"""Calling R from Python — one small, explicit bridge.

Some comparisons in the book use R packages (the ``demography`` package's
Lee–Carter and functional-data forecasts).  R is called only through this
module, always the same way:

1. The inputs are written as CSV files into a fresh working folder
   ``work/r/<tag>/``, together with a ``settings.json`` holding the values
   the script needs (all of which come from ``config/params*.toml``).
2. ``Rscript --vanilla R/<script> work/r/<tag>`` is run as a subprocess.
   The script reads its inputs from the folder and writes its outputs
   there as CSV files, plus ``sessionInfo.txt``.
3. The outputs are read back as DataFrames.  If R exits with an error, a
   :class:`RError` is raised carrying the end of R's error output — R
   failures are never ignored.

The R executable is ``Rscript`` on the ``PATH`` unless the environment
variable ``MDMX_RSCRIPT`` names another one.

Reproducibility: :func:`session_info` records R's ``sessionInfo()`` (R and
package versions) in ``work/r/sessionInfo-<script>.txt`` *before* a cached
step runs, and returns the versions so they can be included in the step's
parameters — a change of R or of a package version then invalidates the
cached results.  The versions returned include, under ``closure``, those of
everything the packages depend on (:func:`package_closure`: ``demography``
needs ``cobs``, ``rainbow``, ``mgcv``, …), so that an update of a dependency
invalidates them too.  After every run, the session information of the run
itself is copied to the same file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from . import paths, provenance

__all__ = ["RError", "rscript", "r_dir", "session_info", "package_closure", "run"]


class RError(RuntimeError):
    """An R script exited with an error."""


def rscript() -> str:
    """The ``Rscript`` executable: ``$MDMX_RSCRIPT`` if set, else ``Rscript``."""
    return os.environ.get("MDMX_RSCRIPT", "Rscript")


def r_dir() -> Path:
    """Working folder for R inputs and outputs (``work/r/``, or ``r/`` under
    ``$MDMX_SCRATCH``, see :func:`mdmx.paths.scratch`)."""
    return paths.scratch() / "r"


def _stem(script: str) -> str:
    return Path(script).stem


def session_info(script: str, packages: Sequence[str]) -> dict:
    """Record R's ``sessionInfo()`` with ``packages`` attached; return the versions.

    Writes the full text to ``work/r/sessionInfo-<script stem>.txt`` and
    returns ``{"R": <version>, "packages": {name: version}, "closure":
    {name: version}}``, suitable for the ``params`` of
    :func:`mdmx.artifacts.stage`.  ``closure`` holds the packages and all
    their dependencies (:func:`package_closure`), so that every R step's key
    covers them.

    Raises :class:`RError` if R or one of the packages is not available.
    """
    paths.ensure(r_dir())
    out_txt = r_dir() / f"sessionInfo-{_stem(script)}.txt"
    pkg_vec = "c(" + ", ".join(f'"{p}"' for p in packages) + ")"
    code = (
        f"pkgs <- {pkg_vec}; "
        "suppressPackageStartupMessages(for (p in pkgs) library(p, character.only = TRUE)); "
        f'writeLines(capture.output(print(sessionInfo())), "{out_txt.as_posix()}"); '
        'v <- sapply(pkgs, function(p) as.character(packageVersion(p))); '
        'cat(jsonlite::toJSON(list(R = paste(R.version$major, R.version$minor, sep = "."), '
        "packages = as.list(v)), auto_unbox = TRUE))"
    )
    out = _run_code(code, what="R session check")
    return {**json.loads(out.strip().splitlines()[-1]), "closure": package_closure(packages)}


def _run_code(code: str, *, what: str) -> str:
    """Run R ``code`` with ``Rscript --vanilla -e`` and return its standard output.

    Raises :class:`RError` if R is not found or exits with an error.
    """
    try:
        proc = subprocess.run([rscript(), "--vanilla", "-e", code], capture_output=True, text=True,
                              check=False)
    except FileNotFoundError as exc:
        raise RError(f"Rscript not found ({rscript()!r}); install R or set MDMX_RSCRIPT.") from exc
    if proc.returncode != 0:
        raise RError(f"{what} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout


# (Rscript, packages) → their closure, for this process (one R call per package set).
_CLOSURES: dict[tuple[str, tuple[str, ...]], dict[str, str | None]] = {}

_CLOSURE_TAG = "MDMXPKG"


def _closure_code(packages: Sequence[str]) -> str:
    """R code that prints ``MDMXPKG<TAB>name<TAB>version`` for ``packages`` and their dependencies."""
    pkg_vec = "c(" + ", ".join(f'"{p}"' for p in packages) + ")"
    return (
        f"pkgs <- {pkg_vec}; "
        # The first library on .libPaths() holding a package is the one library() loads.
        "ip <- installed.packages(); ip <- ip[!duplicated(ip[, 'Package']), , drop = FALSE]; "
        "miss <- setdiff(pkgs, ip[, 'Package']); "
        "if (length(miss)) stop('R packages not installed: ', paste(miss, collapse = ', ')); "
        "deps <- tools::package_dependencies(pkgs, db = ip, which = c('Depends', 'Imports', 'LinkingTo'), "
        "recursive = TRUE); "
        "all <- setdiff(unique(c(pkgs, unlist(deps, use.names = FALSE))), 'R'); "
        "v <- ip[match(all, ip[, 'Package']), 'Version']; "
        f"cat(paste('{_CLOSURE_TAG}', all, ifelse(is.na(v), 'NA', v), sep = '\\t'), sep = '\\n')"
    )


def package_closure(packages: Sequence[str], *, runner: Callable[[str], str] | None = None
                    ) -> dict[str, str | None]:
    """The installed versions of ``packages`` and of everything they need, recursively.

    The packages' ``Depends``, ``Imports`` and ``LinkingTo`` dependencies are followed recursively
    (``tools::package_dependencies``), base packages included, in one ``Rscript`` call (the R of
    :func:`rscript`: ``$MDMX_RSCRIPT`` or ``Rscript`` on the ``PATH``).  :func:`session_info`
    returns it under ``closure``, so it is in the ``params`` of every :func:`mdmx.artifacts.stage`
    that runs R.  It is computed once per process for each R and package set.

    Parameters
    ----------
    packages : the packages the R script loads
    runner : runs R code and returns its standard output (for tests; default: ``Rscript --vanilla
        -e``).  Results from a runner are not cached.

    Returns
    -------
    name → version, sorted by name; None for a dependency that is not installed (the package that
    needs it would fail to load, so the step itself reports it).

    Raises
    ------
    RError if R is not found, fails, or one of ``packages`` is not installed.
    """
    wanted = tuple(sorted(set(packages)))
    key = (rscript(), wanted)
    if runner is None and key in _CLOSURES:
        return dict(_CLOSURES[key])
    code = _closure_code(wanted)
    out = runner(code) if runner is not None else _run_code(code, what="R package-dependency query")
    versions = {}
    for line in out.splitlines():
        parts = line.rstrip("\r").split("\t")
        if len(parts) == 3 and parts[0] == _CLOSURE_TAG:
            versions[parts[1]] = None if parts[2] == "NA" else parts[2]
    absent = [p for p in wanted if versions.get(p) is None]
    if absent:
        raise RError(f"the R package-dependency query returned no version for {absent}:\n{out[-2000:]}")
    versions = dict(sorted(versions.items()))
    if runner is None:
        _CLOSURES[key] = versions
    return dict(versions)


def run(script: str, inputs: Mapping[str, pd.DataFrame], outputs: Sequence[str], *, tag: str,
        settings: Mapping | None = None, timeout: float | None = None) -> dict[str, pd.DataFrame]:
    """Run ``R/<script>`` on CSV inputs and return its CSV outputs.

    Parameters
    ----------
    script : file name of the script in the project's ``R/`` folder
    inputs : name → DataFrame; each is written to ``work/r/<tag>/<name>.csv``
    outputs : names of the CSV files the script must write (without ``.csv``)
    tag : name of the working folder (it is emptied first)
    settings : JSON-serialisable values written to ``settings.json``
    timeout : seconds before the R process is stopped (None: no limit)

    Returns
    -------
    name → DataFrame for every name in ``outputs``.

    Raises
    ------
    RError if R exits with a non-zero status or an expected output is missing.
    """
    work = r_dir() / tag
    if work.exists():
        shutil.rmtree(work)
    paths.ensure(work)
    # LF on every platform: the same input files, byte for byte, on macOS and Windows.
    for name, frame in inputs.items():
        frame.to_csv(work / f"{name}.csv", index=False, lineterminator="\n", encoding="utf-8")
    (work / "settings.json").write_text(json.dumps(dict(settings or {}), indent=1), encoding="utf-8",
                                        newline="\n")

    cmd = [rscript(), "--vanilla", str(paths.root() / "R" / script), str(work)]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except FileNotFoundError as exc:
        raise RError(f"Rscript not found ({rscript()!r}); install R or set MDMX_RSCRIPT.") from exc
    (work / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (work / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    provenance.log("r-run", script=script, tag=tag, returncode=proc.returncode,
                   seconds=round(time.time() - started, 1))
    if proc.returncode != 0:
        raise RError(f"R script {script} failed (exit {proc.returncode}) in {paths.relative(work)}:\n"
                     f"{proc.stderr[-3000:]}")
    if (work / "sessionInfo.txt").exists():
        shutil.copyfile(work / "sessionInfo.txt", r_dir() / f"sessionInfo-{_stem(script)}.txt")
    result = {}
    for name in outputs:
        path = work / f"{name}.csv"
        if not path.exists():
            raise RError(f"R script {script} did not write {paths.relative(path)}.")
        result[name] = pd.read_csv(path, encoding="utf-8")
    return result
