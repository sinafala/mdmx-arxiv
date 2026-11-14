"""Project paths — one place that knows where everything lives.

Every file the pipeline reads or writes is located relative to the
**project root** (the folder that contains ``pyproject.toml`` for the
``mdmx`` project).  Nothing is hard-coded to a particular computer, so the
same code runs unchanged on macOS, Windows and Linux, from Positron, from
``quarto render``, or from the command line.

How the root is found
---------------------
1. If the environment variable ``MDMX_ROOT`` is set, it wins.  (Useful for
   tests and for running the pipeline against a second copy of the data.)
2. Otherwise we walk *upwards* from the current working directory until we
   find a ``pyproject.toml`` whose ``[project] name`` is ``"mdmx"``.
3. As a last resort we walk upwards from this file's own location (this
   covers an editable install where the working directory is elsewhere).

Folder layout (see README.md for the full map)
----------------------------------------------
``raw/``      inputs exactly as downloaded (git-ignored, except MANIFEST.json)
``work/``     intermediate artefacts and caches (git-ignored, regenerable)
``results/``  the outputs of ``book/``: figures, tables, numbers
``book/``     the development book (the record of how the methods were built and chosen)
``companion/`` the stand-alone companion of the article, with its own ``companion/results/``
``article/``  the merged article (LaTeX), built from ``companion/results/``
``config/``   analysis parameters (``params.toml``) and event dictionary
``app/``      the Shiny app

Projects
--------
Each Quarto book in :data:`PROJECTS` writes its own outputs and keeps its own artefact store, so
that rendering one can never change another's results or cache.  The project is chosen by the
environment variable ``MDMX_PROJECT`` (``mdmx render --project companion`` sets it), and is
``book`` when it is not set:

* ``book``: outputs (:func:`results`) in ``results/``, artefacts (:func:`artifacts`) in
  ``work/artifacts/``;
* ``companion``: outputs in ``companion/results/``, artefacts in ``work/companion/artifacts/``.

Everything else in ``work/`` is shared: the R working files, bayesLife's off-site runs, the logs of
``book`` (other projects log to ``work/<project>/logs/``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

__all__ = [
    "root", "raw", "work", "results", "figures", "tables", "numbers",
    "config_dir", "book", "article", "app", "artifacts", "scratch", "logs",
    "ensure", "relative", "PROJECTS", "DEFAULT_PROJECT", "project", "using_project",
]

#: The Quarto books, each with its own outputs and artefact store (module docstring, "Projects").
PROJECTS = ("book", "companion")
#: The project when ``MDMX_PROJECT`` is not set: the development book, as before projects existed.
DEFAULT_PROJECT = "book"

_PROJECT_OVERRIDE: list[str] = []      # set by using_project(), innermost last


def _is_mdmx_root(folder: Path) -> bool:
    """Return True if ``folder`` holds the mdmx ``pyproject.toml``."""
    pyproject = folder / "pyproject.toml"
    if not pyproject.is_file():
        return False
    # A light-weight check (no TOML parser needed): look for name = "mdmx".
    text = pyproject.read_text(encoding="utf-8", errors="ignore")
    return re.search(r'^\s*name\s*=\s*"mdmx"\s*$', text, flags=re.MULTILINE) is not None


@lru_cache(maxsize=1)
def root() -> Path:
    """Return the absolute path of the project root (see module docstring)."""
    env = os.environ.get("MDMX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for folder in (start, *start.parents):
            if _is_mdmx_root(folder):
                return folder
    raise FileNotFoundError(
        "Could not find the MDMx project root (a folder whose pyproject.toml "
        "has name = \"mdmx\").  Run from inside the repository or set the "
        "MDMX_ROOT environment variable."
    )


# ── Projects ─────────────────────────────────────────────────────────────────

def project() -> str:
    """The Quarto project whose outputs and artefacts are in use: :func:`using_project`'s, else the
    environment variable ``MDMX_PROJECT``, else :data:`DEFAULT_PROJECT`.

    Raises
    ------
    ValueError if the name is not one of :data:`PROJECTS`.
    """
    name = _PROJECT_OVERRIDE[-1] if _PROJECT_OVERRIDE else os.environ.get("MDMX_PROJECT") or DEFAULT_PROJECT
    if name not in PROJECTS:
        raise ValueError(f"Unknown MDMx project {name!r}; the projects are {', '.join(PROJECTS)}.")
    return name


@contextmanager
def using_project(name: str) -> Iterator[str]:
    """Use project ``name`` inside a ``with`` block (``mdmx check`` checks every project this way)."""
    if name not in PROJECTS:
        raise ValueError(f"Unknown MDMx project {name!r}; the projects are {', '.join(PROJECTS)}.")
    _PROJECT_OVERRIDE.append(name)
    try:
        yield name
    finally:
        _PROJECT_OVERRIDE.pop()


# ── Named locations ──────────────────────────────────────────────────────────
# These are small functions (not constants) so that MDMX_ROOT can be changed
# in tests before first use.

def raw() -> Path:
    """Raw inputs exactly as downloaded (``raw/``)."""
    return root() / "raw"


def work() -> Path:
    """Regenerable intermediate artefacts (``work/``)."""
    return root() / "work"


def bayeslife() -> Path:
    """Everything of bayesLife's runs, in one folder (``work/bayeslife/``, git-ignored).

    * ``offsite-batches/<made>_hmd-<release>[_<purpose>]/``: each off-site batch as it was run on
      the machines (written by ``tools/bayeslife_bundle.py``; the record, never edited);
    * ``chapter-12-input/``: the copy of one batch that chapter 12 reads
      (``tools/fetch_bayeslife_offsite.py``);
    * ``in-render/<tag>/``: the chains of runs made by :func:`mdmx.bayeslife.run` (inside a
      render, or by ``tools/run_bayeslife.py``);
    * ``checks/`` and ``parity/``: the outputs of ``tools/bayeslife_extra_check.py`` and
      ``tools/bayeslife_offsite_parity.py``.
    """
    return work() / "bayeslife"


def artifacts() -> Path:
    """The project's artefact store, used by :mod:`mdmx.artifacts` (``work/artifacts/`` for
    ``book``, ``work/<project>/artifacts/`` for another project)."""
    name = project()
    return work() / "artifacts" if name == DEFAULT_PROJECT else work() / name / "artifacts"


def scratch() -> Path:
    """Folder for the side outputs of a run: R working files and run logs.

    ``work/`` unless the environment variable ``MDMX_SCRATCH`` names another
    folder.  The experiment lab sets it, so that it can read the artefact
    store while every file it writes stays inside the lab.
    """
    env = os.environ.get("MDMX_SCRATCH")
    return Path(env).expanduser().resolve() if env else work()


def logs() -> Path:
    """Per-chapter run logs (``work/logs/``, or ``logs/`` under :func:`scratch`; another project's
    in ``<project>/logs/`` there)."""
    name = project()
    return scratch() / "logs" if name == DEFAULT_PROJECT else scratch() / name / "logs"


def results() -> Path:
    """The project's paper-facing outputs, committed to git (``results/`` for ``book``,
    ``<project>/results/`` for another project)."""
    name = project()
    return root() / "results" if name == DEFAULT_PROJECT else root() / name / "results"


def figures() -> Path:
    """Figure PDFs read by the paper (``results/figures/``)."""
    return results() / "figures"


def tables() -> Path:
    """LaTeX tables read by the paper (``results/tables/``)."""
    return results() / "tables"


def numbers() -> Path:
    """Generated in-text numbers: ``.tex`` macros and ``.json`` (``results/numbers/``)."""
    return results() / "numbers"


def config_dir() -> Path:
    """Analysis parameters and the event dictionary (``config/``)."""
    return root() / "config"


def book() -> Path:
    """The folder of the project's Quarto book (``book/``, ``companion/``)."""
    return root() / project()


def article() -> Path:
    """The merged article (``article/``, LaTeX), built from the companion's results."""
    return root() / "article"


def app() -> Path:
    return root() / "app"


def ensure(*folders: Path) -> None:
    """Create the given folders (and parents) if they do not exist."""
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """Express ``path`` relative to the project root, with forward slashes.

    Used whenever a path is *recorded* (manifests, logs, table footnotes) so
    that records never contain machine-specific absolute paths.
    """
    try:
        return Path(path).resolve().relative_to(root()).as_posix()
    except ValueError:
        return Path(path).as_posix()
