"""Write chapter 12's off-site bayesLife project from the book's own code.

    uv run python tools/bayeslife_bundle.py --out work/bayeslife/offsite-batches/<made>_hmd-<release>  # every run
    uv run python tools/bayeslife_bundle.py --out work/bayeslife/offsite-batches/<made>_hmd-<release> --split \\
        --hours-from work/bayeslife/chapter-12-input                             # a run list per machine
    uv run python tools/bayeslife_bundle.py --out work/bayeslife/offsite-batches/<made>_test --policies full --origins 1905 1950

A batch is named by the day it is made and the HMD release of its data, e.g.
``2026-11-03_hmd-20261029`` (``_<purpose>`` added when two batches share both).

Why this exists
---------------
bayesLife's 48 runs (24 origins, two fitting periods) take hours each, so they run outside a book
render, as a self-contained R project, and chapter 12 imports them through
:func:`mdmx.bayeslife.offsite_runs`, which accepts a run only if its inputs and settings are those
the book makes.  This tool writes that project from the book alone:

* ``jobs/<policy>-<origin>/``: each run's inputs and settings, made by
  :func:`mdmx.bayeslife.inputs` and :func:`mdmx.bayeslife.config` from the e₀ series (the artefact
  ``forecast_bl_e0``, which chapter 12 stages) and the evaluation's jobs
  (:func:`mdmx.evaluation.context`), written by :func:`mdmx.bayeslife.write_run_inputs`;
* ``R/bayeslife_forecast.R`` and ``R/bayeslife_patches.R``: the book's script and patches, unchanged;
* the runner (``run.sh``, ``R/run_all.R``: a few runs at a time, staggered starts, one retry of a
  failed run in a new R process, issue BL-18), ``R/status.R``, the package check and installer, and
  ``R/pinned.R`` (the R version and the versions of the packages a run loads, from
  ``R/install-packages.R``) — the static files are in ``tools/bayeslife_bundle_files/``;
* ``smoke.json`` (``[forecast.bayeslife_smoke]``), ``README.md`` (how to run, check and bring back);
* ``jobs.csv`` (one row per run) and, with ``--split``, ``machines.csv``: the runs divided between
  the machines of ``[evaluation.bl_machines]`` in proportion to the runs each makes at once, the
  longest first, by expected run time (the hours of an earlier batch with ``--hours-from``, else the
  number of observed population-years of the run's input);
* ``MANIFEST.json``: the MDMx commit, the e₀ artefact, and a SHA-256 for every file
  (:func:`mdmx.bayeslife.verify_offsite`).

It needs the artefacts of chapters 03–04 and ``forecast_bl_e0`` (render chapter 12 to its bayesLife
cell first).  Its output holds life expectancies computed from the HMD: it is for Sam's machines
only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from mdmx import artifacts, paths, provenance
from mdmx import bayeslife as bl
from mdmx import evaluation as ev
from mdmx import svdcomp as sc
from mdmx.config import params

#: The bundle's static files (runner, status, package check and installer, README).
STATIC = Path("tools") / "bayeslife_bundle_files"
#: The book's R code a run needs: the script sources the patches from its own folder.
R_SCRIPTS = ("bayeslife_forecast.R", "bayeslife_patches.R")
#: The packages a bayesLife run loads (the runs' sessionInfo.txt), which R/check_env.R checks.
RUN_PACKAGES = ("abind", "bayesLife", "bayesTFR", "car", "carData", "cli", "coda", "data.table", "Formula", "hett",
                "jsonlite", "lattice", "lifecycle", "MASS", "mvtnorm", "rlang", "rlecuyer", "snow", "snowFT",
                "wpp2019")
#: The artefact of the e₀ series (chapter 12).
E0_ARTEFACT = "forecast_bl_e0"


def pinned_r(install_script: str, packages=RUN_PACKAGES) -> str:
    """``R/pinned.R``: the R version and the pins of ``packages``, read from ``R/install-packages.R``.

    The packages keep the script's order (dependencies first), in which ``R/install.R`` builds any
    version that differs.  Raises ``ValueError`` if the script pins no R version or misses a package.
    """
    version = re.search(r'^r_version\s*<-\s*"([^"]+)"', install_script, flags=re.M)
    start = install_script.find("pinned <- c(")
    if version is None or start < 0:
        raise ValueError("R/install-packages.R has no r_version or no pinned <- c(...)")
    block = install_script[start:install_script.index("\n)", start)]
    pins = re.findall(r'^\s*([A-Za-z][\w.]*)\s*=\s*"([^"]+)"', block, flags=re.M)
    missing = sorted(set(packages) - {p for p, _ in pins})
    if missing:
        raise ValueError(f"R/install-packages.R pins no version for {missing}")
    rows = ",\n            ".join(f'{p} = "{v}"' for p, v in pins if p in packages)
    return ("# The R version and the package versions of MDMx's runs, for the packages a bayesLife run loads\n"
            "# (written by MDMx's tools/bayeslife_bundle.py from R/install-packages.R).\n"
            f'r_version <- "{version.group(1)}"\n'
            f"pinned <- c({rows})\n")


def machine_table(bl_machines: Mapping) -> dict[str, dict]:
    """The machines of ``[evaluation.bl_machines]``: user name → ``machine`` (description), ``at_once``."""
    out = {user: dict(m) for user, m in bl_machines.items() if isinstance(m, Mapping)}
    bad = [u for u, m in out.items() if int(m.get("at_once", 0)) < 1]
    if not out or bad:
        raise ValueError(f"[evaluation.bl_machines] needs machines with at_once ≥ 1 (bad: {bad})")
    return out


def split_runs(expected: pd.Series, machines: Mapping[str, Mapping]) -> pd.Series:
    """Each run's machine: the longest runs first, each to the machine where it would end soonest.

    A machine makes ``at_once`` runs at a time, so its finishing time is taken as its runs' expected
    total over ``at_once`` (longest-processing-time scheduling).  Ties go to the first user name in
    alphabetical order.

    Parameters
    ----------
    expected : tag → expected run time (any unit); machines : :func:`machine_table`.

    Returns
    -------
    tag → user name of the machine, in the order of ``expected``.
    """
    load = dict.fromkeys(machines, 0.0)
    out = {}
    for tag, t in expected.sort_values(ascending=False, kind="stable").items():
        m = min(sorted(machines), key=lambda u: (load[u] + float(t)) / int(machines[u]["at_once"]))
        load[m] += float(t)
        out[tag] = m
    return pd.Series(out).reindex(expected.index)


def offsite_hours(folder: Path, tags) -> pd.Series:
    """Each run's hours in an earlier off-site project (the second line of ``results/<run>/DONE``)."""
    folder = Path(folder)
    missing = [t for t in tags if not (folder / "results" / t / "DONE").is_file()]
    if missing:
        raise ValueError(f"{folder} has no finished run for {missing}")
    return pd.Series({t: float((folder / "results" / t / "DONE").read_text(encoding="utf-8")
                               .splitlines()[1].split()[0]) for t in tags})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_bundle(out: Path, e0: pd.DataFrame, jobs: Mapping[str, pd.DataFrame], families: Mapping[str, str],
                 names: Mapping[str, str], *, horizon: int, settings: Mapping, smoke: Mapping, commit: str,
                 dirty: bool = False, e0_info: Mapping | None = None, machines: Mapping[str, Mapping] | None = None,
                 hours: Mapping[str, float] | None = None, force: bool = False) -> dict:
    """Write the off-site project into ``out``; return its manifest.

    Parameters
    ----------
    out : the project folder; it must not exist (``force``: replace one that holds no results)
    e0 : the e₀ series (:func:`mdmx.bayeslife.e0_series`: ``pop, year, female, male``)
    jobs : policy → its jobs (``pop, origin, first_year, last_train_year``), every origin a run
    families, names : population → its family's aggregate, and → its name
    horizon : years forecast; settings : ``[forecast.bayeslife]``; smoke : ``[forecast.bayeslife_smoke]``
    commit, dirty : the MDMx commit the project is made from, and whether the checkout had changes
    e0_info : provenance of the e₀ series, recorded in the manifest
    machines : :func:`machine_table` to split the runs between machines (``machines.csv``)
    hours : run tag → expected hours (an earlier batch); else the observed population-years

    Returns
    -------
    The manifest (also written as ``MANIFEST.json``).
    """
    out = Path(out)
    if out.exists():
        held = [d for d in ("results", "sims", "results-smoke", "sims-smoke") if (out / d).exists()]
        if not force or held:
            raise FileExistsError(f"{out} exists" + (f" and holds {held}: move them first" if held else
                                                       "; force=True replaces it"))
        shutil.rmtree(out)
    root = paths.root()
    r_settings = {k: v for k, v in settings.items() if k != "concurrent_runs"}
    rows = []
    for policy, jobs_p in jobs.items():
        for origin, jobs_o in jobs_p.groupby("origin", sort=True):
            origin, jobs_o = int(origin), jobs_o.reset_index(drop=True)
            tag = f"{policy}-{origin}"
            inp = bl.inputs(e0, jobs_o, families, names, origin)
            cfg = bl.config(inp, origin, horizon, r_settings, None)
            bl.write_run_inputs(out / "jobs" / tag, inp, cfg)
            years = [c for c in inp["female"].columns if c not in ("country_code", "name", "include_code")]
            rows.append({"tag": tag, "policy": policy, "origin": origin, "n_pops": len(jobs_o),
                         "n_extra": len(cfg["extra"]), "start_year": cfg["start_year"], "end_year": cfg["end_year"],
                         "n_obs": int(inp["female"][years].notna().to_numpy().sum())})
    if not rows:
        raise ValueError("no runs: the jobs are empty")
    index = pd.DataFrame(rows)
    if hours is not None:
        index["expected"] = index["tag"].map(dict(hours)).astype(float)
        if index["expected"].isna().any():
            raise ValueError(f"no expected hours for {index.loc[index['expected'].isna(), 'tag'].tolist()}")
        expected_from = "hours of the same runs in an earlier batch"
    else:
        index["expected"] = index["n_obs"].astype(float)
        expected_from = "observed population-years of the run's input (female e0 values)"
    if machines is not None:
        index["machine"] = split_runs(index.set_index("tag")["expected"], machines).to_numpy()
        table = pd.DataFrame([{"machine": u, "description": m["machine"], "at_once": int(m["at_once"]),
                               "runs": int((index["machine"] == u).sum()),
                               "expected": float(index.loc[index["machine"] == u, "expected"].sum())}
                              for u, m in sorted(machines.items())])
        _write_text(out / "machines.csv", table.to_csv(index=False, lineterminator="\n"))
    _write_text(out / "jobs.csv", index.to_csv(index=False, lineterminator="\n"))

    for rel in ("run.sh", "README.md", "R/run_all.R", "R/status.R", "R/check_env.R", "R/install.R"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / STATIC / rel, out / rel)
    for name in R_SCRIPTS:
        shutil.copyfile(root / "R" / name, out / "R" / name)
    _write_text(out / "R" / "pinned.R", pinned_r((root / "R" / "install-packages.R").read_text(encoding="utf-8")))
    _write_text(out / "smoke.json", json.dumps(dict(smoke), indent=1) + "\n")

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"made": provenance.now(), "by": "tools/bayeslife_bundle.py",
                "commits": f"MDMx {commit[:7]}" + ("+dirty" if dirty else ""), "commit": commit, "dirty": bool(dirty),
                "e0": dict(e0_info) if e0_info is not None else None, "horizon": int(horizon),
                "settings": r_settings, "smoke": dict(smoke), "runs": len(index),
                "policies": sorted(index["policy"].unique().tolist()),
                "origins": sorted(int(o) for o in index["origin"].unique()), "expected": expected_from,
                "machines": dict(machines) if machines is not None else None,
                "sha256": {p.relative_to(out).as_posix(): provenance.sha256_file(p) for p in files}}
    _write_text(out / "MANIFEST.json", json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="the project folder to write (it must not exist)")
    ap.add_argument("--policies", nargs="+", choices=["full", "win"], default=["full", "win"])
    ap.add_argument("--origins", nargs="+", type=int, help="these origins only (default: every origin)")
    ap.add_argument("--split", action="store_true",
                    help="a run list per machine of [evaluation.bl_machines], balanced on expected run time")
    ap.add_argument("--hours-from", type=Path,
                    help="an earlier off-site project whose run times (results/<run>/DONE) predict this one's")
    ap.add_argument("--force", action="store_true", help="replace an existing project folder that holds no results")
    ap.add_argument("--allow-dirty", action="store_true", help="write from uncommitted code (tests only)")
    args = ap.parse_args()

    env = provenance.environment()
    if env["git_commit"] is None:
        raise SystemExit("Cannot read the git commit: run from the MDMx repository")
    if env["git_dirty"] and not args.allow_dirty:
        raise SystemExit("Uncommitted changes: commit first, so that the manifest names the code")
    P = params()
    F, E = P.section("forecast"), P.section("evaluation")
    ctx = ev.context(P)
    if not artifacts.exists(E0_ARTEFACT):
        raise SystemExit(f"No artefact {E0_ARTEFACT}: render chapter 12 to its bayesLife cell first")
    meta = artifacts.meta(E0_ARTEFACT)
    if meta["inputs"].get("forecast_cmp_rates") != artifacts.meta("forecast_cmp_rates")["content_hash"]:
        raise SystemExit(f"{E0_ARTEFACT} was computed from other rates than today's forecast_cmp_rates: "
                         "render chapter 12 to its bayesLife cell first")
    e0 = artifacts.load(E0_ARTEFACT)
    pops = ctx["pops"]
    families = dict(zip(pops, sc.population_families(pops, artifacts.load("populations")), strict=True))
    origins = [int(o) for o in ctx["design"].origins]
    if args.origins:
        unknown = sorted(set(args.origins) - set(origins))
        if unknown:
            raise SystemExit(f"Not origins of the evaluation: {unknown} (they are {origins})")
        origins = sorted(args.origins)
    jobs = {p: ctx["jobs"][p][ctx["jobs"][p]["origin"].isin(origins)] for p in args.policies}
    tags = [f"{p}-{o}" for p, j in jobs.items() for o in sorted(j["origin"].unique())]
    hours = offsite_hours(args.hours_from.expanduser(), tags) if args.hours_from else None
    out = args.out.expanduser()
    manifest = build_bundle(
        out, e0, jobs, families, ctx["names"], horizon=ctx["design"].horizon, settings=F["bayeslife"],
        smoke=F["bayeslife_smoke"], commit=env["git_commit"], dirty=bool(env["git_dirty"]),
        e0_info={"artefact": E0_ARTEFACT, "content_hash": meta["content_hash"], "created": meta["created"],
                 "params": meta["params"], "git_commit": meta["environment"].get("git_commit")},
        machines=machine_table(E["bl_machines"]) if args.split else None, hours=hours, force=args.force)
    print(f"Wrote {manifest['runs']} runs ({', '.join(manifest['policies'])}; origins "
          f"{manifest['origins'][0]}–{manifest['origins'][-1]}) to {out}; {manifest['commits']}.")
    if manifest["machines"]:
        print((out / "machines.csv").read_text(encoding="utf-8"))
    print("Next: copy the folder to each machine (on a Mac with MDMx, into work/bayeslife/offsite-batches/) and follow its README.md (bash run.sh).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
