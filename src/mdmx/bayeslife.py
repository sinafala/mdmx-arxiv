"""The UN's e₀ model (R bayesLife) on HMD data, as a comparator in chapter 12 (decision D2).

What this module does
---------------------
bayesLife (Raftery, Chunn, Gerland & Ševčíková 2013; version 5.3-1) projects
female life expectancy by a Bayesian hierarchical double-logistic model and
male life expectancy by a joint female–male gap model.  The UN runs it on the
World Population Prospects; here it is run on the HMD only
(``use.wpp.data = FALSE``) by ``R/bayeslife_forecast.R``, one R process per
forecast origin and fitting-period policy:

1. :func:`e0_series` computes every population-year's female and male e₀
   from the same HMD death rates as the other comparators (years missing or
   exceptional inside a series are filled from interpolated deaths and
   exposures; chapter 12).
2. :func:`inputs` writes one origin's tables: each population's e₀ over its
   training years, numeric codes, a locations table, and the split into
   populations that estimate the world-level parameters (``include_code``
   2) and the parts of aggregates that are added afterwards
   (:func:`split_overlap`), so that no deaths count twice in the hierarchy.
3. :func:`run` calls R through :mod:`mdmx.rbridge`; :func:`run_many` runs
   several origins at once (each R process runs its chains in parallel) and
   retries a failed one once (issue BL-18).
4. :func:`points` turns the output into the benchmark harness's format
   (``pop, sex, year, e0_fc`` and the 80 / 95 % predictive intervals).

Runs made elsewhere (the off-site project that ``tools/bayeslife_bundle.py``
writes, one folder per run by :func:`write_run_inputs`) are checked and
imported by :func:`offsite_runs`.

Conventions
-----------
* Sexes ``female``, ``male``, ``both`` (the per-trajectory average).
* Population codes are 10001, 10002, … in the order of the populations; no
  HMD population is mapped to a UN country code.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts, lifetable, paths, provenance, rbridge
from . import forecast as fc

__all__ = ["CODE0", "R_PACKAGES", "RUN_FILES", "e0_series", "split_overlap", "inputs", "config", "write_run_inputs",
           "run", "origin_run", "run_many", "benchmark_runs", "verify_offsite", "offsite_run", "offsite_runs",
           "points", "schedules"]

#: First numeric code given to a population (above every UN location code).
CODE0 = 10001
#: The R script's outputs; ``redraws`` and ``lowtail`` count BL-16's redraws and the female
#: trajectories below e₀ 20 (docs/bayeslife-issues.md).
OUTPUTS = ("forecasts", "diagnostics", "gap", "failures", "warnings", "timing", "redraws", "lowtail")
#: R packages whose versions are recorded with (and key the cache of) every run.
R_PACKAGES = ["bayesLife", "bayesTFR", "snowFT", "coda", "data.table", "jsonlite"]
# bayesLife leaves out of the estimation of the gains every annual change of female e0 outside
# this range (its own constant for annual data, e0.mcmc1y.options.default()$outliers).
OUTLIER_LIMITS = (-5.0, 5.0)
#: The input tables of one run, as :func:`write_run_inputs` writes them (``settings.json`` besides).
RUN_FILES = ("female", "male", "locations", "codes")
#: Seconds :func:`run_many` waits before it retries a failed task.  BL-18: snow picks its socket port
#: from the seed and the clock (1,000 ports every 300 s), so a minute later the port is ~200 ports on.
RETRY_DELAY = 60.0


def e0_series(rates: pd.DataFrame, n_ages: int, ax: float) -> pd.DataFrame:
    """Female and male e₀ of every population-year of the comparators' death rates.

    Paper: sec:evaluation:entries

    Parameters
    ----------
    rates : ``pop, sex, year, age, deaths, exposure`` (the artefact
        ``forecast_cmp_rates``: ages 0 … open age, gaps inside a series filled).
    n_ages : single-year ages of the tensor; ax : a_x of the m_x → q_x
        conversion — both as for the other comparators
        (:func:`mdmx.forecast.comparator_schedules`).

    Returns
    -------
    DataFrame ``pop, year, female, male`` (years with both sexes only).
    """
    empty = rates["exposure"] <= 0
    if (rates.loc[empty, "deaths"] > 0).any():
        raise ValueError("deaths with no exposure in the comparators' rates")
    mx = rates.assign(mx=(rates["deaths"] / rates["exposure"].where(~empty)))
    wide = mx.pivot(index=["pop", "sex", "year"], columns="age", values="mx")
    # An age with no exposure (and no deaths) has no one left alive — it happens only at the oldest
    # ages of small populations; it keeps the rate of the age below (a constant force), which
    # changes e0 by far less than 0.001 years since survivorship there is all but zero.
    wide = wide.ffill(axis=1)
    if wide.iloc[:, 0].isna().any():
        raise ValueError("no exposure at age 0 in the comparators' rates")
    wide.columns = [f"a{int(a)}" for a in wide.columns]
    wide = wide.reset_index().assign(method="rates", origin=0)
    index, z = fc.comparator_schedules(wide, n_ages, ax=ax)
    ok = np.isfinite(z).all(axis=(1, 2))
    e0 = fc.e0_and_gradient(z[ok])[0]                                    # (N, 2)
    return pd.DataFrame({"pop": index["pop"].to_numpy()[ok], "year": index["year"].to_numpy(dtype=int)[ok],
                         "female": e0[:, 0], "male": e0[:, 1]})


def split_overlap(pops: Sequence[str], families: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Populations that estimate the world level, and those added afterwards.

    Paper: sec:evaluation:entries

    A family is a population with its HMD parts (``families`` maps each
    population to its family's aggregate).  If the aggregate is among
    ``pops`` it estimates the world level and its parts are added
    afterwards; if not, its parts estimate it (they do not overlap one
    another).

    Returns
    -------
    main, extra : lists of population codes, in the order of ``pops``.
    """
    present = set(pops)
    main, extra = [], []
    for p in pops:
        fam = families[p]
        (extra if p != fam and fam in present else main).append(p)
    return main, extra


def inputs(e0: pd.DataFrame, jobs_o: pd.DataFrame, families: Mapping[str, str], names: Mapping[str, str],
           origin: int) -> dict:
    """The input tables of ``R/bayeslife_forecast.R`` for one origin.

    Each population of ``jobs_o`` contributes its female and male e₀ from
    ``first_year`` to ``last_train_year`` (NA elsewhere); the year columns
    run contiguously from the earliest first year to the origin, which is
    bayesLife's ``present.year``.

    Parameters
    ----------
    e0 : from :func:`e0_series`; jobs_o : this origin's rows of
        :func:`mdmx.forecast.cv_jobs` (``pop, first_year, last_train_year``).
    families : population → its family's aggregate; names : population → name.

    Returns
    -------
    dict with ``female``, ``male``, ``locations`` (DataFrames), ``codes``
    (population → code), ``extra`` (codes added after the world fit) and
    ``start_year``.
    """
    jobs_o = jobs_o.sort_values("pop").reset_index(drop=True)
    if jobs_o["pop"].duplicated().any():
        raise ValueError("a population appears twice among the jobs of one origin")
    pops = list(jobs_o["pop"])
    codes = {p: CODE0 + i for i, p in enumerate(pops)}
    main, extra = split_overlap(pops, families)
    start = int(jobs_o["first_year"].min())
    years = np.arange(start, origin + 1)
    by = e0.set_index(["pop", "year"])
    tables = {}
    for sex in ("female", "male"):
        mat = np.full((len(pops), len(years)), np.nan)
        for i, j in enumerate(jobs_o.itertuples()):
            ys = np.arange(int(j.first_year), int(j.last_train_year) + 1)
            vals = by[sex].reindex(pd.MultiIndex.from_arrays([np.full(len(ys), j.pop), ys]))
            if vals.isna().any():
                raise ValueError(f"{j.pop}: no e0 for some training years ({sex})")
            mat[i, ys - start] = vals.to_numpy()
        if (np.isfinite(mat).sum(axis=1) < 2).any():
            raise ValueError("bayesLife needs at least two observed years per population")
        tables[sex] = pd.concat([pd.DataFrame({"country_code": [codes[p] for p in pops],
                                               "name": [names[p] for p in pops],
                                               "include_code": [2 if p in main else 0 for p in pops]}),
                                 pd.DataFrame(mat, columns=[str(y) for y in years])], axis=1)
    locations = pd.DataFrame({"country_code": [codes[p] for p in pops], "name": [names[p] for p in pops],
                              "location_type": 4})
    return {**tables, "locations": locations, "codes": codes, "extra": [codes[p] for p in extra],
            "start_year": start}


def config(inp: Mapping, origin: int, horizon: int, settings: Mapping, sim_dir: Path | None) -> dict:
    """The ``settings.json`` of one run: what :func:`run` passes to ``R/bayeslife_forecast.R``.

    ``settings`` is ``[forecast.bayeslife]`` without ``concurrent_runs``; ``inp`` comes from
    :func:`inputs`.  The seed is offset by the origin; the chains' forecasts are compared at
    ``origin + check_horizon`` when it is given.
    """
    cfg = {**settings, "origin": int(origin), "end_year": int(origin + horizon),
           "sim_dir": None if sim_dir is None else str(sim_dir), "start_year": int(inp["start_year"]),
           "extra": [int(c) for c in inp["extra"]], "seed": int(settings["seed"]) + int(origin), "jmale": True}
    if "check_horizon" in settings:
        cfg["check_year"] = int(origin + int(settings["check_horizon"]))
    return cfg


def _codes_table(inp: Mapping) -> pd.DataFrame:
    """``pop, country_code`` of one run's populations (``inp`` from :func:`inputs`)."""
    return pd.DataFrame({"pop": list(inp["codes"]), "country_code": list(inp["codes"].values())})


def _csv_text(frame: pd.DataFrame) -> str:
    """A table as CSV text with ``\\n`` line ends.

    pandas ends lines with the platform's separator by default (``\\r\\n`` on Windows); a fixed one
    makes the files of a run, and the comparison of :func:`offsite_runs`, the same on every platform.
    """
    return frame.to_csv(index=False, lineterminator="\n")


def _text_lf(path: Path) -> str:
    """A text file's content with ``\\n`` line ends, whatever a checkout or copy did to them."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def write_run_inputs(folder: Path, inp: Mapping, cfg: Mapping) -> list[Path]:
    """Write one run's input folder: what ``R/bayeslife_forecast.R`` reads, and the population codes.

    Parameters
    ----------
    folder : the run's folder (created; must not hold files of another run)
    inp : :func:`inputs` for the run; cfg : its settings, :func:`config`

    Returns
    -------
    The files written: ``female.csv``, ``male.csv``, ``locations.csv``, ``codes.csv`` (population →
    numeric code) and ``settings.json``, every one UTF-8 with ``\\n`` line ends.  The tables are
    byte-identical to those :func:`offsite_runs` expects back from the run.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"female": inp["female"], "male": inp["male"], "locations": inp["locations"],
              "codes": _codes_table(inp)}
    written = []
    for name in RUN_FILES:
        path = folder / f"{name}.csv"
        path.write_text(_csv_text(tables[name]), encoding="utf-8", newline="\n")
        written.append(path)
    path = folder / "settings.json"
    path.write_text(json.dumps(dict(cfg), indent=1) + "\n", encoding="utf-8", newline="\n")
    return [*written, path]


def run(inp: Mapping, origin: int, horizon: int, settings: Mapping, *, tag: str) -> dict:
    """Run ``R/bayeslife_forecast.R`` on one origin's inputs (:func:`inputs`).

    ``settings`` holds the settings of ``[forecast.bayeslife]``: the MCMC, the
    convergence rule (checked at ``origin + check_horizon`` when it is given,
    else at the end year) and BL-16's bound.  The seed is offset by the
    origin so that the origins do not share draws.  Male e₀ comes from the
    joint gap model, with BL-5 and BL-6 before 1953 (``R/bayeslife_patches.R``).
    The simulation is kept in ``work/bayeslife/in-render/<tag>`` (not in the R work
    folder, which is emptied on every run), so that it can be extended or
    predicted again later.

    Returns
    -------
    dict of the R outputs (:data:`OUTPUTS`) plus ``codes`` (``pop, country_code``).
    """
    sim_dir = paths.bayeslife() / "in-render" / tag
    paths.ensure(sim_dir.parent)
    cfg = config(inp, origin, horizon, settings, sim_dir)
    out = rbridge.run("bayeslife_forecast.R",
                      {"female": inp["female"], "male": inp["male"], "locations": inp["locations"]},
                      OUTPUTS, tag=tag, settings=cfg)
    return {**out, "codes": pd.DataFrame({"pop": list(inp["codes"]),
                                          "country_code": list(inp["codes"].values())})}


def origin_run(e0: pd.DataFrame, jobs_o: pd.DataFrame, families: Mapping[str, str], names: Mapping[str, str],
               origin: int, horizon: int, settings: Mapping, *, tag: str) -> dict:
    """:func:`inputs` and :func:`run` for one origin — the unit that is cached."""
    return run(inputs(e0, jobs_o, families, names, origin), origin, horizon, settings, tag=tag)


def _retry_once(name, fn: Callable[[], dict], delay: float) -> dict:
    """``fn()``; if it fails, log the failure and call ``fn`` once more after ``delay`` seconds.

    BL-18: runs started at the same moment with the same seed pick the same socket port for their
    chains' cluster, and all but one fail at once.  Each call of a task starts a new Rscript process,
    so a second call, later, gets another port and the same random stream; a fixed port
    (``R_PARALLEL_PORT``) would change the chains.  A second failure is raised.
    """
    try:
        return fn()
    except Exception as first:  # logged here; a second failure propagates to run_many's report
        message = str(first).strip().splitlines()
        message = message[0] if message else type(first).__name__
        provenance.log("run-retry", task=str(name), error=str(first)[-2000:], delay_seconds=delay)
        print(f"  [retry] {name} failed ({message}); running it again in {delay:g} s, in a new process",
              flush=True)
        time.sleep(delay)
        return fn()


def run_many(tasks: Sequence[tuple[str, Callable[[], dict]]], *, concurrent: int,
             retry_delay: float = RETRY_DELAY) -> dict[str, dict]:
    """Run several origins at once (each an R process: bayesLife's chains, the comparators' intervals).

    A task that fails is run once more, ``retry_delay`` seconds later (each call starts a new R
    process; issue BL-18); the retry is logged (``run-retry`` in the chapter's run log) with the task
    and its first error.

    Parameters
    ----------
    tasks : (name, zero-argument callable) pairs — typically
        ``artifacts.stage`` calls, so that a cached origin returns at once.
    concurrent : origins run at the same time; with ``nr_chains`` chains
        each, ``concurrent × nr_chains`` should not exceed the cores.
    retry_delay : seconds before a failed task's second attempt (run time only).

    Returns
    -------
    name → result.  A task that fails twice is raised, with every other such task, after the others
    finish.
    """
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {name: pool.submit(_retry_once, name, fn, retry_delay) for name, fn in tasks}
        results, errors = {}, {}
        for name, f in futures.items():
            try:
                results[name] = f.result()
            except Exception as err:  # every failure is reported below, not only the first
                errors[name] = err
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(futures)} runs failed:\n"
                           + "\n".join(f"  {name}: {err}" for name, err in errors.items())) \
            from next(iter(errors.values()))
    return results


def _with_r_message(fn: Callable[[], dict], tag: str) -> dict:
    """Run ``fn``; if it fails, add the R script's own error message.

    ``R/bayeslife_forecast.R`` writes its error to ``failures.csv`` and quits,
    so the exception from :func:`mdmx.rbridge.run` alone does not say what went
    wrong.
    """
    try:
        return fn()
    except Exception as err:
        failures = rbridge.r_dir() / tag / "failures.csv"
        if failures.exists():
            messages = pd.read_csv(failures, encoding="utf-8")["message"].astype(str).tolist()
            if messages:
                raise RuntimeError(f"bayesLife run {tag}: " + "; ".join(messages)) from err
        raise


def benchmark_runs(rates: pd.DataFrame, jobs: Mapping[str, pd.DataFrame], families: Mapping[str, str],
                   names: Mapping[str, str], *, n_ages: int, ax: float, horizon: int, settings: Mapping,
                   prefix: str = "forecast_bl") -> dict[tuple[str, int], dict]:
    """Every origin of every fitting-period policy, cached, several at once (chapter 12).

    The e₀ series are computed (and cached as ``<prefix>_e0``) from the
    comparators' rates; each origin of each policy is the artefact
    ``<prefix>_<policy>_<origin>``, keyed on its jobs, the settings, the R
    script and the R package versions.  The chapter and any launcher call
    this same function, so a run computed by one is found by the other.

    Parameters
    ----------
    rates : the artefact ``forecast_cmp_rates``; jobs : policy → its jobs
        (:func:`mdmx.forecast.cv_jobs`, or :func:`mdmx.forecast.restrict_fit_window`).
    settings : ``[forecast.bayeslife]`` (``concurrent_runs`` is used here).

    Returns
    -------
    (policy, origin) → the result of :func:`run`.
    """
    e0 = artifacts.stage(f"{prefix}_e0", lambda: e0_series(rates, n_ages, ax), inputs=["forecast_cmp_rates"],
                         params={"n_ages": n_ages, "ax": ax}, code=[e0_series, fc.comparator_schedules,
                                                                   fc.e0_and_gradient],
                         desc="Female and male e0 of every population-year of the comparators' rates")
    r_settings = {k: v for k, v in settings.items() if k != "concurrent_runs"}
    key = {"settings": r_settings, "horizon": horizon,
           "script": provenance.sha256_file(paths.root() / "R" / "bayeslife_forecast.R"),
           "patches": provenance.sha256_file(paths.root() / "R" / "bayeslife_patches.R"),   # the script sources it
           "r": rbridge.session_info("bayeslife_forecast.R", R_PACKAGES)}
    tasks = []
    for policy, jobs_p in jobs.items():
        for origin, jobs_o in jobs_p.groupby("origin", sort=True):
            origin = int(origin)
            jobs_o = jobs_o.reset_index(drop=True)
            name, tag = f"{prefix}_{policy}_{origin}", f"{prefix}-{policy}-{origin}".replace("_", "-")
            compute = functools.partial(origin_run, e0, jobs_o, families, names, origin, horizon, r_settings,
                                        tag=tag)
            stage = functools.partial(
                artifacts.stage, name, compute, inputs=[f"{prefix}_e0"],
                params={**key, "jobs": artifacts.content_hash(jobs_o), "policy": policy},
                code=[origin_run, inputs, split_overlap, run, rbridge.run],
                desc=f"bayesLife (HMD only) forecasts from {origin}, policy {policy}")
            tasks.append(((policy, origin), functools.partial(_with_r_message, stage, tag)))
    return run_many(tasks, concurrent=int(settings["concurrent_runs"]))


# ── Runs made elsewhere: the off-site project (tools/bayeslife_bundle.py) ───────────────────────────

def verify_offsite(folder: Path) -> dict:
    """Check an off-site project's files against its ``MANIFEST.json``; return the manifest.

    The manifest lists the SHA-256 of every file the project was shipped with (its R scripts, the
    jobs' inputs and settings, the setup kit); any that changed or is missing is an error.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [rel for rel, digest in manifest["sha256"].items()
           if not (folder / rel).is_file() or provenance.sha256_file(folder / rel) != digest]
    if bad:
        raise ValueError(f"{len(bad)} files of {folder} differ from its manifest: {bad[:5]}")
    return manifest


def offsite_run(folder: Path, tag: str) -> dict:
    """One finished off-site run: the R script's outputs (:data:`OUTPUTS`), ``codes`` and its provenance."""
    d = Path(folder) / "results" / tag
    if not (d / "DONE").is_file():
        raise ValueError(f"off-site run {tag} has not finished (no DONE in {d})")
    out = {name: pd.read_csv(d / f"{name}.csv", encoding="utf-8") for name in OUTPUTS}
    out["codes"] = pd.read_csv(d / "codes.csv", encoding="utf-8")
    finished, hours = (d / "DONE").read_text(encoding="utf-8").splitlines()[:2]
    session = (d / "sessionInfo.txt").read_text(encoding="utf-8").splitlines()
    out["provenance"] = {"finished": finished, "hours": float(hours.split()[0]),
                         "r": session[0], "platform": next((ln for ln in session if ln.startswith("Platform")), ""),
                         "running_under": next((ln for ln in session if ln.startswith("Running under")), "")}
    return out


def offsite_runs(folder: Path, e0: pd.DataFrame, jobs: Mapping[str, pd.DataFrame], families: Mapping[str, str],
                 names: Mapping[str, str], *, horizon: int, settings: Mapping, e0_artefact: str,
                 prefix: str = "evaluation_bl") -> dict[tuple[str, int], dict]:
    """The off-site runs of every origin and policy, checked to be the book's runs, as artefacts.

    Paper: sec:evaluation:entries

    For every policy and origin of ``jobs`` (run ``<policy>-<origin>``):

    * the project's files match its manifest (:func:`verify_offsite`);
    * the run has finished (``DONE``); every missing or failed run is listed in one error;
    * its input files (``female``, ``male``, ``locations`` and ``codes.csv``) are byte-identical,
      line ends aside, to what :func:`write_run_inputs` writes from :func:`inputs` of ``e0`` and
      the jobs, and its settings equal :func:`config`'s from ``settings`` (apart from ``sim_dir``),
      so the run is the one the book would have made;
    * its outputs are read (:func:`offsite_run`) and stored as artefact
      ``<prefix>_<policy>_<origin>``, keyed on the SHA-256 of every output file.

    Parameters
    ----------
    folder : the project (written by ``tools/bayeslife_bundle.py``, run elsewhere, and copied to
        ``work/bayeslife/chapter-12-input`` by ``tools/fetch_bayeslife_offsite.py``)
    e0 : :func:`e0_series` (the artefact ``e0_artefact``); jobs : policy → its jobs
    families, names : population → family, and → name
    horizon : years forecast; settings : ``[forecast.bayeslife]``

    Returns
    -------
    (policy, origin) → the run's outputs, as :func:`run` returns them, plus ``provenance``.
    """
    folder = Path(folder)
    manifest = verify_offsite(folder)
    r_settings = {k: v for k, v in settings.items() if k != "concurrent_runs"}
    tasks = [(policy, int(origin), jobs_o.reset_index(drop=True))
             for policy, jobs_p in jobs.items() for origin, jobs_o in jobs_p.groupby("origin", sort=True)]
    missing = [f"{p}-{o}" + (" (failed)" if (folder / "results" / f"{p}-{o}" / "FAILED").exists() else "")
               for p, o, _ in tasks if not (folder / "results" / f"{p}-{o}" / "DONE").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} of {len(tasks)} off-site runs have not finished: {missing}")
    out = {}
    for policy, origin, jobs_o in tasks:
        tag = f"{policy}-{origin}"
        d = folder / "results" / tag
        inp = inputs(e0, jobs_o, families, names, origin)
        mine_tables = {"female": inp["female"], "male": inp["male"], "locations": inp["locations"],
                       "codes": _codes_table(inp)}
        for name in RUN_FILES:
            # Both sides with \n line ends, so that neither the platform's CSV writer nor a checkout
            # that converts line ends can make the book's own run look different.
            if _csv_text(mine_tables[name]) != _text_lf(d / f"{name}.csv"):
                raise ValueError(f"off-site run {tag}: its {name}.csv is not the book's input")
        theirs = json.loads((d / "settings.json").read_text(encoding="utf-8"))
        mine = config(inp, origin, horizon, r_settings, None)
        differ = sorted(k for k in theirs if k not in ("sim_dir", "mode") and theirs[k] != mine.get(k))
        # check_horizon reaches R as check_year, which is compared above.
        missing = sorted(set(mine) - set(theirs) - {"sim_dir", "check_horizon"})
        if differ or missing:
            raise ValueError(f"off-site run {tag}: settings differ from the book's: {differ}; "
                             f"the book's settings it lacks: {missing}")
        files = {name: provenance.sha256_file(d / f"{name}.csv") for name in (*OUTPUTS, "codes")}
        out[(policy, origin)] = artifacts.stage(
            f"{prefix}_{policy}_{origin}", functools.partial(offsite_run, folder, tag), inputs=[e0_artefact],
            params={"files": files, "settings": r_settings, "horizon": horizon, "policy": policy,
                    "jobs": artifacts.content_hash(jobs_o), "commits": manifest.get("commits")},
            code=[offsite_run], desc=f"bayesLife (HMD only) from {origin}, policy {policy}: the off-site run")
    return out


def run_summary(result: Mapping, jobs_o: pd.DataFrame, families: Mapping[str, str]) -> dict:
    """One run at its last convergence step: populations, convergence, gap model, run time.

    Parameters
    ----------
    result : the output of :func:`run` for one origin and policy.
    jobs_o : that origin's jobs (as passed to the run); families : population → family.

    Returns
    -------
    dict with ``world``/``extra`` (populations in the world fit and added
    afterwards), ``late1953`` (populations whose series starts after 1953),
    ``steps``, ``iter`` (per chain), ``rhat`` (largest R-hat of the world
    parameters other than k and lambda.k), ``rhat_k``, ``rhat_lk``,
    ``ess_min`` and ``ess_min_par`` (smallest effective sample size of a
    world parameter, and which), ``rl_ok`` and ``rl_needed`` (Raftery–Lewis:
    passed, further iterations it asks for, all chains together),
    ``median_spread``/``bound_spread`` (largest difference of the chains'
    forecasts, years), ``n_traj`` (trajectories of the final forecast),
    ``converged``, ``eq2_sigma``/``eq2_wpp`` (equation 2's σ; WPP-derived)
    and ``hours``.

    Raises
    ------
    ValueError if the populations the run checked are not the world-fit
    populations of :func:`split_overlap` for these jobs (a run cached under
    a different family register).
    """
    d = result["diagnostics"]
    last = d[d["loop"] == d["loop"].max()]
    world = last[last["kind"] == "world"].set_index("name")
    fcst = last[last["kind"] == "forecast"]
    main, extra = split_overlap(list(jobs_o["pop"]), families)
    code_to_pop = dict(zip(result["codes"]["country_code"], result["codes"]["pop"], strict=True))
    checked = sorted(code_to_pop[int(c)] for c in fcst["name"])
    if checked != sorted(main):
        raise ValueError(f"the run's world-fit populations {checked} are not those of the family "
                         f"register {sorted(main)}")
    gap = result["gap"].set_index("equation")
    return {"world": len(main), "extra": len(extra), "late1953": int((jobs_o["first_year"] > 1953).sum()),
            "steps": int(last["loop"].iloc[0]), "iter": int(last["iter"].iloc[0]),
            "rhat": float(world["rhat"].drop(["k", "lambda.k"]).max()), "rhat_k": float(world.loc["k", "rhat"]),
            "rhat_lk": float(world.loc["lambda.k", "rhat"]), "ess_min": float(world["ess"].min()),
            "ess_min_par": str(world["ess"].idxmin()), "rl_ok": bool(world["raftery_ok"].iloc[0]),
            "rl_needed": float(world["raftery_iter_needed"].iloc[0]),
            "median_spread": float(fcst["median_spread"].max()),
            "bound_spread": float(fcst[["lower_spread", "upper_spread"]].to_numpy().max()),
            "n_traj": int(last.loc[last["kind"] == "final", "n_traj"].iloc[0]),
            "converged": bool(last["converged"].iloc[0]), "eq2_sigma": float(gap.loc["eq2", "sigma"]),
            "eq2_wpp": bool(gap.loc["eq2", "substituted"]), "hours": float(result["timing"]["seconds"].sum()) / 3600}


def large_changes(e0: pd.DataFrame, jobs_o: pd.DataFrame) -> pd.DataFrame:
    """The annual changes of female e0 that bayesLife leaves out (outside ``OUTLIER_LIMITS``).

    Parameters
    ----------
    e0 : :func:`e0_series` (``pop, year, female, male``); jobs_o : one
        origin's jobs (each population's window ``first_year … last_train_year``).

    Returns
    -------
    DataFrame ``pop, year, change`` (years; the change from year − 1 to year,
    both years inside the window).
    """
    rows = []
    for job in jobs_o.itertuples():
        s = (e0[(e0["pop"] == job.pop) & e0["year"].between(job.first_year, job.last_train_year)]
             .set_index("year")["female"].sort_index())
        consecutive = (s.index.to_series().diff() == 1).to_numpy()
        change = s.diff()[consecutive]
        out = change[(change < OUTLIER_LIMITS[0]) | (change > OUTLIER_LIMITS[1])]
        rows += [{"pop": job.pop, "year": int(y), "change": float(v)} for y, v in out.items()]
    return pd.DataFrame(rows, columns=["pop", "year", "change"])


def points(result: Mapping, origin: int) -> dict:
    """One origin's bayesLife output in the harness's format (:func:`mdmx.forecast.run_benchmark`).

    Parameters
    ----------
    result
        The output of :func:`run` for this origin.
    origin
        The forecast origin (calendar year), recorded with the failures and warnings.

    Returns
    -------
    dict with ``points`` (``pop, sex, year, e0_fc`` = the predictive median,
    ``lower80``/``upper80`` = 10th/90th and ``lower95``/``upper95`` =
    2.5th/97.5th percentiles); ``failures`` and ``warnings`` with the
    comparators' columns ``method, pop, origin, message`` (``pop`` is missing:
    bayesLife reports them per run, not per population); and the R
    ``diagnostics`` and ``gap`` tables.
    """
    fcst = result["forecasts"].merge(result["codes"], on="country_code", how="left", validate="many_to_one")
    if fcst["pop"].isna().any():
        raise ValueError("bayesLife returned a population code that was not in its input")
    pts = pd.DataFrame({"pop": fcst["pop"].astype(str), "sex": fcst["sex"], "year": fcst["year"].astype(int),
                        "e0_fc": fcst["median"], "lower80": fcst["q10"], "upper80": fcst["q90"],
                        "lower95": fcst["q025"], "upper95": fcst["q975"]})
    logs = {k: pd.DataFrame({"method": "BL", "pop": pd.Series(np.nan, index=result[k].index, dtype=object),
                             "origin": int(origin), "message": result[k]["message"].astype(str)})
            for k in ("failures", "warnings")}
    return {"points": pts, **logs, **{k: result[k] for k in ("diagnostics", "gap")}}


def schedules(result: Mapping, smoothed: pd.DataFrame, jobs_o: pd.DataFrame, origin: int, *, n_ages: int,
              ax: float, tag: str, e0_tol: float, max_rounds: int) -> dict:
    """bayesLife's forecast schedules: MortCast's coherent rotated Lee–Carter from its median e₀ path.

    Paper: sec:evaluation:entries

    Decision 13 (Sam, 2026-10-29/30).  ``R/mortcast_schedules.R`` fits each population's coherent
    Lee–Carter parameters to its smoothed rates over the job's fitting window and finds, for every
    forecast year, the schedules whose life expectancy is bayesLife's median female and male e₀,
    rotating b_x as e₀ rises.  The rates become logit(q_x) as the comparators' do
    (:func:`mdmx.forecast.comparator_schedules`).

    MortCast matches e₀ in its own life table (a₀ by Andreev–Kingkade, open age extended to 130),
    whose e₀ exceeds the book's (:mod:`mdmx.lifetable`) by up to 0.4 years where infant mortality is
    high.  The schedules are scored in the book's life table, so MortCast's target is moved by the
    remaining difference until the schedules' e₀ there is the median to ``e0_tol`` years, in at
    most ``max_rounds`` further calls (Sam, 2026-10-31).  MortCast meets its own target only to
    about 0.01 years, which bounds the tolerance that can be reached.  ``e0_check`` gives the schedules' e₀ in the
    book's life table beside the median.

    Parameters
    ----------
    result : one run's outputs (:func:`run` or :func:`offsite_run`)
    smoothed : the comparators' smoothed rates (``forecast_cmp_smoothed``: ``pop, sex, year, age,
        mx``, ages 0 … 100, the last the open group)
    jobs_o : the run's jobs (``pop, first_year, last_train_year``); origin : the forecast origin
    n_ages : single-year ages of the schedules (110); ax : a_x of the m_x → q_x conversion
    tag : the R work folder
    e0_tol : largest |schedule e₀ − median| accepted, years
    max_rounds : calls of MortCast after the first, to close the life-table difference

    Returns
    -------
    dict with ``schedule_index`` (``pop, year``; years after the origin), ``schedules`` (N, 2,
    n_ages) logit(q_x), ``e0_check`` (``pop, year, female, male`` of the schedules and
    ``female_bl, male_bl`` of the median), ``rounds`` (MortCast's calls after the first) and the R
    ``warnings``.

    Raises
    ------
    RuntimeError if the schedules' e₀ is not within ``e0_tol`` of the median after ``max_rounds``.
    """
    pts = points(result, origin)["points"]
    pts = pts[(pts["year"] > origin) & pts["sex"].isin(["female", "male"])]
    med = pts.pivot_table(index=["pop", "year"], columns="sex", values="e0_fc").reset_index()
    med = med[["pop", "year", "female", "male"]]
    win = jobs_o[["pop", "first_year", "last_train_year"]]
    rates = smoothed.merge(win, on="pop", how="inner")
    rates = rates[(rates["year"] >= rates["first_year"]) & (rates["year"] <= rates["last_train_year"])]
    missing = sorted(set(med["pop"]) - set(rates["pop"]))
    if missing:
        raise ValueError(f"no smoothed rates for {missing}")
    rates = rates[["pop", "sex", "year", "age", "mx"]]
    target, rounds = med.copy(), 0
    while True:
        out = rbridge.run("mortcast_schedules.R", {"rates": rates, "e0": target}, ["mx", "failures", "warnings"],
                          tag=tag, settings={"rotate": True})
        wide = out["mx"].pivot_table(index=["pop", "sex", "year"], columns="age", values="mx")
        wide.columns = [f"a{int(a)}" for a in wide.columns]
        wide = wide.reset_index().assign(method="BL", origin=int(origin))
        index, z = fc.comparator_schedules(wide, n_ages, ax)
        e = lifetable.e0_from_logit(z)
        check = index[["pop", "year"]].assign(female=e[:, 0], male=e[:, 1]).merge(
            med.rename(columns={"female": "female_bl", "male": "male_bl"}), on=["pop", "year"], how="left",
            validate="one_to_one")
        gap = check[["female_bl", "male_bl"]].to_numpy() - check[["female", "male"]].to_numpy()
        if np.abs(gap).max() <= e0_tol:
            break
        if rounds == max_rounds:
            raise RuntimeError(f"MortCast's schedules from {origin} miss bayesLife's median e0 by "
                               f"{np.abs(gap).max():.4f} y after {max_rounds} rounds (tolerance {e0_tol})")
        rounds += 1
        # Move each year's target by its remaining difference (the targets keep med's row order).
        moved = target.merge(check[["pop", "year"]].assign(d_f=gap[:, 0], d_m=gap[:, 1]), on=["pop", "year"],
                             how="left", validate="one_to_one")
        target = moved.assign(female=moved["female"] + moved["d_f"], male=moved["male"] + moved["d_m"])[
            ["pop", "year", "female", "male"]]
    return {"schedule_index": index[["pop", "year"]], "schedules": z, "e0_check": check, "rounds": rounds,
            "warnings": out["warnings"]}
