"""Variants and benchmarks reported beside the forecast evaluation, never ranked (paper §12; chapter 12).

The ranked comparison (:mod:`mdmx.evaluation`) is not changed by anything here: this module only adds
runs and summaries (``docs/R1-plan.md``, Stages 4–5).  Every variant is computed and scored at the
**development origins only**, the eras of :class:`mdmx.evaluation.Design` (1900–2000); none is
computed or scored at the held-out origins (Sam, 2026-11-01).  Every run is fitted to data for years
up to its origin, with the exceptions chapter 12 lists for the ranked entries: the pooling blocks of
the data chapter (decision Pool), and the choice of the populations scored at an origin (observed
for at least ``min_test_years`` after it), which also decides Li–Lee's group.

The runs (each origin an artefact, so that the chapter and ``tools/warm_variants.py`` share them):

* :func:`run_flow_variants`: the flow field's other forms (:func:`flow_variant_forms`): FR0-all (the
  flat form), FR2-all (the form the frontier rule as written chose; Deviation 1) and the ablation of
  FR1-50 (planned before its results), with FR1-50 itself as the
  reference (the ranked entry "Flow field"; artefacts ``evaluation_ffvar_<origin>``);
* :func:`run_lilee`: Li–Lee's group form per sex and its sex-coherent form (:mod:`mdmx.lilee`;
  artefacts ``evaluation_llg_<origin>`` and ``evaluation_lls_<origin>``);
* :func:`run_mdmx_sensitivity`: anchored MDMx fitted to every year with one setting changed: the drift
  weights or the number of components (:func:`mdmx_sensitivity_settings`; artefacts
  ``evaluation_mdmx_<tag>_full_<origin>``);
* :func:`run_variants`: all of them from :func:`mdmx.evaluation.context`.

The summaries:

* :func:`variant_results` (the entries' format), :func:`band_errors` and :func:`band_wide` (errors by
  era and horizon band on the comparison's own points), :func:`variant_values` and
  :func:`ratios_to_naive` (every measure of the comparison relative to the naive);
* intervals from past errors only: :func:`past_calibration`, :func:`past_error_intervals`,
  :func:`past_error_quantiles` and :func:`interval_comparison`;
* the windowed naive: :func:`windowed_naive`, :func:`naive_values` and
  :func:`worse_than_naive`;
* the pooling disclosure (decision Pool): :func:`pooling_counts` and :func:`share_without`.

Conventions as in :mod:`mdmx.evaluation`: errors are forecast − observed; sex series ``female``,
``male`` and ``both`` (their average); index 0 of a sex axis is female.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import artifacts, data, flowfield, lifetable, lilee, rbridge
from . import compare as cp
from . import evaluation as ev
from . import forecast as fc
from . import svdcomp as sc

__all__ = [
    "FLOW_REFERENCE", "ABLATION", "OTHER_FORMS", "LILEE_LABELS", "NAIVE_WINDOW", "PARTS", "development_origins",
    "development_jobs", "flow_variant_forms", "run_flow_variants", "run_lilee", "mdmx_sensitivity_settings",
    "masked_event_counts",
    "run_mdmx_sensitivity", "run_variants", "variant_results", "development_keys", "available_keys",
    "band_errors", "band_wide", "variant_values", "ratios_to_naive", "past_calibration",
    "past_error_intervals", "past_error_quantiles", "interval_comparison", "windowed_naive", "naive_values",
    "worse_than_naive", "pooling_counts", "share_without", "FLOW_VARIANT_CODE", "LILEE_RUN_CODE",
    "unpooled_truth", "pooled_to_origin", "origin_tensor", "clean_pooling_runs", "clean_pooling_results",
]

#: The flow field's reference form: FR1-50 (the frontier rule's choice; Deviation 1), the ranked entry "Flow field".
FLOW_REFERENCE = "FR1-50"
#: The ablation's variants (as planned before its results), each one change from FR1-50.
ABLATION = ("no frontier rule", "no shape law", "deviations persist", "deviations vanish", "no anchor")
#: The frontier rule's other candidate forms, every part on every year: the flat form and the written rule's choice.
OTHER_FORMS = ("FR0-all", "FR2-all")
#: Li–Lee's two forms: method code (``[lilee]``) → the label of the tables.
LILEE_LABELS = {"group": "Li–Lee, group", "sex": "Li–Lee, sex-coherent"}
#: The label of the naive whose drift uses only the last ``window`` years.
NAIVE_WINDOW = "Naive, last 50 years"
#: The parts of :func:`run_variants`.
PARTS = ("flow", "lilee", "mdmx")

_KEY3 = ["pop", "origin", "year"]
_KEY4 = ["pop", "sex", "origin", "year"]


# ══════════════════════════════════════════════════════════════════════════════
# The development origins
# ══════════════════════════════════════════════════════════════════════════════

def development_origins(design: ev.Design) -> list[int]:
    """The origins of the eras (the development origins, 1900–2000), in order; never the held-out block."""
    return sorted(int(o) for origins in design.eras.values() for o in origins)


def _check_development(origins: Sequence[int], design: ev.Design) -> None:
    """Raise if an origin is not a development origin: no variant is computed at the held-out block."""
    outside = sorted(set(int(o) for o in origins) - set(development_origins(design)))
    if outside:
        raise ValueError(f"the variants are computed at the development origins only, not at {outside}")


def development_jobs(jobs: pd.DataFrame, design: ev.Design, origins: Sequence[int] | None = None) -> pd.DataFrame:
    """The jobs at the development origins (or at ``origins``, which must be development origins).

    Parameters
    ----------
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (any origins)
    design : the evaluation's design; origins : a subset of its development origins, or None for all
    """
    keep = development_origins(design) if origins is None else [int(o) for o in origins]
    _check_development(keep, design)
    out = jobs[jobs["origin"].isin(keep)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no jobs at the origins {keep}")
    return out


def _by_origin(jobs: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    return [(int(o), j.reset_index(drop=True)) for o, j in jobs.groupby("origin", sort=True)]


# ══════════════════════════════════════════════════════════════════════════════
# The runs
# ══════════════════════════════════════════════════════════════════════════════

def flow_variant_forms(base: flowfield.FlowFieldSettings) -> dict[str, flowfield.FlowFieldSettings]:
    """The flow field's variants: FR1-50 (the reference), FR0-all, FR2-all and the ablation.

    * ``FR1-50``: ``base``, the article's form, which it must be;
    * ``FR0-all``: the speed held at g(s₁*) beyond the frontier point, every part on every year (the
      flat form);
    * ``FR2-all``: the leaders' pace in e₀ through the typical schedule, every part on every year (the
      form the frontier rule as written chose; Deviation 1);
    * the ablation (as planned), one field changed each: ``no frontier rule`` (FR0 with FR1-50's
      windows), ``no shape law`` (s₂…s_p held at their jump-off values), ``deviations persist``
      (α_k = 1), ``deviations vanish`` (α_k = 0) and ``no anchor`` (no jump-off residual).

    Returns
    -------
    label → settings, ``FR1-50`` first.
    """
    if (base.frontier_rule, base.shape_law, base.relax, base.anchor) != ("FR1", True, "estimated", True):
        raise ValueError("the variants are defined from FR1-50: frontier rule FR1 with no part ablated")
    replace = dataclasses.replace
    every_year = {"speed_window": None, "shape_window": None}
    return {FLOW_REFERENCE: base,
            "FR0-all": replace(base, **every_year, frontier_rule="FR0"),
            "FR2-all": replace(base, **every_year, frontier_rule="FR2"),
            "no frontier rule": replace(base, frontier_rule="FR0"),
            "no shape law": replace(base, shape_law=False),
            "deviations persist": replace(base, relax="persist"),
            "deviations vanish": replace(base, relax="vanish"),
            "no anchor": replace(base, anchor=False)}


def run_flow_variants(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                      jobs: pd.DataFrame, mdmx_settings: fc.MDMxSettings,
                      forms: Mapping[str, flowfield.FlowFieldSettings], e0: np.ndarray, *, horizon: int,
                      design: ev.Design, name: str = "evaluation_ffvar", allow_held: bool = False) -> dict[int, dict]:
    """The flow field in each of ``forms`` at every origin of ``jobs``, each origin artefact ``<name>_<origin>``.

    Every form is forecast from the origin's space (:func:`mdmx.forecast.origin_space`, MDMx's
    settings) by :func:`mdmx.flowfield.forecast_origin`, which builds each form's flow with the space's
    linear map (rule FR2 needs it), and keeps its schedules.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); observed : (C, T); pops, years : their labels
    jobs : the jobs (:func:`development_jobs`); development origins only, unless ``allow_held``
    mdmx_settings : the space's settings; forms : :func:`flow_variant_forms`
    e0 : (C, T) e₀ (:func:`mdmx.flowfield.e0_matrix`); horizon : years forecast beyond the origin
    allow_held : also accept the held-out origins, for the held-out block's description of FR0-all and
        FR2-all; every other variant is reported on the development origins only, so the switch is explicit.

    Returns
    -------
    origin → dict with ``runs`` (form → ``points``, ``schedule_index``, ``schedules``, ``flow``),
    ``seconds`` and ``cpu_seconds`` (form → its run time; the space's time is not in them).
    """
    if not allow_held:
        _check_development(jobs["origin"].unique(), design)
    out = {}
    for origin, jo in _by_origin(jobs):
        def compute(origin=origin, jo=jo):
            space = fc.origin_space(M, observed, years, origin, tau=mdmx_settings.tau, age_rank=mdmx_settings.age_rank,
                                    smooth=mdmx_settings.smooth, n_pc=mdmx_settings.n_pc)
            runs, secs, cpus = {}, {}, {}
            for label, settings in forms.items():
                runs[label], secs[label], cpus[label] = ev.timed(
                    flowfield.forecast_origin, M, space, pops, years, jo, origin, horizon, e0, settings,
                    schedules=True)
            return {"runs": runs, "seconds": secs, "cpu_seconds": cpus}
        out[origin] = artifacts.stage(
            f"{name}_{origin}", compute, inputs=["tensor"],
            params={"mdmx": mdmx_settings.as_dict(), "forms": {k: v.as_dict() for k, v in forms.items()},
                    "horizon": int(horizon), "jobs": artifacts.content_hash(jo), "e0": artifacts.content_hash(e0)},
            code=[compute, *FLOW_VARIANT_CODE], desc=f"The flow field's {len(forms)} variant forms from {origin}")
    return out


def run_lilee(smoothed: pd.DataFrame, rates: pd.DataFrame, jobs: pd.DataFrame, settings: lilee.LiLeeSettings,
              families: Mapping[str, str], *, design: ev.Design, name: str = "evaluation") -> dict[str, dict[int, dict]]:
    """Li–Lee's two forms at every origin of ``jobs``, each form and origin an artefact.

    The group form (:func:`mdmx.lilee.group_forecasts`) is artefact ``<name>_<group code>_<origin>`` and
    the sex-coherent form (:func:`mdmx.lilee.sex_coherent_forecasts`) ``<name>_<sex code>_<origin>``,
    the codes from ``[lilee]`` in lower case (``evaluation_llg_1950``, ``evaluation_lls_1950``).

    Parameters
    ----------
    smoothed : the comparators' smoothed rates (artefact ``forecast_cmp_smoothed``)
    rates : the comparators' deaths and exposures (artefact ``forecast_cmp_rates``), Li–Lee's
        ``exposures``
    jobs : the jobs (:func:`development_jobs`); development origins only
    settings : :class:`mdmx.lilee.LiLeeSettings`
    families : population → its family's top-level series (the group form's core)

    Returns
    -------
    method code → origin → :func:`mdmx.lilee.group_forecasts` / :func:`~mdmx.lilee.sex_coherent_forecasts`'s
    output plus ``seconds`` and ``cpu_seconds``.
    """
    _check_development(jobs["origin"].unique(), design)
    out: dict[str, dict[int, dict]] = {settings.method_group: {}, settings.method_sex: {}}
    for origin, jo in _by_origin(jobs):
        fam = {p: families[p] for p in sorted(jo["pop"].unique())}

        def group(jo=jo, fam=fam):
            res, wall, cpu = ev.timed(lilee.group_forecasts, smoothed, jo, settings, families=fam, exposures=rates)
            return {**res, "seconds": wall, "cpu_seconds": cpu}

        def sexes(jo=jo):
            res, wall, cpu = ev.timed(lilee.sex_coherent_forecasts, smoothed, jo, settings, exposures=rates)
            return {**res, "seconds": wall, "cpu_seconds": cpu}

        for code, compute, extra, what in ((settings.method_group, group, {"families": fam}, "group form, per sex"),
                                           (settings.method_sex, sexes, {}, "sex-coherent form")):
            out[code][origin] = artifacts.stage(
                f"{name}_{code.lower()}_{origin}", compute, inputs=["forecast_cmp_rates", "forecast_cmp_smoothed"],
                params={"settings": settings.as_dict(), "jobs": artifacts.content_hash(jo), **extra},
                code=[compute, *LILEE_RUN_CODE], desc=f"Li-Lee forecasts ({what}) from {origin}")
    return out


def mdmx_sensitivity_settings(base: fc.MDMxSettings, F: Mapping) -> dict[str, tuple[str, fc.MDMxSettings]]:
    """MDMx's sensitivity variants: each changes one setting of ``base`` (the settings' earlier values).

    The drift weights ``[w_HMD, w_country]`` of ``[forecast] sens_drift_weights`` and the numbers of
    components of ``sens_n_pc``, each but ``base``'s own value; everything else as ``base``.

    Returns
    -------
    label (``"drift weights 1/0"``, ``"3 components"``, …) → (tag of the artefact names, ``hmd100``,
    ``npc3``, …; settings).
    """
    out = {}
    for w_hmd, w_country in F["sens_drift_weights"]:
        if np.isclose(w_hmd, base.weight_hmd) and np.isclose(w_country, base.weight_country):
            continue
        out[f"drift weights {w_hmd:g}/{w_country:g}"] = (
            f"hmd{round(100 * w_hmd)}", dataclasses.replace(base, weight_hmd=float(w_hmd),
                                                            weight_country=float(w_country)))
    for n in F["sens_n_pc"]:
        if int(n) != base.n_pc:
            out[f"{int(n)} components"] = (f"npc{int(n)}", dataclasses.replace(base, n_pc=int(n)))
    return out


def run_mdmx_sensitivity(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                         jobs: pd.DataFrame, variants: Mapping[str, tuple[str, fc.MDMxSettings]], *,
                         anchor: str | float, workers: int, design: ev.Design, name: str = "evaluation_mdmx",
                         log: Callable[[str], None] = print) -> dict[str, dict[int, dict]]:
    """Anchored MDMx, every year, in each of ``variants`` at every origin of ``jobs``.

    Each variant and origin is artefact ``<name>_<tag>_full_<origin>`` (as ``evaluation_mdmx_full_<origin>``
    for the ranked entry), made by :func:`mdmx.evaluation.run_mdmx`, whose key holds the settings, the
    anchor and the jobs.

    Parameters
    ----------
    jobs : the jobs of the every-year policy (:func:`development_jobs`); development origins only
    variants : :func:`mdmx_sensitivity_settings`; anchor : ``[evaluation] mdmx_anchor``
    workers : processes for the fits (run time only)

    Returns
    -------
    label → origin → :func:`mdmx.forecast.mdmx_origin`'s result, with ``seconds`` and ``cpu_seconds``.
    """
    _check_development(jobs["origin"].unique(), design)
    out = {}
    for label, (tag, settings) in variants.items():
        t0 = time.perf_counter()
        out[label] = ev.run_mdmx(M, observed, pops, years, jobs, settings, anchor=anchor, workers=workers,
                                 name=f"{name}_{tag}_full")
        log(f"MDMx, {label}: {jobs['origin'].nunique()} origins, {time.perf_counter() - t0:.0f} s")
    return out


def run_variants(ctx: Mapping, P, *, parts: Sequence[str] = PARTS, workers: int = 0,
                 origins: Sequence[int] | None = None, log: Callable[[str], None] = print) -> dict:
    """Every variant's runs from :func:`mdmx.evaluation.context` (each origin cached).

    Parts: ``flow`` (:func:`run_flow_variants`), ``lilee`` (:func:`run_lilee`) and ``mdmx``
    (:func:`run_mdmx_sensitivity`, the long one).  The jobs are the every-year policy's
    (``ctx["jobs"]["full"]``) at the development origins, or at ``origins``.

    Parameters
    ----------
    ctx : :func:`mdmx.evaluation.context`; P : the parameters (sections ``forecast``, ``lilee``)
    workers : processes for MDMx's fits (run time only)

    Returns
    -------
    part → its runs (``flow``: origin → result; ``lilee`` and ``mdmx``: form or label → origin → result).
    """
    D = ctx["design"]
    jobs = development_jobs(ctx["jobs"]["full"], D, origins)
    tensor = (ctx["M"], ctx["observed"], ctx["pops"], ctx["years"])
    out = {}
    for part in parts:
        t0 = time.perf_counter()
        if part == "flow":
            out[part] = run_flow_variants(*tensor, jobs, ctx["mdmx"], flow_variant_forms(ctx["flowfield"]), ctx["e0"],
                                          horizon=D.horizon, design=D)
        elif part == "lilee":
            families = dict(zip(ctx["pops"], sc.population_families(ctx["pops"], artifacts.load("populations",
                                                                                               quiet=True)),
                                strict=True))
            out[part] = run_lilee(ctx["smoothed"], ctx["rates"], jobs, lilee.LiLeeSettings.from_params(P.section("lilee")),
                                  families, design=D)
        elif part == "mdmx":
            out[part] = run_mdmx_sensitivity(*tensor, jobs, mdmx_sensitivity_settings(ctx["mdmx"], ctx["F"]),
                                             anchor=ctx["E"]["mdmx_anchor"], workers=workers, design=D, log=log)
        else:
            raise ValueError(f"unknown part {part!r}; the parts are {PARTS}")
        log(f"{part}: {time.perf_counter() - t0:.0f} s")
    return out


def variant_results(runs: Mapping, *, n_ages: int, ax: float,
                    lilee_codes: Mapping[str, str] | None = None) -> dict[str, dict[int, dict]]:
    """Every variant's results per origin in the entries' format (:func:`mdmx.evaluation.entry_results`).

    Parameters
    ----------
    runs : :func:`run_variants` (any of its parts)
    n_ages, ax : the comparators' m → q conversion (:func:`mdmx.forecast.comparator_points`), for Li–Lee
    lilee_codes : ``{"group": code, "sex": code}``, ``[lilee]``'s method codes (needed with ``lilee``)

    Returns
    -------
    label → origin → ``points`` (``pop, sex, year, e0_fc``), ``schedule_index`` (``pop, year``) and
    ``schedules`` (N, 2, A).  Labels: the flow forms (:func:`flow_variant_forms`), :data:`LILEE_LABELS`'
    and ``MDMx, <variant>`` (:func:`mdmx_sensitivity_settings`).
    """
    out: dict[str, dict[int, dict]] = {}
    if "flow" in runs:
        forms = next(iter(runs["flow"].values()))["runs"]
        for label in forms:
            out[label] = {o: r["runs"][label] for o, r in runs["flow"].items()}
    if "lilee" in runs:
        if lilee_codes is None:
            raise ValueError("Li-Lee's results need its method codes (lilee_codes)")
        for form, code in lilee_codes.items():
            out[LILEE_LABELS[form]] = {o: fc.comparator_points(r, code, n_ages, ax) for o, r in runs["lilee"][code].items()}
    if "mdmx" in runs:
        for label, by_origin in runs["mdmx"].items():
            out[f"MDMx, {label}"] = dict(by_origin)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Scoring on the comparison's points
# ══════════════════════════════════════════════════════════════════════════════

def development_keys(keys: pd.DataFrame, design: ev.Design) -> pd.DataFrame:
    """The scored points ``keys`` (``pop, origin, year``, …) at the development origins only."""
    out = keys[keys["origin"].isin(development_origins(design))].reset_index(drop=True)
    if out.empty:
        raise ValueError("no scored point at a development origin")
    return out


def available_keys(results: Mapping[str, Mapping[int, Mapping]], keys: pd.DataFrame) -> pd.DataFrame:
    """The points of ``keys`` at which every entry of ``results`` forecasts all three sex series (finite)
    and has a finite schedule.

    Parameters
    ----------
    results : entry → origin → ``points``, ``schedule_index``, ``schedules`` (N, 2, A)
    keys : the scored points ``pop, origin, year`` (other columns are kept)

    Returns
    -------
    The rows of ``keys`` that every entry can be scored on, in ``keys``' order.
    """
    out = keys
    for by_origin in results.values():
        pts = ev.long_points({"x": by_origin})
        pts = pts[np.isfinite(pts["e0_fc"])]
        n = pts.groupby(_KEY3)["sex"].nunique()
        full = n[n == len(fc.SEXES)].reset_index()[_KEY3]
        sched = pd.concat([r["schedule_index"][["pop", "year"]].assign(
            origin=int(o), _ok=np.isfinite(r["schedules"]).all(axis=(1, 2))) for o, r in by_origin.items()])
        full = full.merge(sched[sched["_ok"]][_KEY3], on=_KEY3)
        out = out.merge(full, on=_KEY3)
    return out.reset_index(drop=True)


def _observed_schedules(M: np.ndarray, pops: Sequence[str], years: Sequence[int], keys: pd.DataFrame) -> np.ndarray:
    """(len(keys), 2, A) observed logit(q_x) at the points ``keys``."""
    pidx = {p: i for i, p in enumerate(pops)}
    years = np.asarray(years)
    return np.moveaxis(M[:, :, keys["pop"].map(pidx).to_numpy(), (keys["year"] - years[0]).to_numpy()], -1, 0)


#: The measures of :func:`band_errors`: column of :func:`mdmx.evaluation.schedule_measures` → name.
_SCHEDULE_MEASURES = {"rmse_schedule_dx": "schedule_dx"}


def band_errors(results: Mapping[str, Mapping[int, Mapping]], truth: pd.DataFrame, keys: pd.DataFrame,
                design: ev.Design, M: np.ndarray, pops: Sequence[str], years: Sequence[int],
                groups: Mapping[str, Sequence[int]], *, old_age: int, common: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Errors by era and horizon band of each entry, on the comparison's points at the development origins.

    The measures (the ablation's plan):

    * ``e0_both``, ``e0_female``, ``e0_male``: e₀ of the average of the sexes, female and male (MAE
      and mean error);
    * ``gap``: the female − male gap of e₀ (MAE and mean error);
    * ``schedule_dx``: the RMSE of logit(q_x) weighted by the observed deaths d_x
      (:func:`mdmx.evaluation.schedule_measures`), its mean over the points in ``mae`` (``me`` is NaN);
    * ``e{old_age}_female``, ``e{old_age}_male``: life expectancy at ``old_age`` (MAE and mean error).

    The e₀ errors are :func:`mdmx.evaluation.errors` and the bands :func:`mdmx.evaluation.band_table`,
    as in the chapter's band tables, so the numbers are comparable with them.

    Parameters
    ----------
    results : entry → origin → ``points``, ``schedule_index``, ``schedules`` (:func:`variant_results`,
        :func:`mdmx.evaluation.entry_results`)
    truth : ``pop, sex, year, e0_obs``; keys : the comparison's scored points (any origins: only the
        development origins are used)
    design : the eras and bands; M : (2, A, C, T) observed logit(q_x); pops, years : its labels
    groups : age group → ages (the weights of :func:`mdmx.forecast.schedule_errors`); old_age : e.g. 65
    common : True scores every entry on the points all of them can be scored on
        (:func:`available_keys`); False each on its own

    Returns
    -------
    (``entry, era, band, measure, mae, me, n``; ``entry, keys, scored``: the development points
    given and those each entry is scored on).
    """
    keys = development_keys(keys, design)
    shared = available_keys(results, keys) if common else None
    frames, used = [], []
    for label, by_origin in results.items():
        k = shared if common else available_keys({label: by_origin}, keys)
        used.append({"entry": label, "keys": len(keys), "scored": len(k)})
        if k.empty:
            continue
        e = ev.errors(ev.long_points({label: by_origin}), truth, k)
        s = ev.schedule_scores({label: by_origin}, k[_KEY3], M, pops, years, groups, old_age=old_age)[label]
        cols = {**_SCHEDULE_MEASURES, f"err_e{old_age}female": f"e{old_age}_female",
                f"err_e{old_age}male": f"e{old_age}_male"}
        s_long = [pd.DataFrame({"entry": label, "pop": s["pop"], "origin": s["origin"], "year": s["year"],
                                "h": s["year"] - s["origin"], "measure": name, "error": s[col]})
                  for col, name in cols.items()]
        frames += [e[["entry", "pop", "origin", "year", "h", "measure", "error"]], *s_long]
    errs = pd.concat(frames, ignore_index=True)
    errs["era"] = errs["origin"].map({o: design.era(o) for o in errs["origin"].unique()})
    tab = ev.band_table(errs, design, ["entry", "era", "measure"])
    tab.loc[tab["measure"].isin(list(_SCHEDULE_MEASURES.values())), "me"] = np.nan   # an RMSE has no sign
    return tab, pd.DataFrame(used)


def band_wide(table: pd.DataFrame, measure: str, stat: str, design: ev.Design,
              entries: Sequence[str] | None = None) -> pd.DataFrame:
    """One measure and statistic of :func:`band_errors` as a table: entry × (era, band), the development eras.

    Parameters
    ----------
    table : :func:`band_errors`' first output; measure : e.g. ``"e0_both"``; stat : ``"mae"``, ``"me"`` or ``"n"``
    entries : the rows, in order (default: as they come)
    """
    t = table[table["measure"] == measure]
    wide = t.pivot_table(index="entry", columns=["era", "band"], values=stat, observed=True, sort=False)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([list(design.eras), design.band_labels()]))
    return wide.reindex(index=list(entries)) if entries is not None else wide


def variant_values(results: Mapping[str, Mapping[int, Mapping]], truth: pd.DataFrame, keys: pd.DataFrame,
                   design: ev.Design, M: np.ndarray, pops: Sequence[str], years: Sequence[int],
                   groups: Mapping[str, Sequence[int]], *, old_age: int, levels: Sequence[float],
                   pi_levels: Sequence[int], calibration: Mapping[int, Sequence[int]], min_points: int,
                   families: Mapping[str, str]) -> pd.DataFrame:
    """Every measure of the overall comparison for each variant, at the development points it can be scored on.

    As the chapter's :func:`mdmx.evaluation.point_values` for the entries: the e₀ measures, the
    schedule measures (:func:`mdmx.evaluation.schedule_scores`) and the weighted interval score of the
    intervals by the comparison's one rule, the empirical quantiles of the variant's own errors at
    the other origins of its era (:func:`mdmx.evaluation.empirical_quantiles`).

    Parameters
    ----------
    results : variant → origin → result (:func:`variant_results`); truth, keys, design, M, pops, years,
        groups, old_age : as :func:`band_errors`
    levels, pi_levels : the quantile levels and nominal coverages; calibration : origin → the origins
        that calibrate it (:func:`mdmx.flowfield.within_era_calibration`); min_points : the fewest
        errors an interval cell needs
    families : population → its family (the bootstrap's cluster)

    Returns
    -------
    :func:`mdmx.evaluation.point_values`' rows for every variant (``method`` is the variant's label).
    """
    keys = development_keys(keys, design)
    points = ev.long_points(results)
    frames = []
    for label, by_origin in results.items():
        k = available_keys({label: by_origin}, keys)
        if k.empty:
            continue
        q, _ = ev.empirical_quantiles(points, truth, k, [label], calibration=calibration, pi_levels=pi_levels,
                                      min_points=min_points)
        s = ev.schedule_scores({label: by_origin}, k[_KEY3], M, pops, years, groups, old_age=old_age)
        frames.append(ev.point_values(points[points["entry"] == label], truth, k, q, s, levels, families, design))
    return pd.concat(frames, ignore_index=True)


def _scaled(values: pd.DataFrame, families: Mapping[str, Sequence[str]], composite: Sequence[str],
            methods: Sequence[str], naive: str) -> pd.DataFrame:
    """Every measure divided by the naive's on the whole sample (:func:`mdmx.compare.overall`'s ``scaled``)."""
    # The scaled measures come from the whole sample, so one draw without a bootstrap gives them exactly.
    return cp.overall(values, families, composite, methods=methods, naive=naive, draws=1, seed=0,
                      bootstrap=False)["scaled"]


def ratios_to_naive(values: pd.DataFrame, extra: pd.DataFrame, *, families: Mapping[str, Sequence[str]],
                    composite: Sequence[str], naive: str) -> pd.DataFrame:
    """Each method of ``extra``: every measure divided by the naive's on the same points (whole sample).

    Each method is compared with the naive on the points both have every measure at
    (:func:`mdmx.evaluation.complete`), so a variant that misses some points is still compared like
    with like.

    Parameters
    ----------
    values : the comparison's values of a view (the naive's rows are used); extra : the variants'
        values (:func:`variant_values`) of the same view
    families, composite : the comparison's families of measures and the composite used
    naive : the naive's label in ``values``

    Returns
    -------
    One row per method of ``extra``: ``method``, ``points`` and one column per measure code (below 1
    beats the naive).
    """
    codes = [c for f in composite for c in families[f]]
    base = values[values["method"] == naive]
    rows = []
    for label in extra["method"].unique():
        v = extra[extra["method"] == label]
        both, _ = ev.complete(pd.concat([v[v["point"].isin(base["point"])], base[base["point"].isin(v["point"])]],
                                        ignore_index=True))
        if both.empty:
            raise ValueError(f"{label} has no point with every measure in common with the naive")
        scaled = _scaled(both, families, composite, [label], naive)
        rows.append({"method": label, "points": int(both["point"].nunique()), **scaled.loc[label, codes].to_dict()})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Intervals from past errors only
# ══════════════════════════════════════════════════════════════════════════════

def past_calibration(origins: Sequence[int], h: int) -> dict[int, list[int]]:
    """The origins whose errors at horizon ``h`` had been observed by each origin (they calibrate its
    interval from past errors only).

    Paper: eq:ev-past-intervals, sec:evaluation:robustness

    At origin o, the error of an earlier origin o′ at horizon h is known once its outcome, the year
    o′ + h, has been observed: o′ + h ≤ o.  The within-era rule
    (:func:`mdmx.flowfield.within_era_calibration`) uses the other origins of the era, later ones
    included.

    Parameters
    ----------
    origins : the origins (calendar years); h : the horizon (years, ≥ 1)

    Returns
    -------
    origin → the origins o′ of ``origins`` with o′ + h ≤ o (never o itself).
    """
    if int(h) < 1:
        raise ValueError(f"the horizon must be at least 1, not {h}")
    ordered = sorted(int(o) for o in origins)
    return {o: [p for p in ordered if p + int(h) <= o] for o in ordered}


def past_error_intervals(points: pd.DataFrame, origins: Sequence[int], pi_levels: Sequence[int],
                         min_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictive quantiles of e₀ from one method's errors that were observed by the origin.

    Paper: *eq:ev-past-intervals, sec:evaluation:robustness

    The empirical rule of :func:`mdmx.flowfield.empirical_intervals` (quantile at level p:
    ``e0_fc − Q_{1−p}(E)``) with E the errors at the same horizon h and sex series of the origins
    o′ with o′ + h ≤ o (:func:`past_calibration`): only outcomes observed by the origin.  A cell
    with fewer than ``min_points`` errors gets no interval and is reported.

    Parameters
    ----------
    points : one method's forecasts, ``pop, sex, origin, year, e0_fc, e0_obs`` (NaN where not scored)
    origins : the origins that may calibrate (the development origins); points at other origins are
        dropped
    pi_levels : nominal coverages (per cent); min_points : the fewest errors a cell needs

    Returns
    -------
    (``points`` at ``origins`` with ``h`` and the quantile columns; the cells without an interval,
    ``origin, sex, h, n_errors``).
    """
    pts = points[points["origin"].isin([int(o) for o in origins])]
    if pts.empty:
        raise ValueError(f"no forecast at the origins {sorted(origins)}")
    h = (pts["year"] - pts["origin"]).astype(int)
    outs, sparse = [], []
    # The calibration set depends on the horizon, so each horizon is one call of the empirical rule.
    for hv, part in pts.groupby(h, sort=True):
        q, sp = flowfield.empirical_intervals(part, past_calibration(origins, int(hv)), pi_levels, min_points)
        outs.append(q)
        sparse.append(sp)
    return pd.concat(outs).sort_index(), pd.concat(sparse, ignore_index=True)


def past_error_quantiles(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame, entries: Sequence[str],
                         *, design: ev.Design, pi_levels: Sequence[int], min_points: int
                         ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every entry's intervals from past errors only (:func:`past_error_intervals`) at the development
    origins.

    Paper: eq:ev-past-intervals, sec:evaluation:robustness

    Only the observations at the scored development points calibrate
    (:func:`mdmx.evaluation.observed_at`), as in :func:`mdmx.evaluation.empirical_quantiles`.

    Parameters
    ----------
    points : forecasts ``entry, pop, sex, origin, year, e0_fc``; truth : ``pop, sex, year, e0_obs``;
    keys : the comparison's scored points; entries : the entries; design : the eras
    pi_levels, min_points : as :func:`past_error_intervals`

    Returns
    -------
    (entry → ``pop, sex, origin, year`` and the quantile columns, NaN in a cell without an interval;
    the cells without one, with an ``entry`` column).
    """
    dev = development_origins(design)
    keys = development_keys(keys, design)
    qcols = cp.quantile_columns(cp.quantile_levels(pi_levels))
    out, sparse = {}, []
    for label in entries:
        pts = points[(points["entry"] == label) & points["origin"].isin(dev)][_KEY4 + ["e0_fc"]]
        q, sp = past_error_intervals(ev.observed_at(pts, truth, keys), dev, pi_levels, min_points)
        out[label] = q[_KEY4 + qcols].reset_index(drop=True)
        sparse.append(sp.assign(entry=label))
    return out, pd.concat(sparse, ignore_index=True)


# The development book (book/, the record frozen at tag dev-book-r1) calls these by their first names.
realtime_calibration, realtime_intervals, realtime_quantiles = (past_calibration, past_error_intervals,
                                                                past_error_quantiles)


def interval_comparison(rules: Mapping[str, Mapping[str, pd.DataFrame]], keys: pd.DataFrame, truth: pd.DataFrame,
                        design: ev.Design, pi_levels: Sequence[int], spans: Mapping[str, tuple[int, int]]
                        ) -> pd.DataFrame:
    """Coverage and weighted interval score of several interval rules, entry by entry, on common points.

    For each entry, the points (population, sex, origin, year) are those at which every rule gives
    it an interval, among the scored development points, so the rules are compared like with like.

    Parameters
    ----------
    rules : rule name → entry → ``pop, sex, origin, year`` and the quantile columns
        (e.g. ``{"within era": …, "past errors": …}``); entries missing from a rule are left out
    keys : the comparison's scored points; truth : ``pop, sex, year, e0_obs``; design : the eras
    pi_levels : nominal coverages, per cent; spans : label → (first, last) horizon, inclusive

    Returns
    -------
    ``rule, entry, era, sex, span, n``, ``in{L}`` (the share inside the level-L interval) for each
    level and ``wis`` (the mean weighted interval score, years).
    """
    levels = cp.quantile_levels(pi_levels)
    qcols = cp.quantile_columns(levels)
    names = dict(zip(np.round(levels, 6), qcols, strict=True))
    keys = development_keys(keys, design)[_KEY3]
    first, *others = rules.values()
    entries = [e for e in first if all(e in r for r in others)]
    rows = []
    for label in entries:
        scored = {}
        for rule, by_entry in rules.items():
            q = by_entry[label].merge(keys, on=_KEY3).merge(truth[["pop", "sex", "year", "e0_obs"]],
                                                            on=["pop", "sex", "year"])
            scored[rule] = q.dropna(subset=qcols)
        common = None
        for q in scored.values():
            common = q[_KEY4] if common is None else common.merge(q[_KEY4], on=_KEY4)
        for rule, q in scored.items():
            s = q.merge(common, on=_KEY4)
            if s.empty:
                continue
            h = s["year"] - s["origin"]
            s = s.assign(era=s["origin"].map({o: design.era(o) for o in s["origin"].unique()}),
                         wis=cp.wis_from_quantiles(s["e0_obs"].to_numpy(), s[qcols].to_numpy(dtype=float), levels))
            inside = []
            for lev in pi_levels:
                lo, hi = round((100 - lev) / 200, 6), round(1 - (100 - lev) / 200, 6)
                s[f"in{lev}"] = (s["e0_obs"] >= s[names[lo]]) & (s["e0_obs"] <= s[names[hi]])
                inside.append(f"in{lev}")
            for span, (a, z) in spans.items():
                g = s[h.between(a, z)].groupby(["era", "sex"])
                t = g[inside + ["wis"]].mean().assign(n=g.size())
                rows.append(t.reset_index().assign(rule=rule, entry=label, span=span))
    cols = ["rule", "entry", "era", "sex", "span", "n", *[f"in{lev}" for lev in pi_levels], "wis"]
    return pd.concat(rows, ignore_index=True)[cols]


# ══════════════════════════════════════════════════════════════════════════════
# The windowed naive
# ══════════════════════════════════════════════════════════════════════════════

def windowed_naive(truth: pd.DataFrame, jobs: pd.DataFrame, window: int, levels: Sequence[float]) -> pd.DataFrame:
    """The naive random walk with drift on e₀, its drift and variance from the last ``window`` years only.

    :func:`mdmx.compare.naive_e0` on the jobs restricted to their last ``window`` years
    (:func:`mdmx.forecast.restrict_fit_window`): the drift is ``(x_b − x_a) / (b − a)`` with a the first
    observed year of ``last_train_year − window + 1 … last_train_year`` and b the last, instead of the
    population's first observed year (the standard naive).

    Parameters
    ----------
    truth : ``pop, sex, year, e0_obs``; jobs : rows of :func:`mdmx.forecast.cv_jobs`
    window : years; levels : quantile levels (:func:`mdmx.compare.quantile_levels`)

    Returns
    -------
    :func:`mdmx.compare.naive_e0`'s frame: ``pop, sex, origin, year, e0_fc, sd`` and the quantile columns.
    """
    return cp.naive_e0(truth, fc.restrict_fit_window(jobs, int(window)), levels)


def naive_values(truth: pd.DataFrame, M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                 jobs: pd.DataFrame, keys: pd.DataFrame, design: ev.Design, groups: Mapping[str, Sequence[int]], *,
                 window: int | None, old_age: int, levels: Sequence[float], pi_levels: Sequence[int],
                 calibration: Mapping[int, Sequence[int]], min_points: int, families: Mapping[str, str],
                 label: str) -> pd.DataFrame:
    """Every measure of the comparison for a naive, at the scored development points.

    The naive as the comparison scores it: a random walk with drift on e₀ (:func:`mdmx.compare.naive_e0`)
    and on each age's logit(q_x) (:func:`mdmx.compare.naive_schedules`), its intervals by the
    comparison's one rule (:func:`mdmx.evaluation.empirical_quantiles`).  With ``window`` every random
    walk is fitted to the job's last ``window`` years only (:func:`mdmx.forecast.restrict_fit_window`;
    the windowed naive, :data:`NAIVE_WINDOW`); with None to the population's whole series (the
    comparison's naive).

    Parameters
    ----------
    truth, M, observed, pops, years : the data; jobs : the every-year jobs (any origins: the
        development origins' are used); keys : the comparison's scored points; design : the eras
    groups, old_age : the schedule measures' age groups and old age; window : years, or None
    levels, pi_levels, calibration, min_points, families : as :func:`variant_values`; label : the
        method's label

    Returns
    -------
    :func:`mdmx.evaluation.point_values`' rows (``method`` = ``label``).
    """
    keys = development_keys(keys, design)
    jw = development_jobs(jobs, design)
    if window is not None:
        jw = fc.restrict_fit_window(jw, int(window))
    nq = cp.naive_e0(truth, jw, levels)
    pts = nq.assign(entry=label, h=nq["year"] - nq["origin"])[["entry", *_KEY4, "h", "e0_fc"]]
    q, _ = ev.empirical_quantiles(pts, truth, keys, [label], calibration=calibration, pi_levels=pi_levels,
                                  min_points=min_points)
    z = cp.naive_schedules(M, observed, pops, years, jw, keys)
    sched = {label: keys[_KEY3].assign(**ev.schedule_measures(z, _observed_schedules(M, pops, years, keys), groups,
                                                                old_age))}
    return ev.point_values(pts, truth, keys, q, sched, levels, families, design)


def worse_than_naive(values: pd.DataFrame, windowed_values: pd.DataFrame, *, families: Mapping[str, Sequence[str]],
                     composite: Sequence[str], methods: Sequence[str], naive: str) -> pd.DataFrame:
    """The number of measures on which each method is worse than the naive, and than the windowed naive.

    The measures are scaled as in the comparison (:func:`mdmx.compare.overall`, whole sample), once by
    the naive of ``values`` (``tbl-ev-measures``' count) and once by the windowed naive.

    Parameters
    ----------
    values : the comparison's values of a view (the methods and ``naive``), complete
    windowed_values : the windowed naive's values (:func:`naive_values`, one method), on the same
        points
    families, composite : the families and the composite (its measures are counted)
    methods : the ranked methods; naive : the naive's label in ``values``

    Returns
    -------
    One row per method: ``method``, ``worse`` (than the naive), ``worse_windowed`` and, for each measure
    code, ``<code>`` (the ratio to the windowed naive).  Raises if the windowed naive lacks a point of
    ``values`` or a measure.
    """
    labels = windowed_values["method"].unique()
    if len(labels) != 1:
        raise ValueError(f"windowed_values must hold one method, not {list(labels)}")
    label = labels[0]
    codes = [c for f in composite for c in families[f]]
    nv = windowed_values[windowed_values["point"].isin(values["point"])]
    missing = set(values["point"]) - set(nv["point"])
    if missing:
        raise ValueError(f"the windowed naive has no values at {len(missing)} points of the view")
    if nv[codes].isna().any().any():
        raise ValueError("the windowed naive lacks a measure at some point of the view")
    standard = _scaled(values, families, composite, methods, naive)
    windowed = _scaled(pd.concat([values[values["method"].isin(methods)], nv], ignore_index=True), families, composite,
                       methods, label)
    out = pd.DataFrame({"method": list(methods),
                        "worse": (standard.loc[list(methods), codes] > 1).sum(axis=1).to_numpy(),
                        "worse_windowed": (windowed.loc[list(methods), codes] > 1).sum(axis=1).to_numpy()})
    return out.join(windowed.loc[list(methods), codes].reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# Pooling blocks that span an origin (decision Pool)
# ══════════════════════════════════════════════════════════════════════════════

def pooling_counts(blocks: pd.DataFrame, jobs: pd.DataFrame, design: ev.Design) -> dict:
    """How often the data chapter's pooling blocks reach past a forecast origin (decision Pool).

    The blocks are formed once on each population's whole series (chapter d5), so a block of years
    ``first … last`` whose deaths and exposures are pooled spans an origin o when first ≤ o < last:
    the pooled schedules of its years up to o contain deaths from after o.  A job (population and
    origin) is affected when its jump-off year (``last_train_year``) is one of the years of a block of
    the population, either sex, that extends past the origin (last > o).

    Parameters
    ----------
    blocks : artefact ``lt_pooled``'s ``blocks``: ``pop, sex, first, last, years`` (the block's years,
        space-separated)
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (``pop, origin, last_train_year``), every origin
    design : the origins and eras

    Returns
    -------
    dict with
      ``overlaps``: one row per block and origin it spans (``pop, sex, first, last, origin, era,
      past`` = last − origin, years);
      ``jobs``: the affected jobs (``pop, origin, era, last_train_year, sexes, last``: the latest year
      of the blocks concerned);
      ``summary``: one row per era (``early``, ``late``, ``held``) and ``all``: ``overlaps``,
      ``overlap_populations``, ``jobs``, ``affected_jobs`` and ``affected_populations``;
      ``populations``: the populations with an overlap (sorted); ``affected_populations``: era →
      the populations with an affected job (sorted).
    """
    b = blocks[["pop", "sex", "first", "last", "years"]].reset_index(drop=True)
    era = {int(o): design.era(o) for o in design.origins}
    rows = [{"pop": r.pop, "sex": r.sex, "first": int(r.first), "last": int(r.last), "origin": o, "era": era[o],
             "past": int(r.last) - o} for r in b.itertuples() for o in era if r.first <= o < r.last]
    overlaps = pd.DataFrame(rows, columns=["pop", "sex", "first", "last", "origin", "era", "past"])
    members = [{int(y) for y in str(r.years).split()} for r in b.itertuples()]
    bad = []
    for job in jobs.itertuples():
        hit = [i for i in np.flatnonzero(b["pop"].to_numpy() == job.pop)
               if int(job.last_train_year) in members[i] and int(b.at[i, "last"]) > int(job.origin)]
        if hit:
            bad.append({"pop": job.pop, "origin": int(job.origin), "era": era[int(job.origin)],
                        "last_train_year": int(job.last_train_year),
                        "sexes": ",".join(sorted(set(b.loc[hit, "sex"]))), "last": int(b.loc[hit, "last"].max())})
    affected = pd.DataFrame(bad, columns=["pop", "origin", "era", "last_train_year", "sexes", "last"])
    job_era = jobs["origin"].astype(int).map(era)
    summary = []
    for name in [*design.eras, "held", "all"]:
        o_ = overlaps if name == "all" else overlaps[overlaps["era"] == name]
        a_ = affected if name == "all" else affected[affected["era"] == name]
        summary.append({"era": name, "overlaps": len(o_), "overlap_populations": o_["pop"].nunique(),
                        "jobs": int(len(jobs) if name == "all" else (job_era == name).sum()),
                        "affected_jobs": len(a_), "affected_populations": a_["pop"].nunique()})
    return {"overlaps": overlaps, "jobs": affected, "summary": pd.DataFrame(summary),
            "populations": sorted(overlaps["pop"].unique()),
            "affected_populations": {name: sorted(affected.loc[affected["era"] == name, "pop"].unique())
                                     for name in [*design.eras, "held"]}}


def masked_event_counts(events: pd.DataFrame, has_data: np.ndarray, observed: np.ndarray, years: Sequence[int],
                        pops: Sequence[str], jobs: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Population-years and forecast years that events masked from the truth leave unscored (decision R1).

    Paper: sec:evaluation:design

    Parameters
    ----------
    events : rows of :func:`mdmx.data.load_events` to count (``event, start, end``), e.g. those masked
        in every population
    has_data : (C, T) bool, the cell has rates (a masked year keeps its data)
    observed : (C, T) bool, the cell is part of the truth (False in a masked year)
    years : the T calendar years; pops : the C populations
    jobs : the scored jobs (``pop, origin``)
    horizon : the longest horizon scored, years

    Returns
    -------
    One row per event: ``event, start, end``, ``popyears`` (cells with data left out of the truth) and
    ``forecast`` (job × year pairs with 0 < year − origin ≤ horizon that fall in those cells).
    """
    year_arr = np.asarray(years)
    c_of = {p: i for i, p in enumerate(pops)}
    rows = []
    for e in events.itertuples():
        t = (year_arr >= e.start) & (year_arr <= e.end)
        cells = has_data[:, t] & ~observed[:, t]
        yrs = year_arr[t]
        n_fc = sum(int(cells[c_of[j.pop], (yrs > j.origin) & (yrs <= j.origin + horizon)].sum())
                   for j in jobs.itertuples())
        rows.append({"event": e.event, "start": int(e.start), "end": int(e.end), "popyears": int(cells.sum()),
                     "forecast": n_fc})
    return pd.DataFrame(rows, columns=["event", "start", "end", "popyears", "forecast"])


def share_without(values: pd.DataFrame, populations: Sequence[str] = (), *, jobs: pd.DataFrame | None = None,
                  families: Mapping[str, Sequence[str]], composite: Sequence[str], methods: Sequence[str],
                  naive: str, draws: int, seed: int) -> dict:
    """The overall comparison of a view again, without some populations or jobs (descriptive; decision Pool).

    Parameters
    ----------
    values : the comparison's values of the view (:func:`mdmx.evaluation.point_values`, complete;
        ``pop`` and ``origin`` columns)
    populations : the populations to leave out entirely (e.g. those whose pooling blocks span an
        origin, :func:`pooling_counts`)
    jobs : rows ``pop, origin`` whose points to leave out (e.g. :func:`pooling_counts`' ``jobs``, the
        jobs whose jump-off year is pooled with later years); None leaves out none
    families, composite, methods, naive, draws, seed : as :func:`mdmx.compare.overall`

    Returns
    -------
    dict with ``first`` and ``mean_rank`` (Series by method), ``points_left_out`` and
    ``populations_left_out`` (the populations with a point left out).
    """
    drop = values["pop"].isin(set(populations))
    if jobs is not None and len(jobs):
        pairs = set(zip(jobs["pop"], jobs["origin"].astype(int), strict=True))
        drop |= pd.Series([(p, int(o)) in pairs for p, o in zip(values["pop"], values["origin"], strict=True)],
                          index=values.index)
    kept = values[~drop]
    if kept.empty:
        raise ValueError("no point is left without these populations")
    res = cp.overall(kept, families, composite, methods=methods, naive=naive, draws=draws, seed=seed)
    return {"first": res["first"], "mean_rank": res["mean_rank"],
            "points_left_out": int(values.loc[drop, "point"].nunique()),
            "populations_left_out": int(values.loc[drop, "pop"].nunique())}


#: The code of a flow-field variant run besides its own (its cache key follows the calls from these).
FLOW_VARIANT_CODE = (ev.timed, flowfield.forecast_origin, fc.origin_space, *flowfield.FLOW_CODE)
#: The code of a Li–Lee run besides its own.
LILEE_RUN_CODE = (ev.timed, *lilee.LILEE_CODE)


# ══════════════════════════════════════════════════════════════════════════════
# The pooling and the truth: the published tables, and pooling planned on the years up to the origin
# ══════════════════════════════════════════════════════════════════════════════

def unpooled_truth(lt_curated: pd.DataFrame, labels: pd.DataFrame, populations: pd.DataFrame,
                   tensor: Mapping, *, q_min: float, min_observed_years: int) -> pd.DataFrame:
    """Life expectancy of every observed country-year from the curated tables without pooling.

    Paper: sec:evaluation:robustness

    The tensor of chapter d6 built from the curated life tables instead of the pooled ones
    (:func:`mdmx.data.build_tensor`, the same floor), scored like the comparison's truth
    (:func:`mdmx.forecast.observed_e0`): the HMD's published tables, after the curation.

    Parameters
    ----------
    lt_curated : the curated life tables; labels, populations : chapter d4's labels and the register
    tensor : the comparison's tensor (``M``, ``observed``, ``pops``, ``years``), whose axes and masks
        the unpooled tensor must share

    Returns
    -------
    ``pop, sex, year, e0_obs`` (sex ``female``, ``male``, ``both``) at the observed country-years.
    """
    un = data.build_tensor(lt_curated, labels, q_min=q_min, min_observed_years=min_observed_years,
                           populations=populations)
    if not (np.array_equal(un["pops"], tensor["pops"]) and np.array_equal(un["years"], tensor["years"])
            and np.array_equal(un["observed"], tensor["observed"])):
        raise ValueError("the unpooled tensor's axes or observed mask differ from the comparison's")
    return fc.observed_e0(un["M"], tensor["observed"], tensor["pops"], tensor["years"])


def pooled_to_origin(lt_curated: pd.DataFrame, lt_pooled: pd.DataFrame, de: pd.DataFrame, labels: pd.DataFrame,
                     valley: np.ndarray, origin: int, *, max_gap: int, raw_rate_age_limit: int) -> pd.DataFrame:
    """The cells whose pooled values differ when the blocks are planned on the years up to ``origin`` only.

    Paper: sec:evaluation:robustness

    Chapter d5 forms its pooling blocks on each population's whole series, so a block can span an
    origin and a year up to the origin can share deaths with later years.  Here the pooling
    (:func:`mdmx.data.adaptive_pooling`, the same valley ages and settings) is run on the curated
    tables of the years up to the origin alone, for the populations that pooling changes at all (the
    others need none at any origin).

    Parameters
    ----------
    lt_curated, lt_pooled : the curated life tables and the same after chapter d5's pooling
    de, labels, valley : the deaths and exposures, the event labels and the valley ages of chapter d5
    origin : the forecast origin; max_gap, raw_rate_age_limit : chapter d5's settings

    Returns
    -------
    ``pop, sex, year, age, mx, qx``: every cell (years ≤ origin) whose values differ from ``lt_pooled``'s.
    """
    key = ["pop", "sex", "year", "age"]
    both = lt_curated[key + ["mx"]].merge(lt_pooled[key + ["mx"]], on=key, suffixes=("", "_p"),
                                          validate="one_to_one")
    pops = both.loc[both["mx"] != both["mx_p"], "pop"].unique()
    cur = lt_curated[lt_curated["pop"].isin(pops) & (lt_curated["year"] <= origin)]
    if cur.empty:
        return pd.DataFrame(columns=key + ["mx", "qx"])
    d = de[de["pop"].isin(pops) & (de["year"] <= origin)]
    pooled_o, _ = data.adaptive_pooling(cur, d, labels, valley, max_gap=max_gap,
                                        raw_rate_age_limit=raw_rate_age_limit)
    m = pooled_o[key + ["mx", "qx"]].merge(lt_pooled[key + ["mx", "qx"]], on=key, suffixes=("", "_full"),
                                           validate="one_to_one")
    changed = (m["mx"] != m["mx_full"]) | (m["qx"] != m["qx_full"])
    return m.loc[changed, key + ["mx", "qx"]].reset_index(drop=True)


def origin_tensor(M: np.ndarray, pops: Sequence[str], years: Sequence[int], ages: Sequence[int],
                  cells: pd.DataFrame, *, q_min: float) -> np.ndarray:
    """The tensor with the values of :func:`pooled_to_origin`'s cells put in.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); pops, years, ages : its axes; cells : ``pop, sex, year, age, qx``
    q_min : the floor of :func:`mdmx.lifetable.logit` (chapter d6's)

    Returns
    -------
    (S, A, C, T) a copy of ``M`` with logit(q_x) of ``cells`` (those of populations on the axis).
    """
    out = M.copy()
    c = cells[cells["pop"].isin(set(pops))]
    if c.empty:
        return out
    pop_idx = {p: i for i, p in enumerate(pops)}
    s_i = c["sex"].astype(str).map({s: i for i, s in enumerate(data.SEXES)}).to_numpy().astype(int)
    a_i = np.searchsorted(np.asarray(ages), c["age"].to_numpy())
    c_i = c["pop"].map(pop_idx).to_numpy().astype(int)
    t_i = c["year"].to_numpy().astype(int) - int(np.asarray(years)[0])
    out[s_i, a_i, c_i, t_i] = lifetable.logit(c["qx"].to_numpy(dtype=float), q_min)
    return out


def clean_pooling_runs(ctx: Mapping, *, lt_curated: pd.DataFrame, lt_pooled: pd.DataFrame, de: pd.DataFrame,
                       labels: pd.DataFrame, valley: np.ndarray, ages: Sequence[int], data_params: Mapping,
                       anchor: str | float, workers: int, name: str) -> dict[int, dict]:
    """Every entry whose inputs are pooled, rerun at every origin with the blocks planned on the years up to it.

    Paper: sec:evaluation:robustness

    At each origin: the cells of :func:`pooled_to_origin` (artefact ``<name>_cells_<origin>``) go into the
    tensor (:func:`origin_tensor`); the flow field (the entry's form) and anchored MDMx under both
    fitting-period policies are forecast from it; Hyndman–Ullah and the coherent model are forecast again
    for the populations whose cells changed, from the smoothed rates of the same pooling
    (:func:`mdmx.forecast.pooled_deaths`; the others' smoothed rates do not change, each year being smoothed
    on its own).  Lee–Carter and bayesLife read the unpooled rates and are not rerun.  Point forecasts only.

    Parameters
    ----------
    ctx : :func:`mdmx.evaluation.context`; lt_curated, lt_pooled, de, labels, valley : chapter d5's inputs
        and output; ages : the tensor's ages; data_params : ``[data]`` (``q_min``, ``pool_max_gap``,
        ``pool_raw_rate_age_limit``); anchor : MDMx's anchor; workers : MDMx's fit processes
    name : the artefacts' prefix

    Returns
    -------
    origin → ``ff`` (the flow field's points), ``mdmx`` (policy → points), ``cmp`` (policy →
    :func:`mdmx.forecast.comparator_forecasts` for the changed populations, or None), ``affected``
    (the populations whose cells changed) and ``n_cells``.
    """
    D, M, observed = ctx["design"], ctx["M"], ctx["observed"]
    pops, years = ctx["pops"], ctx["years"]
    mset = ctx["mdmx"]
    form = ctx["forms"][ev.FLOW_FIELD]
    q_min = float(data_params["q_min"])
    pool_set = {"max_gap": int(data_params["pool_max_gap"]),
                "raw_rate_age_limit": int(data_params["pool_raw_rate_age_limit"])}
    out = {}
    for origin in sorted(int(o) for o in ctx["jobs"]["full"]["origin"].unique()):
        cells = artifacts.stage(
            f"{name}_cells_{origin}",
            functools.partial(pooled_to_origin, lt_curated, lt_pooled, de, labels, valley, origin, **pool_set),
            inputs=["lt_curated", "lt_pooled", "hmd_raw", "event_labels"],
            params={"origin": origin, "valley": [int(a) for a in valley], **pool_set},
            code=[pooled_to_origin, data.adaptive_pooling, data.plan_blocks, data.contiguous_runs,
                  lifetable.qx_from_mx],
            desc=f"Pooling planned on the years up to {origin}: the cells that differ")
        jobs = {pol: ctx["jobs"][pol][ctx["jobs"][pol]["origin"] == origin].reset_index(drop=True)
                for pol in ("full", "win")}
        affected = sorted(set(cells["pop"]) & set(jobs["full"]["pop"]))

        def compute(origin=origin, cells=cells, jobs=jobs, affected=affected):
            M_o = origin_tensor(M, pops, years, ages, cells, q_min=q_min)
            e0_o = flowfield.e0_matrix(M_o, observed)
            sp = fc.origin_space(M_o, observed, years, origin, tau=mset.tau, age_rank=mset.age_rank,
                                 smooth=mset.smooth, n_pc=mset.n_pc)
            flow = flowfield.build_flow(sp["scores"], sp["keep"], years, e0_o, origin, form)
            ff_run = flowfield.forecast_origin(M_o, sp, pops, years, jobs["full"], origin, D.horizon, e0_o, form,
                                               flow=flow)
            md = {pol: fc.mdmx_origin(M_o, observed, pops, years, origin, jobs[pol], mset, workers=workers,
                                      anchor=anchor)["points"] for pol in jobs}
            cmp_runs = {pol: None for pol in jobs}
            if affected:
                key = ["pop", "sex", "year", "age"]
                sub = lt_pooled[lt_pooled["pop"].isin(affected) & (lt_pooled["year"] <= origin)]
                lt_o = sub.merge(cells[key + ["mx"]], on=key, how="left", suffixes=("", "_o"))
                lt_o["mx"] = lt_o["mx_o"].fillna(lt_o["mx"])
                cur = lt_curated[lt_curated["pop"].isin(affected) & (lt_curated["year"] <= origin)]
                de_o = de[de["pop"].isin(affected) & (de["year"] <= origin)]
                rates_o = fc.comparator_data(fc.pooled_deaths(de_o, cur, lt_o.drop(columns="mx_o")), observed,
                                             pops, years, ctx["cmp"]["open_age"])
                sm = fc.smooth_comparator_rates(rates_o)
                if len(sm["failures"]):
                    raise RuntimeError(f"the smoother failed for {sm['failures'].to_dict('records')}")
                for pol, jo in jobs.items():
                    ja = jo[jo["pop"].isin(affected)].reset_index(drop=True)
                    if len(ja):
                        cmp_runs[pol] = fc.comparator_forecasts(ctx["rates"], sm["smoothed"], ja, ctx["cmp"],
                                                                tag=f"{name}-{pol}-{origin}".replace("_", "-"))
            return {"ff": ff_run["points"], "mdmx": md, "cmp": cmp_runs, "affected": affected,
                    "n_cells": len(cells)}

        out[origin] = artifacts.stage(
            f"{name}_{origin}", compute,
            inputs=["tensor", "lt_curated", "lt_pooled", "hmd_raw", f"{name}_cells_{origin}", "forecast_cmp_rates"],
            params={"mdmx": mset.as_dict(), "form": form.as_dict(), "anchor": anchor, "horizon": D.horizon,
                    "q_min": q_min, "jobs": {p: artifacts.content_hash(j) for p, j in jobs.items()},
                    "r": ctx["r_params"]},
            code=[compute, origin_tensor, fc.pooled_deaths, fc.comparator_data, fc.smooth_comparator_rates,
                  fc.comparator_forecasts, rbridge.run, *flowfield.FLOW_CODE, *fc.MDMX_CODE],
            desc=f"The pooled entries from {origin} with the pooling planned on the years up to it")
    return out


def clean_pooling_results(ctx: Mapping, clean: Mapping[int, dict], results: Mapping[str, Mapping[int, Mapping]]
                          ) -> dict[str, dict[int, dict]]:
    """Every entry's points with :func:`clean_pooling_runs` in place of the pooled entries' (e₀ only).

    Parameters
    ----------
    ctx : :func:`mdmx.evaluation.context`; clean : :func:`clean_pooling_runs`
    results : :func:`mdmx.evaluation.entry_results` (Lee–Carter's, bayesLife's and the unchanged populations'
        points are taken from it)

    Returns
    -------
    entry label → origin → ``{"points": …}``; the ensemble rebuilt from its members' new points.
    """
    A, ax = ctx["M"].shape[1], ctx["cmp"]["mx_to_qx_ax"]
    out: dict[str, dict[int, dict]] = {}
    for label, (method, policy) in ev.ENTRIES.items():
        if method == "ENS":
            continue
        by_origin = {}
        for o, r in results[label].items():
            c = clean.get(int(o))
            if c is None:
                raise ValueError(f"no clean-pooling run at origin {o}")
            if method == "MDMx":
                pts = c["mdmx"][policy]
            elif method == "FF":
                pts = c["ff"]
            elif method in ("HU", "CFDM") and c["cmp"][policy] is not None:
                new = fc.comparator_points(c["cmp"][policy], method, A, ax)["points"]
                pts = pd.concat([r["points"][~r["points"]["pop"].isin(new["pop"].unique())], new], ignore_index=True)
            else:
                pts = r["points"]
            by_origin[o] = {"points": pts}
        out[label] = by_origin
    members = list(ctx["E"]["ensemble"])
    out[ev.ENSEMBLE] = {o: {"points": ev.ensemble_points({m: out[m][o]["points"].assign(origin=o)
                                                         for m in members}).drop(columns="origin")}
                        for o in sorted(set.intersection(*(set(out[m]) for m in members)))}
    return out
