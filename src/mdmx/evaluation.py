"""The final evaluation of the forecasters (paper §12; book chapter 12).

The design was frozen before any of its numbers were computed; this module holds the pieces the chapter calls:

* :class:`Design`: the origins, eras, held-out block, horizon bands and windows, from
  ``[evaluation]``;
* the ensemble (:func:`ensemble_points`, :func:`ensemble_schedules`,
  :func:`ensemble_quantiles`): the equally weighted mean of its members;
* where errors come from: :func:`arriaga` and :func:`age_split` (by age group),
  :func:`positions` (by starting position relative to the frontier point),
  :func:`forecast_pace`, :func:`realised_pace` and :func:`frontier_pace` (the e₀ gain per year);
* summaries: :func:`error_curves`, :func:`band_table`, :func:`window_robustness`, :func:`regret`;
* :func:`timed`: wall-clock and CPU time of a computation;
* the runs, each origin cached as an artefact with its own run time: :func:`run_mdmx`,
  :func:`run_flowfield`, :func:`run_comparators`, :func:`run_bl_schedules`;
* :func:`context` (every input of the runs) and :func:`run_entries` (every run but bayesLife's),
  which the chapter and ``tools/warm_evaluation.py`` share, so that a run made by one is found by the
  other.

Conventions
-----------
* Errors are forecast − observed.  Sex series ``female``, ``male`` and ``both`` (the average).
* An origin's era is ``early``, ``late`` (the late development origins) or ``held`` (the held-out
  block, scored once).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import artifacts, flowfield, lifetable, paths, provenance, rbridge
from . import bayeslife as bl
from . import compare as cp
from . import forecast as fc

#: The R packages of ``R/mortcast_schedules.R`` (their versions key bayesLife's schedules).
MORTCAST_PACKAGES = ["MortCast", "data.table", "jsonlite"]

__all__ = [
    "Design", "ensemble_points", "ensemble_schedules", "ensemble_quantiles", "age_groups", "arriaga",
    "age_split", "positions", "forecast_pace", "realised_pace", "frontier_pace", "error_curves",
    "band_table", "window_robustness", "regret", "timed", "run_mdmx", "run_flowfield", "run_comparators",
    "run_bl_schedules", "flow_forms", "context", "run_entries", "ENTRIES", "entry_results", "long_points",
    "paired_keys", "errors", "entry_quantiles", "schedule_scores", "point_values", "machine", "paired_differences",
    "open_interval_schedules",
]


# ══════════════════════════════════════════════════════════════════════════════
# The design
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Design:
    """The evaluation design (read from ``[evaluation]``).

    Paper: sec:evaluation:design

    Attributes
    ----------
    origins : every forecast origin (calendar years)
    horizon : years forecast beyond each origin
    min_train_years, min_test_years : a population is forecast from an origin only with at least
        this many observed years up to it and after it
    window : the fitting window of the second policy (years before the origin)
    eras : era name → its origins; ``held_out`` : the held-out origins, calibrated by the era
        ``held_out_by``
    bands : horizon bands, (first, last) inclusive
    """

    origins: tuple[int, ...]
    horizon: int
    min_train_years: int
    min_test_years: int
    window: int
    eras: dict
    held_out: tuple[int, ...]
    held_out_by: str
    bands: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        listed = [o for v in self.eras.values() for o in v] + list(self.held_out)
        if sorted(listed) != sorted(self.origins) or len(set(listed)) != len(listed):
            raise ValueError("every origin must be in exactly one era or in the held-out block")
        if self.held_out and self.held_out_by not in self.eras:
            raise ValueError(f"held_out_by must name an era, not {self.held_out_by!r}")
        edges = [b for band in self.bands for b in band]
        if edges[0] != 1 or edges[-1] != self.horizon or any(
                b[0] != a[1] + 1 for a, b in zip(self.bands, self.bands[1:], strict=False)):
            raise ValueError("the bands must cover 1 … horizon without gaps or overlaps")

    @classmethod
    def from_params(cls, ev: Mapping) -> Design:
        """Build the design from the ``[evaluation]`` parameter section."""
        first, last, step = (int(v) for v in ev["origins"])
        return cls(origins=tuple(range(first, last + 1, step)), horizon=int(ev["horizon"]),
                   min_train_years=int(ev["min_train_years"]), min_test_years=int(ev["min_test_years"]),
                   window=int(ev["window"]),
                   eras={k: tuple(range(int(v[0]), int(v[1]) + 1, step)) for k, v in ev["eras"].items()},
                   held_out=tuple(range(int(ev["held_out"][0]), int(ev["held_out"][1]) + 1, step)),
                   held_out_by=str(ev["held_out_by"]), bands=tuple((int(a), int(b)) for a, b in ev["bands"]))

    def era(self, origin: int) -> str:
        """``early``/``late``/… (the era of ``origin``) or ``held``."""
        for name, origins in self.eras.items():
            if int(origin) in origins:
                return name
        if int(origin) in self.held_out:
            return "held"
        raise ValueError(f"{origin} is not an origin of the design")

    def band(self, h: np.ndarray) -> np.ndarray:
        """The label ``"a–b"`` of each horizon's band."""
        h = np.asarray(h)
        out = np.full(h.shape, "", dtype=object)
        for a, b in self.bands:
            out[(h >= a) & (h <= b)] = f"{a}–{b}"
        if (out == "").any():
            raise ValueError("a horizon outside the bands")
        return out

    def band_labels(self) -> list[str]:
        """The bands' labels, ``"first–last"``, in order."""
        return [f"{a}–{b}" for a, b in self.bands]

    def as_dict(self) -> dict:
        """Plain dict of the design (what the artefact store hashes)."""
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# The ensemble
# ══════════════════════════════════════════════════════════════════════════════

_POINT_KEY = ["pop", "sex", "origin", "year"]


def ensemble_points(members: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The ensemble's e₀: the mean of its members' forecasts, where every member has one.

    Paper: eq:ev-ensemble, sec:evaluation:entries

    Parameters
    ----------
    members : member → its points ``pop, sex, origin, year, e0_fc``

    Returns
    -------
    ``pop, sex, origin, year, e0_fc``.  "Both" is the mean of the members' "both", which equals
    the mean of the ensemble's female and male e₀.
    """
    stacked = pd.concat([m[_POINT_KEY + ["e0_fc"]].assign(member=name) for name, m in members.items()],
                        ignore_index=True)
    if stacked.duplicated(_POINT_KEY + ["member"]).any():
        raise ValueError("a member has more than one forecast for a point")
    g = stacked.groupby(_POINT_KEY)["e0_fc"].agg(["mean", "size"]).reset_index()
    return g[g["size"] == len(members)].drop(columns="size").rename(columns={"mean": "e0_fc"})


def ensemble_schedules(members: Mapping[str, tuple[pd.DataFrame, np.ndarray]]) -> tuple[pd.DataFrame, np.ndarray]:
    """The ensemble's schedules: the mean of its members' logit(q_x), where every member has one.

    Paper: eq:ev-ensemble, sec:evaluation:entries

    Parameters
    ----------
    members : member → (index ``pop, origin, year``, (N, 2, A) logit(q_x))

    Returns
    -------
    (index ``pop, origin, year``, (n, 2, A) logit(q_x)), on the points every member has.
    """
    key = ["pop", "origin", "year"]
    items = list(members.values())
    common = items[0][0][key]
    for idx, _ in items[1:]:
        common = common.merge(idx[key], on=key)
    common = common.drop_duplicates().sort_values(key).reset_index(drop=True)
    total = np.zeros((len(common),) + items[0][1].shape[1:])
    for idx, z in items:
        pos = idx[key].reset_index(drop=True).reset_index().merge(common, on=key, validate="one_to_one")
        order = common.merge(pos, on=key, how="left")["index"].to_numpy()
        total += z[order]
    return common, total / len(items)


def ensemble_quantiles(members: Mapping[str, pd.DataFrame], qcols: Sequence[str]) -> pd.DataFrame:
    """The ensemble's predictive quantiles: the average of its members' at each level (quantile
    averaging), for every sex series, where every member has them.

    Paper: eq:ev-ensemble, sec:evaluation:entries

    Parameters
    ----------
    members : member → ``pop, sex, origin, year`` and the columns ``qcols``
    qcols : the quantile columns (:func:`mdmx.compare.quantile_columns`)
    """
    stacked = pd.concat([m[_POINT_KEY + list(qcols)].assign(member=name) for name, m in members.items()],
                        ignore_index=True).dropna(subset=list(qcols))
    g = stacked.groupby(_POINT_KEY)
    out = g[list(qcols)].mean()
    out = out[g.size() == len(members)].reset_index()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Where the errors come from: age, starting position, pace
# ══════════════════════════════════════════════════════════════════════════════

def age_groups(starts: Sequence[int], labels: Sequence[str], n_ages: int) -> dict[str, list[int]]:
    """Label → single ages of each group; the last group runs to ``n_ages − 1``."""
    if len(starts) != len(labels) or starts[0] != 0 or list(starts) != sorted(starts):
        raise ValueError("age groups need increasing first ages from 0, one label each")
    ends = list(starts[1:]) + [n_ages]
    return {lab: list(range(a, b)) for lab, a, b in zip(labels, starts, ends, strict=True)}


def _table(z: np.ndarray, a0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """l_x (…, A + 1), L_x and T_x (…, A) of logit(q_x) schedules, radix 1."""
    lx = lifetable.survivorship(lifetable.expit(z))
    Lx = 0.5 * (lx[..., :-1] + lx[..., 1:])
    Lx[..., 0] = a0 * lx[..., 0] + (1.0 - a0) * lx[..., 1]
    Tx = np.flip(np.cumsum(np.flip(Lx, axis=-1), axis=-1), axis=-1)
    return lx, Lx, Tx


def arriaga(z_obs: np.ndarray, z_fc: np.ndarray, a0: float = lifetable.A0) -> np.ndarray:
    """Age contributions to e₀(forecast) − e₀(observed) (Arriaga 1984).

    Paper: eq:ev-arriaga, sec:evaluation:sources

    With population 1 the observed and population 2 the forecast schedule, the contribution of age
    x is

        (l¹ₓ / l¹₀)(L²ₓ / l²ₓ − L¹ₓ / l¹ₓ) + (T²ₓ₊₁ / l¹₀)(l¹ₓ / l²ₓ − l¹ₓ₊₁ / l²ₓ₊₁),

    and, because the life table of :mod:`mdmx.lifetable` has no open interval (T_A = 0), the
    contributions over x = 0 … A − 1 add up to the e₀ difference exactly.

    Parameters
    ----------
    z_obs, z_fc : (..., A) observed and forecast logit(q_x)

    Returns
    -------
    (..., A) contributions in years of e₀.
    """
    l1, L1, _ = _table(z_obs, a0)
    l2, L2, T2 = _table(z_fc, a0)
    T2_next = np.concatenate([T2[..., 1:], np.zeros(T2.shape[:-1] + (1,))], axis=-1)
    main = l1[..., :-1] * (L2 / l2[..., :-1] - L1 / l1[..., :-1])
    rest = T2_next * (l1[..., :-1] / l2[..., :-1] - l1[..., 1:] / l2[..., 1:])
    return main + rest            # l¹₀ = l²₀ = 1


def age_split(index: pd.DataFrame, z_fc: np.ndarray, M: np.ndarray, observed: np.ndarray, pops: Sequence[str],
              years: Sequence[int], groups: Mapping[str, Sequence[int]], a0: float = lifetable.A0) -> pd.DataFrame:
    """Schedule errors and Arriaga contributions by age group, for every point with an observed schedule.

    Paper: eq:ev-arriaga, sec:evaluation:sources

    Parameters
    ----------
    index : (N rows) ``pop, year`` (and any other columns, kept) of the forecast schedules
    z_fc : (N, 2, A) forecast logit(q_x), index 0 = female
    M : (S, A, C, T) observed logit(q_x); observed : (C, T) mask of observed, non-exceptional
        population-years
    pops, years : labels of the population and year axes; groups : :func:`age_groups`

    Returns
    -------
    One row per (point, sex) with an observed schedule: the columns of ``index``, ``sex``,
    ``e0_err`` (forecast − observed e₀), ``rmse_<group>`` (l_x-weighted RMSE of logit(q_x), the
    observed l_x as weights) and ``arr_<group>`` for every group.  Raises if any decomposition does
    not add up to its e₀ error.
    """
    pidx = {str(p): i for i, p in enumerate(pops)}
    yidx = {int(y): i for i, y in enumerate(years)}
    c = index["pop"].astype(str).map(pidx).to_numpy()
    t = index["year"].astype(int).map(yidx)
    ok = t.notna().to_numpy()
    c, t = c[ok], t[ok].to_numpy().astype(int)
    keep = observed[c, t]
    rows = np.flatnonzero(ok)[keep]
    if rows.size == 0:
        return pd.DataFrame()
    zf = z_fc[rows]                                              # (n, 2, A)
    zo = np.moveaxis(M[:, :, c[keep], t[keep]], -1, 0)           # (n, 2, A)
    contrib = arriaga(zo, zf, a0)
    err = lifetable.e0_from_logit(zf, a0) - lifetable.e0_from_logit(zo, a0)
    gap = np.abs(contrib.sum(axis=-1) - err).max()
    if gap > 1e-8:
        raise ValueError(f"Arriaga contributions do not add up to the e₀ error (largest gap {gap:.2e} years)")
    lx = lifetable.survivorship(lifetable.expit(zo))[..., :-1]
    sq = (zf - zo) ** 2
    base = index.iloc[rows].reset_index(drop=True)
    frames = []
    for k, sex in enumerate(("female", "male")):
        out = base.assign(sex=sex, e0_err=err[:, k])
        for label, ages in groups.items():
            w = lx[:, k, list(ages)]
            out[f"rmse_{label}"] = np.sqrt(np.sum(w * sq[:, k, list(ages)], axis=-1) / np.sum(w, axis=-1))
            out[f"arr_{label}"] = contrib[:, k, list(ages)].sum(axis=-1)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def positions(flow: Mapping, scores: np.ndarray, keep: np.ndarray, pops: Sequence[str], years: Sequence[int],
              jobs: pd.DataFrame, near: float) -> pd.DataFrame:
    """Each job's starting position relative to the frontier point s₁*.

    Paper: sec:evaluation:sources

    The position is ``(s₁ₒ(c, t₀) − s₁ₒ*) / range``: the oriented s₁ at the job's last training year
    minus the oriented frontier point, in units of the range of oriented s₁ over the origin's data.

    Parameters
    ----------
    flow : the origin's flow (:func:`mdmx.flowfield.build_flow`); scores, keep : its space
    pops, years : labels; jobs : the origin's jobs (``pop, last_train_year``)
    near : a job below s₁* by at most this share of the range is ``near``, further ``far``

    Returns
    -------
    ``pop, position`` (``beyond``, ``near``, ``far``) and ``pos``, the position.
    """
    pops, years = np.asarray(pops), np.asarray(years).astype(int)
    s1o = flow["sign"] * scores[..., 0]
    star = flow["sign"] * flow["s1_star"]
    s1_range = float(np.ptp(s1o[keep]))
    rows = []
    for job in jobs.itertuples():
        c = int(np.flatnonzero(pops == job.pop)[0])
        t0 = int(np.flatnonzero(years == job.last_train_year)[0])
        pos = (s1o[c, t0] - star) / s1_range
        rows.append({"pop": job.pop, "pos": float(pos),
                     "position": "beyond" if pos > 0 else ("near" if pos >= -near else "far")})
    return pd.DataFrame(rows)


def forecast_pace(points: pd.DataFrame) -> pd.DataFrame:
    """The forecast e₀ gain per year: ``e0_fc(h) − e0_fc(h − 1)`` within each forecast.

    Paper: sec:evaluation:sources

    Parameters
    ----------
    points : ``pop, sex, origin, year, e0_fc``, and ``entry``, ``method`` or ``policy`` where
        several forecasts of the same population, sex and origin are stacked

    Returns
    -------
    The rows of ``points`` that follow another year of the same forecast, with ``h`` (year −
    origin) and ``gain`` (years of e₀ per year).
    """
    # A forecast is named by these columns only: other columns (an interval bound, an sd) must not
    # split it into one-row groups.
    key = ["pop", "sex", "origin"] + [c for c in ("entry", "method", "policy") if c in points.columns]
    p = points.sort_values(key + ["year"])
    prev = p.groupby(key)["year"].shift()
    gain = p["e0_fc"] - p.groupby(key)["e0_fc"].shift()
    out = p.assign(h=p["year"] - p["origin"], gain=gain)
    return out[(prev == p["year"] - 1)].reset_index(drop=True)


def realised_pace(truth: pd.DataFrame) -> pd.DataFrame:
    """The observed e₀ gain per year between consecutive observed years.

    Paper: sec:evaluation:sources

    Parameters
    ----------
    truth : ``pop, sex, year, e0_obs`` (:func:`mdmx.forecast.observed_e0`)

    Returns
    -------
    ``pop, sex, year, gain``: e0_obs(year) − e0_obs(year − 1), where both years are observed.
    """
    t = truth.sort_values(["pop", "sex", "year"])
    g = t.groupby(["pop", "sex"])
    ok = g["year"].shift() == t["year"] - 1
    return t.assign(gain=t["e0_obs"] - g["e0_obs"].shift())[ok][["pop", "sex", "year", "gain"]].reset_index(
        drop=True)


def frontier_pace(e0: np.ndarray, observed: np.ndarray, years: Sequence[int], k: int) -> pd.DataFrame:
    """The realised pace of the e₀ frontier: the annual change of the chain-linked top-k e₀ level.

    Paper: sec:evaluation:sources

    This uses every observed year, after the origins too: it describes what happened, for the
    evaluation, and is never an input to a forecast.

    Parameters
    ----------
    e0 : (C, T) e₀ by population and year (NaN where unobserved); observed : (C, T) bool
    years : the T calendar years; k : the number of leading populations

    Returns
    -------
    ``year, level, gain`` (gain = the level's change from the year before; NaN where the level is
    not linked in both years).
    """
    level = flowfield.frontier_level(e0, e0, observed, k)["level"]
    years = np.asarray(years).astype(int)
    gain = np.concatenate([[np.nan], np.diff(level)])
    return pd.DataFrame({"year": years, "level": level, "gain": gain})


# ══════════════════════════════════════════════════════════════════════════════
# Summaries
# ══════════════════════════════════════════════════════════════════════════════

def error_curves(errors: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """MAE, mean error and number of points at each horizon.

    Parameters
    ----------
    errors : ``h`` and ``error`` (forecast − observed) and the columns ``by``

    Returns
    -------
    ``by``, ``h``, ``mae``, ``me``, ``n``.
    """
    g = errors.groupby(list(by) + ["h"])["error"]
    return pd.DataFrame({"mae": g.apply(lambda x: float(np.mean(np.abs(x)))), "me": g.mean(),
                         "n": g.size()}).reset_index()


def band_table(errors: pd.DataFrame, design: Design, by: Sequence[str]) -> pd.DataFrame:
    """MAE, mean error and number of points in each horizon band (:attr:`Design.bands`).

    Returns
    -------
    ``by``, ``band`` (ordered as the design's bands), ``mae``, ``me``, ``n``.
    """
    e = errors.assign(band=pd.Categorical(design.band(errors["h"].to_numpy()), categories=design.band_labels(),
                                          ordered=True))
    g = e.groupby(list(by) + ["band"], observed=True)["error"]
    return pd.DataFrame({"mae": g.apply(lambda x: float(np.mean(np.abs(x)))), "me": g.mean(),
                         "n": g.size()}).reset_index()


def window_robustness(bands_all: pd.DataFrame, bands_win: pd.DataFrame, margins: Mapping[str, Mapping[str, float]],
                      by: Sequence[str]) -> pd.DataFrame:
    """Generality, property 1, "more history must not hurt": MAE with every year minus with the window.

    Paper: eq:ev-generality

    Parameters
    ----------
    bands_all, bands_win : :func:`band_table` of the same entry with every year and with the window
        (columns ``by``, ``era``, ``band``, ``mae``), on the same points
    margins : era → band → margin (Deviation 1's scale: 2 × the typical SE of a regret between
        forecasters); only the eras and bands with a margin are compared

    Returns
    -------
    ``by``, ``era``, ``band``, ``diff`` (MAE all − MAE window: positive when more history hurts),
    ``margin`` and ``hurts`` (diff > margin).
    """
    key = list(by) + ["era", "band"]
    m = bands_all[key + ["mae"]].merge(bands_win[key + ["mae"]], on=key, suffixes=("_all", "_win"))
    m["margin"] = [margins.get(era, {}).get(str(band), np.nan) for era, band in zip(m["era"], m["band"], strict=True)]
    m = m[m["margin"].notna()].reset_index(drop=True)
    m["diff"] = m["mae_all"] - m["mae_win"]
    m["hurts"] = m["diff"] > m["margin"]
    return m


def regret_margins(errors: pd.DataFrame, design: Design, reference: str, others: Sequence[str],
                   families: Mapping[str, str], bands: Sequence[str], *, multiplier: float, draws: int,
                   seed: int) -> pd.DataFrame:
    """The margins of "more history must not hurt": twice the typical standard error of a regret.

    Paper: eq:ev-generality

    In each era e and band j, for each other entry m, the difference of MAE of e₀ between
    ``reference`` and m on their common points, d = MAE_ref − MAE_m, and its standard error over a
    bootstrap of population families (the clusters resampled with replacement, ``draws`` times, as in
    :func:`mdmx.compare.overall`).  The margin is ``multiplier`` × the median of these standard errors
    over the other entries.

    Parameters
    ----------
    errors : e₀ errors, ``entry, pop, origin, year, h, error`` (one measure); design : the eras and bands
    reference : the entry the others are compared with (the flow field); others : the other entries
    families : population → its population family; bands : the bands with a margin (labels of
        :meth:`Design.band_labels`)
    multiplier, draws, seed : the margin's multiple of the standard error; the bootstrap

    Returns
    -------
    DataFrame, one row per era and band: ``era, band, margin, se_median, se_min, se_max, n_others``.
    """
    keys = ["pop", "origin", "year", "h"]
    e = errors.assign(era=errors["origin"].map({o: design.era(o) for o in errors["origin"].unique()}),
                      band=design.band(errors["h"].to_numpy()))
    e = e[e["band"].isin(list(bands)) & e["era"].isin(list(design.eras))]
    ref = e[e["entry"] == reference]
    if ref.empty:
        raise ValueError(f"no errors for the reference {reference!r}")
    clusters = np.unique([families[p] for p in e["pop"].unique()])
    c_index = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    b = rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=draws).astype(float)
    rows = []
    for (era, band), r in ref.groupby(["era", "band"], sort=False):
        ses = []
        for m in others:
            om = e[(e["entry"] == m) & (e["era"] == era) & (e["band"] == band)]
            if om.empty:
                raise ValueError(f"no errors for {m!r} in era {era}, band {band}")
            both = r[keys + ["error"]].merge(om[keys + ["error"]], on=keys, suffixes=("_r", "_m"))
            ci = both["pop"].map(families).map(c_index).to_numpy()
            diff = np.zeros(len(clusters))
            n = np.zeros(len(clusters))
            np.add.at(diff, ci, both["error_r"].abs().to_numpy() - both["error_m"].abs().to_numpy())
            np.add.at(n, ci, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = (b @ diff) / (b @ n)                 # a draw with no common point is left out
            ses.append(float(np.nanstd(d, ddof=1)))
        ses = np.asarray(ses)
        rows.append({"era": era, "band": band, "margin": multiplier * float(np.median(ses)),
                     "se_median": float(np.median(ses)), "se_min": float(ses.min()), "se_max": float(ses.max()),
                     "n_others": len(ses)})
    out = pd.DataFrame(rows)
    era_order = {name: i for i, name in enumerate(design.eras)}
    band_order = {lab: i for i, lab in enumerate(design.band_labels())}
    out["_o"] = out["era"].map(era_order) * len(band_order) + out["band"].map(band_order)
    return out.sort_values("_o", ignore_index=True).drop(columns="_o")


def regret(bands: pd.DataFrame, entry: str = "entry") -> pd.DataFrame:
    """Each entry's MAE minus the best entry's, in each era and band (generality, property 3).

    Paper: eq:ev-generality

    Parameters
    ----------
    bands : :func:`band_table` by ``entry``, ``era`` (columns ``entry, era, band, mae``)

    Returns
    -------
    ``bands`` with ``best`` (the smallest MAE of the era and band) and ``regret``.
    """
    b = bands.copy()
    b["best"] = b.groupby(["era", "band"], observed=True)["mae"].transform("min")
    b["regret"] = b["mae"] - b["best"]
    return b


def machine() -> str:
    """The computer the runs ran on, for the run-time table: the processor, its cores and the system."""
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                               check=False).stdout.strip()
        cpu = brand or cpu
        system = f"macOS {platform.mac_ver()[0]}"
    else:
        system = f"{platform.system()} {platform.release()}"
    return f"{cpu}, {os.cpu_count()} cores, {system}"


def timed(fn: Callable, *args, **kwargs) -> tuple:
    """Run ``fn`` and return ``(result, wall seconds, CPU seconds)``.

    CPU time is this process's plus that of its finished child processes (R, the fit workers), from
    ``os.times``; a child still running when ``fn`` returns is not counted, and on Windows the
    children's time is not available (only this process's is counted).
    """
    def cpu() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    w0, c0 = time.perf_counter(), cpu()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - w0, cpu() - c0


# ══════════════════════════════════════════════════════════════════════════════
# The runs: every origin cached, with its run time
# ══════════════════════════════════════════════════════════════════════════════

def _origins(jobs: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    return [(int(o), j.reset_index(drop=True)) for o, j in jobs.groupby("origin", sort=True)]


def run_mdmx(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int], jobs: pd.DataFrame,
             settings: fc.MDMxSettings, *, anchor: str | float, workers: int, name: str) -> dict[int, dict]:
    """MDMx at every origin of ``jobs`` (one fitting-period policy), each origin artefact ``<name>_<origin>``.

    The result of :func:`mdmx.forecast.mdmx_origin` with ``anchor``, plus ``seconds`` and
    ``cpu_seconds`` (its run time; the worker processes' CPU included).  The cache key holds the
    settings, the anchor and the jobs, not ``workers`` (the results do not depend on it).
    """
    out = {}
    for origin, jo in _origins(jobs):
        def compute(origin=origin, jo=jo):
            res, wall, cpu = timed(fc.mdmx_origin, M, observed, pops, years, origin, jo, settings, workers=workers,
                                   anchor=anchor)
            return {**res, "seconds": wall, "cpu_seconds": cpu}
        out[origin] = artifacts.stage(
            f"{name}_{origin}", compute, inputs=["tensor"],
            params={"settings": settings.as_dict(), "anchor": anchor, "jobs": artifacts.content_hash(jo)},
            code=[compute, timed, *fc.MDMX_CODE], desc=f"MDMx (anchor {anchor}) forecasts from {origin}")
    return out


def run_flowfield(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                  jobs: pd.DataFrame, mdmx_settings: fc.MDMxSettings, forms: Mapping[str, flowfield.FlowFieldSettings],
                  e0: np.ndarray, *, horizon: int, near: float, name: str,
                  space_windows: Mapping[str, int] | None = None,
                  schedule_forms: Sequence[str] | None = None,
                  leaders_alpha_one: Sequence[str] = ()) -> dict[int, dict]:
    """The flow field in each of its ``forms`` at every origin of ``jobs``, each origin artefact ``<name>_<origin>``.

    Every form is forecast from the origin's space (:func:`mdmx.forecast.origin_space`, MDMx's
    settings), except those named in ``space_windows``, whose space is estimated on their window only
    (the window curve's ``dynspace``).  Schedules are kept for the forms in ``schedule_forms`` (all if None); the
    others keep their e₀ points only.  In the forms of ``leaders_alpha_one`` the jobs that start beyond
    the frontier point keep their deviations from the shape law unrelaxed (α = 1; a
    sensitivity of the design).

    Returns
    -------
    origin → dict with, per form, ``points``, ``schedule_index``, ``schedules`` and ``flow``
    (:func:`mdmx.flowfield.flow_summary`) under ``runs``; ``positions`` of the jobs relative to the
    first form's frontier point (:func:`positions`); ``seconds`` and ``cpu_seconds`` per form.
    """
    space_windows = dict(space_windows or {})
    keep_sched = set(forms) if schedule_forms is None else set(schedule_forms)
    years_arr = np.asarray(years)

    def space_on(origin, obs):
        return fc.origin_space(M, obs, years, origin, tau=mdmx_settings.tau, age_rank=mdmx_settings.age_rank,
                               smooth=mdmx_settings.smooth, n_pc=mdmx_settings.n_pc)

    out = {}
    first = next(iter(forms))
    for origin, jo in _origins(jobs):
        def compute(origin=origin, jo=jo):
            res, secs, cpus = {}, {}, {}
            space = space_on(origin, observed)
            positions_out = None
            for label, settings in forms.items():
                def one(settings=settings, label=label):
                    sp = space
                    if label in space_windows:
                        w = space_windows[label]
                        sp = space_on(origin, observed & (years_arr >= origin - w + 1)[None, :])
                    flow = flowfield.build_flow(sp["scores"], sp["keep"], years, e0, origin, settings)
                    if label not in leaders_alpha_one:
                        run = flowfield.forecast_origin(M, sp, pops, years, jo, origin, horizon, e0, settings,
                                                        flow=flow, schedules=label in keep_sched)
                        return run, flow, sp
                    pos = positions(flow, sp["scores"], sp["keep"], pops, years, jo, near)
                    beyond = set(pos.loc[pos["position"] == "beyond", "pop"])
                    parts = [flowfield.forecast_origin(M, sp, pops, years, part, origin, horizon, e0, settings,
                                                       flow={**flow, "alpha": np.ones_like(flow["alpha"])} if lead
                                                       else flow, schedules=label in keep_sched)
                             for lead, part in ((True, jo[jo["pop"].isin(beyond)]), (False, jo[~jo["pop"].isin(beyond)]))
                             if len(part)]
                    run = {"points": pd.concat([r["points"] for r in parts], ignore_index=True),
                           "flow": flowfield.flow_summary(flow)}
                    if label in keep_sched:
                        run["schedule_index"] = pd.concat([r["schedule_index"] for r in parts], ignore_index=True)
                        run["schedules"] = np.concatenate([r["schedules"] for r in parts], axis=0)
                    return run, flow, sp
                (run, flow, sp), secs[label], cpus[label] = timed(one)
                res[label] = run
                if label == first:
                    positions_out = positions(flow, sp["scores"], sp["keep"], pops, years, jo, near)
            return {"runs": res, "positions": positions_out, "seconds": secs, "cpu_seconds": cpus}
        out[origin] = artifacts.stage(
            f"{name}_{origin}", compute, inputs=["tensor"],
            params={"mdmx": mdmx_settings.as_dict(), "forms": {k: v.as_dict() for k, v in forms.items()},
                    "space_windows": space_windows, "schedule_forms": sorted(keep_sched), "horizon": horizon,
                    "near": near, "leaders_alpha_one": sorted(leaders_alpha_one),
                    "jobs": artifacts.content_hash(jo), "e0": artifacts.content_hash(e0)},
            code=[compute, positions, timed, *flowfield.FLOW_CODE],
            desc=f"The flow field in {len(forms)} forms from {origin}")
    return out


def run_comparators(rates: pd.DataFrame, smoothed: pd.DataFrame, jobs: pd.DataFrame, settings: Mapping,
                    r_params: Mapping, *, name: str, concurrent: int) -> dict[int, dict]:
    """Lee–Carter, Hyndman–Ullah and coherent FDM (one R session per origin) at every origin of ``jobs``.

    Each origin is artefact ``<name>_<origin>``: :func:`mdmx.forecast.comparator_forecasts`'s output
    plus ``seconds`` (wall-clock; the R process's CPU is not separable from the other runs'). Several
    origins run at once (``concurrent`` R processes).
    """
    tasks = []
    for origin, jo in _origins(jobs):
        def compute(origin=origin, jo=jo):
            t0 = time.perf_counter()
            res = fc.comparator_forecasts(rates, smoothed, jo, settings, tag=f"{name}-{origin}".replace("_", "-"))
            return {**res, "seconds": time.perf_counter() - t0}
        stage = functools.partial(
            artifacts.stage, f"{name}_{origin}", compute, inputs=["forecast_cmp_rates", "forecast_cmp_smoothed"],
            params={**r_params, "jobs": artifacts.content_hash(jo)}, code=[compute, fc.comparator_forecasts, rbridge.run],
            desc=f"Lee-Carter, Hyndman-Ullah and coherent FDM forecasts from {origin}")
        tasks.append((origin, stage))
    return bl.run_many(tasks, concurrent=concurrent)


def run_bl_schedules(runs: Mapping[tuple[str, int], dict], smoothed: pd.DataFrame,
                     jobs: Mapping[str, pd.DataFrame], *, n_ages: int, ax: float, e0_tol: float, max_rounds: int,
                     name: str, concurrent: int) -> dict[tuple[str, int], dict]:
    """bayesLife's schedules (:func:`mdmx.bayeslife.schedules`) for every run, artefact ``<name>_<policy>_<origin>``.

    Paper: sec:evaluation:entries

    ``runs`` are :func:`mdmx.bayeslife.offsite_runs`; each is keyed on its forecasts' content.
    ``e0_tol`` and ``max_rounds``: the calibration of the schedules' e₀ to the median.  The key also holds
    the R script and MortCast's version, as the other R runs' keys do.
    """
    script = provenance.sha256_file(paths.root() / "R" / "mortcast_schedules.R")
    r_session = rbridge.session_info("mortcast_schedules.R", MORTCAST_PACKAGES)
    tasks = []
    for (policy, origin), run in sorted(runs.items()):
        jp = jobs[policy]
        jo = jp[jp["origin"] == origin].reset_index(drop=True)
        tag = f"{name}-{policy}-{origin}".replace("_", "-")
        stage = functools.partial(
            artifacts.stage, f"{name}_{policy}_{origin}",
            functools.partial(bl.schedules, run, smoothed, jo, origin, n_ages=n_ages, ax=ax, tag=tag,
                              e0_tol=e0_tol, max_rounds=max_rounds),
            inputs=["forecast_cmp_smoothed"],
            params={"forecasts": artifacts.content_hash(run["forecasts"]), "jobs": artifacts.content_hash(jo),
                    "n_ages": n_ages, "ax": ax, "e0_tol": e0_tol, "max_rounds": max_rounds, "script": script,
                    "r": r_session},
            code=[bl.schedules, bl.points, fc.comparator_schedules, lifetable.e0_from_logit, rbridge.run],
            desc=f"bayesLife's schedules by MortCast from {origin}, policy {policy}")
        tasks.append(((policy, origin), stage))
    return bl.run_many(tasks, concurrent=concurrent)


# ══════════════════════════════════════════════════════════════════════════════
# Everything the runs need, and every run but bayesLife's
# ══════════════════════════════════════════════════════════════════════════════

def flow_forms(base: flowfield.FlowFieldSettings, windows: Sequence[int], window: int) -> tuple[dict, dict]:
    """The flow field's forms: the entry, its windowed variant, and the window curve's forms.

    * ``Flow field``: ``base`` (FR1-50, the entry), and ``Flow field, α = 1 for leaders`` (the same, with
      the deviations of the jobs beyond the frontier point unrelaxed: a sensitivity);
    * ``Flow field, windowed``: the speed and the shape law both on the last ``window`` years (the
      generality table's variant);
    * ``FF all``: every part on every year (the window curve's "all");
    * for each ``w`` of ``windows``: ``FF speed w`` (the speed on w years), ``FF shape w`` (the shape
      law on w years), ``FF dyn w`` (both), ``FF dynspace w`` (both, and the space too).  The frontier's own window stays ``base``'s.

    Returns
    -------
    (label → settings, label → space window) — the second for the ``dynspace`` forms.
    """
    forms = {"Flow field": base,
             "Flow field, α = 1 for leaders": base,
             "Flow field, windowed": dataclasses.replace(base, speed_window=window, shape_window=window),
             "FF all": dataclasses.replace(base, speed_window=None, shape_window=None)}
    space_windows = {}
    for w in windows:
        forms[f"FF speed {w}"] = dataclasses.replace(base, speed_window=w)
        forms[f"FF shape {w}"] = dataclasses.replace(base, shape_window=w)
        forms[f"FF dyn {w}"] = dataclasses.replace(base, speed_window=w, shape_window=w)
        forms[f"FF dynspace {w}"] = dataclasses.replace(base, speed_window=w, shape_window=w)
        space_windows[f"FF dynspace {w}"] = int(w)
    return forms, space_windows


def context(P) -> dict:
    """Every input of the evaluation's runs, from the parameters and the artefacts of chapters 03–04.

    Also stages the comparators' inputs (``forecast_cmp_rates``: deaths and exposures on the
    forecasting data policy; ``forecast_cmp_smoothed``: the rates smoothed year by year in R).

    Returns
    -------
    dict with ``design``, the tensor (``M``, ``observed``, ``pops``, ``years``, ``names``),
    ``mdmx`` (:class:`mdmx.forecast.MDMxSettings`), ``flowfield``, ``forms`` and ``space_windows``
    (:func:`flow_forms`), ``e0`` (:func:`mdmx.flowfield.e0_matrix`), ``jobs`` (policy → jobs:
    ``full``, ``win``), ``curve_jobs`` (window → the development origins' jobs restricted to it),
    ``rates``, ``smoothed``, ``cmp`` (``[forecast.comparators]``), ``r_params`` (their cache key),
    and the parameter sections ``F`` (``[forecast]``) and ``E`` (``[evaluation]``).
    """
    F, E = P.section("forecast"), P.section("evaluation")
    design = Design.from_params(E)
    T = artifacts.load("tensor", quiet=True)
    tucker = artifacts.load("tucker", quiet=True)
    raw = artifacts.load("hmd_raw", quiet=True)
    M, observed = T["M"], T["observed"]
    pops, years = np.asarray(T["pops"]), np.asarray(T["years"]).astype(int)
    mdmx_set = fc.MDMxSettings.from_params(F, P.section("tucker"), age_rank=int(tucker["ranks"][1]),
                                           horizon=design.horizon)
    ff_set = flowfield.FlowFieldSettings.from_params(P.section("flowfield"))
    # One anchor for both forecasters: the same smoothing of the jump-off residual.
    if ff_set.anchor_smoothing != mdmx_set.anchor_smoothing:
        raise ValueError(f"[flowfield] anchor_smoothing ({ff_set.anchor_smoothing}) differs from [forecast]'s "
                         f"({mdmx_set.anchor_smoothing})")
    windows = [int(w) for w in E["window_curve"]]
    forms, space_windows = flow_forms(ff_set, windows, design.window)
    jobs = fc.cv_jobs(observed, pops, years, list(design.origins), design.horizon, design.min_train_years,
                      design.min_test_years)
    development = [o for era in design.eras.values() for o in era]
    dev_jobs = jobs[jobs["origin"].isin(development)]
    cmp = F["comparators"]
    rates = artifacts.stage(
        "forecast_cmp_rates", functools.partial(fc.comparator_data, raw["de"], observed, pops, years, cmp["open_age"]),
        inputs=["tensor", "hmd_raw"], params={"open_age": cmp["open_age"]}, code=[fc.comparator_data],
        desc="HMD deaths and exposures on the forecasting data policy (gaps filled)")
    # The smoother's input: the same, with the pooled rates in the pooled cells (a zero-death cell
    # would drop out of the smoother and bias the curve upwards in the smallest populations).
    lt_cur, lt_pool = artifacts.load("lt_curated", quiet=True)["lt"], artifacts.load("lt_pooled", quiet=True)["lt"]

    def _pooled_rates():
        return fc.comparator_data(fc.pooled_deaths(raw["de"], lt_cur, lt_pool), observed, pops, years, cmp["open_age"])

    rates_pooled = artifacts.stage(
        "forecast_cmp_rates_pooled", _pooled_rates,
        inputs=["tensor", "hmd_raw", "lt_curated", "lt_pooled"], params={"open_age": cmp["open_age"]},
        code=[_pooled_rates, fc.comparator_data, fc.pooled_deaths],
        desc="The comparators' deaths and exposures with the pooled rates in the pooled cells (the smoother's input)")
    r_session = rbridge.session_info("demography_forecasts.R", cp.R_PACKAGES)
    scripts = {s: provenance.sha256_file(paths.root() / "R" / s) for s in ("hmd_smooth.R", "demography_forecasts.R")}
    smooth = artifacts.stage(
        "forecast_cmp_smoothed", functools.partial(fc.smooth_comparator_rates, rates_pooled),
        inputs=["forecast_cmp_rates_pooled"],
        params={"r": r_session, "script": scripts["hmd_smooth.R"]}, code=[fc.smooth_comparator_rates, rbridge.run],
        desc="HMD death rates smoothed year by year (demography::smooth.demogdata)")
    return {"design": design, "M": M, "observed": observed, "pops": pops, "years": years,
            "names": dict(zip(T["pops"], T["pop_names"], strict=True)), "mdmx": mdmx_set, "flowfield": ff_set,
            "forms": forms, "space_windows": space_windows, "e0": flowfield.e0_matrix(M, observed),
            "jobs": {"full": jobs, "win": fc.restrict_fit_window(jobs, design.window)},
            "curve_jobs": {w: fc.restrict_fit_window(dev_jobs, w) for w in windows if w != design.window},
            "rates": rates, "rates_pooled": rates_pooled, "smoothed": smooth["smoothed"], "smooth_failures": smooth["failures"], "cmp": cmp,
            "r_params": {"comparators": cmp, "r": r_session, "scripts": scripts}, "F": F, "E": E}


#: The parts of :func:`run_entries`, in the order they run.
PARTS = ("flowfield", "mdmx", "sensitivity", "comparators", "intervals", "curve", "jumpoff")


def run_entries(ctx: Mapping, *, workers: int, concurrent: int, parts: Sequence[str] = PARTS,
                log: Callable[[str], None] = print) -> dict:
    """Every run of the evaluation except bayesLife's (imported from the off-site batch).

    Parts (each origin cached; a part already computed returns at once):

    * ``flowfield``: every form of :func:`flow_forms` at every origin;
    * ``mdmx``: anchored MDMx under both fitting-period policies;
    * ``sensitivity``: MDMx unanchored and with the decaying anchor (every year);
    * ``comparators``: Lee–Carter, Hyndman–Ullah and coherent FDM under both policies;
    * ``intervals``: the comparators' interval forecasts (:func:`mdmx.compare.interval_runs`);
    * ``curve``: anchored MDMx and the comparators at the window curve's other windows
      (development origins);
    * ``jumpoff``: Lee–Carter and Hyndman–Ullah started from the observed rates of the jump-off year
      (``jumpchoice = "actual"``) under both policies, a sensitivity (Deviation 2; the coherent model
      has no such option and is unchanged).

    Returns
    -------
    part → its results (keyed by policy or window, then origin).
    """
    D, E, F = ctx["design"], ctx["E"], ctx["F"]
    tensor = (ctx["M"], ctx["observed"], ctx["pops"], ctx["years"])
    out = {}
    for part in parts:
        t0 = time.perf_counter()
        if part == "flowfield":
            out[part] = run_flowfield(*tensor, ctx["jobs"]["full"], ctx["mdmx"], ctx["forms"], ctx["e0"],
                                      horizon=D.horizon, near=float(E["positions_near"]), name="evaluation_ff",
                                      space_windows=ctx["space_windows"],
                                      schedule_forms=["Flow field", "Flow field, α = 1 for leaders",
                                                      "Flow field, windowed"],
                                      leaders_alpha_one=["Flow field, α = 1 for leaders"])
        elif part == "mdmx":
            out[part] = {pol: run_mdmx(*tensor, ctx["jobs"][pol], ctx["mdmx"], anchor=E["mdmx_anchor"], workers=workers,
                                       name=f"evaluation_mdmx_{pol}") for pol in ("full", "win")}
        elif part == "sensitivity":
            half = float(E["decaying_anchor_half_life"])
            out[part] = {"none": run_mdmx(*tensor, ctx["jobs"]["full"], ctx["mdmx"], anchor="none", workers=workers,
                                          name="evaluation_mdmxnone_full"),
                         "decaying": run_mdmx(*tensor, ctx["jobs"]["full"], ctx["mdmx"], anchor=half,
                                              workers=workers, name="evaluation_mdmxdecay_full")}
        elif part == "comparators":
            out[part] = {pol: run_comparators(ctx["rates"], ctx["smoothed"], ctx["jobs"][pol], ctx["cmp"],
                                              ctx["r_params"], name=f"evaluation_cmp_{pol}", concurrent=concurrent)
                         for pol in ("full", "win")}
        elif part == "intervals":
            ov = F["overall"]
            out[part] = cp.interval_runs(ctx["rates"], ctx["smoothed"], ctx["jobs"], ctx["cmp"], nsim=int(ov["nsim"]),
                                         sim_seed=int(ov["sim_seed"]), levels=cp.quantile_levels(F["pi_levels"]),
                                         n_ages=ctx["M"].shape[1], ax=ctx["cmp"]["mx_to_qx_ax"],
                                         concurrent=concurrent, prefix="evaluation_pi")
        elif part == "curve":
            out[part] = {w: {"mdmx": run_mdmx(*tensor, jw, ctx["mdmx"], anchor=E["mdmx_anchor"], workers=workers,
                                              name=f"evaluation_mdmx_w{w}"),
                             "comparators": run_comparators(ctx["rates"], ctx["smoothed"], jw, ctx["cmp"],
                                                            ctx["r_params"], name=f"evaluation_cmp_w{w}",
                                                            concurrent=concurrent)}
                         for w, jw in ctx["curve_jobs"].items()}
        elif part == "jumpoff":
            cmp_j = {**ctx["cmp"], "lc_jumpchoice": "actual", "fdm_jumpchoice": "actual"}
            r_j = {**ctx["r_params"], "comparators": cmp_j}
            out[part] = {pol: run_comparators(ctx["rates"], ctx["smoothed"], ctx["jobs"][pol], cmp_j, r_j,
                                              name=f"evaluation_cmpjump_{pol}", concurrent=concurrent)
                         for pol in ("full", "win")}
        else:
            raise ValueError(f"unknown part {part!r}; the parts are {PARTS}")
        log(f"{part}: {time.perf_counter() - t0:.0f} s")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The entries: their points, quantiles and schedules on common points
# ══════════════════════════════════════════════════════════════════════════════

#: The ranked entries, fixed by the design: label → (method, fitting-period policy).  The naive
#: random walk with drift is the yardstick, not an entry.
ENTRIES = {
    "MDMx + anchor, all years": ("MDMx", "full"), "MDMx + anchor, last 50 years": ("MDMx", "win"),
    "Flow field": ("FF", "full"),
    "Lee–Carter, all years": ("LC", "full"), "Lee–Carter, last 50 years": ("LC", "win"),
    "Hyndman–Ullah, all years": ("HU", "full"), "Hyndman–Ullah, last 50 years": ("HU", "win"),
    "Coherent FDM, all years": ("CFDM", "full"), "Coherent FDM, last 50 years": ("CFDM", "win"),
    "bayesLife, all years": ("BL", "full"), "bayesLife, last 50 years": ("BL", "win"),
    "Ensemble": ("ENS", None),
}
NAIVE = "Naive"


def entry_results(ctx: Mapping, runs: Mapping, bl_runs: Mapping | None = None,
                  bl_schedules: Mapping | None = None) -> dict[str, dict[int, dict]]:
    """Every entry's results per origin, in one format: ``points`` (``pop, sex, year, e0_fc``, and
    ``sd`` for MDMx, the interval bounds for bayesLife), ``schedule_index`` (``pop, year``) and
    ``schedules`` (N, 2, A) logit(q_x).

    Parameters
    ----------
    ctx : :func:`context`; runs : :func:`run_entries` (parts ``mdmx``, ``flowfield``, ``comparators``)
    bl_runs : :func:`mdmx.bayeslife.offsite_runs`; bl_schedules : :func:`run_bl_schedules`
        (bayesLife is left out without them)

    Returns
    -------
    entry label (:data:`ENTRIES`) → origin → result.  The ensemble (``[evaluation] ensemble``) is
    built from its members on the points they all have.
    """
    A, ax = ctx["M"].shape[1], ctx["cmp"]["mx_to_qx_ax"]
    out: dict[str, dict[int, dict]] = {}
    for label, (method, policy) in ENTRIES.items():
        if method == "MDMx":
            out[label] = {o: r for o, r in runs["mdmx"][policy].items()}
        elif method == "FF":
            out[label] = {o: r["runs"]["Flow field"] for o, r in runs["flowfield"].items()}
        elif method in ("LC", "HU", "CFDM"):
            out[label] = {o: fc.comparator_points(r, method, A, ax) for o, r in runs["comparators"][policy].items()}
        elif method == "BL" and bl_runs is not None and bl_schedules is not None:
            out[label] = {o: {**bl.points(bl_runs[(p, o)], o), **bl_schedules[(p, o)]}
                          for (p, o) in sorted(bl_runs) if p == policy}
    members = list(ctx["E"]["ensemble"])
    missing = [m for m in members if m not in out]
    if missing:
        raise ValueError(f"the ensemble's members {missing} are not entries")
    ens = {}
    for o in sorted(set.intersection(*(set(out[m]) for m in members))):
        pts = ensemble_points({m: out[m][o]["points"].assign(origin=o) for m in members})
        idx, z = ensemble_schedules({m: (out[m][o]["schedule_index"].assign(origin=o), out[m][o]["schedules"])
                                     for m in members})
        ens[o] = {"points": pts.drop(columns="origin"), "schedule_index": idx[["pop", "year"]], "schedules": z}
    out["Ensemble"] = ens
    return out


def long_points(entries: Mapping[str, Mapping[int, Mapping]]) -> pd.DataFrame:
    """Every entry's forecasts in one frame: ``entry, pop, sex, origin, year, h, e0_fc`` (and any other
    point columns), years after the origin only."""
    frames = [r["points"].assign(entry=label, origin=int(o))
              for label, by_origin in entries.items() for o, r in by_origin.items()]
    out = pd.concat(frames, ignore_index=True)
    out = out[out["year"] > out["origin"]].reset_index(drop=True)
    return out.assign(h=out["year"] - out["origin"])


def paired_keys(points: pd.DataFrame, truth: pd.DataFrame, entries: Sequence[str]) -> pd.DataFrame:
    """The points (``pop, origin, year``) at which every entry in ``entries`` forecasts all three sex
    series and all three are observed: every comparison is made on them (T5b's paired set).

    Paper: sec:evaluation:design
    """
    p = points[points["entry"].isin(entries)].merge(truth[["pop", "sex", "year"]], on=["pop", "sex", "year"])
    n = (p["entry"] + "|" + p["sex"]).groupby([p["pop"], p["origin"], p["year"]]).nunique()
    keys = n[n == 3 * len(entries)].reset_index()[["pop", "origin", "year"]]
    return keys.sort_values(["origin", "pop", "year"]).reset_index(drop=True)


def errors(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Errors (forecast − observed) at ``keys``: ``entry, pop, origin, year, h, measure, forecast,
    observed, error``, with ``measure`` ``e0_female``, ``e0_male``, ``e0_both`` and ``gap`` (female −
    male)."""
    p = points.merge(keys, on=["pop", "origin", "year"]).merge(truth, on=["pop", "sex", "year"])
    long = p.assign(measure="e0_" + p["sex"], forecast=p["e0_fc"], observed=p["e0_obs"])
    w = p.pivot_table(index=["entry", "pop", "origin", "year", "h"], columns="sex",
                      values=["e0_fc", "e0_obs"]).reset_index()
    gap = pd.DataFrame({"entry": w["entry"], "pop": w["pop"], "origin": w["origin"], "year": w["year"], "h": w["h"],
                        "measure": "gap", "forecast": w[("e0_fc", "female")] - w[("e0_fc", "male")],
                        "observed": w[("e0_obs", "female")] - w[("e0_obs", "male")]})
    cols = ["entry", "pop", "origin", "year", "h", "measure", "forecast", "observed"]
    out = pd.concat([long[cols], gap[cols]], ignore_index=True)
    return out.assign(error=out["forecast"] - out["observed"])


def entry_quantiles(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame, *,
                    calibration: Mapping[int, Sequence[int]], pi_levels: Sequence[int], min_points: int,
                    interval_runs: Mapping, lc_q: Mapping[str, pd.DataFrame], naive_q: pd.DataFrame,
                    ensemble: Sequence[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every entry's predictive quantiles of e₀ : each method's own intervals.

    Paper: *eq:ev-intervals, sec:evaluation:intervals

    * MDMx: the delta-method SD scaled by κ, calibrated leaving out one origin, within era
      (:func:`mdmx.forecast.loo_calibrate`), normal quantiles;
    * the flow field: empirical quantiles of its errors at the other origins of the era
      (:func:`mdmx.flowfield.empirical_intervals`);
    * Lee–Carter: its k_t interval mapped through e₀(k) (``lc_q``, policy → frame); Hyndman–Ullah and
      the coherent FDM: their simulations (``interval_runs``); Hyndman–Ullah's "both" from its female
      and male quantiles when the sexes move together (:func:`mdmx.compare.comonotone_both`);
    * bayesLife: its MCMC quantiles; the naive (all years): ``naive_q`` (:func:`mdmx.compare.naive_e0`);
    * the ensemble: the average of its members' quantiles at each level.

    Only the observations at ``keys`` calibrate (the others, the held-out block's included, are
    hidden from the calibration).

    Returns
    -------
    (entry → ``pop, sex, origin, year`` and the quantile columns; the flow field's cells without an
    interval, as :func:`mdmx.flowfield.empirical_intervals` reports them).
    """
    levels = cp.quantile_levels(pi_levels)
    qcols = cp.quantile_columns(levels)
    key4 = ["pop", "sex", "origin", "year"]
    obs = observed_at(points, truth, keys)
    pi = {pol: pd.concat([r["quantiles"] for (p, o), r in interval_runs.items() if p == pol], ignore_index=True)
          for pol in ("full", "win")}
    out, sparse = {}, pd.DataFrame()
    for label, (method, policy) in ENTRIES.items():
        pts = obs[obs["entry"] == label]
        if method == "MDMx":
            out[label] = mdmx_quantiles(pts, pi_levels, calibration)
        elif method == "FF":
            ff_q, sparse = flowfield.empirical_intervals(pts, calibration, pi_levels, min_points)
            out[label] = ff_q[key4 + qcols]
        elif method == "BL" and len(pts):
            out[label] = cp.quantiles_from_intervals(pts, pi_levels)[key4 + qcols]
        elif method == "LC":
            out[label] = lc_q[policy][key4 + qcols]
        elif method in ("HU", "CFDM"):
            out[label] = simulated_quantiles(pi[policy], method, levels, both="comonotone" if method == "HU" else "own")
    out[NAIVE] = naive_q[key4 + qcols]
    out["Ensemble"] = ensemble_quantiles({m: out[m] for m in ensemble}, qcols)
    return out, sparse


def simulated_quantiles(q: pd.DataFrame, method: str, levels: Sequence[float], *, both: str) -> pd.DataFrame:
    """One ``demography`` method's simulated e₀ quantiles, one row per point.

    Paper: eq:ev-intervals, sec:evaluation:intervals

    Parameters
    ----------
    q : the interval runs' quantiles: ``method, pop, sex, origin, year, level, e0`` (long)
    method : ``"HU"`` or ``"CFDM"``
    levels : the quantile levels (:func:`mdmx.compare.quantile_levels`)
    both : ``"own"`` keeps the method's own "both" rows (the coherent model simulates the sexes
        jointly); ``"comonotone"`` builds "both" from the female and male quantiles as if the sexes
        moved together (:func:`mdmx.compare.comonotone_both`), the rule for a method that forecasts
        the sexes separately

    Returns
    -------
    ``pop, sex, origin, year`` and the quantile columns.
    """
    if both not in ("own", "comonotone"):
        raise ValueError(f"both must be 'own' or 'comonotone', not {both!r}")
    key4 = ["pop", "sex", "origin", "year"]
    qcols = cp.quantile_columns(levels)
    names = dict(zip(np.round(levels, 6), qcols, strict=True))
    q = q[q["method"] == method].assign(level=lambda d: np.round(d["level"], 6))
    q = q.pivot(index=key4, columns="level", values="e0").rename(columns=names).reset_index()
    if both == "comonotone":
        sexes = q[q["sex"] != "both"]
        q = pd.concat([sexes, cp.comonotone_both(sexes.assign(method=method), qcols).drop(columns="method")],
                      ignore_index=True)
    return q[key4 + qcols]


def schedule_scores(entries: Mapping[str, Mapping[int, Mapping]], keys: pd.DataFrame, M: np.ndarray,
                    pops: Sequence[str], years: Sequence[int], groups: Mapping[str, Sequence[int]],
                    naive: np.ndarray | None = None, *, old_age: int | None = None) -> dict[str, pd.DataFrame]:
    """The schedule measures at every point of ``keys``, for every entry (and the naive's schedules,
    aligned to ``keys``, if given): the RMSE of logit(q_x) over all ages and of the age-specific male −
    female difference, weighted by the observed survivors l_x (``rmse_schedule``, ``rmse_sexdiff``) and
    by the observed deaths d_x (``rmse_schedule_dx``, ``rmse_sexdiff_dx``), and, with ``old_age``, the
    signed error of life expectancy at that age by sex (``err_e{old_age}female``, ``…male``).

    Parameters
    ----------
    entries : entry → origin → result with ``schedule_index`` and ``schedules`` (N, 2, A)
    keys : the scored points ``pop, origin, year``
    M : (2, A, C, T) observed logit(q_x); pops, years : its populations and years
    groups : age group → ages (the errors' weights); naive : (len(keys), 2, A) or None
    old_age : the age of the old-age measure (e.g. 65), or None

    Returns
    -------
    entry (and ``Naive``) → ``keys`` with the measure columns.
    """
    pidx = {p: i for i, p in enumerate(pops)}
    years = np.asarray(years)
    z_obs = np.moveaxis(M[:, :, keys["pop"].map(pidx).to_numpy(), (keys["year"] - years[0]).to_numpy()], -1, 0)
    zs = {label: cp.aligned_schedules(by_origin, keys) for label, by_origin in entries.items()}
    if naive is not None:
        zs[NAIVE] = naive
    return {label: keys.assign(**schedule_measures(z, z_obs, groups, old_age)) for label, z in zs.items()}


def schedule_measures(z: np.ndarray, z_obs: np.ndarray, groups: Mapping[str, Sequence[int]],
                      old_age: int | None = None) -> dict[str, np.ndarray]:
    """The schedule measures of forecast schedules ``z`` against the observed ``z_obs`` (:func:`schedule_scores`).

    Parameters
    ----------
    z, z_obs : (N, 2, A) forecast and observed logit(q_x); groups : age group → ages;
    old_age : the age of the old-age measure, or None

    Returns
    -------
    column → (N,): ``rmse_schedule``, ``rmse_sexdiff`` (survivors' weights), ``rmse_schedule_dx``,
    ``rmse_sexdiff_dx`` (deaths' weights) and, with ``old_age``, ``err_e{old_age}female``, ``…male``.
    """
    cols = {"rmse_schedule": fc.schedule_errors(z, z_obs, groups)["all"].to_numpy(),
            "rmse_sexdiff": cp.sex_difference_errors(z, z_obs),
            "rmse_schedule_dx": fc.schedule_errors(z, z_obs, groups, weights="deaths")["all"].to_numpy(),
            "rmse_sexdiff_dx": cp.sex_difference_errors(z, z_obs, weights="deaths")}
    if old_age is not None:
        err = lifetable.ex_from_logit(z, old_age) - lifetable.ex_from_logit(z_obs, old_age)
        cols[f"err_e{old_age}female"], cols[f"err_e{old_age}male"] = err[:, 0], err[:, 1]
    return cols


def open_interval_schedules(z: np.ndarray, open_age: int) -> np.ndarray:
    """Schedules with the ages from ``open_age`` on folded into one open interval (a scoring sensitivity).

    The comparators forecast one rate for the open interval and repeat it at every single age above it, so
    a measure summed over single ages counts that one forecast as many times as the interval has ages.
    Folded, the interval is one cell: the logit of the probability of dying between ``open_age`` and the
    schedule's last age, 1 − l_A / l_open.  The survivor and death weights of
    :func:`mdmx.forecast.schedule_errors` then give it l_open and l_open (1 − l_A / l_open), the
    interval's survivors and deaths.

    Parameters
    ----------
    z : (..., A) logit(q_x), ages 0 … A−1; open_age : the first age of the open interval, 0 < open_age < A

    Returns
    -------
    (..., open_age + 1): the ages below ``open_age`` unchanged, then the open interval.
    """
    z = np.asarray(z, dtype=float)
    if not 0 < open_age < z.shape[-1]:
        raise ValueError(f"open_age must lie inside the schedule's ages (1 … {z.shape[-1] - 1}), not {open_age}")
    # log(1 − q_x) = −softplus(z_x): the interval's log survival, then its logit, without forming q near 1.
    log_surv = -np.sum(np.logaddexp(0.0, z[..., open_age:]), axis=-1)
    log_dead = np.log(-np.expm1(log_surv))
    return np.concatenate([z[..., :open_age], (log_dead - log_surv)[..., None]], axis=-1)


def empirical_quantiles(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame, entries: Sequence[str], *,
                        calibration: Mapping[int, Sequence[int]], pi_levels: Sequence[int],
                        min_points: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every entry's predictive quantiles of e₀ by one rule: the empirical quantiles of its own errors.

    Paper: eq:ff-intervals

    The flow field's rule (:func:`mdmx.flowfield.empirical_intervals`) applied to each entry: its errors at
    the other origins of the era (the held-out origins: the late era) at the same horizon and sex, so
    that every entry's intervals are calibrated alike (Deviation 2: the methods' own intervals are
    calibrated in different ways, or not at all).  Only the scored points' observations calibrate.

    Parameters
    ----------
    points : forecasts ``entry, pop, sex, origin, year, e0_fc``; truth : ``pop, sex, year, e0_obs``;
    keys : the scored points; entries : the entries (the naive included if wanted)
    calibration : origin → the origins that calibrate it; pi_levels : nominal coverages, per cent;
    min_points : the fewest errors a cell needs

    Returns
    -------
    (entry → ``pop, sex, origin, year`` and the quantile columns; the cells without enough errors,
    with an ``entry`` column).
    """
    qcols = cp.quantile_columns(cp.quantile_levels(pi_levels))
    out, sparse = {}, []
    for label in entries:
        pts = observed_at(points[points["entry"] == label][["pop", "sex", "origin", "year", "e0_fc"]], truth, keys)
        q, sp = flowfield.empirical_intervals(pts, calibration, pi_levels, min_points)
        out[label] = q[["pop", "sex", "origin", "year", *qcols]]
        sparse.append(sp.assign(entry=label))
    return out, pd.concat(sparse, ignore_index=True)


def point_values(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame,
                 quantiles: Mapping[str, pd.DataFrame], schedules: Mapping[str, pd.DataFrame],
                 levels: Sequence[float], families: Mapping[str, str], design: Design) -> pd.DataFrame:
    """The measures of every entry (and the naive) at every point of ``keys``, for
    :func:`mdmx.compare.overall`: ``method, point, cluster, origin, h, era, cell`` (origin and horizon band,
    for the ``cellbias`` measures) and the measure columns.

    Parameters
    ----------
    points : every entry's forecasts and the naive's (entry ``Naive``): ``entry, pop, sex, origin,
        year, e0_fc``; truth : ``pop, sex, year, e0_obs``; keys : the scored points
    quantiles : entry → quantiles (:func:`entry_quantiles`); schedules : entry → :func:`schedule_scores`
    levels : the quantile levels; families : population → its population family (the bootstrap's
        cluster); design : the eras
    """
    frames = []
    for label in quantiles:
        e0 = points[points["entry"] == label].merge(keys, on=["pop", "origin", "year"]).merge(
            truth, on=["pop", "sex", "year"])[["pop", "origin", "year", "sex", "e0_fc", "e0_obs"]]
        q = quantiles[label].merge(keys, on=["pop", "origin", "year"])
        v = cp.point_values(e0, q, levels, schedules.get(label)).assign(method=label)
        frames.append(v)
    out = pd.concat(frames, ignore_index=True)
    out["point"] = out["pop"] + "|" + out["origin"].astype(str) + "|" + out["year"].astype(str)
    out["cluster"] = out["pop"].map(families)
    out["h"] = out["year"] - out["origin"]
    out["era"] = out["origin"].map({o: design.era(o) for o in out["origin"].unique()})
    out["cell"] = out["origin"].astype(str) + "|" + design.band(out["h"].to_numpy())   # for cellbias measures
    return out


def observed_at(points: pd.DataFrame, truth: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """``points`` with the observed e₀ (``e0_obs``) at the scored points ``keys`` only.

    Elsewhere ``e0_obs`` is NaN, so an observation that is not scored (the held-out block's, while it is
    not scored) can never calibrate an interval.

    Parameters
    ----------
    points : forecasts with ``pop, sex, origin, year``; truth : ``pop, sex, year, e0_obs``;
    keys : the scored points ``pop, origin, year`` (other columns are ignored)
    """
    scored = keys[["pop", "origin", "year"]].drop_duplicates().assign(_scored=True)
    obs = points.merge(truth[["pop", "sex", "year", "e0_obs"]], on=["pop", "sex", "year"], how="left")
    obs = obs.merge(scored, on=["pop", "origin", "year"], how="left")
    obs.loc[obs["_scored"].isna(), "e0_obs"] = np.nan
    return obs.drop(columns="_scored")


def mdmx_quantiles(obs: pd.DataFrame, pi_levels: Sequence[int], calibration: Mapping[int, Sequence[int]], *,
                   kappa: bool = True) -> pd.DataFrame:
    """MDMx's predictive quantiles: normal, with the delta-method SD scaled by κ.

    Paper: eq:fc-kappa, sec:forecasting:intervals

    Parameters
    ----------
    obs : MDMx's points with ``pop, sex, origin, year, e0_fc, sd, e0_obs`` (:func:`observed_at`)
    pi_levels : nominal coverages, per cent; calibration : origin → the origins that calibrate it
        (:func:`mdmx.flowfield.within_era_calibration`)
    kappa : False for κ = 1 (a sensitivity)

    Returns
    -------
    ``pop, sex, origin, year`` and the quantile columns (:func:`mdmx.compare.quantile_columns`).
    """
    levels = cp.quantile_levels(pi_levels)
    cal = fc.loo_calibrate(obs, pi_levels, calibration=calibration)
    sd = cal["sd"] * (cal["kappa_loo"] if kappa else 1.0)
    q = cp.gaussian_quantiles(cal["e0_fc"].to_numpy(), sd.to_numpy(), levels)
    return cal[["pop", "sex", "origin", "year"]].reset_index(drop=True).join(
        pd.DataFrame(q, columns=cp.quantile_columns(levels)))


def coverage_table(quantiles: Mapping[str, pd.DataFrame], keys: pd.DataFrame, truth: pd.DataFrame,
                   pi_levels: Sequence[int], spans: Mapping[str, tuple[int, int]]) -> pd.DataFrame:
    """Share of the observed e₀ (average of the sexes) inside each entry's intervals, by era and span.

    Parameters
    ----------
    quantiles : entry → ``pop, sex, origin, year`` and the quantile columns
    keys : the scored points ``pop, origin, year, era``; truth : ``pop, sex, year, e0_obs``
    pi_levels : nominal coverages, per cent; spans : label → (first, last) horizon, inclusive

    Returns
    -------
    ``entry, era, span, n`` and ``in{L}`` (the share inside the level-L interval) for each level.
    """
    names = dict(zip(np.round(cp.quantile_levels(pi_levels), 6),
                     cp.quantile_columns(cp.quantile_levels(pi_levels)), strict=True))
    rows = []
    for label, q in quantiles.items():
        s = q[q["sex"] == "both"].merge(keys, on=["pop", "origin", "year"]).merge(truth, on=["pop", "sex", "year"])
        h = s["year"] - s["origin"]
        cols = []
        for lev in pi_levels:
            lo, hi = round((100 - lev) / 200, 6), round(1 - (100 - lev) / 200, 6)
            s[f"in{lev}"] = (s["e0_obs"] >= s[names[lo]]) & (s["e0_obs"] <= s[names[hi]])
            cols.append(f"in{lev}")
        for span, (a, z) in spans.items():
            g = s[h.between(a, z)].groupby("era")
            t = g[cols].mean().assign(n=g.size())
            rows.append(t.assign(entry=label, span=span).reset_index())
    return pd.concat(rows, ignore_index=True)[["entry", "era", "span", "n", *[f"in{lev}" for lev in pi_levels]]]


def complete(values: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The points at which every method has every measure, for :func:`mdmx.compare.overall` (T5b's rule).

    A point where some method has no value of some measure (an R failure, a flow-field cell without
    enough errors) or no row at all (a sensitivity's variant) is left out for every method.

    Parameters
    ----------
    values : :func:`point_values` (``method, point`` and the measure columns)

    Returns
    -------
    (the rows of the complete points, the number of points left out).
    """
    meta = {"pop", "origin", "year", "method", "point", "cluster", "h", "era", "cell"}
    cols = [c for c in values.columns if c not in meta]
    per_point = values.groupby("point")["method"].nunique()
    bad = set(values.loc[values[cols].isna().any(axis=1), "point"])
    bad |= set(per_point.index[per_point < values["method"].nunique()])
    return values[~values["point"].isin(bad)], len(bad)


def with_changes(points: pd.DataFrame, quantiles: Mapping[str, pd.DataFrame], schedules: Mapping[str, pd.DataFrame],
                 changes: Mapping[str, tuple]) -> tuple[pd.DataFrame, dict, dict]:
    """The entries' points, quantiles and schedule scores with some entries replaced (a sensitivity).

    Parameters
    ----------
    points : every entry's forecasts (``entry`` column); quantiles, schedules : entry → frame
    changes : entry → ``(points, quantiles, schedule scores)``; ``None`` keeps that part

    Returns
    -------
    (points, quantiles, schedules), new objects; the inputs are not changed.
    """
    pts, q, sc = points.copy(), dict(quantiles), dict(schedules)
    for label, (p_new, q_new, s_new) in changes.items():
        if label not in q:
            raise KeyError(f"no entry {label!r} to replace")
        if p_new is not None:
            pts = pd.concat([pts[pts["entry"] != label], p_new.assign(entry=label)], ignore_index=True)
        if q_new is not None:
            q[label] = q_new
        if s_new is not None:
            sc[label] = s_new
    return pts, q, sc


def jumpoff_schedules(M: np.ndarray, pops: Sequence[str], years: Sequence[int], jobs: pd.DataFrame,
                      keys: pd.DataFrame) -> np.ndarray:
    """The observed schedule of each point's jump-off year, the job's last training year t₀.

    Paper: eq:ff-anchor, eq:ff-path

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); pops, years : its labels
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (``pop, origin, last_train_year``)
    keys : the points ``pop, origin, year``

    Returns
    -------
    (len(keys), S, A) logit(q_x).
    """
    t0 = keys[["pop", "origin"]].merge(jobs[["pop", "origin", "last_train_year"]], on=["pop", "origin"],
                                      how="left", validate="many_to_one")["last_train_year"]
    if t0.isna().any():
        raise ValueError("a point has no job")
    c = keys["pop"].map({p: i for i, p in enumerate(pops)}).to_numpy()
    t = t0.to_numpy(dtype=int) - int(np.asarray(years)[0])
    return np.moveaxis(M[:, :, c, t], -1, 0)


def jumpoff_residuals(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                      jobs: pd.DataFrame, keys: pd.DataFrame, settings: fc.MDMxSettings) -> np.ndarray:
    """The jump-off residual as observed at each point, before the anchor smooths it.

    Paper: eq:fc-anchor, eq:ff-anchor

    r = M_{c,t₀} − z̄ − L s(t₀) in the space of the point's origin (:func:`mdmx.forecast.origin_space` with
    MDMx's settings, the space both anchored forecasters use), t₀ the job's last training year.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); observed : (C, T) the observed, non-exceptional country-years;
        pops, years : their labels
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (``pop, origin, last_train_year``)
    keys : the points ``pop, origin, year``; settings : MDMx's settings (the space)

    Returns
    -------
    (len(keys), S, A) logit(q_x).
    """
    t0 = keys[["pop", "origin"]].merge(jobs[["pop", "origin", "last_train_year"]], on=["pop", "origin"],
                                      how="left", validate="many_to_one")["last_train_year"]
    if t0.isna().any():
        raise ValueError("a point has no job")
    years = np.asarray(years)
    c = keys["pop"].map({p: i for i, p in enumerate(pops)}).to_numpy()
    t = t0.to_numpy(dtype=int) - int(years[0])
    origin = keys["origin"].to_numpy(dtype=int)
    out = np.empty((len(keys),) + M.shape[:2])
    for o in np.unique(origin):
        space = fc.origin_space(M, observed, years, int(o), tau=settings.tau, age_rank=settings.age_rank,
                                smooth=settings.smooth, n_pc=settings.n_pc)
        rows = np.flatnonzero(origin == o)
        s0 = space["scores"][c[rows], t[rows]]                                      # (n, p)
        if not np.isfinite(s0).all():
            raise ValueError(f"a point of origin {o} has no scores at its jump-off year")
        out[rows] = np.moveaxis(M[:, :, c[rows], t[rows]], -1, 0) - fc.scores_to_logit(s0, space["z_mean"],
                                                                                       space["L"])
    return out


def with_observed_residual(z_fc: np.ndarray, resid: np.ndarray, lam: float) -> np.ndarray:
    """Persistently anchored forecasts with the jump-off residual as observed in place of the smoothed one.

    Paper: eq:fc-anchor, eq:ff-anchor

    An anchored forecast is ``z̄ + L s(h) + K r`` (:func:`mdmx.forecast.smooth_residual`), so
    ``z(h) + r − K r`` is the same forecast carrying the residual ``r`` unsmoothed: a sensitivity.

    Parameters
    ----------
    z_fc, resid : (N, S, A) the forecasts and their jump-off residuals as observed (:func:`jumpoff_residuals`);
    lam : the smoothing the forecasts were made with

    Returns
    -------
    (N, S, A) logit(q_x).
    """
    return z_fc + resid - fc.smooth_residual(resid, lam)


def replaced_schedule_values(values: pd.DataFrame, keys: pd.DataFrame, z_old: np.ndarray, z_new: np.ndarray,
                             z_obs: np.ndarray, old_age: int) -> pd.DataFrame:
    """One entry's point values (:func:`point_values`) with forecast schedules ``z_new`` in place of ``z_old``.

    The errors of female and male e₀ and of the gap are shifted by the change of the schedules' life
    expectancies, and the schedule measures are those of ``z_new`` (:func:`schedule_measures`).  The
    measures of the average of the sexes and the interval scores are set to NaN: they are not replaced.

    Parameters
    ----------
    values : the entry's rows of :func:`point_values` (one per point of ``keys``)
    keys : the points ``pop, origin, year``, in the order of the schedules
    z_old, z_new, z_obs : (len(keys), 2, A) the entry's schedules, their replacement and the truth
    old_age : the age of the old-age measure

    Returns
    -------
    ``values`` in the order of ``keys``, with the replaced columns.
    """
    v = keys[["pop", "origin", "year"]].merge(values, on=["pop", "origin", "year"], how="left",
                                              validate="one_to_one")
    if v["method"].isna().any():
        raise ValueError("a point has no values")
    shift = lifetable.e0_from_logit(z_new) - lifetable.e0_from_logit(z_old)            # (N, 2)
    for q, d in (("female", shift[:, 0]), ("male", shift[:, 1]), ("gap", shift[:, 0] - shift[:, 1])):
        v[f"absbias:{q}"] = v[f"absbias:{q}"] + d
        v[f"mae:{q}"] = v[f"absbias:{q}"].abs()
    for c in [c for c in v.columns if c.endswith(":both") or c.startswith("wis:")]:
        v[c] = np.nan
    for name, col in schedule_measures(z_new, z_obs, {}, old_age).items():
        if name.startswith("rmse_"):
            v[f"rmse:{name[5:]}"] = col
        else:
            v[f"mae:{name[4:]}"], v[f"absbias:{name[4:]}"] = np.abs(col), col
    return v


def scaled_measures(values: pd.DataFrame, codes: Sequence[str], methods: Sequence[str], naive: str) -> pd.DataFrame:
    """Each measure's mean over the points divided by the naive's, the full-sample ``scaled`` of
    :func:`mdmx.compare.overall` (``mae`` codes by mean absolute error, ``rmse`` by the mean of the
    points' values), for the given methods on the naive's points.

    Returns
    -------
    DataFrame method × code.
    """
    nv = values[values["method"] == naive].set_index("point")

    def mean(frame: pd.DataFrame, code: str) -> float:
        col = frame[code].to_numpy(dtype=float)
        if not np.all(np.isfinite(col)):
            raise ValueError(f"non-finite values of {code}")
        return float(np.mean(np.abs(col) if code.startswith("mae:") else col))

    out = {}
    for m in methods:
        v = values[values["method"] == m].set_index("point")
        if not v.index.sort_values().equals(nv.index.sort_values()):
            raise ValueError(f"method {m} is not scored on the naive's points")
        v = v.loc[nv.index]
        out[m] = {c: mean(v, c) / mean(nv, c) for c in codes}
    return pd.DataFrame(out).T


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``4th``, …, ``11th``, ``21st``."""
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rank_positions(mean_rank: pd.Series) -> pd.Series:
    """Each method's position by mean SMAA rank (1 = the lowest mean rank; ties share the better one)."""
    return mean_rank.rank(method="min").astype(int)


FLOW_FIELD, MDMX_ALL, MDMX_WIN, ENSEMBLE = ("Flow field", "MDMx + anchor, all years",
                                            "MDMx + anchor, last 50 years", "Ensemble")


def entry_phrase(label: str) -> str:
    """An entry's label as a phrase for a sentence: "the flow field", "anchored MDMx fitted to all years"."""
    if label == FLOW_FIELD:
        return "the flow field"
    if label == ENSEMBLE:
        return "the ensemble"
    name, _, policy = label.partition(", ")
    name = "anchored MDMx" if name == "MDMx + anchor" else name
    return f"{name} fitted to {'all years' if policy == 'all years' else 'the ' + policy}" if policy else name


def headline(result: Mapping, view: str, *, origins: tuple[int, int], horizons: tuple[int, int],
             window: int, ensemble: Sequence[str], post_hoc: bool = False) -> str:
    """The headline sentences of ``docs/changes/12-evaluation.md`` §3.1, approved before any number.

    Parameters
    ----------
    result : :func:`mdmx.compare.overall` for the view (``first``, ``mean_rank``)
    view : ``"primary"`` (the headline: one of the four templates), ``"early"`` (the early era, beside
        it) or ``"held"`` (the held-out block, once scored)
    origins : the view's first and last origin; horizons : its first and last horizon
    window : the second fitting period, years; ensemble : the ensemble's members, in the template's
        order (MDMx on all years, the flow field, Lee–Carter on the last ``window`` years)
    post_hoc : ``view="held"`` only: the comparison was revised after the block had been scored
        (Deviation 2), so the sentence says that instead of calling the block held out from every choice

    Returns
    -------
    The sentence, with ranks as ordinals (position by mean rank among the ranked entries).
    """
    pos = rank_positions(result["mean_rank"])
    first = result["first"].idxmax()
    share = f"{100 * result['first'][first]:.0f}%"
    o1, o2 = origins
    mdmx_ranks = f"{ordinal(pos[MDMX_ALL])} when fitted to all years and {ordinal(pos[MDMX_WIN])} when fitted to " \
                 f"the last {window}"
    if view == "primary":
        lead = f"On forecasts from origins {o1}–{o2} at horizons of {horizons[0]}–{horizons[1]} years, "
        if first == FLOW_FIELD:
            return (lead + f"the flow field ranks first in the overall comparison of age-specific and summary "
                    f"forecasts ({share} of the draws), ahead of anchored MDMx, which ranks {mdmx_ranks}.")
        if first in (MDMX_ALL, MDMX_WIN):
            other = MDMX_WIN if first == MDMX_ALL else MDMX_ALL
            return (lead + f"{entry_phrase(first)} ranks first in the overall comparison of age-specific and "
                    f"summary forecasts ({share} of the draws); the flow field ranks {ordinal(pos[FLOW_FIELD])}, "
                    f"and {entry_phrase(other)} ranks {ordinal(pos[other])}.")
        if first == ENSEMBLE:
            ranks = [ordinal(pos[m]) for m in ensemble]
            return (lead + f"the equally weighted ensemble of anchored MDMx, the flow field and Lee–Carter ranks "
                    f"first in the overall comparison ({share} of the draws).  Its members rank "
                    f"{', '.join(ranks[:-1])} and {ranks[-1]}.")
        return (lead + f"{entry_phrase(first)} ranks first in the overall comparison ({share} of the "
                f"draws); the flow field ranks {ordinal(pos[FLOW_FIELD])}, and anchored MDMx ranks {mdmx_ranks}.")
    if view not in ("early", "held"):
        raise ValueError(f"view must be primary, early or held, not {view!r}")
    if post_hoc and view != "held":
        raise ValueError("post_hoc applies to the held-out view only")
    if view == "early":
        start = f"In the early era (origins {o1}–{o2}), "
    elif post_hoc:
        start = (f"On the held-out origins {o1}–{o2}, scored once under the frozen design and compared here under "
                 "the comparison revised afterwards, ")
    else:
        start = f"On the held-out origins {o1}–{o2}, held out from every formal choice of form and design, "
    where = "" if view == "early" else " in the same view"
    ff = "" if first == FLOW_FIELD else f"the flow field ranks {ordinal(pos[FLOW_FIELD])} and "
    return (start + f"{entry_phrase(first)} ranks first{where} ({share}); {ff}anchored MDMx ranks {mdmx_ranks}.")


def relative_errors(values: pd.DataFrame, methods: Sequence[str], naive: str, codes: Sequence[str]) -> pd.DataFrame:
    """Each method's error relative to the naive's, population by population (a complement to the pooled ratios).

    For each code and population: the method's mean of the code's values over the population's points
    (|error| for ``mae:q``, each point's RMSE over ages for ``rmse:…``) divided by the naive's on the same
    points.  Summaries over the populations: the share where the method beats
    the naive (ratio below 1), and the geometric mean of the ratios (AvgRelMAE), in which each population
    counts once, so a few populations with large errors cannot decide it.

    Parameters
    ----------
    values : :func:`point_values` (``method, pop`` and the codes' columns); methods : the methods;
    naive : the yardstick; codes : ``mae:q`` or ``rmse:…`` codes

    Returns
    -------
    ``method, code, populations, beats_naive`` (share) ``, avg_rel_mae``.
    """
    per_pop = values[values["method"].isin([*methods, naive])].groupby(["method", "pop"])[list(codes)].mean()
    base = per_pop.xs(naive, level="method")
    rows = []
    for m in methods:
        ratio = per_pop.xs(m, level="method").div(base)
        for code in codes:
            r = ratio[code].dropna()
            rows.append({"method": m, "code": code, "populations": len(r), "beats_naive": float((r < 1).mean()),
                         "avg_rel_mae": float(np.exp(np.log(r).mean()))})
    return pd.DataFrame(rows)


def paired_differences(errors: pd.DataFrame, pairs: Sequence[tuple[str, str]], bands: Sequence[tuple[int, int]], *,
                       draws: int, seed: int) -> pd.DataFrame:
    """Paired differences of MAE between entries by horizon band, with a bootstrap over origins.

    For each pair (a, b) and band, on the points where both are scored, d = MAE_a − MAE_b, the mean
    absolute error of a minus that of b over the same points.  Its interval resamples whole origins with
    replacement, ``draws`` times: the points of one origin share its period, so the origins, not the
    points, are the independent units (a block bootstrap).  The share of draws with d < 0 is reported
    beside the 2.5 and 97.5 percentiles; it is a description of the resampling, not a p-value.

    Parameters
    ----------
    errors : one measure's errors, ``entry, pop, origin, year, h, error`` (and ``sex`` if there is one)
    pairs : (a, b) entry labels; bands : (first, last) horizons, inclusive
    draws : bootstrap draws; seed : the random generator's seed

    Returns
    -------
    DataFrame, one row per pair and band: ``a, b, band, n_points, n_origins, diff, lo, hi, share_below``
    (MAE in the errors' units).

    Raises
    ------
    ValueError
        If an entry of a pair has no errors.
    """
    keys = [c for c in ("pop", "sex", "origin", "year", "h") if c in errors.columns]
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        ea = errors[errors["entry"] == a]
        eb = errors[errors["entry"] == b]
        if ea.empty or eb.empty:
            raise ValueError(f"no errors for {a if ea.empty else b!r}")
        both = ea[keys + ["error"]].merge(eb[keys + ["error"]], on=keys, suffixes=("_a", "_b"))
        for lo, hi in bands:
            d = both[both["h"].between(lo, hi)]
            if d.empty:
                continue
            g = d.assign(abs_a=d["error_a"].abs(), abs_b=d["error_b"].abs()).groupby("origin")
            sa, sb, n = g["abs_a"].sum().to_numpy(), g["abs_b"].sum().to_numpy(), g.size().to_numpy()
            k = len(n)
            counts = rng.multinomial(k, np.full(k, 1.0 / k), size=draws)       # (draws, origins)
            boot = (counts @ sa - counts @ sb) / (counts @ n)
            rows.append({"a": a, "b": b, "band": f"{lo}–{hi}", "n_points": int(n.sum()), "n_origins": int(k),
                         "diff": float((sa.sum() - sb.sum()) / n.sum()),
                         "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
                         "share_below": float(np.mean(boot < 0))})
    return pd.DataFrame(rows)

