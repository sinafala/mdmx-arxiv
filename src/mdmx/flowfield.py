"""The flow-field forecaster (paper §11, the second forecaster).

The idea
--------
Every observed schedule is a point in the space of the first ``n_pc`` PCA
scores of the effective cores, the same space as the MDMx forecaster
(:func:`mdmx.forecast.origin_space`).  The first score s₁ is the overall
level of mortality.  Pooled over all populations and years up to the
forecast origin, the observed schedules trace a *flow*: how fast s₁ moves
at each level, and where the other scores sit at each level.  The flow
field forecasts a population by following that flow from its last observed
schedule:

1. **The speed** (:func:`speed_function`).  ds₁/dt = g(s₁), a LOWESS of each
   population's smoothed annual change of s₁ against s₁.  It is never
   allowed to reverse the improvement.
2. **The frontier point** (:func:`orientation`).  s₁* is the median s₁ of
   the population-years whose e₀ is in the top 10 %.  Below it, populations
   move at the pace that populations at the same level moved in the past;
   beyond it, there is no past to learn from.
3. **The pace beyond the frontier** (:func:`frontier_level`,
   :func:`frontier_speed`, :func:`frontier_rule`).  The leading populations
   are the top three by e₀ in each year.  Their mean s₁, chain-linked so that
   populations entering the data do not move it, gives the frontier's pace
   over the last 50 years.  Beyond s₁*, the speed is that pace, shrunk
   towards g(s₁*) by the number of populations that led (rule FR1).  Two
   other rules are variants: FR0 holds the speed at g(s₁*), and FR2 moves s₁
   so that the e₀ of the typical schedule (:func:`typical_e0`) rises at the
   leaders' pace in e₀.
4. **The shape** (:func:`shape_law`).  The other scores follow s₁ along
   s_k = f_k(s₁), a LOWESS of s_k on s₁, and a population's own deviation
   from f_k decays at the pooled rate α_k per year.  The shape law is
   estimated on the last 50 years, so that it describes the current age
   pattern of improvement; beyond the frontier point of those 50 years (not
   s₁*, which uses every year) f_k runs along its tangent.
5. **The path and the anchor** (:func:`integrate`, :func:`forecast_origin`).
   The scores are integrated year by year; the schedules follow from the
   linear map of the space; the jump-off residual (the observed schedule
   minus its projection on the scores), smoothed over age so that it keeps
   the population's detail but not the noise of that one year's rates, is
   kept at every horizon.

Every quantity is estimated from data up to the origin only, and the
forecaster is deterministic.  The form (which parts use which window, and
the rule beyond the frontier) is set by :class:`FlowFieldSettings`.  Its
default, from ``[flowfield]``, is the form chosen for the article: speed on
all years, shape law on the last 50 years, the frontier's pace beyond s₁*.
Variants, such as the flat speed beyond s₁* or every part on a window, are
the same settings with fields replaced (:func:`dataclasses.replace`).

Three switches each remove one part of the form, to measure what it
contributes (the ablation of chapter 12): ``shape_law`` (False: the other
scores stay at their jump-off values), ``relax`` (``"persist"`` or
``"vanish"``: a population's deviation from f_k is kept or dropped instead of
decaying at α_k) and ``anchor`` (False: no jump-off residual).  The speed
without the frontier's pace is ``frontier_rule="FR0"``.  Their defaults are
the article's form.

:func:`empirical_intervals` gives the flow field's predictive intervals:
quantiles of its own errors at other origins.

Each function that implements an equation of the article names its LaTeX
labels in a ``Paper:`` line.

Conventions
-----------
* Index 0 of a sex axis is female and 1 is male.  The e₀ that orients s₁ and
  ranks the populations is the average of the female and male e₀
  (:func:`e0_matrix`).
* Scores are ``(C, T, p)``: populations, calendar years, PCA components;
  NaN outside the population-years used.
* Errors are ``forecast − observed``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from . import compare, lifetable
from . import forecast as fc

__all__ = [
    "FlowFieldSettings", "e0_matrix", "orientation", "speed_function", "shape_law", "frontier_level",
    "frontier_slope", "frontier_speed", "typical_e0", "frontier_rule", "build_flow", "flow_summary",
    "integrate", "forecast_origin", "within_era_calibration", "empirical_intervals",
]

#: The rules for the speed beyond the frontier point: ``FR0`` holds it at
#: g(s₁*); ``FR1`` takes it from the leading populations' pace in s₁;
#: ``FR2`` from their pace in e₀, through the typical schedule.
FRONTIER_RULES = ("FR0", "FR1", "FR2")

#: How a population's deviation from the shape law evolves: ``estimated``
#: decays at the pooled α_k; ``persist`` keeps it (α_k = 1); ``vanish`` drops
#: it from the first forecast year (α_k = 0).
RELAX_MODES = ("estimated", "persist", "vanish")


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FlowFieldSettings:
    """Every tunable value of the flow field (read from ``[flowfield]``).

    The space (``n_pc``, the age rank, the smoothing of the age basis) is
    MDMx's, from :class:`mdmx.forecast.MDMxSettings`.

    Attributes
    ----------
    speed_frac : LOWESS span of the speed g and of the shape law f_k
    smooth_frac_min, smooth_years : each population's s₁ is smoothed over
        time by a LOWESS with span ``max(smooth_frac_min, smooth_years / n)``
        for n observed years, so that every local fit spans at least
        ``smooth_years`` years
    lowess_delta : statsmodels' ``delta`` of g and f_k, as a share of the
        range of s₁: points closer than this are interpolated instead of
        fitted (an approximation that saves time)
    frontier_quantile : s₁* is the median s₁ of the population-years whose
        e₀ is at or above this quantile
    tangent_share : beyond its frontier point, f_k runs along its
        least-squares slope over this share of the range of s₁ before it
    relax_max_lag : calendar lags (years) of the deviations' autocorrelation
    relax_min_beta : lags whose pooled autocorrelation is at most this are
        left out of the fit of α_k
    relax_alpha_max : upper bound of α_k
    min_years : populations with fewer usable years are left out of the
        speed's points, the shape law and the relaxation (they still count
        for the orientation, s₁* and the frontier)
    frontier_k : number of leading populations (by e₀) in the frontier
    frontier_window : years (up to the origin) of the frontier's slope
    frontier_n0 : the pace beyond s₁* is ``w·V + (1 − w)·g(s₁*)`` with
        ``w = n / (n + frontier_n0)`` for n distinct leading populations
        (positive)
    frontier_min_years : linked frontier years needed for its slope; with
        fewer the rule falls back to ``FR0``
    frontier_e0_step : rule ``FR2`` only: the step of the central difference
        e′(s₁) of the typical schedule's e₀, as a share of the range of s₁
        over the population-years up to the origin
    speed_window, shape_window : years (up to the origin) on which the speed
        (g and s₁*) and the shape law (f_k, α_k and their frontier point) are
        estimated; None for every year
    frontier_rule : ``FR0``, ``FR1`` or ``FR2`` (:data:`FRONTIER_RULES`)
    interval_min_points : an empirical interval needs at least this many
        errors in its calibration cell
    shape_law : False holds the other scores s₂…s_p at their jump-off values
        while s₁ moves as before (an ablation; True is the form)
    relax : ``"estimated"`` (the form: the deviations from f_k decay at the
        estimated α_k), ``"persist"`` (α_k = 1) or ``"vanish"`` (α_k = 0)
        (:data:`RELAX_MODES`); must be ``"estimated"`` without the shape law,
        where it has no effect
    anchor : False drops the jump-off residual, so the forecast schedules are
        the projection of the forecast scores (an ablation; True is the form)
    anchor_smoothing : λ of the Whittaker smoother of the jump-off residual
        (:func:`mdmx.forecast.smooth_residual`); 0 keeps the residual as observed
    """

    speed_frac: float
    smooth_frac_min: float
    smooth_years: int
    lowess_delta: float
    frontier_quantile: float
    tangent_share: float
    relax_max_lag: int
    relax_min_beta: float
    relax_alpha_max: float
    min_years: int
    frontier_k: int
    frontier_window: int
    frontier_n0: float
    frontier_min_years: int
    frontier_e0_step: float
    speed_window: int | None
    shape_window: int | None
    frontier_rule: str
    interval_min_points: int
    shape_law: bool = True
    relax: str = "estimated"
    anchor: bool = True
    anchor_smoothing: float = 0.0

    def __post_init__(self) -> None:
        for name in ("speed_frac", "smooth_frac_min", "tangent_share"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)}.")
        if not 0 < self.frontier_quantile < 1:
            raise ValueError(f"frontier_quantile must be in (0, 1), got {self.frontier_quantile}.")
        if not 0 < self.relax_min_beta < 1:
            raise ValueError(f"relax_min_beta must be in (0, 1), got {self.relax_min_beta}.")
        if not 0 < self.relax_alpha_max <= 1:
            raise ValueError(f"relax_alpha_max must be in (0, 1], got {self.relax_alpha_max}.")
        if self.lowess_delta < 0:
            raise ValueError(f"lowess_delta must not be negative, got {self.lowess_delta}.")
        if self.frontier_n0 <= 0:
            raise ValueError(f"frontier_n0 must be positive, got {self.frontier_n0}.")
        if not 0 < self.frontier_e0_step < 1:
            raise ValueError(f"frontier_e0_step must be in (0, 1), got {self.frontier_e0_step}.")
        for name in ("smooth_years", "relax_max_lag", "frontier_k", "interval_min_points"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}.")
        for name in ("min_years", "frontier_window", "frontier_min_years"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2, got {getattr(self, name)}.")
        for name in ("speed_window", "shape_window"):
            value = getattr(self, name)
            if value is not None and value < self.min_years:
                raise ValueError(f"{name} must be None (every year) or at least min_years, got {value}.")
        if self.frontier_min_years > self.frontier_window:
            raise ValueError(f"frontier_min_years ({self.frontier_min_years}) cannot exceed frontier_window "
                             f"({self.frontier_window}).")
        if self.frontier_rule not in FRONTIER_RULES:
            raise ValueError(f"frontier_rule must be one of {FRONTIER_RULES}, got {self.frontier_rule!r}.")
        # bool(...) would turn the string "false" into True, so the switches must be booleans already.
        for name in ("shape_law", "anchor"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false, got {getattr(self, name)!r}.")
        if self.relax not in RELAX_MODES:
            raise ValueError(f"relax must be one of {RELAX_MODES}, got {self.relax!r}.")
        if not (np.isfinite(self.anchor_smoothing) and self.anchor_smoothing >= 0):
            raise ValueError(f"anchor_smoothing must be finite and not negative, got {self.anchor_smoothing}.")
        if not self.shape_law and self.relax != "estimated":
            raise ValueError(f"relax={self.relax!r} has no effect without the shape law (shape_law=False); "
                             "leave it 'estimated'.")

    @classmethod
    def from_params(cls, ff: Mapping, **overrides) -> FlowFieldSettings:
        """Build the settings from the ``[flowfield]`` parameter section.

        A window given as ``"all"`` becomes None (every year up to the origin).
        """
        def window(value):
            return None if value == "all" else int(value)

        values = dict(
            speed_frac=float(ff["speed_frac"]), smooth_frac_min=float(ff["smooth_frac_min"]),
            smooth_years=int(ff["smooth_years"]), lowess_delta=float(ff["lowess_delta"]),
            frontier_quantile=float(ff["frontier_quantile"]), tangent_share=float(ff["tangent_share"]),
            relax_max_lag=int(ff["relax_max_lag"]), relax_min_beta=float(ff["relax_min_beta"]),
            relax_alpha_max=float(ff["relax_alpha_max"]), min_years=int(ff["min_years"]),
            frontier_k=int(ff["frontier_k"]), frontier_window=int(ff["frontier_window"]),
            frontier_n0=float(ff["frontier_n0"]), frontier_min_years=int(ff["frontier_min_years"]),
            frontier_e0_step=float(ff["frontier_e0_step"]),
            speed_window=window(ff["speed_window"]), shape_window=window(ff["shape_window"]),
            frontier_rule=str(ff["frontier_rule"]), interval_min_points=int(ff["interval_min_points"]),
            shape_law=ff["shape_law"], relax=str(ff["relax"]), anchor=ff["anchor"],
            anchor_smoothing=float(ff["anchor_smoothing"]),
        )
        values.update(overrides)
        return cls(**values)

    def as_dict(self) -> dict:
        """Plain dict of the settings (what the artefact store hashes)."""
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Inputs
# ══════════════════════════════════════════════════════════════════════════════

def e0_matrix(M: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """e₀ of every observed schedule, the average of the sexes.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor
    observed : (C, T) mask of the observed, non-exceptional country-years

    Returns
    -------
    (C, T) e₀ in years, the mean of the female and male e₀ of each schedule;
    NaN outside ``observed``.
    """
    e0 = lifetable.e0_from_logit(np.moveaxis(np.nan_to_num(M), 1, -1)).mean(axis=0)
    return np.where(observed, e0, np.nan)


def _window(keep: np.ndarray, years: np.ndarray, origin: int, window: int | None) -> np.ndarray:
    """``keep`` restricted to the years [origin − window + 1, origin] (all of it for None)."""
    if window is None:
        return keep
    return keep & (np.asarray(years) >= origin - window + 1)[None, :]


def _used(keep: np.ndarray, settings: FlowFieldSettings) -> list[int]:
    """Populations with at least ``min_years`` usable years."""
    return [c for c in range(keep.shape[0]) if keep[c].sum() >= settings.min_years]


# ══════════════════════════════════════════════════════════════════════════════
# The speed and the shape law
# ══════════════════════════════════════════════════════════════════════════════

def _lowess_curve(x: np.ndarray, y: np.ndarray, frac: float, delta_share: float) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS of y on x as a sorted grid with unique x: (gx, gy)."""
    fit = lowess(y, x, frac=frac, it=0, delta=delta_share * np.ptp(x), return_sorted=True)
    gx, first = np.unique(fit[:, 0], return_index=True)
    return gx, fit[first, 1]


def _extend(gx: np.ndarray, gy: np.ndarray, sign: float, s1_star: float, *, beyond: str,
            tangent_share: float) -> Callable[[np.ndarray], np.ndarray]:
    """The curve (gx, gy), and beyond s₁* either flat or along its tangent.

    ``sign`` is +1 if a higher s₁ means lower mortality.  The tangent is the
    least-squares slope of the grid over the last ``tangent_share`` of its
    range before s₁*.  Inside the grid the curve is interpolated linearly;
    on the other side of the grid from s₁* it is held at its end value.
    """
    y_star = float(np.interp(s1_star, gx, gy))
    slope = 0.0
    if beyond == "tangent":
        span = gx[-1] - gx[0]
        if sign > 0:
            sel = (gx <= s1_star) & (gx >= s1_star - tangent_share * span)
        else:
            sel = (gx >= s1_star) & (gx <= s1_star + tangent_share * span)
        if sel.sum() < 3:
            raise ValueError("Fewer than 3 grid points to estimate the tangent at the frontier")
        slope = float(np.polyfit(gx[sel], gy[sel], 1)[0])
    elif beyond != "flat":
        raise ValueError(f"beyond must be 'flat' or 'tangent', not {beyond!r}")

    def f(s):
        s = np.asarray(s, dtype=float)
        out = np.interp(s, gx, gy)
        return np.where((s - s1_star) * sign > 0, y_star + slope * (s - s1_star), out)
    return f


def orientation(scores: np.ndarray, keep: np.ndarray, e0: np.ndarray, settings: FlowFieldSettings
                ) -> tuple[float, float]:
    """The direction of improvement along s₁ and the frontier point s₁*.

    Paper: eq:ff-frontier-point, eq:ff-speed

    Parameters
    ----------
    scores : (C, T, p) PCA scores; keep : (C, T) the population-years used;
    e0 : (C, T) e₀ (:func:`e0_matrix`)

    Returns
    -------
    ``(sign, s1_star)``: ``sign`` is +1 if s₁ rises with e₀ over ``keep``
    and −1 otherwise; ``s1_star`` (raw s₁ units) is the median s₁ of the
    population-years whose e₀ is at or above the ``frontier_quantile``.
    """
    e0v, s1v = e0[keep], scores[keep][:, 0]
    sign = 1.0 if np.corrcoef(s1v, e0v)[0, 1] > 0 else -1.0
    s1_star = float(np.median(s1v[e0v >= np.quantile(e0v, settings.frontier_quantile)]))
    return sign, s1_star


def speed_function(scores: np.ndarray, keep: np.ndarray, years: np.ndarray, e0: np.ndarray,
                   settings: FlowFieldSettings) -> dict:
    """The speed of s₁ as a function of s₁, pooled over populations and years.

    Paper: eq:ff-speed

    Each population's s₁ is smoothed over its observed years by a LOWESS,
    and its annual change is computed between consecutive observed years.
    g is a LOWESS of these changes against the smoothed s₁, pooled over the
    populations; beyond the frontier point s₁* it is held at g(s₁*).  It is
    clipped to the improving direction, so a forecast never reverses the
    improvement, and to the largest speed on the grid.

    Parameters
    ----------
    scores : (C, T, p) PCA scores (:func:`mdmx.forecast.origin_space`)
    keep : (C, T) the population-years used (up to the origin, possibly a
        window of them)
    years : (T,) calendar years
    e0 : (C, T) e₀ (:func:`e0_matrix`), to orient s₁ and place s₁*

    Returns
    -------
    dict with ``g`` (s₁ → ds₁/dt, raw s₁ units per year), ``sign``,
    ``s1_star``, ``grid`` ``(gx, gy)``, the LOWESS before extension, and
    ``points`` ``(x, v)``, the pooled smoothed s₁ and annual changes it was
    fitted to.
    """
    sp_x, sp_v = [], []
    for c in _used(keep, settings):
        t_idx = np.flatnonzero(keep[c])
        yc, s1 = years[t_idx].astype(float), scores[c, t_idx, 0]
        frac = max(settings.smooth_frac_min, settings.smooth_years / len(yc))
        s1s = lowess(s1, yc, frac=frac, it=0, return_sorted=False)
        sp_x.append(s1s[:-1])
        sp_v.append(np.diff(s1s) / np.diff(yc))
    sp_x, sp_v = np.concatenate(sp_x), np.concatenate(sp_v)

    sign, s1_star = orientation(scores, keep, e0, settings)
    gx, gy = _lowess_curve(sp_x, sp_v, settings.speed_frac, settings.lowess_delta)
    g_raw = _extend(gx, gy, sign, s1_star, beyond="flat", tangent_share=settings.tangent_share)
    vmax = float(np.max(np.abs(gy)))
    lo, hi = (0.0, vmax) if sign > 0 else (-vmax, 0.0)

    def g(s):
        return np.clip(g_raw(s), lo, hi)
    return {"g": g, "sign": sign, "s1_star": s1_star, "grid": (gx, gy), "points": (sp_x, sp_v)}


def shape_law(scores: np.ndarray, keep: np.ndarray, years: np.ndarray, e0: np.ndarray,
              settings: FlowFieldSettings) -> dict:
    """Where the other scores sit at each level, and how fast deviations from it decay.

    Paper: eq:ff-shape, eq:ff-relax

    For k ≥ 2, f_k is a LOWESS of s_k on s₁ pooled over the population-years
    in ``keep``; beyond the frontier point of ``keep`` it runs along its
    tangent.  α_k is the pooled decay rate of the deviations
    d_k = s_k − f_k(s₁): with β(ℓ) the pooled autocorrelation of d_k at
    calendar lag ℓ = 1 … ``relax_max_lag``, log β(ℓ) ≈ ℓ · log α_k is fitted
    through the origin on the lags where β(ℓ) > ``relax_min_beta``.

    Parameters
    ----------
    As :func:`speed_function`.

    Returns
    -------
    dict with ``fk`` (list of p − 1 functions s₁ → s_k), ``alpha`` (p − 1,)
    per year, and the ``sign`` and ``s1_star`` of ``keep``.
    """
    used = _used(keep, settings)
    sc = np.concatenate([scores[c, np.flatnonzero(keep[c])] for c in used])
    sign, s1_star = orientation(scores, keep, e0, settings)
    p = scores.shape[2]
    fk = []
    for k in range(1, p):
        tx, ty = _lowess_curve(sc[:, 0], sc[:, k], settings.speed_frac, settings.lowess_delta)
        fk.append(_extend(tx, ty, sign, s1_star, beyond="tangent", tangent_share=settings.tangent_share))

    max_lag = settings.relax_max_lag
    lags = np.arange(1, max_lag + 1)
    alpha = []
    for k in range(1, p):
        num, den = np.zeros(max_lag + 1), np.zeros(max_lag + 1)
        for c in used:
            t_idx = np.flatnonzero(keep[c])
            d = scores[c, t_idx, k] - fk[k - 1](scores[c, t_idx, 0])
            # On calendar years, so that a lag skips the years without a schedule.
            ser = np.full(int(years[t_idx[-1]] - years[t_idx[0]]) + 1, np.nan)
            ser[(years[t_idx] - years[t_idx[0]]).astype(int)] = d
            for lag in lags:
                a, b = ser[:-lag], ser[lag:]
                ok = np.isfinite(a) & np.isfinite(b)
                num[lag] += np.sum(a[ok] * b[ok])
                den[lag] += np.sum(a[ok] ** 2)
        beta = num[1:] / den[1:]
        ok = beta > settings.relax_min_beta
        if not ok.any():
            raise ValueError(f"No positive autocorrelation of the deviations of score {k + 1}")
        slope = np.sum(lags[ok] * np.log(beta[ok])) / np.sum(lags[ok] ** 2)
        alpha.append(float(np.clip(np.exp(slope), 0.0, settings.relax_alpha_max)))
    return {"fk": fk, "alpha": np.array(alpha), "sign": sign, "s1_star": s1_star}


# ══════════════════════════════════════════════════════════════════════════════
# The frontier: the leading populations' pace
# ══════════════════════════════════════════════════════════════════════════════

def _topk_mean(rank_by: np.ndarray, values: np.ndarray, members: np.ndarray, k: int) -> float:
    """Mean of ``values`` over the k members with the largest ``rank_by``."""
    top = members[np.argsort(-rank_by[members], kind="stable")[:k]]
    return float(np.mean(values[top]))


def frontier_level(rank_by: np.ndarray, values: np.ndarray, avail: np.ndarray, k: int) -> dict:
    """The chain-linked level of a quantity over the k leading populations.

    Paper: eq:ff-frontier-level, eq:ff-fr1

    In each year the leaders are the k populations with the largest
    ``rank_by`` (e₀).  Populations enter and leave the data, and a raw top-k
    mean jumps when a high-e₀ population enters.  The level is therefore
    chain-linked, like a price index: each year's change is the change of the
    top-k mean among the populations available both in that year and in the
    previous year that has a level, so only movements of the populations
    themselves move it.

    Parameters
    ----------
    rank_by : (C, T) the quantity that ranks the populations (e₀)
    values : (C, T) the quantity whose level is measured (e₀, or oriented s₁)
    avail : (C, T) the population-years that may be used
    k : number of leading populations

    Returns
    -------
    dict with ``level`` (T,): the chain-linked level, starting at the raw
    top-k mean of the first year with k populations, NaN in years that cannot
    be linked; ``raw`` (T,): the raw top-k mean (NaN with fewer than k
    populations); ``bridged``: the number of links that skip one or more
    years; ``top``: by year, the frozenset of the raw top-k populations (or
    None).
    """
    T = avail.shape[1]
    level, raw = np.full(T, np.nan), np.full(T, np.nan)
    top, prev, bridged = [None] * T, None, 0
    for t in range(T):
        here = np.flatnonzero(avail[:, t])
        if here.size >= k:
            raw[t] = _topk_mean(rank_by[:, t], values[:, t], here, k)
            top[t] = frozenset(here[np.argsort(-rank_by[here, t], kind="stable")[:k]].tolist())
        if prev is None:
            if here.size >= k:
                level[t], prev = raw[t], t
            continue
        common = np.flatnonzero(avail[:, t] & avail[:, prev])
        if common.size < k:
            continue            # no link from prev into t; a later year may link to prev
        step = (_topk_mean(rank_by[:, t], values[:, t], common, k)
                - _topk_mean(rank_by[:, prev], values[:, prev], common, k))
        if t - prev > 1:
            bridged += 1
        level[t], prev = level[prev] + step, t
    return {"level": level, "raw": raw, "bridged": bridged, "top": top}


def frontier_slope(years: np.ndarray, level: np.ndarray, first: int, last: int, min_years: int
                   ) -> tuple[float, int]:
    """Least-squares slope (per year) of ``level`` over the years [first, last] where it exists.

    Paper: eq:ff-frontier-speed, eq:ff-fr1

    Parameters
    ----------
    years : (T,) calendar years; level : (T,) a frontier level
        (:func:`frontier_level`), NaN where it is not defined

    Returns
    -------
    ``(slope, years used)``: the slope in units of ``level`` per year, NaN
    with fewer than ``min_years`` years.
    """
    years = np.asarray(years, dtype=float)
    ok = np.isfinite(level) & (years >= first) & (years <= last)
    if ok.sum() < min_years:
        return float("nan"), int(ok.sum())
    return float(np.polyfit(years[ok], level[ok], 1)[0]), int(ok.sum())


def frontier_speed(e0: np.ndarray, s1o: np.ndarray, keep: np.ndarray, years: np.ndarray, origin: int,
                   settings: FlowFieldSettings) -> dict:
    """The leading populations' pace over the ``frontier_window`` years up to the origin.

    Paper: *eq:ff-frontier-speed, eq:ff-fr1

    Parameters
    ----------
    e0 : (C, T) e₀, which ranks the populations (:func:`e0_matrix`)
    s1o : (C, T) oriented s₁ (``sign · s₁``: higher means lower mortality)
    keep : (C, T) the population-years usable at the origin
    years : (T,) calendar years; origin : the forecast origin

    Returns
    -------
    dict with ``V_e`` (frontier e₀, years per year), ``V_s`` (frontier
    oriented s₁ per year), ``n`` (distinct populations among the leaders over
    the window), ``w`` = n / (n + ``frontier_n0``) and ``years_used``.
    """
    years = np.asarray(years)
    k = settings.frontier_k
    lev_e = frontier_level(e0, e0, keep, k)
    lev_s = frontier_level(e0, s1o, keep, k)
    first = origin - settings.frontier_window + 1
    V_e, n_years = frontier_slope(years, lev_e["level"], first, origin, settings.frontier_min_years)
    V_s, _ = frontier_slope(years, lev_s["level"], first, origin, settings.frontier_min_years)
    tops = [lev_e["top"][i] for i in np.flatnonzero((years >= first) & (years <= origin))
            if lev_e["top"][i] is not None]
    n = len(frozenset().union(*tops)) if tops else 0
    return {"V_e": V_e, "V_s": V_s, "n": n, "w": n / (n + settings.frontier_n0), "years_used": n_years}


def typical_e0(fk: Sequence[Callable], z_mean: np.ndarray, L: np.ndarray) -> Callable[[float], float]:
    """e(s): the e₀ of the typical schedule at the level s₁ = s.

    Paper: eq:ff-typical

    The typical schedule at level s is the one on the shape law: its scores
    are [s, f₂(s), …, f_p(s)], and its logit(q_x) is z_mean + L · scores
    (:func:`mdmx.forecast.scores_to_logit`).  Its e₀ is the average of the
    female and male e₀, as in :func:`e0_matrix`.

    Parameters
    ----------
    fk : the p − 1 functions s₁ → s_k of :func:`shape_law`
    z_mean : (S, A) logit(q_x) of the mean core; L : (S, A, ≥ p) the
        loadings' schedules (both from :func:`mdmx.forecast.origin_space`)

    Returns
    -------
    e : a function from s (raw s₁ units, a scalar) to e₀ in years.
    """
    def e(s: float) -> float:
        vec = np.array([s] + [f(s) for f in fk])[None]                      # (1, p)
        return float(lifetable.e0_from_logit(fc.scores_to_logit(vec, z_mean, L)[0]).mean())
    return e


def frontier_rule(g: Callable, sign: float, s1_star: float, frontier: Mapping, rule: str, *,
                  typical: Callable[[float], float] | None = None, step: float | None = None
                  ) -> tuple[Callable, dict]:
    """The speed with its part beyond s₁* set by a frontier rule.

    Paper: eq:ff-fr1, eq:ff-fr2

    ``FR0`` keeps ``g`` (held at g(s₁*) beyond s₁*).  ``FR1`` sets the
    oriented speed beyond s₁* to

        v₁ = max(0, w · V_s + (1 − w) · sign · g(s₁*)),

    the leaders' pace in s₁ shrunk towards g(s₁*) (:func:`frontier_speed`).
    ``FR2`` moves s₁ beyond s₁* so that the e₀ of the typical schedule rises
    by

        v_e = max(0, w · V_e + (1 − w) · g(s₁*) · e′(s₁*))

    years a year: ds₁/dt = v_e / e′(s₁), with e(s) the typical schedule's e₀
    (:func:`typical_e0`) and e′ its central difference
    (e(s + step) − e(s − step)) / (2 · step).  g(s₁*) · e′(s₁*) is the flat
    rule's pace in e₀, so FR2 is the leaders' pace in e₀ shrunk towards it.
    Where the typical schedule stops gaining e₀ (sign · e′(s₁) ≤ 0), FR2 holds
    s₁ (speed 0) and counts the evaluation in ``info["held"]``.

    v₁ and v_e are never negative, so neither rule reverses the improvement.
    Without a frontier pace (fewer than ``frontier_min_years`` linked years)
    ``FR1`` and ``FR2`` fall back to ``FR0``.  Below s₁* the speed is always
    ``g``.

    Parameters
    ----------
    g : the speed (:func:`speed_function`); sign, s1_star : its orientation
        and frontier point; frontier : :func:`frontier_speed`; rule : one of
        :data:`FRONTIER_RULES`
    typical : FR2 only, e(s) (:func:`typical_e0`)
    step : FR2 only, the step of e′ in raw s₁ units (positive)

    Returns
    -------
    ``(speed, info)``: the speed function (raw s₁ units per year; scalar or
    array s₁) and a dict with ``rule``, ``fallback``, ``v1`` (NaN unless FR1
    applies), ``v_e`` (NaN unless FR2 applies; years of e₀ per year) and
    ``held``, the number of FR2's evaluations beyond s₁* that held s₁ (it
    grows as the speed is used).
    """
    if rule not in FRONTIER_RULES:
        raise ValueError(f"rule must be one of {FRONTIER_RULES}, not {rule!r}")
    info = {"rule": rule, "fallback": False, "v1": float("nan"), "v_e": float("nan"), "held": 0}
    if rule == "FR0":
        return g, info
    if rule == "FR2" and (typical is None or step is None or not step > 0):
        raise ValueError("Rule FR2 needs the typical schedule's e₀ (typical) and a positive step for its derivative")
    if not (np.isfinite(frontier["V_e"]) and np.isfinite(frontier["V_s"])):
        info["fallback"] = True
        return g, info
    w, g_star = frontier["w"], float(g(s1_star))
    if rule == "FR1":
        v1 = max(0.0, w * frontier["V_s"] + (1.0 - w) * sign * g_star)
        info["v1"] = v1

        def speed(s):
            s = np.asarray(s, dtype=float)
            return np.where(sign * (s - s1_star) > 0, sign * v1, g(s))
        return speed, info

    def slope(s: float) -> float:
        return (typical(s + step) - typical(s - step)) / (2 * step)

    v_e = max(0.0, w * frontier["V_e"] + (1.0 - w) * g_star * slope(s1_star))
    info["v_e"] = v_e

    # e(s) takes one s₁ at a time, so arrays are evaluated element by element.
    def speed_fr2(s):
        s = np.asarray(s, dtype=float)
        if s.ndim:
            return np.array([speed_fr2(x) for x in s])
        beyond = sign * (s - s1_star) > 0
        if not beyond:
            return g(s)
        d = slope(float(s))
        if sign * d <= 0:
            info["held"] += 1
            return np.asarray(0.0)
        return np.asarray(v_e / d)
    return speed_fr2, info


# ══════════════════════════════════════════════════════════════════════════════
# The flow and the forecasts
# ══════════════════════════════════════════════════════════════════════════════

def build_flow(scores: np.ndarray, keep: np.ndarray, years: np.ndarray, e0: np.ndarray, origin: int,
               settings: FlowFieldSettings, *, z_mean: np.ndarray | None = None, L: np.ndarray | None = None
               ) -> dict:
    """The flow at one origin: speed, frontier rule and shape law, each on its window.

    Paper: sec:fc-flowfield

    The ablation switches act here and in :func:`integrate`: ``relax`` sets
    the α_k the path uses (1 for ``persist``, 0 for ``vanish``), and
    ``shape_law`` is carried in the flow for :func:`integrate`.  f_k are
    always estimated, so FR2's typical schedule is the same with every
    switch.

    Parameters
    ----------
    scores : (C, T, p) PCA scores (:func:`mdmx.forecast.origin_space`)
    keep : (C, T) the population-years observed up to the origin (the
        space's ``keep``); the speed and the shape law use the part of it in
        their windows, the frontier and FR2's step all of it
    years : (T,) calendar years; e0 : (C, T) e₀ (:func:`e0_matrix`)
    origin : the forecast origin
    z_mean : (S, A), L : (S, A, ≥ p) the space's linear map from scores to
        logit(q_x); needed for rule ``FR2`` only (:func:`typical_e0`)

    Returns
    -------
    dict with ``g`` (the speed with the frontier rule applied), ``fk``,
    ``alpha`` (the rates the path uses), ``shape_law``, ``sign``,
    ``s1_star`` (the speed's frontier point), ``s1_star_shape`` (the shape
    law's), ``g_star`` = g(s₁*) before the rule, and ``frontier``:
    :func:`frontier_speed` and :func:`frontier_rule`'s info.
    """
    years = np.asarray(years)
    speed = speed_function(scores, _window(keep, years, origin, settings.speed_window), years, e0, settings)
    shape = shape_law(scores, _window(keep, years, origin, settings.shape_window), years, e0, settings)
    # A window that reverses the direction of improvement would make the
    # parts, and the frontier, describe different directions of s₁.
    sign = orientation(scores, keep, e0, settings)[0]
    if speed["sign"] != sign or shape["sign"] != sign:
        raise ValueError(f"{origin}: s₁ is oriented differently in a window than in every year up to the origin")
    front = frontier_speed(e0, sign * scores[..., 0], keep, years, origin, settings)
    typical = step = None
    if settings.frontier_rule == "FR2":
        if z_mean is None or L is None:
            raise ValueError(f"{origin}: rule FR2 needs the space's z_mean and L (the typical schedule's e₀)")
        typical = typical_e0(shape["fk"], z_mean, L)
        step = settings.frontier_e0_step * float(np.ptp((sign * scores[..., 0])[keep]))
    g, rule = frontier_rule(speed["g"], sign, speed["s1_star"], front, settings.frontier_rule,
                            typical=typical, step=step)
    # The rule's own dict, so that FR2's count of held steps stays current as the flow is used.
    rule.update(front)
    alpha = {"estimated": shape["alpha"], "persist": np.ones_like(shape["alpha"]),
             "vanish": np.zeros_like(shape["alpha"])}[settings.relax]
    return {"g": g, "fk": shape["fk"], "alpha": alpha, "shape_law": settings.shape_law, "sign": sign,
            "s1_star": speed["s1_star"], "s1_star_shape": shape["s1_star"],
            "g_star": float(speed["g"](speed["s1_star"])), "frontier": rule}


def flow_summary(flow: Mapping) -> dict:
    """The fitted flow's scalars, for tables of the flow by origin (oriented where it matters).

    ``s1_star`` and ``s1_star_shape`` are oriented (``sign · s₁*``), and so
    are ``g_star`` and ``v1`` (improvement per year in s₁ units).  ``v_e``
    (FR2) is in years of e₀ per year, and ``held`` counts FR2's held steps
    so far.
    """
    sign, front = flow["sign"], flow["frontier"]
    return {"sign": sign, "s1_star": sign * flow["s1_star"], "s1_star_shape": sign * flow["s1_star_shape"],
            "g_star": sign * flow["g_star"], "v1": front["v1"], "v_e": front["v_e"], "held": front["held"],
            "V_s": front["V_s"], "V_e": front["V_e"], "n": front["n"], "w": front["w"],
            "frontier_years": front["years_used"], "rule": front["rule"], "fallback": front["fallback"],
            **{f"alpha{k + 2}": float(a) for k, a in enumerate(flow["alpha"])}}


def integrate(s0: np.ndarray, flow: Mapping, n: int) -> np.ndarray:
    """The scores for n years from s0: Euler steps of the speed, deviations decaying.

    Paper: eq:ff-path

    s₁(h) = s₁(h − 1) + g(s₁(h − 1)) and, for k ≥ 2,
    s_k(h) = f_k(s₁(h)) + α_k^h · (s_k(0) − f_k(s₁(0))).

    Without the shape law (the flow's ``shape_law`` False, an ablation),
    s_k(h) = s_k(0) for k ≥ 2 while s₁ moves as above.  With α_k = 1 the
    jump-off deviation persists; with α_k = 0 it vanishes from h = 1.

    Parameters
    ----------
    s0 : (p,) the scores at the jump-off year; flow : :func:`build_flow`
        (``g``, ``fk``, ``alpha``, and ``shape_law``, True if absent);
    n : number of years

    Returns
    -------
    (n + 1, p) scores; row 0 is ``s0``.
    """
    p = len(s0)
    s = np.zeros((n + 1, p))
    s[0] = s0
    follow_shape = flow.get("shape_law", True)
    dev0 = np.array([s0[k] - flow["fk"][k - 1](s0[0]) for k in range(1, p)])
    for h in range(1, n + 1):
        s[h, 0] = s[h - 1, 0] + flow["g"](s[h - 1, 0])
        if not follow_shape:
            s[h, 1:] = s0[1:]
            continue
        for k in range(1, p):
            s[h, k] = flow["fk"][k - 1](s[h, 0]) + flow["alpha"][k - 1] ** h * dev0[k - 1]
    return s


def forecast_origin(M: np.ndarray, space: Mapping, pops: Sequence[str], years: Sequence[int], jobs: pd.DataFrame,
                    origin: int, horizon: int, e0: np.ndarray, settings: FlowFieldSettings, *,
                    flow: Mapping | None = None, schedules: bool = False) -> dict:
    """Flow-field forecasts of every job of one origin.

    Paper: eq:ff-anchor, eq:ff-path

    Each job starts from the population's scores in its last training year
    t₀.  The forecast schedules are ``z(h) = z_mean + L s(h) + K r`` with
    ``r = M_{c,t₀} − z_mean − L s(0)``: the jump-off residual, the part of
    the observed schedule that the ``n_pc`` scores do not represent,
    smoothed over age by ``K`` (:func:`mdmx.forecast.smooth_residual`, λ =
    ``settings.anchor_smoothing``), is kept at every horizon (the persistent
    anchor).  With ``settings.anchor`` False (an
    ablation) the residual is dropped and ``z(h) = z_mean + L s(h)``, the
    projection of the forecast scores.  The flow pools every population's
    history up to the origin, so a job's ``first_year`` is not used.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor
    space : :func:`mdmx.forecast.origin_space` at this origin
    pops, years : labels of the population and year axes
    jobs : rows of :func:`mdmx.forecast.cv_jobs` for this origin
    origin, horizon : forecasts run to ``origin + horizon``
    e0 : (C, T) e₀ (:func:`e0_matrix`)
    flow : a flow already built (:func:`build_flow`); None builds it from
        ``space`` with ``settings``.  A flow carries its own frontier rule,
        α_k and shape-law switch; ``settings.anchor`` decides the anchor.
    schedules : also return the forecast schedules

    Returns
    -------
    dict with ``points`` (``pop, sex, year, e0_fc``; sex ``female``,
    ``male`` and ``both``, the average of the two; years after the origin)
    and ``flow`` (:func:`flow_summary`); with ``schedules``, also
    ``schedule_index`` (``pop, year``) and ``schedules`` (N, 2, A) logit(q_x)
    for the same years.
    """
    pops, years = np.asarray(pops), np.asarray(years).astype(int)
    if flow is None:
        flow = build_flow(space["scores"], space["keep"], years, e0, origin, settings, z_mean=space["z_mean"],
                          L=space["L"])
    frames, sched_idx, sched = [], [], []
    for job in jobs.itertuples():
        c = int(np.flatnonzero(pops == job.pop)[0])
        t0 = int(np.flatnonzero(years == job.last_train_year)[0])
        s0 = space["scores"][c, t0]
        if not np.all(np.isfinite(s0)):
            raise ValueError(f"{job.pop}: no scores at its last training year {job.last_train_year}")
        n = int(origin + horizon - job.last_train_year)
        z = fc.scores_to_logit(integrate(s0, flow, n), space["z_mean"], space["L"])   # (n + 1, 2, A)
        if settings.anchor:
            z = z + fc.smooth_residual(M[:, :, c, t0] - z[0], settings.anchor_smoothing)
        e = lifetable.e0_from_logit(z)                                                 # (n + 1, 2)
        yrs = job.last_train_year + np.arange(n + 1)
        after = yrs > origin
        for k, sex in enumerate(("female", "male")):
            frames.append(pd.DataFrame({"pop": job.pop, "sex": sex, "year": yrs[after], "e0_fc": e[after, k]}))
        frames.append(pd.DataFrame({"pop": job.pop, "sex": "both", "year": yrs[after],
                                    "e0_fc": e[after].mean(axis=1)}))
        if schedules:
            sched_idx.append(pd.DataFrame({"pop": job.pop, "year": yrs[after]}))
            sched.append(z[after])
    out = {"points": pd.concat(frames, ignore_index=True), "flow": flow_summary(flow)}
    if schedules:
        out["schedule_index"] = pd.concat(sched_idx, ignore_index=True)
        out["schedules"] = np.concatenate(sched, axis=0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Empirical predictive intervals
# ══════════════════════════════════════════════════════════════════════════════

def within_era_calibration(eras: Mapping[str, Sequence[int]], held_out: Sequence[int] = (),
                           held_out_by: str | None = None) -> dict[int, list[int]]:
    """Leave-one-origin-out calibration sets within each era.

    Paper: eq:ff-intervals, eq:fc-kappa

    The errors that calibrate the intervals of an origin are those of the
    other origins of its era.  Held-out origins are calibrated on every
    origin of the era ``held_out_by``.

    Parameters
    ----------
    eras : era name → its origins (disjoint)
    held_out : origins outside the eras; held_out_by : the era that
        calibrates them (required when ``held_out`` is not empty)

    Returns
    -------
    origin → the origins whose errors calibrate it.
    """
    seen: dict[int, str] = {}
    for era, origins in eras.items():
        for o in origins:
            if int(o) in seen:
                raise ValueError(f"Origin {o} is in two eras ({seen[int(o)]}, {era})")
            seen[int(o)] = era
    cal = {o: [int(x) for x in eras[era] if int(x) != o] for o, era in seen.items()}
    if len(held_out):
        if held_out_by not in eras:
            raise ValueError(f"held_out_by must name an era, got {held_out_by!r}")
        for o in held_out:
            if int(o) in seen:
                raise ValueError(f"Held-out origin {o} is also in era {seen[int(o)]}")
            cal[int(o)] = [int(x) for x in eras[held_out_by]]
    return cal


def empirical_intervals(points: pd.DataFrame, calibration: Mapping[int, Sequence[int]],
                        pi_levels: Sequence[int], min_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictive quantiles of e₀ from one method's errors at other origins.

    Paper: *eq:ff-intervals

    For a forecast at origin o, horizon h = year − o and sex series s, let
    E be the errors (``e0_fc − e0_obs``) at horizon h and series s of the
    origins ``calibration[o]``.  The predictive quantile at level p is
    ``e0_fc − Q_{1−p}(E)``, with Q the empirical quantile (NumPy's default,
    linear interpolation).  The median is ``e0_fc − median(E)``.  An origin
    never calibrates itself.  A cell with fewer than ``min_points`` errors
    gets no quantiles and is reported.

    Parameters
    ----------
    points : one method's forecasts, one row per ``pop, origin, sex,
        year``, with ``e0_fc`` and ``e0_obs`` (NaN where there is no
        observation; those rows get quantiles but give no error)
    calibration : origin → origins whose errors calibrate it
        (:func:`within_era_calibration`); every origin of ``points`` needs one
    pi_levels : nominal coverages (per cent) of the central intervals
    min_points : the fewest errors a cell needs

    Returns
    -------
    ``(out, sparse)``: ``out`` is ``points`` with ``h``, the quantile columns
    of :func:`mdmx.compare.quantile_columns` (the median included) and
    ``lower{L}``, ``upper{L}`` for each level; ``sparse`` lists the cells
    without quantiles (``origin, sex, h, n_errors``).
    """
    if points.duplicated(["pop", "origin", "sex", "year"]).any():
        raise ValueError("points has more than one row per pop, origin, sex and year: pass one method's forecasts")
    levels = compare.quantile_levels(pi_levels)
    qcols = compare.quantile_columns(levels)
    out = points.copy()
    out["h"] = (out["year"] - out["origin"]).astype(int)
    missing = sorted(set(out["origin"].astype(int)) - set(calibration))
    if missing:
        raise ValueError(f"No calibration set for the origins {missing}")
    scored = out[out["e0_obs"].notna()]
    errors = {key: (g["e0_fc"] - g["e0_obs"]).to_numpy()
              for key, g in scored.groupby(["origin", "sex", "h"])}
    q = np.full((len(out), len(levels)), np.nan)
    sparse = []
    for (origin, sex, h), rows in out.groupby(["origin", "sex", "h"]).indices.items():
        cal = [int(o) for o in calibration[int(origin)]]
        if int(origin) in cal:
            raise ValueError(f"Origin {origin} is in its own calibration set")
        # An empty calibration set (the first origin, with past errors only) gives no errors: the cell is reported.
        E = np.concatenate([np.empty(0), *(errors.get((o, sex, h), np.empty(0)) for o in cal)])
        if E.size < min_points:
            sparse.append({"origin": int(origin), "sex": sex, "h": int(h), "n_errors": int(E.size)})
            continue
        q[rows] = out["e0_fc"].to_numpy()[rows, None] - np.quantile(E, 1.0 - levels)[None, :]
    out[qcols] = q
    names = dict(zip(np.round(levels, 6), qcols, strict=True))
    for lev in pi_levels:
        lo = round((100 - lev) / 200, 6)
        out[f"lower{lev}"], out[f"upper{lev}"] = out[names[lo]], out[names[round(1 - lo, 6)]]
    return out, pd.DataFrame(sparse, columns=["origin", "sex", "h", "n_errors"])


#: The code a flow-field forecast runs (its cache key), as :data:`mdmx.forecast.MDMX_CODE` for MDMx.
FLOW_CODE = (forecast_origin, build_flow, flow_summary, integrate, orientation, speed_function, shape_law,
             frontier_level, frontier_slope, frontier_speed, frontier_rule, typical_e0, e0_matrix, _window, _used,
             _lowess_curve, _extend, _topk_mean, fc.scores_to_logit, fc.smooth_residual, fc.anchor_smoother,
             lifetable.e0_from_logit,
             lifetable.survivorship, lifetable.expit, *fc.SPACE_CODE)
