"""Forecasting complete sex–age mortality schedules (paper §11).

The idea
--------
Every observed country-year's mortality schedule is represented exactly (up
to the age rank) by its *effective core* ``G_{c,t} = Sᵀ M_{c,t} A``, an
``r₁ × r₂`` matrix of coordinates in the shared sex and age bases (the
effective core of §6).  Forecasting a population's schedule therefore reduces to
forecasting a short multivariate time series of cores, and every forecast
core maps back to a female *and* a male schedule through the same bases, so
the two sexes are linked by construction.

The forecaster has four parts, each a function below:

1. **The space** (:func:`origin_space`).  For a forecast origin ``o`` the sex
   and age bases, the effective cores and a principal-component analysis
   (PCA) of the vectorised cores are estimated from observed country-years
   with year ``≤ o`` only, so nothing a forecast is scored against has been
   seen (§11.1, the PCA of the vectorised cores).  A forecast is made for the first
   ``n_pc`` PCA scores of a population.
2. **The drift target** (:func:`country_drifts`, :func:`drift_target`).  Each
   population's recent trend in the scores (its *drift*) is blended with the
   average drift of all HMD populations (the two-level drift target of §11.3).
3. **The Kalman filter** (:func:`llt_filter`, :func:`fit_llt`,
   :func:`llt_forecast`).  The scores follow a damped local linear trend whose
   drift reverts towards the target (the damped local linear trend of §11.2); the
   variances and the damping factor ρ are estimated by maximum likelihood,
   on the scale of the standard deviations and of ρ itself, so that the many
   maxima that lie on a bound (ρ at its lower bound, a variance at its
   floor) are reached, and from several starts, because the likelihood can
   have more than one local maximum.
   The filter runs on *calendar* years, and skips the update step in years
   without an observed (non-exceptional) schedule.
4. **Back to schedules and e₀** (:func:`scores_to_logit`,
   :func:`e0_and_gradient`, :func:`mdmx_origin`).  Forecast scores map
   linearly to logit(q_x) (the linear map of §11.1), and from there to e₀ for
   each sex; the forecast standard deviation of e₀ follows from the delta
   method of §11.4 with an exact gradient of e₀.

The evaluation tools are generic, so that any forecasting method can be
compared on the same footing: :func:`observed_e0` (the truth),
:func:`cv_jobs` (the rolling-origin design), :func:`run_benchmark` (the
harness), :func:`summarise`, :func:`loo_calibrate` and
:func:`schedule_errors`.  :func:`comparator_data` and
:func:`comparator_schedules` prepare the inputs for, and read the outputs of,
the R ``demography`` comparators run through :mod:`mdmx.rbridge`.

Adding a forecasting method to the benchmark
--------------------------------------------
A method is one function ``method(origin, jobs_at_origin)``; see
:func:`run_benchmark` for the full contract.  In short:

* **Input.**  ``origin`` (int, calendar year) and the rows of
  :func:`cv_jobs` for that origin — columns ``pop, origin, horizon,
  n_train, n_test, first_year, last_train_year``.  The method may use the
  population's data from ``first_year`` to ``last_train_year`` only
  (``first_year`` is later than the first observed year when the jobs come
  from :func:`restrict_fit_window`), and must forecast at least the years
  ``origin + 1 … origin + horizon``.
* **Output.**  A DataFrame with one row per population, sex and forecast
  year — ``pop, sex, year, e0_fc`` with ``sex`` in ``female``, ``male``
  and ``both`` (``both`` = the average of the female and male e₀ of the
  same forecast) — and, if the method has a predictive distribution,
  ``lower80, upper80`` (its 10th and 90th percentiles), ``lower95,
  upper95`` (2.5th and 97.5th) and optionally ``sd``.  Or a dict whose
  ``"points"`` entry is that DataFrame; anything else in the dict (fitted
  parameters, schedules as ``schedule_index`` + ``schedules`` (N, 2, A)
  logit q_x, R ``failures``/``warnings``) is cached with it.
* **Caching.**  ``run_benchmark(..., cache={name: {"inputs": [...],
  "params": {...}, "code": [...], "desc": "..."}}, cache_prefix=prefix)``
  stores each origin's result as artefact ``<prefix>_<name>_<origin>``
  (prefix ``forecast_bench`` for the full-history policy,
  ``forecast_win50`` for the last-50-years policy of chapter 11).
  ``name`` must contain only letters and digits (e.g. ``"BL"``).

Conventions
-----------
* Index 0 of a sex axis is female and 1 is male; ``"both"`` denotes the
  average of the female and male e₀ (not the e₀ of a combined-sex table).
* Arrays of scores are ``(T, p)``: calendar years by PCA components; rows
  of a year without an observed schedule are NaN.
* Errors are ``forecast − observed``: a positive bias means e₀ was
  over-forecast.
"""

from __future__ import annotations

import functools
import itertools
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from . import artifacts, lifetable, rbridge, smoothing
from . import tensor as tz

__all__ = [
    "SEXES", "MDMxSettings", "observed_e0", "cv_jobs", "restrict_fit_window", "origin_space", "country_drifts",
    "drift_target", "llt_filter", "llt_forecast", "fit_llt", "fit_llt_batch", "scores_to_logit",
    "e0_and_gradient", "origin_inputs", "anchor_weights", "anchor_smoother", "smooth_residual", "mdmx_origin", "SPACE_CODE", "MDMX_CODE", "run_benchmark", "summarise", "loo_calibrate",
    "coverage", "schedule_errors", "comparator_data", "comparator_schedules",
    "smooth_comparator_rates", "comparator_forecasts", "comparator_points",
]

#: The three e₀ series scored for every method: female, male, and their average.
SEXES = ("female", "male", "both")

_LOG_2PI = float(np.log(2.0 * np.pi))


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MDMxSettings:
    """Every tunable value of the MDMx forecaster (read from ``[forecast]``).

    Attributes
    ----------
    horizon : number of years forecast beyond the origin
    n_pc : number of PCA scores forecast (``N_PC``)
    weight_hmd, weight_country : weights of the HMD-wide and the country's
        own drift in the drift target (:func:`drift_target`); they sum to one
    drift_window : length of the drift window in calendar years (``W``)
    drift_min_years : observed years a population needs inside the window
        for its drift to be defined
    rho_bounds : ``(ρ_min, ρ_max)``, the range of the damping factor
    log_var_bounds : box bounds on the log of every variance parameter
    level0_years : the filter's initial level is the mean of the first
        ``level0_years`` observed scores
    p0_level_var : initial variance of each level component
    mle_maxiter : L-BFGS-B iteration limit of each start
    mle_ftol : L-BFGS-B relative-reduction tolerance (set near machine
        precision, so that a start stops on the gradient, not on a flat slope)
    mle_gtol : L-BFGS-B projected-gradient tolerance
    mle_pg_flag : a fit whose largest projected-gradient component exceeds
        this at the optimum is flagged as not converged
    mle_rho_starts : values of ρ at which the optimiser is also started from
        the moment-based starting values (besides the middle of ρ's range)
    mle_grid_rho : grid of ρ for the profile log-likelihood that provides
        further starts (:func:`_profile_starts`)
    mle_grid_sd_log10 : ``(a, b)``: the grid of each standard deviation is its
        floor and ``mle_grid_sd_points`` log-spaced multiples 10^a … 10^b of the
        data's own scale
    mle_grid_sd_points : see ``mle_grid_sd_log10``
    mle_grid_top : number of profile starts (the best grid values of ρ)
    mle_profile_maxiter, mle_profile_maxfun : iteration and evaluation limits
        of the stacked fixed-ρ fit of :func:`_profile_starts`, which has
        ``len(mle_grid_rho)`` times the parameters of one fit
    fd_step : step of the central finite differences used for the gradient
        of the log-likelihood (on the scale of the standard deviations and ρ)
    tau : variance threshold of the age basis (only recorded; the age rank
        is fixed by ``age_rank``)
    age_rank : the age rank r₂, held at its full-data value at every origin
    smooth : keyword arguments of :func:`mdmx.smoothing.smooth_age_basis`
    pi_levels : nominal coverages (per cent) of the prediction intervals
    anchor_smoothing : λ of the Whittaker smoother of the anchor's jump-off
        residual (:func:`smooth_residual`); 0 keeps the residual as observed
    """

    horizon: int
    n_pc: int
    weight_hmd: float
    weight_country: float
    drift_window: int
    drift_min_years: int
    rho_bounds: tuple[float, float]
    log_var_bounds: tuple[float, float]
    level0_years: int
    p0_level_var: float
    mle_maxiter: int
    mle_ftol: float
    mle_gtol: float
    mle_pg_flag: float
    mle_rho_starts: tuple[float, ...]
    mle_grid_rho: tuple[float, ...]
    mle_grid_sd_log10: tuple[float, float]
    mle_grid_sd_points: int
    mle_grid_top: int
    mle_profile_maxiter: int
    mle_profile_maxfun: int
    fd_step: float
    tau: float
    age_rank: int
    smooth: dict = field(default_factory=dict)
    pi_levels: tuple[int, ...] = (80, 95)
    anchor_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if not np.isclose(self.weight_hmd + self.weight_country, 1.0):
            raise ValueError("The drift weights weight_hmd and weight_country must sum to one.")
        if not 0 < self.rho_bounds[0] < self.rho_bounds[1] <= 1:
            raise ValueError(f"Invalid damping bounds {self.rho_bounds}.")
        lo, hi = self.rho_bounds
        for name in ("mle_rho_starts", "mle_grid_rho"):
            values = getattr(self, name)
            if not all(lo <= v <= hi for v in values):
                raise ValueError(f"Every value of {name} must lie in rho_bounds {self.rho_bounds}: {values}.")
        if not 1 <= self.mle_grid_top <= len(self.mle_grid_rho):
            raise ValueError(f"mle_grid_top must be between 1 and len(mle_grid_rho), got {self.mle_grid_top}.")
        if self.mle_grid_sd_points < 1:
            raise ValueError(f"mle_grid_sd_points must be at least 1, got {self.mle_grid_sd_points}.")
        if len(self.mle_grid_sd_log10) != 2 or self.mle_grid_sd_log10[0] > self.mle_grid_sd_log10[1]:
            raise ValueError(f"mle_grid_sd_log10 must be (a, b) with a <= b, got {self.mle_grid_sd_log10}.")
        if min(self.mle_maxiter, self.mle_profile_maxiter, self.mle_profile_maxfun) < 1:
            raise ValueError("The iteration and evaluation limits of the optimiser must be at least 1.")
        if not (np.isfinite(self.anchor_smoothing) and self.anchor_smoothing >= 0):
            raise ValueError(f"anchor_smoothing must be finite and not negative, got {self.anchor_smoothing}.")

    @classmethod
    def from_params(cls, fc: Mapping, tucker: Mapping, age_rank: int, **overrides) -> MDMxSettings:
        """Build the settings from the ``[forecast]`` and ``[tucker]`` parameter sections."""
        values = dict(
            horizon=int(fc["horizon"]), n_pc=int(fc["n_pc"]),
            weight_hmd=float(fc["drift_weight_hmd"]), weight_country=float(fc["drift_weight_country"]),
            drift_window=int(fc["drift_window"]), drift_min_years=int(fc["drift_min_years"]),
            rho_bounds=tuple(fc["rho_bounds"]), log_var_bounds=tuple(fc["log_var_bounds"]),
            level0_years=int(fc["level0_years"]), p0_level_var=float(fc["p0_level_var"]),
            mle_maxiter=int(fc["mle_maxiter"]), mle_ftol=float(fc["mle_ftol"]),
            mle_gtol=float(fc["mle_gtol"]), mle_pg_flag=float(fc["mle_pg_flag"]),
            mle_rho_starts=tuple(float(v) for v in fc["mle_rho_starts"]),
            mle_grid_rho=tuple(float(v) for v in fc["mle_grid_rho"]),
            mle_grid_sd_log10=tuple(float(v) for v in fc["mle_grid_sd_log10"]),
            mle_grid_sd_points=int(fc["mle_grid_sd_points"]), mle_grid_top=int(fc["mle_grid_top"]),
            mle_profile_maxiter=int(fc["mle_profile_maxiter"]),
            mle_profile_maxfun=int(fc["mle_profile_maxfun"]),
            fd_step=float(fc["fd_step"]), tau=float(tucker["tau"]), age_rank=int(age_rank),
            smooth={"components": tucker["smooth_components"], "rest_sigma": tucker["smooth_rest_sigma"],
                    "preserve": tucker["smooth_preserve"], "orth_tol": tucker["smooth_orth_tol"]},
            pi_levels=tuple(int(v) for v in fc["pi_levels"]),
            anchor_smoothing=float(fc["anchor_smoothing"]),
        )
        values.update(overrides)
        return cls(**values)

    def as_dict(self) -> dict:
        """Plain dict of the settings (what the artefact store hashes)."""
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# The truth and the cross-validation design
# ══════════════════════════════════════════════════════════════════════════════

def observed_e0(M: np.ndarray, observed: np.ndarray, pops: Sequence[str],
                years: Sequence[int]) -> pd.DataFrame:
    """Life expectancy at birth of every observed country-year, from the data.

    This is the yardstick every forecasting method is scored against.  It is
    computed from the *observed* schedule in the tensor (the curated, pooled
    HMD life table), never from a model reconstruction.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor (NaN where there are no data)
    observed : (C, T) boolean mask of observed, non-exceptional country-years
    pops, years : labels of the country and year axes

    Returns
    -------
    DataFrame with one row per observed country-year and sex — columns
    ``pop, sex, year, e0_obs``; ``sex`` is ``female``, ``male`` or ``both``
    (the average of the female and male e₀).
    """
    c_idx, t_idx = np.nonzero(observed)
    # M[:, :, c, t] for every observed cell → (S, A, N); move age last for the life table.
    e0 = lifetable.e0_from_logit(np.moveaxis(M[:, :, c_idx, t_idx], 1, -1))   # (S, N)
    values = {"female": e0[0], "male": e0[1], "both": 0.5 * (e0[0] + e0[1])}
    frames = [pd.DataFrame({"pop": np.asarray(pops)[c_idx], "sex": sex,
                            "year": np.asarray(years)[t_idx].astype(int), "e0_obs": v})
              for sex, v in values.items()]
    return pd.concat(frames, ignore_index=True)


def cv_jobs(observed: np.ndarray, pops: Sequence[str], years: Sequence[int], origins: Sequence[int],
            horizon: int, min_train: int, min_test: int) -> pd.DataFrame:
    """The rolling-origin cross-validation design: one job per population and origin.

    Paper: sec:evaluation:design

    A population is forecast from origin ``o`` if it has at least
    ``min_train`` observed (non-exceptional) years up to and including ``o``
    and at least ``min_test`` observed years in ``o+1 … o+horizon`` to score
    the forecast against.

    Returns
    -------
    DataFrame with columns ``pop, origin, horizon, n_train, n_test,
    first_year`` (first observed year) and ``last_train_year`` (last observed
    year ≤ origin; the forecast of a method that cannot skip missing years
    starts there).
    """
    years = np.asarray(years)
    rows = []
    for origin in origins:
        train = observed & (years <= origin)[None, :]
        test = observed & (years > origin)[None, :] & (years <= origin + horizon)[None, :]
        for c in np.flatnonzero((train.sum(axis=1) >= min_train) & (test.sum(axis=1) >= min_test)):
            rows.append({"pop": str(pops[c]), "origin": int(origin), "horizon": int(horizon),
                         "n_train": int(train[c].sum()), "n_test": int(test[c].sum()),
                         "first_year": int(years[train[c]].min()),
                         "last_train_year": int(years[train[c]].max())})
    return pd.DataFrame(rows)


def restrict_fit_window(jobs: pd.DataFrame, years_back: int) -> pd.DataFrame:
    """The same jobs, but every method fitted only to the last ``years_back`` years.

    The training series of each job then starts at
    ``max(first_year, last_train_year − years_back + 1)`` (calendar years).
    The set of jobs and the points scored are unchanged, so results with and
    without the restriction are directly comparable.
    """
    start = np.maximum(jobs["first_year"], jobs["last_train_year"] - years_back + 1)
    return jobs.assign(first_year=start.astype(int))


# ══════════════════════════════════════════════════════════════════════════════
# 1. The space: bases, effective cores and PCA estimated from data ≤ origin
# ══════════════════════════════════════════════════════════════════════════════

def origin_space(M: np.ndarray, observed: np.ndarray, years: Sequence[int], origin: int, *,
                 tau: float, age_rank: int, smooth: Mapping, n_pc: int) -> dict:
    """Sex/age bases, effective cores and their PCA from observed data up to ``origin``.

    Paper: eq:pca_scores, sec:forecasting:pca

    This repeats, on the country-years observed by the origin, exactly the
    steps of the decomposition chapter (04): the sex and age bases from the
    observed schedules (:func:`mdmx.tensor.sex_age_bases`), smoothing of the age basis (:func:`mdmx.smoothing.smooth_age_basis`),
    the effective cores ``G_{c,t} = Sᵀ M_{c,t} A`` (:func:`mdmx.tensor.effective_cores`), and then and
    then the PCA of the vectorised cores..  The age rank is held at
    ``age_rank`` (the full-data rank) at every origin so that forecasts from
    different origins have the same structure.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor (NaN where there are no data)
    observed : (C, T) mask of observed, non-exceptional country-years
    years : (T,) calendar years of the year axis
    origin : last calendar year whose data may be used
    tau : variance threshold passed to :func:`mdmx.tensor.sex_age_bases`
        (it has no effect because the rank bounds are both ``age_rank``)
    age_rank : the age rank r₂
    smooth : ``components, rest_sigma, preserve, orth_tol`` of
        :func:`mdmx.smoothing.smooth_age_basis`
    n_pc : number of principal components kept

    Returns
    -------
    dict with
      ``U_sex`` (S, r₁), ``U_age`` (A, r₂) — the bases;
      ``keep`` (C, T) — the country-years used (observed and ≤ origin);
      ``g_mean`` (r₁·r₂,) — the mean vectorised core ḡ;
      ``V`` (r₁·r₂, n_pc) — the PCA loadings (signs fixed);
      ``var_share`` (r₁·r₂,) — variance share of every component;
      ``scores`` (C, T, n_pc) — PCA scores, NaN outside ``keep``;
      ``z_mean`` (S, A) and ``L`` (S, A, n_pc) — the linear map from scores
      to logit(q_x), ``z = z_mean + L s`` (applied by :func:`scores_to_logit`);
      ``smoothing`` — diagnostics of the age-basis smoothing.
    """
    years = np.asarray(years)
    keep = observed & (years <= origin)[None, :]
    Mz = np.nan_to_num(M)                        # cells outside `keep` are never read
    bases = tz.sex_age_bases(Mz, keep, tau=tau, age_bounds=(age_rank, age_rank))
    U_sex = bases["U_sex"]
    U_age, smooth_diag = smoothing.smooth_age_basis(
        bases["U_age"], smooth["components"], smooth["rest_sigma"], smooth["preserve"],
        smooth["orth_tol"])
    r1, r2 = U_sex.shape[1], U_age.shape[1]

    # Effective cores of the country-years in `keep`, vectorised row-major:
    # element (i, j) of G is element i·r₂ + j of vec(G).
    G = tz.effective_cores(Mz, keep, U_sex, U_age)                 # (r1, r2, C, T)
    X = G[:, :, keep].reshape(r1 * r2, -1).T                        # (N, r1·r2)
    g_mean = X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(X - g_mean, full_matrices=False)
    V = tz.fix_signs(Vt[:n_pc].T)                                   # (r1·r2, n_pc)
    var_share = sv ** 2 / np.sum(sv ** 2)

    scores = np.full(keep.shape + (n_pc,), np.nan)
    scores[keep] = (X - g_mean) @ V

    # The map scores → logit(q_x): column j of L is the schedule S Gⱼ Aᵀ of
    # the j-th loading (the map applied by scores_to_logit).
    z_mean = U_sex @ g_mean.reshape(r1, r2) @ U_age.T               # (S, A)
    L = np.einsum("sp,pqj,aq->saj", U_sex, V.reshape(r1, r2, n_pc), U_age)
    return {"origin": int(origin), "U_sex": U_sex, "U_age": U_age, "keep": keep,
            "g_mean": g_mean, "V": V, "var_share": var_share, "scores": scores,
            "z_mean": z_mean, "L": L, "smoothing": smooth_diag}


def scores_to_logit(s: np.ndarray, z_mean: np.ndarray, L: np.ndarray) -> np.ndarray:
    """logit(q_x) schedules from PCA scores.

    Paper: eq:scores_to_z

    Parameters
    ----------
    s : (..., p) PCA scores
    z_mean : (S, A) schedule of the mean core
    L : (S, A, p) schedules of the PCA loadings (only the first p are used)

    Returns
    -------
    (..., S, A) logit(q_x) schedules.
    """
    p = s.shape[-1]
    return z_mean + np.einsum("...j,saj->...sa", s, L[..., :p])


# ══════════════════════════════════════════════════════════════════════════════
# 2. The drift target (paper §11.3)
# ══════════════════════════════════════════════════════════════════════════════

def country_drifts(scores: np.ndarray, years: Sequence[int], origin: int, window: int,
                   min_years: int) -> np.ndarray:
    """Each population's recent drift of the PCA scores, per calendar year.

    Paper: eq:hier2, sec:forecasting:hierarchy

    The drift of population c is the average annual change of its scores
    over the window ``[origin − window, origin]``, measured between its first
    and last observed years ``a < b`` in the window:
    ``(s_b − s_a) / (b − a)``.  This is the maximum-likelihood drift of a
    random walk with drift observed in years a…b, with or without gaps.

    Parameters
    ----------
    scores : (C, T, p) PCA scores, NaN where not observed (or after the origin)
    years : (T,) calendar years
    origin, window : the window is ``origin − window … origin`` (calendar years)
    min_years : the drift is NaN for a population with fewer observed years
        than this in the window

    Returns
    -------
    (C, p) array of drifts (NaN where undefined).
    """
    years = np.asarray(years)
    in_window = (years >= origin - window) & (years <= origin)
    drifts = np.full((scores.shape[0], scores.shape[2]), np.nan)
    for c in range(scores.shape[0]):
        t_obs = np.flatnonzero(in_window & np.isfinite(scores[c, :, 0]))
        if len(t_obs) >= max(min_years, 2):
            a, b = t_obs[0], t_obs[-1]
            drifts[c] = (scores[c, b] - scores[c, a]) / (years[b] - years[a])
    return drifts


def drift_target(drifts: np.ndarray, c: int, weight_hmd: float, weight_country: float
                 ) -> tuple[np.ndarray, bool]:
    """The drift the forecast reverts to.

    Paper: *eq:hier2, sec:forecasting:hierarchy

    ``δ_hier = w_HMD · δ_HMD + w_country · δ_c``, where ``δ_HMD`` is the
    average drift of every population whose drift is defined.  If the
    population's own drift is undefined (too few observed years in the
    window) the target is ``δ_HMD`` alone; the second return value says
    whether the country drift was available, so that this is reported.

    Parameters
    ----------
    drifts : (C, p) output of :func:`country_drifts`
    c : index of the population being forecast
    """
    hmd = np.nanmean(drifts, axis=0)
    if np.all(np.isfinite(drifts[c])):
        return weight_hmd * hmd + weight_country * drifts[c], True
    return hmd, False


# ══════════════════════════════════════════════════════════════════════════════
# 3. The damped local-linear-trend Kalman filter (paper §11.2)
# ══════════════════════════════════════════════════════════════════════════════
#
# For each PCA component j the state is (level ℓⱼ, drift δⱼ):
#
#     ℓ_t = ℓ_{t−1} + δ_{t−1} + η^ℓ_t,            var η^ℓ = q^ℓⱼ
#     δ_t = ρ δ_{t−1} + (1 − ρ) δ*ⱼ + η^δ_t,       var η^δ = q^δⱼ
#     y_t = ℓ_t + ε_t,                             var ε   = rⱼ
#
# with δ* the drift target.  In matrix form (paper §11.2) the state vector
# stacks all levels and drifts, F = [[I, I], [0, ρI]], H = [I, 0], and the
# covariances Q, R and the initial covariance P₀ are diagonal.  Because of
# that, the p components never interact: the joint filter is exactly p
# independent two-state filters that share ρ, and the joint log-likelihood
# is the sum of theirs.  We exploit this by writing the 2 × 2 algebra out
# explicitly, vectorised over components (and over a batch of parameter
# values, which makes the finite-difference gradient one pass of the
# filter).  tests/test_forecast.py checks the result against the textbook
# matrix filter.

def llt_filter(y: np.ndarray, rho: np.ndarray, q_level: np.ndarray, q_drift: np.ndarray,
               r: np.ndarray, target: np.ndarray, level0: np.ndarray, p0_level: float | np.ndarray,
               p0_drift: np.ndarray) -> dict:
    """Run the damped local-linear-trend Kalman filter over annual data with gaps.

    Paper: eq:fc_level, eq:fc_drift

    Each year is first *predicted* from the previous one (level plus
    drift, the drift damped towards the target).  If the year has an observed score vector the
    prediction is then *updated* with it and the year's contribution to the
    Gaussian log-likelihood (the prediction-error decomposition) is added;
    if the year's row of ``y`` is NaN (no data, or an exceptional year) the
    update is skipped, so the state simply carries its prediction forward
    and its uncertainty grows.

    Parameters
    ----------
    y : (T, p) scores in consecutive calendar years; a row is either fully
        observed or fully NaN
    rho : (B,) damping factors, one per parameter set in the batch
    q_level, q_drift, r : (B, p) level-innovation, drift-innovation and
        observation-noise variances
    target : (p,) or (B, p) drift target δ*
    level0 : (p,) initial level; the initial drift is the target
    p0_level : initial variance of each level component (scalar or (p,))
    p0_drift : (p,) initial variance of each drift component

    Returns
    -------
    dict with ``loglik`` (B,), the filtered state after the last year —
    ``level``, ``drift``, ``P_ll``, ``P_ld``, ``P_dd`` (each (B, p)) — and
    ``n_updates``, the number of years with data.

    Notes
    -----
    The initial state is the state *before* the first year, so the first
    year is predicted from it like every other year.
    """
    y = np.asarray(y, dtype=float)
    observed_rows = np.isfinite(y).all(axis=1)
    if not np.array_equal(observed_rows, np.isfinite(y).any(axis=1)):
        raise ValueError("Each row of y must be either fully observed or fully missing.")
    rho = np.asarray(rho, dtype=float)[:, None]                     # (B, 1)
    shape = np.broadcast_shapes(np.shape(q_level), np.shape(q_drift), np.shape(r),
                                (rho.shape[0], y.shape[1]))
    target = np.broadcast_to(target, shape)
    pull = (1.0 - rho) * target                                     # the intercept b of §11.2
    level = np.broadcast_to(level0, shape).copy()
    drift = target.copy()
    P_ll = np.broadcast_to(np.asarray(p0_level, dtype=float), shape).copy()
    P_ld = np.zeros(shape)
    P_dd = np.broadcast_to(p0_drift, shape).copy()
    loglik = np.zeros(shape[0])

    for t in range(y.shape[0]):
        # Predict: x ← F x + b, P ← F P Fᵀ + Q (written out for the 2 × 2 blocks).
        level = level + drift
        drift = rho * drift + pull
        P_ll, P_ld, P_dd = (P_ll + 2.0 * P_ld + P_dd + q_level,
                            rho * (P_ld + P_dd),
                            rho ** 2 * P_dd + q_drift)
        if not observed_rows[t]:
            continue                                                # no data: skip the update
        # Update with this year's scores: innovation v, its variance S, gain K.
        v = y[t] - level
        S = P_ll + r
        k_level, k_drift = P_ll / S, P_ld / S
        level = level + k_level * v
        drift = drift + k_drift * v
        # P ← (I − K H) P, using the predicted P on the right-hand side.
        P_ll, P_ld, P_dd = (P_ll - k_level * P_ll,
                            P_ld - k_level * P_ld,
                            P_dd - k_drift * P_ld)
        loglik += -0.5 * np.sum(_LOG_2PI + np.log(S) + v ** 2 / S, axis=-1)
    return {"loglik": loglik, "level": level, "drift": drift, "P_ll": P_ll, "P_ld": P_ld,
            "P_dd": P_dd, "n_updates": int(observed_rows.sum())}


def llt_forecast(state: Mapping, rho: float, q_level: np.ndarray, q_drift: np.ndarray,
                 r: np.ndarray, target: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Forecast the scores ``1 … horizon`` years after the filter's last year.

    Paper: eq:fc_level, eq:fc_drift

    The forecast repeats the prediction step without updates.  The
    returned variance is that of a future *observation* of the scores
    (state variance ``P_ll`` plus observation noise ``r``), because the
    forecasts are scored against observed schedules.

    Parameters
    ----------
    state : filtered state with ``level, drift, P_ll, P_ld, P_dd`` (each (p,))
    rho : damping factor
    q_level, q_drift, r, target : (p,) model parameters
    horizon : number of years

    Returns
    -------
    mean (horizon, p) and variance (horizon, p) of the scores; the
    components are independent, so the covariance is diagonal.
    """
    level, drift = np.array(state["level"], float), np.array(state["drift"], float)
    P_ll, P_ld, P_dd = (np.array(state[k], float) for k in ("P_ll", "P_ld", "P_dd"))
    mean = np.empty((horizon, level.size))
    var = np.empty((horizon, level.size))
    for h in range(horizon):
        level = level + drift
        drift = rho * drift + (1.0 - rho) * target
        P_ll, P_ld, P_dd = (P_ll + 2.0 * P_ld + P_dd + q_level, rho * (P_ld + P_dd),
                            rho ** 2 * P_dd + q_drift)
        mean[h], var[h] = level, P_ll + r
    return mean, var


def _unpack(theta: np.ndarray, p: int) -> tuple:
    """Parameter vector(s) θ → (ρ, q_level, q_drift, r).

    θ holds the standard deviations σ^ℓ (p), σ^δ (p), σ^r (p) and ρ itself
    (1); the variances are the squares of the σ.  Both stay in their ranges
    through the optimiser's box bounds.  Many maxima lie on a bound (a
    variance at its floor, i.e. effectively zero, or ρ at its lower bound).
    A logistic ρ would put such a maximum at infinity; on the scale of log q
    the slope ∂ℓ/∂log q = q·∂ℓ/∂q vanishes towards the floor, so an
    optimiser stopped on a small relative decrease of −log L ends far from
    it.  On the σ scale the slope is 2σ·∂ℓ/∂q, flat only within a few
    multiples of the floor.  ``theta`` may be (k,) or (B, k).
    """
    theta = np.atleast_2d(theta)
    return (theta[:, 3 * p], theta[:, :p] ** 2, theta[:, p:2 * p] ** 2, theta[:, 2 * p:3 * p] ** 2)


def fit_llt(y: np.ndarray, target: np.ndarray, p0_drift: np.ndarray, settings: MDMxSettings) -> dict:
    """Maximum-likelihood fit of the drift-constrained local linear trend.

    Paper: sec:forecasting:kalman

    The ``3p + 1`` parameters — the ``p`` variances each of the level
    innovations, the drift innovations and the observation noise, and the
    damping factor ρ — maximise the Kalman-filter log-likelihood
    (prediction-error decomposition) by L-BFGS-B, with ρ in ``rho_bounds``
    and every variance in ``exp(log_var_bounds)``.  The
    optimiser works on the standard deviations and on ρ itself (see
    :func:`_unpack`).  It runs until −log L no longer decreases at machine
    precision (``mle_ftol``), the projected gradient is below ``mle_gtol``,
    or the line search can make no further progress; whether the end point
    is a maximum is then judged by its projected gradient (see
    ``converged``).  The gradient is computed by central finite differences,
    all ``2(3p+1)`` perturbed parameter sets being run through the filter in
    one batch.

    The likelihood can have more than one local maximum — typically one
    with ρ at its lower bound and a higher one inside its range — so the
    optimiser is run from several deterministic starts and the largest
    log-likelihood is kept.  This makes a missed maximum unlikely, not
    impossible.  The starts, in this order:

    * the ``mle_grid_top`` best points of the profile log-likelihood of ρ
      on the grid ``mle_grid_rho`` (:func:`_profile_starts`);
    * moment-based starting values — q^ℓ = var(Δy), q^δ = var(Δ²y)/4,
      r = var(Δy)/10, with the differences taken between consecutive
      calendar years that are both observed — with ρ in the middle of its
      range and at each value in ``mle_rho_starts``.

    Parameters
    ----------
    y : (T, p) annual scores from the population's first observed year to
        the origin (NaN rows in years without an observed schedule)
    target : (p,) drift target δ_hier
    p0_drift : (p,) initial drift variance (the spread of the drifts across
        populations)
    settings : the forecaster's settings

    Returns
    -------
    dict with ``rho``, ``q_level``, ``q_drift``, ``r`` (each (p,) but ρ),
    ``level0``, the filtered ``state`` after the last year and ``loglik``;
    for the best start ``n_iter``, ``n_fev``, the optimiser's ``message`` and
    ``proj_grad``, the largest projected-gradient component of −log L at the
    optimum; ``start_logliks`` (one per start, in the order above; −inf for
    a start that ended at a non-finite value), ``start_kinds`` ("profile"
    or "moment" for each), ``best_start``, ``moment_start`` (the index of
    the moment-based start with ρ in the middle of its range),
    ``n_nonfinite_starts``;
    ``rho_profile`` (the profile log-likelihood at ``mle_grid_rho``) and
    ``profile_converged``, ``profile_proj_grad``, ``profile_message`` for
    the fit that produced it.  ``converged`` is False if the best start
    stopped at the iteration limit or its ``proj_grad`` exceeds
    ``mle_pg_flag``; such a fit is used as it is and the chapter reports how
    many there are.  A line search that stops at the precision of the
    finite-difference gradient (L-BFGS-B's "ABNORMAL" message) at a point
    with a small projected gradient is a converged fit.
    """
    p = y.shape[1]
    level0, sd0 = _initial_values(y, settings)
    bounds = _bounds(settings, p)
    lower, upper = np.array(bounds).T
    starts, rho_profile, profile_fit = _profile_starts(y, target, level0, p0_drift, settings,
                                                       lower[0], upper[0])
    kinds = ["profile"] * len(starts)
    moment_start = len(starts)
    moment_rhos = (0.5 * (lower[-1] + upper[-1]), *settings.mle_rho_starts)
    starts += [np.append(sd0, rho) for rho in moment_rhos]
    kinds += ["moment"] * len(moment_rhos)
    results = _run_starts(y, target, level0, p0_drift, settings, starts)
    start_logliks = np.array([-res.fun for res in results], dtype=float)
    finite = np.isfinite(start_logliks)
    if not finite.any():
        raise FloatingPointError("Non-finite log-likelihood at the end of every start: "
                                 + "; ".join(str(res.message) for res in results))
    # A start that ended at a non-finite value is out of the running; it is counted, not hidden.
    start_logliks = np.where(finite, start_logliks, -np.inf)
    best = int(np.argmax(start_logliks))
    res = results[best]
    proj_grad = float(np.max(np.abs(np.clip(res.x - res.jac, lower, upper) - res.x)))
    rho, q_level, q_drift, r = (a[0] for a in _unpack(res.x, p))
    filt = llt_filter(y, np.array([rho]), q_level[None], q_drift[None], r[None], target, level0,
                      settings.p0_level_var, p0_drift)
    state = {key: filt[key][0] for key in ("level", "drift", "P_ll", "P_ld", "P_dd")}
    at_limit = res.status == 1                      # iteration or evaluation limit reached
    return {"rho": float(rho), "q_level": q_level, "q_drift": q_drift, "r": r, "level0": level0,
            "state": state, "loglik": float(start_logliks[best]), "n_iter": int(res.nit),
            "n_fev": int(res.nfev), "converged": bool(not at_limit and proj_grad <= settings.mle_pg_flag),
            "message": str(res.message), "proj_grad": proj_grad, "start_logliks": start_logliks,
            "start_kinds": tuple(kinds), "best_start": best, "moment_start": moment_start,
            "n_nonfinite_starts": int((~finite).sum()), "rho_profile": rho_profile,
            "profile_converged": profile_fit["converged"], "profile_proj_grad": profile_fit["proj_grad"],
            "profile_message": profile_fit["message"], "n_updates": filt["n_updates"]}


def _initial_values(y: np.ndarray, settings: MDMxSettings) -> tuple[np.ndarray, np.ndarray]:
    """The filter's initial level (p,) and the moment-based starting standard deviations (3p,) for ``y`` (T, p).

    See :func:`fit_llt` for both; the standard deviations are clipped to their bounds.
    """
    observed_rows = np.flatnonzero(np.isfinite(y).all(axis=1))
    if len(observed_rows) < 3:
        raise ValueError(f"Need at least 3 observed years to fit the filter, got {len(observed_rows)}.")
    level0 = y[observed_rows[:settings.level0_years]].mean(axis=0)
    d1 = np.diff(y, axis=0)                         # NaN unless both years observed
    d2 = np.diff(y, n=2, axis=0)
    if np.sum(np.isfinite(d1[:, 0])) < 2 or np.sum(np.isfinite(d2[:, 0])) < 2:
        raise ValueError("Too few consecutive observed years to initialise the variances.")
    lo, hi = settings.log_var_bounds
    var1, var2 = np.nanvar(d1, axis=0), np.nanvar(d2, axis=0)
    return level0, np.clip(np.sqrt(np.concatenate([var1, var2 / 4.0, var1 / 10.0])), np.exp(lo / 2.0),
                           np.exp(hi / 2.0))


def _bounds(settings: MDMxSettings, p: int) -> list[tuple[float, float]]:
    """Box bounds of θ (see :func:`_unpack`): the standard deviations' from ``log_var_bounds``, then ρ's."""
    lo, hi = settings.log_var_bounds
    return [(np.exp(lo / 2.0), np.exp(hi / 2.0))] * (3 * p) + [tuple(settings.rho_bounds)]


def _run_starts(y: np.ndarray, target: np.ndarray, level0: np.ndarray, p0_drift: np.ndarray,
                settings: MDMxSettings, starts: Sequence[np.ndarray]) -> list:
    """L-BFGS-B on −log L from each start (θ vectors, see :func:`_unpack`); one SciPy result per start.

    The gradient is computed by central finite differences, all perturbed
    parameter sets of a step being run through the filter in one batch.
    """
    p = y.shape[1]
    k = 3 * p + 1
    steps = settings.fd_step * np.eye(k)
    bounds = _bounds(settings, p)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        # Row 0: θ; rows 1…k: θ + εeᵢ; rows k+1…2k: θ − εeᵢ.  A σ − ε below the
        # floor (even below zero) is harmless: log L is a smooth, even function
        # of σ on the whole real line, because only σ² enters the filter.
        batch = np.vstack([theta, theta + steps, theta - steps])
        rho, q_level, q_drift, r = _unpack(batch, p)
        ll = llt_filter(y, rho, q_level, q_drift, r, target, level0, settings.p0_level_var,
                        p0_drift)["loglik"]
        grad = -(ll[1:k + 1] - ll[k + 1:]) / (2.0 * settings.fd_step)
        return -ll[0], grad

    return [minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": settings.mle_maxiter, "ftol": settings.mle_ftol,
                              "gtol": settings.mle_gtol})
            for theta0 in starts]


def fit_llt_batch(series: Sequence[tuple[np.ndarray, np.ndarray]], p0_drift: np.ndarray,
                  settings: MDMxSettings, workers: int = 1) -> list[dict]:
    """:func:`fit_llt` for several series, optionally in parallel worker processes.

    Every fit is deterministic and independent of the others, so the results
    do not depend on ``workers``; only the run time does.  The workers are
    started with "spawn", which re-imports the calling script: a script
    (not a notebook) that calls this with ``workers != 1`` must keep its
    top-level work under ``if __name__ == "__main__":``.

    Parameters
    ----------
    series : one ``(y, target)`` pair per fit — ``y`` (T, p) scores and
        ``target`` (p,), as for :func:`fit_llt`
    p0_drift : (p,) initial drift variance, shared by all fits
    settings : the forecaster's settings
    workers : number of worker processes; 1 fits in this process, 0 uses one
        per CPU core

    Returns
    -------
    The :func:`fit_llt` results, in the order of ``series``.
    """
    if workers < 0:
        raise ValueError(f"workers must be 0 (one per CPU core) or positive, got {workers}.")
    n_workers = min(workers or os.cpu_count() or 1, len(series))
    if n_workers <= 1:
        return [fit_llt(y, target, p0_drift, settings) for y, target in series]
    # Longest series first, so that no long fit starts last and keeps the
    # others waiting; the results are put back in the order of `series`.
    order = sorted(range(len(series)), key=lambda i: -series[i][0].shape[0])
    # "spawn" on every platform: fork is unsafe in a process with threads (a Jupyter kernel).
    with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        fits = pool.map(fit_llt, [series[i][0] for i in order], [series[i][1] for i in order],
                        itertools.repeat(p0_drift), itertools.repeat(settings))
        out = dict(zip(order, fits, strict=True))
    return [out[i] for i in range(len(series))]


def _profile_starts(y: np.ndarray, target: np.ndarray, level0: np.ndarray, p0_drift: np.ndarray,
                    settings: MDMxSettings, sd_lo: float, sd_hi: float
                    ) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Starting values for :func:`fit_llt` from the profile log-likelihood of ρ.

    Given ρ the log-likelihood is a sum over the ``p`` components, each of
    which depends only on its own three standard deviations.  Two steps, at
    every ρ of the grid ``mle_grid_rho`` (G values):

    1. *Grid search, component by component.*  Each standard deviation
       takes its floor ``sd_lo`` or one of ``mle_grid_sd_points`` log-spaced
       multiples ``10^a … 10^b`` (``mle_grid_sd_log10`` = (a, b)) of the data's
       own scale — the SD of the annual changes for the level and the noise,
       half the SD of the second differences for the drift; all
       combinations at all grid ρ run through the filter in one batch per
       component, and each component keeps its best combination.
    2. *Exact fit at fixed ρ.*  From there the standard deviations are fitted
       at every grid ρ at once: the G fixed-ρ problems are independent, so
       they are stacked into one L-BFGS-B problem whose filter runs form one
       batch.  The result is the profile log-likelihood of ρ.

    The starts are the ``mle_grid_top`` best grid values of ρ with their
    fitted standard deviations, so that a local maximum of the likelihood at
    a bound of ρ is unlikely to hide a higher one inside the range.  The
    profile is only as good as the fixed-ρ fits: a fixed-ρ problem can
    itself have local maxima, and its fit can stop early, which the returned
    status reports.

    Parameters
    ----------
    y, target, p0_drift : as for :func:`fit_llt`; ``level0`` (p,) its initial level
    settings : the forecaster's settings
    sd_lo, sd_hi : bounds of the standard deviations

    Returns
    -------
    The starts (each a θ vector for :func:`_unpack`, best first), the
    profile log-likelihood (G,), and a dict with the stacked fit's
    ``converged`` (not stopped by a limit, largest projected-gradient
    component at most ``mle_pg_flag``), ``proj_grad`` and ``message``.
    """
    p = y.shape[1]
    rho_grid = np.asarray(settings.mle_grid_rho, dtype=float)
    G = rho_grid.size
    sig0, _ = _grid_sigmas(y, target, level0, p0_drift, settings, sd_lo, sd_hi)
    m = 3 * p
    eye = settings.fd_step * np.eye(m)
    rho_batch = np.repeat(rho_grid, 2 * m + 1)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        # For each grid ρ: its σ, then σ ± εeᵢ (central differences), all in one filter batch.
        batch = np.concatenate([np.vstack([s, s + eye, s - eye]) for s in flat.reshape(G, m)])
        _, q_level, q_drift, r = _unpack(np.column_stack([batch, rho_batch]), p)
        ll_g = llt_filter(y, rho_batch, q_level, q_drift, r, target, level0, settings.p0_level_var,
                          p0_drift)["loglik"].reshape(G, 2 * m + 1)
        grad = -(ll_g[:, 1:m + 1] - ll_g[:, m + 1:]) / (2.0 * settings.fd_step)
        return -ll_g[:, 0].sum(), grad.ravel()

    res = minimize(objective, sig0.ravel(), jac=True, method="L-BFGS-B", bounds=[(sd_lo, sd_hi)] * (G * m),
                   options={"maxiter": settings.mle_profile_maxiter, "maxfun": settings.mle_profile_maxfun,
                            "ftol": settings.mle_ftol, "gtol": settings.mle_gtol})
    proj_grad = float(np.max(np.abs(np.clip(res.x - res.jac, sd_lo, sd_hi) - res.x)))
    status = {"converged": bool(res.status != 1 and proj_grad <= settings.mle_pg_flag),
              "proj_grad": proj_grad, "message": str(res.message)}
    sig = res.x.reshape(G, m)
    _, q_level, q_drift, r = _unpack(np.column_stack([sig, rho_grid]), p)
    profile = llt_filter(y, rho_grid, q_level, q_drift, r, target, level0, settings.p0_level_var,
                         p0_drift)["loglik"]
    top = np.argsort(-profile, kind="stable")[:settings.mle_grid_top]
    return [np.append(sig[g], rho_grid[g]) for g in top], profile, status


def _grid_sigmas(y: np.ndarray, target: np.ndarray, level0: np.ndarray, p0_drift: np.ndarray,
                 settings: MDMxSettings, sd_lo: float, sd_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 of :func:`_profile_starts`: the grid search of the standard deviations at every grid ρ.

    Returns
    -------
    The best grid standard deviations at each of the G grid values of ρ,
    (G, 3p) in the order of θ (level, drift, noise), and the grid's crude
    profile log-likelihood (G,): the sum over components of their best grid
    log-likelihood.
    """
    p = y.shape[1]
    scale_level = np.sqrt(np.nanvar(np.diff(y, axis=0), axis=0))
    scale_drift = 0.5 * np.sqrt(np.nanvar(np.diff(y, n=2, axis=0), axis=0))
    mult = np.logspace(*settings.mle_grid_sd_log10, settings.mle_grid_sd_points)
    cand = [np.clip(np.vstack([np.full((1, p), sd_lo), scale[None] * mult[:, None]]), sd_lo, sd_hi)
            for scale in (scale_level, scale_drift, scale_level)]          # (n, p): level, drift, noise
    n = cand[0].shape[0]
    i_l, i_d, i_r = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"))
    rho_grid = np.asarray(settings.mle_grid_rho, dtype=float)
    G, C = rho_grid.size, n ** 3
    rho_rows = np.repeat(rho_grid, C)                                     # (G·C,)
    ll = np.empty((G, C, p))
    for j in range(p):
        c = [j]
        ll[:, :, j] = llt_filter(y[:, c], rho_rows, cand[0][np.tile(i_l, G)][:, c] ** 2,
                                 cand[1][np.tile(i_d, G)][:, c] ** 2, cand[2][np.tile(i_r, G)][:, c] ** 2,
                                 target[c], level0[c], settings.p0_level_var, p0_drift[c])["loglik"].reshape(G, C)
    best = ll.argmax(axis=1)                                              # (G, p)
    comp = np.arange(p)
    sig = np.stack([np.concatenate([cand[0][i_l[best[g]], comp], cand[1][i_d[best[g]], comp],
                                    cand[2][i_r[best[g]], comp]]) for g in range(G)])   # (G, 3p)
    return sig, ll.max(axis=1).sum(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. From scores to e₀, with its delta-method standard deviation
# ══════════════════════════════════════════════════════════════════════════════

def e0_and_gradient(z: np.ndarray, a0: float = lifetable.A0) -> tuple[np.ndarray, np.ndarray]:
    """e₀ and its exact gradient with respect to logit(q_x).

    Paper: eq:delta_method, sec:fc-mdmx

    With the life-table conventions of :mod:`mdmx.lifetable` (:func:`mdmx.lifetable.e0_from_qx`),
    e₀ is a weighted sum of the survivorship ``l_k`` (k = 0…A):
    ``e₀ = Σ_k w_k l_k`` with ``w₀ = a₀``, ``w₁ = (1 − a₀) + ½``,
    ``w_k = 1`` for ``2 ≤ k ≤ A−1`` and ``w_A = ½``.  Since
    ``l_k = Π_{x<k}(1 − q_x)`` and ``dq_x/dz_x = q_x(1 − q_x)``,

        ∂e₀/∂z_x = −q_x · Σ_{k>x} w_k l_k ,

    a reverse cumulative sum.  The gradient is what the delta-method SD of
    :func:`mdmx_origin` needs; a finite-difference check is in the tests.

    Parameters
    ----------
    z : (..., A) logit(q_x) schedules (A ≥ 3), age on the last axis
    a0 : fraction of the first year lived by infants who die

    Returns
    -------
    e₀ (...) and ∂e₀/∂z (..., A).
    """
    z = np.asarray(z, dtype=float)
    A = z.shape[-1]
    q = lifetable.expit(z)
    lx = lifetable.survivorship(q)                                  # (..., A+1)
    w = np.ones(A + 1)
    w[0], w[1], w[A] = a0, (1.0 - a0) + 0.5, 0.5
    wl = w * lx
    e0 = wl.sum(axis=-1)
    tail = np.flip(np.cumsum(np.flip(wl, axis=-1), axis=-1), axis=-1)   # tail[k] = Σ_{j≥k} w_j l_j
    grad = -q * tail[..., 1:]
    return e0, grad


def _e0_by_sex(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """e₀ and gradient for (..., 2, A) schedules, adding the 'both' series.

    Returns e₀ (..., 3) and gradients (..., 3, 2, A); the 'both' gradient
    is half the female gradient on the female schedule and half the male
    gradient on the male schedule.
    """
    e0, grad = e0_and_gradient(z)                                   # (..., 2), (..., 2, A)
    e0_3 = np.concatenate([e0, 0.5 * e0.sum(axis=-1, keepdims=True)], axis=-1)
    g3 = np.zeros(z.shape[:-2] + (3,) + z.shape[-2:])
    g3[..., 0, 0, :] = grad[..., 0, :]
    g3[..., 1, 1, :] = grad[..., 1, :]
    g3[..., 2, :, :] = 0.5 * grad
    return e0_3, g3


# ══════════════════════════════════════════════════════════════════════════════
# The MDMx forecaster for one origin
# ══════════════════════════════════════════════════════════════════════════════

def origin_inputs(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                  origin: int, jobs: pd.DataFrame, settings: MDMxSettings) -> dict:
    """What the fits of one origin start from: the space, the drifts and each job's series.

    Parameters as for :func:`mdmx_origin`.

    Returns
    -------
    dict with ``space`` (:func:`origin_space`), ``drifts`` (C, p)
    (:func:`country_drifts`), ``p0_drift`` (p,) (their variance across
    populations, the initial drift variance), ``t_origin`` (index of the
    origin in ``years``), and per job, in the order of ``jobs``: ``series``,
    the ``(y, target)`` pairs that :func:`fit_llt` is given — ``y`` (T, p)
    the annual scores from the job's first year to the origin (NaN in years
    without an observed schedule), ``target`` (p,) the drift target — and
    ``own_drift``, whether the population's own drift entered the target.
    """
    years = np.asarray(years)
    pidx = {p: i for i, p in enumerate(np.asarray(pops))}
    space = origin_space(M, observed, years, origin, tau=settings.tau, age_rank=settings.age_rank,
                         smooth=settings.smooth, n_pc=settings.n_pc)
    scores = space["scores"]                                        # (C, T, p)
    drifts = country_drifts(scores, years, origin, settings.drift_window, settings.drift_min_years)
    t_origin = int(np.flatnonzero(years == origin)[0])
    series, own_drifts = [], []
    for j, pop in enumerate(jobs["pop"]):
        c = pidx[pop]
        # The training series starts at the job's first year (by default the
        # population's first observed year; later if the job restricts the
        # fitting window) and runs, year by year, to the origin.
        t_first = int(np.flatnonzero(space["keep"][c])[0])
        if "first_year" in jobs.columns:
            t_first = max(t_first, int(np.searchsorted(years, jobs["first_year"].iloc[j])))
        target, own_drift = drift_target(drifts, c, settings.weight_hmd, settings.weight_country)
        series.append((scores[c, t_first:t_origin + 1], target))
        own_drifts.append(own_drift)
    return {"space": space, "drifts": drifts, "p0_drift": np.maximum(np.nanvar(drifts, axis=0), 1e-8),
            "t_origin": t_origin, "series": series, "own_drift": own_drifts}


def anchor_weights(years_after: np.ndarray, anchor: str | float) -> np.ndarray:
    """Weight of the jump-off residual at each forecast year.

    Paper: eq:fc-anchor

    ``"none"``: 0; ``"persistent"``: 1 at every horizon (ρ = 1, decision C); a number: a decay with
    that half-life in years, 2^(−h / half-life), h counted from the jump-off year.

    Parameters
    ----------
    years_after : (H,) years since the jump-off year (the job's last training year)
    anchor : ``"none"``, ``"persistent"`` or a half-life (years, positive)
    """
    h = np.asarray(years_after, dtype=float)
    if anchor == "none":
        return np.zeros_like(h)
    if anchor == "persistent":
        return np.ones_like(h)
    if isinstance(anchor, str) or not float(anchor) > 0:
        raise ValueError(f"anchor must be 'none', 'persistent' or a positive half-life, not {anchor!r}")
    return 2.0 ** (-h / float(anchor))


def anchor_smoother(n_ages: int, lam: float) -> np.ndarray:
    """The Whittaker smoother of the anchor's jump-off residual, as a matrix.

    Paper: eq:fc-anchor-smooth

    Row a of K gives the smoothed value at age a: K = (I + λ D₂ᵀD₂)⁻¹ on ages 1..A−1, with D₂ the
    second differences (equal weights), the minimiser of Σ_a (x̃_a − x_a)² + λ Σ_a (Δ²x̃_a)²; age 0
    keeps its observed value, because the step from infancy to childhood is real.  The smoother keeps
    constants and straight lines over ages 1..A−1.

    Parameters
    ----------
    n_ages : A, the number of single ages; lam : λ ≥ 0 (0 gives the identity)

    Returns
    -------
    (A, A) K, with x̃ = K x along the age axis.
    """
    if not (np.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be finite and not negative, got {lam}")
    K = np.eye(n_ages)
    if lam > 0:
        m = n_ages - 1
        D2 = np.diff(np.eye(m), n=2, axis=0)
        K[1:, 1:] = np.linalg.solve(np.eye(m) + lam * D2.T @ D2, np.eye(m))
    return K


def smooth_residual(resid: np.ndarray, lam: float) -> np.ndarray:
    """The jump-off residual the anchored forecasters carry: each sex's residual smoothed over age.

    Paper: eq:fc-anchor-smooth, eq:fc-anchor, eq:ff-anchor

    The residual r = M_{c,t₀} − z̄ − L s(0), the part of the jump-off year's observed schedule that the
    scores do not represent, holds the population's own detail and the noise of that one year's
    single-age rates.  Smoothed over age (:func:`anchor_smoother`), it keeps the detail and drops
    most of the noise, which the forecasts would otherwise carry at every horizon.

    Parameters
    ----------
    resid : (..., S, A) logit(q_x) residuals; lam : λ of :func:`anchor_smoother` (0 returns a copy)

    Returns
    -------
    (..., S, A) the smoothed residuals.
    """
    resid = np.asarray(resid, dtype=float)
    if lam == 0:
        return resid.copy()
    return resid @ anchor_smoother(resid.shape[-1], lam).T


def mdmx_origin(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                origin: int, jobs: pd.DataFrame, settings: MDMxSettings, *,
                mc_draws: int = 0, mc_seed: int = 0, workers: int = 1, anchor: str | float = "none") -> dict:
    """Fit and forecast every population of one cross-validation origin.

    Paper: *eq:fc-anchor, sec:fc-mdmx, eq:delta_method

    Steps: estimate the space from data ≤ origin
    (:func:`origin_space`); compute every population's drift and the
    HMD-wide drift; for each job, run the maximum-likelihood fit of the
    Kalman filter on the population's annual scores from its first observed
    year to the origin (:func:`fit_llt`), forecast ``horizon`` years
    (:func:`llt_forecast`), map the scores to schedules and e₀, and compute
    the delta-method standard deviation of e₀.

    Parameters
    ----------
    M, observed, pops, years : the tensor and its labels (see
        :func:`observed_e0`)
    origin : forecast origin (calendar year)
    jobs : the rows of :func:`cv_jobs` for this origin (column ``pop``; if
        there is a column ``first_year``, the training series starts there
        rather than at the population's first observed year)
    settings : the forecaster's settings
    mc_draws : if positive, also compute a Monte-Carlo standard deviation of
        e₀ from this many draws of the forecast scores (to check the delta
        method); 0 skips it
    mc_seed : seed of the Monte-Carlo draws (combined with origin and job)
    workers : worker processes for the fits (see :func:`fit_llt_batch`; the
        results do not depend on it)
    anchor : the jump-off anchor (:func:`anchor_weights`).  With an anchor, the
        residual of the observed schedule at the job's last training year from
        its ``n_pc``-score projection, ``M[:, :, c, t₀] − z_mean − L s(t₀)``,
        smoothed over age (:func:`smooth_residual`, λ = ``settings.anchor_smoothing``),
        is added to every forecast schedule with that year's weight: the part of
        the schedule the scores do not represent is kept, without the noise of
        that one year's single-age rates.  e₀, the delta-method
        SD and the intervals are then those of the anchored schedules.  Needs
        the jobs' ``last_train_year``.

    Returns
    -------
    dict with
      ``points`` — one row per job, forecast year and sex (``female``,
      ``male``, ``both``): ``pop, sex, year, e0_fc, sd`` (delta method),
      ``sd_mc`` (Monte Carlo, NaN unless requested) and raw prediction
      intervals ``lower80, upper80, lower95, upper95`` (``e0_fc ± z·sd``);
      ``jobs`` — one row per job with the fitted parameters and optimiser
      diagnostics (with an anchor, also ``anchor_year`` and
      ``anchor_medabs``, the median |residual| in logit(q_x));
      ``schedule_index`` (pop, year) and ``schedules`` (N, 2, A) — the
      forecast logit(q_x) schedules;
      ``jumpoff`` — for each job whose origin year is observed, how far the
      schedule the forecast starts from (the filtered level at the origin)
      is from the observed schedule there, and how much of that is PCA
      truncation (the observed core's own projection on the ``n_pc``
      components): median and maximum |Δ logit(q_x)| and the e₀ differences;
      ``jumpoff_schedules`` (n, 3, 2, A) — the observed, projected and
      filtered schedules at the origin for those jobs;
      ``var_share``, ``z_mean``, ``L`` — the space's PCA variance shares and
      score-to-schedule map.
    """
    pidx = {p: i for i, p in enumerate(np.asarray(pops))}
    yidx = {int(y): i for i, y in enumerate(np.asarray(years))}
    H, p = settings.horizon, settings.n_pc
    anchor_weights(np.zeros(1), anchor)                             # rejects an invalid anchor early
    if anchor != "none" and "last_train_year" not in jobs.columns:
        raise ValueError("an anchor needs the jobs' last_train_year")
    inputs = origin_inputs(M, observed, pops, years, origin, jobs, settings)
    space, drifts, t_origin = inputs["space"], inputs["drifts"], inputs["t_origin"]
    scores = space["scores"]                                        # (C, T, p)
    zq = {lev: float(norm.ppf(0.5 + lev / 200.0)) for lev in settings.pi_levels}
    fits = fit_llt_batch(inputs["series"], inputs["p0_drift"], settings, workers)

    point_frames, job_rows, sched, sched_idx, jump_rows, jump_sched = [], [], [], [], [], []
    for j, pop in enumerate(jobs["pop"]):
        c = pidx[pop]
        (y, target), own_drift, fit = inputs["series"][j], inputs["own_drift"][j], fits[j]
        s_mean, s_var = llt_forecast(fit["state"], fit["rho"], fit["q_level"], fit["q_drift"],
                                     fit["r"], target, H)

        # Schedules, e₀ and the delta-method SD:
        # σ² = Σⱼ Jⱼ² var(sⱼ) because the score covariance is diagonal.
        z_fc = scores_to_logit(s_mean, space["z_mean"], space["L"])    # (H, 2, A)
        fc_years = origin + np.arange(1, H + 1)
        anchor_info = {}
        if anchor == "none":
            e0, grad = _e0_by_sex(z_fc)                                 # (H, 3), (H, 3, 2, A)
        else:
            t0 = yidx[int(jobs["last_train_year"].iloc[j])]
            if not np.all(np.isfinite(scores[c, t0])):
                raise ValueError(f"{pop}: no scores at its last training year {years[t0]}")
            resid = smooth_residual(M[:, :, c, t0] - scores_to_logit(scores[c, t0][None], space["z_mean"],
                                                                      space["L"])[0], settings.anchor_smoothing)
            w = anchor_weights(fc_years - int(years[t0]), anchor)
            z_fc = z_fc + w[:, None, None] * resid if anchor != "persistent" else z_fc + resid
            # e₀ as lifetable computes it (the anchored forecast's own definition); the gradient for
            # the SD from e0_and_gradient, which agrees with it to 1e-13.
            e_sex = lifetable.e0_from_logit(z_fc)                          # (H, 2)
            e0 = np.concatenate([e_sex, e_sex.mean(axis=1, keepdims=True)], axis=1)
            grad = _e0_by_sex(z_fc)[1]
            anchor_info = {"anchor_year": int(years[t0]), "anchor_medabs": float(np.median(np.abs(resid)))}
        J = np.einsum("hksa,saj->hkj", grad, space["L"][..., :p])      # (H, 3, p)
        sd = np.sqrt(np.einsum("hkj,hj->hk", J ** 2, s_var))
        sd_mc = np.full_like(sd, np.nan)
        if mc_draws > 0:
            rng = np.random.default_rng([mc_seed, origin, j])
            draws = s_mean + np.sqrt(s_var) * rng.standard_normal((mc_draws, H, p))
            e0_draws, _ = _e0_by_sex(scores_to_logit(draws, space["z_mean"], space["L"]))
            sd_mc = e0_draws.std(axis=0)

        for k, sex in enumerate(SEXES):
            frame = pd.DataFrame({"pop": pop, "sex": sex, "year": fc_years, "e0_fc": e0[:, k],
                                  "sd": sd[:, k], "sd_mc": sd_mc[:, k]})
            for lev, zv in zq.items():
                frame[f"lower{lev}"] = e0[:, k] - zv * sd[:, k]
                frame[f"upper{lev}"] = e0[:, k] + zv * sd[:, k]
            point_frames.append(frame)
        sched.append(z_fc)
        sched_idx.append(pd.DataFrame({"pop": pop, "year": fc_years}))

        job_rows.append({
            "pop": pop, "origin": int(origin), "rho": fit["rho"], "loglik": fit["loglik"],
            "n_iter": fit["n_iter"], "n_fev": fit["n_fev"], "converged": fit["converged"],
            "message": fit["message"], "proj_grad": fit["proj_grad"],
            "n_starts": len(fit["start_logliks"]), "best_start_kind": fit["start_kinds"][fit["best_start"]],
            # How far below the maximum the worst start, and the single moment-based start with ρ in
            # the middle of its range, ended (0: it found the maximum; inf: it ended at a non-finite value).
            "start_gap_max": float(fit["loglik"] - np.min(fit["start_logliks"])),
            "moment_gap": float(fit["loglik"] - fit["start_logliks"][fit["moment_start"]]),
            "n_nonfinite_starts": fit["n_nonfinite_starts"],
            "profile_converged": fit["profile_converged"], "profile_proj_grad": fit["profile_proj_grad"],
            "n_years": int(y.shape[0]), "n_updates": fit["n_updates"],
            "own_drift": own_drift,
            **{f"q_level_{i + 1}": fit["q_level"][i] for i in range(p)},
            **{f"q_drift_{i + 1}": fit["q_drift"][i] for i in range(p)},
            **{f"r_{i + 1}": fit["r"][i] for i in range(p)},
            **{f"target_{i + 1}": target[i] for i in range(p)},
            **anchor_info,
        })

        # Jump-off: the schedule the forecast starts from (the filtered level
        # at the origin) against the observed schedule there, and the part of
        # the gap that is PCA truncation (the observed core's own projection).
        if observed[c, t_origin]:
            z_obs = M[:, :, c, t_origin]
            z_filt = scores_to_logit(fit["state"]["level"], space["z_mean"], space["L"])
            z_proj = scores_to_logit(scores[c, t_origin], space["z_mean"], space["L"])
            e_obs, e_filt, e_proj, e_h1 = (_e0_by_sex(z)[0] for z in (z_obs, z_filt, z_proj, z_fc[0]))
            jump_rows.append({
                "pop": pop, "origin": int(origin),
                "medabs_filtered": float(np.median(np.abs(z_filt - z_obs))),
                "maxabs_filtered": float(np.max(np.abs(z_filt - z_obs))),
                "medabs_projection": float(np.median(np.abs(z_proj - z_obs))),
                "medabs_h1": float(np.median(np.abs(z_fc[0] - z_obs))),
                "de0_filtered": float(e_filt[2] - e_obs[2]),
                "de0_projection": float(e_proj[2] - e_obs[2]),
                "de0_h1": float(e_h1[2] - e_obs[2]),
            })
            jump_sched.append(np.stack([z_obs, z_proj, z_filt]))

    return {
        "points": pd.concat(point_frames, ignore_index=True),
        "jobs": pd.DataFrame(job_rows),
        "schedule_index": pd.concat(sched_idx, ignore_index=True),
        "schedules": np.concatenate(sched, axis=0),
        "jumpoff": pd.DataFrame(jump_rows),
        "jumpoff_schedules": (np.stack(jump_sched) if jump_sched
                              else np.zeros((0, 3) + M.shape[:2])),
        "var_share": space["var_share"], "z_mean": space["z_mean"], "L": space["L"],
        "hmd_drift": np.nanmean(drifts, axis=0), "n_drift_countries": int(np.isfinite(drifts[:, 0]).sum()),
    }


#: The functions whose source code determines the space of an origin
#: (:func:`origin_space`) and an MDMx forecast (:func:`mdmx_origin`).  Chapter
#: 11's artefact cache keys include them, so that changing any of them
#: recomputes what depends on it; tests/test_forecast.py checks that every
#: mdmx function they call is listed.
SPACE_CODE = (origin_space, tz.sex_age_bases, tz.effective_cores, tz.fix_signs, tz.leading_left_vectors,
              tz.unfold, tz.select_rank, smoothing.smooth_age_basis, smoothing.bandwidths,
              smoothing.kernel_matrix)
MDMX_CODE = (mdmx_origin, anchor_weights, anchor_smoother, smooth_residual, origin_inputs, fit_llt_batch, country_drifts, drift_target, fit_llt,
             _initial_values, _bounds, _run_starts, _profile_starts, _grid_sigmas, llt_filter, llt_forecast,
             _unpack, scores_to_logit, e0_and_gradient, _e0_by_sex, lifetable.survivorship, lifetable.expit,
             lifetable.e0_from_logit, lifetable.e0_from_qx,
             *SPACE_CODE)


# ══════════════════════════════════════════════════════════════════════════════
# The benchmark harness
# ══════════════════════════════════════════════════════════════════════════════

_POINT_COLUMNS = ["method", "pop", "sex", "origin", "year", "horizon", "e0_obs", "e0_fc", "sd",
                  "lower80", "upper80", "lower95", "upper95"]


def run_benchmark(methods: Mapping[str, Callable[[int, pd.DataFrame], pd.DataFrame | dict]],
                  jobs: pd.DataFrame, truth: pd.DataFrame, *,
                  cache: Mapping[str, Mapping] | None = None,
                  cache_prefix: str = "forecast_bench") -> pd.DataFrame:
    """Run forecasting methods on the same jobs and line their forecasts up with the truth.

    The method contract (how a new method plugs in)
    ------------------------------------------------
    A method is a function ``method(origin: int, jobs_at_origin: DataFrame)``
    that forecasts every population in ``jobs_at_origin`` from ``origin``.

    * ``jobs_at_origin`` holds the rows of :func:`cv_jobs` (or of
      :func:`restrict_fit_window`) for that origin: ``pop, origin, horizon,
      n_train, n_test, first_year, last_train_year``.  Training data are the
      population's observed years from ``first_year`` to ``last_train_year``
      (≤ origin); nothing after the origin may be used.
    * It returns a DataFrame with one row per (``pop``, ``sex``, ``year``):
      ``e0_fc`` for ``sex`` = ``female``, ``male`` and ``both`` (the average
      of the female and male e₀ of the same forecast), for at least the years
      ``origin + 1 … origin + horizon``.  Optional columns, for a method with
      a predictive distribution of e₀: ``lower80``/``upper80`` (10th/90th
      percentiles), ``lower95``/``upper95`` (2.5th/97.5th) and ``sd``.
      Missing columns are filled with NaN.
    * Or it returns a dict whose ``"points"`` entry is that DataFrame;
      everything else in the dict is cached too (chapter 11 reads
      ``schedule_index``/``schedules`` for schedule-level errors and
      ``failures``/``warnings`` for R methods when present).

    Adding a method to the comparison is therefore one function plus one
    entry in ``methods`` (and, for caching, one entry in ``cache``).

    The expected points are every (job, sex, year) for which the truth has
    an observed e₀ within the job's horizon.  A point a method did not
    produce, or produced as NaN, is kept with ``e0_fc = NaN``; nothing is
    dropped silently.  The column ``paired`` marks the points (pop, origin,
    year) for which *every* method produced a finite e₀ for all three sex
    series — the set on which methods must be compared.

    Parameters
    ----------
    methods : name → forecasting function
    jobs : rows of :func:`cv_jobs` (``pop, origin, horizon`` are used)
    truth : output of :func:`observed_e0` (``pop, sex, year, e0_obs``)
    cache : optional name → keyword arguments of :func:`mdmx.artifacts.stage`
        (``inputs``, ``params``, ``code``, ``desc``).  A method with an entry
        is cached per origin as artefact ``<cache_prefix>_<name>_<origin>``
        and recomputed only when its inputs, parameters or code change; a
        method without one is simply run.  ``code`` must list the functions
        whose source determines the result (a ``functools.partial`` has no
        source of its own).
    cache_prefix : first part of the artefact names, so that the same method
        can be cached under several data policies (e.g. ``forecast_bench``
        and ``forecast_win50``).

    Returns
    -------
    Tidy DataFrame with one row per method and expected point: ``method,
    pop, sex, origin, year, horizon, e0_obs, e0_fc, sd, lower80, upper80,
    lower95, upper95, paired``.
    """
    cache = cache or {}
    expected = (jobs[["pop", "origin", "horizon"]]
                .merge(truth[["pop", "sex", "year", "e0_obs"]], on="pop"))
    expected = expected[(expected["year"] > expected["origin"])
                        & (expected["year"] <= expected["origin"] + expected["horizon"])]
    expected = expected.assign(horizon=expected["year"] - expected["origin"])

    frames = []
    for name, method in methods.items():
        for origin, jobs_o in jobs.groupby("origin", sort=True):
            origin = int(origin)
            jobs_o = jobs_o.reset_index(drop=True)
            compute = functools.partial(method, origin, jobs_o)
            if name in cache:
                meta = dict(cache[name])
                meta.setdefault("code", [method])
                result = artifacts.stage(f"{cache_prefix}_{name}_{origin}", compute, **meta)
            else:
                result = compute()
            points = result["points"] if isinstance(result, dict) else result
            missing = {"pop", "sex", "year", "e0_fc"} - set(points.columns)
            if missing:
                raise ValueError(f"Method {name!r} returned no column(s) {sorted(missing)}.")
            if points.duplicated(["pop", "sex", "year"]).any():
                raise ValueError(f"Method {name!r} returned duplicate (pop, sex, year) rows at {origin}.")
            keep = [c for c in ("pop", "sex", "year", "e0_fc", "sd", "lower80", "upper80",
                                "lower95", "upper95") if c in points.columns]
            merged = expected[expected["origin"] == origin].merge(
                points[keep], on=["pop", "sex", "year"], how="left")
            frames.append(merged.assign(method=name))
    out = pd.concat(frames, ignore_index=True)
    for col in _POINT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[_POINT_COLUMNS]

    # A (pop, origin, year) is paired if every method has a finite forecast
    # for every sex series there.
    finite = np.isfinite(out["e0_fc"].to_numpy(dtype=float))
    per_point = (out.assign(ok=finite).groupby(["pop", "origin", "year"])["ok"]
                 .agg(["all", "size"]).reset_index())
    n_expected = len(methods) * len(SEXES)
    per_point["paired"] = per_point["all"] & (per_point["size"] == n_expected)
    out = out.merge(per_point[["pop", "origin", "year", "paired"]], on=["pop", "origin", "year"])
    return out.sort_values(["method", "origin", "pop", "year", "sex"]).reset_index(drop=True)


def summarise(points: pd.DataFrame, by: Sequence[str] = ("method",)) -> pd.DataFrame:
    """Point-forecast accuracy: n, MAE, RMSE and bias (forecast − observed) by group."""
    err = points["e0_fc"] - points["e0_obs"]
    g = points.assign(err=err, abs_err=err.abs(), sq_err=err ** 2).groupby(list(by), sort=True)
    out = g.agg(n=("err", "size"), MAE=("abs_err", "mean"), RMSE=("sq_err", "mean"), Bias=("err", "mean"))
    out["RMSE"] = np.sqrt(out["RMSE"])
    return out.reset_index()


def loo_calibrate(points: pd.DataFrame, levels: Sequence[int] = (80, 95),
                  calibration: Mapping[int, Sequence[int]] | None = None) -> pd.DataFrame:
    """Leave-one-origin-out calibration of delta-method intervals.

    Paper: *eq:fc-kappa, sec:forecasting:intervals

    The interval half-width is ``z_α · κ · sd``.  The scale factor κ is the
    standard deviation of the standardised errors ``(e0_obs − e0_fc)/sd``,
    but for the points of origin o it is estimated
    only from the points of the *other* origins, so that no point helps to
    calibrate its own interval.  κ is estimated separately for each sex
    series.

    Parameters
    ----------
    points : one method's scored points with ``origin, sex, e0_obs, e0_fc,
        sd`` (e.g. the MDMx rows of :func:`run_benchmark`); rows without an
        observation get an interval but give no standardised error
    levels : nominal coverages, per cent
    calibration : origin → the origins whose errors calibrate it (e.g.
        :func:`mdmx.flowfield.within_era_calibration`); None: every other
        origin.  An origin never calibrates itself.

    Returns
    -------
    ``points`` with columns ``z`` (standardised error), ``kappa_loo`` and
    ``lower{L}_loo``, ``upper{L}_loo`` for each level.
    """
    out = points.copy()
    out["z"] = (out["e0_obs"] - out["e0_fc"]) / out["sd"]
    out["kappa_loo"] = np.nan
    if calibration is not None:
        missing = sorted(set(out["origin"].astype(int)) - set(calibration))
        if missing:
            raise ValueError(f"No calibration set for the origins {missing}")
    for sex in out["sex"].unique():
        is_sex = out["sex"] == sex
        for origin in out.loc[is_sex, "origin"].unique():
            mine = is_sex & (out["origin"] == origin)
            if calibration is None:
                others = is_sex & (out["origin"] != origin)
            else:
                cal = [int(o) for o in calibration[int(origin)]]
                if int(origin) in cal:
                    raise ValueError(f"Origin {origin} is in its own calibration set")
                others = is_sex & out["origin"].isin(cal)
            out.loc[mine, "kappa_loo"] = float(np.std(out.loc[others, "z"].dropna()))
    for lev in levels:
        half = norm.ppf(0.5 + lev / 200.0) * out["kappa_loo"] * out["sd"]
        out[f"lower{lev}_loo"] = out["e0_fc"] - half
        out[f"upper{lev}_loo"] = out["e0_fc"] + half
    return out


def coverage(points: pd.DataFrame, level: int, suffix: str = "") -> float:
    """Share of observed e₀ inside the ``level``% interval (columns ``lower{level}{suffix}``…)."""
    lo, hi = points[f"lower{level}{suffix}"], points[f"upper{level}{suffix}"]
    return float(np.mean((points["e0_obs"] >= lo) & (points["e0_obs"] <= hi)))


def schedule_errors(z_fc: np.ndarray, z_obs: np.ndarray, groups: Mapping[str, Sequence[int]],
                    weights: str = "survivors") -> pd.DataFrame:
    """Weighted RMSE of forecast logit(q_x) by age group, both sexes pooled.

    Paper: *eq:ev-schedule-rmse

    In each age group the squared errors are weighted by the observed
    survivorship l_x of each sex, so that ages few people reach count
    little (the age-pattern error of the forecast evaluation, §12), or by the observed
    life-table deaths d_x = l_x q_x, so that each age counts by the deaths at
    it: the old ages, where most deaths are, and infancy.

    Parameters
    ----------
    z_fc, z_obs : (N, 2, A) forecast and observed logit(q_x) schedules
    groups : age-group label → list of ages
    weights : ``"survivors"`` (l_x) or ``"deaths"`` (d_x)

    Returns
    -------
    DataFrame with one row per schedule and a column per age group, plus
    ``all`` (every age).
    """
    lx = _age_weights(z_obs, weights)                               # (N, 2, A)
    sq = (z_fc - z_obs) ** 2
    out = {}
    for label, ages in {**groups, "all": list(range(z_obs.shape[-1]))}.items():
        w = lx[..., ages]
        out[label] = np.sqrt(np.sum(w * sq[..., ages], axis=(1, 2)) / np.sum(w, axis=(1, 2)))
    return pd.DataFrame(out)


def _age_weights(z_obs: np.ndarray, weights: str) -> np.ndarray:
    """(N, 2, A) weights of the ages: the observed l_x (``survivors``) or d_x = l_x q_x (``deaths``)."""
    qx = lifetable.expit(z_obs)
    lx = lifetable.survivorship(qx)[..., :-1]
    if weights == "survivors":
        return lx
    if weights == "deaths":
        return lx * qx
    raise ValueError(f"weights must be 'survivors' or 'deaths', not {weights!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparators (R demography): input data and reading their forecasts
# ══════════════════════════════════════════════════════════════════════════════

def comparator_data(de: pd.DataFrame, observed: np.ndarray, pops: Sequence[str],
                    years: Sequence[int], open_age: int) -> pd.DataFrame:
    """HMD deaths and exposures for the comparators, on the MDMx data policy.

    Paper: sec:evaluation:entries

    * Only observed (non-exceptional) country-years are used — the same
      years the MDMx filter is updated with.
    * Ages ``open_age`` and above (including the HMD's 110+) are summed into
      one open age group, the ``demography`` default for Lee–Carter.
    * The comparators need an unbroken annual series, so within each
      population's span (first to last observed year) a year without an
      observed schedule is filled by linear interpolation, in calendar
      time, of the deaths and exposures of the neighbouring observed years
      (column ``filled`` is True for those rows).  A filled year uses only
      the two observed years around it, so a training series that ends at
      an origin never contains information from after the origin.

    Parameters
    ----------
    de : HMD deaths and exposures (``pop, sex, year, age, deaths, exposure``)
    observed : (C, T) observed mask of the tensor
    pops, years : labels of the tensor's country and year axes
    open_age : first age of the open age group

    Returns
    -------
    Long DataFrame ``pop, sex, year, age, deaths, exposure, filled``.
    """
    years = np.asarray(years)
    c_idx, t_idx = np.nonzero(observed)
    obs_cy = pd.DataFrame({"pop": np.asarray(pops)[c_idx].astype(str),
                           "year": years[t_idx].astype(int)})
    d = de.assign(age=np.minimum(de["age"].to_numpy(), open_age), sex=de["sex"].astype(str),
                  pop=de["pop"].astype(str), year=de["year"].astype(int))
    d = d.merge(obs_cy, on=["pop", "year"])
    d = d.groupby(["pop", "sex", "age", "year"], as_index=False)[["deaths", "exposure"]].sum()

    frames = []
    for (pop, sex), g in d.groupby(["pop", "sex"], sort=True):
        span = np.arange(g["year"].min(), g["year"].max() + 1)
        for col in ("deaths", "exposure"):
            wide = g.pivot(index="age", columns="year", values=col).reindex(columns=span)
            # Linear in calendar time between observed years (columns are consecutive years).
            filled = wide.interpolate(axis=1, method="linear", limit_area="inside")
            long = filled.stack(future_stack=True).rename(col).reset_index()
            if col == "deaths":
                base = long
            else:
                base[col] = long[col].to_numpy()
        base.insert(0, "sex", sex)
        base.insert(0, "pop", pop)
        base["filled"] = ~base["year"].isin(g["year"].unique())
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    if out[["deaths", "exposure"]].isna().any().any():
        raise ValueError("Interpolation left missing deaths or exposures inside a population's span.")
    return out[["pop", "sex", "year", "age", "deaths", "exposure", "filled"]]


def pooled_deaths(de: pd.DataFrame, lt_curated: pd.DataFrame, lt_pooled: pd.DataFrame) -> pd.DataFrame:
    """The HMD's deaths with the pooled rates of chapter d5 put in, for the comparators' smoother.

    Paper: sec:evaluation:entries

    ``demography``'s smoother weights each cell by its deaths on the log scale, so a cell with no
    death drops out and the curve is fitted through the positive cells only: in the smallest
    populations the smoothed rates then sit above the truth at the ages where deaths are rare.  The
    pooled cells (eq:pooling; the cells whose m_x pooling changed) get the deaths the pooled rate
    implies, ``m_pool × E`` with the year's own exposure, so the smoother sees the rate that the
    tensor, the truth and the new forecasters use; every other cell keeps its count.

    Parameters
    ----------
    de : HMD deaths and exposures (``pop, sex, year, age, deaths, exposure``)
    lt_curated, lt_pooled : the curated life tables and the same after pooling (``pop, sex, year,
        age, mx``, same rows)

    Returns
    -------
    ``de`` with ``deaths`` replaced in the pooled cells (same rows and columns).
    """
    key = ["pop", "sex", "year", "age"]
    lt = lt_curated[key + ["mx"]].merge(lt_pooled[key + ["mx"]], on=key, suffixes=("", "_pooled"),
                                        validate="one_to_one")
    pooled = lt.loc[lt["mx"] != lt["mx_pooled"], key + ["mx_pooled"]]
    out = de.merge(pooled, on=key, how="left", validate="one_to_one")
    if len(out) != len(de):
        raise ValueError("pooled cells matched more than one row of the deaths and exposures")
    hit = out["mx_pooled"].notna()
    if hit.sum() != len(pooled):
        raise ValueError(f"{len(pooled) - int(hit.sum())} pooled cells have no deaths and exposures")
    out.loc[hit, "deaths"] = out.loc[hit, "mx_pooled"] * out.loc[hit, "exposure"]
    return out.drop(columns="mx_pooled")[list(de.columns)]


def comparator_schedules(mx: pd.DataFrame, n_ages: int, ax: float = 0.5,
                         q_min: float = lifetable.Q_MIN_DEFAULT) -> tuple[pd.DataFrame, np.ndarray]:
    """Turn the comparators' forecast death rates into logit(q_x) schedules.

    Paper: sec:evaluation:entries

    The R methods forecast m_x at ages 0 … ``open_age`` (the last one being
    the open group).  The open group's rate is used at every age from
    ``open_age`` to ``n_ages − 1`` (a constant force of mortality beyond it),
    and ``q_x = m_x / (1 + (1 − a_x) m_x)`` with ``a_x = ax`` above age 0 (the
    HMD's own single-year value) and the book's a₀ (:data:`mdmx.lifetable.A0`)
    at age 0.  a₀ matters when infant mortality is high: before 1950, a₀ = 0.5
    put the comparators' e₀ below the book's e₀ of the same population-years
    (chapter 12 reports the median, ``evAzeroOffsetEarly``); with the book's
    a₀ the offset is close to zero.

    Parameters
    ----------
    mx : wide DataFrame with columns ``method, pop, origin, sex, year`` and
        one column ``a0 … a{open_age}`` per age
    n_ages : number of single-year ages of the tensor (110)

    Returns
    -------
    The index (``method, pop, origin, year``) and an array (N, 2, n_ages) of
    logit(q_x), female then male.  A schedule with a missing sex is NaN.
    """
    age_cols = sorted((c for c in mx.columns if c.startswith("a") and c[1:].isdigit()),
                      key=lambda c: int(c[1:]))
    key = ["method", "pop", "origin", "year"]
    if mx.duplicated(key + ["sex"]).any():
        raise ValueError("Duplicate forecast rows for one method, population, origin, year and sex.")
    index = mx[key].drop_duplicates().sort_values(key).reset_index(drop=True)
    m = np.full((len(index), 2, len(age_cols)), np.nan)              # (N, 2, open_age+1)
    for s, sex in enumerate(("female", "male")):
        rows = index.merge(mx[mx["sex"] == sex], on=key, how="left")   # keeps the order of `index`
        m[:, s] = rows[age_cols].to_numpy(dtype=float)
    open_rate = m[..., -1:]
    m_full = np.concatenate([m[..., :-1], np.repeat(open_rate, n_ages - m.shape[-1] + 1, axis=-1)],
                            axis=-1)
    q = lifetable.qx_from_mx(m_full, lifetable.conversion_ax(n_ages, ax))
    return index, lifetable.logit(q, q_min)


def smooth_comparator_rates(rates: pd.DataFrame) -> dict:
    """Smooth every population-sex series year by year in R (``R/hmd_smooth.R``).

    Parameters
    ----------
    rates : output of :func:`comparator_data`

    Returns
    -------
    dict with ``smoothed`` (``pop, sex, year, age, mx``) and ``failures``
    (series R could not smooth, with R's message).
    """
    return rbridge.run("hmd_smooth.R", {"rates": rates.drop(columns="filled")},
                       ["smoothed", "failures"], tag="hmd-smooth")


def comparator_forecasts(rates: pd.DataFrame, smoothed: pd.DataFrame, jobs: pd.DataFrame,
                         settings: Mapping, *, tag: str) -> dict:
    """Lee–Carter, Hyndman–Ullah and coherent FDM forecasts in R for a set of jobs.

    Paper: sec:evaluation:entries

    Each job is fitted to the population's series from its first observed
    year to its last observed year at or before the origin
    (``last_train_year``), and forecast far enough to reach
    ``origin + horizon``.  Only the rows the jobs need are sent to R.

    Parameters
    ----------
    rates : output of :func:`comparator_data`
    smoothed : the ``smoothed`` output of :func:`smooth_comparator_rates`
    jobs : rows of :func:`cv_jobs` (``pop, origin, horizon, first_year,
        last_train_year``)
    settings : the ``[forecast.comparators]`` parameters
    tag : name of the working folder under ``work/r/``

    Returns
    -------
    dict with ``forecasts`` (wide m_x: ``method, pop, origin, sex, year,
    a0 …``), ``failures`` and ``warnings`` (``method, pop, origin,
    message``) as returned by ``R/demography_forecasts.R``.
    """
    r_jobs = pd.DataFrame({
        "pop": jobs["pop"], "origin": jobs["origin"], "first_year": jobs["first_year"],
        "last_year": jobs["last_train_year"],
        "h": jobs["origin"] + jobs["horizon"] - jobs["last_train_year"],
    })
    last = r_jobs.set_index("pop")["last_year"]
    if last.index.duplicated().any():
        raise ValueError("comparator_forecasts expects one job per population (one origin).")

    def needed(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame["pop"].isin(last.index)]
        return frame[frame["year"] <= frame["pop"].map(last)]

    return rbridge.run("demography_forecasts.R",
                       {"rates": needed(rates).drop(columns="filled"), "smoothed": needed(smoothed),
                        "jobs": r_jobs},
                       ["forecasts", "failures", "warnings"], tag=tag, settings=dict(settings))


def comparator_points(run: Mapping, method: str, n_ages: int, ax: float) -> dict:
    """e₀ forecasts and schedules of one R method, in the harness's format.

    Parameters
    ----------
    run : output of :func:`comparator_forecasts`
    method : ``"LC"``, ``"HU"`` or ``"CFDM"``
    n_ages : number of single-year ages of the tensor
    ax : a_x used to convert m_x to q_x (see :func:`comparator_schedules`)

    Returns
    -------
    dict with ``points`` (``pop, sex, year, e0_fc``), ``schedule_index``
    (``pop, year``) and ``schedules`` (N, 2, A) logit(q_x), and the R
    ``failures`` and ``warnings`` of this method.
    """
    mx = run["forecasts"]
    mx = mx[mx["method"] == method]
    index, z = comparator_schedules(mx, n_ages, ax=ax)
    e0, _ = _e0_by_sex(z)                                           # (N, 3)
    points = pd.concat([pd.DataFrame({"pop": index["pop"].astype(str), "sex": sex,
                                      "year": index["year"].astype(int), "e0_fc": e0[:, k]})
                        for k, sex in enumerate(SEXES)], ignore_index=True)
    pick = {name: frame[frame["method"] == method].reset_index(drop=True)
            for name, frame in run.items() if name in ("failures", "warnings")}
    return {"points": points, "schedule_index": index[["pop", "year"]].astype({"year": int}),
            "schedules": z, **pick}
