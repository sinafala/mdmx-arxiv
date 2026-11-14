"""The overall comparison of the forecasting methods (chapter 11, task T5b).

The procedure was specified before any of it was computed
(``docs/changes/11-forecasting.md`` §9, ``[forecast.overall]``):

* every method is scored on *families* of measures — e₀; the sexes (female
  and male e₀ and the female–male gap); age-specific sex coherence; the
  age pattern of the schedules; the uncertainty (weighted interval score) —
  each measure divided by the error of a naive random walk with drift for
  the same quantity, and a family's loss is the mean of its measures;
* the methods are ranked over all weightings of the families (weights from
  Dirichlet(1, …, 1)) jointly with a bootstrap of population families —
  stochastic multicriteria acceptability analysis (SMAA; Lahdelma &
  Salminen 2001) — so that no weighting has to be chosen.

This module holds the scores (:func:`weighted_interval_score`), the naive
benchmark (:func:`rw_drift`, :func:`naive_e0`, :func:`naive_schedules`), the
comparators' interval forecasts (:func:`interval_runs`,
:func:`lc_quantiles`), the age-specific sex-difference error and its
long-run rate in MDMx (:func:`sex_difference_errors`,
:func:`sex_difference_slope`) and the ranking (:func:`overall`).
"""

from __future__ import annotations

import functools
import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from . import artifacts, lifetable, paths, provenance, rbridge
from . import forecast as fc
from .bayeslife import run_many

#: R packages of the comparators' interval forecasts (recorded in the cache key).
R_PACKAGES = ["demography", "forecast", "ftsa", "data.table", "jsonlite"]
#: Fraction of the first year lived by infants who die: the book's one a₀ (:data:`mdmx.lifetable.A0`).
A0 = lifetable.A0


# ══════════════════════════════════════════════════════════════════════════════
# Scores of predictive distributions
# ══════════════════════════════════════════════════════════════════════════════

def quantile_levels(pi_levels: Sequence[int]) -> np.ndarray:
    """Quantile levels of central prediction intervals and the median.

    ``(80, 95)`` → ``(0.025, 0.10, 0.50, 0.90, 0.975)``.
    """
    lower = sorted((100.0 - lev) / 200.0 for lev in pi_levels)
    return np.array(lower + [0.5] + [1.0 - p for p in reversed(lower)])


def quantile_columns(levels: Sequence[float]) -> list[str]:
    """Column names of quantiles: 0.025 → ``q025``, 0.5 → ``q500``."""
    return [f"q{round(1000 * lev):03d}" for lev in levels]


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha) -> np.ndarray:
    """Interval score of a central (1 − α) prediction interval (Gneiting & Raftery 2007).

    ``IS_α = (u − l) + (2/α)(l − y)·1{y < l} + (2/α)(y − u)·1{y > u}``, in the
    units of y.  All arguments broadcast.
    """
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    alpha = np.asarray(alpha, dtype=float)
    return (upper - lower) + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)


def weighted_interval_score(y: np.ndarray, median: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                            alphas: Sequence[float]) -> np.ndarray:
    """Weighted interval score (Bracher et al. 2021).

    Paper: eq:ev-wis, sec:evaluation:intervals

    ``WIS = [½|y − m| + Σ_k (α_k/2)·IS_αk] / (K + ½)``: a proper score of the
    predictive distribution given by its median and K central intervals,
    approximating the CRPS (in the units of y; lower is better).

    Parameters
    ----------
    y, median : (N,) observations and predictive medians
    lower, upper : (N, K) bounds of the central (1 − α_k) intervals
    alphas : (K,) the α of each interval
    """
    y, median = np.asarray(y, dtype=float), np.asarray(median, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if np.any(lower > upper):
        raise ValueError("a lower bound exceeds its upper bound")
    scores = interval_score(y[:, None], lower, upper, alphas[None, :])
    return (0.5 * np.abs(y - median) + (0.5 * alphas * scores).sum(axis=1)) / (alphas.size + 0.5)


def wis_from_quantiles(y: np.ndarray, q: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """:func:`weighted_interval_score` from predictive quantiles.

    Parameters
    ----------
    y : (N,) observations
    q : (N, L) quantiles at ``levels`` (ascending, symmetric about the median
        0.5, which must be included)
    """
    levels = np.asarray(levels, dtype=float)
    q = np.asarray(q, dtype=float)
    K = (levels.size - 1) // 2
    if levels.size % 2 == 0 or not np.isclose(levels[K], 0.5) or not np.allclose(levels + levels[::-1], 1.0):
        raise ValueError("the quantile levels must be symmetric about an included median")
    if np.any(np.diff(q, axis=1) < -1e-9):
        raise ValueError("the quantiles are not non-decreasing")
    return weighted_interval_score(y, q[:, K], q[:, :K], q[:, ::-1][:, :K], 2.0 * levels[:K])


def gaussian_quantiles(mean: np.ndarray, sd: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Quantiles (..., L) of normal distributions with the given means and SDs (...)."""
    return np.asarray(mean, dtype=float)[..., None] + np.asarray(sd, dtype=float)[..., None] * norm.ppf(levels)


def crps_gaussian(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score of a normal forecast (closed form)."""
    z = (np.asarray(y, dtype=float) - mean) / sd
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


# ══════════════════════════════════════════════════════════════════════════════
# The naive benchmark: a random walk with drift
# ══════════════════════════════════════════════════════════════════════════════

def rw_drift(t: np.ndarray, x: np.ndarray) -> dict:
    """Fit a random walk with drift to a series observed in calendar years, with gaps.

    Paper: sec:evaluation:entries

    The increments are ``Δx_j ~ N(d·Δt_j, σ²·Δt_j)``, so the drift is
    ``d = (x_b − x_a)/(b − a)`` (first and last observed years a, b) and
    ``σ² = Σ_j (Δx_j − d·Δt_j)²/Δt_j / (n_inc − 1)``.

    Parameters
    ----------
    t : (n,) increasing calendar years (n ≥ 3)
    x : (n, ...) values (any trailing shape, e.g. (n, 2, A) logit schedules)

    Returns
    -------
    dict with ``drift`` and ``sigma2`` (...), ``t_last``, ``x_last`` (...)
    and ``span`` (b − a, years).
    """
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    if t.ndim != 1 or t.size < 3 or t.size != x.shape[0] or np.any(np.diff(t) <= 0):
        raise ValueError("a random walk with drift needs at least three increasing years, one value each")
    span = t[-1] - t[0]
    drift = (x[-1] - x[0]) / span
    dt = np.diff(t).reshape((-1,) + (1,) * (x.ndim - 1))
    resid = np.diff(x, axis=0) - drift * dt
    sigma2 = np.sum(resid ** 2 / dt, axis=0) / (t.size - 2)
    return {"drift": drift, "sigma2": sigma2, "t_last": t[-1], "x_last": x[-1], "span": span}


def rw_drift_forecast(fit: Mapping, t_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast a fitted random walk with drift.

    Paper: sec:evaluation:entries

    ``mean = x_b + d·h`` and ``var = σ²(h + h²/(b − a))`` with ``h = t − b``
    in years from the last observed year (innovations plus the uncertainty
    of the estimated drift).

    Returns
    -------
    mean and SD, each (H, ...).
    """
    h = np.asarray(t_new, dtype=float) - fit["t_last"]
    if np.any(h <= 0):
        raise ValueError("forecast years must come after the last observed year")
    h = h.reshape((-1,) + (1,) * np.ndim(fit["x_last"]))
    mean = fit["x_last"] + fit["drift"] * h
    sd = np.sqrt(fit["sigma2"] * (h + h ** 2 / fit["span"]))
    return mean, sd


def naive_e0(truth: pd.DataFrame, jobs: pd.DataFrame, levels: Sequence[float]) -> pd.DataFrame:
    """The naive benchmark's e₀ forecasts and predictive quantiles for a set of jobs.

    Paper: sec:evaluation:entries, eq:ev-intervals, sec:evaluation:intervals

    For each job a random walk with drift (:func:`rw_drift`) is fitted to the
    population's observed e₀ in its window ``first_year … last_train_year``
    — female, male and their average ("both"), each as its own series — and
    forecast to ``origin + 1 … origin + horizon`` with normal quantiles.

    Parameters
    ----------
    truth : :func:`mdmx.forecast.observed_e0` (``pop, sex, year, e0_obs``)
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (``pop, origin, horizon,
        first_year, last_train_year``)
    levels : quantile levels (:func:`quantile_levels`)

    Returns
    -------
    DataFrame ``pop, sex, origin, year, e0_fc, sd`` and one column per
    quantile (:func:`quantile_columns`).
    """
    series = {key: g.set_index("year")["e0_obs"].sort_index() for key, g in truth.groupby(["pop", "sex"])}
    qcols = quantile_columns(levels)
    frames = []
    for job in jobs.itertuples():
        years_fc = np.arange(job.origin + 1, job.origin + job.horizon + 1)
        for sex in fc.SEXES:
            s = series[(job.pop, sex)]
            s = s[(s.index >= job.first_year) & (s.index <= job.last_train_year)]
            mean, sd = rw_drift_forecast(rw_drift(s.index.to_numpy(), s.to_numpy()), years_fc)
            frame = pd.DataFrame({"pop": job.pop, "sex": sex, "origin": int(job.origin), "year": years_fc,
                                  "e0_fc": mean, "sd": sd})
            frame[qcols] = gaussian_quantiles(mean, sd, levels)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def naive_schedules(M: np.ndarray, observed: np.ndarray, pops: Sequence[str], years: Sequence[int],
                    jobs: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """The naive benchmark's logit(q_x) forecasts at the given points.

    Paper: sec:evaluation:entries

    A random walk with drift (:func:`rw_drift`) on logit(q_x) of each age and
    sex, fitted to the population's observed schedules in the job's window.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor; observed : (C, T) mask of observed,
        non-exceptional country-years; pops, years : their labels
    jobs : rows of :func:`mdmx.forecast.cv_jobs`, one per (pop, origin)
    keys : the points ``pop, origin, year``

    Returns
    -------
    (len(keys), S, A) forecast logit(q_x).
    """
    years = np.asarray(years)
    pidx = {p: i for i, p in enumerate(pops)}
    keys = keys[["pop", "origin", "year"]].reset_index(drop=True)
    windows = jobs.set_index(["pop", "origin"])[["first_year", "last_train_year"]]
    out = np.full((len(keys),) + M.shape[:2], np.nan)
    for (pop, origin), rows in keys.groupby(["pop", "origin"]):
        first, last = windows.loc[(pop, origin)]
        c = pidx[pop]
        t_sel = np.nonzero(observed[c] & (years >= first) & (years <= last))[0]
        fit = rw_drift(years[t_sel], np.moveaxis(M[:, :, c, t_sel], -1, 0))      # x: (n, S, A)
        out[rows.index.to_numpy()] = rw_drift_forecast(fit, rows["year"].to_numpy())[0]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The comparators' interval forecasts (R demography)
# ══════════════════════════════════════════════════════════════════════════════

def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def job_seeds(jobs: pd.DataFrame, base_seed: int, policy: str) -> pd.DataFrame:
    """Per-job seeds for R's ``set.seed`` (Hyndman–Ullah, coherent FDM).

    Derived from the base seed, the policy, the origin and the population —
    not from the order of the jobs — so that a job's draws do not change when
    other jobs are added or removed.

    Returns
    -------
    DataFrame ``pop, origin, seed_hu, seed_cfdm`` (integers below 2³¹ − 1).
    """
    rows = []
    for job in jobs.itertuples():
        state = np.random.SeedSequence([int(base_seed), _crc(policy), int(job.origin), _crc(job.pop)])
        hu, cfdm = (int(v) % (2 ** 31 - 1) for v in state.generate_state(2))
        rows.append({"pop": job.pop, "origin": int(job.origin), "seed_hu": hu, "seed_cfdm": cfdm})
    return pd.DataFrame(rows)


def comparator_intervals(rates: pd.DataFrame, smoothed: pd.DataFrame, jobs: pd.DataFrame,
                         settings: Mapping, *, seeds: pd.DataFrame, nsim: int, levels: Sequence[float],
                         n_ages: int, ax: float, tag: str) -> dict:
    """Interval forecasts of e₀ by Lee–Carter, Hyndman–Ullah and the coherent FDM for one origin.

    Runs ``R/demography_intervals.R``, which refits the three methods exactly
    as :func:`mdmx.forecast.comparator_forecasts` does and returns, per job:
    Lee–Carter's k_t forecast (mean and standard error, drift uncertainty
    included) and age pattern, from which :func:`lc_quantiles` computes e₀
    quantiles; and the e₀ quantiles of ``nsim`` draws of
    ``demography::simulate()`` for Hyndman–Ullah and the coherent FDM, with
    e₀ computed by the chapter's life-table convention.

    Parameters
    ----------
    rates, smoothed, jobs, settings : as for :func:`mdmx.forecast.comparator_forecasts`
    seeds : :func:`job_seeds` of these jobs
    nsim : draws per job and method; levels : quantile levels
    n_ages, ax : the life-table convention (ages 0 … n_ages − 1; a_x)
    tag : name of the working folder under ``work/r/``

    Returns
    -------
    dict with ``quantiles`` (``method, pop, origin, sex, year, level, e0``;
    sex ``female``, ``male``, ``both`` — the per-draw average, which for
    Hyndman–Ullah pairs independent draws), ``points`` (``method, pop,
    origin, sex, year, e0_point, n_finite``: the refitted point forecast's
    e₀ and the number of finite draws), ``lc_kt`` (``pop, origin, sex, year,
    kt_mean, kt_se``), ``lc_age`` (``pop, origin, sex, age, log_jump, bx``),
    ``failures`` and ``warnings``.
    """
    r_jobs = pd.DataFrame({
        "pop": jobs["pop"], "origin": jobs["origin"], "first_year": jobs["first_year"],
        "last_year": jobs["last_train_year"],
        "h": jobs["origin"] + jobs["horizon"] - jobs["last_train_year"],
    }).merge(seeds, on=["pop", "origin"], how="left", validate="one_to_one")
    if r_jobs[["seed_hu", "seed_cfdm"]].isna().any().any():
        raise ValueError("a job has no seed")
    last = r_jobs.set_index("pop")["last_year"]
    if last.index.duplicated().any():
        raise ValueError("comparator_intervals expects one job per population (one origin)")

    def needed(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame["pop"].isin(last.index)]
        return frame[frame["year"] <= frame["pop"].map(last)]

    cfg = {**dict(settings), "nsim": int(nsim), "levels": [float(v) for v in levels], "n_ages": int(n_ages),
           "ax": float(ax), "a0": A0, "q_min": lifetable.Q_MIN_DEFAULT}
    return rbridge.run("demography_intervals.R",
                       {"rates": needed(rates).drop(columns="filled"), "smoothed": needed(smoothed),
                        "jobs": r_jobs},
                       ["quantiles", "points", "lc_kt", "lc_age", "failures", "warnings"], tag=tag, settings=cfg)


def interval_runs(rates: pd.DataFrame, smoothed: pd.DataFrame, jobs: Mapping[str, pd.DataFrame],
                  settings: Mapping, *, nsim: int, sim_seed: int, levels: Sequence[float], n_ages: int,
                  ax: float, concurrent: int, prefix: str = "forecast_pi") -> dict[tuple[str, int], dict]:
    """Every origin of every fitting-period policy, cached, several at once (chapter 11).

    Each origin of each policy is the artefact ``<prefix>_<policy>_<origin>``,
    keyed on its jobs, the comparator settings, nsim, the seed, the quantile
    levels, the R script and the R package versions.  The point forecasts
    (:func:`mdmx.forecast.comparator_forecasts`) are not touched.

    Parameters
    ----------
    jobs : policy → its jobs; settings : ``[forecast.comparators]``
    concurrent : R processes at once (run time only; not in the key)

    Returns
    -------
    (policy, origin) → the result of :func:`comparator_intervals`.
    """
    key = {"comparators": dict(settings), "nsim": int(nsim), "sim_seed": int(sim_seed),
           "levels": [float(v) for v in levels], "n_ages": int(n_ages), "ax": float(ax),
           "script": provenance.sha256_file(paths.root() / "R" / "demography_intervals.R"),
           "r": rbridge.session_info("demography_intervals.R", R_PACKAGES)}
    tasks = []
    for policy, jobs_p in jobs.items():
        for origin, jobs_o in jobs_p.groupby("origin", sort=True):
            origin = int(origin)
            jobs_o = jobs_o.reset_index(drop=True)
            tag = f"{prefix}-{policy}-{origin}".replace("_", "-")
            compute = functools.partial(comparator_intervals, rates, smoothed, jobs_o, settings,
                                        seeds=job_seeds(jobs_o, sim_seed, policy), nsim=nsim, levels=levels,
                                        n_ages=n_ages, ax=ax, tag=tag)
            tasks.append(((policy, origin), functools.partial(
                artifacts.stage, f"{prefix}_{policy}_{origin}", compute,
                inputs=["forecast_cmp_rates", "forecast_cmp_smoothed"],
                params={**key, "jobs": artifacts.content_hash(jobs_o), "policy": policy},
                code=[comparator_intervals, job_seeds, rbridge.run],
                desc=f"Lee-Carter, Hyndman-Ullah and coherent FDM interval forecasts of e0 from {origin} "
                     f"(policy {policy})")))
    return run_many(tasks, concurrent=concurrent)


def e0_from_mx(m: np.ndarray, n_ages: int, ax: float) -> np.ndarray:
    """e₀ of death-rate schedules by the comparators' convention in this chapter.

    The last column is the open age group, whose rate is used at every age
    up to ``n_ages − 1``; then :func:`mdmx.forecast.comparator_schedules`'s
    conversion (``q = m/(1 + (1 − a)m)``, logit with the floor) and
    :func:`mdmx.lifetable.e0_from_logit` (the e₀ of
    :func:`mdmx.forecast.e0_and_gradient`, without the gradient).

    Parameters
    ----------
    m : (..., n_open) death rates at ages 0 … open age
    """
    m = np.asarray(m, dtype=float)
    m_full = np.concatenate([m[..., :-1], np.repeat(m[..., -1:], n_ages - m.shape[-1] + 1, axis=-1)], axis=-1)
    z = lifetable.logit(lifetable.qx_from_mx(m_full, lifetable.conversion_ax(n_ages, ax, A0)))
    return lifetable.e0_from_logit(z, A0)


def lc_quantiles(lc_kt: pd.DataFrame, lc_age: pd.DataFrame, levels: Sequence[float], *, n_ages: int,
                 ax: float, both_rule: str, n_grid: int = 2000, n_indep: int = 20000,
                 seed: int = 0) -> pd.DataFrame:
    """Lee–Carter's predictive e₀ quantiles from its k_t forecast.

    Paper: eq:ev-intervals, sec:evaluation:intervals

    Lee–Carter's interval comes from the random walk with drift of k_t alone
    (``demography::forecast.lca``, drift uncertainty included): every age
    moves with k, ``m_x(k) = exp(log jumprate_x + b_x k)``, so e₀ is a
    function g(k) of a normal k.  Its quantiles are those of g(k) over an
    equal-probability grid of k (``n_grid`` points at Φ⁻¹((i − ½)/n)): when g
    is monotone they are g at the k quantiles; when b_x changes sign (adult
    mortality that rose while child mortality fell), g can turn within the
    interval and only the distribution of g(k) gives the right quantiles.

    Parameters
    ----------
    lc_kt : ``pop, origin, sex, year, kt_mean, kt_se``; lc_age : ``pop,
        origin, sex, age, log_jump, bx`` (from :func:`comparator_intervals`)
    levels : quantile levels; n_ages, ax : the life-table convention
    both_rule : ``"comonotone"`` (average of the female and male quantiles)
        or ``"independent"`` (quantiles of the average of independent female
        and male draws, ``n_indep`` normal draws of k per sex and year)
    seed : seed of the independent draws.  Each job (population, origin) draws from its own stream,
        derived from ``seed``, the origin and the population code, so that its quantiles do not
        depend on which other jobs are in ``lc_kt``

    Returns
    -------
    DataFrame ``pop, origin, sex, year, e0_point`` (e₀ at the k mean),
    ``monotone`` (whether g is monotone over the grid, for the job and sex)
    and one column per quantile; sex ``female``, ``male`` and ``both``.
    """
    if both_rule not in ("comonotone", "independent"):
        raise ValueError(f"unknown both_rule {both_rule!r}")
    qcols = quantile_columns(levels)
    z_grid = norm.ppf((np.arange(n_grid) + 0.5) / n_grid)
    out = []
    for (pop, origin), kt in lc_kt.groupby(["pop", "origin"], sort=True):
        rng = np.random.default_rng([seed, int(origin), _crc(str(pop))])
        per_sex, draws = {}, {}
        for sex in ("female", "male"):
            k = kt[kt["sex"] == sex].sort_values("year")
            age = lc_age[(lc_age["pop"] == pop) & (lc_age["origin"] == origin) & (lc_age["sex"] == sex)]
            age = age.sort_values("age")
            log_jump, bx = age["log_jump"].to_numpy(), age["bx"].to_numpy()

            def e0_at(kv, log_jump=log_jump, bx=bx):
                return e0_from_mx(np.exp(log_jump + np.multiply.outer(kv, bx)), n_ages, ax)

            mean, se = k["kt_mean"].to_numpy()[:, None], k["kt_se"].to_numpy()[:, None]
            e_grid = e0_at(mean + se * z_grid[None, :])                   # (years, n_grid), k ascending
            steps = np.diff(e_grid, axis=1)
            monotone = bool(np.all(steps <= 0) or np.all(steps >= 0))
            q = np.quantile(e_grid, levels, axis=1).T                     # (years, L)
            per_sex[sex] = (k["year"].to_numpy(), e0_at(k["kt_mean"].to_numpy()), q, monotone)
            if both_rule == "independent":
                draws[sex] = e0_at(mean + se * rng.standard_normal((len(k), n_indep)))
        years_f, point_f, q_f, mono_f = per_sex["female"]
        years_m, point_m, q_m, mono_m = per_sex["male"]
        if not np.array_equal(years_f, years_m):
            raise ValueError(f"Lee–Carter female and male forecast years differ for {pop} {origin}")
        if both_rule == "comonotone":
            q_b = 0.5 * (q_f + q_m)
        else:
            q_b = np.quantile(0.5 * (draws["female"] + draws["male"]), levels, axis=1).T
        for sex, point, q, mono in (("female", point_f, q_f, mono_f), ("male", point_m, q_m, mono_m),
                                    ("both", 0.5 * (point_f + point_m), q_b, mono_f and mono_m)):
            frame = pd.DataFrame({"pop": pop, "origin": int(origin), "sex": sex, "year": years_f.astype(int),
                                  "e0_point": point, "monotone": mono})
            frame[qcols] = q
            out.append(frame)
    return pd.concat(out, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Schedules: alignment, age-specific sex differences, long-run rate
# ══════════════════════════════════════════════════════════════════════════════

def aligned_schedules(results: Mapping[int, Mapping], keys: pd.DataFrame) -> np.ndarray:
    """A method's forecast schedules at the given points.

    Parameters
    ----------
    results : origin → the method's result for that origin, with
        ``schedule_index`` (``pop, year``) and ``schedules`` (N, 2, A)
    keys : the points ``pop, origin, year``

    Returns
    -------
    (len(keys), 2, A) logit(q_x).
    """
    index = pd.concat([r["schedule_index"][["pop", "year"]].assign(origin=int(o),
                                                                   row=np.arange(len(r["schedule_index"])))
                       for o, r in results.items()], ignore_index=True)
    rows = keys[["pop", "origin", "year"]].merge(index, on=["pop", "origin", "year"], how="left",
                                                 validate="one_to_one")
    if rows["row"].isna().any():
        raise ValueError("a point has no forecast schedule")
    return np.stack([results[o]["schedules"][int(r)] for o, r in zip(rows["origin"], rows["row"], strict=True)])


def sex_difference_errors(z_fc: np.ndarray, z_obs: np.ndarray, weights: str = "survivors") -> np.ndarray:
    """Weighted RMSE of the forecast age-specific male − female difference.

    Paper: eq:ev-schedule-rmse

    ``D_x = logit q_x(male) − logit q_x(female)``; for each schedule
    ``sqrt(Σ_x w_x (D_fc − D_obs)² / Σ_x w_x)`` with ``w_x`` the average of
    the two sexes' observed weights: survivorship l_x (as in
    :func:`mdmx.forecast.schedule_errors`, ages few reach count little) or
    life-table deaths d_x.

    Parameters
    ----------
    z_fc, z_obs : (N, 2, A) forecast and observed logit(q_x), female then male
    weights : ``"survivors"`` (l_x) or ``"deaths"`` (d_x)

    Returns
    -------
    (N,) in logit units.
    """
    wx = fc._age_weights(z_obs, weights)                                   # (N, 2, A)
    w = 0.5 * (wx[:, 0] + wx[:, 1])
    d_err = (z_fc[:, 1] - z_fc[:, 0]) - (z_obs[:, 1] - z_obs[:, 0])
    return np.sqrt(np.sum(w * d_err ** 2, axis=-1) / np.sum(w, axis=-1))


def smooth_sex_difference(z: np.ndarray, ages: int) -> np.ndarray:
    """Schedules with their male − female difference smoothed over age, the average of the sexes unchanged.

    Paper: sec:evaluation:overall

    ``D_x = z_male,x − z_female,x`` is replaced by its centred moving average over ``ages`` single ages
    (the window shrinks at the youngest and oldest ages); the sexes become ``avg ∓ D̃/2``.  A difference
    that is linear in age is kept as it is, except at the edges, where the window is one-sided.

    Parameters
    ----------
    z : (..., 2, A) logit(q_x), female then male; ages : the window, an odd number of ages ≥ 1

    Returns
    -------
    (..., 2, A) logit(q_x).
    """
    if ages < 1 or ages % 2 == 0:
        raise ValueError(f"ages must be an odd number ≥ 1, not {ages}")
    z = np.asarray(z, dtype=float)
    diff = z[..., 1, :] - z[..., 0, :]
    avg = 0.5 * (z[..., 0, :] + z[..., 1, :])
    n, h = diff.shape[-1], ages // 2
    csum = np.concatenate([np.zeros(diff.shape[:-1] + (1,)), np.cumsum(diff, axis=-1)], axis=-1)
    lo = np.maximum(np.arange(n) - h, 0)
    hi = np.minimum(np.arange(n) + h + 1, n)
    smooth = (csum[..., hi] - csum[..., lo]) / (hi - lo)
    return np.stack([avg - smooth / 2, avg + smooth / 2], axis=-2)


def sex_difference_by_age(z_fc: np.ndarray, z_obs: np.ndarray, bands: Sequence[tuple[int, int]],
                          weights: str = "deaths") -> pd.DataFrame:
    """The error of the forecast age-specific male − female difference, by band of ages.

    Paper: eq:ev-schedule-rmse

    For each band ``[a, b)``: the mean error of ``D_x`` (forecast − observed, logit units) over the
    points and the band's ages, its root mean square, and the band's share of the weighted mean squared
    error that :func:`sex_difference_errors` takes the root of (averaged over the points).

    Parameters
    ----------
    z_fc, z_obs : (N, 2, A) forecast and observed logit(q_x), female then male
    bands : (first age, last age + 1) of each band; weights : ``"deaths"`` or ``"survivors"``

    Returns
    -------
    DataFrame indexed by ``"a–(b−1)"`` with ``mean_error``, ``rmse`` and ``share``.
    """
    w = fc._age_weights(z_obs, weights)
    w = 0.5 * (w[:, 0] + w[:, 1])
    err = (z_fc[:, 1] - z_fc[:, 0]) - (z_obs[:, 1] - z_obs[:, 0])
    total = np.sum(w * err ** 2, axis=1) / np.sum(w, axis=1)
    rows = {}
    for a, b in bands:
        part = np.sum(w[:, a:b] * err[:, a:b] ** 2, axis=1) / np.sum(w, axis=1)
        rows[f"{a}–{b - 1}" if b - a > 1 else f"{a}"] = {
            "mean_error": float(err[:, a:b].mean()), "rmse": float(np.sqrt((err[:, a:b] ** 2).mean())),
            "share": float(part.mean() / total.mean())}
    return pd.DataFrame(rows).T


def sex_difference_slope(L: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """Long-run annual change of MDMx's male − female logit(q_x) difference.

    Paper: sec:forecasting:sexdiff

    The forecast drift of the PCA scores reverts to the drift target δ
    (:func:`mdmx.forecast.drift_target`), so far ahead the scores grow by δ a year and the schedules by
    ``L δ`` (:func:`mdmx.forecast.scores_to_logit`); the male − female
    difference by ``(L_male − L_female) δ``.

    Parameters
    ----------
    L : (2, A, P) schedules of the PCA loadings (the first p are used)
    drift : (..., p) drift targets (per year)

    Returns
    -------
    (..., A) logit units per year.
    """
    drift = np.asarray(drift, dtype=float)
    p = drift.shape[-1]
    return np.einsum("...j,aj->...a", drift, L[1, :, :p] - L[0, :, :p])


def group_means(x: np.ndarray, groups: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Mean over the ages of each age group: (..., A) → (..., G), in the order of ``groups``."""
    return np.stack([np.asarray(x)[..., list(ages)].mean(axis=-1) for ages in groups.values()], axis=-1)


def observed_sex_difference_change(M: np.ndarray, observed: np.ndarray, years: Sequence[int], c_index: Sequence[int],
                                   groups: Mapping[str, Sequence[int]], *, lag: int, window: int) -> np.ndarray:
    """Observed change of the male − female logit(q_x) difference over ``lag`` years, by age group.

    Paper: sec:forecasting:sexdiff

    For each population: the mean difference over its observed years in the
    last ``window`` calendar years up to its last observed year, minus the
    mean over the observed years in the ``window`` years ending ``lag``
    years earlier.  Populations without observed years in both are left out.

    Parameters
    ----------
    M : (2, A, C, T) logit(q_x); observed : (C, T) mask; years : (T,) labels
    c_index : the populations (indices on the C axis)

    Returns
    -------
    (n, G) changes in logit units, one row per population with data.
    """
    years = np.asarray(years)
    out = []
    for c in c_index:
        t_obs = np.flatnonzero(observed[c])
        last = years[t_obs[-1]]
        recent = t_obs[years[t_obs] > last - window]
        past = t_obs[(years[t_obs] <= last - lag) & (years[t_obs] > last - lag - window)]
        if recent.size and past.size:
            d = M[1, :, c, :] - M[0, :, c, :]                                  # (A, T)
            out.append(group_means(d[:, recent].mean(axis=1) - d[:, past].mean(axis=1), groups))
    return np.stack(out)


def quantiles_from_intervals(frame: pd.DataFrame, pi_levels: Sequence[int], median: str = "e0_fc") -> pd.DataFrame:
    """Quantile columns from a method's central intervals and median.

    ``lower{L}``/``upper{L}`` (central L % intervals) and the median column
    become the quantile columns of :func:`quantile_columns`
    (:func:`quantile_levels` of ``pi_levels``); other columns are kept.
    """
    levels = quantile_levels(pi_levels)
    names = dict(zip(np.round(levels, 6), quantile_columns(levels), strict=True))
    out = frame.copy()
    out[names[0.5]] = frame[median]
    for lev in pi_levels:
        lo = round((100 - lev) / 200, 6)
        out[names[lo]], out[names[round(1 - lo, 6)]] = frame[f"lower{lev}"], frame[f"upper{lev}"]
    return out


def comonotone_both(q: pd.DataFrame, qcols: Sequence[str]) -> pd.DataFrame:
    """Quantiles of the average of the sexes' e₀ when their errors move together.

    For a method that forecasts the sexes separately, the average of the
    female and male quantiles at each level is the quantile of the average if
    the two are comonotone (perfectly dependent).

    Parameters
    ----------
    q : rows ``method, pop, origin, sex, year`` and ``qcols``, sex ``female``
        and ``male`` (exactly one row each)

    Returns
    -------
    The same columns with sex ``both``.
    """
    key = ["method", "pop", "origin", "year"]
    sexes = q[q["sex"].isin(["female", "male"])]
    counts = sexes.groupby(key)["sex"].nunique()
    if (counts != 2).any() or sexes.duplicated(key + ["sex"]).any():
        raise ValueError("comonotone_both needs exactly one female and one male row per point")
    return sexes.groupby(key)[list(qcols)].mean().reset_index().assign(sex="both")[q.columns]


# ══════════════════════════════════════════════════════════════════════════════
# Families of measures and the weight-robust ranking (SMAA)
# ══════════════════════════════════════════════════════════════════════════════

def point_values(e0: pd.DataFrame, quantiles: pd.DataFrame, levels: Sequence[float],
                 schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """One method's measure values at each point, for :func:`overall`.

    Parameters
    ----------
    e0 : ``pop, origin, year, sex, e0_fc, e0_obs`` with sex ``female``,
        ``male`` and ``both`` (the point forecasts and the truth)
    quantiles : ``pop, origin, year, sex`` and the quantile columns
        (:func:`quantile_columns` of ``levels``) for the same sexes
    schedules : ``pop, origin, year`` and the point's schedule measures:
        each ``rmse_<name>`` column becomes ``rmse:<name>`` (e.g. the
        schedule and age-specific sex-difference errors) and each
        ``err_<q>`` column (a signed error, e.g. of e₆₅) becomes ``mae:<q>``
        and ``absbias:<q>``; for the methods that forecast schedules

    Returns
    -------
    DataFrame ``pop, origin, year`` and the columns ``mae:q`` (|error|) and
    ``absbias:q`` (error) for q = both, female, male, gap (female − male),
    ``wis:q`` for q = female, male, both, and the schedule measures when
    ``schedules`` is given.
    """
    key = ["pop", "origin", "year"]
    wide = e0.pivot(index=key, columns="sex", values=["e0_fc", "e0_obs"])
    err = {sex: wide[("e0_fc", sex)] - wide[("e0_obs", sex)] for sex in fc.SEXES}
    err["gap"] = err["female"] - err["male"]
    out = pd.DataFrame(index=wide.index)
    for q in ("both", "female", "male", "gap"):
        out[f"mae:{q}"] = err[q].abs()
        out[f"absbias:{q}"] = err[q]
    qcols = quantile_columns(levels)
    scored = quantiles.merge(e0[key + ["sex", "e0_obs"]], on=key + ["sex"], how="inner", validate="one_to_one")
    scored["wis"] = wis_from_quantiles(scored["e0_obs"].to_numpy(), scored[qcols].to_numpy(dtype=float), levels)
    wis = scored.pivot(index=key, columns="sex", values="wis")
    for q in ("female", "male", "both"):
        out[f"wis:{q}"] = wis[q].reindex(out.index)
    if schedules is not None:
        cols = [c for c in schedules.columns if c.startswith(("rmse_", "err_"))]
        sched = schedules.set_index(key)[cols].reindex(out.index)
        for c in cols:
            if c.startswith("rmse_"):
                out[f"rmse:{c[5:]}"] = sched[c]
            else:
                out[f"mae:{c[4:]}"], out[f"absbias:{c[4:]}"] = sched[c].abs(), sched[c]
    return out.reset_index()


def _split(code: str) -> tuple[str, str]:
    kind, quantity = code.split(":")
    if kind not in ("mae", "absbias", "cellbias", "rmse", "wis"):
        raise ValueError(f"unknown measure kind in {code!r}")
    return kind, quantity


def overall(values: pd.DataFrame, families: Mapping[str, Sequence[str]], composite: Sequence[str], *,
            methods: Sequence[str], naive: str, draws: int, seed: int, bootstrap: bool = True) -> dict:
    """Scaled family losses and the weight-robust ranking of the methods (SMAA).

    Paper: eq:ev-smaa

    Each measure is a mean over the points: ``mae:q`` of |error|,
    ``absbias:q`` the absolute value of the mean error, ``rmse:…`` and
    ``wis:…`` of the points' values.  ``cellbias:q`` is the bias without
    cancellation: the mean error within each cell (``values["cell"]``, e.g.
    an origin and horizon band), then the mean of the cells' absolute values,
    each cell counting once (a cell with no points in a bootstrap sample is
    left out).  It reads the signed errors of ``absbias:q``.  A measure is
    divided by the naive's MAE of the same quantity (``mae``, ``absbias``,
    ``cellbias``) or by the naive's own value (``rmse``, ``wis``); a family's
    loss is the mean of its scaled measures;
    a method's composite loss for weights w is ``Σ_f w_f L_f``.  Each draw
    takes w from Dirichlet(1, …, 1) and — with ``bootstrap`` — resamples the
    clusters (population families) with replacement, recomputing every
    measure (the naive's included).

    Parameters
    ----------
    values : one row per method and point, columns ``method, point,
        cluster`` and one per measure code (``mae:q`` → |error|,
        ``absbias:q`` → error, ``rmse:…``/``wis:…`` → the value), ``cell``
        if any ``cellbias`` code is used, and optionally ``weight`` (each
        point's weight in every mean; default 1); every method must have the
        same points
    families : family → measure codes; composite : the families used
    methods : the methods ranked (in this order); naive : the yardstick
        (in ``values``, not ranked)
    draws, seed : number of draws and their seed

    Returns
    -------
    dict with ``losses`` (DataFrame method × family, full sample, naive
    included), ``scaled`` (DataFrame method × measure), and over the draws:
    ``first`` (rank-1 share, ties shared), ``mean_rank``, ``beats_naive``
    (share of draws with a composite loss below the naive's) — Series by
    method — ``central_weights`` (DataFrame method × family: mean weights of
    the draws in which the method ranks first), ``pairwise`` (DataFrame,
    P(loss of row < loss of column)) and ``rank_acceptability`` (DataFrame
    method × rank).
    """
    codes = [c for f in composite for c in families[f]]
    quantities = {_split(c)[1] for c in codes if _split(c)[0] in ("mae", "absbias", "cellbias")}
    needed = list(dict.fromkeys(codes + [f"mae:{q}" for q in sorted(quantities)]))
    summed = [c for c in needed if _split(c)[0] != "cellbias"]
    cell_codes = [c for c in needed if _split(c)[0] == "cellbias"]
    if cell_codes and "cell" not in values:
        raise ValueError("cellbias measures need a 'cell' column in values")
    all_methods = list(methods) + [naive]
    clusters = np.unique(values["cluster"])
    c_index = {c: i for i, c in enumerate(clusters)}
    points = None
    sums = np.zeros((len(all_methods), len(needed), len(clusters)))
    counts = np.zeros(len(clusters))
    cells = np.unique(values["cell"]) if cell_codes else np.array([])
    g_index = {g: j for j, g in enumerate(cells)}
    cell_sums = np.zeros((len(cell_codes), len(all_methods), len(clusters), len(cells)))
    cell_counts = np.zeros((len(clusters), len(cells)))
    for i, m in enumerate(all_methods):
        v = values[values["method"] == m].sort_values("point")
        ci = v["cluster"].map(c_index).to_numpy()
        wt = v["weight"].to_numpy(dtype=float) if "weight" in v else np.ones(len(v))
        if points is None:
            points = v["point"].to_numpy()
            np.add.at(counts, ci, wt)
            if cell_codes:
                np.add.at(cell_counts, (ci, v["cell"].map(g_index).to_numpy()), wt)
        elif not np.array_equal(points, v["point"].to_numpy()):
            raise ValueError(f"method {m} is not scored on the same points as {all_methods[0]}")
        data = v[summed].to_numpy(dtype=float)
        for k, code in enumerate(summed):
            col = np.abs(data[:, k]) if _split(code)[0] == "mae" else data[:, k]
            if not np.all(np.isfinite(col)):
                raise ValueError(f"method {m} has non-finite values of {code}")
            np.add.at(sums[i, needed.index(code)], ci, wt * col)
        for j, code in enumerate(cell_codes):
            err = v[f"absbias:{_split(code)[1]}"].to_numpy(dtype=float)
            if not np.all(np.isfinite(err)):
                raise ValueError(f"method {m} has non-finite values of absbias:{_split(code)[1]}")
            np.add.at(cell_sums[j, i], (ci, v["cell"].map(g_index).to_numpy()), wt * err)

    fam_index = [[needed.index(c) for c in families[f]] for f in composite]
    kinds = [_split(c)[0] for c in needed]
    denom_index = [needed.index(f"mae:{_split(c)[1]}") if _split(c)[0] in ("mae", "absbias", "cellbias")
                   else needed.index(c) for c in needed]
    cell_pos = [needed.index(c) for c in cell_codes]
    naive_i = len(all_methods) - 1

    def losses(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled measures (D, methods, codes) and family losses (D, methods, families) for counts b (D, C)."""
        vals = np.einsum("dc,mkc->dmk", b, sums) / (b @ counts)[:, None, None]
        vals = np.where(np.array(kinds) == "absbias", np.abs(vals), vals)
        if cell_codes:
            n_cell = b @ cell_counts                                        # (D, G) points per cell
            present = n_cell > 0
            for j, k in enumerate(cell_pos):
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_cell = np.einsum("dc,mcg->dmg", b, cell_sums[j]) / n_cell[:, None, :]
                vals[:, :, k] = (np.where(present[:, None, :], np.abs(mean_cell), 0.0).sum(axis=-1)
                                 / present.sum(axis=-1)[:, None])
        scaled = vals / vals[:, naive_i, :][:, None, denom_index]
        fam = np.stack([scaled[..., idx].mean(axis=-1) for idx in fam_index], axis=-1)
        return scaled, fam

    scaled_full, fam_full = losses(np.ones((1, len(clusters))))
    rng = np.random.default_rng(seed)
    b = (rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=draws).astype(float)
         if bootstrap else np.ones((draws, len(clusters))))
    w = rng.dirichlet(np.ones(len(composite)), size=draws)                  # (D, F)
    _, fam = losses(b)
    score = np.einsum("df,dmf->dm", w, fam)                                 # (D, methods + naive)
    ranked = score[:, :-1]
    best = ranked == ranked.min(axis=1, keepdims=True)
    share = best / best.sum(axis=1, keepdims=True)                          # ties share the first place
    ranks = rankdata(ranked, axis=1, method="average")
    # Methods tied over ranks lo … hi share each of those ranks equally.
    lo, hi = rankdata(ranked, axis=1, method="min"), rankdata(ranked, axis=1, method="max")
    r_all = np.arange(1, len(methods) + 1)
    acceptability = (((lo[..., None] <= r_all) & (hi[..., None] >= r_all)) / (hi - lo + 1)[..., None]).mean(axis=0)
    with np.errstate(invalid="ignore"):
        central = (share.T @ w) / share.sum(axis=0)[:, None]
    pairwise = (ranked[:, :, None] < ranked[:, None, :]).mean(axis=0)
    fam_names = list(composite)
    return {
        "losses": pd.DataFrame(fam_full[0], index=all_methods, columns=fam_names),
        "scaled": pd.DataFrame(scaled_full[0], index=all_methods, columns=needed),
        "first": pd.Series(share.mean(axis=0), index=list(methods)),
        "mean_rank": pd.Series(ranks.mean(axis=0), index=list(methods)),
        "beats_naive": pd.Series((ranked < score[:, -1:]).mean(axis=0), index=list(methods)),
        "central_weights": pd.DataFrame(central, index=list(methods), columns=fam_names),
        "pairwise": pd.DataFrame(pairwise, index=list(methods), columns=list(methods)),
        "rank_acceptability": pd.DataFrame(acceptability, index=list(methods),
                                           columns=range(1, len(methods) + 1)),
    }
