"""Schedules at any level of mortality: the typical schedule plus a regime's deviation (paper §7).

What this module does
---------------------
Paper §7 generates a complete sex–age schedule of mortality from a regime
``k`` and a target life expectancy ``x`` (the article's level ξ).  A regime
is a deviation pattern from the typical schedule at the same e₀ (chapter 06),
so the generator is

    core_k(x) = Ḡ(x) + D_k(x),        z_k(x) = S core_k(x) Aᵀ   (S, A) logit(q_x)

* ``Ḡ(x)`` — the e₀-conditional mean of the effective core, a LOWESS of
  every core entry against e₀ (:func:`mdmx.clustering.conditional_mean`):
  the typical schedule of every population together;
* ``D_k(x)`` — the deviation of regime ``k``, non-zero only at its
  *positions*: the level-free feature positions of chapter 06 (sex
  components × age columns 2…k) and, optionally, the sex contrast of the
  level column (:func:`deviation_positions`).  Three kinds are compared
  (:func:`deviation_grids`,
  :func:`neural_deviation_grids`): ``"constant"`` (the regime's mean
  deviation), ``"lowess"`` (a LOWESS of the members' features against e₀,
  so the deviation changes with the level of mortality) and ``"neural"``
  (a small network from regime and e₀ to the features).  Outside a
  regime's central e₀ range (its *support*) a deviation is held at its value
  at the boundary.

A last step shifts the level coordinate ``G[0,0]`` so that the schedule's
sex-average e₀ equals ``x`` exactly (:func:`calibrate_level`); the shape at
``x`` is kept.  :func:`holdout_scores` compares the kinds by rebuilding the
schedules of held-out population families from their regime and e₀, and
:func:`invert` finds the regime and e₀ that best reproduce a schedule
(chapter 09 builds on it).

A *model* (:func:`build_model`) is a dict of grids, so that it can be stored
as an artefact: ``e0_grid`` (g,), ``typical`` (g, r₁, r₂), ``deviation``
(K, g, p) on the positions ``feature_index`` (p, 2) — chapter 06's
features, plus the sex level if the chosen candidate uses it, so *not*
always the same as ``clustering["features"]["index"]`` — ``support``
(K, 2), ``clusters`` (the regimes it serves), ``U_sex`` (S, r₁) and
``U_age`` (A, r₂).

Conventions
-----------
* Index 0 of the sex axis is female, 1 male; e₀ is the average of the
  female and male e₀ (as in chapter 06).
* Arrays of cores are ``(n, r₁, r₂)``; schedules ``(n, S, A)`` logit(q_x).
* Regimes are chapter 06's cluster labels ``0 … K−1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import clustering as clus
from . import forecast as fc
from . import lifetable
from . import nn as mnn

__all__ = [
    "schedules", "sex_e0", "calibrate_level", "deviation_grids", "neural_deviation_grids",
    "deviation_positions", "estimate_deviation", "build_model", "trajectory_cores", "generate",
    "uncalibrated_e0_gap", "in_support", "invert", "holdout_scores",
    "paired_rmse_difference", "age_contributions",
]


# ══════════════════════════════════════════════════════════════════════════════
# Schedules, e₀ and the level calibration
# ══════════════════════════════════════════════════════════════════════════════

def schedules(cores: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray) -> np.ndarray:
    """Logit(q_x) schedules (n, S, A) from effective cores (n, r₁, r₂): ``S G Aᵀ``."""
    return np.einsum("sp,npq,aq->nsa", U_sex, np.asarray(cores, dtype=float), U_age)


def sex_e0(z: np.ndarray) -> np.ndarray:
    """e₀ of each sex (..., S) of logit(q_x) schedules (..., S, A) (:func:`mdmx.lifetable.e0_from_logit`)."""
    return lifetable.e0_from_logit(z)


def calibrate_level(cores: np.ndarray, targets: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray, *,
                    bracket: float, iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Shift the level coordinate ``G[0,0]`` of each core so that its sex-average e₀ hits the target.

    Paper: sec:reconstruction:trajectory_eval

    The level coordinate moves logit(q_x) in the same direction at (almost)
    every age at once, so e₀ moves with it; the other coordinates, which
    carry the shape, are left alone.  The shift is found by bisection in
    ``[−bracket, +bracket]``, for all cores at once; a target that the e₀ at
    the two ends of the bracket do not enclose is an error.

    Parameters
    ----------
    cores : (n, r₁, r₂) effective cores.
    targets : (n,) target sex-average e₀ (years).
    U_sex, U_age : (S, r₁) and (A, r₂) bases.
    bracket : largest shift tried, in level units.
    iterations : bisection steps (60 halve the bracket to below 1e-15 of it).

    Returns
    -------
    calibrated : (n, r₁, r₂) cores; shift : (n,) the shift applied to ``G[0,0]``.

    Raises
    ------
    ValueError if a target cannot be bracketed (the e₀ at the two ends of the
    bracket lie on the same side of it).
    """
    cores = np.asarray(cores, dtype=float)
    targets = np.asarray(targets, dtype=float)

    def gap(shift: np.ndarray) -> np.ndarray:
        moved = cores.copy()
        moved[:, 0, 0] += shift
        return sex_e0(schedules(moved, U_sex, U_age)).mean(axis=-1) - targets

    lo, hi = np.full(len(cores), -bracket), np.full(len(cores), bracket)
    g_lo, g_hi = gap(lo), gap(hi)
    bad = np.flatnonzero(np.sign(g_lo) == np.sign(g_hi))
    if bad.size:
        raise ValueError(f"Cannot bracket the target e0 for {bad.size} core(s) (first: row {bad[0]}, "
                         f"target {targets[bad[0]]:.2f}); widen calibration_bracket.")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        g_mid = gap(mid)
        same = np.sign(g_mid) == np.sign(g_lo)
        lo, g_lo = np.where(same, mid, lo), np.where(same, g_mid, g_lo)
        hi = np.where(same, hi, mid)
    shift = 0.5 * (lo + hi)
    calibrated = cores.copy()
    calibrated[:, 0, 0] += shift
    return calibrated, shift


# ══════════════════════════════════════════════════════════════════════════════
# The deviations
# ══════════════════════════════════════════════════════════════════════════════

def _support(e0: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    return np.quantile(e0, quantiles)


def deviation_grids(R: np.ndarray, e0: np.ndarray, labels: np.ndarray, clusters: Sequence[int],
                    grid: np.ndarray, *, kind: str, frac: float | None = None, iterations: int = 3,
                    delta_share: float = 0.01, support_quantiles: Sequence[float] = (0.05, 0.95)
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Each regime's deviation from the typical schedule on the e₀ grid.

    Paper: eq:trajectory_smooth

    Parameters
    ----------
    R : (n, p) deviations of the observed country-years from the typical
        schedule at their positions (core entries minus the typical core's;
        the level-free features of chapter 06, plus the sex level if used).
    e0 : (n,) sex-average e₀ of the observed schedules.
    labels : (n,) regime of each country-year.
    clusters : the regimes to estimate.
    grid : (g,) increasing e₀ grid.
    kind : ``"constant"`` — the regime's mean feature vector at every e₀;
        ``"lowess"`` — a LOWESS of each feature against e₀ over the regime's
        members (span ``frac`` of them, ``iterations`` robustifying
        iterations; :func:`mdmx.clustering.conditional_mean`).
    support_quantiles : the regime's support is this central range of its
        members' e₀; outside it a ``"lowess"`` deviation is held at its
        boundary value (the smoother has too few points there to follow).

    Returns
    -------
    deviation : (len(clusters), g, p); support : (len(clusters), 2) e₀ range.
    """
    out = np.empty((len(clusters), len(grid), R.shape[1]))
    support = np.empty((len(clusters), 2))
    for i, k in enumerate(clusters):
        m = labels == k
        if m.sum() < 2:
            raise ValueError(f"Regime {k} has {int(m.sum())} member(s); a deviation needs at least 2.")
        support[i] = _support(e0[m], support_quantiles)
        if kind == "constant":
            out[i] = R[m].mean(axis=0)
        elif kind == "lowess":
            if frac is None:
                raise ValueError("A LOWESS deviation needs a span `frac`.")
            # Evaluated on the grid clipped to the support: constant beyond it.
            _, out[i] = clus.conditional_mean(e0[m], R[m], frac=frac, iterations=iterations,
                                              delta_share=delta_share, grid=np.clip(grid, *support[i]))
        else:
            raise ValueError(f"Unknown deviation kind {kind!r}.")
    return out, support


def _e0_inputs(e0: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Polynomial features (n, 3) of e₀ scaled to [0, 1] on the training range."""
    t = (np.asarray(e0, dtype=float) - lo) / (hi - lo)
    return np.column_stack([t, t ** 2, t ** 3])


def neural_deviation_grids(R: np.ndarray, e0: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                           clusters: Sequence[int], grid: np.ndarray, *, hidden: Sequence[int], lr: float,
                           epochs: int, batch_size: int, l2: float, val_frac: float, patience: int,
                           seed: int, support_quantiles: Sequence[float] = (0.05, 0.95)
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Each regime's deviation from a neural network of (regime, e₀), on the e₀ grid.

    One network for all regimes (:func:`mdmx.nn.fit_mlp`): inputs are the
    regime as a one-hot vector and three polynomial terms of e₀ (scaled to
    [0, 1] on the training range), outputs the ``p`` features; inputs and
    outputs are standardised on the training rows.  Early stopping holds out
    whole ``groups`` (population families).  Outside a regime's support the
    deviation is held at its boundary value, as for :func:`deviation_grids`.

    Parameters as for :func:`deviation_grids`, plus ``groups`` (n,) and the
    training settings of :func:`mdmx.nn.fit_mlp`.

    Returns
    -------
    deviation : (len(clusters), g, p); support : (len(clusters), 2).
    """
    clusters = list(clusters)
    m = np.isin(labels, clusters)
    onehot = (labels[m, None] == np.asarray(clusters)[None]).astype(float)
    lo, hi = float(e0[m].min()), float(e0[m].max())
    X = np.column_stack([onehot, _e0_inputs(e0[m], lo, hi)])
    sx, sy = mnn.Standardizer.fit(X), mnn.Standardizer.fit(R[m])
    net, _ = mnn.fit_mlp(sx.transform(X), sy.transform(R[m]), hidden, lr=lr, epochs=epochs,
                         batch_size=batch_size, l2=l2, seed=seed, val_frac=val_frac, patience=patience,
                         groups=groups[m])
    out = np.empty((len(clusters), len(grid), R.shape[1]))
    support = np.empty((len(clusters), 2))
    for i, k in enumerate(clusters):
        support[i] = _support(e0[labels == k], support_quantiles)
        x = np.clip(grid, *support[i])
        Xg = np.column_stack([np.tile(np.eye(len(clusters))[i], (len(grid), 1)), _e0_inputs(x, lo, hi)])
        out[i] = sy.inverse_transform(net.predict(sx.transform(Xg)))
    return out, support


def deviation_positions(feature_index: np.ndarray, r1: int, *, sex_level: bool) -> np.ndarray:
    """The core positions a deviation covers: chapter 06's features, optionally with the sex level.

    Paper: eq:trajectory

    Chapter 06's regimes are built without the level column, so a deviation
    on its feature positions alone cannot shift how far apart male and female
    mortality sit overall.  ``sex_level`` adds the level column's other sex
    components — 0-based positions (i, 0) for i = 1 … r₁ − 1, a difference
    between the sexes, not a level — so that a regime keeps its own overall
    sex differential.

    Parameters
    ----------
    feature_index : (p, 2) 0-based (sex component, age column) of chapter 06's features.
    r1 : number of sex components.
    sex_level : add the level column's sex contrasts.

    Returns
    -------
    (p′, 2) positions: ``feature_index`` followed by the added ones (none
    added twice).
    """
    feature_index = np.asarray(feature_index, dtype=int)
    if not sex_level:
        return feature_index
    extra = [(i, 0) for i in range(1, r1) if not ((feature_index[:, 0] == i) & (feature_index[:, 1] == 0)).any()]
    return np.vstack([feature_index, np.array(extra, dtype=int).reshape(-1, 2)])


def estimate_deviation(R: np.ndarray, e0: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       clusters: Sequence[int], grid: np.ndarray, spec: Mapping, *, iterations: int,
                       delta_share: float, support_quantiles: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """The deviations of one candidate (``spec``), on the e₀ grid — the one place that dispatches by kind.

    ``spec["kind"]`` is ``"typical"`` (no deviation: zeros, support the whole
    grid), ``"constant"`` or ``"lowess"`` (:func:`deviation_grids`, with
    ``spec["frac"]``), or ``"neural"`` (:func:`neural_deviation_grids`, with the
    training settings in ``spec``); other keys (``sex_level``) are ignored here.

    Parameters
    ----------
    R : (n, p) deviations from the typical schedule at the candidate's positions.
    e0 : (n,) sex-average e₀; labels : (n,) regimes; groups : (n,) families
        (used by the neural candidate's early stopping).
    clusters : the regimes to estimate; grid : (g,) e₀ grid.
    iterations, delta_share : LOWESS settings; support_quantiles : see
        :func:`deviation_grids`.

    Returns
    -------
    deviation : (len(clusters), g, p); support : (len(clusters), 2).
    """
    kind = spec["kind"]
    if kind == "typical":
        return (np.zeros((len(clusters), len(grid), R.shape[1])),
                np.tile([grid[0], grid[-1]], (len(clusters), 1)).astype(float))
    if kind == "neural":
        settings = {k: v for k, v in spec.items() if k not in ("kind", "sex_level")}
        return neural_deviation_grids(R, e0, labels, groups, clusters, grid, support_quantiles=support_quantiles,
                                      **settings)
    return deviation_grids(R, e0, labels, clusters, grid, kind=kind, frac=spec.get("frac"), iterations=iterations,
                           delta_share=delta_share, support_quantiles=support_quantiles)


# ══════════════════════════════════════════════════════════════════════════════
# The model: generating and inverting schedules
# ══════════════════════════════════════════════════════════════════════════════

def build_model(e0_grid: np.ndarray, typical: np.ndarray, deviation: np.ndarray, support: np.ndarray,
                clusters: Sequence[int], feature_index: np.ndarray, U_sex: np.ndarray,
                U_age: np.ndarray, kind: str) -> dict:
    """Assemble a model (see the module docstring) from its grids."""
    if deviation.shape[:2] != (len(clusters), len(e0_grid)) or typical.shape[0] != len(e0_grid):
        raise ValueError("The deviation and typical grids do not match the regimes and the e0 grid.")
    return {"kind": kind, "e0_grid": np.asarray(e0_grid, dtype=float), "typical": typical,
            "deviation": deviation, "support": support, "clusters": np.asarray(clusters, dtype=int),
            "feature_index": np.asarray(feature_index), "U_sex": U_sex, "U_age": U_age}


def _regime_row(model: Mapping, k: int) -> int:
    rows = np.flatnonzero(model["clusters"] == k)
    if rows.size == 0:
        raise ValueError(f"Regime {k} is not in this model (regimes {model['clusters'].tolist()}).")
    return int(rows[0])


def trajectory_cores(model: Mapping, k: int | None, x: np.ndarray | float) -> np.ndarray:
    """Uncalibrated cores ``Ḡ(x) + D_k(x)`` (q, r₁, r₂) at e₀ values ``x``; ``k=None``: the typical schedule.

    Paper: *eq:trajectory
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    grid = model["e0_grid"]
    cores = clus.conditional_core_at(x, grid, model["typical"])
    if k is not None:
        dev = clus.conditional_core_at(x, grid, model["deviation"][_regime_row(model, k)])   # (q, p)
        r1, r2 = model["typical"].shape[1:]
        cores = cores + clus.features_to_cores(dev, model["feature_index"], r1, r2)
    return cores


def generate(model: Mapping, k: int | None, x: np.ndarray | float, *, bracket: float) -> np.ndarray:
    """Calibrated schedules (q, S, A) logit(q_x) of regime ``k`` (``None``: typical) at e₀ values ``x``.

    Paper: eq:reconstruct_trajectory

    Each schedule's sex-average e₀ equals its ``x`` (:func:`calibrate_level`).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    cores, _ = calibrate_level(trajectory_cores(model, k, x), x, model["U_sex"], model["U_age"], bracket=bracket)
    return schedules(cores, model["U_sex"], model["U_age"])


def uncalibrated_e0_gap(model: Mapping, k: int | None, x: np.ndarray | float) -> np.ndarray:
    """Sex-average e₀ of ``Ḡ(x) + D_k(x)`` minus ``x``, before the level calibration (q,)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = schedules(trajectory_cores(model, k, x), model["U_sex"], model["U_age"])
    return sex_e0(z).mean(axis=-1) - x


def in_support(model: Mapping, k: int, x: np.ndarray | float) -> np.ndarray:
    """True where e₀ ``x`` (scalar or (q,)) lies in regime ``k``'s support, (q,) bool.

    Outside the support the deviation is held at its boundary value, so a
    schedule generated there extrapolates the regime.
    """
    lo, hi = model["support"][_regime_row(model, k)]
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return (x >= lo) & (x <= hi)


def invert(z: np.ndarray, model: Mapping, *, bracket: float, step: float = 1.0) -> dict:
    """The regime and e₀ whose generated schedule is closest to a given schedule.

    Paper: sec:results:inversion, eq:forward

    For each regime of the model, the e₀ that minimises the root-mean-square
    difference in logit(q_x) between ``z`` and the calibrated generated
    schedule is found by a search on a grid of ``step`` years over the e₀
    grid, refined by a bounded scalar minimisation; the regime with the
    smallest such difference wins.

    Parameters
    ----------
    z : (S, A) logit(q_x) schedule.
    model : a model (:func:`build_model`).

    Returns
    -------
    dict with ``regime``, ``e0``, ``rmse`` and ``by_regime`` (a DataFrame with
    each regime's best e₀ and RMSE).
    """
    z = np.asarray(z, dtype=float)
    lo, hi = model["e0_grid"][0], model["e0_grid"][-1]
    coarse = np.arange(lo, hi + 1e-9, step)
    rows = []
    for k in model["clusters"]:
        def rmse(x: float, k: int = int(k)) -> float:
            return float(np.sqrt(np.mean((generate(model, k, x, bracket=bracket)[0] - z) ** 2)))
        errs = np.sqrt(np.mean((generate(model, int(k), coarse, bracket=bracket) - z) ** 2, axis=(1, 2)))
        i = int(np.argmin(errs))
        res = minimize_scalar(rmse, bounds=(coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)]),
                              method="bounded", options={"xatol": 1e-4})
        rows.append({"regime": int(k), "e0": float(res.x), "rmse": float(res.fun)})
    by_regime = pd.DataFrame(rows)
    best = by_regime.loc[by_regime["rmse"].idxmin()]
    return {"regime": int(best["regime"]), "e0": float(best["e0"]), "rmse": float(best["rmse"]),
            "by_regime": by_regime}


# ══════════════════════════════════════════════════════════════════════════════
# Choosing the deviation by a hold-out over population families
# ══════════════════════════════════════════════════════════════════════════════

def holdout_scores(G_obs: np.ndarray, z_obs: np.ndarray, e0: np.ndarray, labels: np.ndarray,
                   groups: np.ndarray, folds: np.ndarray, clusters: Sequence[int],
                   candidates: Mapping[str, Mapping], *, feature_index: np.ndarray, grid: np.ndarray,
                   typical: Mapping, U_sex: np.ndarray, U_age: np.ndarray, bracket: float,
                   support_quantiles: Sequence[float], generate_labels: np.ndarray | None = None) -> pd.DataFrame:
    """Rebuild every held-out schedule from its regime and e₀, for each candidate deviation.

    Paper: sec:reconstruction:holdout

    For each fold the typical schedule (a LOWESS of every core entry against
    e₀, settings ``typical``) and each candidate's deviations are estimated
    from the other folds only.  Each held-out country-year's schedule is then
    generated at its observed e₀ with its regime, calibrated to that e₀, and
    compared with its observed schedule.  The regime labels themselves come
    from the clustering of all data (chapter 06): the hold-out asks how well
    a schedule is rebuilt *given* its regime.  With ``generate_labels`` a held-out schedule is rebuilt with
    that regime instead (e.g. :func:`nearest_regimes`, which does not look at the held-out schedule); the
    deviations are still estimated from the training folds' own labels.

    Parameters
    ----------
    G_obs : (n, r₁, r₂) effective cores; z_obs : (n, S, A) observed logit(q_x).
    e0 : (n,) sex-average e₀; labels : (n,) regimes; groups : (n,) families.
    folds : (n,) fold of each country-year (whole families per fold).
    clusters : regimes for which deviations are estimated (rows of other
        regimes are not scored).
    candidates : name → settings for :func:`estimate_deviation`, plus
        ``"sex_level": True`` to include the sex contrast of the level column
        in the deviation (:func:`deviation_positions`).
    feature_index : (p, 2) chapter 06's feature positions.
    typical : ``frac``, ``iterations``, ``delta_share`` of the typical schedule.
    generate_labels : (n,) the regime each country-year is rebuilt with when held out (default
        ``labels``); every value of a scored row must be one of ``clusters``.

    Returns
    -------
    DataFrame, one row per scored country-year and candidate: ``row`` (index
    into the inputs), ``candidate``, ``fold``, ``regime`` (the one it was
    rebuilt with), ``mse`` (mean squared logit difference over sexes and
    ages), ``e0_gap`` (e₀ before the calibration minus the observed e₀).
    """
    r1 = G_obs.shape[1]
    scored = np.isin(labels, clusters)
    gen = labels if generate_labels is None else np.asarray(generate_labels)
    if not np.isin(gen[scored], clusters).all():
        raise ValueError("generate_labels gives a scored row a regime outside clusters")
    out = []
    for f in np.unique(folds):
        train, test = folds != f, (folds == f) & scored
        if not test.any():
            continue
        fit_train, typ_grid = clus.conditional_mean(e0[train], G_obs[train], grid=grid, **typical)
        for name, spec in candidates.items():
            positions = deviation_positions(feature_index, r1, sex_level=spec.get("sex_level", False))
            R_train = (clus.cores_to_features(G_obs[train], positions)
                       - clus.cores_to_features(fit_train, positions))
            dev, support = estimate_deviation(R_train, e0[train], labels[train], groups[train], clusters, grid,
                                              spec, iterations=typical["iterations"],
                                              delta_share=typical["delta_share"],
                                              support_quantiles=support_quantiles)
            model = build_model(grid, typ_grid, dev, support, clusters, positions, U_sex, U_age, spec["kind"])
            for k in clusters:
                rows = np.flatnonzero(test & (gen == k))
                if rows.size == 0:
                    continue
                cores = trajectory_cores(model, int(k), e0[rows])
                gap = sex_e0(schedules(cores, U_sex, U_age)).mean(axis=-1) - e0[rows]
                calibrated, _ = calibrate_level(cores, e0[rows], U_sex, U_age, bracket=bracket)
                err = schedules(calibrated, U_sex, U_age) - z_obs[rows]
                out.append(pd.DataFrame({"row": rows, "candidate": name, "fold": int(f), "regime": int(k),
                                         "mse": np.mean(err ** 2, axis=(1, 2)), "e0_gap": gap}))
    return pd.concat(out, ignore_index=True)


def nearest_regimes(e0: np.ndarray, years: np.ndarray, labels: np.ndarray, folds: np.ndarray,
                    clusters: Sequence[int]) -> np.ndarray:
    """A regime for each country-year that does not look at its schedule: the nearest one's of the other folds.

    Paper: sec:reconstruction:holdout

    For each fold, each of its country-years gets the regime of the nearest country-year of the other
    folds, among those of ``clusters``, in (e₀, calendar year), each standardised by its standard
    deviation over all country-years.  The label-blind variant of the hold-out
    (:func:`holdout_scores` with ``generate_labels``): how much the regimes gain when the regime must be
    guessed from e₀ and the date alone.

    Parameters
    ----------
    e0, years, labels, folds : (n,) the sex-average e₀, the calendar year, the regime and the fold of
        each country-year (whole population families per fold)
    clusters : the regimes a country-year may be given

    Returns
    -------
    (n,) the assigned regimes.
    """
    from scipy.spatial import cKDTree

    e0, years = np.asarray(e0, dtype=float), np.asarray(years, dtype=float)
    z = np.column_stack([(e0 - e0.mean()) / e0.std(), (years - years.mean()) / years.std()])
    candidate = np.isin(labels, clusters)
    out = np.asarray(labels).copy()
    for f in np.unique(folds):
        test, train = folds == f, (folds != f) & candidate
        if not train.any():
            raise ValueError(f"fold {f}: no country-year of the other folds belongs to the regimes")
        _, idx = cKDTree(z[train]).query(z[test])
        out[test] = np.asarray(labels)[train][idx]
    return out


def paired_rmse_difference(mse_a: np.ndarray, mse_b: np.ndarray, groups: np.ndarray, *, n_resamples: int,
                           seed: int) -> dict:
    """Pooled RMSE under candidate ``a`` minus that under ``b``, with a bootstrap over groups.

    Paper: sec:reconstruction:holdout

    The rows are the same country-years scored under both candidates; whole
    groups (population families) are resampled with replacement, because
    the years of one population are not independent.

    Parameters
    ----------
    mse_a, mse_b : (n,) mean squared errors of the same rows.
    groups : (n,) group of each row.
    n_resamples, seed : bootstrap resamples and their seed.

    Returns
    -------
    dict with ``difference`` (negative: ``a`` is more accurate), ``lower``
    and ``upper`` (95 % percentile interval), and ``share_better`` (share of
    groups whose mean squared error is lower under ``a``).
    """
    _, inv = np.unique(groups, return_inverse=True)
    sum_a, sum_b, count = np.bincount(inv, mse_a), np.bincount(inv, mse_b), np.bincount(inv)
    draws = np.random.default_rng(seed).integers(0, len(count), (n_resamples, len(count)))
    n = count[draws].sum(axis=1)
    boot = np.sqrt(sum_a[draws].sum(axis=1) / n) - np.sqrt(sum_b[draws].sum(axis=1) / n)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(np.sqrt(sum_a.sum() / count.sum()) - np.sqrt(sum_b.sum() / count.sum())),
            "lower": float(lower), "upper": float(upper),
            "share_better": float(np.mean(sum_a / count < sum_b / count))}


# ══════════════════════════════════════════════════════════════════════════════
# Where the gains in e₀ come from along a trajectory
# ══════════════════════════════════════════════════════════════════════════════

def age_contributions(model: Mapping, k: int | None, x: np.ndarray | float, *, bracket: float,
                      h: float = 0.25, normalise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The rate of change of logit(q_x) along a trajectory, and each age's share of the e₀ gain.

    Paper: eq:gain_shares, sec:reconstruction:properties

    Along a calibrated trajectory ``z(x)`` the sex-average e₀ is ``x``, so by
    the chain rule ``1 = Σ_{s,a} ½ ∂e₀ˢ/∂z_{s,a} · dz_{s,a}/dx``: each term
    is the share of a small gain in e₀ that comes from the fall of mortality
    at age ``a`` of sex ``s`` (the exact gradient of
    :func:`mdmx.forecast.e0_and_gradient`; the derivative along the
    trajectory by central differences of step ``h`` years).

    Returns
    -------
    slope : (q, S, A) dz/dx, logit units per year of e₀.
    share : (q, S, A) contributions, summing to ≈ 1 over sexes and ages (the
        chain rule is exact as ``h`` → 0); with ``normalise`` they are divided
        by their sum, which a wider ``h`` — a slope over a range of e₀ rather
        than at a point — needs.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    up, down = generate(model, k, x + h, bracket=bracket), generate(model, k, x - h, bracket=bracket)
    slope = (up - down) / (2.0 * h)
    _, grad = fc.e0_and_gradient(generate(model, k, x, bracket=bracket))
    share = 0.5 * grad * slope
    if normalise:
        share = share / share.sum(axis=(1, 2), keepdims=True)
    return slope, share
