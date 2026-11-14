"""Level-free clustering of mortality schedules, and level epochs (paper §6).

What this module does
---------------------
Paper §6 groups the observed country-years of the HMD into *mortality
regimes* and separately classifies how the level of mortality changes over
time within each country.  A regime is a **deviation pattern relative
to the typical schedule at the same life expectancy**: two country-years
are in the same cluster if their schedules depart from the schedule that is
typical for their e₀ in the same way.  The steps and their functions:

1. **The unit of clustering** (§6.1): every observed (non-exceptional)
   country-year, listed once by :func:`observed_index`; its e₀ is that of the
   *observed* schedule (:func:`observed_e0`), averaged over the sexes.
2. **The smooth, low-order part of the effective core.**  Only the first
   ``k`` age columns of ``G_{c,t}`` are used, with ``k`` the smallest number
   of age components that capture a share τ of the sum of squares of the
   observed schedules (:func:`column_energy_share`, :func:`choose_n_columns`).
   :func:`basis_roughness` and :func:`year_to_year_noise` give the evidence
   that these columns are smooth and signal-dominated.
3. **Level-free features.**  The level column (age component 1) is dropped
   and, for each remaining entry of columns 2…k (both sex components), the
   e₀-conditional mean — a LOWESS curve of the entry against e₀ fitted
   across all observed country-years (:func:`conditional_mean`) — is
   subtracted.  The residuals are the features (:func:`feature_index`,
   :func:`cores_to_features`).
4. **PCA** of the features (:func:`pca_fit`).
5. **Gaussian mixtures** for a range of K (§6.3; :func:`bic_sweep`,
   :func:`gmm_n_parameters`), K chosen as the largest number of clusters
   whose partition is reproducible from other starting values
   (:func:`reproducibility_sweep`, :func:`choose_k_reproducible`; BIC is
   reported); whether the clusters survive resampling of whole population
   families is measured by :func:`group_bootstrap` and
   :func:`cluster_jaccard`.  Clusters numbered by :func:`order_clusters`.
6. **Ward's clustering** as a cross-check (:func:`ward_clustering`,
   :func:`matched_agreement`, :func:`node_label_counts`).
7. **Derived labels** by majority vote (:func:`majority_label`) and
   **diagnostics** of level independence and of noise (:func:`eta_squared`,
   :func:`linear_r_squared`, :func:`switch_share`, :func:`roughness`).
8. **Level epochs** (§6.6; :func:`classify_epochs`), which use the level
   coordinate ``G_{c,t}[1,1]`` and e₀ only, not the clustering features.

Generating a cluster's schedule at any e₀ (the artefact contract)
-----------------------------------------------------------------
The book chapter 06 saves an artefact ``clustering`` from which a schedule
of cluster ``k`` at a target life expectancy ``x`` is built as

    core(x, k) = conditional_core_at(x, e0_grid, conditional_core)      # typical core at e₀ = x
                 + residual_core[k]                                      # cluster k's mean deviation
    schedule   = tensor.schedules_from_cores(core, U_sex, U_age)         # (S, A) logit q_x

where ``conditional_core`` (g, r₁, r₂) holds the e₀-conditional mean of the
*full* effective core on the grid ``e0_grid`` (g,), and ``residual_core``
(K, r₁, r₂) is each cluster's mean feature vector mapped back to core
coordinates by :func:`features_to_cores` (zero outside the feature columns).
The e₀ of the resulting schedule is close to, but not exactly, ``x``
(the life table is non-linear in the core); the chapter reports the size
of the difference.

Conventions
-----------
* Arrays indexed by observation have the observed country-years as rows in
  the order of :func:`observed_index` (country first, then year).
* Cluster labels are integers ``0 … K−1`` in arrays; figures and the paper
  number them ``1 … K``.  ``-1`` always means "no label" (missing).
* Age columns of the core are 0-based in code (column 0 = the level) and
  1-based in the paper and the book's text.
* Index 0 of the sex axis is female, 1 is male.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from statsmodels.nonparametric.smoothers_lowess import lowess
from threadpoolctl import threadpool_limits

from . import lifetable
from .tensor import fix_signs

__all__ = [
    "EPOCH_CATEGORIES", "observed_index", "observed_cores", "age_structure_features",
    "observed_e0", "column_energy_share", "choose_n_columns", "basis_roughness",
    "year_to_year_noise", "feature_index", "cores_to_features", "features_to_cores",
    "e0_grid", "conditional_mean", "conditional_core_at",
    "pca_fit", "pca_transform", "gmm_n_parameters", "fit_gmm", "bic_sweep",
    "order_clusters", "cluster_means", "pairwise_ari", "reproducibility_sweep", "choose_k_reproducible",
    "cluster_jaccard", "group_resample", "group_bootstrap", "cluster_stability", "bootstrap_summary", "label_counts", "majority_label", "matched_agreement",
    "ward_clustering", "node_label_counts", "eta_squared", "linear_r_squared", "switch_share",
    "roughness", "noise_diagnostics", "region_of",
    "level_orientation", "lowess_at", "window_slopes", "classify_slopes", "yearly_categories",
    "runs", "classify_epochs",
]

#: The five epoch categories of paper §6.6, in the order of their integer
#: codes 0–4.  The order runs from fastest improvement to fastest worsening;
#: it is also the tie-break order used when a calendar year's windows are
#: split evenly between categories (see :func:`yearly_categories`).
EPOCH_CATEGORIES: tuple[str, ...] = (
    "Rapid improvement", "Slow improvement", "Stagnation", "Slow worsening", "Rapid worsening",
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. The unit of clustering: observed country-years
# ══════════════════════════════════════════════════════════════════════════════

def observed_index(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Country and year indices of every observed country-year.

    Paper: sec:clustering:unit

    Parameters
    ----------
    observed : (C, T) boolean observed mask ``O`` of the
        non-exceptional country-years.

    Returns
    -------
    c_idx, t_idx : two integer arrays of length ``n_obs``.  Row ``i`` of
        every per-observation array in this module refers to country
        ``c_idx[i]`` in year ``t_idx[i]``.  The order is country first, then
        year (``np.nonzero`` order), so consecutive rows of one country are
        its observed years in calendar order.
    """
    observed = np.asarray(observed, dtype=bool)
    if observed.ndim != 2:
        raise ValueError(f"observed must be a (C, T) mask, got shape {observed.shape}")
    c_idx, t_idx = np.nonzero(observed)
    return c_idx, t_idx


def observed_cores(G_ct: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Effective cores of the observed country-years, one matrix per row.

    Parameters
    ----------
    G_ct : (r₁, r₂, C, T) effective cores from the decomposition (artefact
        ``tucker``).  For observed country-years these are the exact
        projections ``Sᵀ M_{c,t} A`` of the observed schedules (:func:`mdmx.tensor.effective_cores`).
    c_idx, t_idx : observation index from :func:`observed_index`.

    Returns
    -------
    (n_obs, r₁, r₂) array: ``G[i]`` is the effective core of observation ``i``.
    """
    G = np.moveaxis(G_ct[:, :, c_idx, t_idx], -1, 0)
    if not np.all(np.isfinite(G)):
        raise ValueError("Effective cores contain non-finite values at observed country-years.")
    return G


def age_structure_features(G_obs: np.ndarray) -> np.ndarray:
    """The weights of the age-specific residual of each core: every age column but the level column.

    Paper: eq:level_residual_decomposition

    The first column of each effective core holds the weights on the first
    age basis function ``a₁`` — approximately the mean logit(q_x) schedule —
    for each sex component, i.e. the *level component* of the
    split of a schedule into a level component and an age-specific residual..  Dropping it leaves the weights that
    generate the *age-specific residual*.

    Parameters
    ----------
    G_obs : (n_obs, r₁, r₂) effective cores from :func:`observed_cores`.

    Returns
    -------
    (n_obs, r₁ (r₂ − 1)) array.  Each row is ``vec(G_{c,t}[:, 2:r₂])`` with
    the paper's column-stacking ``vec``: element ``j·r₁ + i`` (0-based) is
    the weight on sex component ``i+1`` and age component ``j+2``.

    Notes
    -----
    Because the sex and age bases are orthonormal, the Euclidean distance
    between two feature vectors equals the Frobenius distance between the
    two age-specific residual schedules in logit(q_x) space.  The features
    therefore need no further scaling.
    """
    n, r1, r2 = G_obs.shape
    # (n, r1, r2-1) -> (n, r2-1, r1) so that a C-order reshape stacks columns.
    return G_obs[:, :, 1:].transpose(0, 2, 1).reshape(n, r1 * (r2 - 1))


def observed_e0(M: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray, a0: float = lifetable.A0) -> np.ndarray:
    """Life expectancy at birth of each *observed* schedule, by sex.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor (artefact ``tensor``); NaN where not
        observed.
    c_idx, t_idx : observation index from :func:`observed_index`.
    a0 : fraction of the first year lived by infants who die (see
        :func:`mdmx.lifetable.e0_from_qx`).

    Returns
    -------
    (n_obs, S) array of e₀ in years (column 0 female, column 1 male),
    computed with :func:`mdmx.lifetable.e0_from_logit` from the data
    themselves, not from any model reconstruction.
    """
    schedules = np.moveaxis(M[:, :, c_idx, t_idx], 1, -1)          # (S, n_obs, A)
    if not np.all(np.isfinite(schedules)):
        raise ValueError("The tensor has missing values at country-years marked observed.")
    return lifetable.e0_from_logit(schedules, a0=a0).T


# ══════════════════════════════════════════════════════════════════════════════
# 1b. Level-free features: low-order columns minus their e0-conditional mean
# ══════════════════════════════════════════════════════════════════════════════

def column_energy_share(G_obs: np.ndarray, M_obs: np.ndarray) -> np.ndarray:
    """Share of the sum of squares of the observed schedules carried by each age column.

    Paper: eq:clus_columns

    Because the sex and age bases are orthonormal, the squared Frobenius norm
    of a schedule splits exactly over the entries of its effective core,
    plus a remainder outside the r₂ retained age components.  Summing over
    observed country-years and the two sex components gives each age
    column's share — the same (uncentred) quantity as the τ rule used to
    choose r₂ (:func:`mdmx.tensor.select_rank`)..

    Parameters
    ----------
    G_obs : (n_obs, r₁, r₂) effective cores of the observed country-years.
    M_obs : (S, A, n_obs) their observed logit(q_x) schedules.

    Returns
    -------
    (r₂,) shares; their sum is below 1 by the share outside the basis.
    """
    return np.sum(G_obs ** 2, axis=(0, 1)) / np.sum(M_obs ** 2)


def choose_n_columns(share: np.ndarray, tau: float) -> int:
    """Smallest number of leading age columns whose cumulative share reaches ``tau``.

    Paper: eq:clus_columns

    Raises ``ValueError`` if even all columns fall short.
    """
    cumulative = np.cumsum(share)
    if cumulative[-1] < tau:
        raise ValueError(f"All {len(share)} columns capture only {cumulative[-1]:.6f} < tau = {tau}")
    return int(np.searchsorted(cumulative, tau) + 1)


def basis_roughness(U_age: np.ndarray, first_age: int = 2) -> np.ndarray:
    """Roughness of each age basis function: sum of squared second differences.

    Ages below ``first_age`` are skipped because the infant–child
    discontinuity is genuine, not noise.  Returns an (r₂,) array.
    """
    return np.sum(np.diff(U_age[first_age:], n=2, axis=0) ** 2, axis=0)


def year_to_year_noise(G_obs: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """How much of each core entry's variation is year-to-year jitter (0 = smooth, ≈1 = noise).

    For every pair of *consecutive calendar years* observed in the same
    population, the change of each core entry is computed.  The ratio

        mean(change²) / (2 · variance across all observed country-years)

    is close to 0 for an entry that moves smoothly over time (a signal) and
    close to 1 for an entry whose values are independent from one year to
    the next (pure noise), because the difference of two independent draws
    has twice their variance.

    Parameters
    ----------
    G_obs : (n_obs, r₁, r₂) effective cores in :func:`observed_index` order.
    c_idx, t_idx : the observation index (country, year index).

    Returns
    -------
    (r₁, r₂) array of ratios.
    """
    consecutive = (c_idx[1:] == c_idx[:-1]) & (np.diff(t_idx) == 1)
    change = np.diff(G_obs, axis=0)[consecutive]
    return np.mean(change ** 2, axis=0) / (2.0 * G_obs.var(axis=0))


def feature_index(r1: int, n_columns: int) -> np.ndarray:
    """Core coordinates of the level-free features: age columns 2…k, all sex components.

    Parameters
    ----------
    r1 : number of sex components (2).
    n_columns : ``k``, the number of leading age columns kept, including the
        level column, which is then dropped.

    Returns
    -------
    (p, 2) integer array with ``p = r₁ (k − 1)``; row ``m`` gives the
    (sex component, age column) — 0-based — of feature ``m``.  The order
    stacks the columns (paper's ``vec``): (0, 1), (1, 1), (0, 2), (1, 2), ….
    """
    if n_columns < 2:
        raise ValueError("At least two age columns are needed (the level column is dropped).")
    return np.array([(i, j) for j in range(1, n_columns) for i in range(r1)], dtype=int)


def cores_to_features(G: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Pick the feature entries out of effective cores.

    Paper: eq:level_free_features

    ``G`` has shape (..., r₁, r₂); the result has shape (..., p), with
    entry ``m`` equal to ``G[..., index[m, 0], index[m, 1]]``.
    """
    return G[..., index[:, 0], index[:, 1]]


def features_to_cores(F: np.ndarray, index: np.ndarray, r1: int, r2: int) -> np.ndarray:
    """Put feature vectors back into effective-core coordinates (inverse of :func:`cores_to_features`).

    Paper: eq:regime_schedule

    ``F`` has shape (..., p); the result (..., r₁, r₂) is zero everywhere
    except at the feature positions.  Used to turn a cluster's mean residual
    into a deviation that can be added to a core.
    """
    F = np.asarray(F, dtype=float)
    G = np.zeros(F.shape[:-1] + (r1, r2))
    G[..., index[:, 0], index[:, 1]] = F
    return G


def e0_grid(e0: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced e₀ values covering the observed range, in steps of ``step`` years.

    The grid starts at the observed minimum rounded down and ends at the
    maximum rounded up to a multiple of ``step``.
    """
    lo = np.floor(np.min(e0) / step) * step
    hi = np.ceil(np.max(e0) / step) * step
    return np.round(np.arange(lo, hi + step / 2, step), 10)


def conditional_mean(x: np.ndarray, Y: np.ndarray, *, frac: float, iterations: int,
                     delta_share: float, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E[Y | x] for every column of ``Y``, by LOWESS, at the data and on a grid.

    Paper: *eq:level_free_features

    Each column of ``Y`` is smoothed against ``x`` (here: e₀ of the observed
    schedule) across all observations with a LOWESS span of ``frac`` of the
    points and ``iterations`` robustifying iterations (Cleveland 1979).  For
    speed, points within ``delta_share`` of the range of ``x`` of a point
    already fitted are linearly interpolated (statsmodels' ``delta``).  The
    curve is then evaluated on ``grid`` by linear interpolation between the
    fitted values; beyond the observed range it is held constant.

    Parameters
    ----------
    x : (n,) explanatory variable (e₀); values must be distinct.
    Y : (n,) or (n, m) responses (feature or core entries).
    frac, iterations, delta_share : LOWESS settings.
    grid : (g,) points at which to report the curve.

    Returns
    -------
    at_data : (n, m) fitted values at the observations (Y's shape kept).
    on_grid : (g, m) the curve on the grid.
    """
    x = np.asarray(x, dtype=float)
    if len(np.unique(x)) != len(x):
        raise ValueError("conditional_mean needs distinct x values (ties make the grid curve ambiguous)")
    Y = np.asarray(Y, dtype=float)
    Y2 = Y[:, None] if Y.ndim == 1 else Y.reshape(len(Y), -1)
    order = np.argsort(x)
    delta = delta_share * np.ptp(x)
    at_data = np.empty_like(Y2)
    on_grid = np.empty((len(grid), Y2.shape[1]))
    for m in range(Y2.shape[1]):
        fit = lowess(Y2[:, m], x, frac=frac, it=iterations, delta=delta, return_sorted=False)
        at_data[:, m] = fit
        on_grid[:, m] = np.interp(grid, x[order], fit[order])
    if Y.ndim == 1:
        return at_data[:, 0], on_grid[:, 0]
    return at_data.reshape(Y.shape), on_grid.reshape((len(grid),) + Y.shape[1:])


def conditional_core_at(e0: float | np.ndarray, grid: np.ndarray, core_grid: np.ndarray) -> np.ndarray:
    """The e₀-conditional mean effective core at any e₀, by linear interpolation on the grid.

    Paper: eq:typical_schedule

    Parameters
    ----------
    e0 : a target life expectancy, or an array of them, shape (q,).
    grid : (g,) increasing e₀ grid (artefact ``clustering``, ``e0_grid``).
    core_grid : (g, r₁, r₂) conditional mean core on the grid
        (``conditional_core``).

    Returns
    -------
    (r₁, r₂) for a scalar ``e0``, else (q, r₁, r₂).  Add a cluster's
    ``residual_core`` to obtain that cluster's core at the same e₀.
    """
    e0 = np.asarray(e0, dtype=float)
    flat = core_grid.reshape(len(grid), -1)
    out = np.column_stack([np.interp(e0.ravel(), grid, flat[:, m]) for m in range(flat.shape[1])])
    return out.reshape(e0.shape + core_grid.shape[1:])


# ══════════════════════════════════════════════════════════════════════════════
# 2. Principal component analysis of the features
# ══════════════════════════════════════════════════════════════════════════════

def pca_fit(F: np.ndarray, variance: float) -> dict:
    """Principal components of the feature matrix, truncated by variance share.

    Paper: sec:clustering:features

    Parameters
    ----------
    F : (n_obs, p) feature matrix (rows ``f_{c,t}ᵀ``).
    variance : share of the total variance the retained components must
        explain (``[clustering] pca_variance``).  ``d`` is the smallest number of
        components whose cumulative share reaches it.

    Returns
    -------
    dict with
        ``mean`` (p,) — column means subtracted before the SVD;
        ``components`` (p, d) — orthonormal loadings, signs fixed with
        :func:`mdmx.tensor.fix_signs`;
        ``explained_variance_ratio`` (p,) and ``cumulative`` (p,) — for all
        components;
        ``d`` — the number retained;
        ``scores`` (n_obs, d) — the reduced features ``F̃``.

    Notes
    -----
    Computed by a thin SVD of the centred matrix, which is deterministic
    (unlike randomised PCA solvers).
    """
    F = np.asarray(F, dtype=float)
    mean = F.mean(axis=0)
    _, sv, Vt = np.linalg.svd(F - mean, full_matrices=False)
    V = fix_signs(Vt.T)                                   # (p, p), one column per component
    evr = sv ** 2 / np.sum(sv ** 2)
    cumulative = np.cumsum(evr)
    d = int(min(np.searchsorted(cumulative, variance) + 1, len(evr)))
    components = V[:, :d]
    return {"mean": mean, "components": components, "explained_variance_ratio": evr,
            "cumulative": cumulative, "d": d, "scores": (F - mean) @ components}


def pca_transform(F: np.ndarray, pca: Mapping) -> np.ndarray:
    """Project new feature rows onto a fitted PCA: ``(F − mean) · components``."""
    return (np.asarray(F, dtype=float) - pca["mean"]) @ pca["components"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Gaussian mixtures and BIC
# ══════════════════════════════════════════════════════════════════════════════

def gmm_n_parameters(k: int, d: int, covariance_type: str) -> int:
    """Number of free parameters ``p_k`` of a K-component Gaussian mixture (the penalty count of the BIC, :func:`bic_sweep`).

    ``K − 1`` mixing weights, ``K·d`` means, plus the covariance parameters:
    ``K·d(d+1)/2`` (``"full"``: one unrestricted covariance per component),
    ``d(d+1)/2`` (``"tied"``: one shared covariance), ``K·d`` (``"diag"``),
    or ``K`` (``"spherical"``).
    """
    cov = {"full": k * d * (d + 1) // 2, "tied": d * (d + 1) // 2,
           "diag": k * d, "spherical": k}
    if covariance_type not in cov:
        raise ValueError(f"Unknown covariance_type {covariance_type!r}")
    return (k - 1) + k * d + cov[covariance_type]


def fit_gmm(X: np.ndarray, k: int, *, covariance_type: str, n_init: int, max_iter: int,
            tol: float, reg_covar: float, seed: int) -> GaussianMixture:
    """Fit one K-component Gaussian mixture by EM (scikit-learn).

    ``n_init`` EM runs are started from k-means solutions with seeds derived
    from ``seed``; the run with the highest likelihood is kept.
    ``reg_covar`` is added to the diagonal of every covariance matrix for
    numerical stability.
    """
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init,
                         max_iter=max_iter, tol=tol, reg_covar=reg_covar, random_state=seed)
    return gm.fit(X)


def bic_sweep(X: np.ndarray, k_values: Sequence[int], *, covariance_type: str, n_init: int,
              max_iter: int, tol: float, reg_covar: float, seed: int
              ) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    """Fit a Gaussian mixture for every K and tabulate BIC and AIC.

    Paper: eq:bic, sec:clustering:gmm

    Parameters
    ----------
    X : (n, d) reduced features ``F̃``.
    k_values : the candidate numbers of components (``[clustering] k_min`` … ``k_max``).
    covariance_type, n_init, max_iter, tol, reg_covar, seed : passed to
        :func:`fit_gmm`.

    Returns
    -------
    table : one row per K with the maximised log-likelihood ``ℓ``, the
        number of parameters ``p_k``, ``BIC = −2ℓ + p_k log n`` and
        ``AIC = −2ℓ + 2 p_k``, whether EM converged, the number of EM
        iterations, and the size of the smallest cluster.
    models : the fitted mixtures, keyed by K.

    Raises
    ------
    RuntimeError if the BIC computed here (−2ℓ + p_k log n) disagrees with
    scikit-learn's own value (which would mean ``p_k`` is mis-counted).
    """
    n, d = X.shape
    rows, models = [], {}
    for k in k_values:
        gm = fit_gmm(X, k, covariance_type=covariance_type, n_init=n_init, max_iter=max_iter,
                     tol=tol, reg_covar=reg_covar, seed=seed)
        loglik = float(gm.score(X) * n)                    # score() is the mean log-likelihood
        p = gmm_n_parameters(k, d, covariance_type)
        bic = -2.0 * loglik + p * np.log(n)
        if not np.isclose(bic, gm.bic(X), rtol=1e-9, atol=1e-6):
            raise RuntimeError(f"K={k}: BIC {bic} differs from scikit-learn's {gm.bic(X)}")
        sizes = np.bincount(gm.predict(X), minlength=k)
        rows.append({"K": int(k), "log_likelihood": loglik, "n_parameters": p, "bic": bic,
                     "aic": -2.0 * loglik + 2.0 * p, "converged": bool(gm.converged_),
                     "n_iter": int(gm.n_iter_), "smallest_cluster": int(sizes.min())})
        models[int(k)] = gm
    return pd.DataFrame(rows), models


def order_clusters(labels: np.ndarray, key: np.ndarray, n_labels: int) -> np.ndarray:
    """Permutation that numbers clusters by increasing mean of ``key``.

    Mixture components come out of EM in an arbitrary order.  Numbering them
    by, for example, the mean e₀ of their members makes the numbering
    reproducible and meaningful.

    Parameters
    ----------
    labels : (n,) integer labels ``0 … n_labels−1``.
    key : (n,) values whose cluster means define the order.
    n_labels : number of clusters.

    Returns
    -------
    order : (n_labels,) array; ``order[new] = old``.  The relabelled labels
        are ``np.argsort(order)[labels]``.

    Raises
    ------
    ValueError if a cluster has no members (its mean is undefined).
    """
    counts = np.bincount(labels, minlength=n_labels)
    if np.any(counts == 0):
        raise ValueError(f"Empty cluster(s): {np.flatnonzero(counts == 0).tolist()}")
    means = np.bincount(labels, weights=key, minlength=n_labels) / counts
    return np.argsort(means, kind="stable")


def cluster_means(values: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Mean of ``values`` over the members of each cluster.

    Paper: *eq:regime_schedule

    Parameters
    ----------
    values : (n, ...) array, one entry per observation — e.g. the effective
        cores (n, r₁, r₂), giving the cluster centroids in G-space.
    labels : (n,) integer labels ``0 … n_labels−1``.
    n_labels : number of clusters.

    Returns
    -------
    (n_labels, ...) array of means.

    Notes
    -----
    Because a schedule is linear in its effective core
    (``M = S G Aᵀ``), the schedule of the mean core equals the mean of the
    members' schedules — so the G-space centroid *is* the cluster
    centroid of §6.5 (the mean of the members' schedules), expressed in the shared basis., expressed in the shared basis.
    """
    values = np.asarray(values, dtype=float)
    flat = values.reshape(len(values), -1)
    onehot = np.zeros((len(labels), n_labels))
    onehot[np.arange(len(labels)), labels] = 1.0
    counts = onehot.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError(f"Empty cluster(s): {np.flatnonzero(counts == 0).tolist()}")
    return ((onehot.T @ flat) / counts[:, None]).reshape((n_labels,) + values.shape[1:])


# ══════════════════════════════════════════════════════════════════════════════
# 3b. The choice of K: reproducibility, and stability under resampling
# ══════════════════════════════════════════════════════════════════════════════

def pairwise_ari(labelings: Sequence[np.ndarray]) -> tuple[float, float]:
    """Mean and smallest adjusted Rand index over all pairs of partitions of the same points.

    Parameters
    ----------
    labelings : two or more (n,) label arrays.

    Returns
    -------
    (mean, smallest) adjusted Rand index over the pairs.
    """
    if len(labelings) < 2:
        raise ValueError("Need at least two partitions to compare.")
    scores = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labelings, 2)]
    return float(np.mean(scores)), float(np.min(scores))


def _fit_predict(task: tuple) -> np.ndarray:
    """Labels of every row of ``X`` from a mixture fitted to ``X[rows]`` (all rows if ``rows`` is None).

    The OpenMP loops of the k-means starting values run on one thread, so
    that parallel worker processes do not oversubscribe the cores and the
    result does not depend on how many fits run at once (see :func:`_map`).
    (threadpoolctl cannot limit Apple Accelerate's BLAS; the results were
    checked to be identical for any number of workers there too.)
    """
    X, k, settings, rows = task
    with threadpool_limits(limits=1):
        return fit_gmm(X if rows is None else X[rows], k, **settings).predict(X)


def _map(tasks: list[tuple], workers: int) -> list[np.ndarray]:
    """:func:`_fit_predict` for every task, in ``workers`` processes (0: one per CPU core; 1: none)."""
    if workers < 0:
        raise ValueError(f"workers must be 0 (one per CPU core) or positive, got {workers}.")
    n_workers = min(workers or os.cpu_count() or 1, len(tasks))
    if n_workers <= 1:
        return [_fit_predict(task) for task in tasks]
    # "spawn" on every platform: fork is unsafe in a process with threads (a Jupyter kernel).
    with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_fit_predict, tasks))


def reproducibility_sweep(X: np.ndarray, k_values: Sequence[int], seeds: Sequence[int], *,
                          covariance_type: str, n_init: int, max_iter: int, tol: float, reg_covar: float,
                          workers: int = 1) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Refit the mixture for every K from several seeds, and measure how far the partitions agree.

    Paper: eq:clus_k_rule, sec:clustering:gmm

    The likelihood of a mixture has many local maxima; from different
    k-means starting values EM may reach different partitions of similar
    likelihood.  A K whose refits agree has a reproducible partition.

    Parameters
    ----------
    X : (n, d) reduced features ``F̃``.
    k_values : the candidate numbers of components.
    seeds : seeds of the starting values, two or more; each refit keeps the
        best of ``n_init`` EM starts (:func:`fit_gmm`).
    covariance_type, max_iter, tol, reg_covar : as for :func:`fit_gmm`.
    workers : worker processes (the results do not depend on it).

    Returns
    -------
    table : one row per K: ``ari_mean`` and ``ari_min``, the mean and the
        smallest adjusted Rand index over all pairs of refits.
    labels : K → (len(seeds), n) labels of the refits, in the order of ``seeds``.
    """
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds to compare refits.")
    settings = {"covariance_type": covariance_type, "n_init": n_init, "max_iter": max_iter, "tol": tol,
                "reg_covar": reg_covar}
    tasks = [(X, k, {**settings, "seed": s}, None) for k in k_values for s in seeds]
    fitted = iter(_map(tasks, workers))
    labels = {int(k): np.stack([next(fitted) for _ in seeds]) for k in k_values}
    rows = [{"K": k, **dict(zip(("ari_mean", "ari_min"), pairwise_ari(list(lab)), strict=True))}
            for k, lab in labels.items()]
    return pd.DataFrame(rows), labels


def choose_k_reproducible(table: pd.DataFrame, threshold: float) -> int:
    """The largest K such that K and every smaller K have refits agreeing at mean ARI ≥ ``threshold``.

    Paper: *eq:clus_k_rule, sec:clustering:gmm

    Parameters
    ----------
    table : the table of :func:`reproducibility_sweep` (columns ``K``, ``ari_mean``).
    threshold : smallest acceptable mean pairwise adjusted Rand index.

    Raises
    ------
    ValueError if even the smallest K fails.
    """
    t = table.sort_values("K")
    failing = t.loc[t["ari_mean"] < threshold, "K"]
    if failing.empty:
        return int(t["K"].max())
    first = int(failing.min())
    if first == int(t["K"].min()):
        raise ValueError(f"Even K = {first} is not reproducible (mean ARI below {threshold}).")
    return int(t.loc[t["K"] < first, "K"].max())


def cluster_jaccard(ref: np.ndarray, lab: np.ndarray, n_ref: int, n_lab: int, mask: np.ndarray) -> np.ndarray:
    """Hennig's cluster-wise Jaccard similarity of a reference partition with another partition.

    Paper: eq:clus_jaccard, sec:clustering:gmm

    For each reference cluster A, the largest Jaccard similarity
    ``|A ∩ B| / |A ∪ B|`` with any cluster B of ``lab``, both restricted to
    the points in ``mask`` (in a bootstrap: the points in the resample).
    Hennig (2007), *Computational Statistics & Data Analysis* 52:258–271.

    Parameters
    ----------
    ref : (n,) reference labels ``0 … n_ref−1``.
    lab : (n,) labels of the other partition, ``0 … n_lab−1``.
    n_ref, n_lab : numbers of clusters of the two partitions.
    mask : (n,) True for the points compared.

    Returns
    -------
    (n_ref,) similarities in [0, 1]; NaN for a reference cluster with no
    point in ``mask``.
    """
    out = np.full(n_ref, np.nan)
    for a in range(n_ref):
        in_a = (ref == a) & mask
        if not in_a.any():
            continue
        out[a] = max(np.sum(in_a & (lab == b)) / np.sum(in_a | ((lab == b) & mask)) for b in range(n_lab))
    return out


def group_resample(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw whole groups with replacement, as many as there are distinct groups.

    Parameters
    ----------
    groups : (n,) group label of each point.
    rng : the random generator of this resample.

    Returns
    -------
    drawn : (G,) the drawn group labels (with repeats).
    rows : indices of the points of the drawn groups, a group's points
        repeated as often as the group was drawn.
    """
    uniq = np.unique(groups)
    drawn = rng.choice(uniq, size=uniq.size, replace=True)
    return drawn, np.concatenate([np.flatnonzero(groups == g) for g in drawn])


def group_bootstrap(X: np.ndarray, ref: np.ndarray, groups: np.ndarray, k: int, *, n_resamples: int,
                    seed: int, covariance_type: str, n_init: int, max_iter: int, tol: float,
                    reg_covar: float, workers: int = 1) -> pd.DataFrame:
    """Stability of a partition when whole groups of points are resampled (Hennig 2007).

    Paper: eq:clus_jaccard, sec:clustering:gmm

    Each resample draws as many groups as there are, with replacement, and
    takes every point of each drawn group (repeated groups repeat their
    points), so that the dependence between the years of one population is
    respected.  A mixture with ``k`` components (``n_init`` EM starts) is
    fitted to the resample and labels every point; the reference partition
    is then compared with it on the distinct points of the resample.

    Parameters
    ----------
    X : (n, d) reduced features.
    ref : (n,) reference labels ``0 … k−1`` (e.g. the chapter's partition).
    groups : (n,) group of each point (population family).
    k : number of mixture components refitted.
    n_resamples : number of resamples B.
    seed : resample ``b`` uses ``np.random.default_rng([seed, k, b])`` for
        the draw and the seed of its fit, so results do not depend on
        ``workers``.
    covariance_type, n_init, max_iter, tol, reg_covar : as for :func:`fit_gmm`.
    workers : worker processes.

    Returns
    -------
    DataFrame, one row per resample: ``b``, ``n_groups`` (distinct groups
    drawn), ``ari`` (with ``ref`` on the distinct points of the resample), and
    ``jaccard_0 … jaccard_{k−1}`` (:func:`cluster_jaccard`; NaN for a cluster
    with no member in the resample).
    """
    settings = {"covariance_type": covariance_type, "n_init": n_init, "max_iter": max_iter, "tol": tol,
                "reg_covar": reg_covar}
    draws, tasks = [], []
    for b in range(n_resamples):
        rng = np.random.default_rng([seed, k, b])
        drawn, rows = group_resample(groups, rng)
        draws.append((drawn, rows))
        tasks.append((X, k, {**settings, "seed": int(rng.integers(2 ** 31))}, rows))
    out = []
    for b, ((drawn, rows), lab) in enumerate(zip(draws, _map(tasks, workers), strict=True)):
        mask = np.zeros(len(X), dtype=bool)
        mask[rows] = True
        jac = cluster_jaccard(ref, lab, k, k, mask)
        out.append({"b": b, "n_groups": int(np.unique(drawn).size), "ari": adjusted_rand_score(ref[mask], lab[mask]),
                    **{f"jaccard_{c}": jac[c] for c in range(k)}})
    return pd.DataFrame(out)


def cluster_stability(resamples: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each cluster's mean Jaccard over the resamples of :func:`group_bootstrap`, and how many held it.

    Returns
    -------
    jaccard : (k,) mean over the resamples that hold at least one member.
    present : (k,) number of those resamples.
    """
    jac = resamples[[f"jaccard_{c}" for c in range(k)]].to_numpy()
    present = np.isfinite(jac).sum(axis=0)
    if np.any(present == 0):
        raise ValueError(f"Cluster(s) {np.flatnonzero(present == 0).tolist()} absent from every resample.")
    return np.nanmean(jac, axis=0), present


def bootstrap_summary(resamples_by_k: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per K of :func:`group_bootstrap` results: the clusters' mean Jaccard and the mean ARI.

    Returns
    -------
    DataFrame with ``K``, ``jaccard_min`` (the least stable cluster's mean
    Jaccard), ``jaccard_mean`` (average over the clusters) and ``ari``.
    """
    rows = []
    for k, res in sorted(resamples_by_k.items()):
        jac, _ = cluster_stability(res, k)
        rows.append({"K": int(k), "jaccard_min": float(jac.min()), "jaccard_mean": float(jac.mean()),
                     "ari": float(res["ari"].mean())})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Majority labels, agreement between clusterings, Ward's method
# ══════════════════════════════════════════════════════════════════════════════

def label_counts(labels: np.ndarray, groups: np.ndarray, n_groups: int, n_labels: int) -> np.ndarray:
    """Count how often each label occurs in each group.

    Parameters
    ----------
    labels : (n,) integer labels; entries ``< 0`` mean "no label" and are
        ignored.
    groups : (n,) integer group of each entry, ``0 … n_groups−1``.
    n_groups, n_labels : sizes of the output.

    Returns
    -------
    (n_groups, n_labels) integer array of counts.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    keep = labels >= 0
    counts = np.zeros((n_groups, n_labels), dtype=int)
    np.add.at(counts, (groups[keep], labels[keep]), 1)
    return counts


def majority_label(labels: np.ndarray, groups: np.ndarray, n_groups: int, n_labels: int) -> np.ndarray:
    """Most frequent label in each group.

    Paper: eq:country_cluster, eq:year_cluster

    This is the single implementation of every majority vote in the chapter:
    the country cluster ``g_c`` (groups = countries), the year cluster
    ``g_t`` (groups = years), the modal cluster of a country in a decade, and
    the modal epoch category of a calendar year.

    Parameters
    ----------
    labels : (n,) integer labels ``0 … n_labels−1``; ``-1`` = missing.
        Only labelled entries vote — for clusters, only *observed*
        country-years.
    groups : (n,) group index of each entry, ``0 … n_groups−1``.
    n_groups, n_labels : number of groups and of possible labels.

    Returns
    -------
    (n_groups,) integer array: the label with the most votes in each group,
    or ``-1`` if the group has no labelled entries.

    Tie-break
    ---------
    If two or more labels share the largest count, the **smallest label
    number** wins (``argmax`` returns the first maximum).  Callers choose the
    label numbering so that this rule means something stated: clusters are
    numbered by increasing mean e₀ (:func:`order_clusters`), epoch
    categories in the order of :data:`EPOCH_CATEGORIES`.
    """
    counts = label_counts(labels, groups, n_groups, n_labels)
    winner = counts.argmax(axis=1)
    winner[counts.sum(axis=1) == 0] = -1
    return winner


def matched_agreement(a: np.ndarray, b: np.ndarray, n_labels: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Share of observations on which two clusterings agree after matching their labels.

    Cluster numbers of two methods are arbitrary, so the labels of ``a`` are
    first matched one-to-one to those of ``b`` so as to maximise the number
    of agreements (the Hungarian algorithm on the confusion matrix).

    Parameters
    ----------
    a, b : (n,) integer labels ``0 … n_labels−1`` from two clusterings.
    n_labels : number of clusters in each.

    Returns
    -------
    share : fraction of the n observations with matching labels.
    confusion : (n_labels, n_labels) counts, rows = ``a``, columns = ``b``.
    mapping : (n_labels,) array; label ``i`` of ``a`` corresponds to label
        ``mapping[i]`` of ``b``.
    """
    confusion = np.zeros((n_labels, n_labels), dtype=int)
    np.add.at(confusion, (a, b), 1)
    rows, cols = linear_sum_assignment(-confusion)
    mapping = np.empty(n_labels, dtype=int)
    mapping[rows] = cols
    return float(confusion[rows, cols].sum() / len(a)), confusion, mapping


def ward_clustering(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ward's minimum-variance hierarchical clustering, cut into ``k`` groups (Remark ``ward``).

    Parameters
    ----------
    X : (n, d) reduced features ``F̃`` (all observations; no subsampling).
    k : number of groups at which the tree is cut.

    Returns
    -------
    Z : (n−1, 4) SciPy linkage matrix (row ``m`` creates node ``n+m``).
    labels : (n,) integer labels ``0 … k−1``.
    """
    Z = linkage(X, method="ward")
    labels = fcluster(Z, k, criterion="maxclust") - 1
    return Z, labels


def node_label_counts(Z: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Label composition of every node of a hierarchical-clustering tree.

    Parameters
    ----------
    Z : (n−1, 4) SciPy linkage matrix of ``n`` observations.
    labels : (n,) labels (e.g. GMM clusters) of the observations.
    n_labels : number of distinct labels.

    Returns
    -------
    (2n−1, n_labels) counts: row ``i < n`` is leaf ``i``; row ``n+m`` is the
    node formed by merge ``m``.  Used to colour a dendrogram by the GMM
    clusters below each branch.

    Notes
    -----
    The merges must be processed in order (each node needs its children),
    so this is a plain loop over the n−1 merges.
    """
    n = len(labels)
    counts = np.zeros((2 * n - 1, n_labels), dtype=int)
    counts[np.arange(n), labels] = 1
    for m, (left, right) in enumerate(Z[:, :2].astype(int)):
        counts[n + m] = counts[left] + counts[right]
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# 5. How independent of level are the features and the clusters?
# ══════════════════════════════════════════════════════════════════════════════

def eta_squared(y: np.ndarray, labels: np.ndarray) -> float:
    """Share of the variance of ``y`` explained by cluster membership (η²).

    ``η² = Σ_k n_k (ȳ_k − ȳ)² / Σ_i (y_i − ȳ)²`` — the R² of a one-way
    analysis of variance of ``y`` on the labels.  0 means the clusters have
    identical means; 1 means ``y`` is constant within clusters.
    """
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    sums = np.bincount(labels, weights=y)
    present = counts > 0
    means = sums[present] / counts[present]
    between = np.sum(counts[present] * (means - y.mean()) ** 2)
    return float(between / np.sum((y - y.mean()) ** 2))


def linear_r_squared(y: np.ndarray, X: np.ndarray) -> float:
    """R² of the least-squares regression of ``y`` on the columns of ``X`` (with intercept)."""
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))


def switch_share(labels: np.ndarray, c_idx: np.ndarray) -> float:
    """Share of consecutive observed years of the same country with different labels.

    ``labels`` and ``c_idx`` must be in :func:`observed_index` order, so that
    neighbouring rows of one country are consecutive observed years (gaps in
    a series are bridged).
    """
    same_country = c_idx[1:] == c_idx[:-1]
    changed = labels[1:] != labels[:-1]
    return float(changed[same_country].mean())


def roughness(M: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray, first_age: int,
              last_age: int) -> np.ndarray:
    """Year-to-year noise in each observed schedule: mean squared second difference over age.

    For a smooth schedule the second differences
    ``z_{x+1} − 2 z_x + z_{x−1}`` of ``z = logit(q_x)`` are small; random
    variation from few deaths (small populations, low mortality) makes them
    large.  Used to check whether a cluster is distinguished by noise rather
    than by the shape of mortality.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor.
    c_idx, t_idx : observation index from :func:`observed_index`.
    first_age, last_age : ages (inclusive) whose second differences are
        averaged; ``first_age ≥ 1`` so that the infant–child discontinuity
        at age 0 is excluded.

    Returns
    -------
    (n_obs,) mean over both sexes and the ages of the squared second
    differences.
    """
    if first_age < 1:
        raise ValueError("first_age must be at least 1 (the age-0 discontinuity is genuine)")
    z = M[:, first_age - 1:last_age + 2, c_idx, t_idx]           # one extra age on each side
    return np.mean(np.diff(z, n=2, axis=1) ** 2, axis=(0, 1))


def noise_diagnostics(labels: np.ndarray, n_labels: int, excess_log_roughness: np.ndarray,
                      small: np.ndarray, *, roughness_ratio: float, small_share: float) -> pd.DataFrame:
    """Flag clusters that collect noisy schedules of small populations.

    Paper: sec:clustering:interpretation

    Parameters
    ----------
    labels : (n,) cluster labels ``0 … n_labels−1``.
    n_labels : number of clusters.
    excess_log_roughness : (n,) log₁₀ roughness of each observed schedule
        (:func:`roughness`) minus its e₀-conditional mean — how much rougher
        the schedule is than typical schedules with the same e₀.
    small : (n,) True for country-years in the smallest quarter of
        population size.
    roughness_ratio, small_share : flagging thresholds.

    Returns
    -------
    DataFrame, one row per cluster: ``n``; ``roughness_ratio`` =
    10^median(excess) (1 = as smooth as typical at the same e₀);
    ``small_share`` = share of members from the smallest population
    quarter; ``noise`` = True if both thresholds are exceeded.
    """
    rows = []
    for k in range(n_labels):
        m = labels == k
        ratio = float(10 ** np.median(excess_log_roughness[m]))
        share = float(np.mean(small[m]))
        rows.append({"cluster": k, "n": int(m.sum()), "roughness_ratio": ratio, "small_share": share,
                     "noise": bool(ratio > roughness_ratio and share > small_share)})
    return pd.DataFrame(rows)


def region_of(pops: Sequence[str], regions: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Region name of every population, from a region → population-codes table.

    Raises ``ValueError`` if a population is in no region or in more than
    one, so that no population is silently dropped from a regional summary.
    """
    owner: dict[str, str] = {}
    for region, codes in regions.items():
        for code in codes:
            if code in owner:
                raise ValueError(f"{code} is listed in both {owner[code]!r} and {region!r}")
            owner[code] = region
    missing = [p for p in pops if p not in owner]
    if missing:
        raise ValueError(f"Populations without a region: {missing}")
    return np.array([owner[p] for p in pops])


# ══════════════════════════════════════════════════════════════════════════════
# 6. Level epochs (paper §6.6)
# ══════════════════════════════════════════════════════════════════════════════

def level_orientation(level: np.ndarray, e0: np.ndarray) -> int:
    """Sign that makes the level signal increase with life expectancy.

    Paper: eq:level_signal

    The level loading ``G_{c,t}[1,1]`` inherits the arbitrary sign of the
    singular vectors ``s₁`` and ``a₁``.  Multiplying it by the sign of its
    correlation with e₀ gives a signal that *rises when mortality improves*,
    whatever sign convention the decomposition used.

    Returns
    -------
    +1 or −1.

    Raises
    ------
    ValueError if the correlation is exactly zero (the level would carry no
    information about e₀).
    """
    r = np.corrcoef(level, e0)[0, 1]
    if not np.isfinite(r) or r == 0:
        raise ValueError(f"Cannot orient the level signal: corr(level, e0) = {r}")
    return int(np.sign(r))


def lowess_at(x: np.ndarray, y: np.ndarray, *, frac: float, min_points: int,
              iterations: int) -> np.ndarray:
    """LOWESS fit of ``y`` on ``x``, evaluated at the given ``x`` (the smoothing step of the epoch classification, :func:`classify_epochs`).

    The smoothing span is ``max(frac, min_points / len(x))`` of the points,
    so that short series are still smoothed over at least ``min_points``
    observations.  ``iterations`` robustifying iterations down-weight
    outliers (Cleveland 1979).
    """
    span = min(1.0, max(frac, min_points / len(x)))
    return lowess(y, x, frac=span, it=iterations, return_sorted=False)


def window_slopes(years: np.ndarray, values: np.ndarray, *, window: int,
                  min_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS slopes of ``values`` on ``years`` in every rolling window (the window-slope step of the epoch classification).

    Windows cover the calendar years ``[start, start + window − 1]`` for every
    ``start`` from the first observed year to the last one that still fits a
    whole window.  A window is kept only if it contains at least
    ``min_points`` observed years.

    Parameters
    ----------
    years : (m,) observed calendar years of one country, increasing.
    values : (m,) or (m, q) series observed in those years (e.g. the
        smoothed level and the smoothed e₀).
    window, min_points : W and the minimum number of points.

    Returns
    -------
    starts : (w,) first calendar year of each kept window.
    n_points : (w,) observed years in each kept window.
    slopes : (w,) or (w, q) least-squares slope per calendar year.
    """
    if min_points < 2:
        raise ValueError("A slope needs at least two points: min_points must be >= 2")
    years = np.asarray(years, dtype=float)
    vals = np.asarray(values, dtype=float)
    vals2 = vals[:, None] if vals.ndim == 1 else vals
    starts = np.arange(int(years.min()), int(years.max()) - window + 2)
    # inside[w, i]: observed year i lies in window w.
    inside = (years[None, :] >= starts[:, None]) & (years[None, :] <= starts[:, None] + window - 1)
    w = inside.astype(float)
    n = w.sum(axis=1)
    keep = n >= min_points
    w, n, starts = w[keep], n[keep], starts[keep]
    # Slope = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)², written with window sums.
    sx = w @ years
    sxx = w @ years ** 2
    sy = w @ vals2
    sxy = w @ (years[:, None] * vals2)
    slopes = (n[:, None] * sxy - sx[:, None] * sy) / (n * sxx - sx ** 2)[:, None]
    return starts, n.astype(int), (slopes[:, 0] if vals.ndim == 1 else slopes)


def classify_slopes(slopes: np.ndarray, delta: float, delta_rapid: float) -> np.ndarray:
    """Epoch category of each window slope (the fixed thresholds δ and δ_rapid of the epoch classification).

    Paper: eq:epoch_categories

    The slopes must come from the *oriented* level signal
    (:func:`level_orientation`), so that a positive slope always means
    improving mortality (rising e₀).  With ``s`` the slope:

    =====  ===================  ==============================
    code   category             rule
    =====  ===================  ==============================
    0      Rapid improvement    s > δ_rapid
    1      Slow improvement     δ < s ≤ δ_rapid
    2      Stagnation           |s| ≤ δ
    3      Slow worsening       −δ_rapid ≤ s < −δ
    4      Rapid worsening      s < −δ_rapid
    =====  ===================  ==============================

    These are the boundaries of the epoch classification for the oriented
    level signal, which rises as mortality improves.
    """
    s = np.asarray(slopes, dtype=float)
    return np.select([s > delta_rapid, s > delta, s >= -delta, s >= -delta_rapid],
                     [0, 1, 2, 3], default=4)


def yearly_categories(starts: np.ndarray, codes: np.ndarray, *, window: int, first_year: int,
                      last_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal category of the windows containing each calendar year (the year-assignment step of the epoch classification).

    Parameters
    ----------
    starts, codes : first year and category code of each kept window.
    window : W.
    first_year, last_year : the calendar years to classify (a country's
        first and last observed year).

    Returns
    -------
    category : (n_years,) modal code per calendar year, ``-1`` if no kept
        window contains the year.  Ties go to the category listed first in
        :data:`EPOCH_CATEGORIES` (rule of :func:`majority_label`).
    n_windows : (n_years,) number of windows containing each year.
    tied : (n_years,) True where the modal count was shared by two or more
        categories (reported so the effect of the tie-break can be judged).
    """
    n_years = last_year - first_year + 1
    covered = starts[:, None] + np.arange(window)[None, :]           # years in each window
    year_code = np.broadcast_to(np.asarray(codes)[:, None], covered.shape)
    inside = (covered >= first_year) & (covered <= last_year)
    groups = (covered - first_year)[inside]
    votes = year_code[inside]
    counts = label_counts(votes, groups, n_years, len(EPOCH_CATEGORIES))
    category = majority_label(votes, groups, n_years, len(EPOCH_CATEGORIES))
    top = counts.max(axis=1)
    tied = (top > 0) & ((counts == top[:, None]).sum(axis=1) > 1)
    return category, counts.sum(axis=1), tied


def runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of equal, non-missing codes (the epochs of the epoch classification).

    Returns a list of ``(code, first_index, last_index)``; entries equal to
    ``-1`` (no category) end a run and start none.
    """
    codes = np.asarray(codes)
    if len(codes) == 0:
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks - 1, [len(codes) - 1]])
    return [(int(codes[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True) if codes[s] >= 0]


def classify_epochs(c_idx: np.ndarray, years: np.ndarray, level: np.ndarray, e0: np.ndarray,
                    pops: Sequence[str], *, window: int, min_points: int, min_years: int,
                    delta: float, delta_rapid: float, lowess_frac: float,
                    lowess_min_points: int, lowess_iterations: int) -> dict:
    """The complete epoch classification, for every country.

    Paper: sec:clustering:epochs

    Parameters
    ----------
    c_idx : (n_obs,) country index of each observed country-year
        (:func:`observed_index` order: by country, then year).
    years : (n_obs,) calendar year of each observed country-year.
    level : (n_obs,) **oriented** level signal ``σ·G_{c,t}[1,1]``, rising
        with e₀ (:func:`level_orientation`).
    e0 : (n_obs,) life expectancy of the observed schedule (sex average).
    pops : population code of each country index.
    window, min_points : W and the minimum observed years per window.
    min_years : countries with fewer observed years are not classified.
    delta, delta_rapid : δ and δ_rapid in level units per year.
    lowess_frac, lowess_min_points, lowess_iterations : see :func:`lowess_at`.

    Returns
    -------
    dict of DataFrames
        ``series`` — one row per observed country-year of a classified
        country: ``pop, year, level, e0, level_smooth, e0_smooth``;
        ``windows`` — one row per kept window: ``pop, start, end,
        n_points, slope_level, slope_e0, category``;
        ``years`` — one row per calendar year from each country's first to
        last observed year: ``pop, year, category, n_windows, tied,
        level_smooth, e0_smooth`` (smoothed values interpolated linearly
        between observed years);
        ``epochs`` — one row per epoch: ``pop, category, label, start, end,
        n_years, e0_start, e0_end``;
    and ``skipped``, the list of populations with fewer than ``min_years``
    observed years.
    """
    series, windows, yearly, epochs, skipped = [], [], [], [], []
    for c in np.unique(c_idx):
        rows = c_idx == c
        yr = years[rows].astype(int)
        if np.any(np.diff(yr) <= 0):
            raise ValueError(f"{pops[c]}: years must be strictly increasing")
        if len(yr) < min_years:
            skipped.append(str(pops[c]))
            continue
        lv_s = lowess_at(yr, level[rows], frac=lowess_frac, min_points=lowess_min_points,
                         iterations=lowess_iterations)
        e0_s = lowess_at(yr, e0[rows], frac=lowess_frac, min_points=lowess_min_points,
                         iterations=lowess_iterations)
        series.append(pd.DataFrame({"pop": pops[c], "year": yr, "level": level[rows],
                                    "e0": e0[rows], "level_smooth": lv_s, "e0_smooth": e0_s}))

        starts, n_pts, slopes = window_slopes(yr, np.column_stack([lv_s, e0_s]), window=window,
                                              min_points=min_points)
        codes = classify_slopes(slopes[:, 0], delta, delta_rapid)
        windows.append(pd.DataFrame({"pop": pops[c], "start": starts, "end": starts + window - 1,
                                     "n_points": n_pts, "slope_level": slopes[:, 0],
                                     "slope_e0": slopes[:, 1], "category": codes}))

        cal = np.arange(yr[0], yr[-1] + 1)
        cat, n_win, tied = yearly_categories(starts, codes, window=window, first_year=int(yr[0]),
                                             last_year=int(yr[-1]))
        e0_cal = np.interp(cal, yr, e0_s)
        yearly.append(pd.DataFrame({"pop": pops[c], "year": cal, "category": cat,
                                    "n_windows": n_win, "tied": tied,
                                    "level_smooth": np.interp(cal, yr, lv_s), "e0_smooth": e0_cal}))
        for code, s, e in runs(cat):
            epochs.append({"pop": pops[c], "category": code, "label": EPOCH_CATEGORIES[code],
                           "start": int(cal[s]), "end": int(cal[e]), "n_years": int(e - s + 1),
                           "e0_start": float(e0_cal[s]), "e0_end": float(e0_cal[e])})
    return {"series": pd.concat(series, ignore_index=True),
            "windows": pd.concat(windows, ignore_index=True),
            "years": pd.concat(yearly, ignore_index=True),
            "epochs": pd.DataFrame(epochs),
            "skipped": skipped}
