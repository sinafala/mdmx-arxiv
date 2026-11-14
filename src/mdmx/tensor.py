"""Tensor algebra and the Tucker decomposition (paper §4).

Notation (paper §4 and the Notation appendix)
---------------------------------------------
The mortality tensor ``M`` has shape ``(S, A, C, T)`` = (sex, age,
country, year) and holds logit(q_x).  A Tucker decomposition writes

    M ≈ G ×₁ S ×₂ A ×₃ C ×₄ T   (the Tucker decomposition of the mortality tensor, §4.2)

with orthonormal *factor matrices* ``S (S×r₁)``, ``A (A×r₂)``, ``C (C×r₃)``,
``T (T×r₄)`` and a small *core tensor* ``G (r₁×r₂×r₃×r₄)``.  In code the
factor matrices are held in a list ``U = [U_sex, U_age, U_country, U_year]``.

The mode-n product ``X ×ₙ B`` multiplies every mode-n fibre of ``X`` by the
matrix ``B``; :func:`mode_product` implements it.  :func:`project` computes
``X ×₁ U₁ᵀ ×₂ U₂ᵀ …`` (data → core) and :func:`reconstruct` computes
``G ×₁ U₁ ×₂ U₂ …`` (core → data).

Missing data (paper §3.3 and §4.7)
----------------------------------
About 70% of the country-year cells of the HMD tensor are unobserved (a
series has not started yet, or the year is a war/pandemic year set aside).
Filling them with country means would let those made-up values shape the
decomposition (and a core computed from a differently weighted tensor than
the factors would not match them).  The module therefore separates two jobs:

1. **Shared sex and age bases** (:func:`sex_age_bases`).  Each observed
   country-year is a complete sex × age schedule, so the sex and age
   unfoldings restricted to observed country-years need no imputation.  Their
   leading singular vectors give ``S`` and ``A`` (rank by the τ rule).
2. **Effective cores** (:func:`effective_cores`).  Every observed schedule is
   projected onto those bases: ``G_{c,t} = Sᵀ M_{c,t} A``.  All downstream
   analyses (clustering, trajectories, fitting, prediction, forecasting) use
   these, so each observed country-year is represented as exactly as the age
   rank allows, with no smoothing across countries or years.
3. **A completion model** (:func:`complete_cores`).  Unobserved country-years
   are filled by a Tucker model of the country and year modes fitted by EM
   to the observed effective cores, with ranks chosen by hiding observed
   country-years and predicting them (:func:`holdout_rank_search`).  This is
   the article's completion model for unobserved country-years (§3.3).

The general-purpose :func:`em_tucker` (all four modes estimated by EM) is
kept for comparison in the book.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "unfold", "fold", "mode_product", "project", "reconstruct", "leading_left_vectors",
    "select_rank", "hosvd", "em_tucker", "fix_signs", "TuckerFit", "country_mean_impute",
    "sex_age_bases", "effective_cores", "complete_cores", "holdout_rank_search", "schedules_from_cores",
]


# ══════════════════════════════════════════════════════════════════════════════
# Basic multilinear algebra
# ══════════════════════════════════════════════════════════════════════════════

def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """Mode-``mode`` unfolding: rows index that mode, columns everything else.

    Uses the same column ordering as ``np.moveaxis(X, mode, 0).reshape(...)``
    throughout the project, so :func:`fold` is its exact inverse.
    """
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def fold(Xn: np.ndarray, mode: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`unfold` for a tensor of the given full ``shape``."""
    moved = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(Xn.reshape(moved), 0, mode)


def mode_product(X: np.ndarray, B: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n product ``X ×ₙ B``: replaces dimension ``mode`` (size I) by B's rows.

    Paper: eq:nmode_product

    ``B`` has shape ``(J, I)``; the result has size ``J`` along ``mode``.
    Implemented with ``tensordot`` (no explicit unfolding copy).
    """
    out = np.tensordot(B, X, axes=(1, mode))          # new axis first
    return np.moveaxis(out, 0, mode)


def project(X: np.ndarray, U: Sequence[np.ndarray]) -> np.ndarray:
    """Core of ``X`` in the bases ``U``: ``X ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ ×₄ U₄ᵀ``.

    Paper: eq:core_projection
    """
    G = X
    for mode, Un in enumerate(U):
        G = mode_product(G, Un.T, mode)
    return G


def reconstruct(G: np.ndarray, U: Sequence[np.ndarray]) -> np.ndarray:
    """Tensor from core and factors: ``G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄``.

    Paper: eq:tucker_mortality
    """
    X = G
    for mode, Un in enumerate(U):
        X = mode_product(X, Un, mode)
    return X


def leading_left_vectors(X: np.ndarray, mode: int) -> tuple[np.ndarray, np.ndarray]:
    """All left singular vectors and singular values of the mode unfolding.

    Computed from the eigen-decomposition of the small Gram matrix
    ``X₍ₙ₎ X₍ₙ₎ᵀ`` (size I_n × I_n, at most 274 × 274 here), which is much
    faster than an SVD of the wide unfolding and gives the same vectors.
    Singular values are returned in decreasing order.
    """
    Xn = unfold(X, mode)
    gram = Xn @ Xn.T
    evals, evecs = np.linalg.eigh(gram)                # ascending order
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    return evecs[:, order], np.sqrt(evals)


def fix_signs(U: np.ndarray) -> np.ndarray:
    """Make each column's largest-magnitude entry positive.

    Singular vectors are defined only up to sign, and different linear-algebra
    libraries may return either sign.  Fixing a convention makes every
    downstream result (loadings, core entries, epoch slopes) reproducible
    across machines.
    """
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def select_rank(sigma: np.ndarray, tau: float, r_min: int, r_max: int) -> tuple[int, np.ndarray]:
    """Smallest rank whose cumulative variance share reaches ``tau``.

    Paper: eq:variance_fraction, eq:rank_selection, eq:shared_bases

    Returns the rank (bounded to ``[r_min, r_max]``) and the cumulative
    variance curve ``sum_{i<=r} σ_i² / sum_i σ_i²``.
    """
    var = sigma ** 2
    cumvar = np.cumsum(var) / var.sum()
    r = int(np.searchsorted(cumvar, tau) + 1)
    return int(min(max(r, r_min), r_max)), cumvar


def country_mean_impute(M: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill unobserved country-years with the country's mean schedule.

    Paper: eq:imputation, eq:completion

    Parameters
    ----------
    M : (S, A, C, T) tensor; values in unobserved cells are ignored
    observed : (C, T) boolean mask of observed (non-exceptional) country-years
    """
    X = M.copy()
    for c in range(M.shape[2]):
        obs_t = observed[c]
        mean_c = M[:, :, c, obs_t].mean(axis=-1)
        X[:, :, c, ~obs_t] = mean_c[..., None]
    return X


# ══════════════════════════════════════════════════════════════════════════════
# HOSVD and EM Tucker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TuckerFit:
    """Result of a Tucker decomposition.

    Attributes
    ----------
    U : factor matrices [U_sex (S×r₁), U_age (A×r₂), U_country (C×r₃), U_year (T×r₄)]
    G : core tensor (r₁, r₂, r₃, r₄)
    ranks : (r₁, r₂, r₃, r₄)
    sigma : per-mode singular values of the (completed) tensor
    cumvar : per-mode cumulative variance curves
    completed : the completed tensor (observed cells = data, others = model)
    history : per-iteration convergence diagnostics (EM only)
    """
    U: list[np.ndarray]
    G: np.ndarray
    ranks: tuple[int, ...]
    sigma: list[np.ndarray]
    cumvar: list[np.ndarray]
    completed: np.ndarray
    history: list[dict] = field(default_factory=list)

    def reconstruct(self) -> np.ndarray:
        return reconstruct(self.G, self.U)


def hosvd(X: np.ndarray, ranks: Sequence[int] | None = None, *, tau: float | None = None,
          rank_bounds: Sequence[tuple[int, int]] | None = None) -> TuckerFit:
    """Truncated higher-order SVD of a complete tensor.

    Paper: eq:mode_svd, sec:tucker:definition

    Either give ``ranks`` explicitly, or give ``tau`` (variance threshold)
    and ``rank_bounds`` (one ``(r_min, r_max)`` per mode) to select them.
    """
    U_full, sigma, cumvar, chosen = [], [], [], []
    for mode in range(X.ndim):
        vecs, sv = leading_left_vectors(X, mode)
        U_full.append(vecs)
        sigma.append(sv)
        r, cv = select_rank(sv, tau if tau is not None else 1.0, *(rank_bounds[mode] if rank_bounds else (1, len(sv))))
        cumvar.append(cv)
        chosen.append(r)
    if ranks is None:
        ranks = chosen
    U = [fix_signs(U_full[m][:, :ranks[m]]) for m in range(X.ndim)]
    G = project(X, U)
    return TuckerFit(U=U, G=G, ranks=tuple(int(r) for r in ranks), sigma=sigma, cumvar=cumvar,
                     completed=X)


def em_tucker(M: np.ndarray, observed: np.ndarray, *, tau: float,
              rank_bounds: Sequence[tuple[int, int]], max_iter: int = 500, tol: float = 1e-6,
              max_rank_rounds: int = 6, verbose: bool = True) -> TuckerFit:
    """Tucker decomposition fitted to the observed country-years only (EM).

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor; only cells with ``observed`` are used
    observed : (C, T) boolean mask of observed, non-exceptional country-years
    tau : per-mode variance threshold for rank selection (the τ rule of :func:`select_rank`)
    rank_bounds : ``[(r_min, r_max)]`` for each of the four modes
    max_iter : maximum EM iterations per rank round
    tol : convergence threshold on the relative change of the imputed cells
    max_rank_rounds : ranks are re-selected from the completed tensor after
        each EM run; the procedure stops when the ranks no longer change

    Returns
    -------
    TuckerFit whose ``completed`` tensor equals ``M`` on observed cells and
    the model reconstruction elsewhere, with ``G = project(completed, U)``.

    Notes
    -----
    Rank selection and imputation depend on each other: the ranks are chosen
    by the τ rule on the completed tensor, and the completion depends on the
    ranks.  We therefore alternate: (i) choose ranks from the current
    completed tensor; (ii) run EM to convergence at those ranks; repeat until
    the ranks are stable.  Every round is recorded in ``history``.
    """
    obs4 = np.broadcast_to(observed[None, None, :, :], M.shape)
    missing = ~obs4
    X = country_mean_impute(M, observed)            # starting point: country-mean imputation
    history: list[dict] = []
    ranks = None
    for rnd in range(max_rank_rounds):
        new_ranks = hosvd(X, tau=tau, rank_bounds=rank_bounds).ranks
        if ranks is not None and new_ranks == ranks:
            break
        ranks = new_ranks
        if verbose:
            print(f"    rank round {rnd + 1}: ranks = {ranks}")
        for it in range(max_iter):
            fit = hosvd(X, ranks=ranks)
            Xhat = fit.reconstruct()
            old = X[missing]
            change = np.linalg.norm(Xhat[missing] - old) / max(np.linalg.norm(old), 1e-12)
            X[missing] = Xhat[missing]
            obs_rmse = float(np.sqrt(np.mean((Xhat[obs4] - M[obs4]) ** 2)))
            history.append({"round": rnd + 1, "iter": it + 1, "ranks": list(ranks),
                            "rel_change_missing": float(change), "rmse_observed": obs_rmse})
            if verbose and (it % 25 == 0):
                print(f"      iter {it + 1:4d}: Δ(imputed) = {change:.2e}, RMSE(observed) = {obs_rmse:.4f}")
            if change < tol:
                break
    final = hosvd(X, tau=tau, rank_bounds=rank_bounds) if ranks is None else hosvd(X, ranks=ranks)
    # Keep the τ-curves of the completed tensor for reporting.
    final.history = history
    final.completed = X
    return final


# ══════════════════════════════════════════════════════════════════════════════
# The MDMx decomposition (paper §4.7): shared bases from observed data,
# plus a low-rank completion model for the country and year modes
# ══════════════════════════════════════════════════════════════════════════════

def sex_age_bases(M: np.ndarray, observed: np.ndarray, *, tau: float,
                  age_bounds: tuple[int, int]) -> dict:
    """Shared sex and age factor matrices, estimated from observed schedules only.

    Paper: eq:shared_bases

    Every observed country-year contributes one complete (S × A) schedule,
    so the sex and age unfoldings restricted to observed country-years are
    complete matrices — no imputation is needed to compute their SVDs.

    Signs follow :func:`fix_signs`, except the first age column, which is
    oriented like the schedules themselves (its inner product with the mean
    observed logit(q_x) schedule is positive): it then has the shape of a
    mortality schedule, negative at almost every age, and the core entries on
    it rise as mortality falls.  Any orientation gives the same schedules;
    this one makes the first basis function read like the data.

    Returns ``U_sex`` (S × S, full rank), ``U_age`` (A × r₂, r₂ by the τ
    rule within ``age_bounds``), and the singular values / cumulative
    variance curves of both unfoldings.
    """
    Xobs = M[:, :, observed]                                 # (S, A, n_observed)
    out = {}
    for mode, name in ((0, "sex"), (1, "age")):
        vecs, sv = leading_left_vectors(Xobs, mode)
        if name == "sex":
            r = Xobs.shape[0]
            cumvar = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        else:
            r, cumvar = select_rank(sv, tau, *age_bounds)
        U = fix_signs(vecs[:, :r])
        if name == "age" and U[:, 0] @ Xobs.mean(axis=(0, 2)) < 0:
            U[:, 0] = -U[:, 0]
        out[f"U_{name}"] = U
        out[f"sigma_{name}"] = sv
        out[f"cumvar_{name}"] = cumvar
    return out


def effective_cores(M: np.ndarray, observed: np.ndarray, U_sex: np.ndarray,
                    U_age: np.ndarray) -> np.ndarray:
    """Effective core ``G_{c,t} = Sᵀ M_{c,t} A`` of every observed country-year.

    Paper: eq:Gct

    Returns an array (r₁, r₂, C, T) with NaN in unobserved country-years.
    For an observed country-year this is the exact projection of its
    schedule onto the shared sex–age
    basis.
    """
    G = np.full((U_sex.shape[1], U_age.shape[1]) + M.shape[2:], np.nan)
    G[:, :, observed] = np.einsum("sp,saN,aq->pqN", U_sex, M[:, :, observed], U_age)
    return G


def complete_cores(Y: np.ndarray, observed: np.ndarray, r_country: int, r_year: int, *,
                   max_iter: int, tol: float, init: np.ndarray | None = None
                   ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[dict]]:
    """Fill unobserved country-years with a low-rank country × year model (EM).

    Paper: eq:completion

    ``Y`` is the (r₁, r₂, C, T) array of effective cores, valid where
    ``observed``.  The model is a Tucker decomposition of ``Y`` with the
    first two modes at full rank and ranks ``(r_country, r_year)`` in the
    country and year modes.  EM: start from country means, then repeatedly
    (i) fit the truncated decomposition to the completed array and (ii)
    replace only the unobserved slices by the fit, until the relative change
    of the filled values is below ``tol``.

    Because the sex and age bases are orthonormal, fitting ``Y`` in this
    reduced space is equivalent to fitting the full logit(q_x) tensor with
    those bases held fixed — at a fraction of the cost.

    Returns the completed array, the factors ``[U_country, U_year]``, the
    core (r₁, r₂, r_country, r_year), and per-iteration diagnostics.
    """
    X = country_mean_impute(np.nan_to_num(Y), observed) if init is None else init.copy()
    miss = ~np.broadcast_to(observed[None, None], X.shape)
    history = []
    for it in range(max_iter):
        U_c = fix_signs(leading_left_vectors(X, 2)[0][:, :r_country])
        U_t = fix_signs(leading_left_vectors(X, 3)[0][:, :r_year])
        core = mode_product(mode_product(X, U_c.T, 2), U_t.T, 3)
        Xh = mode_product(mode_product(core, U_c, 2), U_t, 3)
        old = X[miss]
        change = float(np.linalg.norm(Xh[miss] - old) / max(np.linalg.norm(old), 1e-12))
        X[miss] = Xh[miss]
        history.append({"iter": it + 1, "rel_change": change})
        if change < tol:
            break
    return X, [U_c, U_t], core, history


def holdout_rank_search(Y: np.ndarray, observed: np.ndarray, grid: Sequence[tuple[int, int]],
                        *, share: float, seed: int, max_iter: int, tol: float,
                        loss: callable) -> list[dict]:
    """Choose the completion ranks by hiding observed country-years.

    Paper: eq:completion

    A random ``share`` of observed country-years is hidden; for each
    ``(r_country, r_year)`` in ``grid`` the completion model is fitted to the
    rest, and ``loss(Y_completed, hidden_mask)`` measures how well the hidden
    country-years are predicted.  Returns one dict per grid point.
    """
    rng = np.random.default_rng(seed)
    ci, ti = np.nonzero(observed)
    pick = rng.choice(len(ci), size=max(1, int(round(share * len(ci)))), replace=False)
    hidden = np.zeros_like(observed)
    hidden[ci[pick], ti[pick]] = True
    train = observed & ~hidden
    rows = []
    for rc, rt in grid:
        Xc, _, _, hist = complete_cores(Y, train, rc, rt, max_iter=max_iter, tol=tol)
        rows.append({"r_country": rc, "r_year": rt, "iterations": len(hist), **loss(Xc, hidden)})
    return rows


def schedules_from_cores(G: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray) -> np.ndarray:
    """Logit(q_x) schedules from effective cores: ``M_{c,t} = S G_{c,t} Aᵀ``.

    Paper: eq:reconstruction

    ``G`` may be (r₁, r₂) for one country-year, or (r₁, r₂, …) for many; the
    result has shape (S, A, …).
    """
    return np.einsum("sp,pq...,aq->sa...", U_sex, G, U_age)
