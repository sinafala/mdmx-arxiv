"""Higher-order orthogonal iteration in the completion model: the check of the truncated HOSVD.

The completion model (:func:`mdmx.tensor.complete_cores`) computes its country and year factor
matrices by a truncated HOSVD at every EM iteration.  A truncated HOSVD is quasi-optimal, not
optimal; higher-order orthogonal iteration (alternating least squares; Kroonenberg and de Leeuw
1980, De Lathauwer, De Moor and Vandewalle 2000b) can lower the error with the same ranks.  This
module fits the same EM with that update instead, so that the article can compare the two on the
observed country-years and on held-out ones.  The completion model itself does not use it.
"""

from __future__ import annotations

import numpy as np

from mdmx.tensor import country_mean_impute, fix_signs, leading_left_vectors, mode_product


def complete_cores_hooi(Y: np.ndarray, observed: np.ndarray, r_country: int, r_year: int, *,
                        max_iter: int, tol: float, init: np.ndarray | None = None
                        ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[dict]]:
    """The completion model's EM with higher-order orthogonal iteration in place of the HOSVD.

    Paper: sec:tucker:definition

    The EM of :func:`mdmx.tensor.complete_cores` (eq:completion): start from the population means
    of the observed cores, fit the model, replace only the unobserved slices by the fit, and repeat
    until the filled values change by less than ``tol`` (relative) or ``max_iter`` is reached.  The
    fit differs: instead of recomputing the country and year factor matrices as the leading left
    singular vectors of the filled array's unfoldings, each iteration updates them by one sweep of
    higher-order orthogonal iteration, started from the previous iteration's factors (from the
    HOSVD of the starting array in the first).  The sweep sets the country factor matrix to the
    leading left singular vectors of the country unfolding of the array projected on the year
    factors, then the year factor matrix likewise given the new country factors; each step is the
    least-squares optimum given the other factor matrix, so a sweep never moves the model away from
    the filled array.  It follows that the residual of the observed cores never increases from one
    iteration to the next (the HOSVD update has no such guarantee).

    Parameters
    ----------
    Y : (r₁, r₂, C, T) array
        Effective cores; valid where ``observed`` (other values are ignored).
    observed : (C, T) bool array
        Observed country-years.
    r_country, r_year : int
        Ranks of the country and year modes.
    max_iter, tol : int, float
        As in :func:`mdmx.tensor.complete_cores`.
    init : (r₁, r₂, C, T) array, optional
        Starting array instead of the population means.

    Returns
    -------
    completed : (r₁, r₂, C, T) array
        Observed cores unchanged, unobserved ones the model's prediction.
    factors : [U_country (C, r_country), U_year (T, r_year)]
        Orthonormal factor matrices.
    core : (r₁, r₂, r_country, r_year) array
    history : list of dict
        Per iteration: ``iter``, ``rel_change`` (relative change of the filled values) and
        ``resid_obs`` (Frobenius norm of the observed cores minus the fit, before the fill).
    """
    X = country_mean_impute(np.nan_to_num(Y), observed) if init is None else init.copy()
    obs = np.broadcast_to(observed[None, None], X.shape)
    miss = ~obs
    U_c = leading_left_vectors(X, 2)[0][:, :r_country]
    U_t = leading_left_vectors(X, 3)[0][:, :r_year]
    history = []
    for it in range(max_iter):
        U_c = fix_signs(leading_left_vectors(mode_product(X, U_t.T, 3), 2)[0][:, :r_country])
        U_t = fix_signs(leading_left_vectors(mode_product(X, U_c.T, 2), 3)[0][:, :r_year])
        core = mode_product(mode_product(X, U_c.T, 2), U_t.T, 3)
        Xh = mode_product(mode_product(core, U_c, 2), U_t, 3)
        old = X[miss]
        change = float(np.linalg.norm(Xh[miss] - old) / max(np.linalg.norm(old), 1e-12))
        resid = float(np.linalg.norm((X - Xh)[obs]))
        X[miss] = Xh[miss]
        history.append({"iter": it + 1, "rel_change": change, "resid_obs": resid})
        if change < tol:
            break
    return X, [U_c, U_t], core, history


def observed_residual(completed: np.ndarray, observed: np.ndarray, factors: list[np.ndarray],
                      core: np.ndarray) -> float:
    """Frobenius norm of the observed cores minus a completion model's fit.

    ``completed`` (r₁, r₂, C, T) holds the observed cores where ``observed`` (C, T); ``factors`` and
    ``core`` are a fitted model as returned by :func:`mdmx.tensor.complete_cores` or
    :func:`complete_cores_hooi`.  The fit is ``core ×₃ U_country ×₄ U_year`` (eq:completion).
    """
    U_c, U_t = factors
    fit = mode_product(mode_product(core, U_c, 2), U_t, 3)
    obs = np.broadcast_to(observed[None, None], completed.shape)
    return float(np.linalg.norm((completed - fit)[obs]))
