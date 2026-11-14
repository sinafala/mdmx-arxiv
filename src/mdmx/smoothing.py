"""Smoothing the age basis functions (paper §4.8).

The higher-order columns of the age factor matrix ``A`` pick up some
high-frequency wiggle at ages where data are noisy.  Each column except the
first is smoothed with a Gaussian kernel whose width σ varies with age:

    σ(x) = σ_max · [ s_min + (1 − s_min) · min(x / x_ramp, 1) ]   (the age-varying bandwidth of §4.8)

so that young ages (infant peak, child minimum, accident hump) are smoothed
lightly and the old-age tail more strongly.  The smoothed basis is then
re-orthonormalised and the core recomputed (chapter 04).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["bandwidths", "kernel_matrix", "smooth_age_basis"]


def bandwidths(n_ages: int, sigma_max: float, s_min: float, x_ramp: float) -> np.ndarray:
    """Kernel standard deviation σ(x) for each age x = 0 … n_ages−1.

    Paper: eq:varbw_sigma
    """
    x = np.arange(n_ages, dtype=float)
    ramp = np.minimum(x / x_ramp, 1.0) if x_ramp > 0 else np.ones_like(x)
    return sigma_max * (s_min + (1.0 - s_min) * ramp)


def kernel_matrix(sigma: np.ndarray) -> np.ndarray:
    """Row-normalised Gaussian smoothing matrix K with age-varying width.

    Paper: eq:varbw_kernel

    Row x of ``K`` holds the weights that produce the smoothed value at age
    x: ``w(x, y) ∝ exp(−½ ((y − x)/σ(x))²)``, normalised to sum to one.  A
    zero σ gives the identity row (no smoothing).  Smoothing a vector ``v``
    is then simply ``K @ v``.
    """
    n = len(sigma)
    x = np.arange(n, dtype=float)
    K = np.eye(n)
    for i in np.flatnonzero(sigma > 0):
        w = np.exp(-0.5 * ((x - i) / sigma[i]) ** 2)
        K[i] = w / w.sum()
    return K


def smooth_age_basis(U_age: np.ndarray, components: Sequence[Sequence[float]], rest_sigma: float,
                     preserve: int, orth_tol: float) -> tuple[np.ndarray, dict]:
    """Smooth columns 2… of the age basis and re-orthonormalise.

    Paper: sec:tucker:bsmooth

    Parameters
    ----------
    U_age : (A, r₂) orthonormal age basis from the decomposition
    components : ``[x_ramp, s_min, sigma_max]`` for columns 2, 3, … in turn
    rest_sigma : constant σ for all remaining columns
    preserve : ages 0 … preserve−1 are reset to their unsmoothed values
        before re-orthonormalisation (the infant–child discontinuity is real)
    orth_tol : re-orthonormalise (QR) if max|AᵀA − I| exceeds this

    Returns
    -------
    The smoothed, orthonormal basis and a diagnostics dict (orthonormality
    error before/after, largest change at the preserved ages caused by QR).
    """
    A, r = U_age.shape
    out = U_age.copy()
    for k in range(1, r):                            # column 0 is left as is
        if k - 1 < len(components):
            x_ramp, s_min, sigma_max = components[k - 1]
            sig = bandwidths(A, sigma_max, s_min, x_ramp)
        else:
            sig = np.full(A, rest_sigma)
        out[:, k] = kernel_matrix(sig) @ U_age[:, k]
        out[:preserve, k] = U_age[:preserve, k]
    orth_before = float(np.max(np.abs(out.T @ out - np.eye(r))))
    if orth_before > orth_tol:
        Q, R = np.linalg.qr(out)
        out = Q * np.sign(np.diag(R))               # keep each column's orientation
    orth_after = float(np.max(np.abs(out.T @ out - np.eye(r))))
    return out, {
        "orth_error_before_qr": orth_before,
        "orth_error_after_qr": orth_after,
        # With preserve = 0 no age is reset, so there is no change to measure.
        "max_change_preserved_ages": (float(np.max(np.abs(out[:preserve] - U_age[:preserve])))
                                      if preserve > 0 else 0.0),
    }
