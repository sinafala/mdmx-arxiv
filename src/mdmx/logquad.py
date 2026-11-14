"""The Log-Quad model (Wilmoth et al. 2012), re-estimated on training data.

Log-Quad is the standard benchmark for predicting a life table from child
mortality, optionally with adult mortality.  For each abridged age group
x (0, 1–4, 5–9, …) it models the log central death rate as a quadratic in
h = log ₅q₀ plus a second, adult-mortality dimension k:

    log ₙm_x = a_x + b_x h + c_x h² + v_x k                 (Wilmoth et al. 2012)

* With ₅q₀ alone, k = 0 (the "typical" adult mortality for that ₅q₀).
* With ₅q₀ and ₄₅q₁₅, k is chosen so that the predicted ₄₅q₁₅ equals the
  input.
* The group 1–4 is not modelled: ₄q₁ is set so that ₁q₀ and ₄q₁ together
  reproduce the input ₅q₀ exactly.

Estimation (Wilmoth et al. 2012, §3): a_x, b_x, c_x by ordinary least
squares, age group by age group, of log ₙm_x on (1, h, h²) across the
training life tables; then v_x is the leading left singular vector of the
matrix of residuals, restricted to ages 5–89 and zero elsewhere (so that k
changes neither ₅q₀ nor the oldest ages), scaled to unit length.  The
published coefficients (MortCast's ``LQcoef``) have exactly this structure.

Why re-estimate?  The published coefficients were fitted to HMD life tables
— including every population we test on — so using them in a
cross-validation would let the benchmark see the test data.  Here the model
is fitted to the training folds only, exactly like MDMx.

Life-table conventions.  To make the comparison about the *models* and not
about life-table arithmetic, every conversion uses the book's single-year
conventions (:mod:`mdmx.lifetable`): linear survivorship within each year
of age and a₀ = 0.3 for infants.  The training ₙm_x are the central death
rates of each abridged group implied by an observed single-year schedule,
and a predicted ₙm_x is spread over the single years of its group as a
constant rate.  (MortCast instead uses the Andreev–Kingkade a₀ and
Greville's ₙa_x; those choices would change the predictions slightly but
not the comparison.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import lifetable

__all__ = ["abridged_starts", "abridged_mx", "logquad_schedule", "fit_logquad", "predict_logquad"]


def abridged_starts(A: int) -> list[int]:
    """First age of each abridged group: ``[0, 1, 5, 10, …]`` below ``A``."""
    return [0, 1, *range(5, A, 5)]


def _group_bounds(starts: Sequence[int], A: int) -> list[tuple[int, int]]:
    ends = list(starts[1:]) + [A]
    return list(zip(starts, ends, strict=True))


def abridged_mx(q: np.ndarray, starts: Sequence[int], a0: float = lifetable.A0) -> np.ndarray:
    """Central death rate of each abridged age group from single-year q_x.

    ``ₙm_x = (l_x − l_{x+n}) / Σ L`` over the single years of the group,
    with ``L`` computed exactly as in :func:`mdmx.lifetable.e0_from_qx`.

    Parameters
    ----------
    q : (..., A) single-year probabilities of dying
    starts : first age of each group (see :func:`abridged_starts`)
    a0 : fraction of the first year lived by infants who die

    Returns
    -------
    (..., G) group death rates
    """
    q = np.asarray(q, dtype=float)
    A = q.shape[-1]
    lx = lifetable.survivorship(q)                             # (..., A+1)
    Lx = 0.5 * (lx[..., :-1] + lx[..., 1:])
    Lx[..., 0] = a0 * lx[..., 0] + (1.0 - a0) * lx[..., 1]
    groups = _group_bounds(starts, A)
    return np.stack([(lx[..., a] - lx[..., b]) / Lx[..., a:b].sum(axis=-1) for a, b in groups],
                    axis=-1)


def logquad_schedule(coefs: dict[str, Any], q5: np.ndarray, k: np.ndarray, A: int) -> np.ndarray:
    """Single-year q_x implied by Log-Quad for given ₅q₀ and k.

    Parameters
    ----------
    coefs : output of :func:`fit_logquad`
    q5, k : (n,) child mortality and adult-mortality parameter
    A : number of single-year ages

    Returns
    -------
    (n, A) probabilities of dying.  Age 0 from ₁m₀ with a₀; ages 1–4 share
    the single-year q that makes ₅q₀ exact; every other group's rate is
    spread evenly over its single years, ``q = m / (1 + m/2)``.
    """
    q5 = np.asarray(q5, dtype=float)[:, None]
    k = np.asarray(k, dtype=float)[:, None]
    h = np.log(q5)
    log_m = coefs["a"] + coefs["b"] * h + coefs["c"] * h ** 2 + coefs["v"] * k   # (n, G)
    m = np.exp(log_m)
    a0 = coefs["a0"]
    q = np.empty((q5.shape[0], A))
    for g, (lo, hi) in enumerate(_group_bounds(coefs["starts"], A)):
        if lo == 0:
            q[:, 0] = m[:, g] / (1.0 + (1.0 - a0) * m[:, g])
        elif lo == 1:
            continue                                   # set below from ₅q₀
        else:
            q[:, lo:hi] = (m[:, g] / (1.0 + 0.5 * m[:, g]))[:, None]
    q4_1 = 1.0 - (1.0 - q5[:, 0]) / (1.0 - q[:, 0])
    if np.any(q4_1 <= 0):
        raise ValueError("predicted 1q0 exceeds the input 5q0 for some inputs")
    q[:, 1:5] = (1.0 - (1.0 - q4_1) ** 0.25)[:, None]
    return q


def fit_logquad(q: np.ndarray, *, v_ages: tuple[int, int], a0: float = lifetable.A0) -> dict[str, Any]:
    """Estimate Log-Quad from single-year training schedules of ONE sex.

    Paper: sec:svdcomp:benchmarks

    Parameters
    ----------
    q : (n, A) single-year probabilities of dying (training life tables)
    v_ages : ``(first, stop)`` — v_x is estimated for groups starting at
        ages ``first ≤ x < stop`` (5 and 90 in Wilmoth et al. 2012) and is
        zero elsewhere
    a0 : infant a₀ used in every conversion

    Returns
    -------
    dict with ``starts`` (G,), ``a``, ``b``, ``c``, ``v`` (G,) — the 1–4
    group has a = b = c = v = 0 because it is never used — ``k_train`` (n,),
    the k of each training schedule, and ``a0``.
    """
    q = np.asarray(q, dtype=float)
    A = q.shape[1]
    starts = abridged_starts(A)
    h = np.log(lifetable.nqx(q, 0, 5))
    log_m = np.log(abridged_mx(q, starts, a0))                         # (n, G)
    design = np.column_stack([np.ones_like(h), h, h ** 2])
    modelled = np.array([s != 1 for s in starts])
    abc = np.zeros((3, len(starts)))
    abc[:, modelled], *_ = np.linalg.lstsq(design, log_m[:, modelled], rcond=None)
    resid = log_m - design @ abc
    in_v = np.array([v_ages[0] <= s < v_ages[1] for s in starts])
    U, sigma, Vt = np.linalg.svd(resid[:, in_v].T, full_matrices=False)
    sign = 1.0 if U[:, 0].sum() >= 0 else -1.0      # v_x positive: k up ⇒ adult mortality up
    v = np.zeros(len(starts))
    v[in_v] = sign * U[:, 0]
    return {"starts": np.array(starts), "a": abc[0], "b": abc[1], "c": abc[2], "v": v,
            "k_train": sign * sigma[0] * Vt[0], "a0": float(a0)}


def predict_logquad(coefs: dict[str, Any], q5: np.ndarray, q45: np.ndarray | None, A: int,
                    n_bisect: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Predicted single-year q_x from ₅q₀ (and optionally ₄₅q₁₅).

    Paper: sec:svdcomp:benchmarks

    With ``q45`` given, k is found for each row by bisection so that the
    predicted ₄₅q₁₅ equals the input.  Because v_x ≥ 0 at ages 15–59 for a
    fitted model (checked here), ₄₅q₁₅ increases with k and the bisection
    is well defined.  The search interval is the training range of k
    widened by its own length on each side; inputs whose ₄₅q₁₅ cannot be
    reached inside it end at the interval's edge, which the caller can
    detect from the returned residuals.

    Returns
    -------
    q : (n, A) probabilities of dying
    k : (n,) the adult-mortality parameter used (zeros without ``q45``)
    """
    q5 = np.asarray(q5, dtype=float)
    if q45 is None:
        k = np.zeros_like(q5)
        return logquad_schedule(coefs, q5, k, A), k
    starts = coefs["starts"]
    adult = (starts >= 15) & (starts < 60)
    if np.any(coefs["v"][adult] < 0):
        raise ValueError("v_x is negative at some adult age; 45q15 is not monotone in k")
    q45 = np.asarray(q45, dtype=float)
    k_train = coefs["k_train"]
    span = k_train.max() - k_train.min()
    lo = np.full_like(q5, k_train.min() - span)
    hi = np.full_like(q5, k_train.max() + span)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        too_high = lifetable.nqx(logquad_schedule(coefs, q5, mid, A), 15, 45) > q45
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    k = 0.5 * (lo + hi)
    return logquad_schedule(coefs, q5, k, A), k
