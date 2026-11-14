"""Life-table arithmetic — the single implementation used everywhere.

All functions are vectorised: the age axis is always the **last** axis, so
``qx`` may be one schedule ``(A,)``, a sex pair ``(2, A)``, or a whole
tensor slice ``(S, A, C, T)`` moved so that age is last.

Conventions (§2.3, the e₀ of the life table)
------------------------------
* Single-year ages ``x = 0, 1, …, A−1`` (A = 110: ages 0–109).
* ``q_x`` is the probability of dying between exact ages x and x+1.
* Person-years lived in each interval use the usual linear assumption
  ``L_x = (l_x + l_{x+1}) / 2`` for x ≥ 1, and for infants
  ``L_0 = a_0 l_0 + (1 − a_0) l_1`` with a fixed ``a_0`` (default 0.3).
* e₀ is the sum of ``L_x`` over ages 0–109; survivors to exact age 110 are
  not followed further.  Against the HMD's own e₀ the two simplifications
  move e₀ very little (chapter d6 reports the median difference over the
  curated life tables, and the few tables of small populations, with high
  survival to 110, where it exceeds 0.01 years).

Using one function for every e₀ in the project — data summaries,
trajectories, fitting, forecasting, evaluation, and the Shiny app — means
an e₀ computed in one place can be compared with an e₀ computed anywhere
else.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Q_MIN_DEFAULT", "logit", "expit", "qx_from_mx", "survivorship", "e0_from_qx",
    "e0_from_logit", "life_table_from_qx", "life_table_from_mx", "nqx",
]

Q_MIN_DEFAULT = 1e-8   # floor/cap used before the logit (q_min of §2.5)
#: The book's a₀, the average fraction of the first year lived by infants who die: in L₀ of every e₀
#: and in every conversion of a forecast m₀ to q₀ (one value everywhere; decision R1, 2026-11-01).
#: HMD's own a₀ is 0.29–0.31 when infant mortality is high, where the choice matters for e₀.
#: Mirrored by ``[data] a0`` in ``config/params.toml`` (a test keeps them equal).
A0 = 0.3


def logit(q: np.ndarray, q_min: float = Q_MIN_DEFAULT) -> np.ndarray:
    """``log(q / (1 − q))`` after clipping ``q`` to ``[q_min, 1 − q_min]``.

    Paper: eq:logit, eq:qx_floor

    The clip keeps the transform finite: HMD reports ``q_x = 0`` in a few
    small-population cells (no deaths), which would otherwise give −∞.
    """
    q = np.clip(np.asarray(q, dtype=float), q_min, 1.0 - q_min)
    return np.log(q) - np.log1p(-q)


def expit(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`logit`: ``1 / (1 + exp(−z))`` (numerically stable)."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def qx_from_mx(mx: np.ndarray, ax: np.ndarray) -> np.ndarray:
    """Convert central death rates to probabilities of dying (1-year ages).

    ``q_x = m_x / (1 + (1 − a_x) m_x)`` — the standard identity relating the
    death rate, the probability of dying and the average fraction of the
    year lived by those who die (Preston et al. 2001, Eq. 3.5).
    """
    mx = np.asarray(mx, dtype=float)
    ax = np.asarray(ax, dtype=float)
    return np.clip(mx / (1.0 + (1.0 - ax) * mx), 0.0, 1.0)


def conversion_ax(n_ages: int, ax: float = 0.5, a0: float = A0) -> np.ndarray:
    """The a_x of the book's m_x → q_x conversion: ``a0`` at age 0, ``ax`` above.

    Parameters
    ----------
    n_ages : number of single-year ages (A)
    ax : a_x at ages 1 … A − 1 (0.5: deaths spread evenly over the year of age)
    a0 : a_x at age 0 (the book's :data:`A0`; infant deaths cluster early in the year)

    Returns
    -------
    (A,) array, to broadcast over schedules ``(..., A)`` in :func:`qx_from_mx`.
    """
    out = np.full(n_ages, float(ax))
    out[0] = a0
    return out


def survivorship(qx: np.ndarray) -> np.ndarray:
    """Return ``l_x`` for x = 0 … A (one more age than ``qx``), with ``l_0 = 1``."""
    qx = np.asarray(qx, dtype=float)
    ones = np.ones(qx.shape[:-1] + (1,))
    return np.concatenate([ones, np.cumprod(1.0 - qx, axis=-1)], axis=-1)


def e0_from_qx(qx: np.ndarray, a0: float = A0) -> np.ndarray:
    """Life expectancy at birth from ``q_x``, vectorised.

    Paper: eq:e0

    Parameters
    ----------
    qx : array (..., A) of probabilities of dying, ages 0 … A−1
    a0 : average fraction of the first year lived by infants who die

    Returns
    -------
    array (...) of e₀ values in years
    """
    lx = survivorship(qx)                       # (..., A+1)
    Lx = 0.5 * (lx[..., :-1] + lx[..., 1:])     # linear within each year
    Lx[..., 0] = a0 * lx[..., 0] + (1.0 - a0) * lx[..., 1]
    return Lx.sum(axis=-1)


def e0_from_logit(z: np.ndarray, a0: float = A0) -> np.ndarray:
    """e₀ from a logit(q_x) schedule (age on the last axis).

    Paper: eq:forward

    Returns e₀ for each sex (each row of the leading axes); the article's e₀ of a
    schedule is the average of the sexes', which the callers take.
    """
    return e0_from_qx(expit(z), a0=a0)


def ex_from_qx(qx: np.ndarray, age: int) -> np.ndarray:
    """Life expectancy at ``age`` (≥ 1) from ``q_x``, linear within each year as :func:`e0_from_qx`.

    Parameters
    ----------
    qx : array (..., A) of probabilities of dying, ages 0 … A−1; age : the age, 1 ≤ age < A

    Returns
    -------
    array (...): ``Σ_{x ≥ age} L_x / l_age``.
    """
    if not 1 <= age < np.shape(qx)[-1]:
        raise ValueError(f"age must be between 1 and {np.shape(qx)[-1] - 1}, got {age}")
    lx = survivorship(qx)
    Lx = 0.5 * (lx[..., :-1] + lx[..., 1:])
    return Lx[..., age:].sum(axis=-1) / lx[..., age]


def ex_from_logit(z: np.ndarray, age: int) -> np.ndarray:
    """Life expectancy at ``age`` from a logit(q_x) schedule (:func:`ex_from_qx`)."""
    return ex_from_qx(expit(z), age)


def nqx(qx: np.ndarray, start: int, width: int) -> np.ndarray:
    """Probability of dying between exact ages ``start`` and ``start + width``.

    Examples: ``nqx(q, 0, 5)`` is ₅q₀ (child mortality) and
    ``nqx(q, 15, 45)`` is ₄₅q₁₅ (adult mortality).
    """
    qx = np.asarray(qx, dtype=float)
    return 1.0 - np.prod(1.0 - qx[..., start:start + width], axis=-1)


def life_table_from_qx(qx: np.ndarray, a0: float = A0, radix: float = 100_000) -> dict[str, np.ndarray]:
    """Full life table implied by ``q_x`` under the conventions of :func:`e0_from_qx`.

    Returns a dict with ``qx, lx, dx, Lx, Tx, ex`` (each shape ``(..., A)``).
    ``ex[..., 0]`` equals :func:`e0_from_qx` exactly, so a table displayed in
    the app always agrees with the e₀ used everywhere else.
    """
    qx = np.asarray(qx, dtype=float)
    l_full = survivorship(qx)
    lx = l_full[..., :-1]
    dx = lx - l_full[..., 1:]
    Lx = 0.5 * (lx + l_full[..., 1:])
    Lx[..., 0] = a0 * lx[..., 0] + (1.0 - a0) * l_full[..., 1]
    Tx = np.flip(np.cumsum(np.flip(Lx, axis=-1), axis=-1), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(lx > 0, Tx / lx, 0.0)
    return {"qx": qx, "lx": radix * lx, "dx": radix * dx, "Lx": radix * Lx,
            "Tx": radix * Tx, "ex": ex}


def life_table_from_mx(mx: np.ndarray, ax: np.ndarray, radix: float = 1.0) -> dict[str, np.ndarray]:
    """Life table from central death rates ``m_x`` and ``a_x`` (1-year ages).

    The last age is treated as **closed** (``q`` from the usual identity) —
    callers pass ages 0–109 only.  The pooling step itself converts the pooled
    rates with :func:`qx_from_mx`; this builds the whole table from rates.
    """
    qx = qx_from_mx(mx, ax)
    l_full = survivorship(qx)
    lx = l_full[..., :-1]
    dx = lx - l_full[..., 1:]
    Lx = l_full[..., 1:] + np.asarray(ax, dtype=float) * dx
    Tx = np.flip(np.cumsum(np.flip(Lx, axis=-1), axis=-1), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(lx > 0, Tx / lx, 0.0)
    return {"qx": qx, "lx": radix * lx, "dx": radix * dx, "Lx": radix * Lx,
            "Tx": radix * Tx, "ex": ex}
