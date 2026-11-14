"""Li–Lee coherent forecasts: a benchmark of the forecast evaluation (chapter 12).

Li & Lee (2005, *Demography* 42(3), "Coherent mortality forecasts for a
group of populations: an extension of the Lee–Carter method") forecast a
group of populations whose mortality is expected to stay close by giving
them one common Lee–Carter factor plus a factor of their own that is
forecast to stop drifting away: the *augmented common factor* (ACF) model.
The evaluation reports it beside the comparison, never ranked, at the
development origins only, in two forms (Sam, 2026-11-01):

* **the group form, per sex** (:func:`group_forecasts`): the group is the
  origin's populations, fitted separately for females and males;
* **the sex-coherent form, per population** (:func:`sex_coherent_forecasts`):
  the group is the two sexes of one population.

Both are the same model with a different group, fitted by :func:`fit_acf`
and forecast by :func:`forecast_acf`.

The model, exactly as implemented
---------------------------------
For member i of a group (a population, or one sex of a population), age x
and calendar year t of the fitting window,

    log m_{x,t,i} = a_{x,i} + B_x K_t + b_{x,i} k_{t,i} + ε_{x,t,i}.

1. **The common factor (B, K)** (:func:`common_factor`) is a Lee–Carter fit
   to the group's aggregate rate m_{x,t} over the window, from the *core*
   members only (members with rates in every year of the window):
   ``A_x`` is the time-mean of log m_{x,t}, and (B, K⁰) is the first term of
   the singular value decomposition of ``log m_{x,t} − A_x`` (the first
   stage); K is K⁰ refitted to the data in a second stage (item 4).  The
   aggregate is the combined population's death rate
   ``Σ_i m_{x,t,i} E_{x,t,i} / Σ_i E_{x,t,i}``, which is ``ΣD/ΣE`` when
   m = D/E (``aggregate = "exposure"``, Li & Lee's group rate), or the mean
   of the members' log rates (``"logmean"``).  At an age and year with no
   exposure in the whole core (the open age group of a small population in
   its early years, in the sex-coherent form) the members' rates are
   averaged with equal weights, with a warning.
2. **a_{x,i}** is the mean over the member's own years of
   ``log m_{x,t,i} − B_x K⁰_t``, with the first-stage K⁰.  For a member with
   every year of the window this is Li & Lee's eq. (4), the time-mean of
   log m_{x,t,i}, because K⁰ is centred over the window; for a member with
   fewer years (group form) it is the least-squares a_{x,i} of their
   eq. (3) given B and K⁰.  As in Lee–Carter, where a_x stays the time mean
   while k_t is refitted, a is not refitted after the second stage: a
   refitted a would take back the level the second stage puts into K.
3. **(b_i, k_i)** is the first SVD term of the member's residual
   ``log m_{x,t,i} − a_{x,i} − B_x K_t`` over its own years, with the
   refitted K (Li & Lee's "residual matrix of the common factor model").
4. **Second stage** (``refit``; section "Second stage" below).
   ``"deaths"``: each K_t is refitted so that the core's
   implied deaths equal its observed total deaths, then each k_{t,i} so that
   member i's implied deaths equal its own (:func:`_refit_to_deaths`).
   ``"deaths_common"``: K_t only, k_i not refitted (Li & Lee's structure
   with Lee & Carter's criterion).  ``"e0"`` (the default): Li & Lee (2005) as published —
   K_t refitted to the life expectancy of the aggregate rate
   (:func:`_refit_to_e0`), k_i not refitted.  ``"none"``: no second stage
   (K = K⁰, k_i as the SVD gives it).
5. **Normalisation** (Lee & Carter 1992's): ``Σ_x B_x = 1`` and
   ``Σ_x b_{x,i} = 1``; K⁰ and, without a second stage, each k_i have mean
   zero over the years they are fitted to (the rows of the matrices
   decomposed are centred); the refitted K and k_i are not re-centred, as in
   ``demography::lca``.  The normalisation also fixes the signs.  If a
   residual's age pattern sums to (numerically) zero, its normalisation is
   undefined: b_i is then the unit singular vector with the sign of
   :func:`mdmx.tensor.fix_signs`, and a warning is recorded.  The forecasts
   depend on the products B K and b_i k_i only, which neither normalisation
   changes.
6. **Forecast.**  K is a random walk with drift, ``d = (K_last − K_first) /
   (T − 1)`` over the common factor's T years (refitted K); each k_i is an
   AR(1), ``k_t = c₀ + c₁ k_{t−1}``,
   fitted by least squares on the member's consecutive years.  With
   ``|c₁| < 1`` k_i decays geometrically to ``c₀ / (1 − c₁)``, so the
   member's log rates stay a bounded distance from the common trend: the
   coherence of the forecast.  **Otherwise** (``|c₁| ≥ 1``, or k_i constant so
   that c₁ is undefined) k_i is held at its last fitted value, the forecast
   of a random walk without drift: the member's deviation from the common
   trend persists unchanged but never grows.  The rule used is in the
   ``fits`` output (``rule``: ``ar1`` or ``hold``) and every ``hold`` is a
   warning.
7. **Jump-off.**  The forecast starts from the fitted value
   ``a + B K + b k`` of the member's last year, after the second stage,
   with no further jump-off adjustment: like the book's Lee–Carter
   (``demography::lca(adjust = "dt")`` then ``jumpchoice = "fit"``), whose
   fitted value in the jump-off year reproduces that year's total deaths.
   Point forecasts only; no intervals.

Second stage: what Li & Lee (2005) did, and what is done here
-------------------------------------------------------------
Read in the authors' manuscript on PubMed Central (PMC1356525, NIHMS3338;
https://pmc.ncbi.nlm.nih.gov/articles/PMC1356525/, read 2026-11-01), the
version funded by the NIA; page numbers of the journal version were not
checked.

* The common factor ("Extending the LC method"): "the B(x) and K(t) should
  be obtained from applying the ordinary LC method to the whole group, with
  the K(t) adjusted to fit the group's average life expectancy".  Their
  ordinary LC second stage fits life expectancy ("Background"): "(The
  original LC method fits the observed total number of deaths in the second
  stage, but fitting life expectancy is simpler and works just as well.
  Some analysts prefer to skip the second stage and forecast the original
  k(t) directly.)"
* The populations' own factors ("Extending the LC method"): "it may not be
  possible to choose a value of k(t,i) that exactly fits a given level of
  life expectancy, or if such a level exists, it may correspond to a
  significant reduction in R_AC(i).  For these reasons, we do not adjust
  k(t,i) to fit the life expectancy of the ith population" — because b(x,i)
  takes both signs, so that raising k(t,i) has "offsetting effects".
* The jump-off, their eq. (11): ``log m(x,t,i) = log m(x,T,i) +
  B(x)[K(t) − K(T)] + b(x,i)[k(t,i) − k(T,i)]`` for t > T, obtained by
  "subtracting (6) at T from that at t".  The text does not say whether
  m(x,T,i) is the observed or the fitted rate of the last year T.

The default is the paper's own second stage (``refit = "e0"``; Sam,
2026-11-01, after the paper was checked).  The alternative ``refit = "deaths"``
follows the book's Lee–Carter rather than the letter of the paper: K is refitted to the core's
total deaths (Lee & Carter 1992's criterion, which Li & Lee call equivalent
to theirs) in Li & Lee's way, as the ordinary Lee–Carter second stage of the
group taken as one population, ``Σ_x Ē_{x,t} exp(A_x + B_x K_t) = Σ_x D̄_{x,t}``
with the core's summed exposures Ē and deaths D̄; and, departing from Li & Lee,
each k_{t,i} is also refitted, ``Σ_x E_{x,t,i} exp(a_{x,i} + B_x K_t +
b_{x,i} k_{t,i}) = Σ_x D_{x,t,i}``, so that every member's fit, and so its
jump-off, reproduces its observed deaths.  The paper's own procedure is
``refit = "e0"``; ``refit = "deaths_common"`` keeps its structure (K
refitted, the k_i not) with Lee & Carter's criterion.

The equation for K_t or k_{t,i} has the form ``Σ_x c_x exp(β_x κ) = D``
with c_x ≥ 0, whose logarithm is a convex function of κ.  When the β_x
(B, or b_i) are all of one sign it is monotone and has exactly one root.
When they take both signs — b_i nearly always does, the case Li & Lee warn
about — it has at most two roots, one on each side of its minimum; the root
taken is the one on the side of the first-stage (SVD) value, the value from
which ``demography``'s ``findroot`` also starts its search.  If no κ
reproduces D (the minimum of the implied deaths is above D), κ is the
minimiser — the implied deaths closest to D, as ``demography`` keeps "the
closest" value — and the year is counted (``n_inexact``) and reported as a
warning.

Data up to the origin
---------------------
The rates are the comparators' smoothed rates (the ``smoothed`` output of
:func:`mdmx.forecast.smooth_comparator_rates`; each year smoothed on its own,
no zero rates), with deaths and exposures from
:func:`mdmx.forecast.comparator_data` (``exposures=``) for the
exposure-weighted aggregate and the second stage.  Ages at and above the
open age are summed there.  Years whose deaths and exposures are
interpolated (``filled``) are refitted like the others, because the book's
Lee–Carter fits the same filled series (``R/demography_forecasts.R`` passes
every row to ``lca``).  ``lca`` computes its deaths as exposure times rate
after replacing zero rates by interpolated estimates
(``interpolate = TRUE``); here the observed deaths are used as they are.
A job (a row of :func:`mdmx.forecast.cv_jobs`) reads only its population's
years up to its ``last_train_year`` (≤ the origin): the smoothed series'
rows for years after it are filled from later observations and are never
read.  The forecast covers ``last_train_year + 1 … origin + horizon``, like
the R comparators.

* **Group form.**  The common factor's window is ``[origin − window + 1,
  origin]``.  Its core is the jobs' populations with rates in every year of
  the window (first year at or before its start, last observed year at the
  origin, both sexes) reduced to non-overlapping series by
  :func:`mdmx.bayeslife.split_overlap`, the rule of bayesLife's world level:
  an aggregate in the candidate set represents its family and its parts
  leave the core; without the aggregate its parts enter (they do not overlap
  one another; the register's only nested series, the civilian ones, are not
  in the tensor).  If fewer than ``min_core`` populations qualify, the
  window ends at the latest earlier year at which enough do
  (:func:`_common_window`): this happens when the origin year is unobserved
  nearly everywhere — 1918–1920 are masked in every population, so the
  common window of the origin 1920 ends in 1917 — and K is its drift
  forecast after it.  Every job's population — aggregates, parts and
  populations with part of the window included — is forecast with the
  core's B and K and its own a, b and k, fitted on its own years from the
  common window's first year to its last observed year.  Between a
  population's last observed year and the origin K is the core's fitted K
  (data up to the origin); after the origin it is the drift forecast.  The
  same core serves both sexes.  In a member's years after the common
  window (origin 1920) K is the drift forecast, not refitted (the core is
  not complete there); with ``refit = "deaths"`` the member's own k,
  refitted to its deaths, carries its level there.
* **Sex-coherent form.**  Per population, the window is the last ``window``
  years up to its ``last_train_year`` (from its ``first_year`` at the
  earliest); the core is both sexes.

Comparison with MortCast
------------------------
MortCast's ``lileecarter.estimate`` (used for bayesLife's schedules) is not
the ACF model: it fits a Lee–Carter model to each sex (``k_{t,s} =
Σ_x (log m_{x,t,s} − a_{x,s})``, b_{x,s} by regression on k) and shares
``B = (b_F + b_M) / 2``, with no sex-specific factor; its forecasts take k
from an e₀ path (``mortcast``).  Without sex-specific factors and with the
``"logmean"`` aggregate the two give the same a, B and K; with them, a is
still the same and MortCast's ``k_{t,s} = K_t + k_{t,s} + Σ_x ε_{x,t,s}``,
because both normalisations make the age patterns sum to one
(``tools/lilee_mortcast_check.py``).

Conventions
-----------
* Index 0 of a sex axis is female and 1 is male.
* Rates are central death rates m_x per person-year; ``log`` is natural.
* Arrays of a group are ``(N, X, T)``: members, ages, the window's years.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from . import bayeslife, lifetable
from . import tensor as tz

__all__ = [
    "AGGREGATES", "REFITS", "LiLeeSettings", "common_factor", "fit_acf", "fitted_acf", "forecast_acf",
    "implied_deaths", "group_forecasts", "sex_coherent_forecasts", "LILEE_CODE",
]

#: The aggregates whose Lee–Carter fit is the common factor.
AGGREGATES = ("exposure", "logmean")
#: The second stages (``[lilee] refit``) and what each refits: (K's target, whether each k_i is
#: refitted to its member's deaths).
REFITS = {"deaths": ("deaths", True), "deaths_common": ("deaths", False), "e0": ("e0", False),
          "none": ("none", False)}
_COMMON_REFITS = ("deaths", "e0", "none")

_SEXES = ("female", "male")
# A unit singular vector whose entries sum to less than this has no usable
# Lee–Carter normalisation (Σ_x b_x = 1 would divide by ~0).
_SUM_TOL = 1e-8
# The deaths refit has converged when |log implied − log observed deaths| is below this: a relative
# error of 1e-12 in the deaths, a hundred times the rounding of the logarithm of a total of 10⁷.
_REFIT_TOL = 1e-12
_REFIT_MAX_ITER = 100
# Log rates are clipped to this range inside the e₀ of the second stage only, so that the bracket
# search for K can step far without overflow (q is 1 above m = 2 anyway).
_LOG_M_CLIP = 50.0
_FAIL_COLUMNS = ["method", "pop", "origin", "message"]
_FIT_COLUMNS = ["method", "pop", "origin", "sex", "core", "common_last_year", "first_year", "last_year",
                "n_years", "drift", "c0", "c1", "rule", "b_norm", "explained_cf", "explained_acf", "refit",
                "n_inexact"]


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LiLeeSettings:
    """Every tunable value of the Li–Lee benchmark (read from ``[lilee]``).

    Attributes
    ----------
    window : fitting window in calendar years (group form: the common
        factor's, ending at the origin or, if fewer than ``min_core``
        populations are complete there, at the latest earlier year where
        enough are; sex-coherent form: ending at the population's last
        observed year)
    min_years : a member with fewer years in its window is not forecast
        (at least 3: the AR(1) of k needs two pairs of years)
    min_core : the group form needs at least this many core populations
    aggregate : ``"exposure"`` or ``"logmean"`` (:data:`AGGREGATES`)
    refit : the second stage (:data:`REFITS`): ``"deaths"`` (K to the
        core's total deaths, each k_i to its member's), ``"deaths_common"``
        (K only), ``"e0"`` (Li & Lee 2005: K to the aggregate's life
        expectancy, k_i not refitted) or ``"none"``.  A deaths refit of K
        needs the ``"exposure"`` aggregate, whose rate is the core's ΣD/ΣE:
        the ``"logmean"`` aggregate has no deaths.
    method_group, method_sex : method codes of the two forms in the
        forecasts' ``method`` column (letters and digits only)
    """

    window: int
    min_years: int
    min_core: int
    aggregate: str
    refit: str
    method_group: str
    method_sex: str

    def __post_init__(self) -> None:
        if self.min_years < 3:
            raise ValueError(f"min_years must be at least 3, got {self.min_years}.")
        if self.window < self.min_years:
            raise ValueError(f"window ({self.window}) must be at least min_years ({self.min_years}).")
        if self.min_core < 1:
            raise ValueError(f"min_core must be at least 1, got {self.min_core}.")
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {self.aggregate!r}.")
        if self.refit not in REFITS:
            raise ValueError(f"refit must be one of {tuple(REFITS)}, got {self.refit!r}.")
        if REFITS[self.refit][0] == "deaths" and self.aggregate != "exposure":
            raise ValueError(f"refit = '{self.refit}' needs aggregate = 'exposure' (the 'logmean' aggregate has "
                             "no deaths).")
        for name in ("method_group", "method_sex"):
            code = getattr(self, name)
            if not (code.isascii() and code.isalnum()):
                raise ValueError(f"{name} must contain letters and digits only, got {code!r}.")
        if self.method_group == self.method_sex:
            raise ValueError("method_group and method_sex must differ.")

    @classmethod
    def from_params(cls, section: Mapping, **overrides) -> LiLeeSettings:
        """Build the settings from the ``[lilee]`` parameter section."""
        values = dict(window=int(section["window"]), min_years=int(section["min_years"]),
                      min_core=int(section["min_core"]), aggregate=str(section["aggregate"]),
                      refit=str(section["refit"]), method_group=str(section["method_group"]),
                      method_sex=str(section["method_sex"]))
        values.update(overrides)
        return cls(**values)

    def as_dict(self) -> dict:
        """Plain dict of the settings (what the artefact store hashes)."""
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# The model
# ══════════════════════════════════════════════════════════════════════════════

def _leading_term(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """First SVD term of a row-centred (X, n) matrix, normalised as Lee–Carter.

    Returns ``(b, k, norm)`` with ``b k ≈ R``: ``Σ b = 1`` (``norm = "sum"``),
    or, when the singular vector sums to ~0, ``b`` of unit length with the
    sign of :func:`mdmx.tensor.fix_signs` (``norm = "unit"``).
    """
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    u, v = U[:, 0], s[0] * Vt[0]
    total = float(u.sum())
    if abs(total) >= _SUM_TOL:
        return u / total, v * total, "sum"
    sign = float(np.sign(tz.fix_signs(u[:, None])[:, 0] @ u))
    return sign * u, sign * v, "unit"


# ══════════════════════════════════════════════════════════════════════════════
# The second stage
# ══════════════════════════════════════════════════════════════════════════════

def _log_exposure(E: np.ndarray) -> np.ndarray:
    """log E with −inf where E = 0 (a cell with no exposure implies no deaths)."""
    with np.errstate(divide="ignore"):
        return np.log(E)


def implied_deaths(log_m: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Total deaths implied by log rates and exposures, ``Σ_x E_{x,t} exp(log m_{x,t})``.

    Parameters
    ----------
    log_m, exposures : (..., X, T) log central death rates and exposures
        (person-years)

    Returns
    -------
    (..., T) deaths per year.
    """
    return (np.asarray(exposures, dtype=float) * np.exp(np.asarray(log_m, dtype=float))).sum(axis=-2)


def _refit_to_deaths(log_c: np.ndarray, beta: np.ndarray, deaths: np.ndarray,
                     guess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lee–Carter's second stage: solve ``Σ_x exp(log_c[x, t] + β_x κ_t) = D_t`` for every year t.

    Paper: eq:lilee

    ``g_t(κ) = log Σ_x exp(log_c[x, t] + β_x κ) − log D_t`` is convex in κ
    (a log-sum-exp of affine functions), with slope the implied-deaths
    weighted mean of β.  Newton's method on g, started from the first-stage
    value, keeps to the monotone side of g on which it starts (the side of
    the slope's sign there) and converges to that side's root: after at most
    one step every iterate lies on the far side of the root from g's
    minimum, where g > 0 and the tangent of a convex function cannot
    overshoot the root.  When β is of one sign g is
    monotone and that root is the only one.  When β takes both signs and
    that side has no root — the implied deaths are above D_t at every κ — an
    iterate crosses g's minimum; κ_t is then the minimiser, the implied
    deaths closest to D_t, and the year is flagged.

    Parameters
    ----------
    log_c : (X, n) log of the implied deaths at κ = 0 of each age and year,
        ``log E_{x,t} + offset_{x,t}`` (−inf where the exposure is 0)
    beta : (X,) the age pattern that multiplies κ (B, or a member's b)
    deaths : (n,) observed total deaths D_t (> 0)
    guess : (n,) the first-stage values of κ

    Returns
    -------
    ``(kappa, exact)``: (n,) the refitted values and (n,) bool, False in the
    years where no κ reproduces D_t (κ is the minimiser there).
    """
    log_c, beta = np.asarray(log_c, dtype=float), np.asarray(beta, dtype=float)
    deaths, kappa = np.asarray(deaths, dtype=float), np.array(guess, dtype=float)
    X, n = log_c.shape
    if beta.shape != (X,) or deaths.shape != (n,) or kappa.shape != (n,):
        raise ValueError(f"Shapes do not match: log_c {log_c.shape}, beta {beta.shape}, deaths {deaths.shape}, "
                         f"guess {kappa.shape}.")
    if np.any(np.isnan(log_c)) or np.any(log_c == np.inf) or not np.all(np.isfinite(beta)):
        raise ValueError("The second stage's offsets and age pattern must be finite (log_c may be −inf).")
    if np.any(np.all(log_c == -np.inf, axis=0)):
        raise ValueError("A year with no exposure at any age cannot be refitted to its deaths.")
    if not np.all(np.isfinite(deaths)) or np.any(deaths <= 0):
        raise ValueError("Observed total deaths must be positive and finite in every year refitted.")
    log_d = np.log(deaths)

    def value_slope(k: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = log_c[:, cols] + beta[:, None] * k[None, :]
        top = z.max(axis=0)
        w = np.exp(z - top)
        total = w.sum(axis=0)
        return top + np.log(total) - log_d[cols], (w * beta[:, None]).sum(axis=0) / total

    every = np.arange(n)
    g, slope = value_slope(kappa, every)
    side = np.where(slope >= 0.0, 1.0, -1.0)          # the monotone side of g on which each year starts
    exact = np.ones(n, dtype=bool)
    active = np.abs(g) > _REFIT_TOL
    flat = active & (slope == 0.0)                    # started exactly at g's minimum
    if np.any(flat & (g < 0.0)):
        raise ValueError("A first-stage value is exactly at the minimum of the implied deaths, below the "
                         "observed deaths: the side of the second stage's root is undefined.")
    exact[flat] = False                               # minimum above D: the guess is the closest value
    active &= ~flat
    for _ in range(_REFIT_MAX_ITER):
        cols = np.flatnonzero(active)
        if cols.size == 0:
            break
        trial = kappa[cols] - g[cols] / slope[cols]
        g_new, slope_new = value_slope(trial, cols)
        crossed = side[cols] * slope_new <= 0.0
        for j, c in zip(np.flatnonzero(crossed), cols[crossed], strict=True):
            kappa[c], exact[c] = _deaths_minimum(value_slope, c, kappa[c], trial[j])
            active[c] = False
        keep = cols[~crossed]
        kappa[keep], g[keep], slope[keep] = trial[~crossed], g_new[~crossed], slope_new[~crossed]
        active[keep] = np.abs(g[keep]) > _REFIT_TOL
    if active.any():
        raise RuntimeError(f"The deaths refit did not converge in {_REFIT_MAX_ITER} Newton steps in "
                           f"{int(active.sum())} years (largest |log implied/observed| "
                           f"{float(np.abs(g[active]).max()):.3g}).")
    return kappa, exact


def _deaths_minimum(value_slope, col: int, before: float, after: float) -> tuple[float, bool]:
    """κ of one year when a Newton step crossed the minimum of the convex g (:func:`_refit_to_deaths`).

    The slope of g changes sign between ``before`` and ``after``, so its
    minimiser lies between them.  If g is still above the tolerance there no
    κ reproduces the deaths: the minimiser is returned, flagged inexact.
    (A minimum below zero cannot happen in exact arithmetic — a root on the
    starting side would have stopped the crossing — but is handled by a
    bracketed root search between the minimiser and ``before``, where g > 0
    because only a step from g > 0 can cross the minimum.)
    """
    cols = np.array([col])

    def g(k: float) -> float:
        return float(value_slope(np.array([k]), cols)[0][0])

    def slope(k: float) -> float:
        return float(value_slope(np.array([k]), cols)[1][0])

    lo, hi = min(before, after), max(before, after)
    k_min = after if slope(after) == 0.0 else brentq(slope, lo, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    g_min = g(k_min)
    if g_min > _REFIT_TOL:
        return float(k_min), False
    if g_min >= 0.0:                                   # the minimum touches D within the tolerance
        return float(k_min), True
    return float(brentq(g, *sorted((k_min, before)), xtol=1e-14, rtol=4 * np.finfo(float).eps)), True


def _e0_open(log_m: np.ndarray) -> np.ndarray:
    """Life expectancy at birth of single-year rates whose last age is an open interval.

    The book's life table (:mod:`mdmx.lifetable`: a₀ = ``lifetable.A0``,
    a_x = 0.5 above) at ages 0 … X − 2, and ``l_{X−1} / m_{X−1}`` person-years
    in the open interval.  Log rates are clipped to ±``_LOG_M_CLIP`` first,
    which leaves every e₀ of real rates unchanged.

    Parameters
    ----------
    log_m : (..., X) log central death rates, age last

    Returns
    -------
    (...) e₀ in years.
    """
    m = np.exp(np.clip(np.asarray(log_m, dtype=float), -_LOG_M_CLIP, _LOG_M_CLIP))
    X = m.shape[-1]
    q = lifetable.qx_from_mx(m[..., :-1], lifetable.conversion_ax(X - 1))
    return lifetable.e0_from_qx(q) + lifetable.survivorship(q)[..., -1] / m[..., -1]


def _root_near(f, guess: float, step: float, max_doublings: int = 60) -> tuple[float, bool]:
    """The root of a continuous f nearest a guess, searching outward on both sides.

    The bracket ``[guess, guess ± step · 2^j]`` doubles until f changes sign
    on either side (the strategy of ``demography``'s ``findroot``); Brent's
    method then finds the root in the first bracket found (the nearer one if
    both sides change sign at the same doubling).  If f never changes sign,
    the value returned minimises |f| — found among the points visited, then
    refined between that point's neighbours — and is flagged.

    Returns
    -------
    ``(x, exact)``: the root, or the closest value with ``exact = False``.
    """
    f_guess = f(guess)
    if f_guess == 0.0:
        return float(guess), True
    visited = [(guess, f_guess)]
    inner = {-1.0: (guess, f_guess), 1.0: (guess, f_guess)}
    for j in range(max_doublings):
        roots = []
        for direction in (-1.0, 1.0):
            x = guess + direction * step * 2.0 ** j
            fx = f(x)
            visited.append((x, fx))
            x_in, f_in = inner[direction]
            if fx == 0.0:
                roots.append(x)
            elif np.sign(fx) != np.sign(f_in):
                roots.append(brentq(f, min(x_in, x), max(x_in, x), xtol=1e-12, rtol=4 * np.finfo(float).eps))
            inner[direction] = (x, fx)
        if roots:
            return float(min(roots, key=lambda r: abs(r - guess))), True
    xs, fs = (np.array(v) for v in zip(*sorted(visited), strict=True))
    best = int(np.argmin(np.abs(fs)))
    lo, hi = xs[max(best - 1, 0)], xs[min(best + 1, len(xs) - 1)]
    closest = minimize_scalar(lambda x: abs(f(x)), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10 * max(1.0, abs(xs[best]))})
    if not closest.success:
        raise RuntimeError(f"The search for the value closest to a root near {guess:.6g} failed: {closest.message}")
    x = float(closest.x) if abs(f(closest.x)) <= abs(fs[best]) else float(xs[best])
    return x, False


def _refit_to_e0(A: np.ndarray, B: np.ndarray, log_agg: np.ndarray,
                 guess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Li & Lee (2005)'s second stage: K_t such that the model's e₀ is the aggregate's in every year.

    Paper: eq:lilee

    Solves ``e₀(A + B K_t) = e₀(log m_{·,t})`` (:func:`_e0_open`) for each
    year t by :func:`_root_near` from the first-stage K_t, with steps of the
    first-stage K's mean absolute yearly change.  e₀ falls as K rises when B
    is positive at every age, so the root is then unique.  When B takes both
    signs (falling child and rising adult mortality: eastern Europe around
    1990–2000) e₀(K) has a maximum, which a year's e₀ may exceed: K_t is then
    the value whose e₀ is closest, flagged.

    Parameters
    ----------
    A, B : (X,) the aggregate's mean log rate and age pattern
    log_agg : (X, T) the aggregate's log rates
    guess : (T,) the first-stage K

    Returns
    -------
    ``(K, exact)``: (T,) the refitted K and (T,) bool, False where no K
    reproduces the year's e₀.
    """
    target = _e0_open(log_agg.T)                                     # (T,)
    step = max(float(np.abs(np.diff(guess)).mean()), 1e-6)
    found = [_root_near(lambda k, t=t: float(_e0_open(A + B * k)) - target[t], float(guess[t]), step)
             for t in range(len(guess))]
    return np.array([k for k, _ in found]), np.array([ok for _, ok in found])


# ══════════════════════════════════════════════════════════════════════════════
# The model
# ══════════════════════════════════════════════════════════════════════════════

def common_factor(log_m: np.ndarray, weights: np.ndarray | None = None, *, refit: str = "none",
                  exposures: np.ndarray | None = None, deaths: np.ndarray | None = None) -> dict:
    """The common factor: a Lee–Carter fit to the group's aggregate rates, with its second stage.

    Paper: eq:lilee

    Parameters
    ----------
    log_m : (N, X, T) log central death rates of the core members over the
        window (complete: no NaN)
    weights : (N, X, T) exposures (person-years, ≥ 0) for the aggregate
        ``Σ_i m_i E_i / Σ_i E_i``; None for the mean of the log rates.  At an
        age and year where the core has no exposure at all (the open age
        group of a small population in its early years) the aggregate is the
        members' rates averaged with equal weights.
    refit : the second stage of K: ``"deaths"``, so that
        ``Σ_x Ē_{x,t} exp(A_x + B_x K_t)`` equals the core's total deaths
        ``Σ_x D̄_{x,t}`` (Ē, D̄ summed over the core; needs ``weights``, the
        exposure aggregate); ``"e0"``, so that the e₀ of ``A + B K_t`` is the
        aggregate's (Li & Lee 2005); ``"none"``
    exposures, deaths : (N, X, T) the core's exposures and observed deaths,
        for ``refit = "deaths"``

    Returns
    -------
    dict with ``A`` (X,), the aggregate's time-mean log rate; ``B`` (X,),
    ``Σ B = 1``; ``K0`` (T,), the first-stage K, mean zero, and ``drift0``,
    its drift; ``K`` (T,), K after the second stage (``K0`` without one),
    and ``drift``, its random-walk drift per year; ``exact`` (T,) bool,
    False where no K reproduces the year's deaths or e₀ (:func:`_refit_to_deaths`,
    :func:`_refit_to_e0`; K is then the closest value);
    ``log_agg`` (X, T), the aggregate's log rate; ``n_unweighted``, the
    number of age-year cells averaged with equal weights.
    """
    log_m = np.asarray(log_m, dtype=float)
    if log_m.ndim != 3 or log_m.shape[0] < 1 or log_m.shape[2] < 2:
        raise ValueError(f"log_m must be (N ≥ 1, X, T ≥ 2), got shape {log_m.shape}.")
    if not np.all(np.isfinite(log_m)):
        raise ValueError("The core's log rates must be complete and finite over the window.")
    if refit not in _COMMON_REFITS:
        raise ValueError(f"refit must be one of {_COMMON_REFITS}, got {refit!r}.")
    n_unweighted = 0
    if weights is None:
        log_agg = log_m.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != log_m.shape:
            raise ValueError(f"weights must have the shape of log_m {log_m.shape}, got {weights.shape}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Exposures of the core must be finite and not negative.")
        # The smoothed rates exist where the exposure is zero; with no exposure in the whole
        # core the weighted mean is 0/0, so those cells fall back to equal weights.
        empty = weights.sum(axis=0) <= 0
        n_unweighted = int(empty.sum())
        weights = np.where(empty[None], 1.0, weights)
        log_agg = np.log((np.exp(log_m) * weights).sum(axis=0) / weights.sum(axis=0))
    A = log_agg.mean(axis=1)
    B, K0, norm = _leading_term(log_agg - A[:, None])
    if norm != "sum":
        raise ValueError("The common age pattern sums to zero: the Lee–Carter normalisation Σ B = 1 is undefined.")
    drift0 = float((K0[-1] - K0[0]) / (len(K0) - 1))
    exact = np.ones(len(K0), dtype=bool)
    if refit == "none":
        K, drift = K0, drift0
    else:
        if refit == "deaths":
            if weights is None:
                raise ValueError("refit = 'deaths' needs the exposure aggregate (weights): the mean of the log "
                                 "rates has no deaths.")
            E, D = _second_stage_data(exposures, deaths, np.ones(log_m.shape[::2], dtype=bool))
            K, exact = _refit_to_deaths(_log_exposure(E.sum(axis=0)) + A[:, None], B, D.sum(axis=(0, 1)), K0)
        else:
            K, exact = _refit_to_e0(A, B, log_agg, K0)
        drift = float((K[-1] - K[0]) / (len(K) - 1))
    return {"A": A, "B": B, "K0": K0, "drift0": drift0, "K": K, "drift": drift, "exact": exact,
            "log_agg": log_agg, "n_unweighted": n_unweighted}


def _second_stage_data(exposures: np.ndarray | None, deaths: np.ndarray | None,
                       used: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check the exposures and deaths of a deaths refit: (N, X, T), finite and ≥ 0 where used.

    ``used`` (N, T) marks each member's years; elsewhere the arrays may hold
    anything (NaN) and are returned as zeros, which no refit reads.
    """
    if exposures is None or deaths is None:
        raise ValueError("A deaths refit needs the exposures and the observed deaths.")
    E, D = np.asarray(exposures, dtype=float), np.asarray(deaths, dtype=float)
    N, T = used.shape
    if E.shape != D.shape or E.shape[0] != N or E.shape[2] != T:
        raise ValueError(f"exposures {E.shape} and deaths {D.shape} must be (N = {N}, X, T = {T}).")
    mask = used[:, None, :]
    for name, a in (("exposures", E), ("deaths", D)):
        inside = np.where(mask, a, 0.0)
        if not np.all(np.isfinite(inside)) or np.any(inside < 0):
            raise ValueError(f"The {name} must be finite and not negative in every member's years.")
    return np.where(mask, E, 0.0), np.where(mask, D, 0.0)


def _ar1(k: np.ndarray) -> tuple[float, float, str]:
    """Least-squares AR(1) ``k_t = c₀ + c₁ k_{t−1}`` and the forecast rule it implies."""
    x, y = k[:-1], k[1:]
    dx = x - x.mean()
    sxx = float(dx @ dx)
    if sxx == 0.0:                       # k constant: c₁ undefined, k is held
        return float(y.mean()), float("nan"), "hold"
    c1 = float(dx @ (y - y.mean())) / sxx
    c0 = float(y.mean() - c1 * x.mean())
    return c0, c1, ("ar1" if abs(c1) < 1.0 else "hold")


def fit_acf(log_m: np.ndarray, core: np.ndarray, weights: np.ndarray | None = None, *,
            n_common: int | None = None, refit_common: str = "none", refit_own: bool = False,
            exposures: np.ndarray | None = None, deaths: np.ndarray | None = None) -> dict:
    """Fit the augmented common factor model to a group of members.

    Paper: *eq:lilee

    The common factor (B, K) comes from the core members' aggregate over
    the first ``n_common`` years of the window, with K's second stage
    (:func:`common_factor`); in the window's later years, if any, K is its
    random-walk-with-drift forecast.  Every member, core or not, gets its
    own ``a_{x,i}``, ``b_{x,i}``, ``k_{t,i}`` on its own years (module
    docstring, items 2–4), k_i refitted to its deaths if ``refit_own``, and
    the AR(1) of its k (item 6).

    Parameters
    ----------
    log_m : (N, X, T) log central death rates of the members over the
        window's T calendar years; NaN in the years a member does not use.
        Each member's years are consecutive, at least 3, and its rates there
        complete.  Core members have every one of the first ``n_common``
        years.
    core : (N,) bool, the members whose rates estimate the common factor
    weights : (N, X, T) exposures for an exposure-weighted aggregate (read
        for the core's first ``n_common`` years only), or None for the mean of
        the core's log rates
    n_common : years (from the window's first) of the common factor, at
        least 2; None for all T
    refit_common : K's second stage, ``"deaths"``, ``"e0"`` or ``"none"``
        (:func:`common_factor`)
    refit_own : refit each k_{t,i} so that ``Σ_x E_{x,t,i} exp(a_{x,i} +
        B_x K_t + b_{x,i} k_{t,i})`` equals the member's observed deaths
        ``Σ_x D_{x,t,i}`` in each of its years (:func:`_refit_to_deaths`)
    exposures, deaths : (N, X, T) exposures (person-years) and observed
        deaths, read in each member's years; needed by a deaths refit

    Returns
    -------
    dict with the common factor (``A``, ``B``, ``drift``, ``n_unweighted``,
    ``K0``, ``drift0``, :func:`common_factor`), ``K`` (T,), fitted (and
    refitted) over the first ``n_common`` years and its drift forecast after,
    ``n_common``, ``n_inexact_common``, the common factor's years whose
    deaths (or e₀) no K reproduces, and, per member: ``first``, ``last`` (N,)
    indices of its first and last year in the window; ``a``, ``b`` (N, X);
    ``k`` (N, T), NaN outside its years (after the second stage), and
    ``k0``, the first-stage k; ``n_inexact`` (N,), the member's years whose
    deaths no k reproduces; ``c0``, ``c1`` (N,) of the AR(1) and ``rule``
    (N,) ``"ar1"`` or ``"hold"``; ``b_norm`` (N,) ``"sum"`` or ``"unit"``;
    ``explained_cf`` and ``explained_acf`` (N,), Li & Lee's explanation
    ratios ``1 − Σ ε² / Σ (log m − mean_t log m)²`` without and with the
    member's own factor, with the factors after the second stage (as theirs).
    """
    log_m = np.asarray(log_m, dtype=float)
    core = np.asarray(core, dtype=bool)
    if log_m.ndim != 3:
        raise ValueError(f"log_m must be (N, X, T), got shape {log_m.shape}.")
    N, X, T = log_m.shape
    n_common = T if n_common is None else int(n_common)
    if not 2 <= n_common <= T:
        raise ValueError(f"n_common must be in [2, T = {T}], got {n_common}.")
    if core.shape != (N,) or not core.any():
        raise ValueError("core must be an (N,) boolean mask with at least one member.")
    used = np.isfinite(log_m).all(axis=1)                          # (N, T) years each member uses
    if np.any(np.isfinite(log_m).any(axis=1) & ~used):
        raise ValueError("A member's rates are incomplete in some year of its window.")
    if not used[core, :n_common].all():
        raise ValueError("Every core member must have rates in every year of the common factor's window.")
    first = used.argmax(axis=1)
    last = T - 1 - used[:, ::-1].argmax(axis=1)
    n_years = used.sum(axis=1)
    if np.any(n_years != last - first + 1):
        raise ValueError("Each member's years must be consecutive.")
    if np.any(n_years < 3):
        raise ValueError("Each member needs at least 3 years (the AR(1) of k needs two pairs).")
    if refit_common not in _COMMON_REFITS:
        raise ValueError(f"refit_common must be one of {_COMMON_REFITS}, got {refit_common!r}.")
    if refit_common == "deaths" or refit_own:
        exposures, deaths = _second_stage_data(exposures, deaths, used)

    in_common = (core, slice(None), slice(0, n_common))
    common = common_factor(log_m[in_common], None if weights is None else np.asarray(weights, dtype=float)[in_common],
                           refit=refit_common,
                           exposures=None if refit_common != "deaths" else exposures[in_common],
                           deaths=None if refit_common != "deaths" else deaths[in_common])
    B = common["B"]
    after = np.arange(1, T - n_common + 1)
    K = np.concatenate([common["K"], common["K"][-1] + common["drift"] * after])
    K0 = np.concatenate([common["K0"], common["K0"][-1] + common["drift0"] * after])
    a = np.empty((N, X))
    b = np.empty((N, X))
    k, k0 = np.full((N, T), np.nan), np.full((N, T), np.nan)
    c0, c1 = np.empty(N), np.empty(N)
    rule, b_norm = np.empty(N, dtype=object), np.empty(N, dtype=object)
    explained_cf, explained_acf = np.empty(N), np.empty(N)
    n_inexact = np.zeros(N, dtype=int)
    for i in range(N):
        sl = slice(first[i], last[i] + 1)
        y = log_m[i, :, sl]                                        # (X, n)
        # a from the first-stage K (Li & Lee's eq. 4); the residual with the refitted K.
        a[i] = (y - np.outer(B, K0[sl])).mean(axis=1)
        common_i = np.outer(B, K[sl])
        R = y - a[i][:, None] - common_i
        b[i], k0[i, sl], b_norm[i] = _leading_term(R)
        k[i, sl] = k0[i, sl]
        if refit_own:
            log_c = _log_exposure(exposures[i, :, sl]) + a[i][:, None] + common_i
            k[i, sl], exact = _refit_to_deaths(log_c, b[i], deaths[i, :, sl].sum(axis=0), k0[i, sl])
            n_inexact[i] = int((~exact).sum())
        c0[i], c1[i], rule[i] = _ar1(k[i, sl])
        # Li & Lee's explanation ratios: share of the variation over time explained.
        total = float(((y - y.mean(axis=1, keepdims=True)) ** 2).sum())
        rest = R - np.outer(b[i], k[i, sl])
        explained_cf[i] = 1.0 - float((R ** 2).sum()) / total if total > 0 else np.nan
        explained_acf[i] = 1.0 - float((rest ** 2).sum()) / total if total > 0 else np.nan
    return {**{name: common[name] for name in ("A", "B", "drift", "drift0", "n_unweighted")}, "K": K, "K0": K0,
            "n_common": n_common, "n_inexact_common": int((~common["exact"]).sum()), "first": first,
            "last": last, "a": a, "b": b, "k": k, "k0": k0, "n_inexact": n_inexact, "c0": c0, "c1": c1,
            "rule": rule, "b_norm": b_norm, "explained_cf": explained_cf, "explained_acf": explained_acf}


def fitted_acf(fit: Mapping) -> np.ndarray:
    """The fitted log rates ``a + B K + b k`` of every member over its own years.

    Returns
    -------
    (N, X, T) log rates, NaN outside each member's years.
    """
    return (fit["a"][:, :, None] + (fit["B"][:, None] * fit["K"][None, :])[None]
            + fit["b"][:, :, None] * fit["k"][:, None, :])


def forecast_acf(fit: Mapping, steps: int) -> dict:
    """Point forecasts of every member from its last year to ``steps`` years after the window.

    Paper: eq:lilee

    K continues its random walk with drift after the window (``fit["K"]``
    already holds the fitted K and, after the common factor's years, its
    drift forecast); a member whose years end before the window's last year
    takes K from ``fit["K"]`` in between.  Each k_i follows its AR(1) from
    its last fitted value,
    ``k_{l+h} = μ + c₁^h (k_l − μ)`` with ``μ = c₀ / (1 − c₁)``, or is held at
    k_l (``rule == "hold"``).  The forecast starts from the fitted value of
    the member's last year l.

    Parameters
    ----------
    fit : output of :func:`fit_acf`
    steps : years forecast after the window's last year (≥ 0)

    Returns
    -------
    dict with ``log_m`` (N, X, T + steps), the forecast log rates (NaN up to
    each member's last year), ``K`` (T + steps,), the fitted then forecast K,
    and ``k`` (N, T + steps), each member's fitted then forecast k (NaN
    before its first year).
    """
    if steps < 0:
        raise ValueError(f"steps must not be negative, got {steps}.")
    a, b, k_fit, B, K = fit["a"], fit["b"], fit["k"], fit["B"], fit["K"]
    N, X = a.shape
    T = len(K)
    K_path = np.concatenate([K, K[-1] + fit["drift"] * np.arange(1, steps + 1)])
    k_path = np.full((N, T + steps), np.nan)
    k_path[:, :T] = k_fit
    log_m = np.full((N, X, T + steps), np.nan)
    for i in range(N):
        last = int(fit["last"][i])
        h = np.arange(1, T + steps - last)
        k_last = k_fit[i, last]
        if fit["rule"][i] == "ar1":
            mu = fit["c0"][i] / (1.0 - fit["c1"][i])
            k_path[i, last + 1:] = mu + fit["c1"][i] ** h * (k_last - mu)
        elif fit["rule"][i] == "hold":
            k_path[i, last + 1:] = k_last
        else:
            raise ValueError(f"Unknown forecast rule {fit['rule'][i]!r} of member {i}.")
        log_m[i, :, last + 1:] = (a[i][:, None] + B[:, None] * K_path[None, last + 1:]
                                  + b[i][:, None] * k_path[i, None, last + 1:])
    return {"log_m": log_m, "K": K_path, "k": k_path}


# ══════════════════════════════════════════════════════════════════════════════
# The two forms on the comparators' data
# ══════════════════════════════════════════════════════════════════════════════

def _cube(frame: pd.DataFrame, value: str, pops: Sequence[str], years: np.ndarray,
          ages: np.ndarray) -> np.ndarray:
    """(P, 2, X, T) array of one column for populations × sexes × ages × years; NaN where absent."""
    sub = frame[frame["pop"].isin(pops) & (frame["year"] >= years[0]) & (frame["year"] <= years[-1])]
    if sub.duplicated(["pop", "sex", "year", "age"]).any():
        raise ValueError(f"Duplicate rows of {value!r} for one population, sex, year and age.")
    p_idx = pd.Index(pops).get_indexer(sub["pop"])
    s_idx = pd.Index(_SEXES).get_indexer(sub["sex"])
    x_idx = pd.Index(ages).get_indexer(sub["age"].astype(int))
    if np.any(s_idx < 0):
        raise ValueError(f"Unknown sex labels: {sorted(set(sub['sex']) - set(_SEXES))}.")
    if np.any(x_idx < 0):
        raise ValueError(f"Ages of {value!r} not among the smoothed rates' ages.")
    cube = np.full((len(pops), 2, len(ages), len(years)), np.nan)
    cube[p_idx, s_idx, x_idx, sub["year"].to_numpy(dtype=int) - int(years[0])] = sub[value].to_numpy(dtype=float)
    return cube


def _ages(smoothed: pd.DataFrame) -> np.ndarray:
    ages = np.sort(smoothed["age"].astype(int).unique())
    if len(ages) < 2 or np.any(np.diff(ages) != 1):
        raise ValueError("The smoothed rates must have consecutive single-year ages.")
    return ages


def _check_jobs(jobs: pd.DataFrame) -> None:
    need = {"pop", "origin", "horizon", "first_year", "last_train_year"}
    if not need <= set(jobs.columns):
        raise ValueError(f"jobs lack the columns {sorted(need - set(jobs.columns))}.")
    if (jobs["last_train_year"] > jobs["origin"]).any():
        raise ValueError("A job's last training year is after its origin.")
    if jobs.duplicated(["pop", "origin"]).any():
        raise ValueError("A population appears twice among the jobs of one origin.")


def _member_rates(mx: np.ndarray, start: int, end: int) -> tuple[np.ndarray | None, str]:
    """A member's (2, X, n) rates over its years, or None and the reason it cannot be fitted."""
    m = mx[..., start:end + 1]
    if not np.all(np.isfinite(m)):
        return None, "smoothed rates missing in its window"
    if np.any(m <= 0):
        return None, "smoothed rates not positive in its window"
    return m, ""


def _emit(blocks: list, method: str, pop: str, origin: int, sex: str, years: np.ndarray,
          log_m: np.ndarray) -> None:
    """Append one member-sex's forecast rows (years × ages)."""
    blocks.append((method, pop, origin, sex, years, np.exp(log_m.T)))


def _forecast_frame(blocks: list, ages: np.ndarray) -> pd.DataFrame:
    cols = [f"a{x}" for x in ages]
    if not blocks:
        return pd.DataFrame(columns=["method", "pop", "origin", "sex", "year", *cols])
    keys = pd.DataFrame({
        "method": np.concatenate([np.repeat(b[0], len(b[4])) for b in blocks]),
        "pop": np.concatenate([np.repeat(b[1], len(b[4])) for b in blocks]),
        "origin": np.concatenate([np.repeat(b[2], len(b[4])) for b in blocks]).astype(int),
        "sex": np.concatenate([np.repeat(b[3], len(b[4])) for b in blocks]),
        "year": np.concatenate([b[4] for b in blocks]).astype(int),
    })
    values = pd.DataFrame(np.concatenate([b[5] for b in blocks], axis=0), columns=cols)
    return pd.concat([keys, values], axis=1)


def _frames(blocks: list, ages: np.ndarray, failures: list, warnings: list, fits: list) -> dict:
    return {"forecasts": _forecast_frame(blocks, ages),
            "failures": pd.DataFrame(failures, columns=_FAIL_COLUMNS),
            "warnings": pd.DataFrame(warnings, columns=_FAIL_COLUMNS),
            "fits": pd.DataFrame(fits, columns=_FIT_COLUMNS)}


def _record_fit(fits: list, warnings: list, method: str, pop: str, origin: int, sex: str, core: bool,
                common_last: int, years: np.ndarray, fit: Mapping, i: int, refit: str) -> None:
    """One member-sex's row of ``fits`` and its warnings."""
    first, last = int(years[fit["first"][i]]), int(years[fit["last"][i]])
    fits.append([method, pop, origin, sex, core, int(common_last), first, last, last - first + 1, fit["drift"],
                 fit["c0"][i], fit["c1"][i], fit["rule"][i], fit["b_norm"][i], fit["explained_cf"][i],
                 fit["explained_acf"][i], refit, int(fit["n_inexact"][i])])
    if fit["rule"][i] == "hold":
        why = ("k is constant, so its AR(1) coefficient is undefined" if np.isnan(fit["c1"][i]) else
               f"k's AR(1) coefficient c1 = {fit['c1'][i]:.4g} is not inside (-1, 1)")
        warnings.append([method, pop, origin, f"{sex}: {why}; k held at its last fitted value"])
    if fit["b_norm"][i] == "unit":
        warnings.append([method, pop, origin,
                         f"{sex}: the own age pattern b sums to ~0; b normalised to unit length instead of sum 1"])
    if fit["n_inexact"][i]:
        warnings.append([method, pop, origin, f"{sex}: in {int(fit['n_inexact'][i])} years no k reproduces the "
                         "observed total deaths; k is the value whose implied deaths are closest"])


def _record_inexact_common(warnings: list, method: str, pop: str, origin: int, sex: str, fit: Mapping,
                           refit: str) -> None:
    """A warning if in some years no K reproduces the core's total deaths (or the aggregate's e₀)."""
    if fit["n_inexact_common"]:
        target = "the core's total deaths" if REFITS[refit][0] == "deaths" else "the aggregate's e0"
        warnings.append([method, pop, origin, f"{sex}: in {fit['n_inexact_common']} years no K reproduces "
                         f"{target}; K is the value that comes closest"])


def _deaths_and_exposures(settings: LiLeeSettings, exposures: pd.DataFrame | None, pops: Sequence[str],
                          years: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The (P, 2, X, T) exposure and deaths cubes a run needs (None where it needs none)."""
    refit_common, refit_own = REFITS[settings.refit]
    need_d = refit_common == "deaths" or refit_own
    need_e = settings.aggregate == "exposure" or need_d
    if need_e and exposures is None:
        raise ValueError(f"The '{settings.aggregate}' aggregate with refit = '{settings.refit}' needs the "
                         "exposures (the comparator_data frame).")
    if need_d and "deaths" not in exposures.columns:
        raise ValueError(f"refit = '{settings.refit}' needs a 'deaths' column in the exposures frame "
                         "(comparator_data).")
    E = _cube(exposures, "exposure", pops, years, ages) if need_e else None
    D = _cube(exposures, "deaths", pops, years, ages) if need_d else None
    return E, D


def _check_second_stage(pop: str, p: int, E: np.ndarray | None, D: np.ndarray | None, lo: int, hi: int,
                        first: int, last: int) -> None:
    """Exposures and deaths must exist wherever a member's smoothed rates do (both come from one frame).

    ``E``, ``D``: (P, 2, X, T) cubes or None; ``p``: the population's row;
    ``lo``, ``hi``: its first and last year's column; ``first``, ``last``:
    the same years, for the message.
    """
    for name, cube in (("exposures", E), ("deaths", D)):
        if cube is not None and not np.all(np.isfinite(cube[p, ..., lo:hi + 1])):
            raise ValueError(f"{pop}: {name} missing where its smoothed rates are complete ({first}-{last}).")


def _record_unweighted(warnings: list, method: str, pop: str, origin: int, sex: str, fit: Mapping) -> None:
    """A warning if some cells of the aggregate had no exposure (equal weights there)."""
    if fit["n_unweighted"]:
        warnings.append([method, pop, origin, f"{sex}: {fit['n_unweighted']} age-year cells of the aggregate have "
                         "no exposure in the core; their rates are averaged with equal weights"])


def group_forecasts(smoothed: pd.DataFrame, jobs: pd.DataFrame, settings: LiLeeSettings, *,
                    families: Mapping[str, str], exposures: pd.DataFrame | None = None) -> dict:
    """Li–Lee's group form, per sex: the origin's populations share one common factor.

    Paper: eq:lilee

    At each origin of ``jobs`` the common factor of each sex is estimated
    from the core (complete data over ``[origin − window + 1, origin]``, or
    over the latest earlier window with ``min_core`` such populations;
    non-overlapping series; module docstring, "Group form"), and every job's
    population is forecast with it and its own terms.

    Parameters
    ----------
    smoothed : ``pop, sex, year, age, mx`` — the ``smoothed`` output of
        :func:`mdmx.forecast.smooth_comparator_rates` (per person-year)
    jobs : rows of :func:`mdmx.forecast.cv_jobs` (``pop, origin, horizon,
        first_year, last_train_year``), any number of origins
    settings : :class:`LiLeeSettings`
    families : population → its family's top-level series
        (:func:`mdmx.svdcomp.population_families`), for the core
    exposures : ``pop, sex, year, age, exposure`` and, for ``refit =
        "deaths"``, ``deaths`` (the output of
        :func:`mdmx.forecast.comparator_data`); required by the
        ``"exposure"`` aggregate and the deaths refit

    Returns
    -------
    dict in the format of :func:`mdmx.forecast.comparator_forecasts`:
    ``forecasts`` (wide m_x: ``method, pop, origin, sex, year, a0 …``, the
    years ``last_train_year + 1 … origin + horizon``), ``failures`` and
    ``warnings`` (``method, pop, origin, message``; a warning about the
    common factor itself has ``pop = "(core)"``); and ``fits``, one row per
    population, origin and sex: ``core``, ``common_last_year`` (the common
    window's last year), ``first_year``, ``last_year`` and ``n_years`` of its
    own fit, ``drift`` of K, ``c0``, ``c1`` and ``rule`` of its k,
    ``b_norm``, ``explained_cf`` and ``explained_acf``, ``refit`` (the
    setting) and ``n_inexact`` (its years whose deaths no k reproduces).
    """
    _check_jobs(jobs)
    method, W = settings.method_group, settings.window
    refit_common, refit_own = REFITS[settings.refit]
    ages = _ages(smoothed)
    blocks, failures, warnings, fits = [], [], [], []
    for origin, jo in jobs.groupby("origin", sort=True):
        origin = int(origin)
        jo = jo.sort_values("pop").reset_index(drop=True)
        pops = list(jo["pop"].astype(str))
        first = jo["first_year"].to_numpy(dtype=int)
        last = jo["last_train_year"].to_numpy(dtype=int)
        # Every year the search for the common window can reach (it ends at most W − 1 years
        # before the origin); nothing after the origin is read.
        years = np.arange(origin - 2 * W + 2, origin + 1)
        mx = _cube(smoothed, "mx", pops, years, ages)                  # (P, 2, X, 2W − 1)
        E, D = _deaths_and_exposures(settings, exposures, pops, years, ages)
        found = _common_window(pops, first, last, mx, E, years, families, W, settings.min_core)
        if found is None:
            failures.extend([method, pop, origin, f"no year in {origin - W + 1}-{origin} ends a {W}-year window "
                             f"in which at least min_core = {settings.min_core} non-overlapping populations "
                             "have rates in every year"] for pop in pops)
            continue
        common_last, core_pops = found
        win = np.arange(common_last - W + 1, origin + 1)             # the common window, then to the origin
        off = int(win[0] - years[0])
        # Each population's own years run from the common window's first year (or its own first
        # year) to its last observed year: W years or fewer, unless the common window ends before
        # the origin, so that a core member's own years always contain the common window.
        start = np.maximum(first, win[0])
        member = np.zeros(len(pops), dtype=bool)
        for p, pop in enumerate(pops):
            n = last[p] - start[p] + 1
            if n < settings.min_years:
                failures.append([method, pop, origin, f"{max(n, 0)} years in its window {start[p]}-{last[p]} "
                                 f"(fewer than min_years = {settings.min_years})"])
                continue
            m, why = _member_rates(mx[p], start[p] - years[0], last[p] - years[0])
            if m is None:
                failures.append([method, pop, origin, why])
                continue
            _check_second_stage(pop, p, E, D, start[p] - years[0], last[p] - years[0], start[p], last[p])
            member[p] = True

        idx = np.flatnonzero(member)
        core = np.array([pops[p] in core_pops for p in idx])
        end_year = (jo["origin"] + jo["horizon"]).to_numpy(dtype=int)
        steps = int(end_year[idx].max()) - origin
        for s, sex in enumerate(_SEXES):
            log_m = np.full((len(idx), len(ages), len(win)), np.nan)
            for j, p in enumerate(idx):
                lo, hi = start[p] - win[0], last[p] - win[0]
                log_m[j, :, lo:hi + 1] = np.log(mx[p, s, :, off + lo:off + hi + 1])
            weights = E[idx, s, :, off:] if settings.aggregate == "exposure" else None
            second = {} if D is None else {"exposures": E[idx, s, :, off:], "deaths": D[idx, s, :, off:]}
            fit = fit_acf(log_m, core, weights, n_common=W, refit_common=refit_common, refit_own=refit_own,
                          **second)
            _record_unweighted(warnings, method, "(core)", origin, sex, fit)
            _record_inexact_common(warnings, method, "(core)", origin, sex, fit, settings.refit)
            path = forecast_acf(fit, steps)
            for j, p in enumerate(idx):
                _record_fit(fits, warnings, method, pops[p], origin, sex, bool(core[j]), common_last, win, fit, j,
                            settings.refit)
                fc_years = np.arange(last[p] + 1, end_year[p] + 1)
                _emit(blocks, method, pops[p], origin, sex, fc_years, path["log_m"][j][:, fc_years - win[0]])
    return _frames(blocks, ages, failures, warnings, fits)


def _common_window(pops: Sequence[str], first: np.ndarray, last: np.ndarray, mx: np.ndarray,
                   E: np.ndarray | None, years: np.ndarray, families: Mapping[str, str], W: int,
                   min_core: int) -> tuple[int, set[str]] | None:
    """The last year of the group form's common window and its core.

    The latest year ``t ≤ origin`` (``years[-1]``), at most ``W − 1`` years
    before the origin, at which at least ``min_core`` non-overlapping
    populations (:func:`mdmx.bayeslife.split_overlap`) have positive rates in
    every year of ``[t − W + 1, t]``, each up to its own last observed year.
    At the origin itself unless the origin year is unobserved nearly
    everywhere (1918–1920 are masked in every population, so the common
    window of the origin 1920 ends in 1917).

    Parameters
    ----------
    pops : (P,) codes; first, last : (P,) each job's first and last training
        year; mx, E : (P, 2, X, len(years)) rates and exposures (E None for
        the ``"logmean"`` aggregate); years : (len(years),) calendar years

    Returns
    -------
    ``(t, core)`` or None if no such year exists.
    """
    origin = int(years[-1])
    for t in range(origin, origin - W, -1):
        lo, hi = t - W + 1 - int(years[0]), t - int(years[0])
        candidates = []
        for p, pop in enumerate(pops):
            if first[p] > t - W + 1 or last[p] < t:
                continue
            m = mx[p, ..., lo:hi + 1]
            if not (np.all(np.isfinite(m)) and np.all(m > 0)):
                continue
            if E is not None and not np.all(np.isfinite(E[p, ..., lo:hi + 1])):
                raise ValueError(f"{pop}: exposures missing where its smoothed rates are complete "
                                 f"({t - W + 1}-{t}).")
            candidates.append(pop)
        core = bayeslife.split_overlap(candidates, families)[0]
        if len(core) >= min_core:
            return t, set(core)
    return None


def sex_coherent_forecasts(smoothed: pd.DataFrame, jobs: pd.DataFrame, settings: LiLeeSettings, *,
                           exposures: pd.DataFrame | None = None) -> dict:
    """Li–Lee's sex-coherent form: the two sexes of one population share one common factor.

    Paper: eq:lilee

    For each job, the window is the last ``window`` years up to its
    ``last_train_year`` (from its ``first_year`` at the earliest); the common
    factor is the Lee–Carter fit to the aggregate of both sexes, and each sex
    has its own a, b and k (module docstring).

    Parameters
    ----------
    smoothed, jobs, settings, exposures : as for :func:`group_forecasts`

    Returns
    -------
    dict in the format of :func:`mdmx.forecast.comparator_forecasts`
    (``forecasts``, ``failures``, ``warnings``) and ``fits``, as
    :func:`group_forecasts` (``core`` is True for both sexes).
    """
    _check_jobs(jobs)
    method, W = settings.method_sex, settings.window
    refit_common, refit_own = REFITS[settings.refit]
    ages = _ages(smoothed)
    blocks, failures, warnings, fits = [], [], [], []
    for origin, jo in jobs.groupby("origin", sort=True):
        origin = int(origin)
        jo = jo.sort_values("pop").reset_index(drop=True)
        pops = list(jo["pop"].astype(str))
        last = jo["last_train_year"].to_numpy(dtype=int)
        first = np.maximum(jo["first_year"].to_numpy(dtype=int), last - W + 1)
        years = np.arange(int(first.min()), origin + 1)
        mx = _cube(smoothed, "mx", pops, years, ages)                  # (P, 2, X, T)
        E, D = _deaths_and_exposures(settings, exposures, pops, years, ages)
        end_year = (jo["origin"] + jo["horizon"]).to_numpy(dtype=int)
        for p, pop in enumerate(pops):
            n = last[p] - first[p] + 1
            if n < settings.min_years:
                failures.append([method, pop, origin, f"{max(n, 0)} years in the window {first[p]}-{last[p]} "
                                 f"(fewer than min_years = {settings.min_years})"])
                continue
            lo, hi = first[p] - years[0], last[p] - years[0]
            m, why = _member_rates(mx[p], lo, hi)
            if m is None:
                failures.append([method, pop, origin, why])
                continue
            _check_second_stage(pop, p, E, D, lo, hi, first[p], last[p])
            w = E[p, :, :, lo:hi + 1] if settings.aggregate == "exposure" else None
            second = {} if D is None else {"exposures": E[p, :, :, lo:hi + 1], "deaths": D[p, :, :, lo:hi + 1]}
            fit = fit_acf(np.log(m), np.ones(2, dtype=bool), w, refit_common=refit_common, refit_own=refit_own,
                          **second)
            _record_unweighted(warnings, method, pop, origin, "both sexes", fit)
            _record_inexact_common(warnings, method, pop, origin, "both sexes", fit, settings.refit)
            path = forecast_acf(fit, int(end_year[p] - last[p]))
            win_years = np.arange(first[p], last[p] + 1)
            fc_years = np.arange(last[p] + 1, end_year[p] + 1)
            for s, sex in enumerate(_SEXES):
                _record_fit(fits, warnings, method, pop, origin, sex, True, last[p], win_years, fit, s,
                            settings.refit)
                _emit(blocks, method, pop, origin, sex, fc_years, path["log_m"][s][:, n:])
    return _frames(blocks, ages, failures, warnings, fits)


#: The functions whose source code is part of a cached Li–Lee run's key.
LILEE_CODE = (group_forecasts, sex_coherent_forecasts, fit_acf, forecast_acf, common_factor, _leading_term,
              _ar1, _cube, _ages, _member_rates, _check_jobs, _emit, _forecast_frame, _frames, _record_fit,
              _record_unweighted, _record_inexact_common, _common_window, _deaths_and_exposures,
              _check_second_stage, _second_stage_data, _log_exposure, _refit_to_deaths, _deaths_minimum,
              _refit_to_e0, _root_near, _e0_open, implied_deaths, bayeslife.split_overlap, lifetable.qx_from_mx,
              lifetable.conversion_ax, lifetable.survivorship, lifetable.e0_from_qx)
