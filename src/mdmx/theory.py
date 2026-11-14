"""What the flow field's fitted objects say about mortality change (paper §13, sec:theory).

The flow field (:mod:`mdmx.flowfield`) makes the pace of improvement a function of the level of
mortality (the speed g) and the age pattern a function of the level (the shape law f_k).  Read
together they trace one path through the space of schedules, the *typical path*, along which the
level s₁ moves at the speed g(s₁) and the other scores sit on the shape law.  This module computes
that path's life expectancies, the annual gain in e₀ it implies at each level, and the gap between
the sexes, and the same quantities observed in the data, so that the article can set them side by
side: the implied gain against e₀ is the flow field's phase portrait, its counterpart of the gain
function of bayesLife, and the gap along the level is its account of the widening and narrowing of
the female advantage.

Nothing here forecasts: the functions describe a fitted flow and the observed schedules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from . import forecast as fc
from . import lifetable

__all__ = ["typical_path", "observed_levels", "binned_medians", "at_levels"]


def _schedule(s: float, flow: Mapping, z_mean: np.ndarray, L: np.ndarray) -> np.ndarray:
    """The (S, A) logit(q_x) of the typical schedule at the level s (raw s₁ units)."""
    vec = np.array([s] + [f(s) for f in flow["fk"]])[None]                      # (1, p)
    return fc.scores_to_logit(vec, z_mean, L)[0]


def typical_path(flow: Mapping, z_mean: np.ndarray, L: np.ndarray, s_grid: Sequence[float]) -> pd.DataFrame:
    """The typical path of a fitted flow and the annual gain in e₀ it implies at each level.

    Paper: eq:theory-gain

    At the level s (raw s₁ units) the typical schedule has scores [s, f₂(s), …, f_p(s)] and
    logit(q_x) = z_mean + L · scores (as :func:`mdmx.flowfield.typical_e0`).  One year later the
    flow has moved the level to s + v(s), with v the flow's speed (``flow["g"]``; the article's law
    of motion), so the implied annual gain in life expectancy is

        Δe(s) = e(s + v(s)) − e(s),

    for each sex and for their average.  The deviations from the shape law, which relax towards it
    (α_k), are zero on the typical path, so Δe is the gain of a population on the path.

    Parameters
    ----------
    flow : :func:`mdmx.flowfield.build_flow` (``g``, ``fk`` and ``sign`` are used)
    z_mean : (S, A) logit(q_x) of the mean core; L : (S, A, ≥ p) the loadings' schedules (both from
        :func:`mdmx.forecast.origin_space`)
    s_grid : levels in raw s₁ units

    Returns
    -------
    DataFrame, one row per level, sorted by e₀: ``s1`` (oriented: higher is lower mortality),
    ``e0_female``, ``e0_male``, ``e0`` (their average), ``gap`` (female − male), ``gain_female``,
    ``gain_male`` and ``gain`` (years of e₀ per year).
    """
    rows = []
    for s in np.asarray(s_grid, dtype=float):
        now = lifetable.e0_from_logit(_schedule(s, flow, z_mean, L))
        nxt = lifetable.e0_from_logit(_schedule(s + flow["g"](s), flow, z_mean, L))
        rows.append({"s1": flow["sign"] * s, "e0_female": now[0], "e0_male": now[1], "e0": now.mean(),
                     "gap": now[0] - now[1], "gain_female": nxt[0] - now[0], "gain_male": nxt[1] - now[1],
                     "gain": nxt.mean() - now.mean()})
    return pd.DataFrame(rows).sort_values("e0", ignore_index=True)


def observed_levels(M: np.ndarray, observed: np.ndarray, years: Sequence[int], pops: Sequence[str],
                    half_window: int) -> pd.DataFrame:
    """Each observed country-year's e₀ by sex, its sex gap and its centred gain in e₀.

    Paper: sec:theory:pace

    The gain at year t is (e₀(t + w) − e₀(t − w)) / (2w) with w = ``half_window``, when both years
    are observed; otherwise it is missing.  Centring and averaging over 2w years smooths the
    year-to-year noise of single populations, as the speed of the flow field is smoothed.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); observed : (C, T) bool, the observed country-years
    years : (T,) calendar years; pops : (C,) population codes
    half_window : w, in years

    Returns
    -------
    DataFrame, one row per observed country-year: ``pop``, ``year``, ``e0_female``, ``e0_male``,
    ``e0`` (their average), ``gap`` (female − male) and ``gain`` (years of e₀ per year; NaN where
    the window is not observed).
    """
    years = np.asarray(years)
    C, T = observed.shape
    e0 = np.full((C, T, 2), np.nan)
    for c, t in zip(*np.nonzero(observed), strict=True):
        e0[c, t] = lifetable.e0_from_logit(M[:, :, c, t])
    mean = e0.mean(axis=2)
    gain = np.full((C, T), np.nan)
    w = int(half_window)
    if w < 1:
        raise ValueError(f"half_window must be at least 1, got {half_window}")
    gain[:, w:T - w] = (mean[:, 2 * w:] - mean[:, :T - 2 * w]) / (2 * w)
    c_idx, t_idx = np.nonzero(observed)
    return pd.DataFrame({"pop": np.asarray(pops)[c_idx], "year": years[t_idx],
                         "e0_female": e0[c_idx, t_idx, 0], "e0_male": e0[c_idx, t_idx, 1],
                         "e0": mean[c_idx, t_idx], "gap": e0[c_idx, t_idx, 0] - e0[c_idx, t_idx, 1],
                         "gain": gain[c_idx, t_idx]})


def binned_medians(df: pd.DataFrame, x: str, columns: Sequence[str], edges: Sequence[float]) -> pd.DataFrame:
    """Medians of ``columns`` within bands of ``x`` (left-open, right-closed: (a, b]).

    Returns
    -------
    DataFrame, one row per band with at least one value: ``low``, ``high``, ``n`` (rows in the band)
    and the median of each column (missing values ignored).
    """
    band = pd.cut(df[x], list(edges))
    out = df.groupby(band, observed=True)[list(columns)].median()
    out.insert(0, "n", df.groupby(band, observed=True)[x].count())
    out.insert(0, "high", [iv.right for iv in out.index])
    out.insert(0, "low", [iv.left for iv in out.index])
    return out.reset_index(drop=True)


def at_levels(path: pd.DataFrame, levels: Sequence[float], columns: Sequence[str]) -> pd.DataFrame:
    """The typical path's ``columns`` at given levels of e₀, by linear interpolation along e₀.

    Raises
    ------
    ValueError
        If a level lies outside the path's range of e₀ (no extrapolation).
    """
    lo, hi = float(path["e0"].min()), float(path["e0"].max())
    bad = [x for x in levels if not lo <= x <= hi]
    if bad:
        raise ValueError(f"levels {bad} lie outside the typical path's e0 range [{lo:.1f}, {hi:.1f}]")
    out = {"e0": list(levels)}
    for c in columns:
        out[c] = np.interp(levels, path["e0"], path[c])
    return pd.DataFrame(out)
