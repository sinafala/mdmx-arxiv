"""Exceptional mortality: baselines, disruption profiles and intensities (paper §8).

What this module does
---------------------
Wars and epidemics are set aside from the decomposition (chapter d4).  Paper
§8 describes an exceptional country-year as its *baseline* — the schedule
it would have had without the event — plus a disruption:

    y_{c,t} = ŷ_{c,t} + λ_{c,t} δ_d          (S, A) logit(q_x)

with ``δ_d`` the unit-length profile of the event's type ``d`` and ``λ`` its
intensity.  The module has three parts.

1. **Baselines** (:func:`baseline_estimates`).  The candidates of paper
   §8.1: temporal interpolation of the population's effective cores
   (``"linear"``, ``"local"``), the naïve and penalized projections onto the
   population's slice of the completion model (``"projection"``,
   ``"penalized"``), the neural core (``"neural"``) and the completion model
   itself (``"completion"``).  They are compared on *pseudo-events*
   (:func:`pseudo_events`): normal years next to the real events, hidden
   from every candidate, with the residual of a real event of the same type
   added (:func:`inject`) so that a candidate that reads the year's own
   schedule meets a disruption, as it would in a real exceptional year.
2. **Residuals, profiles and intensities** (:func:`residuals`,
   :func:`profile`, :func:`sg_smooth`, :func:`intensities`,
   :func:`explained_fraction`), with cells at the q_min floor left out.
3. **Heterogeneity and extensions**: profiles by event, Ward sub-clusters
   (:func:`ward_subclusters`), their comparison on held-out population
   families (:func:`holdout_heterogeneity`), two disruptions at once
   (:func:`composite_fit`) and a disruption applied to any schedule
   (:func:`apply`).

Conventions
-----------
* Index 0 of the sex axis is female, 1 male.
* Schedules and residuals are ``(n, S, A)`` logit(q_x); effective cores
  ``(r₁, r₂, C, T)`` as in the ``tucker`` artefact; masks over country-years
  ``(C, T)``; ``c_idx``, ``t_idx`` index populations and years.
* A cell is *floored* when its observed q_x was at the floor q_min (logit
  ``logit(q_min)``).  Residuals are NaN there and wherever the expected
  number of deaths is below a threshold (:func:`expected_deaths`).
* Disruption types are the labels of chapter d4: 1 war, 2 respiratory,
  3 enteric, 4 other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.metrics import silhouette_score

from . import clustering as clus
from . import lifetable
from . import nn as mnn
from . import tensor as tz

__all__ = [
    "spells", "pooling_groups", "pseudo_events", "inject", "interpolate_cores", "completion_span", "project_schedules",
    "year_features", "neural_core_cores", "baseline_estimates", "score_estimates", "with_prior",
    "low_information", "masked", "baseline_holdout", "pseudo_residuals",
    "exposures", "expected_deaths", "residuals", "floored", "profile", "sg_smooth", "intensities", "explained_fraction", "null_band",
    "ward_subclusters", "holdout_heterogeneity", "composite_fit", "composite_holdout", "apply",
]


# ══════════════════════════════════════════════════════════════════════════════
# Spells and pseudo-events
# ══════════════════════════════════════════════════════════════════════════════

def spells(mask: np.ndarray) -> pd.DataFrame:
    """Runs of consecutive True years in each row of a mask.

    Parameters
    ----------
    mask : (C, T) boolean, e.g. the exceptional country-years.

    Returns
    -------
    DataFrame with one row per run: ``c``, ``start``, ``end`` (year indices,
    inclusive) and ``length``.
    """
    rows = []
    for c, row in enumerate(np.asarray(mask, dtype=bool)):
        padded = np.r_[False, row, False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1) - 1
        rows += [{"c": c, "start": int(s), "end": int(e), "length": int(e - s + 1)}
                 for s, e in zip(starts, ends, strict=True)]
    return pd.DataFrame(rows, columns=["c", "start", "end", "length"])


def pooling_groups(blocks: pd.DataFrame, pops: Sequence[str], years: Sequence[int]) -> np.ndarray:
    """Which country-years share deaths and exposures through the pooling of chapter d5.

    A pooling block assigns one ΣD/ΣE rate to every year of the block (at
    its pooling ages), so each of those years carries the deaths of the
    others.  Years linked through a block of either sex form one group.

    Parameters
    ----------
    blocks : the ``blocks`` table of the ``lt_pooled`` artefact (columns
        ``pop`` and ``years``, the block's years separated by spaces).
    pops, years : tensor order of populations and calendar years.

    Returns
    -------
    (C, T) integer labels: 0 for a year in no block, otherwise a group number
    (unique across populations).
    """
    pos = {p: i for i, p in enumerate(pops)}
    col = {int(y): j for j, y in enumerate(years)}
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in blocks.itertuples():
        if b.pop not in pos:
            continue
        cells = [(pos[b.pop], col[int(y)]) for y in str(b.years).split() if int(y) in col]
        for cell in cells:
            parent.setdefault(cell, cell)
        for cell in cells[1:]:
            parent[find(cell)] = find(cells[0])
    out = np.zeros((len(pops), len(years)), dtype=int)
    roots = {}
    for cell in parent:
        out[cell] = roots.setdefault(find(cell), len(roots) + 1)
    return out


def pseudo_events(event_of: np.ndarray, observed: np.ndarray, *, buffer: int = 1,
                  pooled: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Normal years next to the real events, to be hidden from the baseline candidates.

    Paper: sec:exceptional:holdout

    Every event (all country-years carrying its id, across populations) is
    shifted as one block by ``span + buffer`` years to each side, where
    ``span`` is the number of calendar years the event covers; moving the
    whole block keeps its structure — an event that hits every population
    in a year (an influenza pandemic, COVID-19) becomes a year in which no
    population is available.  A shifted country-year is a *pseudo-event* if
    that year is observed (a normal year with data) and was not pooled with
    other years (``pooled``): the other years of a pooling block carry its
    deaths, which a real exceptional year's neighbours never do (exceptional
    years are never pooled), and hiding the whole block instead would turn
    the long blocks of the smallest populations (Iceland and Luxembourg
    since about 1980) into extrapolations over decades.  If the real spell
    has no observed year after it (before it), every year after (before) the
    shifted copy is hidden too — the extrapolation needed for COVID-19 at
    the end of a series.

    Parameters
    ----------
    event_of : (C, T) array of event ids (str), ``""`` for non-exceptional
        country-years.
    observed : (C, T) mask of observed, non-exceptional country-years.
    buffer : years between a real event and its shifted copy.
    pooled : (C, T) country-years in a multi-year pooling block
        (``pooling_groups(...) > 0``); they are not used as pseudo-events.

    Returns
    -------
    dict ``{"before": (hidden, rows), "after": (hidden, rows)}``.  ``hidden``
    (C, T) marks the country-years no candidate may use; ``rows`` has one row
    per pseudo-event: ``c``, ``t``, ``event_id`` and ``one_sided`` (no
    available year on one side of it, given ``hidden``).
    """
    event_of = np.asarray(event_of, dtype=object)
    observed = np.asarray(observed, dtype=bool)
    C, T = observed.shape
    pooled = np.zeros((C, T), dtype=bool) if pooled is None else np.asarray(pooled, dtype=bool)
    exceptional = event_of != ""
    # Is the real spell containing (c, t) open on its later / earlier side?
    no_after = np.zeros((C, T), dtype=bool)
    no_before = np.zeros((C, T), dtype=bool)
    for sp in spells(exceptional).itertuples():
        no_after[sp.c, sp.start:sp.end + 1] = not observed[sp.c, sp.end + 1:].any()
        no_before[sp.c, sp.start:sp.end + 1] = not observed[sp.c, :sp.start].any()
    out = {}
    for side, sign in (("before", -1), ("after", 1)):
        hidden = np.zeros((C, T), dtype=bool)
        rows = []
        for ev in sorted(set(event_of[exceptional])):
            ci, ti = np.nonzero(event_of == ev)
            shift = sign * (ti.max() - ti.min() + 1 + buffer)
            for c, t in zip(ci, ti, strict=True):
                s = t + shift
                if not (0 <= s < T and observed[c, s]) or pooled[c, s]:
                    continue
                hidden[c, s] = True
                if no_after[c, t]:
                    hidden[c, s + 1:] = True
                if no_before[c, t]:
                    hidden[c, :s] = True
                rows.append({"c": int(c), "t": int(s), "event_id": ev})
        rows = pd.DataFrame(rows, columns=["c", "t", "event_id"]).drop_duplicates(["c", "t"])
        avail = observed & ~hidden
        rows["one_sided"] = [not avail[c, t + 1:].any() or not avail[c, :t].any()
                             for c, t in zip(rows["c"], rows["t"], strict=True)]
        out[side] = (hidden, rows.reset_index(drop=True))
    return out


def inject(clean: np.ndarray, types: np.ndarray, families: np.ndarray, donor_R: np.ndarray,
           donor_types: np.ndarray, donor_families: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Add a real disruption to clean schedules: the residual of a real event of the same type.

    Paper: sec:exceptional:holdout

    Each clean schedule receives the residual of one real exceptional
    country-year drawn at random among those of the same type from *another*
    population family (so that a population's own events never test it).
    NaN cells of a donor residual (floored cells) add nothing.

    Parameters
    ----------
    clean : (n, S, A) observed logit(q_x) of normal (pseudo-event) years.
    types, families : (n,) disruption type and population family of each.
    donor_R : (m, S, A) residuals of real exceptional country-years.
    donor_types, donor_families : (m,) their type and family.
    seed : seed of the draws.

    Returns
    -------
    injected : (n, S, A); donor : (n,) index of the donor of each row.
    """
    rng = np.random.default_rng(seed)
    donor = np.empty(len(clean), dtype=int)
    for i, (d, f) in enumerate(zip(types, families, strict=True)):
        pool = np.flatnonzero((donor_types == d) & (donor_families != f))
        if pool.size == 0:
            raise ValueError(f"no donor of type {d} outside family {f}")
        donor[i] = rng.choice(pool)
    return clean + np.nan_to_num(donor_R[donor]), donor


# ══════════════════════════════════════════════════════════════════════════════
# Baseline candidates
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_cores(G: np.ndarray, available: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray,
                      years: np.ndarray, *, method: str, end_rule: str = "carry", trend_years: int = 10,
                      window: float = 10.0) -> np.ndarray:
    """A population's effective core in a year, from its other available years.

    Paper: eq:baseline_temporal, eq:baseline_penalized

    The year's own schedule is never read.  Because the sex and age bases are
    fixed, interpolating cores is the same as interpolating the schedules
    they map to.

    * ``"linear"``: linear interpolation between the nearest available years
      before and after.  With available years on one side only, ``end_rule``
      decides: ``"carry"`` repeats the nearest available core, ``"trend"``
      extends a least-squares line through the ``trend_years`` nearest
      available years on that side.
    * ``"local"``: local-linear regression on calendar year with tricube
      weights over ``±window`` years (widened to reach the three nearest
      available years when fewer lie inside), which smooths the year-to-year
      noise of the neighbours and extrapolates the local trend at a series
      end.

    Parameters
    ----------
    G : (r₁, r₂, C, T) effective cores, valid where ``available``.
    available : (C, T) mask of the country-years that may be used.
    c_idx, t_idx : (n,) population and year index of each target.
    years : (T,) calendar years.

    Returns
    -------
    (n, r₁, r₂) cores.
    """
    years = np.asarray(years, dtype=float)
    out = np.empty((len(c_idx),) + G.shape[:2])
    for i, (c, t) in enumerate(zip(c_idx, t_idx, strict=True)):
        ok = np.flatnonzero(available[c])
        if ok.size == 0:
            raise ValueError(f"population index {c} has no available year")
        Y = G[:, :, c, ok]                                   # (r1, r2, m)
        x = years[ok] - years[t]
        if method == "linear":
            before, after = np.flatnonzero(x < 0), np.flatnonzero(x > 0)
            if before.size and after.size:
                b, a = before[-1], after[0]
                w = -x[b] / (x[a] - x[b])
                out[i] = (1 - w) * Y[:, :, b] + w * Y[:, :, a]
            elif end_rule == "carry":
                out[i] = Y[:, :, before[-1] if before.size else after[0]]
            elif end_rule == "trend":
                side = before[-trend_years:] if before.size else after[:trend_years]
                out[i] = _local_linear(x[side], Y[:, :, side], np.ones(side.size))
            else:
                raise ValueError(f"unknown end rule {end_rule!r}")
        elif method == "local":
            near = np.flatnonzero(np.abs(x) <= window)
            if near.size < 3:
                near = np.sort(np.argsort(np.abs(x))[:3])
            h = max(window, np.abs(x[near]).max()) + 1.0
            w = (1 - (np.abs(x[near]) / h) ** 3) ** 3
            out[i] = _local_linear(x[near], Y[:, :, near], w)
        else:
            raise ValueError(f"unknown interpolation method {method!r}")
    return out


def _local_linear(x: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Intercept (value at x = 0) of a weighted least-squares line through ``Y`` (r₁, r₂, m) against ``x`` (m,)."""
    if x.size == 1:
        return Y[:, :, 0]
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], (Y.reshape(-1, x.size) * np.sqrt(w)).T, rcond=None)[0]
    return beta[0].reshape(Y.shape[:2])


def completion_span(core: np.ndarray, U_country: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray,
                    c: int) -> np.ndarray:
    """Schedules the completion model can produce for population ``c``: one per year direction.

    Paper: sec:exceptional:holdout

    The completion model (chapter 04) writes a population's core in year t
    as ``core ×₃ u_c ×₄ τ_t``.  With the country loading ``u_c`` fixed, the
    schedules it can produce for that population span the ``r₄`` columns
    returned here — the set ``𝓡`` onto which :func:`project_schedules` projects.

    Parameters
    ----------
    core : (r₁, r₂, r₃, r₄) core of the completion model.
    U_country : (C, r₃) its country loadings.
    U_sex, U_age : (S, r₁), (A, r₂) bases.

    Returns
    -------
    (S·A, r₄) basis, sex-major (female ages then male ages).
    """
    B = np.einsum("pqiy,i->pqy", core, U_country[c])
    return np.einsum("sp,pqy,aq->say", U_sex, B, U_age).reshape(-1, core.shape[-1])


def project_schedules(y: np.ndarray, bases: Sequence[np.ndarray], *, alpha: float = 0.0,
                      prior: np.ndarray | None = None) -> np.ndarray:
    """Naïve (α = 0) or penalized projection of schedules onto their populations' spans.

    Paper: eq:baseline_projection, eq:baseline_penalized

    For each row, ``ẑ = Zτ`` with

        τ = argmin  Σ_valid (y − Zτ)²  +  α ‖Zτ − p‖²,

    the first sum over the cells where ``y`` is finite (floored cells are
    NaN and left out) and the penalty over all cells.  As α → ∞ the estimate
    tends to the projection of the prior ``p`` onto the span — not to ``p``
    itself, unless ``p`` lies in the span.

    Parameters
    ----------
    y : (n, S, A) observed logit(q_x), NaN where not to be fitted.
    bases : n arrays (S·A, r) from :func:`completion_span`.
    alpha : penalty weight (≥ 0).
    prior : (n, S, A) prior schedules (required if ``alpha > 0``).

    Returns
    -------
    (n, S, A) projected schedules.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if alpha > 0 and prior is None:
        raise ValueError("a penalized projection needs a prior")
    n, S, A = y.shape
    out = np.empty_like(y, dtype=float)
    for i in range(n):
        Z = bases[i]
        yi = y[i].ravel()
        v = np.isfinite(yi)
        lhs = Z[v].T @ Z[v]
        rhs = Z[v].T @ yi[v]
        if alpha > 0:
            lhs = lhs + alpha * (Z.T @ Z)
            rhs = rhs + alpha * (Z.T @ prior[i].ravel())
        out[i] = (Z @ np.linalg.solve(lhs, rhs)).reshape(S, A)
    return out


def year_features(years: np.ndarray, t_min: float, t_max: float, freqs: Sequence[float] = (1, 2, 5)
                  ) -> np.ndarray:
    """Polynomial and Fourier features of calendar year, the inputs of the neural core (:func:`neural_core_cores`).

    With ``t̃ = (t − t_min)/(t_max − t_min)``: ``[t̃, t̃², t̃³, sin 2πf t̃, cos 2πf t̃
    for f in freqs]`` — nine features for the default frequencies 1, 2, 5.

    Returns
    -------
    (n, 3 + 2·len(freqs)).
    """
    tt = (np.asarray(years, dtype=float) - t_min) / (t_max - t_min)
    cols = [tt, tt ** 2, tt ** 3]
    for f in freqs:
        cols += [np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)]
    return np.column_stack(cols)


def neural_core_cores(G: np.ndarray, available: np.ndarray, U_country: np.ndarray, years: np.ndarray,
                      c_idx: np.ndarray, t_idx: np.ndarray, *, hidden: Sequence[int],
                      lr: float, epochs: int, batch_size: int, l2: float, val_frac: float, patience: int,
                      seed: int, freqs: Sequence[float]) -> np.ndarray:
    """The neural core (a baseline candidate compared in the companion): cores predicted from country loading and year features.

    An MLP (:func:`mdmx.nn.fit_mlp`) maps ``[u_c, φ(t)]`` to the ``r₁·r₂``
    core entries (the effective cores: the exact projection ``Sᵀ M_{c,t} A`` of
    an observed schedule), trained on the available country-years only;
    inputs and targets are standardised on them.  Early stopping holds out
    whole calendar years — the network's task is a year hidden in every
    population, not a population it has never seen.  The schedule of an
    exceptional year is never read.

    Parameters
    ----------
    G : (r₁, r₂, C, T) effective cores, valid where ``available``.
    available : (C, T) training country-years.
    U_country : (C, r₃) country loadings of the completion model fitted to
        the same available country-years.
    years : (T,) calendar years.
    c_idx, t_idx : (n,) targets.

    Returns
    -------
    (n, r₁, r₂) predicted cores.
    """
    years = np.asarray(years, dtype=float)
    t_min, t_max = years.min(), years.max()
    ci, ti = np.nonzero(available)
    X = np.column_stack([U_country[ci], year_features(years[ti], t_min, t_max, freqs)])
    Y = G[:, :, ci, ti].reshape(-1, ci.size).T
    sx, sy = mnn.Standardizer.fit(X), mnn.Standardizer.fit(Y)
    net, _ = mnn.fit_mlp(sx.transform(X), sy.transform(Y), hidden, lr=lr, epochs=epochs, batch_size=batch_size,
                         l2=l2, seed=seed, val_frac=val_frac, patience=patience, groups=ti)
    Xt = np.column_stack([U_country[c_idx], year_features(years[t_idx], t_min, t_max, freqs)])
    return sy.inverse_transform(net.predict(sx.transform(Xt))).reshape(len(c_idx), *G.shape[:2])


def baseline_estimates(y: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray, specs: Mapping[str, Mapping], *,
                       G: np.ndarray, available: np.ndarray, years: np.ndarray, U_sex: np.ndarray,
                       U_age: np.ndarray, completion: Mapping, neural_cores: np.ndarray | None = None
                       ) -> dict[str, np.ndarray]:
    """Baseline schedules of target country-years under each candidate — the one place that dispatches by kind.

    Paper: sec:exceptional:holdout

    ``specs`` maps a candidate's name to its settings; ``spec["kind"]`` is

    * ``"linear"`` / ``"local"`` — :func:`interpolate_cores` (with
      ``end_rule``, ``trend_years`` or ``window``);
    * ``"projection"`` — :func:`project_schedules` of ``y`` with α = 0;
    * ``"penalized"`` — the same with ``spec["alpha"]`` and the estimate of
      candidate ``spec["prior"]`` (which must come earlier in ``specs``) as
      the prior;
    * ``"neural"`` — the cores in ``neural_cores`` (from
      :func:`neural_core_cores`, fitted on the same ``available`` mask);
    * ``"completion"`` — the completion model's own value at (c, t).

    Parameters
    ----------
    y : (n, S, A) the targets' observed schedules, NaN at floored cells
        (read only by the projections).
    c_idx, t_idx : (n,) targets.
    G : (r₁, r₂, C, T) effective cores, valid where ``available``.
    available : (C, T) country-years the candidates may use; the targets
        must not be among them.
    completion : dict with ``core`` (r₁, r₂, r₃, r₄), ``U_country`` (C, r₃)
        and ``U_year`` (T, r₄) of a completion model fitted on ``available``.

    Returns
    -------
    dict name → (n, S, A) baseline schedules.
    """
    if available[c_idx, t_idx].any():
        raise ValueError("a target country-year is among the available ones")
    out: dict[str, np.ndarray] = {}
    bases = None
    for name, spec in specs.items():
        kind = spec["kind"]
        if kind in ("linear", "local"):
            cores = interpolate_cores(G, available, c_idx, t_idx, years, method=kind,
                                      end_rule=spec.get("end_rule", "carry"),
                                      trend_years=spec.get("trend_years", 10), window=spec.get("window", 10.0))
            out[name] = tz.schedules_from_cores(np.moveaxis(cores, 0, -1), U_sex, U_age).transpose(2, 0, 1)
        elif kind in ("projection", "penalized"):
            if bases is None:
                bases = [completion_span(completion["core"], completion["U_country"], U_sex, U_age, c)
                         for c in c_idx]
            if kind == "projection":
                out[name] = project_schedules(y, bases)
            else:
                if spec["prior"] not in out:
                    raise ValueError(f"prior {spec['prior']!r} of {name!r} must be estimated first")
                out[name] = project_schedules(y, bases, alpha=spec["alpha"], prior=out[spec["prior"]])
        elif kind == "neural":
            if neural_cores is None:
                raise ValueError("the neural candidate needs neural_cores")
            out[name] = tz.schedules_from_cores(np.moveaxis(neural_cores, 0, -1), U_sex, U_age).transpose(2, 0, 1)
        elif kind == "completion":
            cores = np.einsum("pqiy,ni,ny->npq", completion["core"], completion["U_country"][c_idx],
                              completion["U_year"][t_idx])
            out[name] = tz.schedules_from_cores(np.moveaxis(cores, 0, -1), U_sex, U_age).transpose(2, 0, 1)
        else:
            raise ValueError(f"unknown baseline kind {kind!r}")
    return out


def score_estimates(clean: np.ndarray, estimates: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Errors of baseline estimates against the clean schedules they should recover.

    Paper: sec:exceptional:holdout

    Parameters
    ----------
    clean : (n, S, A) the true (clean) schedules.
    estimates : name → (n, S, A).

    Returns
    -------
    Long DataFrame: ``row`` (0 … n−1), ``candidate``, ``mse`` (mean squared
    error of logit(q_x) over both sexes and all ages) and ``e0_error``
    (sex-average e₀ of the estimate minus that of the clean schedule, years).
    """
    e0_clean = lifetable.e0_from_logit(clean).mean(axis=1)
    frames = []
    for name, est in estimates.items():
        frames.append(pd.DataFrame({
            "row": np.arange(len(clean)), "candidate": name,
            "mse": np.mean((est - clean) ** 2, axis=(1, 2)),
            "e0_error": lifetable.e0_from_logit(est).mean(axis=1) - e0_clean}))
    return pd.concat(frames, ignore_index=True)


def with_prior(specs: Mapping[str, Mapping], name: str) -> dict[str, Mapping]:
    """The specs needed to compute candidate ``name``: itself, preceded by its prior if it is penalized."""
    spec = specs[name]
    return {spec["prior"]: specs[spec["prior"]], name: spec} if spec["kind"] == "penalized" else {name: spec}


def low_information(exposure: np.ndarray, c_idx: np.ndarray, t_idx: np.ndarray, *, G: np.ndarray,
                    available: np.ndarray, years: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray,
                    min_expected_deaths: float) -> tuple[np.ndarray, np.ndarray]:
    """Cells with too few expected deaths to carry a signal, judged without the year's own deaths.

    Paper: sec:exceptional:residual_def

    The expected deaths (:func:`expected_deaths`) use the linear
    interpolation of the population's available years (end rule
    ``"carry"``), which never reads the target year — so the rule cannot
    select cells by their outcome.

    Parameters
    ----------
    exposure : (n, S, A) HMD exposures of the targets (:func:`exposures`).
    c_idx, t_idx : (n,) targets; G (r₁, r₂, C, T) effective cores, valid
        where ``available`` (C, T); years (T,); U_sex, U_age the bases.

    Returns
    -------
    mask : (n, S, A) cells to leave out; baseline : (n, S, A) the
    interpolation used.
    """
    cores = interpolate_cores(G, available, c_idx, t_idx, years, method="linear", end_rule="carry")
    base = np.einsum("sp,npq,aq->nsa", U_sex, cores, U_age)
    return expected_deaths(exposure, base) < min_expected_deaths, base


def masked(y: np.ndarray, mask: np.ndarray, q_min: float) -> np.ndarray:
    """Schedules (n, S, A) with floored and ``mask`` cells set to NaN — what the projections fit."""
    return np.where(floored(y, q_min) | mask, np.nan, y)


def baseline_holdout(M: np.ndarray, G: np.ndarray, observed: np.ndarray, event_of: np.ndarray,
                     type_of_event: Mapping[str, int], families: np.ndarray, de: pd.DataFrame,
                     specs: Mapping[str, Mapping], *, pops: Sequence[str], years: np.ndarray, U_sex: np.ndarray,
                     U_age: np.ndarray, ranks: tuple[int, int], em: Mapping, neural: Mapping, buffer: int,
                     injection_seed: int, min_expected_deaths: float, q_min: float,
                     pooled: np.ndarray | None = None) -> dict:
    """Score baseline candidates on pseudo-events, with and without an injected real disruption.

    Paper: sec:exceptional:holdout

    For each side (events shifted earlier, later; :func:`pseudo_events`):

    1. the pseudo-events (none in a pooling block, ``pooled``) are hidden;
    2. the completion model (:func:`mdmx.tensor.complete_cores`, ranks and EM
       settings ``ranks``, ``em``) and the neural core are refitted on the
       remaining observed country-years;
    3. each pseudo-event receives the residual of a real exceptional
       country-year of its type from another family (:func:`inject`), the
       donors' residuals taken against linear interpolation and without
       their low-information cells;
    4. every candidate in ``specs`` estimates the baseline from the injected
       and from the clean schedule (low-information cells left out of what
       the projections fit), and is scored against the clean schedule
       (:func:`score_estimates`).

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x); read for the real exceptional years (the
        donors) and the pseudo-events only.
    G : (r₁, r₂, C, T) effective cores, read only where available.
    observed : (C, T); event_of : (C, T) event ids, "" if none;
    type_of_event : event id → type; families : (C,) family of each population;
    de : HMD deaths and exposures (``hmd_raw["de"]``).
    specs : candidates, as for :func:`baseline_estimates`.
    neural : settings of :func:`neural_core_cores`.

    Returns
    -------
    dict with ``scores`` (long; ``row`` within its side, ``candidate``,
    ``mse``, ``e0_error``, ``side``, ``injection``), ``pseudo`` (one row per
    pseudo-event: ``side``, ``c``, ``t``, ``event_id``, ``one_sided``,
    ``type``, ``family``, ``donor``) and, per side, the context needed to
    recompute any candidate: ``hidden`` and the refitted ``core``,
    ``U_country``, ``U_year`` and ``neural`` cores of its pseudo-events.
    """
    A = M.shape[1]
    common = {"years": years, "U_sex": U_sex, "U_age": U_age}
    ci, ti = np.nonzero(np.asarray(event_of, dtype=object) != "")
    y_exc = np.moveaxis(M[:, :, ci, ti], -1, 0)
    low, lin = low_information(exposures(de, pops, years, A, ci, ti), ci, ti, G=G, available=observed,
                               min_expected_deaths=min_expected_deaths, **common)
    donor_R = residuals(y_exc, lin, q_min, low)
    donor_types = np.array([type_of_event[e] for e in np.asarray(event_of)[ci, ti]])
    out: dict = {"scores": [], "pseudo": []}
    for k, (side, (hidden, rows)) in enumerate(pseudo_events(event_of, observed, buffer=buffer,
                                                             pooled=pooled).items()):
        avail = observed & ~hidden
        _, (U_c, U_t), core, _ = tz.complete_cores(G, avail, ranks[0], ranks[1], max_iter=em["max_iter"],
                                                   tol=em["tol"])
        comp = {"core": core, "U_country": U_c, "U_year": U_t}
        c_, t_ = rows["c"].to_numpy(), rows["t"].to_numpy()
        typ = rows["event_id"].map(type_of_event).to_numpy()
        clean = np.moveaxis(M[:, :, c_, t_], -1, 0)
        injected, donor = inject(clean, typ, families[c_], donor_R, donor_types, families[ci],
                                 seed=injection_seed + k)
        nc = neural_core_cores(G, avail, U_c, years, c_, t_, **neural)
        low_p, _ = low_information(exposures(de, pops, years, A, c_, t_), c_, t_, G=G, available=avail,
                                   min_expected_deaths=min_expected_deaths, **common)
        for what, y in (("injected", injected), ("clean", clean)):
            est = baseline_estimates(masked(y, low_p, q_min), c_, t_, specs, G=G, available=avail,
                                     completion=comp, neural_cores=nc, **common)
            out["scores"].append(score_estimates(clean, est).assign(side=side, injection=what))
        out["pseudo"].append(rows.assign(side=side, type=typ, family=families[c_], donor=donor))
        out[side] = {"hidden": hidden, **comp, "neural": nc}
    out["scores"] = pd.concat(out["scores"], ignore_index=True)
    out["pseudo"] = pd.concat(out["pseudo"], ignore_index=True)
    return out


def pseudo_residuals(holdout: Mapping, M: np.ndarray, G: np.ndarray, observed: np.ndarray, de: pd.DataFrame,
                     specs: Mapping[str, Mapping], name: str, *, pops: Sequence[str], years: np.ndarray,
                     U_sex: np.ndarray, U_age: np.ndarray, min_expected_deaths: float, q_min: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Residuals of the clean pseudo-events under candidate ``name`` — what "no disruption" looks like.

    Paper: sec:exceptional:estimation

    Recomputes the candidate on each side's pseudo-events from the context
    stored by :func:`baseline_holdout` (no refit), with the same rules for
    low-information cells as for the real exceptional years.

    Returns
    -------
    residuals : (n, S, A) in the order of ``holdout["pseudo"]``; types : (n,).
    """
    A = M.shape[1]
    common = {"years": years, "U_sex": U_sex, "U_age": U_age}
    need = with_prior(specs, name)
    R, types = [], []
    for side in ("before", "after"):
        ctx = holdout[side]
        rows = holdout["pseudo"][holdout["pseudo"]["side"] == side]
        c_, t_ = rows["c"].to_numpy(), rows["t"].to_numpy()
        avail = observed & ~ctx["hidden"]
        low_p, _ = low_information(exposures(de, pops, years, A, c_, t_), c_, t_, G=G, available=avail,
                                   min_expected_deaths=min_expected_deaths, **common)
        clean = np.moveaxis(M[:, :, c_, t_], -1, 0)
        est = baseline_estimates(masked(clean, low_p, q_min), c_, t_, need, G=G, available=avail,
                                 completion={k: ctx[k] for k in ("core", "U_country", "U_year")},
                                 neural_cores=ctx["neural"], **common)[name]
        R.append(residuals(clean, est, q_min, low_p))
        types.append(rows["type"].to_numpy())
    return np.concatenate(R), np.concatenate(types)


# ══════════════════════════════════════════════════════════════════════════════
# Residuals, profiles and intensities
# ══════════════════════════════════════════════════════════════════════════════

def floored(y: np.ndarray, q_min: float) -> np.ndarray:
    """Cells of logit(q_x) schedules at the floor ``logit(q_min)`` of :func:`mdmx.lifetable.logit`.

    Paper: sec:exceptional:residual_def
    """
    return y <= lifetable.logit(np.array(q_min), q_min) + 1e-9


def exposures(de: pd.DataFrame, pops: Sequence[str], years: Sequence[int], n_ages: int, c_idx: np.ndarray,
              t_idx: np.ndarray) -> np.ndarray:
    """HMD person-years of exposure of the given country-years, as ``(n, S, A)``.

    Parameters
    ----------
    de : HMD deaths and exposures (artefact ``hmd_raw["de"]``), columns
        ``pop``, ``sex`` ("female"/"male"), ``year``, ``age``, ``exposure``.
    pops, years : tensor order of populations and years.
    n_ages : ages 0 … n_ages−1 are kept (the tensor's ages).
    c_idx, t_idx : (n,) country-years.

    Returns
    -------
    (n, S, A) exposures; raises if any is missing.
    """
    want = pd.DataFrame({"pop": np.asarray(pops)[c_idx], "year": np.asarray(years)[t_idx], "row": np.arange(len(c_idx))})
    sub = de[de["age"] < n_ages].merge(want, on=["pop", "year"])
    out = np.full((len(c_idx), 2, n_ages), np.nan)
    sex = sub["sex"].map({"female": 0, "male": 1}).to_numpy(dtype=int)
    out[sub["row"].to_numpy(dtype=int), sex, sub["age"].to_numpy(dtype=int)] = sub["exposure"].to_numpy()
    if np.isnan(out).any():
        raise ValueError(f"{int(np.isnan(out).any(axis=(1, 2)).sum())} country-years lack HMD exposures")
    return out


def expected_deaths(exposure: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Deaths expected in each cell if the baseline held: ``E · m`` with ``m = −log(1 − q)``.

    Paper: sec:exceptional:residual_def

    The death rate is that of a constant force of mortality over the year of
    age with probability ``q = expit(baseline)``.  The number is used only to
    screen out cells with too little information, so this approximation is
    ample.

    Parameters
    ----------
    exposure : (n, S, A) person-years; baseline : (n, S, A) logit(q_x).
    """
    return exposure * -np.log1p(-1.0 / (1.0 + np.exp(-baseline)))


def residuals(y: np.ndarray, baseline: np.ndarray, q_min: float, mask: np.ndarray | None = None
              ) -> np.ndarray:
    """Residuals ``r = y − ŷ`` on the logit scale, NaN where they carry no signal.

    Paper: eq:residual, sec:exceptional:residual_def, eq:disruption_model

    A floored cell records q_x = 0 replaced by q_min: its logit (about −18)
    says nothing about the size of a disruption, so it is left out.  Leaving
    out only the zeros, however, keeps the small cells that happened to have
    deaths and biases their residuals upward; ``mask`` therefore also leaves
    out cells chosen *before* looking at the year's deaths — those with few
    expected deaths (:func:`expected_deaths`).

    Parameters
    ----------
    y, baseline : (n, S, A) observed and baseline logit(q_x).
    mask : (n, S, A) cells to leave out besides the floored ones.

    Returns
    -------
    (n, S, A) residuals.
    """
    r = np.asarray(y, dtype=float) - baseline
    r[floored(y, q_min)] = np.nan
    if mask is not None:
        r[mask] = np.nan
    return r


def profile(R: np.ndarray, *, positive: bool = True) -> tuple[np.ndarray, int]:
    """Unit-length mean residual.

    Paper: eq:mean_residual, eq:profile_norm, sec:exceptional:estimation

    The mean is taken cell by cell over the rows where the cell is not NaN;
    a cell with no residual in any row (e.g. a very old age in small
    nineteenth-century populations, all masked) is set to 0 and counted.
    With ``positive`` the sign is chosen so that the profile's entries sum
    to a positive number — a positive intensity then means excess mortality.

    Parameters
    ----------
    R : (n, S, A) residuals, NaN at left-out cells.

    Returns
    -------
    (S, A) profile with ‖δ‖ = 1, and the number of cells set to 0.
    """
    empty = np.all(np.isnan(R), axis=0)
    mean = np.zeros(R.shape[1:])
    mean[~empty] = np.nanmean(R[:, ~empty], axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("the mean residual is zero")
    delta = mean / norm
    return (-delta if positive and delta.sum() < 0 else delta), int(empty.sum())


def sg_smooth(delta: np.ndarray, *, window: int, poly: int) -> np.ndarray:
    """Savitzky–Golay smoothing of a profile, each sex separately.

    Paper: eq:profile_smooth, sec:exceptional:estimation

    The filter (window ``window`` ages, polynomial of degree ``poly``) is
    applied along age; the first and last ``window // 2`` ages of each sex
    keep their raw values (no ringing at the boundaries), and the result is
    rescaled to unit length.

    Parameters
    ----------
    delta : (S, A) profile.

    Returns
    -------
    (S, A) smoothed profile with ‖δ‖ = 1.
    """
    h = window // 2
    out = np.array(delta, dtype=float)
    for s in range(out.shape[0]):
        sm = savgol_filter(delta[s], window, poly)
        out[s, h:-h] = sm[h:-h]
    return out / np.linalg.norm(out)


def intensities(R: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Least-squares intensity of each residual along a profile.

    Paper: *eq:disruption_model, eq:intensity, sec:exceptional:estimation

    ``λ = Σ r δ / Σ δ²`` over the cells where ``r`` is not NaN — ``rᵀδ``
    when no cell is missing, since ``‖δ‖ = 1``.

    Parameters
    ----------
    R : (n, S, A) residuals; delta : (S, A) or (n, S, A) profile(s).

    Returns
    -------
    (n,) intensities.
    """
    v = np.isfinite(R)
    d = np.broadcast_to(delta, R.shape)
    return np.where(v, R * d, 0).sum(axis=(1, 2)) / np.where(v, d * d, 0).sum(axis=(1, 2))


def explained_fraction(R: np.ndarray, fitted: np.ndarray) -> tuple[np.ndarray, float]:
    """Share of the residuals' sum of squares that a fit accounts for (uncentred).

    Paper: sec:exceptional:estimation, eq:explained_share

    ``1 − ‖r − fit‖² / ‖r‖²`` over the non-NaN cells, per row and pooled
    over rows (the pooled value weights each row by its ``‖r‖²``).  It is
    uncentred, as the disruption model has no intercept, and can be negative
    for a held-out row that the fit makes worse.

    Parameters
    ----------
    R : (n, S, A) residuals; fitted : (n, S, A) the fit (e.g. ``λ δ``).

    Returns
    -------
    per_row : (n,); pooled : float.
    """
    v = np.isfinite(R)
    sse = np.where(v, (R - fitted) ** 2, 0).sum(axis=(1, 2))
    sst = np.where(v, R ** 2, 0).sum(axis=(1, 2))
    return 1 - sse / sst, float(1 - sse.sum() / sst.sum())


def null_band(lam: np.ndarray, quantiles: Sequence[float]) -> tuple[float, float]:
    """Central range of intensities where there is no disruption.

    Paper: sec:exceptional:estimation

    Parameters
    ----------
    lam : (m,) intensities of pseudo-events (true λ = 0); quantiles : (lower, upper).
    """
    lo, hi = np.quantile(lam, quantiles)
    return float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════════════
# Within-type heterogeneity: profiles by event and Ward sub-clusters
# ══════════════════════════════════════════════════════════════════════════════

def ward_subclusters(R: np.ndarray, *, k_range: Sequence[int], min_size: int, max_pca: int, n_use: int
                     ) -> tuple[np.ndarray, int, dict[int, float]]:
    """Sub-clusters of one type's residuals by Ward's method, K by silhouette (scored against type and event profiles by :func:`holdout_heterogeneity`).

    Paper: sec:exceptional:heterogeneity

    The residual vectors (cells left out, NaN, set to 0) are reduced by PCA to
    ``min(max_pca, n − 1)`` components, of which the first ``n_use`` are
    clustered; each K in ``k_range`` is cut from Ward's tree, and the K with
    the highest silhouette among those whose sub-clusters all have at least
    ``min_size`` members is chosen.  If no K qualifies the type is one
    cluster (K = 1).

    Parameters
    ----------
    R : (n, S, A) residuals of one type.

    Returns
    -------
    labels : (n,) sub-cluster of each row; K; silhouette of each admissible K.
    """
    X = np.nan_to_num(R).reshape(len(R), -1)
    X = X - X.mean(axis=0)
    n_pca = min(max_pca, len(R) - 1)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    scores = (U * s)[:, :min(n_use, n_pca)]
    best, best_k, sil = np.zeros(len(R), dtype=int), 1, {}
    for k in k_range:
        if k >= len(R):
            continue
        labels = clus.ward_clustering(scores, k)[1]
        if np.bincount(labels).min() < min_size:
            continue
        sil[k] = float(silhouette_score(scores, labels))
        if sil[k] == max(sil.values()):
            best, best_k = labels, k
    return best, best_k, sil


def _fit_profiles(R: np.ndarray, types: np.ndarray, groups: np.ndarray, settings: Mapping) -> dict:
    """Profiles of every kind from one set of residual rows (types other than those listed are skipped)."""
    smooth = {"window": settings["sg_window"], "poly": settings["sg_poly"]}
    out = {"type": {}, "group": {}, "sub": {}}
    for d in settings["profile_types"]:
        m = types == d
        out["type"][d] = sg_smooth(profile(R[m])[0], **smooth)
        for g in np.unique(groups[m]):
            mg = m & (groups == g)
            if mg.sum() >= settings["min_event_n"]:
                out["group"][g] = sg_smooth(profile(R[mg])[0], **smooth)
        lab, k, _ = ward_subclusters(R[m], k_range=settings["k_range"], min_size=settings["min_size"],
                                     max_pca=settings["max_pca"], n_use=settings["n_use"])
        out["sub"][d] = np.stack([sg_smooth(profile(R[m][lab == j])[0], **smooth) for j in range(k)])
    return out


def _fits(R: np.ndarray, types: np.ndarray, groups: np.ndarray, prof: Mapping) -> dict[str, np.ndarray]:
    """``λ δ`` of each row under the three alternatives (one profile per type, by event, best sub-profile)."""
    fits = {k: np.zeros_like(R) for k in ("type", "event", "subcluster")}
    for i in range(len(R)):
        d, r = types[i], R[i:i + 1]
        dt = prof["type"][d]
        fits["type"][i] = intensities(r, dt)[0] * dt
        de = prof["group"].get(groups[i], dt)
        fits["event"][i] = intensities(r, de)[0] * de
        subs = prof["sub"][d]
        lam = np.array([intensities(r, s)[0] for s in subs])
        sse = [np.nansum((r[0] - lam[j] * subs[j]) ** 2) for j in range(len(subs))]
        j = int(np.argmin(sse))
        fits["subcluster"][i] = lam[j] * subs[j]
    return fits


def holdout_heterogeneity(R: np.ndarray, types: np.ndarray, groups: np.ndarray, families: np.ndarray,
                          settings: Mapping) -> tuple[pd.DataFrame, dict]:
    """One profile per type, per event, or per Ward sub-cluster — scored on held-out population families.

    Paper: sec:exceptional:heterogeneity

    For each family, every profile is estimated from the residuals of the
    other families (types in ``settings["profile_types"]`` only) and the
    family's own residuals are fitted: (a) by their type's profile; (b) by
    their event group's profile where the group has at least
    ``settings["min_event_n"]`` training rows, else by the type's; (c) by
    whichever of the type's sub-cluster profiles fits best — a choice made
    with the held-out residual itself, which favours (c) somewhat.  The same
    is done in sample (all rows) for comparison.

    Parameters
    ----------
    R : (n, S, A) residuals; types, groups, families : (n,) of each row.
    settings : ``profile_types``, ``min_event_n``, ``sg_window``,
        ``sg_poly``, ``k_range``, ``min_size``, ``max_pca``, ``n_use``.

    Returns
    -------
    scores : long DataFrame — ``row``, ``type``, ``alternative``, ``sample``
    (``"held-out"`` or ``"in-sample"``), ``sse`` and ``sst`` over the row's
    valid cells; profiles : the in-sample profiles (``type``, ``group``,
    ``sub``).
    """
    keep = np.isin(types, settings["profile_types"])
    idx = np.flatnonzero(keep)
    R, types, groups, families = R[idx], types[idx], groups[idx], families[idx]
    v = np.isfinite(R)
    sst = np.where(v, R ** 2, 0).sum(axis=(1, 2))
    frames = []

    def _collect(rows, fits, sample):
        for alt, f in fits.items():
            sse = np.where(v[rows], (R[rows] - f) ** 2, 0).sum(axis=(1, 2))
            frames.append(pd.DataFrame({"row": idx[rows], "type": types[rows], "alternative": alt,
                                        "sample": sample, "sse": sse, "sst": sst[rows]}))

    for fam in np.unique(families):
        test = families == fam
        train = ~test
        if not np.all(np.isin(np.unique(types[test]), types[train])):
            raise ValueError(f"family {fam} holds every row of some type")
        prof = _fit_profiles(R[train], types[train], groups[train], settings)
        _collect(np.flatnonzero(test), _fits(R[test], types[test], groups[test], prof), "held-out")
    prof_all = _fit_profiles(R, types, groups, settings)
    _collect(np.arange(len(R)), _fits(R, types, groups, prof_all), "in-sample")
    return pd.concat(frames, ignore_index=True), prof_all


def composite_fit(R: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Non-negative intensities of several profiles at once.

    Paper: eq:composite

    ``r ≈ Σ_j λ_j δ_j`` with every ``λ_j ≥ 0``, by non-negative least squares
    over each row's non-NaN cells.

    Parameters
    ----------
    R : (n, S, A) residuals; deltas : (m, S, A) profiles.

    Returns
    -------
    (n, m) intensities.
    """
    out = np.empty((len(R), len(deltas)))
    D = deltas.reshape(len(deltas), -1).T                    # (S·A, m)
    for i, r in enumerate(R.reshape(len(R), -1)):
        v = np.isfinite(r)
        out[i] = nnls(D[v], r[v])[0]
    return out


def composite_holdout(R: np.ndarray, types: np.ndarray, families: np.ndarray, rows: np.ndarray, *,
                      pair: tuple[int, int], sg_window: int, sg_poly: int) -> dict[str, np.ndarray]:
    """The composite model (:func:`composite_fit`) against one profile, with the profiles held out.

    Paper: sec:exceptional:complete

    For each row in ``rows`` the profiles of the two types in ``pair`` are
    estimated from the other population families' residuals (smoothed as in
    :func:`sg_smooth`); the row is then fitted by the first profile alone
    (least squares) and by both at once (:func:`composite_fit`).  The
    intensities are fitted on the row itself — only the profiles are held
    out.

    Parameters
    ----------
    R : (n, S, A) residuals; types, families : (n,); rows : indices to fit.

    Returns
    -------
    dict with ``lam_single`` (m,), ``lam_pair`` (m, 2), ``fit_single`` and
    ``fit_pair`` (m, S, A).
    """
    m = len(rows)
    out = {"lam_single": np.empty(m), "lam_pair": np.empty((m, 2)),
           "fit_single": np.empty((m,) + R.shape[1:]), "fit_pair": np.empty((m,) + R.shape[1:])}
    for k, i in enumerate(rows):
        other = families != families[i]
        d1, d2 = (sg_smooth(profile(R[other & (types == d)])[0], window=sg_window, poly=sg_poly) for d in pair)
        r = R[i:i + 1]
        out["lam_single"][k] = intensities(r, d1)[0]
        out["fit_single"][k] = out["lam_single"][k] * d1
        out["lam_pair"][k] = composite_fit(r, np.stack([d1, d2]))[0]
        out["fit_pair"][k] = out["lam_pair"][k, 0] * d1 + out["lam_pair"][k, 1] * d2
    return out


def apply(z: np.ndarray, delta: np.ndarray, lam: float | np.ndarray) -> np.ndarray:
    """A disruption of profile ``delta`` and intensity ``lam`` (≥ 0) added to schedule(s) ``z``.

    Paper: eq:full_model

    Parameters
    ----------
    z : (S, A) or (n, S, A) baseline logit(q_x); delta : (S, A); lam : scalar
        or (n,) intensities (≥ 0 for a disruption).

    Returns
    -------
    Disrupted logit(q_x), shape of ``z`` broadcast with ``lam``.
    """
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("a disruption's intensity must be non-negative")
    return z + lam[..., None, None] * delta
