"""Life-table fitting: regime, e₀ and disruption from a single schedule (paper §9).

What this module does
---------------------
Given one observed schedule ``y`` (S, A) logit(q_x) and nothing else, the
fitter estimates the parameters of the model

    y = z_k(e₀*) + λ δ_h + ε,      λ ≥ 0,

where ``z_k(e₀*)`` is the calibrated schedule of regime ``k`` at life
expectancy ``e₀*`` (chapter 07, :func:`mdmx.trajectories.generate`) and
``δ_h`` is a disruption profile (chapter 08) — or no disruption.

1. :func:`prepare` evaluates every regime's schedule on a fine grid of e₀
   once.
2. :func:`fit_many` finds, for every regime and every hypothesis, the e₀*
   (and λ̂ ≥ 0) with the smallest residual sum of squares, exactly on the
   grid and refined between grid points — no linearization.
3. :func:`laplace_log_bf` and :func:`evidence` turn each fit into a Laplace
   approximation of its log evidence, with a half-normal prior on λ; the
   best pair over all regimes and hypotheses is chosen
   (:func:`summarise`), and the *score* of a schedule is the log Bayes
   factor of its best disrupted fit against its best undisrupted one.
4. :func:`cross_fitted_thresholds` sets the score above which a schedule is
   called disrupted from normal schedules of *other* population families,
   so that a stated share of normal schedules is flagged; :func:`roc`
   summarises the trade-off.

:func:`fold_model` and :func:`fold_profiles` rebuild the generator and the
profiles from training families only, so the evaluation is out of sample;
:func:`identifiability` measures how far each profile can be told apart
from a change of e₀.

Conventions
-----------
* Index 0 of the sex axis is female, 1 male; e₀ is the average of the two.
* Schedules are ``(S, A)`` or ``(n, S, A)`` logit(q_x), flattened sex-major
  to ``S·A`` cells where noted.
* A cell left out of a fit (floored, or masked) does not enter the residual
  sum of squares; ``p`` is the number of cells that do.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from . import clustering as clus
from . import disruption as ds
from . import lifetable
from . import trajectories as tr

__all__ = [
    "spec_from_model", "fold_model", "fold_profiles", "prepare", "fit_many", "laplace_log_bf", "evidence",
    "summarise", "baseline_at", "expected_deaths_mask", "fit_two_pass", "fit_schedule", "evaluate_folds",
    "operating_point", "cross_fitted_thresholds", "roc", "identifiability",
    "schedule_e0",
]


# ══════════════════════════════════════════════════════════════════════════════
# The generator and the profiles, rebuilt from training families
# ══════════════════════════════════════════════════════════════════════════════

def spec_from_model(model: Mapping) -> dict:
    """The candidate specification of chapter 07's chosen generator (its ``candidate`` name and ``kind``).

    Chapter 07 names its candidates ``"<kind> [<span>] [+ sex level]"``;
    this reads the name back so that a fold's generator is rebuilt with the
    same settings.  Raises if the name does not match the stored kind.
    """
    name = str(model["candidate"])
    parts = name.replace("+ sex level", "").split()
    if not parts or parts[0] != model["kind"]:
        raise ValueError(f"cannot read the candidate {name!r} of a {model['kind']!r} model")
    spec = {"kind": parts[0], "sex_level": name.endswith("+ sex level")}
    if spec["kind"] == "lowess":
        spec["frac"] = float(parts[1])
    return spec


def fold_model(G_obs: np.ndarray, e0: np.ndarray, labels: np.ndarray, groups: np.ndarray, train: np.ndarray,
               clusters: Sequence[int], grid: np.ndarray, spec: Mapping, *, typical: Mapping,
               feature_index: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray,
               support_quantiles: Sequence[float]) -> dict:
    """Chapter 07's generator estimated from the ``train`` rows only.

    Paper: sec:fitting:evaluation

    The typical schedule (an e₀-conditional LOWESS of the cores, settings
    ``typical``) and the regimes' deviations (``spec``, as chosen in chapter
    07) are estimated exactly as there, on the training country-years.

    Parameters
    ----------
    G_obs : (n, r₁, r₂) effective cores of the observed country-years.
    e0 : (n,) their sex-average e₀; labels : (n,) chapter 06 regimes;
    groups : (n,) population families; train : (n,) bool.
    clusters : the regimes the model serves; grid : (g,) e₀ grid.
    feature_index : (p, 2) chapter 06's feature positions.

    Returns
    -------
    A model dict (:func:`mdmx.trajectories.build_model`).
    """
    train = np.asarray(train, dtype=bool)
    fit_train, typ_grid = clus.conditional_mean(e0[train], G_obs[train], grid=grid, **typical)
    positions = tr.deviation_positions(feature_index, G_obs.shape[1], sex_level=spec.get("sex_level", False))
    R_train = clus.cores_to_features(G_obs[train], positions) - clus.cores_to_features(fit_train, positions)
    dev, support = tr.estimate_deviation(R_train, e0[train], labels[train], groups[train], clusters, grid, spec,
                                         iterations=typical["iterations"], delta_share=typical["delta_share"],
                                         support_quantiles=support_quantiles)
    return tr.build_model(grid, typ_grid, dev, support, clusters, positions, U_sex, U_age, spec["kind"])


def fold_profiles(R: np.ndarray, types: np.ndarray, groups: np.ndarray, train: np.ndarray, *,
                  profile_types: Sequence[int], min_event_n: int, sg_window: int, sg_poly: int
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """Disruption profiles and their prior scales from the ``train`` rows only (chapter 08's method).

    Paper: eq:fitting:log-bf

    Two kinds of hypothesis: one profile per type (``"type"``), and one per
    event group with at least ``min_event_n`` training country-years
    (``"event"``).  Each profile is the smoothed unit mean residual
    (:func:`mdmx.disruption.profile`, :func:`mdmx.disruption.sg_smooth`).
    Its prior scale σ_λ is the root mean square of the positive intensities
    of its own training country-years on it.  An event whose training
    country-years have a mean intensity ≤ 0 on its own profile — on average
    a deficit, not an excess — has no disruption to detect and is skipped;
    a type in that state raises.

    Parameters
    ----------
    R : (n, S, A) residuals of exceptional country-years (NaN = left out).
    types, groups : (n,) type and event group; train : (n,) bool.

    Returns
    -------
    table : one row per hypothesis — ``name``, ``kind``, ``type``, ``sigma``,
    ``n_train``; deltas : (H, S, A) profiles in the same order.  The events
    skipped are listed in ``table.attrs["skipped"]``.
    """
    train = np.asarray(train, dtype=bool)
    rows, deltas, skipped = [], [], []

    def _add(name, kind, d, members):
        delta = ds.sg_smooth(ds.profile(R[members])[0], window=sg_window, poly=sg_poly)
        lam = ds.intensities(R[members], delta)
        if lam.mean() <= 0 or not (lam > 0).any():
            if kind == "type":
                raise ValueError(f"type {d}: its training country-years show no excess on their profile")
            skipped.append(name)
            return
        rows.append({"name": name, "kind": kind, "type": int(d),
                     "sigma": float(np.sqrt(np.mean(lam[lam > 0] ** 2))), "n_train": int(members.sum())})
        deltas.append(delta)

    for d in profile_types:
        _add(f"type {d}", "type", d, train & (types == d))
    for d in profile_types:
        for g in np.unique(groups[train & (types == d)]):
            members = train & (types == d) & (groups == g)
            if members.sum() >= min_event_n:
                _add(str(g), "event", d, members)
    table = pd.DataFrame(rows)
    table.attrs["skipped"] = skipped
    return table, np.stack(deltas)


# ══════════════════════════════════════════════════════════════════════════════
# The fit
# ══════════════════════════════════════════════════════════════════════════════

def prepare(model: Mapping, deltas: np.ndarray, sigmas: np.ndarray, *, grid_step: float, bracket: float) -> dict:
    """Everything a fit needs, computed once: each regime's schedules on a fine e₀ grid, and the profiles.

    Paper: sec:fitting:algorithm

    Parameters
    ----------
    model : a chapter 07 model; deltas : (H, S, A) profiles; sigmas : (H,)
        their prior scales.
    grid_step : spacing (years) of the e₀ grid, which spans the model's
        e₀ grid.
    bracket : largest shift of the level coordinate tried when calibrating
        each grid schedule to its e₀ (:func:`mdmx.trajectories.calibrate_level`).

    Returns
    -------
    dict with ``x`` (g,), ``Z`` (K, g, S·A) calibrated schedules,
    ``clusters`` (K,), ``D`` (H, S·A), ``sigma`` (H,), ``shape`` (S, A).
    """
    lo, hi = float(model["e0_grid"][0]), float(model["e0_grid"][-1])
    x = np.arange(np.ceil(lo / grid_step) * grid_step, hi + 1e-9, grid_step)
    clusters = np.asarray(model["clusters"], dtype=int)
    Z = np.stack([tr.generate(model, int(k), x, bracket=bracket).reshape(len(x), -1) for k in clusters])
    return {"x": x, "Z": Z, "clusters": clusters, "D": np.asarray(deltas, dtype=float).reshape(len(deltas), -1),
            "sigma": np.asarray(sigmas, dtype=float), "shape": tuple(np.shape(deltas)[1:])}


def _refine(rss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid index of each column's minimum and the parabolic offset (−½ … ½ grid steps) around it.

    Paper: sec:fitting:algorithm

    rss : (g, m).  Returns (m,) integer indices and (m,) offsets; at the ends
    of the grid the offset is 0.
    """
    g = rss.shape[0]
    i = np.argmin(rss, axis=0)
    inner = (i > 0) & (i < g - 1)
    j = np.clip(i, 1, g - 2)
    cols = np.arange(rss.shape[1])
    a, b, c = rss[j - 1, cols], rss[j, cols], rss[j + 1, cols]
    denom = a - 2 * b + c
    off = np.where(inner & (denom > 0), 0.5 * (a - c) / np.where(denom > 0, denom, 1.0), 0.0)
    return i, np.clip(off, -0.5, 0.5)


def fit_many(Y: np.ndarray, prep: Mapping, valid: np.ndarray | None = None) -> dict:
    """Fit every schedule under every regime and hypothesis, exactly (no linearization).

    Paper: *sec:fitting:algorithm, eq:fitting:exact-resid, eq:fitting:lam-proj

    For a schedule, regime ``k`` and hypothesis ``h`` (0: no disruption,
    1…H: the profiles), the residual sum of squares over the valid cells

        RSS_{k,h}(x) = Σ_v (y − z_k(x) − λ δ_h)²,  λ = max(0, Σ_v r δ_h / Σ_v δ_h²),

    is evaluated at every grid e₀ ``x``; e₀* is the minimum, refined by a
    parabola through its neighbours, and the RSS and λ̂ are recomputed there
    with the schedule interpolated linearly between the two grid points.

    Parameters
    ----------
    Y : (n, S, A) schedules; prep : from :func:`prepare`.
    valid : (n, S, A) bool, cells to use (default: the finite ones — floored
        cells are *not* removed here; :func:`fit_schedule` and
        :func:`fit_two_pass` do that, and the decision thresholds of chapter
        09 assume it).

    Returns
    -------
    dict of arrays: ``e0`` (n, K, H+1), ``lam`` (n, K, H+1) (0 for h = 0),
    ``rss`` (n, K, H+1), ``p`` (n,) cells used, ``dd`` (n, H) Σ_v δ²,
    ``seconds`` (n,) time per schedule.
    """
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    valid = np.isfinite(Y) if valid is None else np.asarray(valid, dtype=bool).reshape(len(Y), -1) & np.isfinite(Y)
    x, Z, D = prep["x"], prep["Z"], prep["D"]
    step = x[1] - x[0]
    K, H = Z.shape[0], D.shape[0]
    out = {key: np.zeros((len(Y), K, H + 1)) for key in ("e0", "lam", "rss")}
    out["p"] = valid.sum(axis=1)
    out["dd"] = np.zeros((len(Y), H))
    out["seconds"] = np.zeros(len(Y))
    for n in range(len(Y)):
        t0 = time.perf_counter()
        v = valid[n]
        if v.sum() < 2:
            raise ValueError(f"schedule {n} has fewer than two usable cells")
        y, Dv = Y[n, v], D[:, v]
        dd = (Dv ** 2).sum(axis=1)                                     # (H,)
        out["dd"][n] = dd
        for ki in range(K):
            Rg = y - Z[ki][:, v]                                       # (g, p)
            C = Rg @ Dv.T                                              # (g, H)
            lam = np.maximum(C / dd, 0.0)
            rss0 = (Rg ** 2).sum(axis=1)                               # (g,)
            rss = np.column_stack([rss0, rss0[:, None] - 2 * lam * C + lam ** 2 * dd])   # (g, H+1)
            i, off = _refine(rss)
            # The schedule at the refined e₀, interpolated between neighbouring grid points.
            lo = np.clip(np.where(off < 0, i - 1, i), 0, len(x) - 2)
            w = np.where(off < 0, 1 + off, off)
            w = np.where(i == len(x) - 1, 1.0, w)
            zr = (1 - w)[:, None] * Z[ki][lo][:, v] + w[:, None] * Z[ki][lo + 1][:, v]   # (H+1, p)
            r = y - zr
            c = np.einsum("hp,hp->h", r[1:], Dv)
            lam_r = np.maximum(c / dd, 0.0)
            rss_r = np.empty(H + 1)
            rss_r[0] = (r[0] ** 2).sum()
            rss_r[1:] = (r[1:] ** 2).sum(axis=1) - 2 * lam_r * c + lam_r ** 2 * dd
            out["e0"][n, ki] = x[lo] + w * step
            out["rss"][n, ki] = rss_r
            out["lam"][n, ki, 1:] = lam_r
        out["seconds"][n] = time.perf_counter() - t0
    return out


def laplace_log_bf(rss0: np.ndarray, rss: np.ndarray, lam: np.ndarray, dd: np.ndarray, sigma: np.ndarray,
                   p: np.ndarray) -> np.ndarray:
    """Laplace log Bayes factor of a disruption (λ ≥ 0, half-normal prior) against none.

    Paper: eq:fitting:log-bf

    With Gaussian errors of variance σ̂² = RSS_d / p and λ ~ half-normal(σ_λ):

        log BF = (p/2) log(RSS₀ / RSS_d) + log(2 / (σ_λ √(2π))) − λ̂² / (2σ_λ²)
                 + ½ log 2π − ½ log I,       I = Σ δ² / σ̂²,

    and log 2 less when λ̂ = 0 (the maximum sits on the boundary, where only
    half of the Laplace Gaussian lies inside the prior's support).  The
    second term is the prior's normalising constant: without it a wider
    prior would never be penalised.  ``RSS₀`` must be the
    null model's own minimum (at its own e₀*), not the RSS at the
    alternative's e₀*.  All arguments broadcast; e₀* is profiled out in both
    models alike.
    """
    rss0, rss, lam, dd, sigma, p = (np.asarray(a, dtype=float) for a in (rss0, rss, lam, dd, sigma, p))
    info = dd / (rss / p)
    log_bf = (p / 2 * np.log(rss0 / rss) + np.log(2.0 / (sigma * np.sqrt(2 * np.pi))) - lam ** 2 / (2 * sigma ** 2)
              + 0.5 * np.log(2 * np.pi) - 0.5 * np.log(info))
    return np.where(lam > 0, log_bf, log_bf - np.log(2.0))


def evidence(fit: Mapping, sigma: np.ndarray) -> np.ndarray:
    """Log evidence of every (regime, hypothesis) pair, up to a constant shared by all: (n, K, H+1).

    Paper: eq:fitting:score

    ``log E(k, 0) = −(p/2) log RSS_{k,0}``; ``log E(k, h) = log E(k, 0) +
    log BF(k, h)`` with the Bayes factor taken against the null of the
    *same* regime at its own e₀* (:func:`laplace_log_bf`), which reduces to
    ``−(p/2) log RSS_{k,h}`` plus the prior and Occam terms.
    """
    p = fit["p"][:, None, None]
    logE = -p / 2 * np.log(fit["rss"])
    rss0 = fit["rss"][:, :, :1]
    logE[:, :, 1:] = logE[:, :, :1] + laplace_log_bf(rss0, fit["rss"][:, :, 1:], fit["lam"][:, :, 1:],
                                                       fit["dd"][:, None, :], np.asarray(sigma)[None, None, :], p)
    return logE


def summarise(fit: Mapping, sigma: np.ndarray, hyp: np.ndarray | None = None) -> pd.DataFrame:
    """The best fit of each schedule over all (regime, hypothesis) pairs, and its score.

    Paper: eq:fitting:score

    Parameters
    ----------
    fit : from :func:`fit_many`; sigma : (H,) prior scales.
    hyp : indices (0-based, into the profiles) of the hypotheses allowed
        (default: all).

    Returns
    -------
    DataFrame, one row per schedule: ``k0``, ``e0_null`` — the best fit
    without disruption; ``k``, ``h`` (an index into all the profiles, also
    when ``hyp`` restricts them), ``e0``, ``lam`` — the best disrupted fit,
    whose intensity can be 0 (then it fits no better than the null);
    ``score`` = its log evidence minus the best null's (the log Bayes factor
    of the best disrupted against the best undisrupted fit).
    """
    logE = evidence(fit, sigma)
    H = logE.shape[2] - 1
    hyp = np.arange(H) if hyp is None else np.asarray(hyp, dtype=int)
    n = logE.shape[0]
    null = logE[:, :, 0]
    k0 = null.argmax(axis=1)
    alt = logE[:, :, 1 + hyp].reshape(n, -1)
    best = alt.argmax(axis=1)
    k, j = np.divmod(best, len(hyp))
    rows = np.arange(n)
    return pd.DataFrame({
        "k0": k0, "e0_null": fit["e0"][rows, k0, 0],
        "k": k, "h": hyp[j], "e0": fit["e0"][rows, k, 1 + hyp[j]], "lam": fit["lam"][rows, k, 1 + hyp[j]],
        "score": alt[rows, best] - null[rows, k0]})


def baseline_at(prep: Mapping, k_index: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """Grid schedules of regimes ``k_index`` (indices into ``prep["clusters"]``) at e₀ ``e0``, interpolated: (n, S, A).

    Paper: sec:fitting:algorithm
    """
    x = prep["x"]
    pos = np.clip((np.asarray(e0) - x[0]) / (x[1] - x[0]), 0, len(x) - 1)
    lo = np.minimum(pos.astype(int), len(x) - 2)
    w = (pos - lo)[:, None]
    Z = prep["Z"]
    z = (1 - w) * Z[k_index, lo] + w * Z[k_index, lo + 1]
    return z.reshape(len(z), *prep["shape"])


def expected_deaths_mask(baseline: np.ndarray, exposure: np.ndarray, threshold: float) -> np.ndarray:
    """Cells whose deaths expected under a fitted baseline fall below ``threshold`` (n, S, A) bool.

    Paper: sec:fitting:algorithm

    The baseline is the schedule's own fit without disruption, never its
    observed deaths, so the cells left out are not chosen by their outcome
    (as in chapter 08; :func:`mdmx.disruption.expected_deaths`).
    """
    return ds.expected_deaths(exposure, baseline) < threshold


def fit_two_pass(Y: np.ndarray, exposure: np.ndarray | None, prep: Mapping, *, q_min: float,
                 min_expected_deaths: float) -> tuple[dict, np.ndarray, dict]:
    """Fit schedules leaving out floored cells, then (with exposures) the cells with few expected deaths.

    Paper: sec:fitting:algorithm

    The first fit uses every cell that is not floored.  If ``exposure`` is
    given, the deaths expected under each schedule's best undisrupted fit
    are computed (:func:`expected_deaths_mask`) and the schedule is fitted
    again without the cells where they fall below ``min_expected_deaths``.

    Parameters
    ----------
    Y : (n, S, A) logit(q_x); exposure : (n, S, A) person-years or None.

    Returns
    -------
    fit : the final fit (:func:`fit_many`), whose ``seconds`` count both
    passes; valid : (n, S, A) cells it used; first : the first pass (the
    fit without exposures; the same as ``fit`` if ``exposure`` is None).
    """
    valid = ~ds.floored(Y, q_min)
    first = fit_many(Y, prep, valid)
    if exposure is None:
        return first, valid, first
    head = summarise(first, prep["sigma"])
    base = baseline_at(prep, head["k0"].to_numpy(), head["e0_null"].to_numpy())
    valid = valid & ~expected_deaths_mask(base, exposure, min_expected_deaths)
    fit = fit_many(Y, prep, valid)
    fit["seconds"] = fit["seconds"] + first["seconds"]
    return fit, valid, first


def fit_schedule(y: np.ndarray, prep: Mapping, threshold: float, *, q_min: float, exposure: np.ndarray | None = None,
                 min_expected_deaths: float = 5.0, hyp: np.ndarray | None = None) -> dict:
    """Fit one schedule and decide whether it carries a disruption — the procedure chapter 09 evaluates.

    Paper: sec:fitting:algorithm

    Floored cells are left out; with ``exposure`` the schedule is fitted a
    second time without the cells with few expected deaths
    (:func:`fit_two_pass`).  The schedule is called disrupted if its score
    exceeds ``threshold``, which must have been calibrated for the same
    procedure (with or without exposures; the ``fitting`` artefact holds
    both).

    Parameters
    ----------
    y : (S, A) logit(q_x); prep : from :func:`prepare`; threshold : on the score.
    exposure : (S, A) person-years, or None.
    hyp : indices of the hypotheses allowed (default: all in ``prep``).

    Returns
    -------
    dict with ``disrupted`` (bool), ``score``, ``regime`` and ``e0`` of the
    chosen fit, ``h`` and ``lam`` (−1 and 0 if not disrupted), ``e0_null``
    and ``regime_null`` (the best fit without disruption), ``cells`` used.
    """
    E = None if exposure is None else np.asarray(exposure)[None]
    fit, valid, _ = fit_two_pass(np.asarray(y, dtype=float)[None], E, prep, q_min=q_min,
                                 min_expected_deaths=min_expected_deaths)
    s = summarise(fit, prep["sigma"], hyp).iloc[0]
    disrupted = bool(s["score"] > threshold)
    regime_null = int(prep["clusters"][int(s["k0"])])
    return {"disrupted": disrupted, "score": float(s["score"]),
            "regime": int(prep["clusters"][int(s["k"])]) if disrupted else regime_null,
            "e0": float(s["e0"]) if disrupted else float(s["e0_null"]),
            "h": int(s["h"]) if disrupted else -1, "lam": float(s["lam"]) if disrupted else 0.0,
            "e0_null": float(s["e0_null"]), "regime_null": regime_null, "cells": int(valid.sum())}


def evaluate_folds(Y_obs: np.ndarray, E_obs: np.ndarray, folds_obs: np.ndarray, Y_exc: np.ndarray,
                   E_exc: np.ndarray, folds_exc: np.ndarray, R: np.ndarray, types: np.ndarray, groups: np.ndarray,
                   *, G_obs: np.ndarray, e0: np.ndarray, labels: np.ndarray, families_obs: np.ndarray,
                   clusters: Sequence[int], grid: np.ndarray, spec: Mapping, typical: Mapping,
                   feature_index: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray,
                   support_quantiles: Sequence[float], profiles: Mapping, sets: Mapping[str, Sequence[str]],
                   grid_step: float, bracket: float, q_min: float, min_expected_deaths: float) -> dict:
    """Fit every normal and exceptional schedule out of fold.

    Paper: sec:fitting:evaluation

    For each fold: the generator (:func:`fold_model`) is estimated from the
    other folds' observed country-years, and the profiles and prior scales
    (:func:`fold_profiles`, settings ``profiles``) from the other folds'
    exceptional ones; then every schedule of the fold — normal and
    exceptional — is fitted (:func:`fit_two_pass`) and summarised for each
    hypothesis set (``sets``: name → the kinds of hypothesis it allows).

    Parameters
    ----------
    Y_obs, E_obs : (n, S, A) observed normal schedules and their exposures
        (None: floored cells are the only ones left out); folds_obs : (n,).
    Y_exc, E_exc : (m, S, A) exceptional schedules and exposures; folds_exc : (m,).
    R, types, groups : (m, …) chapter 08's residuals, types and event groups
        of the exceptional schedules.
    G_obs, e0, labels, families_obs : the observed country-years' cores, e₀,
        regimes and families (for :func:`fold_model`).

    Returns
    -------
    dict with ``fits`` (one row per schedule, set and procedure: ``row``,
    ``kind``, ``fold``, ``set``, ``passes`` — ``"two"`` with exposures,
    ``"one"`` without —, the columns of :func:`summarise`, ``hname``,
    ``htype``, ``regime``, ``regime_null``, ``seconds``, ``p``) and
    ``folds`` (fold → its ``model``, ``hypotheses``, ``deltas``, ``skipped``
    events and ``prepare_seconds``).
    """
    frames, per_fold = [], {}
    for f in np.unique(folds_obs):
        model = fold_model(G_obs, e0, labels, families_obs, folds_obs != f, clusters, grid, spec,
                           typical=typical, feature_index=feature_index, U_sex=U_sex, U_age=U_age,
                           support_quantiles=support_quantiles)
        hyp, deltas = fold_profiles(R, types, groups, folds_exc != f, **profiles)
        t0 = time.perf_counter()
        prep = prepare(model, deltas, hyp["sigma"].to_numpy(), grid_step=grid_step, bracket=bracket)
        prepare_seconds = time.perf_counter() - t0
        for kind, Y, E, test in (("normal", Y_obs, E_obs, folds_obs == f), ("exceptional", Y_exc, E_exc, folds_exc == f)):
            idx = np.flatnonzero(test)
            if idx.size == 0:
                continue
            fit, _, first = fit_two_pass(Y[idx], None if E is None else E[idx], prep, q_min=q_min,
                                         min_expected_deaths=min_expected_deaths)
            passes = {"two": fit, "one": first} if E is not None else {"one": fit}
            for npass, fit_ in passes.items():
                for name, kinds in sets.items():
                    allowed = np.flatnonzero(hyp["kind"].isin(kinds).to_numpy())
                    summ = summarise(fit_, prep["sigma"], allowed)
                    frames.append(summ.assign(row=idx, kind=kind, fold=int(f), set=name, passes=npass,
                                              hname=hyp["name"].to_numpy()[summ["h"]],
                                              htype=hyp["type"].to_numpy()[summ["h"]],
                                              regime_null=prep["clusters"][summ["k0"]],
                                              regime=prep["clusters"][summ["k"]],
                                              seconds=fit_["seconds"], p=fit_["p"]))
        per_fold[int(f)] = {"model": model, "hypotheses": hyp, "deltas": deltas, "skipped": list(hyp.attrs["skipped"]),
                            "prepare_seconds": prepare_seconds}
    return {"fits": pd.concat(frames, ignore_index=True), "folds": per_fold}


# ══════════════════════════════════════════════════════════════════════════════
# Decision thresholds, ROC and identifiability
# ══════════════════════════════════════════════════════════════════════════════

def cross_fitted_thresholds(scores: np.ndarray, folds: np.ndarray, budget: float) -> dict[int, float]:
    """For each fold, the score above which only ``budget`` of the *other* folds' normal schedules lie.

    Paper: eq:fitting:threshold, sec:fitting:evaluation

    Parameters
    ----------
    scores : (m,) scores of normal schedules; folds : (m,) their folds.
    budget : share of normal schedules allowed above the threshold.

    Returns
    -------
    dict fold → threshold.
    """
    folds = np.asarray(folds)
    return {int(f): float(np.quantile(scores[folds != f], 1 - budget)) for f in np.unique(folds)}


def roc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC curve and area of a score separating positives from negatives.

    Returns false-positive rates (m,), true-positive rates (m,) over all
    thresholds (decreasing), and the AUC — the probability that a random
    positive scores above a random negative (ties count one half).
    """
    pos, neg = np.asarray(scores_pos, dtype=float), np.asarray(scores_neg, dtype=float)
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    # Counts of scores at or above each threshold, by sorting once.
    tpr = np.r_[0.0, (len(pos) - np.searchsorted(np.sort(pos), thr, side="left")) / len(pos)]
    fpr = np.r_[0.0, (len(neg) - np.searchsorted(np.sort(neg), thr, side="left")) / len(neg)]
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return fpr, tpr, float(auc)


def operating_point(normal: pd.DataFrame, exceptional: pd.DataFrame, budget: float, *, truth: np.ndarray,
                    detectable: np.ndarray, profile_types: Sequence[int], labels: np.ndarray,
                    regimes: Sequence[int], e0_data: np.ndarray, e0_base: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """The fitter's performance at one false-positive budget, with cross-fitted thresholds.

    Paper: sec:fitting:evaluation

    Parameters
    ----------
    normal, exceptional : out-of-fold summaries (:func:`evaluate_folds`) of
        one hypothesis set and procedure, with ``row`` (index into the
        arrays below), ``fold``, ``score``, ``htype``, ``e0``, ``e0_null``,
        ``regime_null``.
    budget : share of other folds' normal schedules above the threshold.
    truth, detectable, e0_base : (m,) per exceptional row — its type, whether
        its disruption is detectable, its counterfactual e₀ (chapter 08).
    labels, e0_data : (n,) per normal row — chapter 06's regime and the
        schedule's own e₀.
    regimes : the regimes the generator serves (for the agreement).

    Returns
    -------
    metrics : dict of rates and their counts (``n_*``); a rate whose
    denominator is 0 is NaN.  flags_normal, flags_exceptional : bool arrays
    in the order of ``normal`` and ``exceptional``.
    """
    th = cross_fitted_thresholds(normal["score"].to_numpy(), normal["fold"].to_numpy(), budget)
    n_flag = normal["score"].to_numpy() > normal["fold"].map(th).to_numpy()
    e_flag = exceptional["score"].to_numpy() > exceptional["fold"].map(th).to_numpy()
    nrow, erow = normal["row"].to_numpy(), exceptional["row"].to_numpy()
    tt, det = truth[erow], detectable[erow]
    right = exceptional["htype"].to_numpy() == tt
    scored = np.isin(tt, profile_types)
    known = np.isin(labels[nrow], regimes)

    def rate(num_mask, den_mask):
        den = int(den_mask.sum())
        return (num_mask & den_mask).sum() / den if den else np.nan, int((num_mask & den_mask).sum()), den

    out = {}
    out["fp"], out["n_fp"], out["n_normal"] = rate(n_flag, np.ones_like(n_flag))
    out["detect"], out["n_detect"], out["n_detectable"] = rate(e_flag, det)
    out["detect_all"], out["n_detect_all"], out["n_scored"] = rate(e_flag, scored)
    out["type_acc"], out["n_type_right"], out["n_type_den"] = rate(right, e_flag & det)
    out["regime_agree"], _, _ = rate(normal["regime_null"].to_numpy() == labels[nrow], known)
    e0_fit_n = np.where(n_flag, normal["e0"], normal["e0_null"])
    e0_fit_e = np.where(e_flag, exceptional["e0"], exceptional["e0_null"])
    out["e0_mae_normal"] = float(np.abs(e0_fit_n - e0_data[nrow]).mean())
    out["e0_mae_exc"] = float(np.abs(e0_fit_e - e0_base[erow])[scored].mean())
    out["e0_mae_exc_null"] = float(np.abs(exceptional["e0_null"].to_numpy() - e0_base[erow])[scored].mean())
    out["n_unscored_flagged"], out["n_unscored"] = int(e_flag[~scored].sum()), int((~scored).sum())
    for d in profile_types:
        m = det & (tt == d)
        out[f"detect_{d}"], out[f"n_detect_{d}"], out[f"n_detectable_{d}"] = rate(e_flag, m)
        out[f"type_acc_{d}"], out[f"n_type_right_{d}"], out[f"n_type_den_{d}"] = rate(right, e_flag & m)
    return out, n_flag, e_flag


def identifiability(model: Mapping, deltas: np.ndarray, *, bracket: float, step: float, h: float) -> pd.DataFrame:
    """How far each profile can be told apart from a change of e₀ along each regime's trajectory.

    Paper: *eq:fitting:tangent, sec:fitting:identifiability

    The tangent ``t_k(x) = ∂z_k/∂e₀`` is a central difference of half-width
    ``h`` years of the calibrated generator, at points ``step`` apart over
    each regime's support.  For each profile ``δ`` the cosine ρ between
    ``t_k`` and ``δ`` (both as S·A vectors), the orthogonal fraction
    ``‖δ⊥‖/‖δ‖ = √(1 − ρ²)`` and the variance inflation ``1/(1 − ρ²)`` are
    returned.

    Returns
    -------
    DataFrame — ``regime``, ``e0``, ``profile`` (index into ``deltas``),
    ``cos``, ``orthogonal``, ``vif``.
    """
    D = np.asarray(deltas, dtype=float).reshape(len(deltas), -1)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    rows = []
    for k in model["clusters"]:
        lo, hi = model["support"][list(model["clusters"]).index(k)]
        xs = np.arange(lo + h, hi - h + 1e-9, step)
        t = ((tr.generate(model, int(k), xs + h, bracket=bracket) - tr.generate(model, int(k), xs - h, bracket=bracket))
             / (2 * h)).reshape(len(xs), -1)
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        cos = t @ D.T                                                     # (len(xs), H)
        for j in range(len(D)):
            rows.append(pd.DataFrame({"regime": int(k), "e0": xs, "profile": j, "cos": cos[:, j],
                                      "orthogonal": np.sqrt(1 - cos[:, j] ** 2),
                                      "vif": 1 / (1 - cos[:, j] ** 2)}))
    return pd.concat(rows, ignore_index=True)


def schedule_e0(z: np.ndarray) -> np.ndarray:
    """Sex-average e₀ of logit(q_x) schedules (n, S, A) → (n,)."""
    return lifetable.e0_from_logit(np.asarray(z, dtype=float)).mean(axis=-1)
