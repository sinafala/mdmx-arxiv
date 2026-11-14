"""Predicting complete schedules from summary indicators (paper §10).

Given child mortality ₅q₀ — or ₅q₀ and adult mortality ₄₅q₁₅ — predict a
complete single-year, sex-specific logit(q_x) schedule.  This module holds

1. **The MDMx predictor** ("SVD-Comp via the Tucker decomposition").  A
   small neural network maps the logit of the input indicators to the
   leading ``c_age`` × 2 core weights of the schedule in the shared sex and
   age bases (article §10).  The schedule is the linear reconstruction

       z = R · vec(G),   R = S ⊗ A_c   (the reconstruction matrix of the article's §10)

   with vec(G) the core weights stacked row by row (one sex component after the other), as in the article's notation,

   where ``S`` (2 × 2) is the sex basis, ``A_c`` (A × c_age) the leading
   columns of the smoothed age basis, and ``z`` stacks the female then the
   male schedule (length 2A).  The network is trained with the
   reconstruction loss of the article's §10 on the full schedule, with the
   ages 0–4 up-weighted by ``alpha`` (the article's w_child).  At prediction time the core weights
   are then changed — in the directions that move the indicators most per
   unit change of schedule (:func:`enforce_indicators`) — so that each
   predicted schedule reproduces its input indicators *exactly*
   — ₅q₀ of each sex, and ₄₅q₁₅ of each sex when it is an input — as
   Log-Quad does by construction.

2. **Cross-validation blocked by population**, plain and nested (the
   nested version chooses ``c_age`` and ``alpha`` inside each training
   set, so that the reported error is not optimistic).  Folds are formed from
   whole *population families*: a population together with every HMD
   population that overlaps it (e.g. the United Kingdom with England &
   Wales, Scotland and Northern Ireland).  All years of a family are in the
   same fold, so a model is never tested on a year whose neighbouring years
   — or whose overlapping sub-population — it was trained on.

3. **The classic SVD-Comp model** (Clark 2019), re-estimated on any set of
   training schedules so that it can be compared with MDMx on exactly the
   same out-of-fold points.  (Log-Quad is in :mod:`mdmx.logquad`.)

4. **Accuracy measures** used in the book and the paper's tables.

Layout of schedules
-------------------
A set of n schedules is a matrix ``Z`` of shape ``(n, 2A)``: row i holds
logit(q_x) for females at ages 0…A−1 followed by males at ages 0…A−1 (the
paper's vector z, sex varying before age in vec(G)).  Index 0 is female.

The model artefact (``svdcomp_model1``, ``svdcomp_model2``)
-----------------------------------------------------------
:func:`fit_core_model` returns a plain dict that is stored as an artefact
and is everything needed to make predictions (the Shiny app loads it):

=================  ============================================================
key                meaning
=================  ============================================================
``inputs``         list of input names in order, e.g. ``["5q0_F", "5q0_M"]``;
                   the network input is the **logit** of each indicator
``x_scaler``       ``{"mean", "scale"}`` (d,) standardisation of the inputs
``network``        ``{"sizes", "W": [...], "b": [...]}`` — see :class:`mdmx.nn.MLP`
``core_scaler``    ``{"mean", "scale"}`` (2·c_age,): the network predicts
                   standardised core weights; ``core = out·scale + mean``
``U_sex``          (2, 2) sex basis S
``U_age``          (A, c_age) leading columns of the smoothed age basis A
``ages``           (A,) ages 0…A−1
``loss_sexes``     sexes whose cells enter the loss (``[0, 1]`` = both)
``enforce``        ``{"tol", "max_iter"}`` of the step that makes predictions
                   reproduce the inputs named in ``inputs``
``settings``       hyper-parameters used (c_age, alpha, hidden, lr, …)
``history``        training curve and early-stopping summary
``description``    one line saying which model this is (added by the book)
=================  ============================================================

and :func:`predict_schedules` turns it back into schedules: the network
cores are ``g₀ = out · scale + mean``, :func:`enforce_indicators` moves
them to the ``g`` near g₀ whose schedule reproduces the inputs, and
``z = kron(U_sex, U_age) · g``.  Everything needed is in the dict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import lifetable, logquad, nn

__all__ = [
    "INDICATOR_COLUMNS", "schedules_matrix", "indicators", "features", "reconstruction_matrix",
    "loss_weights", "project_cores", "TRAINING_KEYS", "ENFORCE_KEYS", "MODEL_KEYS",
    "model_settings", "fit_core_model", "predict_cores", "indicator_constraints",
    "constraint_values", "enforce_indicators", "predict_with_diagnostics", "predict_schedules",
    "population_families", "assign_folds", "cross_validate", "cross_validate_core_model",
    "cross_validate_own_sex", "nested_cross_validate_core_model", "cross_validate_benchmark",
    "e0_by_sex", "accuracy_by_sex", "summarise_accuracy", "truncation_oracle",
    "abridged_logit_q", "fit_svdcomp_classic", "predict_svdcomp_classic",
]

#: Summary indicators and the ages they span: (start, width) for ``nqx``.
_INDICATOR_AGES = {"5q0": (0, 5), "45q15": (15, 45)}
_SEX_SUFFIX = ("F", "M")
#: Column names of :func:`indicators`, e.g. ``"5q0_F"``.
INDICATOR_COLUMNS = [f"{ind}_{sx}" for ind in _INDICATOR_AGES for sx in _SEX_SUFFIX]


# ══════════════════════════════════════════════════════════════════════════════
# Data in the shape the models use
# ══════════════════════════════════════════════════════════════════════════════

def schedules_matrix(M: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack the schedules of the selected country-years as rows.

    Parameters
    ----------
    M : (S, A, C, T) logit(q_x) tensor
    mask : (C, T) boolean, the country-years to take (e.g. ``observed``)

    Returns
    -------
    Z : (n, S·A) one row per selected country-year: female ages 0…A−1, then
        male ages 0…A−1
    cell : (n, 2) integer (country index, year index) of each row, in the
        row-major order of ``np.nonzero(mask)``
    """
    S, A = M.shape[:2]
    ci, ti = np.nonzero(mask)
    Z = np.moveaxis(M[:, :, ci, ti], -1, 0).reshape(len(ci), S * A)
    return Z, np.column_stack([ci, ti])


def indicators(Z: np.ndarray, A: int) -> pd.DataFrame:
    """₅q₀ and ₄₅q₁₅ of each sex for every schedule.

    Parameters
    ----------
    Z : (n, 2A) logit(q_x) schedules (female block, then male block)
    A : number of single-year ages

    Returns
    -------
    DataFrame (n rows) with columns ``5q0_F, 5q0_M, 45q15_F, 45q15_M``
    (probabilities), computed with :func:`mdmx.lifetable.nqx`.
    """
    q = lifetable.expit(Z.reshape(Z.shape[0], 2, A))
    cols = {}
    for ind, (start, width) in _INDICATOR_AGES.items():
        values = lifetable.nqx(q, start, width)               # (n, 2)
        for s, sx in enumerate(_SEX_SUFFIX):
            cols[f"{ind}_{sx}"] = values[:, s]
    return pd.DataFrame(cols)


def features(ind: pd.DataFrame, inputs: Sequence[str]) -> np.ndarray:
    """Network inputs: the logit of the named indicators, (n, len(inputs)).

    The logit puts indicators that span two orders of magnitude (₅q₀ from
    about 0.001 to 0.5) on a scale where differences are comparable
    (the inputs of the network of :func:`fit_core_model`).
    """
    unknown = [name for name in inputs if name not in ind.columns]
    if unknown:
        raise KeyError(f"unknown indicator(s) {unknown}; available: {list(ind.columns)}")
    return lifetable.logit(ind[list(inputs)].to_numpy())


# ══════════════════════════════════════════════════════════════════════════════
# The MDMx predictor
# ══════════════════════════════════════════════════════════════════════════════

def reconstruction_matrix(U_sex: np.ndarray, U_age: np.ndarray, c_age: int) -> np.ndarray:
    """Truncated reconstruction matrix ``R = S ⊗ A_c``.

    Paper: eq:svdcomp:recon

    Parameters
    ----------
    U_sex : (2, 2) sex factor matrix S
    U_age : (A, r₂) age factor matrix A (orthonormal columns)
    c_age : number of leading age components kept

    Returns
    -------
    R : (2A, 2·c_age) with ``R[s·A + a, i·c_age + j] = S[s, i] · A[a, j]``, so
        that ``z = R @ vec(G)`` with sex varying before age in both vectors.
        Because S and A_c have orthonormal columns, so does R; hence
        ``Rᵀ z`` is the least-squares core of a schedule z.
    """
    if not 1 <= c_age <= U_age.shape[1]:
        raise ValueError(f"c_age = {c_age} must be between 1 and {U_age.shape[1]}")
    return np.kron(U_sex, U_age[:, :c_age])


def loss_weights(A: int, alpha: float, young_ages: int, sexes: Sequence[int] = (0, 1)) -> np.ndarray:
    """Weight of each of the 2A schedule cells in the reconstruction loss.

    Paper: eq:svdcomp:loss

    The reconstruction loss with both sexes:

        L = 1/(2A) Σ_cells (ẑ − z)²  +  α · 1/(2·5) Σ_{s, a<5} (ẑ − z)²,

    i.e. every cell has weight 1/(2A) and the 10 cells at ages 0–4 have an
    extra α/10.  When ``sexes`` names only one sex (a sex-specific model
    that sees only that sex's indicator), the other sex's cells get weight
    0 and the normalising counts use one sex: 1/A and α/5.

    Returns an array (2A,) in the female-then-male layout of ``Z``.
    """
    n_sex = len(sexes)
    w = np.zeros((2, A))
    for s in sexes:
        w[s, :] = 1.0 / (n_sex * A)
        w[s, :young_ages] += alpha / (n_sex * young_ages)
    return w.ravel()


def project_cores(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Least-squares truncated cores of schedules: ``Z @ R`` → (n, 2·c_age).

    Valid because R has orthonormal columns (see :func:`reconstruction_matrix`).
    """
    return Z @ R


#: Training settings used by :func:`fit_core_model`.
TRAINING_KEYS = ("c_age", "alpha", "young_ages", "hidden", "lr", "max_epochs", "batch_size",
                 "l2", "val_frac", "patience")
#: Settings of the step that makes predictions reproduce their inputs
#: (:func:`enforce_indicators`).
ENFORCE_KEYS = ("enforce_tol", "enforce_max_iter")
#: Every setting a model needs (all from the ``[svdcomp]`` parameter section).
MODEL_KEYS = TRAINING_KEYS + ENFORCE_KEYS


def model_settings(section: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """The model settings from a parameter section, optionally overridden.

    Returns a new dict with exactly the keys in :data:`MODEL_KEYS`, e.g.
    ``model_settings(P.section("svdcomp"), c_age=20, alpha=0.0)`` for one
    point of a sensitivity analysis.
    """
    unknown = set(overrides) - set(MODEL_KEYS)
    if unknown:
        raise KeyError(f"not model settings: {sorted(unknown)}")
    return {key: overrides.get(key, section[key]) for key in MODEL_KEYS}


def fit_core_model(X: np.ndarray, Z: np.ndarray, *, inputs: Sequence[str], U_sex: np.ndarray,
                   U_age: np.ndarray, settings: dict[str, Any], seed: int,
                   groups: np.ndarray | None = None,
                   loss_sexes: Sequence[int] = (0, 1)) -> dict[str, Any]:
    """Train the MDMx summary-indicator model.

    Paper: sec:svdcomp:nn, eq:svdcomp:nn

    Steps:

    1. Standardise the inputs ``X`` (fitted on these rows only).
    2. Compute the least-squares truncated cores of the training schedules
       and their mean and standard deviation.  The network predicts
       *standardised* core weights; un-standardising is a fixed affine map,
       which is folded into the decoder.  (For HMD schedules the first core
       weight is of order −70 and the later ones of order ±1; without this
       step the output layer would spend much of training just reaching the
       right scale.)
    3. Train the MLP with :func:`mdmx.nn.fit_mlp` under the reconstruction
       loss: the decoded schedule ``ẑ = R (out ⊙ scale + mean)`` is compared
       with the observed schedule ``z`` cell by cell, with the cell weights
       of :func:`loss_weights`.  Early stopping
       uses a validation split of whole population families when ``groups``
       is given.

    The network is trained on the *unconstrained* reconstruction; the step
    that makes predictions reproduce their inputs exactly
    (:func:`enforce_indicators`) is applied afterwards, at prediction time,
    with the settings stored in the model.

    Parameters
    ----------
    X : (n, d) network inputs — logit indicators (see :func:`features`)
    Z : (n, 2A) observed logit(q_x) schedules of the same rows
    inputs : names of the d inputs (e.g. ``"5q0_F"``), stored with the
        model; they also say which indicators predictions must reproduce
    U_sex, U_age : sex basis (2, 2) and smoothed age basis (A, r₂)
    settings : dict with the keys of :data:`MODEL_KEYS` — ``c_age``
        (leading age components predicted), ``alpha`` and ``young_ages``
        (up-weighting of ages 0…young_ages−1 in the loss), the network and
        training settings passed to :func:`mdmx.nn.fit_mlp`, and the
        tolerance and iteration limit of :func:`enforce_indicators`
    seed : integer seed of the network (initial weights, validation split,
        batch order)
    groups : (n,) population-family label of each row, for the validation split
    loss_sexes : sexes whose cells enter the loss (both by default)

    Returns
    -------
    The self-contained model dict described in the module docstring.
    """
    A = U_age.shape[0]
    if Z.shape[1] != 2 * A:
        raise ValueError(f"Z has {Z.shape[1]} columns but the age basis implies {2 * A}")
    s = {key: settings[key] for key in MODEL_KEYS}
    indicator_constraints(inputs)                     # fails early on an unknown input name
    x_scaler = nn.Standardizer.fit(X)
    R = reconstruction_matrix(U_sex, U_age, s["c_age"])
    core_scaler = nn.Standardizer.fit(project_cores(Z, R))
    loss = nn.LinearDecoderLoss(D=R * core_scaler.scale, offset=R @ core_scaler.mean,
                                weights=loss_weights(A, s["alpha"], s["young_ages"], loss_sexes))
    net, history = nn.fit_mlp(x_scaler.transform(X), Z, s["hidden"], lr=s["lr"],
                              epochs=s["max_epochs"], batch_size=s["batch_size"], l2=s["l2"],
                              seed=seed, val_frac=s["val_frac"], patience=s["patience"],
                              loss=loss, out_dim=R.shape[1], groups=groups)
    return {"inputs": list(inputs), "x_scaler": x_scaler.to_dict(), "network": net.to_dict(),
            "core_scaler": core_scaler.to_dict(), "U_sex": np.array(U_sex, dtype=float),
            "U_age": np.array(U_age[:, :s["c_age"]], dtype=float), "ages": np.arange(A),
            "loss_sexes": [int(x) for x in loss_sexes],
            "enforce": {"tol": float(s["enforce_tol"]), "max_iter": int(s["enforce_max_iter"])},
            "settings": {**s, "seed": int(seed)}, "history": history}


def predict_cores(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """The network's truncated core weights, (n, 2·c_age) = vec(Ĝ) row by row.

    Paper: eq:svdcomp:nn

    ``X`` holds the logit inputs in the order of ``model["inputs"]``.  These
    are the cores *before* :func:`enforce_indicators`.
    """
    x_scaler = nn.Standardizer.from_dict(model["x_scaler"])
    core_scaler = nn.Standardizer.from_dict(model["core_scaler"])
    net = nn.MLP.from_dict(model["network"])
    return core_scaler.inverse_transform(net.predict(x_scaler.transform(X)))


# ── Reproducing the input indicators exactly ─────────────────────────────────

def indicator_constraints(inputs: Sequence[str]) -> list[tuple[int, int, int]]:
    """``(sex, first age, width)`` of each input indicator, e.g. ``"45q15_M"`` → (1, 15, 45)."""
    out = []
    for name in inputs:
        ind, _, sx = name.partition("_")
        if ind not in _INDICATOR_AGES or sx not in _SEX_SUFFIX:
            raise KeyError(f"unknown indicator {name!r}; expected e.g. '5q0_F' or '45q15_M'")
        out.append((_SEX_SUFFIX.index(sx), *_INDICATOR_AGES[ind]))
    return out


def constraint_values(z: np.ndarray, constraints: Sequence[tuple[int, int, int]], A: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """logit of each constrained indicator and its gradient with respect to z.

    Paper: eq:svdcomp:enforce

    For an indicator ``h = 1 − Π_{a∈W} (1 − q_a)`` over the age window W of
    one sex, with ``q_a = expit(z_a)``, the derivative of ``logit(h)`` with
    respect to the logit ``z_a`` is ``q_a / h`` for a in W and 0 elsewhere.

    Parameters
    ----------
    z : (n, 2A) logit(q_x) schedules
    constraints : m tuples ``(sex, first age, width)`` (:func:`indicator_constraints`)
    A : number of single-year ages

    Returns
    -------
    values : (n, m) logit of each indicator
    grad : (n, m, 2A) its gradient with respect to z
    """
    n, m = z.shape[0], len(constraints)
    values = np.empty((n, m))
    grad = np.zeros((n, m, z.shape[1]))
    for j, (s, first, width) in enumerate(constraints):
        cols = slice(s * A + first, s * A + first + width)
        q = lifetable.expit(z[:, cols])
        h = 1.0 - np.prod(1.0 - q, axis=1)
        values[:, j] = lifetable.logit(h)
        grad[:, j, cols] = q / h[:, None]
    return values, grad


def enforce_indicators(cores: np.ndarray, R: np.ndarray, constraints: Sequence[tuple[int, int, int]],
                       targets: np.ndarray, *, tol: float, max_iter: int,
                       max_halvings: int = 30) -> dict[str, np.ndarray]:
    """Change the core weights so that the schedules reproduce the input indicators.

    Paper: *eq:svdcomp:enforce

    For one schedule with network cores ``g₀`` (its truncated core weights)
    and m target indicators ``t_j`` (₅q₀ of each sex, and ₄₅q₁₅ of each sex
    when it is an input), write ``f_j(g) = logit h_j(R g) − logit t_j``,
    where ``h_j(z)`` is the indicator implied by the schedule z.

    *Directions.*  To first order, the smallest change of core weights that
    removes the errors ``f(g₀)`` is ``Δg = −Jᵀ (J Jᵀ)⁻¹ f(g₀)``, with
    ``J = ∂f/∂g`` (m × 2c) at g₀: a combination of the m columns of
    ``D = Jᵀ``.  Column j of D is the gradient of indicator j with respect
    to the schedule, projected onto the truncated basis — the change of
    schedule that moves indicator j furthest per unit of change, so it is
    concentrated in that indicator's age window as far as the basis allows.
    Because R has orthonormal columns, ``‖Δg‖`` is the root-sum-of-squares
    change of the whole logit(q_x) schedule, and because the span of
    R = S ⊗ A_c treats the sexes separately, a female indicator's direction
    changes only the female schedule.

    *Coefficients.*  The correction is ``g = g₀ + D μ`` with the m
    coefficients μ chosen so that every indicator is reproduced *exactly*:
    the m equations ``f(g₀ + D μ) = 0`` in m unknowns are solved by Newton's
    method, ``μ ← μ − (J(g) D)⁻¹ f(g)``, starting from μ = 0, halving a step
    (up to ``max_halvings`` times) whenever it would not reduce the largest
    |f_j| by at least the fraction 10⁻⁴ of the step.  The directions are
    fixed at g₀, so the problem stays small and well posed; the solution is
    the first-order smallest change, made exact for the nonlinear
    indicators.  (Re-computing the directions at every step would reach only
    a stationary point of the size of the change among the exact solutions,
    and that iteration can fail far from the solution; in tests the two
    differ in size by about 1%.)

    Parameters
    ----------
    cores : (n, k) network core weights g₀ (k = 2·c_age)
    R : (2A, k) reconstruction matrix
    constraints : m tuples ``(sex, first age, width)`` (:func:`indicator_constraints`)
    targets : (n, m) target indicator values (probabilities)
    tol : convergence tolerance on |logit h − logit t| (every constraint)
    max_iter : maximum number of Newton iterations
    max_halvings : maximum number of step halvings within one iteration

    Returns
    -------
    dict with ``cores`` (n, k) adjusted cores, ``converged`` (n,) boolean,
    ``max_residual`` (n,) the largest |logit h − logit t| at the end, and
    ``iterations`` (the number of iterations used, an int).  Rows that did
    not converge — including any whose m × m Newton matrix became singular —
    keep their last iterate; the caller must check ``converged`` (see
    :func:`predict_schedules`).
    """
    A = R.shape[0] // 2
    g0 = np.asarray(cores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (g0.shape[0], len(constraints)):
        raise ValueError(f"targets have shape {targets.shape}, expected {(g0.shape[0], len(constraints))}")
    target_logit = lifetable.logit(targets)

    def errors(mu: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = g0 + np.einsum("nkm,nm->nk", D, mu)
        values, grad_z = constraint_values(g @ R.T, constraints, A)
        return values - target_logit, grad_z

    f, grad_z = errors(np.zeros_like(target_logit), np.zeros(g0.shape + (len(constraints),)))
    D = (grad_z @ R).transpose(0, 2, 1)                       # (n, k, m), fixed directions
    mu = np.zeros_like(target_logit)
    stuck = np.zeros(g0.shape[0], dtype=bool)                 # singular Newton matrix
    iterations = 0
    for iterations in range(max_iter + 1):
        norm = np.abs(f).max(axis=1)
        active = (norm >= tol) & ~stuck
        if not active.any() or iterations == max_iter:
            break
        # Newton matrix ∂f/∂μ = (∂f/∂z) R D, one m × m matrix per row.
        jac = (grad_z @ R) @ D
        det = np.linalg.det(jac)
        stuck |= active & ~(np.isfinite(det) & (np.abs(det) > 1e-300))
        active &= ~stuck
        delta = np.zeros_like(mu)
        delta[active] = -np.linalg.solve(jac[active], f[active][..., None])[..., 0]
        step = np.where(active, 1.0, 0.0)
        for halving in range(max_halvings + 1):
            f_try, grad_try = errors(mu + step[:, None] * delta, D)
            worse = active & (np.abs(f_try).max(axis=1) > (1.0 - 1e-4 * step) * norm)
            if not worse.any() or halving == max_halvings:
                break
            step = np.where(worse, step / 2.0, step)
        mu = mu + step[:, None] * delta              # f_try and grad_try belong to this step
        f, grad_z = f_try, grad_try
    residual = np.abs(f).max(axis=1)
    return {"cores": g0 + np.einsum("nkm,nm->nk", D, mu), "converged": residual < tol,
            "max_residual": residual, "iterations": int(iterations)}


def predict_with_diagnostics(model: dict[str, Any], X: np.ndarray) -> dict[str, Any]:
    """Predicted schedules before and after :func:`enforce_indicators`, with diagnostics.

    Parameters
    ----------
    model : a model dict from :func:`fit_core_model`
    X : (n, d) logit inputs in the order of ``model["inputs"]``

    Returns
    -------
    dict with ``Z_raw`` (n, 2A) the network's reconstruction, ``Z_hat``
    (n, 2A) the schedules that reproduce the inputs, and ``converged``,
    ``max_residual``, ``iterations`` from :func:`enforce_indicators`.
    """
    R = np.kron(model["U_sex"], model["U_age"])
    cores = predict_cores(model, X)
    res = enforce_indicators(cores, R, indicator_constraints(model["inputs"]), lifetable.expit(X),
                             tol=model["enforce"]["tol"], max_iter=model["enforce"]["max_iter"])
    return {"Z_raw": cores @ R.T, "Z_hat": res["cores"] @ R.T, "converged": res["converged"],
            "max_residual": res["max_residual"], "iterations": res["iterations"]}


def predict_schedules(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predicted logit(q_x) schedules (n, 2A) that reproduce the inputs exactly.

    Paper: eq:svdcomp:enforce

    ``X`` holds the logit inputs (n, d) in the order of ``model["inputs"]``.
    Raises ``RuntimeError`` if the constraint step fails to converge for any
    row (it has not for any HMD country-year; the error names the rows).
    """
    out = predict_with_diagnostics(model, X)
    if not out["converged"].all():
        bad = np.flatnonzero(~out["converged"])
        raise RuntimeError(f"the inputs could not be reproduced for rows {bad.tolist()} "
                           f"(largest residual {out['max_residual'].max():.2e})")
    return out["Z_hat"]


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation blocked by population family
# ══════════════════════════════════════════════════════════════════════════════

def population_families(pops: Sequence[str], register: pd.DataFrame) -> np.ndarray:
    """Family label of each population: the aggregate it belongs to, or itself.

    Paper: sec:svdcomp:eval

    Parameters
    ----------
    pops : population codes in tensor order
    register : the population register (``config/populations.csv``) with
        columns ``code`` and ``aggregate_of`` (empty for top-level series)

    Returns
    -------
    array of str (len(pops),).  For example GBRTENW, GBR_SCO, GBR_NIR and
    GBR_NP all get ``"GBR_NP"``; DEUTE, DEUTW and DEUTNP get ``"DEUTNP"``.
    Nested aggregates resolve to the top-level series (GBRCENW, part of
    GBRTENW, gets ``"GBR_NP"``), so that no family is split across folds.
    """
    parent = dict(zip(register["code"], register["aggregate_of"].fillna(""), strict=True))
    missing = [p for p in pops if p not in parent]
    if missing:
        raise KeyError(f"populations not in the register: {missing}")

    def root(code: str) -> str:
        seen = [code]
        while parent.get(code):
            code = parent[code]
            if code in seen:
                raise ValueError(f"the register's aggregate_of links form a cycle: {' -> '.join(seen + [code])}")
            seen.append(code)
        return code

    return np.array([root(p) for p in pops])


def assign_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign rows to ``n_folds`` folds so that each group is in one fold.

    Paper: sec:svdcomp:eval

    Groups are visited from largest to smallest (ties in a random order set
    by ``seed``) and each is put in the fold that currently has the fewest
    rows.  This keeps the folds close to equal in size even though the
    populations have very different numbers of observed years.

    Parameters
    ----------
    groups : (n,) group label of each row (e.g. population family)

    Returns
    -------
    fold : (n,) integer fold 0…n_folds−1 of each row
    """
    labels, sizes = np.unique(groups, return_counts=True)
    if n_folds > len(labels):
        raise ValueError(f"{n_folds} folds but only {len(labels)} groups")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(labels))
    order = shuffled[np.argsort(-sizes[shuffled], kind="stable")]
    fold_of_label, totals = {}, np.zeros(n_folds, dtype=int)
    for i in order:
        k = int(np.argmin(totals))
        fold_of_label[labels[i]] = k
        totals[k] += sizes[i]
    return np.array([fold_of_label[g] for g in groups])


def cross_validate(folds: np.ndarray,
                   fit_predict: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
                   width: int) -> np.ndarray:
    """Out-of-fold predictions: each row predicted by a model not trained on its fold.

    Parameters
    ----------
    folds : (n,) fold of each row
    fit_predict : ``fit_predict(k, train_mask, test_mask)`` trains on the
        rows in ``train_mask`` and returns predictions (n_test, width) for
        the rows in ``test_mask`` (in row order); ``k`` is the fold number,
        used for example to vary the seed
    width : number of columns of a prediction

    Returns
    -------
    (n, width) array; every row is filled exactly once.
    """
    out = np.full((len(folds), width), np.nan)
    for k in np.unique(folds):
        test = folds == k
        out[test] = fit_predict(int(k), ~test, test)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("some out-of-fold predictions are missing or not finite")
    return out


def cross_validate_core_model(X: np.ndarray, Z: np.ndarray, *, inputs: Sequence[str],
                              folds: np.ndarray, groups: np.ndarray, settings: dict[str, Any],
                              seed: int, U_sex: np.ndarray | None = None,
                              U_age: np.ndarray | None = None,
                              basis: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                              loss_sexes: Sequence[int] = (0, 1),
                              fold_settings: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Out-of-fold predictions of the MDMx model, with the same training as the final fit.

    Paper: sec:svdcomp:eval

    For each fold k the model is trained on the other folds — with early
    stopping on a validation split of whole population families drawn from
    those training folds — and predicts the schedules of fold k, which are
    then made to reproduce their inputs (:func:`enforce_indicators`).  The
    seed of fold k is ``seed + k``.

    Parameters
    ----------
    X, Z, inputs, settings, loss_sexes : as in :func:`fit_core_model`
    folds : (n,) fold of each row (:func:`assign_folds`)
    groups : (n,) population family of each row
    U_sex, U_age : the sex and age bases, used for every fold, or
    basis : alternatively a function ``basis(train_mask) -> (U_sex, U_age)``
        that re-estimates the bases from the training rows of each fold
    fold_settings : optional fold → settings; fold k then uses
        ``fold_settings[k]`` instead of ``settings`` (used to apply the
        settings chosen by a nested cross-validation)

    Returns
    -------
    dict with ``Z_hat`` (n, 2A) out-of-fold schedules that reproduce their
    inputs, ``Z_raw`` (n, 2A) the network's schedules before that step, and
    ``runs``, a DataFrame with one row per fold: ``fold``, ``n_train``,
    ``best_epoch``, ``epochs_run``, ``stopped_early``, ``c_age``, ``alpha``,
    ``enforce_iterations`` and ``not_converged`` (rows of the fold for which
    the inputs could not be reproduced; their last iterate is kept and the
    count is reported, never hidden).
    """
    if (basis is None) == (U_sex is None or U_age is None):
        raise ValueError("give either fixed bases (U_sex and U_age) or a basis function")
    width = Z.shape[1]
    raw = np.full((len(folds), width), np.nan)
    runs = []

    def fit_predict(k: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        S, Aa = basis(train) if basis is not None else (U_sex, U_age)
        s = fold_settings[k] if fold_settings is not None else settings
        model = fit_core_model(X[train], Z[train], inputs=inputs, U_sex=S, U_age=Aa,
                               settings=s, seed=seed + k, groups=groups[train],
                               loss_sexes=loss_sexes)
        out = predict_with_diagnostics(model, X[test])
        raw[test] = out["Z_raw"]
        h = model["history"]
        runs.append({"fold": k, "n_train": int(train.sum()), "best_epoch": h["best_epoch"],
                     "epochs_run": h["epochs_run"], "stopped_early": h["stopped_early"],
                     "c_age": int(s["c_age"]), "alpha": float(s["alpha"]),
                     "enforce_iterations": out["iterations"],
                     "not_converged": int((~out["converged"]).sum())})
        return out["Z_hat"]

    Z_hat = cross_validate(folds, fit_predict, width)
    return {"Z_hat": Z_hat, "Z_raw": raw, "runs": pd.DataFrame(runs)}


def cross_validate_own_sex(ind: pd.DataFrame, Z: np.ndarray, *, indicator_names: Sequence[str],
                           folds: np.ndarray, groups: np.ndarray, settings: dict[str, Any],
                           seed: int, U_sex: np.ndarray | None = None,
                           U_age: np.ndarray | None = None,
                           basis: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                           fold_settings: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Out-of-fold predictions of sex-specific MDMx models.

    Paper: sec:svdcomp:benchmarks

    Log-Quad and SVD-Comp predict each sex's schedule from that sex's
    indicators only, whereas MDMx Models 1 and 2 use both sexes' indicators.
    To separate the effect of the model from the effect of the extra
    information, this variant trains one network per sex whose inputs are
    that sex's indicators only (e.g. ``5q0_F`` for females), whose loss
    covers that sex's schedule only (:func:`loss_weights` with one sex), and
    whose predictions are made to reproduce those inputs.  The two halves
    are put together into one (n, 2A) prediction.

    Parameters
    ----------
    indicator_names : e.g. ``["5q0"]`` or ``["5q0", "45q15"]``
    other arguments as in :func:`cross_validate_core_model`

    Returns
    -------
    dict with ``Z_hat`` (n, 2A) and ``runs`` (with a ``sex`` column).
    """
    A = Z.shape[1] // 2
    Z_hat = np.empty_like(Z)
    runs = []
    for s, sx in enumerate(_SEX_SUFFIX):
        inputs = [f"{name}_{sx}" for name in indicator_names]
        res = cross_validate_core_model(features(ind, inputs), Z, inputs=inputs, folds=folds,
                                        groups=groups, settings=settings, seed=seed, U_sex=U_sex,
                                        U_age=U_age, basis=basis, loss_sexes=(s,),
                                        fold_settings=fold_settings)
        Z_hat[:, s * A:(s + 1) * A] = res["Z_hat"][:, s * A:(s + 1) * A]
        runs.append(res["runs"].assign(sex=sx))
    return {"Z_hat": Z_hat, "runs": pd.concat(runs, ignore_index=True)}


def nested_cross_validate_core_model(X: np.ndarray, Z: np.ndarray, *, inputs: Sequence[str],
                                     folds: np.ndarray, groups: np.ndarray,
                                     settings: dict[str, Any], grid: Sequence[dict[str, Any]],
                                     n_inner: int, inner_seed: int, seed: int,
                                     U_sex: np.ndarray | None = None, U_age: np.ndarray | None = None,
                                     basis: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                                     ) -> dict[str, Any]:
    """Out-of-fold predictions with the settings chosen inside each training set.

    Paper: sec:svdcomp:eval

    Choosing settings (e.g. ``c_age`` and ``alpha``) by the out-of-fold
    error and then quoting that error is optimistic: the test folds helped
    choose.  A nested cross-validation avoids this.  For each outer fold k:

    1. the population families of the *training* folds are dealt into
       ``n_inner`` inner folds (:func:`assign_folds`, seed ``inner_seed``);
    2. every candidate in ``grid`` is cross-validated over those inner
       folds (training rows only), and scored by the mean over both sexes
       of the mean absolute error of e₀ of its schedules — after they have
       been made to reproduce their inputs;
    3. the best candidate is trained on all training rows of fold k and
       predicts fold k.

    The test fold therefore plays no part in choosing its own settings.
    With ``basis``, every network is trained in bases estimated from its own
    training rows: those of the inner fold in step 2, those of the outer
    fold in step 3, so the test fold also plays no part in the bases.

    Parameters
    ----------
    X, Z, inputs, folds, groups : as in :func:`cross_validate_core_model`
    U_sex, U_age : fixed bases for every network, or
    basis : a function ``basis(train_mask) -> (U_sex, U_age)`` of a mask
        over all n rows, called with the training rows of every inner and
        outer fit
    settings : base model settings (:data:`MODEL_KEYS`)
    grid : candidate overrides, e.g. ``[{"c_age": 6, "alpha": 10.0}, …]``
    n_inner, inner_seed : number of inner folds and their seed
    seed : seeds as in :func:`cross_validate_core_model`; the inner
        cross-validation of outer fold k uses ``seed + 1000 (k + 1)``

    Returns
    -------
    dict with ``Z_hat``, ``Z_raw`` and ``runs`` as in
    :func:`cross_validate_core_model`, ``fold_settings`` (fold → chosen
    settings) and ``inner``, a DataFrame with the inner score of every
    candidate in every outer fold (columns ``fold``, the grid keys,
    ``inner_e0_mae``, ``not_converged``, ``networks``, ``at_cap`` (networks
    stopped by the epoch cap) and ``chosen``).
    """
    if (basis is None) == (U_sex is None or U_age is None):
        raise ValueError("give either fixed bases (U_sex and U_age) or a basis function")
    A = Z.shape[1] // 2
    inner_rows, fold_settings = [], {}
    for k in np.unique(folds):
        k = int(k)
        train = folds != k
        inner_folds = assign_folds(groups[train], n_inner, inner_seed)
        e0_obs = e0_by_sex(Z[train], A)
        inner_basis = None
        if basis is not None:
            idx = np.flatnonzero(train)

            def inner_basis(inner_train: np.ndarray, idx: np.ndarray = idx):
                # The inner fit sees the rows of X[train]; the bases need the
                # same rows as a mask over all n rows.
                full = np.zeros(len(folds), dtype=bool)
                full[idx[inner_train]] = True
                return basis(full)
        rows = []
        for cand in grid:
            res = cross_validate_core_model(X[train], Z[train], inputs=inputs, folds=inner_folds,
                                            groups=groups[train], settings={**settings, **cand},
                                            seed=seed + 1000 * (k + 1),
                                            U_sex=None if basis is not None else U_sex,
                                            U_age=None if basis is not None else U_age,
                                            basis=inner_basis)
            cap = {**settings, **cand}["max_epochs"]
            rows.append({"fold": k, **cand,
                         "inner_e0_mae": float(np.mean(np.abs(e0_by_sex(res["Z_hat"], A) - e0_obs))),
                         "not_converged": int(res["runs"]["not_converged"].sum()),
                         "networks": len(res["runs"]),
                         "at_cap": int((res["runs"]["epochs_run"] >= cap).sum())})
        best = int(np.argmin([r["inner_e0_mae"] for r in rows]))
        for i, r in enumerate(rows):
            r["chosen"] = i == best
        inner_rows += rows
        fold_settings[k] = {**settings, **grid[best]}
    outer = cross_validate_core_model(X, Z, inputs=inputs, folds=folds, groups=groups,
                                      settings=settings, seed=seed, U_sex=U_sex, U_age=U_age,
                                      basis=basis, fold_settings=fold_settings)
    return {**outer, "fold_settings": fold_settings, "inner": pd.DataFrame(inner_rows)}


def cross_validate_benchmark(method: str, ind: pd.DataFrame, Z: np.ndarray, *, folds: np.ndarray,
                             two_parameter: bool, settings: dict[str, Any]) -> np.ndarray:
    """Out-of-fold predictions of Log-Quad or classic SVD-Comp, re-estimated per fold.

    Paper: sec:svdcomp:benchmarks

    Each sex is fitted to the training folds' schedules of that sex and
    predicts the test fold's schedules from that sex's ₅q₀ (and ₄₅q₁₅ if
    ``two_parameter``), exactly as the published models are used.

    Parameters
    ----------
    method : ``"logquad"`` or ``"svdcomp"``
    ind, Z : indicators and observed schedules of all rows
    settings : the ``[svdcomp]`` parameter section (uses
        ``logquad_v_ages``, ``svdcomp_components``, ``svdcomp_offset``)

    Returns
    -------
    (n, 2A) out-of-fold logit(q_x) schedules.
    """
    if method not in ("logquad", "svdcomp"):
        raise ValueError(f"unknown benchmark {method!r}")
    A = Z.shape[1] // 2

    def fit_predict(k: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        out = np.empty((int(test.sum()), 2 * A))
        for s, sx in enumerate(_SEX_SUFFIX):
            Zs = Z[:, s * A:(s + 1) * A]
            q5 = ind[f"5q0_{sx}"].to_numpy()
            q45 = ind[f"45q15_{sx}"].to_numpy()
            q45_in = q45[test] if two_parameter else None
            if method == "logquad":
                coefs = logquad.fit_logquad(lifetable.expit(Zs[train]),
                                            v_ages=tuple(settings["logquad_v_ages"]))
                q, _ = logquad.predict_logquad(coefs, q5[test], q45_in, A)
                out[:, s * A:(s + 1) * A] = lifetable.logit(q)
            else:
                coefs = fit_svdcomp_classic(Zs[train], q5[train], q45[train],
                                            n_components=settings["svdcomp_components"],
                                            offset=settings["svdcomp_offset"])
                out[:, s * A:(s + 1) * A] = predict_svdcomp_classic(coefs, q5[test], q45_in)
        return out

    return cross_validate(folds, fit_predict, 2 * A)


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def e0_by_sex(Z: np.ndarray, A: int) -> np.ndarray:
    """Life expectancy at birth of each schedule and sex, (n, 2)."""
    return lifetable.e0_from_logit(Z.reshape(Z.shape[0], 2, A))


def accuracy_by_sex(Z_hat: np.ndarray, Z: np.ndarray, A: int,
                    age_groups: Sequence[Sequence[int]]) -> pd.DataFrame:
    """Accuracy of predicted schedules against observed ones, by sex.

    Parameters
    ----------
    Z_hat, Z : (n, 2A) predicted and observed logit(q_x) schedules
    A : number of single-year ages
    age_groups : list of inclusive ``[first, last]`` ages, e.g.
        ``[[0, 0], [1, 4], [5, 14], [15, 59], [60, 109]]``

    Returns
    -------
    DataFrame with one row per sex (index ``"female"``, ``"male"``) and
    columns ``e0_mae`` and ``e0_bias`` (years; bias = mean of predicted −
    observed), ``rmse_all`` and ``rmse_<first>_<last>`` (root-mean-square
    error of logit(q_x) over all country-years and the single-year ages in
    the group), ``q5_mae`` and ``q45_mae`` (absolute error of the implied
    ₅q₀ and ₄₅q₁₅).
    """
    n = Z.shape[0]
    err = (Z_hat - Z).reshape(n, 2, A)
    de0 = e0_by_sex(Z_hat, A) - e0_by_sex(Z, A)
    ind_hat, ind = indicators(Z_hat, A), indicators(Z, A)
    rows = []
    for s, sx in enumerate(_SEX_SUFFIX):
        row = {"e0_mae": np.mean(np.abs(de0[:, s])), "e0_bias": np.mean(de0[:, s]),
               "rmse_all": np.sqrt(np.mean(err[:, s, :] ** 2))}
        for first, last in age_groups:
            row[f"rmse_{first}_{last}"] = np.sqrt(np.mean(err[:, s, first:last + 1] ** 2))
        row["q5_mae"] = np.mean(np.abs(ind_hat[f"5q0_{sx}"] - ind[f"5q0_{sx}"]))
        row["q45_mae"] = np.mean(np.abs(ind_hat[f"45q15_{sx}"] - ind[f"45q15_{sx}"]))
        rows.append(row)
    return pd.DataFrame(rows, index=["female", "male"])


def abridged_logit_q(Z: np.ndarray, A: int, starts: Sequence[int]) -> np.ndarray:
    """logit(ₙq_x) in abridged age groups from single-year schedules.

    Paper: sec:svdcomp:benchmarks

    Parameters
    ----------
    Z : (n, 2A) logit(q_x) schedules
    starts : first age of each group, e.g. ``[0, 1, 5, 10, …, 105]``; each
        group runs to the next start (the last to age A−1)

    Returns
    -------
    (n, 2, G) logit of the probability of dying within each group.  Used to
    compare models on the five-year age groups in which Log-Quad works.
    """
    q = lifetable.expit(Z.reshape(Z.shape[0], 2, A))
    bounds = list(starts) + [A]
    out = np.stack([lifetable.nqx(q, a, b - a) for a, b in zip(bounds[:-1], bounds[1:], strict=True)], axis=-1)
    return lifetable.logit(out)


def summarise_accuracy(predictions: dict[str, np.ndarray], Z: np.ndarray, A: int,
                       age_groups: Sequence[Sequence[int]],
                       abridged_starts: Sequence[int]) -> pd.DataFrame:
    """Accuracy of several sets of predictions, one row per (method, sex).

    Parameters
    ----------
    predictions : method name → (n, 2A) predicted logit(q_x) schedules
    Z : (n, 2A) observed schedules
    age_groups : single-year age groups for :func:`accuracy_by_sex`
    abridged_starts : abridged groups for :func:`abridged_logit_q`

    Returns
    -------
    DataFrame with columns ``method``, ``sex`` and every column of
    :func:`accuracy_by_sex`, plus ``rmse_abridged``: the RMSE of
    logit(ₙq_x) over the abridged age groups (the scale on which Log-Quad
    is defined, so the fairest single summary for comparing with it).
    """
    observed_ab = abridged_logit_q(Z, A, abridged_starts)
    frames = []
    for name, Z_hat in predictions.items():
        acc = accuracy_by_sex(Z_hat, Z, A, age_groups)
        err_ab = abridged_logit_q(Z_hat, A, abridged_starts) - observed_ab
        acc["rmse_abridged"] = np.sqrt(np.mean(err_ab ** 2, axis=(0, 2)))
        frames.append(acc.rename_axis("sex").reset_index().assign(method=name))
    out = pd.concat(frames, ignore_index=True)
    return out[["method", "sex"] + [c for c in out.columns if c not in ("method", "sex")]]


def truncation_oracle(Z: np.ndarray, U_sex: np.ndarray, U_age: np.ndarray, c_age: int,
                      weights: np.ndarray) -> np.ndarray:
    """The best schedules any predictor of the truncated core could produce.

    Paper: sec:svdcomp:framework

    For each observed schedule z, the core weights g minimising the
    weighted reconstruction loss ``Σ_j w_j (R g − z)_j²`` are found by
    weighted least squares,

        g = (Rᵀ W R)⁻¹ Rᵀ W z,     W = diag(weights),

    and the schedule ``R g`` is returned.  A network trained with the same
    loss weights can at best reach this; the gap between the oracle and the
    observed schedule is the price of the truncation (and of the weights)
    alone, before any prediction error.

    Parameters
    ----------
    Z : (n, 2A) observed schedules
    U_sex, U_age, c_age : bases and truncation (see :func:`reconstruction_matrix`)
    weights : (2A,) loss weights (:func:`loss_weights`); all-equal weights
        give the ordinary projection ``R Rᵀ z``

    Returns
    -------
    (n, 2A) oracle schedules.
    """
    R = reconstruction_matrix(U_sex, U_age, c_age)
    RW = R * weights[:, None]
    G = np.linalg.solve(R.T @ RW, RW.T @ Z.T).T            # (n, 2·c_age)
    return G @ R.T


# ══════════════════════════════════════════════════════════════════════════════
# The classic SVD-Comp model (Clark 2019), re-estimated on training data
# ══════════════════════════════════════════════════════════════════════════════

#: Terms of the SVD-Comp weight regressions (Clark 2019, Eq. 12), in the
#: order used by the R package svdComp5q0: cm = ₅q₀, cml = logit ₅q₀, am =
#: ₄₅q₁₅, aml = logit ₄₅q₁₅; "s" = squared, "c" = cubed.
SVDCOMP_TERMS = ["(Intercept)", "cm", "cml", "cmls", "cmlc", "am", "amls", "amlc", "cmlaml"]
SVDCOMP_AML_TERMS = ["(Intercept)", "cm", "cml", "cmls", "cmlc"]
SVDCOMP_Q0_TERMS = ["(Intercept)", "cml", "cmls"]


def _svdcomp_design(q5: np.ndarray, q45: np.ndarray | None) -> dict[str, np.ndarray]:
    """All candidate regression terms as columns (``q45`` may be None)."""
    cml = lifetable.logit(q5)
    terms = {"(Intercept)": np.ones_like(q5), "cm": q5, "cml": cml, "cmls": cml ** 2, "cmlc": cml ** 3}
    if q45 is not None:
        aml = lifetable.logit(q45)
        terms.update({"am": q45, "amls": aml ** 2, "amlc": aml ** 3, "cmlaml": cml * aml})
    return terms


def _ols(design: dict[str, np.ndarray], names: Sequence[str], y: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients of ``y`` (n, ...) on the named terms."""
    Xd = np.column_stack([design[name] for name in names])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return coef


def fit_svdcomp_classic(Zs: np.ndarray, q5: np.ndarray, q45: np.ndarray, *, n_components: int,
                        offset: float) -> dict[str, Any]:
    """Estimate the classic, single-sex SVD-Comp model (Clark 2019).

    Paper: sec:svdcomp:benchmarks

    Following the R package svdComp5q0:

    1. Subtract a constant ``offset`` from the logit(q_x) schedules and take
       the SVD of the age × schedule matrix ``X = U Σ Vᵀ``.  The model keeps
       ``n_components`` scaled components ``u_i σ_i`` (one curve over age
       each) and the matching right singular vectors ``v_i`` (one weight
       per training schedule).
    2. Regress each weight ``v_i`` on eight terms in ₅q₀ and ₄₅q₁₅ plus an
       intercept (Clark 2019, Eq. 12; :data:`SVDCOMP_TERMS`), by OLS.
    3. For use with ₅q₀ alone, regress logit(₄₅q₁₅) on a cubic in
       logit(₅q₀) plus ₅q₀ (:data:`SVDCOMP_AML_TERMS`); its prediction
       stands in for the missing adult mortality.
    4. Regress logit(q₀) on a quadratic in logit(₅q₀); this replaces the
       age-0 value of the reconstruction (infant mortality is predicted
       directly, as in the package).

    The package additionally smooths components 2–4 over age; that is not
    done here (equivalent to ``predictLT(smooth = FALSE)``).

    Parameters
    ----------
    Zs : (n, A) logit(q_x) training schedules of ONE sex
    q5, q45 : (n,) their ₅q₀ and ₄₅q₁₅
    n_components : number of SVD components kept (4 in Clark 2019)
    offset : constant subtracted before the SVD (10 in the package)

    Returns
    -------
    dict with ``components`` (n_components, A), ``offset``, ``v_coef``
    (len(SVDCOMP_TERMS), n_components), ``aml_coef`` and ``q0_coef``.
    """
    X = (Zs - offset).T                                        # (A, n)
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    components = (U[:, :n_components] * sigma[:n_components]).T
    design = _svdcomp_design(q5, q45)
    return {"components": components, "offset": float(offset),
            "v_coef": _ols(design, SVDCOMP_TERMS, Vt[:n_components].T),
            "aml_coef": _ols(design, SVDCOMP_AML_TERMS, lifetable.logit(q45)),
            "q0_coef": _ols(design, SVDCOMP_Q0_TERMS, Zs[:, 0])}


def predict_svdcomp_classic(coefs: dict[str, Any], q5: np.ndarray,
                            q45: np.ndarray | None = None) -> np.ndarray:
    """Predicted logit(q_x) schedules (n, A) of one sex from SVD-Comp.

    Paper: sec:svdcomp:benchmarks

    With ``q45 = None`` (the one-parameter model) adult mortality is first
    predicted from ₅q₀ with the ``aml`` regression.
    """
    q5 = np.asarray(q5, dtype=float)
    design = _svdcomp_design(q5, None)
    if q45 is None:
        aml = np.column_stack([design[t] for t in SVDCOMP_AML_TERMS]) @ coefs["aml_coef"]
        q45 = lifetable.expit(aml)
    design = _svdcomp_design(q5, np.asarray(q45, dtype=float))
    v = np.column_stack([design[t] for t in SVDCOMP_TERMS]) @ coefs["v_coef"]   # (n, k)
    Z = v @ coefs["components"] + coefs["offset"]
    Z[:, 0] = np.column_stack([design[t] for t in SVDCOMP_Q0_TERMS]) @ coefs["q0_coef"]
    return Z
