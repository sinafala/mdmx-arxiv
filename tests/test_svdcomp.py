"""Tests for mdmx.svdcomp (MDMx summary-indicator models) and mdmx.logquad."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from mdmx import artifacts, lifetable, logquad, svdcomp

A = 110


def _orthonormal(rows, cols, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(rows, cols)))
    return q


def test_reconstruction_matrix_matches_bilinear_form():
    S = _orthonormal(2, 2, 0)
    Ab = _orthonormal(A, 8, 1)
    c = 5
    R = svdcomp.reconstruction_matrix(S, Ab, c)
    G = np.random.default_rng(2).normal(size=(2, c))
    # vec with sex varying before age, on both sides (paper Eq. svdcomp:recon)
    np.testing.assert_allclose(R @ G.ravel(), (S @ G @ Ab[:, :c].T).ravel(), atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(2 * c), atol=1e-12)


def test_loss_weights_follow_the_paper():
    w = svdcomp.loss_weights(A, alpha=10.0, young_ages=5).reshape(2, A)
    np.testing.assert_allclose(w[:, 5:], 1 / (2 * A))
    np.testing.assert_allclose(w[:, :5], 1 / (2 * A) + 10.0 / 10)
    one = svdcomp.loss_weights(A, alpha=10.0, young_ages=5, sexes=(1,)).reshape(2, A)
    assert np.all(one[0] == 0)
    np.testing.assert_allclose(one[1, 5:], 1 / A)


def test_truncation_oracle_with_equal_weights_is_the_projection():
    S, Ab = _orthonormal(2, 2, 3), _orthonormal(A, 10, 4)
    Z = np.random.default_rng(5).normal(size=(7, 2 * A))
    R = svdcomp.reconstruction_matrix(S, Ab, 4)
    oracle = svdcomp.truncation_oracle(Z, S, Ab, 4, np.ones(2 * A))
    np.testing.assert_allclose(oracle, Z @ R @ R.T, atol=1e-10)


SETTINGS = {"c_age": 6, "alpha": 1.0, "young_ages": 5, "hidden": [16, 16], "lr": 1e-2,
            "max_epochs": 200, "batch_size": 64, "l2": 0.0, "val_frac": 0.2, "patience": 20,
            "enforce_tol": 1e-10, "enforce_max_iter": 50}


def _poly_basis(c):
    ages = np.arange(A, dtype=float)
    Ab, _ = np.linalg.qr(np.column_stack([(ages / A) ** k for k in range(c)]))
    return Ab


def _synthetic_schedules(n, seed):
    """Smooth schedules whose truncated core is a nonlinear function of ₅q₀."""
    rng = np.random.default_rng(seed)
    ages = np.arange(A)
    base = -7.0 + 0.09 * ages - 3.0 * np.exp(-ages / 2.0)
    level = rng.uniform(-1.5, 1.5, size=n)
    shape = np.tanh(level)[:, None] * np.exp(-ages / 20.0)
    Zf = base + level[:, None] * (1 - ages / 150.0) + shape
    Zm = Zf + 0.3
    return np.hstack([Zf, Zm])


def test_core_model_learns_and_is_storable():
    Z = _synthetic_schedules(400, 6)
    ind = svdcomp.indicators(Z, A)
    inputs = ["5q0_F", "5q0_M"]
    X = svdcomp.features(ind, inputs)
    Ab = _poly_basis(6)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    groups = np.repeat(np.arange(40), 10)
    model = svdcomp.fit_core_model(X, Z, inputs=inputs, U_sex=S, U_age=Ab, settings=SETTINGS,
                                   seed=1, groups=groups)
    out = svdcomp.predict_with_diagnostics(model, X)
    # The inputs determine the schedules exactly, so the network should get
    # close to the best truncated representation under its own loss weights.
    oracle = svdcomp.truncation_oracle(Z, S, Ab, 6, svdcomp.loss_weights(A, 1.0, 5))
    assert np.sqrt(np.mean((out["Z_raw"] - oracle) ** 2)) < 0.02
    # After the constraint step the predictions reproduce the inputs exactly.
    assert out["converged"].all()
    Z_hat = svdcomp.predict_schedules(model, X)
    np.testing.assert_allclose(svdcomp.indicators(Z_hat, A)[inputs], ind[inputs], rtol=1e-9)
    # The model dict is plain data that the artefact store can hold.
    assert len(artifacts.content_hash(model)) == 64
    assert model["U_age"].shape == (A, 6) and model["inputs"] == inputs


def test_population_families_and_folds():
    register = pd.DataFrame({"code": ["GBR_NP", "GBRTENW", "GBR_SCO", "SWE", "NOR"],
                             "aggregate_of": [None, "GBR_NP", "GBR_NP", None, None]})
    fam = svdcomp.population_families(["GBRTENW", "SWE", "GBR_SCO", "NOR", "GBR_NP"], register)
    assert list(fam) == ["GBR_NP", "SWE", "GBR_NP", "NOR", "GBR_NP"]
    nested = pd.concat([register, pd.DataFrame({"code": ["GBRCENW"], "aggregate_of": ["GBRTENW"]})])
    assert list(svdcomp.population_families(["GBRCENW", "SWE"], nested)) == ["GBR_NP", "SWE"]
    cycle = pd.DataFrame({"code": ["A", "B"], "aggregate_of": ["B", "A"]})
    with pytest.raises(ValueError, match="cycle"):
        svdcomp.population_families(["A"], cycle)
    rng = np.random.default_rng(7)
    groups = rng.integers(0, 30, size=1000)
    folds = svdcomp.assign_folds(groups, 5, seed=3)
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1          # never split
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= np.bincount(groups).max()
    np.testing.assert_array_equal(folds, svdcomp.assign_folds(groups, 5, seed=3))


def test_cross_validate_fills_every_row_from_other_folds():
    folds = np.repeat(np.arange(4), 5)

    def fit_predict(k, train, test):
        assert not np.any(train & test)
        return np.full((test.sum(), 2), float(k))

    out = svdcomp.cross_validate(folds, fit_predict, 2)
    np.testing.assert_array_equal(out[:, 0], folds)


def test_classic_svdcomp_recovers_its_own_model():
    """Schedules generated exactly by an SVD-Comp model are predicted exactly."""
    rng = np.random.default_rng(8)
    n = 300
    q5 = np.exp(rng.uniform(np.log(0.003), np.log(0.3), n))
    q45 = lifetable.expit(lifetable.logit(0.6 * q5 ** 0.3) + rng.normal(0, 0.2, n))
    design = svdcomp._svdcomp_design(q5, q45)
    terms = np.column_stack([design[t] for t in svdcomp.SVDCOMP_TERMS])
    true_coef = rng.normal(size=(terms.shape[1], 4)) * 0.01
    comps = _orthonormal(A, 4, 9).T * np.array([[1000.0], [300.0], [90.0], [80.0]])
    Zs = (terms @ true_coef) @ comps + 10.0                  # rank 4 after the offset
    coefs = svdcomp.fit_svdcomp_classic(Zs, q5, q45, n_components=4, offset=10.0)
    pred = svdcomp.predict_svdcomp_classic(coefs, q5, q45)
    # The right singular vectors are linear combinations of the weight
    # columns, which are linear in the regression terms, so ages ≥ 1 are
    # reproduced exactly; age 0 is the separate quadratic q0 regression.
    np.testing.assert_allclose(pred[:, 1:], Zs[:, 1:], atol=1e-8)
    q0_terms = np.column_stack([design[t] for t in svdcomp.SVDCOMP_Q0_TERMS])
    q0_fit = q0_terms @ np.linalg.lstsq(q0_terms, Zs[:, 0], rcond=None)[0]
    np.testing.assert_allclose(pred[:, 0], q0_fit, atol=1e-8)


def test_logquad_matches_inputs_and_recovers_generating_model():
    rng = np.random.default_rng(10)
    starts = np.array(logquad.abridged_starts(A))
    G = len(starts)
    x = starts / 100.0
    true = {"starts": starts, "a": -7.0 + 7.0 * x ** 1.5, "b": 0.8 * np.exp(-3 * x),
            "c": 0.02 * np.exp(-3 * x), "a0": 0.3}
    v = np.where((starts >= 5) & (starts < 90), np.exp(-((starts - 35) / 25.0) ** 2), 0.0)
    true["v"] = v / np.linalg.norm(v)
    for coef in "abcv":
        true[coef][1] = 0.0                                   # the 1–4 group is not modelled
    n = 400
    q5 = np.exp(rng.uniform(np.log(0.004), np.log(0.25), n))
    k = rng.normal(0, 0.8, n)
    q = logquad.logquad_schedule(true, q5, k, A)
    coefs = logquad.fit_logquad(q, v_ages=(5, 90))
    # ₅q₀ (one-parameter) and ₄₅q₁₅ (two-parameter) are reproduced exactly.
    q_one, _ = logquad.predict_logquad(coefs, q5, None, A)
    np.testing.assert_allclose(lifetable.nqx(q_one, 0, 5), q5, rtol=1e-10)
    q45 = lifetable.nqx(q, 15, 45)
    q_two, k_hat = logquad.predict_logquad(coefs, q5, q45, A)
    np.testing.assert_allclose(lifetable.nqx(q_two, 15, 45), q45, rtol=1e-8)
    # Data generated by the model are fitted (almost) exactly: the only
    # approximation is the round trip between group and single-year rates.
    assert np.max(np.abs(lifetable.logit(q_two) - lifetable.logit(q))) < 0.02
    assert G == 23


def test_indicator_constraints_parse_names():
    assert svdcomp.indicator_constraints(["5q0_F", "45q15_M"]) == [(0, 0, 5), (1, 15, 45)]
    with pytest.raises(KeyError):
        svdcomp.indicator_constraints(["5q1_F"])


def _enforcement_problem(constraints, seed):
    """Truncated schedules perturbed so that they miss their targets."""
    Z = _synthetic_schedules(50, seed)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    R = svdcomp.reconstruction_matrix(S, _poly_basis(8), 8)
    g_true = Z @ R                                       # cores whose schedules give the targets
    Zt = g_true @ R.T
    targets = np.column_stack([lifetable.nqx(lifetable.expit(Zt[:, s * A:(s + 1) * A]), a, w)
                               for s, a, w in constraints])
    g0 = g_true + np.random.default_rng(seed).normal(0, 0.3, size=g_true.shape)
    return R, g0, targets


@pytest.mark.parametrize("constraints", [[(0, 0, 5), (1, 0, 5)],
                                         [(0, 0, 5), (1, 0, 5), (0, 15, 45), (1, 15, 45)]])
def test_enforcement_is_exact_and_minimal(constraints):
    R, g0, targets = _enforcement_problem(constraints, 12)
    res = svdcomp.enforce_indicators(g0, R, constraints, targets, tol=1e-10, max_iter=50)
    assert res["converged"].all()
    Zs = res["cores"] @ R.T
    for j, (s, a, w) in enumerate(constraints):
        got = lifetable.nqx(lifetable.expit(Zs[:, s * A:(s + 1) * A]), a, w)
        np.testing.assert_allclose(got, targets[:, j], rtol=1e-8)
    # The change is close to the exactly smallest change that meets the
    # constraints (found here by a general-purpose constrained optimiser):
    # never smaller, and only slightly larger.
    tl = lifetable.logit(targets)
    for i in range(5):
        def gap(g, i=i):
            return svdcomp.constraint_values((g @ R.T)[None], constraints, A)[0][0] - tl[i]
        exact = minimize(lambda g, i=i: np.sum((g - g0[i]) ** 2), res["cores"][i], method="SLSQP",
                         constraints={"type": "eq", "fun": gap}, options={"ftol": 1e-14, "maxiter": 500})
        assert exact.success
        ours, best = np.linalg.norm(res["cores"][i] - g0[i]), np.linalg.norm(exact.x - g0[i])
        assert best * (1 - 1e-6) <= ours <= 1.05 * best


def test_enforcement_for_one_sex_leaves_the_other_unchanged():
    constraints = [(0, 0, 5)]
    R, g0, targets = _enforcement_problem(constraints, 13)
    res = svdcomp.enforce_indicators(g0, R, constraints, targets, tol=1e-10, max_iter=50)
    before, after = g0 @ R.T, res["cores"] @ R.T
    np.testing.assert_allclose(after[:, A:], before[:, A:], atol=1e-10)
    assert np.abs(after[:, :A] - before[:, :A]).max() > 1e-3


def test_nested_cross_validation_chooses_from_the_grid():
    Z = _synthetic_schedules(160, 14)
    ind = svdcomp.indicators(Z, A)
    inputs = ["5q0_F", "5q0_M"]
    X = svdcomp.features(ind, inputs)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    groups = np.repeat(np.arange(16), 10)
    folds = svdcomp.assign_folds(groups, 4, seed=1)
    settings = {**SETTINGS, "hidden": [8], "max_epochs": 5, "patience": 3}
    grid = [{"c_age": 2, "alpha": 0.0}, {"c_age": 6, "alpha": 0.0}]
    res = svdcomp.nested_cross_validate_core_model(
        X, Z, inputs=inputs, folds=folds, groups=groups, settings=settings, grid=grid,
        n_inner=2, inner_seed=3, seed=5, U_sex=S, U_age=_poly_basis(6))
    assert len(res["inner"]) == 4 * len(grid)
    assert res["inner"].groupby("fold")["chosen"].sum().eq(1).all()
    for k, s in res["fold_settings"].items():
        chosen = res["inner"][(res["inner"].fold == k) & res["inner"].chosen].iloc[0]
        assert s["c_age"] == chosen["c_age"]
    assert np.all(np.isfinite(res["Z_hat"])) and res["runs"]["not_converged"].sum() == 0
    np.testing.assert_allclose(svdcomp.indicators(res["Z_hat"], A)[inputs], ind[inputs], rtol=1e-9)


def test_nested_cross_validation_estimates_bases_from_training_rows_only():
    Z = _synthetic_schedules(160, 14)
    ind = svdcomp.indicators(Z, A)
    inputs = ["5q0_F", "5q0_M"]
    X = svdcomp.features(ind, inputs)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Ab = _poly_basis(6)
    groups = np.repeat(np.arange(16), 10)
    folds = svdcomp.assign_folds(groups, 4, seed=1)
    settings = {**SETTINGS, "hidden": [8], "max_epochs": 5, "patience": 3}
    grid = [{"c_age": 2, "alpha": 0.0}, {"c_age": 6, "alpha": 0.0}]
    masks = []

    def basis(train):
        assert train.shape == (len(folds),) and train.dtype == bool
        masks.append(train.copy())
        return S, Ab

    kw = dict(inputs=inputs, folds=folds, groups=groups, settings=settings, grid=grid,
              n_inner=2, inner_seed=3, seed=5)
    fixed = svdcomp.nested_cross_validate_core_model(X, Z, U_sex=S, U_age=Ab, **kw)
    fold_basis = svdcomp.nested_cross_validate_core_model(X, Z, basis=basis, **kw)
    # The same bases through the function give the same predictions.
    np.testing.assert_array_equal(fixed["Z_hat"], fold_basis["Z_hat"])
    # One call per inner fit (outer folds × inner folds × candidates) and per outer fit.
    assert len(masks) == 4 * 2 * len(grid) + 4
    # No basis is ever estimated from rows of the outer test fold it serves:
    # every mask leaves out at least one whole outer fold.
    for m in masks:
        assert any(not m[folds == k].any() for k in range(4))
    with pytest.raises(ValueError, match="either fixed bases"):
        svdcomp.nested_cross_validate_core_model(X, Z, U_sex=S, U_age=Ab, basis=basis, **kw)


def test_own_sex_cross_validation_accepts_a_basis_function():
    Z = _synthetic_schedules(120, 15)
    ind = svdcomp.indicators(Z, A)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Ab = _poly_basis(6)
    groups = np.repeat(np.arange(12), 10)
    folds = svdcomp.assign_folds(groups, 3, seed=2)
    settings = {**SETTINGS, "hidden": [8], "max_epochs": 5, "patience": 3}
    kw = dict(indicator_names=["5q0"], folds=folds, groups=groups, settings=settings, seed=4)
    fixed = svdcomp.cross_validate_own_sex(ind, Z, U_sex=S, U_age=Ab, **kw)
    by_fn = svdcomp.cross_validate_own_sex(ind, Z, basis=lambda train: (S, Ab), **kw)
    np.testing.assert_array_equal(fixed["Z_hat"], by_fn["Z_hat"])


def test_abridged_logit_q_agrees_with_nqx():
    Z = _synthetic_schedules(5, 11)
    ab = svdcomp.abridged_logit_q(Z, A, logquad.abridged_starts(A))
    q = lifetable.expit(Z[:, :A])
    np.testing.assert_allclose(lifetable.expit(ab[:, 0, 2]), lifetable.nqx(q, 5, 5))


@pytest.mark.parametrize("bad", [0, 7])
def test_reconstruction_matrix_rejects_bad_truncation(bad):
    with pytest.raises(ValueError):
        svdcomp.reconstruction_matrix(np.eye(2), _orthonormal(A, 6, 0), bad)
