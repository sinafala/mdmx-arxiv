"""Unit tests for mdmx.forecast (book chapter 11, paper §11).

They use small synthetic inputs only, so they run without the HMD data.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from mdmx import forecast as fc
from mdmx import lifetable
from mdmx.config import params

# ══════════════════════════════════════════════════════════════════════════════
# The Kalman filter
# ══════════════════════════════════════════════════════════════════════════════

def _matrix_filter(y, rho, q_level, q_drift, r, target, level0, p0_level, p0_drift):
    """Textbook Kalman filter in matrix form (paper §11.2), for comparison.

    State x = [levels (p), drifts (p)], F = [[I, I], [0, ρI]],
    b = [0, (1 − ρ)δ*], H = [I, 0]; rows of y that are NaN skip the update.
    """
    p = y.shape[1]
    eye, zero = np.eye(p), np.zeros((p, p))
    F = np.block([[eye, eye], [zero, rho * eye]])
    H = np.hstack([eye, zero])
    Q = np.diag(np.concatenate([q_level, q_drift]))
    R = np.diag(r)
    b = np.concatenate([np.zeros(p), (1 - rho) * target])
    x = np.concatenate([level0, target])
    P = np.diag(np.concatenate([np.full(p, p0_level), p0_drift]))
    loglik = 0.0
    for t in range(y.shape[0]):
        x = F @ x + b
        P = F @ P @ F.T + Q
        if np.isnan(y[t]).any():
            continue
        v = y[t] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ v
        P = (np.eye(2 * p) - K @ H) @ P
        loglik += -0.5 * (p * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + v @ np.linalg.solve(S, v))
    return loglik, x, P


@pytest.fixture
def llt_case():
    rng = np.random.default_rng(1)
    T_, p = 40, 3
    y = np.cumsum(rng.normal(0.3, 0.5, size=(T_, p)), axis=0)
    y[[5, 6, 7, 20]] = np.nan                       # gaps: missing / exceptional years
    pars = dict(rho=0.9, q_level=np.array([0.2, 0.1, 0.05]), q_drift=np.array([0.01, 0.02, 0.005]),
                r=np.array([0.1, 0.3, 0.2]), target=np.array([0.3, 0.2, 0.1]),
                level0=y[0], p0_level=1.0, p0_drift=np.array([0.04, 0.03, 0.02]))
    return y, pars


def test_filter_matches_matrix_form_with_gaps(llt_case):
    """The component-wise filter equals the matrix filter, gaps included."""
    y, k = llt_case
    ll_m, x_m, P_m = _matrix_filter(y, **k)
    out = fc.llt_filter(y, np.array([k["rho"]]), k["q_level"][None], k["q_drift"][None], k["r"][None],
                        k["target"], k["level0"], k["p0_level"], k["p0_drift"])
    p = y.shape[1]
    assert out["loglik"][0] == pytest.approx(ll_m, rel=1e-10)
    np.testing.assert_allclose(out["level"][0], x_m[:p], rtol=1e-10)
    np.testing.assert_allclose(out["drift"][0], x_m[p:], rtol=1e-10)
    np.testing.assert_allclose(out["P_ll"][0], np.diag(P_m)[:p], rtol=1e-10)
    np.testing.assert_allclose(out["P_ld"][0], np.diag(P_m[:p, p:]), rtol=1e-10)
    np.testing.assert_allclose(out["P_dd"][0], np.diag(P_m)[p:], rtol=1e-10)
    assert out["n_updates"] == np.isfinite(y[:, 0]).sum()


def test_filter_skips_update_for_nan_year(llt_case):
    """A NaN year adds nothing to the log-likelihood and is a pure prediction step."""
    y, k = llt_case
    args = (np.array([k["rho"]]), k["q_level"][None], k["q_drift"][None], k["r"][None],
            k["target"], k["level0"], k["p0_level"], k["p0_drift"])
    head = fc.llt_filter(y[:10], *args)                 # ends with an observed year
    with_gap = fc.llt_filter(np.vstack([y[:10], np.full((1, y.shape[1]), np.nan)]), *args)
    assert with_gap["loglik"][0] == pytest.approx(head["loglik"][0], rel=1e-14)
    state = {key: head[key][0] for key in ("level", "drift", "P_ll", "P_ld", "P_dd")}
    mean, var = fc.llt_forecast(state, k["rho"], k["q_level"], k["q_drift"], k["r"], k["target"], 1)
    np.testing.assert_allclose(with_gap["level"][0], mean[0])
    np.testing.assert_allclose(with_gap["P_ll"][0] + k["r"], var[0])


def test_filter_rejects_partly_missing_rows(llt_case):
    y, k = llt_case
    y = y.copy()
    y[3, 0] = np.nan                                     # one component only: not allowed
    with pytest.raises(ValueError, match="fully observed or fully missing"):
        fc.llt_filter(y, np.array([k["rho"]]), k["q_level"][None], k["q_drift"][None], k["r"][None],
                      k["target"], k["level0"], k["p0_level"], k["p0_drift"])


def test_forecast_drift_reverts_to_target():
    """With ρ < 1 the forecast drift converges to the target δ* (paper §11.2)."""
    state = {"level": np.array([0.0]), "drift": np.array([1.0]), "P_ll": np.array([0.0]),
             "P_ld": np.array([0.0]), "P_dd": np.array([0.0])}
    mean, var = fc.llt_forecast(state, 0.8, np.array([0.1]), np.array([0.01]), np.array([0.05]),
                                np.array([0.2]), 200)
    assert np.diff(mean[:, 0])[-1] == pytest.approx(0.2, abs=1e-8)
    assert np.all(np.diff(var[:, 0]) > 0)                # uncertainty grows with the horizon


# ══════════════════════════════════════════════════════════════════════════════
# Maximum-likelihood fit of the filter
# ══════════════════════════════════════════════════════════════════════════════

def _settings(**overrides):
    """The forecaster's settings from the configuration (age rank and N_PC are not used here)."""
    P = params()
    return fc.MDMxSettings.from_params(P.section("forecast"), P.section("tucker"), age_rank=10, **overrides)


def _simulate_llt(seed, T_, rho, q_level, q_drift, r, target):
    """Scores from the damped local linear trend of paper §11.2 (drift starting at the target)."""
    rng = np.random.default_rng(seed)
    level, drift = np.zeros(len(target)), np.array(target, dtype=float)
    y = np.empty((T_, len(target)))
    for t in range(T_):
        level = level + drift + rng.normal(0.0, np.sqrt(q_level))
        drift = rho * drift + (1.0 - rho) * target + rng.normal(0.0, np.sqrt(q_drift))
        y[t] = level + rng.normal(0.0, np.sqrt(r))
    return y


def _loglik(theta, y, fit, target, p0_drift, settings):
    """Log-likelihood at θ = (σ^ℓ, σ^δ, σ^r, ρ), with the fit's initial state."""
    rho, q_level, q_drift, r = fc._unpack(np.asarray(theta), y.shape[1])
    return fc.llt_filter(y, rho, q_level, q_drift, r, target, fit["level0"], settings.p0_level_var,
                         p0_drift)["loglik"][0]


def _theta(fit):
    return np.concatenate([np.sqrt(fit["q_level"]), np.sqrt(fit["q_drift"]), np.sqrt(fit["r"]),
                           [fit["rho"]]])


@pytest.fixture
def interior_case():
    target = np.array([0.1, -0.05])
    y = _simulate_llt(11, 150, 0.95, np.array([0.05, 0.02]), np.array([0.002, 0.001]),
                      np.array([0.1, 0.05]), target)
    return y, target, np.array([0.001, 0.001])


def test_unpack_maps_standard_deviations_and_rho():
    theta = np.array([[1.0, 2.0, 0.1, 0.2, 3.0, 4.0, 0.9]])
    rho, q_level, q_drift, r = fc._unpack(theta, 2)
    assert rho[0] == 0.9
    np.testing.assert_array_equal(q_level[0], [1.0, 4.0])
    np.testing.assert_allclose(q_drift[0], [0.01, 0.04])
    np.testing.assert_array_equal(r[0], [9.0, 16.0])


def test_fit_reaches_the_maximum(interior_case):
    """The fit is a maximum: no feasible small move, nor another optimiser, raises log L."""
    y, target, p0_drift = interior_case
    s = _settings()
    fit = fc.fit_llt(y, target, p0_drift, s)
    assert fit["converged"]
    theta = _theta(fit)
    ll = _loglik(theta, y, fit, target, p0_drift, s)
    assert ll == pytest.approx(fit["loglik"], abs=1e-9)
    truth = np.concatenate([np.sqrt([0.05, 0.02]), np.sqrt([0.002, 0.001]), np.sqrt([0.1, 0.05]), [0.95]])
    assert ll >= _loglik(truth, y, fit, target, p0_drift, s)
    lower = np.r_[np.full(6, np.exp(s.log_var_bounds[0] / 2)), s.rho_bounds[0]]
    upper = np.r_[np.full(6, np.exp(s.log_var_bounds[1] / 2)), s.rho_bounds[1]]
    for i in range(theta.size):
        for step in (-1e-3, 1e-3):
            moved = theta.copy()
            moved[i] = np.clip(theta[i] + step * max(abs(theta[i]), 1e-3), lower[i], upper[i])
            assert _loglik(moved, y, fit, target, p0_drift, s) <= ll + 1e-7
    nm = minimize(lambda th: -_loglik(th, y, fit, target, p0_drift, s), theta, method="Nelder-Mead",
                  bounds=list(zip(lower, upper, strict=True)),
                  options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert -nm.fun <= ll + 1e-6


def test_fit_reaches_maxima_on_the_bounds():
    """A maximum on the bounds is reached exactly, and it is a maximum (moving inward lowers log L).

    Data without drift innovations (q^δ = 0); for this seed the maximum has
    ρ at its lower bound and the first drift variance at the floor — the
    situation that a logistic ρ and log variances could only approach.
    """
    target = np.array([0.1, -0.05])
    p0_drift = np.array([0.01, 0.01])
    y = _simulate_llt(5, 200, 0.3, np.array([0.02, 0.01]), np.array([0.0, 0.0]),
                      np.array([0.1, 0.05]), target)
    s = _settings()
    fit = fc.fit_llt(y, target, p0_drift, s)
    assert fit["converged"]
    assert fit["rho"] == s.rho_bounds[0]
    assert fit["q_drift"][0] == pytest.approx(np.exp(s.log_var_bounds[0]), rel=1e-12)
    theta = _theta(fit)
    ll = _loglik(theta, y, fit, target, p0_drift, s)
    for i in (2, theta.size - 1):                        # σ^δ₁ and ρ, both at their lower bound
        moved = theta.copy()
        moved[i] += 1e-3 * max(abs(theta[i]), 1e-3)
        assert _loglik(moved, y, fit, target, p0_drift, s) < ll


def test_fit_is_stable_to_the_last_bit(interior_case):
    """A 1-ulp change of the data moves the fitted forecast by far less than any reported digit."""
    y, target, p0_drift = interior_case
    s = _settings()
    y_pert = y * (1 + np.finfo(float).eps * np.random.default_rng(2).choice([-1.0, 1.0], size=y.shape))
    means = []
    for data in (y, y_pert):
        fit = fc.fit_llt(data, target, p0_drift, s)
        mean, _ = fc.llt_forecast(fit["state"], fit["rho"], fit["q_level"], fit["q_drift"], fit["r"],
                                  target, s.horizon)
        means.append(mean)
    np.testing.assert_allclose(means[0], means[1], rtol=0, atol=1e-6)


def test_fit_keeps_the_best_start(interior_case):
    """Every start is reported, and the fit is the best of them."""
    y, target, p0_drift = interior_case
    s = _settings()
    fit = fc.fit_llt(y, target, p0_drift, s)
    assert len(fit["start_logliks"]) == s.mle_grid_top + 1 + len(s.mle_rho_starts)
    assert fit["start_kinds"].count("profile") == s.mle_grid_top
    assert fit["start_kinds"][fit["moment_start"]] == "moment"
    assert fit["loglik"] == np.max(fit["start_logliks"])
    assert fit["start_logliks"][fit["best_start"]] == fit["loglik"]
    assert fit["n_nonfinite_starts"] == 0
    assert fit["rho_profile"].shape == (len(s.mle_grid_rho),)
    # A profile value maximises over the σ at one ρ, so it cannot exceed the free maximum.
    assert np.all(fit["rho_profile"] <= fit["loglik"] + 1e-8)


def test_profile_is_the_maximum_at_fixed_rho(interior_case):
    """Each profile value equals an independent maximisation over the σ with ρ held fixed.

    The check uses the separability of the likelihood at fixed ρ (each
    component has its own three σ) and Powell's method from several starts,
    sharing nothing with the profile code but the filter.
    """
    y, target, p0_drift = interior_case
    s = _settings()
    fit = fc.fit_llt(y, target, p0_drift, s)
    assert fit["profile_converged"]
    sd_lo, sd_hi = np.exp(np.asarray(s.log_var_bounds) / 2)

    def component_loglik(sig, j, rho):
        out = fc.llt_filter(y[:, [j]], np.array([rho]), np.array([[sig[0] ** 2]]), np.array([[sig[1] ** 2]]),
                            np.array([[sig[2] ** 2]]), target[[j]], fit["level0"][[j]], s.p0_level_var,
                            p0_drift[[j]])
        return out["loglik"][0]

    for g in (0, 6, 11):
        rho = s.mle_grid_rho[g]
        total = 0.0
        for j in range(y.shape[1]):
            best = -np.inf
            for start in (0.01, 0.1, 0.5):
                res = minimize(lambda sig, j=j, rho=rho: -component_loglik(sig, j, rho), np.full(3, start),
                               method="Powell", bounds=[(sd_lo, sd_hi)] * 3,
                               options={"xtol": 1e-10, "ftol": 1e-13, "maxfev": 20000})
                best = max(best, -res.fun)
            total += best
        assert fit["rho_profile"][g] == pytest.approx(total, abs=1e-5)


def test_profile_starts_find_a_maximum_the_moment_starts_miss():
    """A likelihood with two maxima: every start from the moment-based starting values (ρ anywhere) ends
    at the lower one; the profile starts find the higher one, as for NLD 1960 (last 50 years)."""
    target = np.array([0.1])
    y = _simulate_llt(7, 100, 0.99, np.array([0.02]), np.array([5e-5]), np.array([0.05]), target)
    s = _settings()
    fit = fc.fit_llt(y, target, np.array([0.001]), s)
    kinds = np.array(fit["start_kinds"])
    assert fit["converged"]
    assert fit["start_kinds"][fit["best_start"]] == "profile"
    assert fit["loglik"] - fit["start_logliks"][kinds == "moment"].max() > 0.1
    assert fit["loglik"] - fit["start_logliks"][fit["moment_start"]] > 0.1


def test_fit_stopped_by_the_iteration_limit_is_not_converged(interior_case):
    y, target, p0_drift = interior_case
    fit = fc.fit_llt(y, target, p0_drift, _settings(mle_maxiter=2))
    assert not fit["converged"]


def test_settings_reject_starts_outside_the_bounds():
    with pytest.raises(ValueError, match="mle_rho_starts"):
        _settings(mle_rho_starts=(0.5,))
    with pytest.raises(ValueError, match="mle_grid_top"):
        _settings(mle_grid_top=0)
    with pytest.raises(ValueError, match="mle_grid_sd_log10"):
        _settings(mle_grid_sd_log10=(0.5, -3.0))


def test_parallel_fits_equal_serial_fits(interior_case):
    """Worker processes change only the run time: every fit is identical to the last bit."""
    y, target, p0_drift = interior_case
    s = _settings()
    series = [(y[:60], target), (y, target), (y[40:], -target)]
    serial = fc.fit_llt_batch(series, p0_drift, s, workers=1)
    parallel = fc.fit_llt_batch(series, p0_drift, s, workers=2)
    for a, b in zip(serial, parallel, strict=True):
        assert a["loglik"] == b["loglik"]
        np.testing.assert_array_equal(_theta(a), _theta(b))
        np.testing.assert_array_equal(a["state"]["level"], b["state"]["level"])
    assert serial[1]["loglik"] == fc.fit_llt(y, target, p0_drift, s)["loglik"]
    with pytest.raises(ValueError, match="workers"):
        fc.fit_llt_batch(series, p0_drift, s, workers=-1)


def _code_lists():
    """Every ``*_CODE`` tuple of the package: (qualified name, tuple)."""
    import importlib
    import pkgutil

    import mdmx
    found = []
    for info in pkgutil.iter_modules(mdmx.__path__):
        module = importlib.import_module(f"mdmx.{info.name}")
        found += [(f"{module.__name__}.{name}", value) for name, value in vars(module).items()
                  if name.endswith("_CODE") and isinstance(value, tuple)]
    return sorted(found, key=lambda pair: pair[0])


def _loaded_names(code):
    """Global, attribute and free-variable names a code object loads, nested code included."""
    names = set(code.co_names) | set(code.co_freevars)
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= _loaded_names(const)
    return names


def _bytecode_callees(func):
    """An oracle independent of the package's ast-based search: the mdmx functions that ``func``'s
    bytecode can load, as a global name or as an attribute of an mdmx module it loads."""
    names = _loaded_names(func.__code__)
    values = [func.__globals__[n] for n in names if n in func.__globals__]
    modules = [v for v in values if inspect.ismodule(v) and v.__name__.startswith("mdmx")]
    values += [vars(m)[n] for m in modules for n in names if n in vars(m)]
    return {inspect.unwrap(v) for v in values
            if inspect.isfunction(inspect.unwrap(v)) and inspect.unwrap(v).__module__.startswith("mdmx.")}


def test_cache_code_lists_are_complete():
    """The code a key hashes for every ``*_CODE`` tuple includes every mdmx function that any of its
    functions calls, transitively, so that changing one changes the keys (a missing one would leave
    stale forecasts in the cache).  The listed tuples need not be complete themselves: the key
    follows the calls (:func:`mdmx.artifacts._code_closure`)."""
    from mdmx import artifacts
    lists = _code_lists()
    assert {name for name, _ in lists} >= {"mdmx.flowfield.FLOW_CODE", "mdmx.forecast.SPACE_CODE",
                                           "mdmx.forecast.MDMX_CODE", "mdmx.lilee.LILEE_CODE",
                                           "mdmx.variants.FLOW_VARIANT_CODE", "mdmx.variants.LILEE_RUN_CODE"}
    for name, listed in lists:
        items, _ = artifacts._code_closure(listed)
        functions = [o for o in items if inspect.isfunction(o)]
        assert {artifacts._unwrap(o) for o in listed if inspect.isfunction(artifacts._unwrap(o))} <= set(functions)
        missing = set().union(*(_bytecode_callees(f) for f in functions)) - set(functions)
        assert not missing, (name, sorted(f"{f.__module__}.{f.__qualname__}" for f in missing))


def test_cache_keys_cover_the_callees_review_b_found_missing():
    """The examples of review B (F5): code the listed functions call but the lists did not name."""
    from mdmx import (
        artifacts,
        bayeslife,
        clustering,
        disruption,
        fitting,
        flowfield,
        rbridge,
        svdcomp,
        trajectories,
    )

    def names(code):
        return {artifacts._qualified_name(o) for o in artifacts._code_closure(code)[0]}

    assert "mdmx.lifetable.e0_from_qx" in names(flowfield.FLOW_CODE)
    bl_schedules = names([bayeslife.schedules, bayeslife.points, fc.comparator_schedules,
                          lifetable.e0_from_logit, rbridge.run])          # evaluation.run_bl_schedules
    assert {"mdmx.lifetable.logit", "mdmx.lifetable.qx_from_mx"} <= bl_schedules
    ch09 = names([fitting, disruption, trajectories, clustering, lifetable, svdcomp.assign_folds,
                  svdcomp.population_families])                         # chapter 09's fitting_eval, less its cell
    assert {"mdmx.tensor.complete_cores", "mdmx.nn.fit_mlp", "mdmx.nn.validation_split"} <= ch09


# ══════════════════════════════════════════════════════════════════════════════
# e₀ gradient and the delta method
# ══════════════════════════════════════════════════════════════════════════════

def _schedule(A=110):
    ages = np.arange(A)
    q = np.clip(np.exp(-9.5 + 0.09 * ages) + 0.004 * np.exp(-ages), 1e-6, 0.9)   # Gompertz + infancy
    return lifetable.logit(q)


def test_e0_gradient_matches_finite_differences():
    z = np.stack([_schedule(), _schedule() + 0.2])         # (2, A): a "female" and a "male" schedule
    e0, grad = fc.e0_and_gradient(z)
    np.testing.assert_allclose(e0, lifetable.e0_from_logit(z), rtol=1e-12)
    eps = 1e-6
    for x in (0, 1, 2, 30, 80, 109):
        zp, zm = z.copy(), z.copy()
        zp[:, x] += eps
        zm[:, x] -= eps
        numeric = (lifetable.e0_from_logit(zp) - lifetable.e0_from_logit(zm)) / (2 * eps)
        np.testing.assert_allclose(grad[:, x], numeric, rtol=1e-5, atol=1e-10)
    assert np.all(grad < 0)                                 # higher mortality, lower e₀


def test_delta_method_sd_positive_and_matches_monte_carlo():
    """σ² = J Cov Jᵀ is positive and agrees with simulation for small variances."""
    rng = np.random.default_rng(3)
    A, p = 110, 4
    z_mean = np.stack([_schedule(A), _schedule(A) + 0.25])
    L = rng.normal(0, 0.05, size=(2, A, p))
    s = rng.normal(0, 1, size=p)
    var = np.full(p, 0.01)
    z = fc.scores_to_logit(s, z_mean, L)
    e0, grad = fc._e0_by_sex(z)
    J = np.einsum("ksa,saj->kj", grad, L)
    sd = np.sqrt(np.sum(J ** 2 * var, axis=-1))
    assert np.all(sd > 0)
    draws = s + np.sqrt(var) * rng.standard_normal((20000, p))
    e0_draws, _ = fc._e0_by_sex(fc.scores_to_logit(draws, z_mean, L))
    np.testing.assert_allclose(e0_draws.std(axis=0), sd, rtol=0.03)
    assert e0[2] == pytest.approx(0.5 * (e0[0] + e0[1]))


# ══════════════════════════════════════════════════════════════════════════════
# Drifts on calendar time
# ══════════════════════════════════════════════════════════════════════════════

def test_country_drift_uses_calendar_years():
    years = np.arange(1980, 2001)
    scores = np.full((1, len(years), 1), np.nan)
    keep = np.isin(years, [1981, 1985, 1990, 1991, 1999])   # irregular observed years
    scores[0, keep, 0] = 2.0 * (years[keep] - 1980)         # slope 2 per calendar year
    d = fc.country_drifts(scores, years, 2000, window=20, min_years=3)
    assert d[0, 0] == pytest.approx(2.0)
    assert np.isnan(fc.country_drifts(scores, years, 2000, window=20, min_years=6)[0, 0])


# ══════════════════════════════════════════════════════════════════════════════
# The benchmark harness
# ══════════════════════════════════════════════════════════════════════════════

def _truth():
    rows = [(pop, sex, year, 70.0 + 0.2 * (year - 2000))
            for pop in ("AAA", "BBB") for sex in fc.SEXES for year in range(1995, 2010)]
    return pd.DataFrame(rows, columns=["pop", "sex", "year", "e0_obs"])


def _jobs():
    return pd.DataFrame({"pop": ["AAA", "BBB", "AAA"], "origin": [2000, 2000, 2004],
                         "horizon": [3, 3, 3]})


def _perfect(origin, jobs):
    rows = [(pop, sex, origin + h, 70.0 + 0.2 * (origin + h - 2000))
            for pop in jobs["pop"] for sex in fc.SEXES for h in (1, 2, 3)]
    return pd.DataFrame(rows, columns=["pop", "sex", "year", "e0_fc"])


def _flawed(origin, jobs):
    out = _perfect(origin, jobs)
    out["e0_fc"] += 1.0
    # Misses one point entirely and returns NaN for another (one sex only).
    drop = (out["pop"] == "AAA") & (out["year"] == origin + 2) & (out["sex"] == "male")
    out.loc[(out["pop"] == "BBB") & (out["year"] == origin + 3) & (out["sex"] == "both"), "e0_fc"] = np.nan
    return out[~drop]


def test_benchmark_pairs_points():
    out = fc.run_benchmark({"good": _perfect, "bad": _flawed}, _jobs(), _truth())
    # Every expected point appears for every method: 3 jobs × 3 years × 3 sexes.
    assert out.groupby("method").size().to_dict() == {"bad": 27, "good": 27}
    assert set(out.columns) >= {"method", "pop", "sex", "origin", "year", "horizon", "e0_obs", "e0_fc",
                                "lower80", "upper80", "lower95", "upper95", "paired"}
    assert out["horizon"].between(1, 3).all()
    unpaired = out.loc[~out["paired"], ["pop", "origin", "year"]].drop_duplicates()
    expected = {("AAA", 2000, 2002), ("AAA", 2004, 2006), ("BBB", 2000, 2003)}
    assert set(map(tuple, unpaired.to_numpy())) == expected
    # Unpaired for *both* methods, and all three sexes, even though "good" produced them.
    assert (~out["paired"]).sum() == 2 * 3 * len(expected)
    paired = out[out["paired"]]
    s = fc.summarise(paired).set_index("method")
    assert s.loc["good", "MAE"] == pytest.approx(0.0)
    assert s.loc["bad", "Bias"] == pytest.approx(1.0)
    assert s.loc["good", "n"] == s.loc["bad", "n"]


def test_benchmark_rejects_bad_method_output():
    def broken(origin, jobs):
        return _perfect(origin, jobs).rename(columns={"e0_fc": "e0"})
    with pytest.raises(ValueError, match="e0_fc"):
        fc.run_benchmark({"broken": broken}, _jobs(), _truth())


def test_loo_calibration_excludes_own_origin():
    pts = pd.DataFrame({"origin": [1, 1, 2, 2, 3, 3], "sex": "both", "e0_fc": 0.0, "sd": 1.0,
                        "e0_obs": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]})
    out = fc.loo_calibrate(pts)
    # κ for origin 1 comes from origins 2 and 3 only: z = ±2, ±3 → SD = sqrt(6.5).
    assert out.loc[out["origin"] == 1, "kappa_loo"].iloc[0] == pytest.approx(np.sqrt(6.5))
    assert fc.coverage(out, 95, "_loo") == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Comparator inputs
# ══════════════════════════════════════════════════════════════════════════════

def test_comparator_data_fills_gaps_linearly():
    years = np.arange(2000, 2005)
    observed = np.array([[True, False, False, True, True]])
    rows = [("AAA", sex, y, a, 10.0 * (y - 1999) + a, 1000.0 + y - 2000)
            for sex in ("female", "male") for y in years for a in (0, 1, 2, 3)]
    de = pd.DataFrame(rows, columns=["pop", "sex", "year", "age", "deaths", "exposure"])
    de.loc[de["year"].isin([2001, 2002]), "deaths"] = 999.0     # unusable years: must be ignored
    out = fc.comparator_data(de, observed, ["AAA"], years, open_age=2)
    assert sorted(out["age"].unique()) == [0, 1, 2]
    f = out[(out["sex"] == "female") & (out["age"] == 0)].set_index("year")
    np.testing.assert_allclose(f["deaths"].to_numpy(), [10, 20, 30, 40, 50])   # 2001–02 interpolated
    assert f["filled"].tolist() == [False, True, True, False, False]
    open_group = out[(out["sex"] == "male") & (out["age"] == 2) & (out["year"] == 2000)]
    assert open_group["deaths"].iloc[0] == pytest.approx((10 + 2) + (10 + 3))   # ages 2 and 3 summed


# ══════════════════════════════════════════════════════════════════════════════
# The jump-off anchor (decision C revised) and within-era calibration
# ══════════════════════════════════════════════════════════════════════════════

def test_anchor_weights_are_zero_one_or_a_half_life_decay():
    h = np.arange(0, 7)
    np.testing.assert_array_equal(fc.anchor_weights(h, "none"), 0.0)
    np.testing.assert_array_equal(fc.anchor_weights(h, "persistent"), 1.0)
    np.testing.assert_allclose(fc.anchor_weights(h, 2.0), 0.5 ** (h / 2))
    for bad in ("decaying", 0.0, -1.0):
        with pytest.raises(ValueError):
            fc.anchor_weights(h, bad)


def test_anchor_smoother_keeps_age_zero_constants_and_lines():
    A = 110
    K = fc.anchor_smoother(A, 1e3)
    x = np.random.default_rng(3).normal(size=A)
    y = K @ x
    assert y[0] == x[0] and np.array_equal(K[0], np.eye(A)[0]) and not K[1:, 0].any()
    np.testing.assert_allclose(K @ np.ones(A), np.ones(A), atol=1e-10)
    line = np.r_[4.0, 0.25 * np.arange(1, A)]
    np.testing.assert_allclose(K @ line, line, atol=1e-9)
    # The minimiser of Σ (x̃ − x)² + λ Σ (Δ²x̃)² over ages 1..A−1: (I + λ D₂ᵀD₂) x̃ = x there.
    D2 = np.diff(np.eye(A - 1), n=2, axis=0)
    np.testing.assert_allclose((np.eye(A - 1) + 1e3 * D2.T @ D2) @ y[1:], x[1:], atol=1e-9)
    assert np.array_equal(fc.anchor_smoother(A, 0.0), np.eye(A))
    with pytest.raises(ValueError):
        fc.anchor_smoother(A, -1.0)


def test_smooth_residual_acts_on_each_sex_along_age():
    r = np.random.default_rng(4).normal(size=(3, 2, 110))
    K = fc.anchor_smoother(110, 1e3)
    out = fc.smooth_residual(r, 1e3)
    np.testing.assert_allclose(out[1, 0], K @ r[1, 0], atol=1e-12)
    np.testing.assert_allclose(out[2, 1], K @ r[2, 1], atol=1e-12)
    assert np.array_equal(fc.smooth_residual(r, 0.0), r) and fc.smooth_residual(r, 0.0) is not r
    # Smoothing removes single-age noise: the second differences shrink.
    assert np.mean(np.diff(out, n=2, axis=-1) ** 2) < 0.01 * np.mean(np.diff(r, n=2, axis=-1) ** 2)


def test_one_anchor_smoothing_for_both_forecasters():
    P = params()
    lam = float(P.section("forecast")["anchor_smoothing"])
    assert lam > 0 and float(P.section("flowfield")["anchor_smoothing"]) == lam
    with pytest.raises(ValueError):
        fc.MDMxSettings.from_params(P.section("forecast"), P.section("tucker"), age_rank=4, anchor_smoothing=-1.0)


def test_mdmx_origin_anchors_the_jump_off_residual_without_look_ahead():
    """On a synthetic tensor: the anchored schedules are the unanchored ones plus the weighted
    residual at the last training year, smoothed over age; the fits do not change; data after the origin
    change nothing."""
    from synthetic import tensor

    M, observed, years, pops = tensor()
    origin = 1960
    settings = fc.MDMxSettings.from_params(params().section("forecast"), params().section("tucker"), age_rank=4,
                                           n_pc=3, horizon=10)
    assert settings.anchor_smoothing > 0                            # the configured form smooths the residual
    jobs = fc.cv_jobs(observed, pops, years, [origin], 10, 30, 3).head(3)
    runs = {a: fc.mdmx_origin(M, observed, pops, years, origin, jobs, settings, anchor=a)
            for a in ("none", "persistent", 2.0)}
    space = fc.origin_space(M, observed, years, origin, tau=settings.tau, age_rank=4, smooth=settings.smooth, n_pc=3)
    idx = runs["none"]["schedule_index"]
    for job in jobs.itertuples():
        c, t0 = int(np.flatnonzero(pops == job.pop)[0]), int(np.flatnonzero(years == job.last_train_year)[0])
        resid = fc.smooth_residual(M[:, :, c, t0] - fc.scores_to_logit(space["scores"][c, t0], space["z_mean"],
                                                                       space["L"]), settings.anchor_smoothing)
        rows = (idx["pop"] == job.pop).to_numpy()
        h = idx.loc[rows, "year"].to_numpy() - job.last_train_year
        diff_p = runs["persistent"]["schedules"][rows] - runs["none"]["schedules"][rows]
        diff_d = runs[2.0]["schedules"][rows] - runs["none"]["schedules"][rows]
        np.testing.assert_allclose(diff_p, np.broadcast_to(resid, diff_p.shape), atol=1e-12)
        np.testing.assert_allclose(diff_d, 0.5 ** (h / 2)[:, None, None] * resid, atol=1e-12)
    for a in ("persistent", 2.0):
        pd.testing.assert_series_equal(runs[a]["jobs"]["rho"], runs["none"]["jobs"]["rho"])
        z = runs[a]["schedules"]
        e = lifetable.e0_from_logit(z)
        both = runs[a]["points"].query("sex == 'both'")["e0_fc"].to_numpy()
        assert np.array_equal(np.sort(both), np.sort(e.mean(axis=1)))
        assert (runs[a]["points"]["sd"] > 0).all()
    assert {"anchor_year", "anchor_medabs"} <= set(runs["persistent"]["jobs"].columns)

    after = years > origin
    rng = np.random.default_rng(5)
    M_noise, obs_noise = M.copy(), observed.copy()
    M_noise[..., after] = rng.normal(0.0, 3.0, M_noise[..., after].shape)
    obs_noise[:, after] = rng.random(obs_noise[:, after].shape) < 0.5
    noisy = fc.mdmx_origin(M_noise, obs_noise, pops, years, origin, jobs, settings, anchor="persistent")
    pd.testing.assert_frame_equal(noisy["points"], runs["persistent"]["points"], check_exact=True)


def test_loo_calibrate_within_era_uses_only_the_calibration_set():
    rows = []
    for origin, scale in ((1, 1.0), (2, 1.0), (3, 3.0), (4, 3.0)):     # two "eras" of different spread
        for k, z in enumerate((-1.0, 1.0)):
            rows.append({"origin": origin, "sex": "both", "year": origin + 1 + k, "e0_fc": 70.0, "sd": 1.0,
                         "e0_obs": 70.0 + scale * z})
    pts = pd.DataFrame(rows)
    cal = {1: [2], 2: [1], 3: [4], 4: [3]}
    out = fc.loo_calibrate(pts, (80,), calibration=cal)
    kappa = out.groupby("origin")["kappa_loo"].first()
    assert kappa.tolist() == pytest.approx([1.0, 1.0, 3.0, 3.0])
    pooled = fc.loo_calibrate(pts, (80,)).groupby("origin")["kappa_loo"].first()
    assert pooled[1] == pytest.approx(np.std([-1, 1, -3, 3, -3, 3]))      # every other origin
    with pytest.raises(ValueError, match="own calibration"):
        fc.loo_calibrate(pts, (80,), calibration={1: [1], 2: [1], 3: [4], 4: [3]})
    with pytest.raises(ValueError, match="No calibration"):
        fc.loo_calibrate(pts, (80,), calibration={1: [2]})


def test_ex_from_qx_known_answers():
    # No deaths before the last age, which everyone dies in: e_x = (A - x) - 0.5 (linear within the year).
    qx = np.zeros(10)
    qx[-1] = 1.0
    assert lifetable.ex_from_qx(qx, 3) == pytest.approx(6.5)
    # A constant q: e_x does not depend on x away from the end (geometric series, the tail cut at A).
    q = np.full(400, 0.1)
    assert lifetable.ex_from_qx(q, 5) == pytest.approx(lifetable.ex_from_qx(q, 50), rel=1e-12)
    assert lifetable.ex_from_logit(np.log(q / (1 - q)), 5) == pytest.approx(lifetable.ex_from_qx(q, 5))
    with pytest.raises(ValueError):
        lifetable.ex_from_qx(qx, 0)


def test_schedule_errors_weight_by_survivors_or_deaths():
    rng = np.random.default_rng(0)
    q = np.clip(np.linspace(0.001, 0.6, 50), 1e-6, 0.99)
    z_obs = np.broadcast_to(np.log(q / (1 - q)), (3, 2, 50)).copy()
    same = z_obs + 0.2                                              # the same error at every age
    for w in ("survivors", "deaths"):
        np.testing.assert_allclose(fc.schedule_errors(same, z_obs, {}, weights=w)["all"], 0.2)
    old = z_obs.copy()
    old[..., 40:] += 1.0                                            # an error at old ages only
    lx = fc.schedule_errors(old, z_obs, {}, weights="survivors")["all"].to_numpy()
    dx = fc.schedule_errors(old, z_obs, {}, weights="deaths")["all"].to_numpy()
    assert np.all(dx > lx)                                          # deaths weight the old ages more
    with pytest.raises(ValueError):
        fc.schedule_errors(same, z_obs, {}, weights="people")
    assert rng is not None


# ══════════════════════════════════════════════════════════════════════════════
# a₀: one value in every e₀ and every m → q conversion (decision R1)
# ══════════════════════════════════════════════════════════════════════════════

def test_the_book_has_one_a0_in_code_and_params():
    """``[data] a0`` in params and :data:`mdmx.lifetable.A0` are the same value, and every default
    that names a₀ uses it."""
    import inspect

    from mdmx import clustering, compare, evaluation, logquad
    from mdmx.config import params

    assert params().section("data")["a0"] == lifetable.A0 == compare.A0 == 0.3
    for fn in (lifetable.e0_from_qx, lifetable.e0_from_logit, lifetable.life_table_from_qx,
               fc.e0_and_gradient, logquad.abridged_mx, logquad.fit_logquad, clustering.observed_e0,
               evaluation.arriaga):
        assert inspect.signature(fn).parameters["a0"].default == lifetable.A0, fn.__qualname__


def test_comparator_schedules_convert_age_0_with_the_books_a0():
    """q₀ = m₀ / (1 + (1 − a₀) m₀) with a₀ = 0.3 and q_x = m_x / (1 + ½ m_x) above: with m₀ = 0.2
    (infant mortality as before 1900) q₀ = 0.2 / 1.14, where a₀ = 0.5 would give 0.2 / 1.1."""
    ages = 5
    row = {"method": "LC", "pop": "X", "origin": 1900, "year": 1901}
    mx = pd.DataFrame([{**row, "sex": s, **{f"a{a}": (0.2 if a == 0 else 0.01) for a in range(ages)}}
                       for s in ("female", "male")])
    _, z = fc.comparator_schedules(mx, ages, ax=0.5)
    q = lifetable.expit(z[0])
    assert q[:, 0] == pytest.approx(0.2 / 1.14, rel=1e-12)
    assert q[:, 1:] == pytest.approx(np.full((2, ages - 1), 0.01 / 1.005), rel=1e-12)
    # The e₀ of the converted schedule is the book's e₀ of q with the same a₀ in L₀.
    e0 = fc.e0_and_gradient(z)[0]                                    # (1, 2)
    assert e0[0] == pytest.approx(lifetable.e0_from_qx(q), rel=1e-12)


@pytest.mark.parametrize("sex", ["lt_female", "lt_male"])
def test_a0_conversion_reproduces_the_hmd_e0_before_1950(sex):
    """Review B's check, kept: the e₀ of HMD's own m_x converted with the book's a₀ (0.3 at age 0, 0.5
    above) is the book's e₀ of HMD's own q_x within 0.02 y (median over population-years before 1950,
    ages 0–109; in the 20261029 release +0.010 for females, −0.001 for males); with a₀ = 0.5 the median
    offset is more than 0.1 y (−0.13 and −0.19).  Needs the HMD release in
    ``raw/`` (skipped without it)."""
    from mdmx import data
    from mdmx.config import params

    try:
        source = data.hmd_source(str(params().section("data")["hmd_release"]))
    except FileNotFoundError:
        pytest.skip("the HMD release is not in raw/")
    with data.open_member(source, data.HMD_MEMBERS[sex]) as fh:
        lt, _ = data.read_hmd_text(fh)
    lt = lt[(lt["Year"] < 1950) & (lt["Age"] < 110)]
    mx = lt.pivot_table(index=["PopName", "Year"], columns="Age", values="mx")
    qx = lt.pivot_table(index=["PopName", "Year"], columns="Age", values="qx").reindex_like(mx)
    keep = mx.notna().all(axis=1) & qx.notna().all(axis=1) & (mx.shape[1] == 110)
    mx, qx = mx[keep].to_numpy(), qx[keep].to_numpy()
    assert len(mx) > 1000                                              # population-years before 1950
    e0_hmd = lifetable.e0_from_qx(qx)
    e0_book = lifetable.e0_from_qx(lifetable.qx_from_mx(mx, lifetable.conversion_ax(110, 0.5)))
    e0_half = lifetable.e0_from_qx(lifetable.qx_from_mx(mx, lifetable.conversion_ax(110, 0.5, a0=0.5)))
    assert abs(np.median(e0_book - e0_hmd)) < 0.02
    assert abs(np.median(e0_half - e0_hmd)) > 0.1


def test_e0_from_mx_converts_age_0_with_the_books_a0():
    """compare.e0_from_mx (Lee–Carter's quantiles) uses the same age-0 conversion."""
    from mdmx import compare as cp

    m = np.full(6, 0.01)
    m[0] = 0.2
    q = lifetable.qx_from_mx(m, lifetable.conversion_ax(6, 0.5))
    assert q[0] == pytest.approx(0.2 / 1.14) and q[1] == pytest.approx(0.01 / 1.005)
    assert cp.e0_from_mx(m, 6, 0.5) == pytest.approx(lifetable.e0_from_logit(lifetable.logit(q)), rel=1e-12)


def test_pooled_deaths_put_in_the_pooled_rate_times_the_years_exposure_only_where_pooling_changed_m():
    key = {"pop": ["P"] * 4, "sex": ["female"] * 4, "year": [2000, 2000, 2001, 2001], "age": [1, 2, 1, 2]}
    de = pd.DataFrame({**key, "deaths": [0.0, 5.0, 4.0, 6.0], "exposure": [1000.0, 1000.0, 3000.0, 2000.0]})
    cur = pd.DataFrame({**key, "mx": [0.0, 0.005, 4 / 3000, 0.003]})
    pooled = cur.assign(mx=[0.001, 0.005, 0.001, 0.003])            # age 1 pooled: 4/4000
    out = fc.pooled_deaths(de, cur, pooled)
    assert list(out.columns) == list(de.columns) and len(out) == len(de)
    np.testing.assert_allclose(out["deaths"], [1.0, 5.0, 3.0, 6.0])   # 0.001 × 1000, unchanged, 0.001 × 3000
    np.testing.assert_array_equal(out["exposure"], de["exposure"])
