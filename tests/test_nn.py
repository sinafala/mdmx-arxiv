"""Tests for mdmx.nn: gradients, standardisation, reproducibility, storage."""

import numpy as np
import pytest

from mdmx import nn


def _numerical_gradients(net, X, Y, loss, eps=1e-6):
    """Central finite differences of loss(net(X), Y) for every parameter."""
    grads = []
    for p in net.params():
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = loss(net.predict(X), Y)[0]
            p[idx] = old - eps
            down = loss(net.predict(X), Y)[0]
            p[idx] = old
            g[idx] = (up - down) / (2 * eps)
        grads.append(g)
    return grads


def _analytic_gradients(net, X, Y, loss):
    out, acts = net.forward(X)
    _, d_out = loss(out, Y)
    dW, db = net.backward(acts, d_out)
    grads = []
    for gW, gb in zip(dW, db, strict=True):
        grads += [gW, gb]
    return grads


@pytest.mark.parametrize("which", ["mse", "decoder"])
def test_backprop_matches_finite_differences(which):
    rng = np.random.default_rng(1)
    net = nn.MLP([3, 5, 4, 2], seed=7)
    # Shift the biases so that no ReLU unit sits exactly at its kink.
    for b in net.b[:-1]:
        b += 0.1
    X = rng.normal(size=(6, 3))
    if which == "mse":
        Y = rng.normal(size=(6, 2))
        loss = nn.mse_loss
    else:
        D = rng.normal(size=(7, 2))
        loss = nn.LinearDecoderLoss(D, rng.normal(size=7), rng.uniform(0.1, 2.0, size=7))
        Y = rng.normal(size=(6, 7))
    analytic = _analytic_gradients(net, X, Y, loss)
    numeric = _numerical_gradients(net, X, Y, loss)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_standardizer_round_trip_and_storage():
    rng = np.random.default_rng(2)
    X = rng.normal(loc=[1.0, -3.0, 10.0], scale=[0.1, 2.0, 5.0], size=(50, 3))
    sc = nn.Standardizer.fit(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(sc.inverse_transform(Z), X, atol=1e-12)
    again = nn.Standardizer.from_dict(sc.to_dict())
    np.testing.assert_array_equal(again.transform(X), Z)


def test_standardizer_rejects_constant_column():
    with pytest.raises(ValueError):
        nn.Standardizer.fit(np.column_stack([np.arange(5.0), np.ones(5)]))


def _toy_problem():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(300, 2))
    Y = np.column_stack([np.sin(X[:, 0]) + X[:, 1] ** 2, X[:, 0] * X[:, 1]])
    return X, Y


def test_training_is_deterministic_and_learns():
    X, Y = _toy_problem()
    kw = dict(lr=1e-2, epochs=60, batch_size=32, l2=1e-5, seed=11, val_frac=0.2, patience=10)
    net1, h1 = nn.fit_mlp(X, Y, [16, 16], **kw)
    net2, h2 = nn.fit_mlp(X, Y, [16, 16], **kw)
    for a, b in zip(net1.params(), net2.params(), strict=True):
        np.testing.assert_array_equal(a, b)
    curve = h1["curve"]
    assert curve["train_loss"].iloc[-1] < 0.2 * curve["train_loss"].iloc[0]
    # The kept weights are those of the best validation epoch.
    best = curve.loc[curve["epoch"] == h1["best_epoch"], "val_loss"].item()
    assert best == pytest.approx(curve["val_loss"].min())
    # A different seed gives a different network.
    net3, _ = nn.fit_mlp(X, Y, [16, 16], **{**kw, "seed": 12})
    assert not np.array_equal(net1.W[0], net3.W[0])


def test_group_validation_split_holds_out_whole_groups():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(10), 20)
    held = nn.validation_split(len(groups), 0.25, rng, groups)
    for g in np.unique(groups):
        assert held[groups == g].all() or not held[groups == g].any()
    assert held.mean() >= 0.25


def test_network_round_trips_through_dict():
    X, Y = _toy_problem()
    net, _ = nn.fit_mlp(X, Y, [8], lr=1e-2, epochs=3, batch_size=64, l2=0.0, seed=1)
    again = nn.MLP.from_dict(net.to_dict())
    np.testing.assert_array_equal(again.predict(X), net.predict(X))


def test_patience_without_validation_is_an_error():
    X, Y = _toy_problem()
    with pytest.raises(ValueError):
        nn.fit_mlp(X, Y, [8], lr=1e-2, epochs=3, batch_size=64, l2=0.0, seed=1, patience=5)
