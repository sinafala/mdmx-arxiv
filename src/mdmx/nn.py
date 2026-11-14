"""A small, transparent neural network in pure NumPy (paper §10).

Several parts of MDMx map a handful of inputs (summary indicators, life
expectancy, cluster labels) to a vector of Tucker core weights.  A
multilayer perceptron (MLP) is a convenient, flexible way to do that, and
the networks involved are tiny (two hidden layers of 64 units, a dozen
outputs), so there is no need for a deep-learning framework.  Keeping the
implementation in plain NumPy means

* every number in a trained network is an ordinary array that can be
  stored as an artefact (``MLP.to_dict``) and read back by any language —
  no pickles;
* the gradient code is short enough to read and is checked against finite
  differences in ``tests/test_nn.py``;
* results are bit-for-bit reproducible from a seed.

Contents
--------
:class:`Standardizer`
    Centre and scale columns (fitted on training data only).
:class:`MLP`
    ReLU hidden layers, linear output layer, He initialisation.
:class:`Adam`
    The Adam optimiser (Kingma & Ba 2015).
:func:`mse_loss`, :class:`LinearDecoderLoss`
    Loss functions.  A loss is any callable ``loss(out, Y) -> (value, grad)``
    returning the scalar loss averaged over the rows of the batch and its
    gradient with respect to the network output ``out``.
:func:`fit_mlp`
    The one training loop used everywhere: mini-batch Adam with weight
    decay and early stopping on a validation split.

Conventions
-----------
Rows are observations, columns are features: ``X`` is ``(n, d_in)`` and the
network output is ``(n, d_out)``.  All randomness comes from the integer
``seed`` passed in, split into independent streams with
``numpy.random.SeedSequence`` so that, for example, changing the validation
share does not change the initial weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["Standardizer", "MLP", "Adam", "mse_loss", "LinearDecoderLoss", "fit_mlp",
           "validation_split"]

#: Signature of a loss: ``loss(out, Y) -> (value, d value / d out)``.
LossFn = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]


# ══════════════════════════════════════════════════════════════════════════════
# Standardisation of inputs (and, where useful, of targets)
# ══════════════════════════════════════════════════════════════════════════════

class Standardizer:
    """Centre and scale each column: ``(X − mean) / scale``.

    Neural networks train much more reliably when every input has mean 0
    and standard deviation 1.  The mean and scale must be computed from the
    *training* rows only (otherwise information about the test rows leaks
    into the model), stored with the model, and re-applied unchanged to any
    new input.

    Attributes
    ----------
    mean, scale : arrays (d,)
        Column means and standard deviations of the data passed to
        :meth:`fit`.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        if self.mean.shape != self.scale.shape:
            raise ValueError(f"mean {self.mean.shape} and scale {self.scale.shape} differ in shape")
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("every column needs a positive, finite scale "
                             "(a constant column cannot be standardised)")

    @classmethod
    def fit(cls, X: np.ndarray) -> Standardizer:
        """Column means and standard deviations of ``X`` (n, d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError(f"need a 2-D array with at least two rows, got shape {X.shape}")
        return cls(X.mean(axis=0), X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """``(X − mean) / scale``; ``X`` is (n, d)."""
        return (np.asarray(X, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        """Undo :meth:`transform`: ``Xs · scale + mean``."""
        return np.asarray(Xs, dtype=float) * self.scale + self.mean

    def to_dict(self) -> dict[str, np.ndarray]:
        """Plain arrays for the artefact store."""
        return {"mean": self.mean.copy(), "scale": self.scale.copy()}

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> Standardizer:
        return cls(d["mean"], d["scale"])


# ══════════════════════════════════════════════════════════════════════════════
# The network
# ══════════════════════════════════════════════════════════════════════════════

class MLP:
    """Multilayer perceptron: ``input → [linear → ReLU] × H → linear → output``.

    Parameters
    ----------
    sizes : layer widths ``[d_in, h_1, …, h_H, d_out]``.  ``[2, 64, 64, 12]``
        is §10's summary-indicator network with c_age = 6 (:func:`mdmx.svdcomp.fit_core_model`).
    seed : integer seed for the initial weights.

    Notes
    -----
    Weights are initialised with He scaling, ``W ~ N(0, 2 / fan_in)``, which
    keeps the variance of the activations roughly constant through ReLU
    layers; biases start at zero.  ``W[l]`` has shape ``(fan_in, fan_out)``
    so that a layer is ``h_next = h @ W[l] + b[l]`` for a batch of rows.
    """

    def __init__(self, sizes: Sequence[int], seed: int) -> None:
        sizes = [int(s) for s in sizes]
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError(f"invalid layer sizes {sizes}")
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
                  for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)]
        self.b = [np.zeros(n_out) for n_out in sizes[1:]]

    # ── forward and backward passes ──────────────────────────────────────────
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Network output for a batch, plus what the backward pass needs.

        Parameters
        ----------
        X : (n, d_in) inputs

        Returns
        -------
        out : (n, d_out) outputs (linear, no activation)
        acts : the input to every layer, ``[X, h_1, …, h_H]`` — the
            post-ReLU activations, kept for :meth:`backward`.
        """
        h = np.asarray(X, dtype=float)
        acts = [h]
        last = len(self.W) - 1
        for layer, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = h @ W + b
            if layer < last:
                h = np.maximum(z, 0.0)           # ReLU
                acts.append(h)
            else:
                h = z                            # linear output layer
        return h, acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Output for inputs ``X`` (n, d_in) → (n, d_out)."""
        return self.forward(X)[0]

    def backward(self, acts: list[np.ndarray], d_out: np.ndarray
                 ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Gradients of a scalar loss with respect to every weight and bias.

        Parameters
        ----------
        acts : the list returned by :meth:`forward` for the same batch
        d_out : (n, d_out) gradient of the loss with respect to the output.
            Any averaging over the batch is already in ``d_out`` (the loss
            functions below return the gradient of the *batch-mean* loss),
            so no further division by n happens here.

        Returns
        -------
        dW, db : lists matching ``self.W`` and ``self.b``.
        """
        n_layers = len(self.W)
        dW: list[np.ndarray] = [np.empty(0)] * n_layers
        db: list[np.ndarray] = [np.empty(0)] * n_layers
        delta = d_out
        for layer in range(n_layers - 1, -1, -1):
            h_in = acts[layer]
            dW[layer] = h_in.T @ delta
            db[layer] = delta.sum(axis=0)
            if layer > 0:
                # Back through the linear map, then through the ReLU of the
                # previous layer (its derivative is 1 where the unit was
                # active, i.e. where its output h_in is positive, else 0).
                delta = (delta @ self.W[layer].T) * (h_in > 0.0)
        return dW, db

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def params(self) -> list[np.ndarray]:
        """All trainable arrays, ``[W_0, b_0, W_1, b_1, …]`` (views, not copies)."""
        out = []
        for W, b in zip(self.W, self.b, strict=True):
            out += [W, b]
        return out

    def n_params(self) -> int:
        """Number of trainable parameters."""
        return int(sum(p.size for p in self.params()))

    def copy_weights(self) -> list[np.ndarray]:
        """Deep copy of all parameters (used to remember the best epoch)."""
        return [p.copy() for p in self.params()]

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        """Overwrite every parameter, in the order of :meth:`params`."""
        for target, source in zip(self.params(), weights, strict=True):
            target[...] = source

    def to_dict(self) -> dict[str, Any]:
        """The network as plain arrays: ``{"sizes", "W": [...], "b": [...]}``."""
        return {"sizes": np.array(self.sizes), "W": [W.copy() for W in self.W],
                "b": [b.copy() for b in self.b]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MLP:
        """Rebuild a network saved with :meth:`to_dict`."""
        net = cls(list(d["sizes"]), seed=0)
        for layer, (W, b) in enumerate(zip(d["W"], d["b"], strict=True)):
            if W.shape != net.W[layer].shape or b.shape != net.b[layer].shape:
                raise ValueError(f"layer {layer}: stored shapes {W.shape}, {b.shape} do not "
                                 f"match sizes {net.sizes}")
            net.W[layer] = np.array(W, dtype=float)
            net.b[layer] = np.array(b, dtype=float)
        return net


# ══════════════════════════════════════════════════════════════════════════════
# Optimiser
# ══════════════════════════════════════════════════════════════════════════════

class Adam:
    """Adam optimiser (Kingma & Ba 2015) for a list of NumPy arrays.

    Keeps running averages of each parameter's gradient (``m``) and squared
    gradient (``v``), corrects their start-up bias, and moves each parameter
    by ``lr · m̂ / (√v̂ + eps)``.  The step size is therefore roughly ``lr``
    per parameter regardless of the scale of its gradient.
    """

    def __init__(self, params: Sequence[np.ndarray], lr: float, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = list(params)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Update every parameter in place from its gradient."""
        self.t += 1
        c1 = 1.0 - self.beta1 ** self.t
        c2 = 1.0 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def mse_loss(out: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error over all entries, and its gradient.

    ``value = mean((out − Y)²)``; ``grad = 2 (out − Y) / out.size``.
    """
    resid = out - Y
    return float(np.mean(resid ** 2)), 2.0 * resid / resid.size


class LinearDecoderLoss:
    """Weighted squared error measured *after* a fixed linear decoder.

    Paper: eq:svdcomp:loss

    The network output ``out`` (n, k) is not compared with the target
    directly.  It is first decoded,

        Ŷ = out · Dᵀ + offset            (n, p),

    and the loss is a weighted sum of squared errors over the p decoded
    columns, averaged over the rows of the batch:

        value = (1/n) Σ_i Σ_j w_j (Ŷ_ij − Y_ij)².

    In the summary-indicator model (:func:`mdmx.svdcomp.fit_core_model`)
    the network outputs (standardised) truncated Tucker core weights, the decoder is the reconstruction matrix ``R`` (:func:`mdmx.svdcomp.reconstruction_matrix`)
    and ``Y`` is the observed logit(q_x) schedule, so this is that
    model's reconstruction loss, with ``w`` holding the cell weights of :func:`mdmx.svdcomp.loss_weights`.  The
    gradient with respect to the network output follows from the chain rule:

        ∂value/∂out = (2/n) ((Ŷ − Y) ⊙ w) · D.

    Parameters
    ----------
    D : (p, k) decoder matrix
    offset : (p,) added after decoding (zeros if ``None``)
    weights : (p,) non-negative weight of each decoded column
    """

    def __init__(self, D: np.ndarray, offset: np.ndarray | None, weights: np.ndarray) -> None:
        self.D = np.asarray(D, dtype=float)
        p = self.D.shape[0]
        self.offset = np.zeros(p) if offset is None else np.asarray(offset, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        if self.offset.shape != (p,) or self.weights.shape != (p,):
            raise ValueError(f"offset {self.offset.shape} and weights {self.weights.shape} "
                             f"must both have shape ({p},)")
        if np.any(self.weights < 0):
            raise ValueError("loss weights must be non-negative")

    def decode(self, out: np.ndarray) -> np.ndarray:
        """``out · Dᵀ + offset``: network output (n, k) → decoded (n, p)."""
        return out @ self.D.T + self.offset

    def __call__(self, out: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        resid = self.decode(out) - Y
        n = resid.shape[0]
        weighted = resid * self.weights
        value = float(np.sum(weighted * resid) / n)
        grad = (2.0 / n) * weighted @ self.D
        return value, grad


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def validation_split(n: int, val_frac: float, rng: np.random.Generator,
                     groups: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask (n,) marking the rows held out for early stopping.

    Paper: sec:svdcomp:nn

    Without ``groups`` a random ``val_frac`` of the rows is held out.  With
    ``groups`` (one label per row) whole groups are held out — groups are
    taken in random order until at least ``val_frac`` of the rows are
    covered — so that, for example, all years of a population are either
    used for fitting or for validation, never split between the two.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if groups is None:
        held = np.zeros(n, dtype=bool)
        held[rng.choice(n, size=max(1, int(round(val_frac * n))), replace=False)] = True
        return held
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {groups.shape}")
    labels = rng.permutation(np.unique(groups))
    sizes = np.array([np.sum(groups == g) for g in labels])
    n_take = int(np.searchsorted(np.cumsum(sizes), val_frac * n) + 1)
    if n_take >= len(labels):
        raise ValueError(f"val_frac = {val_frac} would hold out every group")
    return np.isin(groups, labels[:n_take])


def fit_mlp(X: np.ndarray, Y: np.ndarray, hidden: Sequence[int], *, lr: float, epochs: int,
            batch_size: int, l2: float, seed: int, val_frac: float = 0.0,
            patience: int | None = None, loss: str | LossFn = "mse", out_dim: int | None = None,
            groups: np.ndarray | None = None) -> tuple[MLP, dict[str, Any]]:
    """Train an :class:`MLP` by mini-batch Adam, with optional early stopping.

    Paper: sec:svdcomp:nn

    This is the single training loop of the project.  Each epoch visits the
    training rows once, in a fresh random order, in mini-batches; each batch
    takes one Adam step on

        objective = loss(batch) + (l2 / 2) Σ_layers ‖W‖²,

    i.e. the batch-mean loss plus weight decay on the weight matrices
    (biases are not penalised).  After every epoch the training loss (mean
    of that epoch's batch losses) and, if ``val_frac > 0``, the loss on the
    validation rows are recorded.

    Early stopping: when a validation split is used, the weights
    from the epoch with the lowest validation loss are kept, and training
    stops once the validation loss has not improved for ``patience``
    consecutive epochs (or after ``epochs`` epochs).

    Parameters
    ----------
    X : (n, d_in) inputs, already standardised (see :class:`Standardizer`)
    Y : (n, ·) targets passed to the loss.  For ``loss="mse"`` this is what
        the network output should match; for a :class:`LinearDecoderLoss`
        it is the decoded target (e.g. the full logit(q_x) schedule).
    hidden : widths of the hidden layers, e.g. ``[64, 64]``
    lr, epochs, batch_size, l2 : learning rate, maximum number of epochs,
        mini-batch size and weight-decay coefficient
    seed : integer seed; initial weights, validation split and batch order
        use independent streams derived from it
    val_frac : share of rows held out for early stopping (0 = none)
    patience : epochs without improvement before stopping (requires
        ``val_frac > 0``); ``None`` trains for all ``epochs`` but still
        returns the best-validation weights if there is a validation split
    loss : ``"mse"`` or a callable ``loss(out, Y) -> (value, grad)``
    out_dim : network output width; defaults to ``Y.shape[1]`` for
        ``"mse"`` and must be given for a custom loss (e.g. the number of
        core weights, which differs from the width of ``Y``)
    groups : optional (n,) labels; if given the validation split holds out
        whole groups (see :func:`validation_split`)

    Returns
    -------
    net : the trained network
    history : dict with ``"curve"`` (a DataFrame with columns ``epoch``,
        ``train_loss``, ``val_loss``), ``"best_epoch"``, ``"epochs_run"``,
        ``"stopped_early"``, ``"n_train"``, ``"n_val"``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"X has {n} rows but Y has {Y.shape[0]}")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("X and Y must be finite (no NaN or infinity)")
    if patience is not None and val_frac <= 0:
        raise ValueError("early stopping (patience) needs a validation split (val_frac > 0)")
    if loss == "mse":
        loss_fn: LossFn = mse_loss
        out_dim = Y.shape[1] if out_dim is None else out_dim
    elif callable(loss):
        loss_fn = loss
        if out_dim is None:
            raise ValueError("out_dim must be given with a custom loss")
    else:
        raise ValueError(f"unknown loss {loss!r}")

    init_ss, split_ss, order_ss = np.random.SeedSequence(seed).spawn(3)
    net = MLP([X.shape[1], *hidden, out_dim], seed=int(init_ss.generate_state(1)[0]))
    held = (validation_split(n, val_frac, np.random.default_rng(split_ss), groups)
            if val_frac > 0 else np.zeros(n, dtype=bool))
    X_tr, Y_tr, X_va, Y_va = X[~held], Y[~held], X[held], Y[held]
    order_rng = np.random.default_rng(order_ss)

    params = net.params()
    opt = Adam(params, lr=lr)
    n_tr = X_tr.shape[0]
    rows: list[dict[str, float]] = []
    best_val, best_epoch, best_weights, since_best = np.inf, 0, net.copy_weights(), 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        perm = order_rng.permutation(n_tr)
        loss_sum = 0.0
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            out, acts = net.forward(X_tr[idx])
            batch_loss, d_out = loss_fn(out, Y_tr[idx])
            loss_sum += batch_loss * len(idx)
            dW, db = net.backward(acts, d_out)
            grads = []
            for W, gW, gb in zip(net.W, dW, db, strict=True):
                grads += [gW + l2 * W, gb]           # weight decay on W only
            opt.step(grads)

        # Training loss: the row-weighted mean of the batch losses of this
        # epoch (the usual "running" training loss; no extra pass needed).
        train_loss = loss_sum / n_tr
        val_loss = loss_fn(net.predict(X_va), Y_va)[0] if held.any() else np.nan
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if held.any():
            if val_loss < best_val:
                best_val, best_epoch, best_weights, since_best = val_loss, epoch, net.copy_weights(), 0
            else:
                since_best += 1
                if patience is not None and since_best >= patience:
                    stopped_early = True
                    break
    if held.any():
        net.set_weights(best_weights)
    else:
        best_epoch = len(rows)
    history = {"curve": pd.DataFrame(rows), "best_epoch": int(best_epoch),
               "epochs_run": len(rows), "stopped_early": stopped_early,
               "n_train": int(n_tr), "n_val": int(held.sum())}
    return net, history
