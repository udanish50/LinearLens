from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _as_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_inputs_weights(inputs: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return validated 2D arrays with shapes [samples, features] and [neurons, features]."""
    x = _as_numpy(inputs).astype(np.float64, copy=False)
    w = _as_numpy(weights).astype(np.float64, copy=False)
    if x.ndim != 2:
        raise ValueError(f"inputs must have shape [samples, features], got {x.shape}")
    if w.ndim != 2:
        raise ValueError(f"weights must have shape [neurons, features], got {w.shape}")
    if x.shape[1] != w.shape[1]:
        raise ValueError(
            f"feature mismatch: inputs have {x.shape[1]} features, weights expect {w.shape[1]}"
        )
    if x.shape[0] == 0:
        raise ValueError("inputs must contain at least one sample")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("inputs and weights must contain only finite values")
    return x, w


def mean_absolute_influence(inputs: ArrayLike, weights: ArrayLike) -> np.ndarray:
    """Compute Eq. 9 style mean absolute feature-to-neuron influence.

    For neuron i and feature j:
        mu[i, j] = mean_k |x[k, j] * W[i, j]|.
    """
    x, w = validate_inputs_weights(inputs, weights)
    mean_abs_x = np.mean(np.abs(x), axis=0)
    return np.abs(w) * mean_abs_x[None, :]


def normalize_influence(influence: ArrayLike, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize each neuron row to a probability-style feature distribution."""
    mu = _as_numpy(influence).astype(np.float64, copy=False)
    if mu.ndim != 2:
        raise ValueError("influence must be a 2D [neurons, features] array")
    if np.any(mu < 0) or not np.all(np.isfinite(mu)):
        raise ValueError("influence must be finite and non-negative")
    denom = mu.sum(axis=1, keepdims=True)
    out = np.divide(
        mu,
        np.maximum(denom, epsilon),
        out=np.zeros_like(mu),
        where=np.ones_like(mu, dtype=bool),
    )
    zero_rows = denom[:, 0] <= epsilon
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / mu.shape[1]
    return out


def influence_distribution(inputs: ArrayLike, weights: ArrayLike) -> np.ndarray:
    """Compute normalized influence distributions corresponding to Eq. 10."""
    return normalize_influence(mean_absolute_influence(inputs, weights))
