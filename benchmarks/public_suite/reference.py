"""Small NumPy reference for browser/public-evidence verification of Linear Lens.

This module intentionally covers dense feed-forward networks only.  The main
``linearlens`` package remains the source of truth for supported PyTorch model
adapters, capture, tracing, QSM, and validation workflows.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

EPSILON = 1e-12
Z_CRITICAL_90_ONE_TAILED = NormalDist().inv_cdf(0.90)


def normalized_influence(
    x: np.ndarray,
    weight: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Return neuron × input normalized absolute pre-activation influence.

    For neuron k and input j, the unnormalized influence is the sample mean of
    ``abs(x_ij * w_kj)``.  Rows are normalized to sum to one.
    """
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(
            "Expected x=(samples, inputs), weight=(neurons, inputs)."
        )
    mu = np.mean(np.abs(x[:, None, :] * weight[None, :, :]), axis=0)
    denom = np.sum(mu, axis=1, keepdims=True)
    return np.divide(
        mu,
        np.maximum(denom, epsilon),
        out=np.zeros_like(mu),
        where=True,
    )


def entropy_rows(p: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log(p + epsilon), axis=1)


def layer_relative_zscore(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values))
    if sd <= epsilon:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / sd


def classify_roles(
    z: np.ndarray,
    *,
    deeper_layer: bool = False,
    z_critical: float = Z_CRITICAL_90_ONE_TAILED,
) -> list[str]:
    """Apply the paper's layer-relative low/middle/high entropy regions."""
    low = "unimodal" if deeper_layer else "monosemantic"
    middle = "multimodal" if deeper_layer else "polysemantic"
    high = "muted" if deeper_layer else "dead/flat"
    return [
        low if value < -z_critical else high if value > z_critical else middle
        for value in np.asarray(z)
    ]


def role_report(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    deeper_layer: bool = False,
) -> dict[str, Any]:
    p = normalized_influence(x, weight)
    h = entropy_rows(p)
    z = layer_relative_zscore(h)
    return {
        "influence": p,
        "entropy": h,
        "zscore": z,
        "roles": classify_roles(z, deeper_layer=deeper_layer),
    }


def apply_activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    if name in {"identity", "linear"}:
        return x
    raise ValueError(f"Unsupported activation in public reference: {name}")


def analyze_exported_dense_model(
    x: np.ndarray,
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Analyze exported dense layers without altering weights or activations."""
    current = np.asarray(x, dtype=float)
    reports: list[dict[str, Any]] = []
    hidden_index = 0
    for layer in model["layers"]:
        if layer["type"] == "linear":
            weight = np.asarray(layer["weight"], dtype=float)
            bias = np.asarray(layer["bias"], dtype=float)
            preactivation = current @ weight.T + bias
            if layer.get("is_output"):
                reports.append({"kind": "output", "preactivation": preactivation})
            else:
                hidden_index += 1
                report = role_report(current, weight, deeper_layer=hidden_index > 1)
                report.update(
                    {
                        "kind": "hidden",
                        "layer_index": hidden_index,
                        "preactivation": preactivation,
                    }
                )
                reports.append(report)
            current = preactivation
        elif layer["type"] == "activation":
            current = apply_activation(current, layer["name"])
            if reports and reports[-1]["kind"] == "hidden":
                reports[-1]["activation"] = current
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")
    return reports
