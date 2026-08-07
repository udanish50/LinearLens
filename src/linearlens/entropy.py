from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RoleMode = Literal["zscore", "illustrative"]


@dataclass(frozen=True)
class RoleThresholds:
    """Classification thresholds.

    `z_cutoff=1.645` follows Equation 13 in the publication.
    The dominance fields support an explicitly optional illustrative mode.
    """

    z_cutoff: float = 1.645
    monosemantic_dominance: float = 0.80
    diffuse_max_share: float = 0.20


def shannon_entropy(distribution: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    p = np.asarray(distribution, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("distribution must be 2D [neurons, features]")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("distribution must be finite and non-negative")
    rows = p.sum(axis=1)
    if not np.allclose(rows, 1.0, atol=1e-6):
        raise ValueError("each distribution row must sum to 1")
    return -np.sum(p * np.log(np.maximum(p, epsilon)), axis=1)


def entropy_zscores(entropy: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(entropy, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("entropy must be a 1D array")
    if values.size == 0:
        raise ValueError("entropy cannot be empty")
    std = float(values.std(ddof=0))
    if std <= epsilon:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def classify_first_layer(
    distribution: np.ndarray,
    *,
    mode: RoleMode = "zscore",
    thresholds: RoleThresholds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify first-layer neurons as monosemantic/polysemantic/dead.

    `zscore` applies Equation 13 exactly. `illustrative` is a convenience mode based on
    dominant-share examples discussed in the article and is not Equation 13.
    """
    p = np.asarray(distribution, dtype=np.float64)
    thresholds = thresholds or RoleThresholds()
    entropy = shannon_entropy(p)
    z = entropy_zscores(entropy)
    if mode == "zscore":
        labels = np.full(p.shape[0], "polysemantic", dtype=object)
        labels[z < -thresholds.z_cutoff] = "monosemantic"
        labels[z > thresholds.z_cutoff] = "dead"
    elif mode == "illustrative":
        max_share = p.max(axis=1)
        labels = np.full(p.shape[0], "polysemantic", dtype=object)
        labels[max_share >= thresholds.monosemantic_dominance] = "monosemantic"
        labels[max_share <= thresholds.diffuse_max_share] = "dead"
    else:
        raise ValueError(f"unsupported role mode: {mode}")
    return labels, entropy, z


def classify_hidden_layer(
    distribution: np.ndarray,
    *,
    mode: RoleMode = "zscore",
    thresholds: RoleThresholds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify deeper neurons as unimodal/multimodal/muted.

    The publication says deeper layers reuse the influence, entropy, and z-score pipeline.
    This implementation maps the corresponding low/middle/high entropy regions to the
    deeper-layer terminology.
    """
    first, entropy, z = classify_first_layer(distribution, mode=mode, thresholds=thresholds)
    mapping = {"monosemantic": "unimodal", "polysemantic": "multimodal", "dead": "muted"}
    labels = np.array([mapping[str(role)] for role in first], dtype=object)
    return labels, entropy, z
