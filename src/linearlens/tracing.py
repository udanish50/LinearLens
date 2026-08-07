from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy import RoleMode, RoleThresholds, classify_hidden_layer
from .influence import influence_distribution


@dataclass(frozen=True)
class HiddenLayerReport:
    distribution: np.ndarray
    roles: np.ndarray
    entropy: np.ndarray
    zscores: np.ndarray
    top_upstream: tuple[tuple[int, ...], ...]


def analyze_hidden_layer(
    upstream_activations: np.ndarray,
    weights: np.ndarray,
    *,
    role_mode: RoleMode = "zscore",
    thresholds: RoleThresholds | None = None,
    contribution_threshold: float = 0.10,
) -> HiddenLayerReport:
    """Apply Linear Lens to a deeper linear component using upstream activations as inputs."""
    thresholds = thresholds or RoleThresholds()
    dist = influence_distribution(upstream_activations, weights)
    roles, entropy, zscores = classify_hidden_layer(dist, mode=role_mode, thresholds=thresholds)
    paths: list[tuple[int, ...]] = []
    for row in dist:
        indices = np.flatnonzero(row >= contribution_threshold)
        if indices.size == 0:
            indices = np.array([int(np.argmax(row))])
        paths.append(tuple(int(i) for i in indices))
    return HiddenLayerReport(dist, roles, entropy, zscores, tuple(paths))


def output_composition(
    upstream_activations: np.ndarray,
    output_weights: np.ndarray,
    *,
    contribution_threshold: float = 0.10,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return normalized upstream influence and compositional sources for output neurons."""
    dist = influence_distribution(upstream_activations, output_weights)
    sources: list[tuple[int, ...]] = []
    for row in dist:
        idx = np.flatnonzero(row >= contribution_threshold)
        if idx.size == 0:
            idx = np.array([int(np.argmax(row))])
        sources.append(tuple(int(i) for i in idx))
    return dist, tuple(sources)
