from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionValidation:
    neuron_index: int
    selected_features: tuple[int, ...]
    r2: float
    coefficients: tuple[float, ...]
    intercept: float


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return beta[1:], float(beta[0]), r2


def validate_polysemantic_neurons(
    inputs: np.ndarray,
    preactivations: np.ndarray,
    distribution: np.ndarray,
    roles: np.ndarray,
    *,
    feature_threshold: float = 0.15,
    min_features: int = 2,
) -> list[RegressionValidation]:
    """Regression validation corresponding to Phase 2 / Eq. 14.

    Selected features are those whose normalized influence exceeds `feature_threshold`.
    If fewer than `min_features` pass, the largest influences are selected instead.
    """
    x = np.asarray(inputs, dtype=np.float64)
    p = np.asarray(preactivations, dtype=np.float64)
    dist = np.asarray(distribution, dtype=np.float64)
    role_array = np.asarray(roles, dtype=object)
    if x.ndim != 2 or p.ndim != 2 or dist.ndim != 2:
        raise ValueError("inputs, preactivations, and distribution must all be 2D")
    if p.shape[0] != x.shape[0]:
        raise ValueError("inputs and preactivations must have the same number of samples")
    if p.shape[1] != dist.shape[0] or role_array.shape[0] != dist.shape[0]:
        raise ValueError("neuron dimensions do not align")

    reports: list[RegressionValidation] = []
    for neuron in np.flatnonzero(role_array == "polysemantic"):
        selected = np.flatnonzero(dist[neuron] > feature_threshold)
        if selected.size < min_features:
            selected = np.argsort(dist[neuron])[-min_features:]
        selected = np.sort(selected)
        coef, intercept, r2 = _fit_ols(x[:, selected], p[:, neuron])
        reports.append(
            RegressionValidation(
                neuron_index=int(neuron),
                selected_features=tuple(int(i) for i in selected),
                r2=float(r2),
                coefficients=tuple(float(v) for v in coef),
                intercept=intercept,
            )
        )
    return reports
