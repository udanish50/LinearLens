from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .entropy import RoleMode, RoleThresholds, classify_first_layer
from .influence import influence_distribution, mean_absolute_influence, validate_inputs_weights
from .qsm import QSMConfig, qsm_to_markdown, qualitative_symbolic_matrix
from .validation import RegressionValidation, validate_polysemantic_neurons


@dataclass
class ComponentReport:
    feature_names: tuple[str, ...]
    influence: np.ndarray
    distribution: np.ndarray
    entropy: np.ndarray
    zscores: np.ndarray
    roles: np.ndarray
    preactivations: np.ndarray
    regression_validations: list[RegressionValidation]
    qsm: np.ndarray

    def qsm_markdown(self) -> str:
        return qsm_to_markdown(self.qsm, self.feature_names, self.roles)


def analyze_linear_component(
    inputs: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    bias: np.ndarray | torch.Tensor | None = None,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    role_mode: RoleMode = "zscore",
    thresholds: RoleThresholds | None = None,
    polysemantic_feature_threshold: float = 0.15,
    qsm_config: QSMConfig | None = None,
) -> ComponentReport:
    """Analyze a linear forward-computation component without modifying it."""
    x, w = validate_inputs_weights(inputs, weights)
    thresholds = thresholds or RoleThresholds()
    qsm_config = qsm_config or QSMConfig()
    if feature_names is None:
        names = tuple(f"x{i}" for i in range(x.shape[1]))
    else:
        if len(feature_names) != x.shape[1]:
            raise ValueError("feature_names length must equal the input feature count")
        names = tuple(feature_names)

    b = np.zeros(w.shape[0], dtype=np.float64)
    if bias is not None:
        if isinstance(bias, torch.Tensor):
            b = bias.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            b = np.asarray(bias, dtype=np.float64)
        if b.shape != (w.shape[0],):
            raise ValueError(f"bias must have shape {(w.shape[0],)}, got {b.shape}")

    preactivations = x @ w.T + b
    mu = mean_absolute_influence(x, w)
    dist = influence_distribution(x, w)
    roles, entropy, zscores = classify_first_layer(dist, mode=role_mode, thresholds=thresholds)
    regression = validate_polysemantic_neurons(
        x,
        preactivations,
        dist,
        roles,
        feature_threshold=polysemantic_feature_threshold,
    )
    qsm = qualitative_symbolic_matrix(dist, qsm_config)
    return ComponentReport(
        feature_names=names,
        influence=mu,
        distribution=dist,
        entropy=entropy,
        zscores=zscores,
        roles=roles,
        preactivations=preactivations,
        regression_validations=regression,
        qsm=qsm,
    )
