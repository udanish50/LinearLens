from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import ComponentReport, analyze_linear_component


@dataclass(frozen=True)
class SyntheticExperiment:
    inputs: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    expected_roles: tuple[str, ...]


def generate_controlled_experiment(samples: int = 5000, seed: int = 7) -> SyntheticExperiment:
    """Generate the article's controlled-regression structure (Eqs. 17–24)."""
    if samples < 10:
        raise ValueError("samples must be at least 10")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(samples, 8))
    epsilon = rng.normal(0.0, 0.1, size=samples)
    y = 3 * x[:, 0] + 2 * x[:, 1] - 1.5 * x[:, 2] + 0.7 * x[:, 0] * x[:, 1] + epsilon
    weights = np.array(
        [
            [2.5, 0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00],
            [0.05, 2.2, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00],
            [1.4, 1.3, 0.10, 0.00, 0.00, 0.05, 0.00, 0.00],
            [0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
        ],
        dtype=np.float64,
    )
    bias = np.zeros(4, dtype=np.float64)
    expected = ("monosemantic", "monosemantic", "polysemantic", "diffuse")
    return SyntheticExperiment(x, y, weights, bias, expected)


def analyze_controlled_experiment(
    samples: int = 5000,
    seed: int = 7,
    *,
    role_mode: str = "zscore",
) -> tuple[SyntheticExperiment, ComponentReport]:
    experiment = generate_controlled_experiment(samples=samples, seed=seed)
    report = analyze_linear_component(
        experiment.inputs,
        experiment.weights,
        experiment.bias,
        feature_names=[f"x{i}" for i in range(1, 9)],
        role_mode=role_mode,  # type: ignore[arg-type]
    )
    return experiment, report
