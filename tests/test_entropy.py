import numpy as np

from linearlens.entropy import (
    RoleThresholds,
    classify_first_layer,
    entropy_zscores,
    shannon_entropy,
)


def test_entropy_low_for_concentrated_distribution() -> None:
    p = np.array([[0.99, 0.01], [0.5, 0.5]])
    entropy = shannon_entropy(p)
    assert entropy[0] < entropy[1]


def test_zscores_are_centered() -> None:
    z = entropy_zscores(np.array([0.1, 0.2, 0.5, 0.8]))
    assert abs(float(z.mean())) < 1e-12


def test_equation_13_thresholds() -> None:
    p = np.array(
        [
            [0.999, 0.001, 0.0, 0.0],
            [0.6, 0.2, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.26, 0.24, 0.25, 0.25],
            [0.55, 0.15, 0.15, 0.15],
            [0.52, 0.18, 0.15, 0.15],
        ]
    )
    roles, _, z = classify_first_layer(p, thresholds=RoleThresholds(z_cutoff=0.5))
    assert roles[np.argmin(z)] == "monosemantic"
    assert roles[np.argmax(z)] == "dead"
