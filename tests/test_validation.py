import numpy as np

from linearlens.validation import validate_polysemantic_neurons


def test_regression_validation_recovers_linear_behavior() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 3))
    pre = (2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.2).reshape(-1, 1)
    dist = np.array([[0.55, 0.40, 0.05]])
    roles = np.array(["polysemantic"], dtype=object)
    reports = validate_polysemantic_neurons(x, pre, dist, roles, feature_threshold=0.15)
    assert len(reports) == 1
    assert reports[0].selected_features == (0, 1)
    assert reports[0].r2 > 0.999999
