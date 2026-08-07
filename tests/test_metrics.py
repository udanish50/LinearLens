import numpy as np

from linearlens.metrics import cronbach_alpha, regression_metrics


def test_regression_metrics_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, y.copy())
    assert metrics == {"mae": 0.0, "rmse": 0.0, "smape": 0.0}


def test_cronbach_alpha_returns_finite_value() -> None:
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(200, 1))
    items = latent + rng.normal(scale=0.2, size=(200, 3))
    assert cronbach_alpha(items) > 0.8
