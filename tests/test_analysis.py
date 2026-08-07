import numpy as np

from linearlens import analyze_linear_component


def test_analysis_report_shapes() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(200, 5))
    w = rng.normal(size=(16, 5))
    report = analyze_linear_component(x, w, feature_names=["a", "b", "c", "d", "e"])
    assert report.distribution.shape == (16, 5)
    assert report.roles.shape == (16,)
    assert report.qsm.shape == (16, 5)
    assert report.preactivations.shape == (200, 16)
