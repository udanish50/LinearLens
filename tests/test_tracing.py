import numpy as np

from linearlens.tracing import analyze_hidden_layer, output_composition


def test_hidden_layer_report() -> None:
    rng = np.random.default_rng(3)
    upstream = rng.normal(size=(300, 6))
    weights = rng.normal(size=(8, 6))
    report = analyze_hidden_layer(upstream, weights)
    assert report.distribution.shape == (8, 6)
    assert len(report.top_upstream) == 8


def test_output_composition() -> None:
    rng = np.random.default_rng(5)
    upstream = rng.normal(size=(100, 4))
    weights = rng.normal(size=(2, 4))
    dist, sources = output_composition(upstream, weights)
    assert dist.shape == (2, 4)
    assert len(sources) == 2
