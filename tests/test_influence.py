import numpy as np

from linearlens.influence import influence_distribution, mean_absolute_influence


def test_mean_absolute_influence_matches_manual_formula() -> None:
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    w = np.array([[2.0, -1.0]])
    mu = mean_absolute_influence(x, w)
    expected = np.array([[4.0, 3.0]])
    np.testing.assert_allclose(mu, expected)


def test_distribution_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    dist = influence_distribution(rng.normal(size=(100, 5)), rng.normal(size=(8, 5)))
    np.testing.assert_allclose(dist.sum(axis=1), 1.0)
