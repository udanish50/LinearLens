from linearlens.synthetic import analyze_controlled_experiment, generate_controlled_experiment


def test_synthetic_structure() -> None:
    experiment = generate_controlled_experiment(samples=1000, seed=1)
    assert experiment.inputs.shape == (1000, 8)
    assert experiment.weights.shape == (4, 8)
    assert experiment.expected_roles[2] == "polysemantic"


def test_illustrative_mode_recovers_intended_structure() -> None:
    _, report = analyze_controlled_experiment(samples=20000, seed=2, role_mode="illustrative")
    assert report.roles[0] == "monosemantic"
    assert report.roles[1] == "monosemantic"
    assert report.roles[2] == "polysemantic"
    assert report.roles[3] == "dead"
