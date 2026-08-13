import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks" / "public_suite"
sys.path.insert(0, str(SUITE))
from reference import classify_roles, entropy_rows, normalized_influence  # noqa: E402


def test_public_manifest_and_counts():
    manifest = json.loads((SUITE / "manifest.json").read_text())
    assert manifest["public_dataset_count"] == 45
    assert manifest["model_analysis_runs"] == 180
    assert manifest["controlled_role_recovery_scenarios"] == 36
    assert manifest["fixed_model_bootstrap_dataset_count"] == 45


def test_reference_influence_is_distribution():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 6))
    w = rng.normal(size=(12, 6))
    p = normalized_influence(x, w)
    assert p.shape == (12, 6)
    assert np.allclose(p.sum(axis=1), 1.0)
    h = entropy_rows(p)
    assert np.all(np.isfinite(h))


def test_role_labels_cover_published_regions():
    roles = classify_roles(np.array([-2.0, 0.0, 2.0]))
    assert roles == ["monosemantic", "polysemantic", "dead/flat"]
