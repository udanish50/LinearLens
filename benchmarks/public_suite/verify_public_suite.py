from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "benchmarks" / "public_suite"
sys.path.insert(0, str(SUITE))
from reference import analyze_exported_dense_model  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_numeric(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    data = np.array(
        [
            [
                float(value)
                if value not in {"", "nan", "NaN"}
                else np.nan
                for value in row
            ]
            for row in rows[1:]
        ],
        dtype=float,
    )
    return data[:, :-1], header[:-1]


def main() -> None:
    manifest = json.loads((SUITE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_dataset_count"] == 45
    assert manifest["real_public_dataset_count"] == 6
    assert manifest["controlled_fixture_count"] == 39
    assert manifest["model_analysis_runs"] == 180

    evidence_manifest = json.loads(
        (SUITE / "results" / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    for item in evidence_manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256(path) == item["sha256"], item["path"]

    for entry in manifest["datasets"]:
        dpath = ROOT / entry["dataset_file"]
        mpath = ROOT / entry["model_file"]
        assert dpath.is_file() and mpath.is_file()
        assert sha256(dpath) == entry["dataset_sha256"]
        assert sha256(mpath) == entry["model_sha256"]
        x, features = read_csv_numeric(dpath)
        model = json.loads(mpath.read_text(encoding="utf-8"))
        pp = model["preprocessing"]
        med = np.asarray(pp["imputer_median"], dtype=float)
        mean = np.asarray(pp["mean"], dtype=float)
        std = np.asarray(pp["std"], dtype=float)
        x = np.where(np.isnan(x), med[None, :], x)
        x = (x - mean) / std
        reports = analyze_exported_dense_model(x, model)
        hidden = [r for r in reports if r["kind"] == "hidden"]
        assert len(hidden) == 2
        for report in hidden:
            p = report["influence"]
            assert np.all(np.isfinite(p))
            assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-9
            assert len(report["roles"]) == p.shape[0]
        assert features == model["feature_names"]

    def count_rows(name: str) -> int:
        with (SUITE / "results" / name).open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.reader(handle)) - 1

    assert count_rows("benchmark_runs.csv") == 180
    assert count_rows("controlled_role_recovery.csv") == 36
    assert count_rows("fixed_model_bootstrap_stability.csv") == 45
    assert count_rows("neuron_roles.csv") > 9000
    print(
        "PASS Linear Lens public evidence: "
        "45 datasets · 180 trained model analyses · "
        "36 controlled recovery scenarios"
    )


if __name__ == "__main__":
    main()
