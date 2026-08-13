"""Rebuild controlled role-recovery and fixed-model bootstrap checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from reference import role_report

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "benchmarks" / "public_suite"
RESULTS = SUITE / "results"
Z_CRITICAL = NormalDist().inv_cdf(0.90)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_inputs(
    family: str,
    samples: int,
    features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if family == "gaussian":
        return rng.normal(size=(samples, features))
    if family == "lognormal":
        return rng.lognormal(0.0, 0.8, size=(samples, features))
    if family == "student_t":
        return rng.standard_t(3.0, size=(samples, features))
    if family == "zero_inflated":
        x = rng.normal(size=(samples, features))
        x[rng.random(x.shape) < 0.55] = 0.0
        return x
    if family == "correlated":
        correlation = np.fromfunction(
            lambda i, j: 0.65 ** np.abs(i - j),
            (features, features),
        )
        return rng.multivariate_normal(
            np.zeros(features),
            correlation,
            size=samples,
        )
    if family == "heterogeneous":
        scales = np.logspace(-1, 1, features)
        return rng.normal(size=(samples, features)) * scales
    raise ValueError(f"Unknown family: {family}")


def controlled_weights(
    features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weights: list[np.ndarray] = []
    truth: list[str] = []

    for index in range(10):
        row = np.ones(features) * 0.005
        row[index % features] = 15.0
        weights.append(row)
        truth.append("monosemantic")

    for _ in range(44):
        row = np.ones(features) * 0.01
        count = min(features, max(2, int(rng.integers(2, min(6, features) + 1))))
        selected = rng.choice(features, size=count, replace=False)
        row[selected] = rng.uniform(0.7, 1.4, count)
        weights.append(row)
        truth.append("polysemantic")

    for _ in range(10):
        weights.append(np.ones(features))
        truth.append("dead/flat")

    return np.asarray(weights), np.asarray(truth)


def build_controlled_recovery() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_index = 0
    families = [
        "gaussian",
        "lognormal",
        "student_t",
        "zero_inflated",
        "correlated",
        "heterogeneous",
    ]
    for family in families:
        for features in (6, 12, 24):
            for samples in (250, 800):
                scenario_index += 1
                rng = np.random.default_rng(9000 + scenario_index)
                x = synthetic_inputs(family, samples, features, rng)
                weight, truth = controlled_weights(features, rng)
                report = role_report(x, weight)
                predicted = np.asarray(report["roles"])
                rows.append(
                    {
                        "scenario": f"{family}_d{features}_n{samples}",
                        "distribution": family,
                        "features": features,
                        "samples": samples,
                        "neurons": len(truth),
                        "accuracy": float(np.mean(predicted == truth)),
                        "mono_recall": float(
                            np.mean(
                                predicted[truth == "monosemantic"]
                                == "monosemantic"
                            )
                        ),
                        "poly_recall": float(
                            np.mean(
                                predicted[truth == "polysemantic"]
                                == "polysemantic"
                            )
                        ),
                        "flat_recall": float(
                            np.mean(
                                predicted[truth == "dead/flat"]
                                == "dead/flat"
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_fixture(entry: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    dataset = pd.read_csv(ROOT / str(entry["dataset_file"]))
    model = json.loads((ROOT / str(entry["model_file"])).read_text())
    x = dataset.iloc[:, :-1].to_numpy(float)
    preprocessing = model["preprocessing"]
    median = np.asarray(preprocessing["imputer_median"], dtype=float)
    mean = np.asarray(preprocessing["mean"], dtype=float)
    std = np.asarray(preprocessing["std"], dtype=float)
    x = np.where(np.isnan(x), median[None, :], x)
    return (x - mean) / std, model


def first_linear_weight(model: dict[str, object]) -> np.ndarray:
    for layer in model["layers"]:
        if layer["type"] == "linear":
            return np.asarray(layer["weight"], dtype=float)
    raise ValueError("No linear layer in exported model.")


def build_bootstrap_stability(manifest: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in manifest["datasets"]:
        x, model = load_fixture(entry)
        weight = first_linear_weight(model)
        baseline = np.asarray(role_report(x, weight)["roles"])
        rng = np.random.default_rng(7878)
        agreements: list[float] = []
        for _ in range(40):
            indices = rng.choice(len(x), size=len(x), replace=True)
            sampled = np.asarray(role_report(x[indices], weight)["roles"])
            agreements.append(float(np.mean(sampled == baseline)))
        rows.append(
            {
                "dataset": entry["dataset"],
                "task": entry["task"],
                "rows": len(x),
                "features": x.shape[1],
                "neurons": len(baseline),
                "bootstrap_replicates": 40,
                "mean_role_agreement": float(np.mean(agreements)),
                "min_role_agreement": float(np.min(agreements)),
                "max_role_agreement": float(np.max(agreements)),
            }
        )
    return pd.DataFrame(rows)


def refresh_evidence_manifest() -> None:
    output = RESULTS / "evidence_manifest.json"
    files = []
    for path in sorted(SUITE.rglob("*")):
        if path.is_file() and path != output:
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_on": "2026-08-13",
                "method": "Linear Lens",
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    manifest_path = SUITE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    recovery = build_controlled_recovery()
    recovery.to_csv(RESULTS / "controlled_role_recovery.csv", index=False)

    bootstrap = build_bootstrap_stability(manifest)
    bootstrap.to_csv(
        RESULTS / "fixed_model_bootstrap_stability.csv",
        index=False,
    )

    manifest["controlled_role_recovery_scenarios"] = len(recovery)
    manifest["controlled_role_recovery_mean_accuracy"] = float(
        recovery["accuracy"].mean()
    )
    manifest["fixed_model_bootstrap_dataset_count"] = len(bootstrap)
    manifest["fixed_model_bootstrap_mean_agreement"] = float(
        bootstrap["mean_role_agreement"].mean()
    )
    manifest["z_critical_one_tailed_90"] = Z_CRITICAL
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    refresh_evidence_manifest()

    print(
        "PASS validation suites: "
        f"{len(recovery)} controlled scenarios · "
        f"{len(bootstrap)} fixed-model bootstrap datasets"
    )


if __name__ == "__main__":
    main()
