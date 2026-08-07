from __future__ import annotations

import argparse

from .synthetic import analyze_controlled_experiment


def _synthetic(args: argparse.Namespace) -> int:
    experiment, report = analyze_controlled_experiment(
        samples=args.samples,
        seed=args.seed,
        role_mode=args.role_mode,
    )
    print("Linear Lens controlled synthetic experiment")
    print(f"samples: {experiment.inputs.shape[0]}")
    print(f"mode: {args.role_mode}")
    print()
    for i, expected in enumerate(experiment.expected_roles):
        top = report.distribution[i].argsort()[::-1][:3]
        top_text = ", ".join(f"x{j + 1}={report.distribution[i, j]:.3f}" for j in top)
        print(
            f"p{i + 1}: expected={expected:13s} recovered={str(report.roles[i]):13s} "
            f"entropy={report.entropy[i]:.4f} z={report.zscores[i]:+.4f} [{top_text}]"
        )
    print()
    print(report.qsm_markdown())
    if args.role_mode == "zscore":
        print()
        print("Note: exact Equation-13 z-score thresholds can differ from the article's")
        print("illustrative/controlled-table labels. See docs/reproducibility.md.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linear-lens")
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="run the controlled synthetic example")
    synthetic.add_argument("--samples", type=int, default=5000)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--role-mode", choices=["zscore", "illustrative"], default="zscore")
    synthetic.set_defaults(func=_synthetic)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
