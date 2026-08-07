import torch

from linearlens import analyze_linear_component


def main() -> None:
    torch.manual_seed(11)
    x = torch.randn(512, 5)
    layer = torch.nn.Linear(5, 12)
    report = analyze_linear_component(
        x,
        layer.weight,
        layer.bias,
        feature_names=["temperature", "hour", "energy", "day_of_year", "day_of_week"],
    )
    print(report.qsm_markdown())


if __name__ == "__main__":
    main()
