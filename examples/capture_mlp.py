import torch
from torch import nn

from linearlens.capture import ActivationRecorder


class DemoMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    model = DemoMLP().eval()
    batch = torch.randn(32, 5)
    with ActivationRecorder(model) as recorder:
        prediction = model(batch)
    print("prediction shape:", tuple(prediction.shape))
    for name, snapshot in recorder.snapshots.items():
        print(name, tuple(snapshot.inputs.shape), "->", tuple(snapshot.output.shape))


if __name__ == "__main__":
    main()
