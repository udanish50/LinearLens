import torch

from linearlens.capture import ActivationRecorder


def test_recorder_does_not_change_output() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    x = torch.randn(5, 3)
    expected = model(x)
    with ActivationRecorder(model) as recorder:
        observed = model(x)
    torch.testing.assert_close(observed, expected)
    assert set(recorder.snapshots) == {"0", "2"}
