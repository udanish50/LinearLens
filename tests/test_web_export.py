import json

import torch.nn as nn

from linearlens.web import export_dense_model


def test_dense_web_export(tmp_path):
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    path = export_dense_model(
        model,
        tmp_path / "model.json",
        feature_names=["a", "b", "c"],
    )
    payload = json.loads(path.read_text())
    assert payload["method"] == "Linear Lens"
    assert payload["schema_version"] == 1
    assert payload["feature_names"] == ["a", "b", "c"]
    assert sum(layer["type"] == "linear" for layer in payload["layers"]) == 2
    assert payload["layers"][-1]["is_output"] is True
