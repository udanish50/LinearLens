"""Export a supported dense PyTorch model for the Linear Lens website analyzer."""

import torch.nn as nn

from linearlens.web import export_dense_model


model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

feature_names = ["temperature", "humidity", "ghi", "load_history"]

export_dense_model(
    model,
    "linear_lens_web_model.json",
    feature_names=feature_names,
    task="regression",
    preprocessing={
        "imputer_median": [20.0, 50.0, 400.0, 3.0],
        "mean": [20.0, 50.0, 400.0, 3.0],
        "std": [8.0, 18.0, 300.0, 1.2],
    },
    metadata={
        "note": "Example only. Export the fitted preprocessing used by your own model.",
    },
)
