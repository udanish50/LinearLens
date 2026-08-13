"""Export supported dense PyTorch models for Linear Lens's browser analyzer.

The web format is deliberately narrow: ``nn.Sequential``-style dense networks
containing Linear plus ReLU/Tanh/Sigmoid activations.  Full Linear Lens analysis
remains available in Python for broader architectures and adapter workflows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch.nn as nn

_SUPPORTED_ACTIVATIONS = {
    nn.ReLU: "relu",
    nn.Tanh: "tanh",
    nn.Sigmoid: "sigmoid",
    nn.Identity: "identity",
}


def _flatten_modules(model: nn.Module) -> list[nn.Module]:
    if isinstance(model, nn.Sequential):
        return list(model.children())
    children = list(model.children())
    if len(children) == 1 and isinstance(children[0], nn.Sequential):
        return list(children[0].children())
    raise TypeError(
        "Web export currently supports nn.Sequential dense models. "
        "Use the Python package for other architectures."
    )


def export_dense_model(
    model: nn.Module,
    path: str | Path,
    *,
    feature_names: Iterable[str],
    task: str = "regression",
    preprocessing: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a browser-safe JSON representation without changing ``model``."""
    modules = _flatten_modules(model)
    layers: list[dict[str, Any]] = []
    linear_positions = [i for i, module in enumerate(modules) if isinstance(module, nn.Linear)]
    if not linear_positions:
        raise ValueError("No nn.Linear layers found.")
    last_linear_position = linear_positions[-1]
    for index, module in enumerate(modules):
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": module.weight.detach().cpu().numpy().astype(float).tolist(),
                    "bias": (
                        module.bias.detach().cpu().numpy().astype(float).tolist()
                        if module.bias is not None
                        else np.zeros(module.out_features, dtype=float).tolist()
                    ),
                    "is_output": index == last_linear_position,
                }
            )
        elif type(module) in _SUPPORTED_ACTIVATIONS:
            layers.append({"type": "activation", "name": _SUPPORTED_ACTIVATIONS[type(module)]})
        else:
            raise TypeError(f"Unsupported web-export module: {type(module).__name__}")
    names = [str(name) for name in feature_names]
    first_linear = next(module for module in modules if isinstance(module, nn.Linear))
    if len(names) != first_linear.in_features:
        raise ValueError("feature_names length must equal the first Linear layer's input width.")
    payload = {
        "schema_version": 1,
        "method": "Linear Lens",
        "task": task,
        "feature_names": names,
        "preprocessing": preprocessing or {},
        "layers": layers,
        "metadata": metadata or {},
    }
    destination = Path(path)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
