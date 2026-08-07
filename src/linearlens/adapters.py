from __future__ import annotations

from typing import Literal

import torch
from torch import nn

LSTMGate = Literal["input", "forget", "cell", "output"]


def linear_weight(module: nn.Linear) -> torch.Tensor:
    """Return a detached view of a Linear layer's weight matrix."""
    if not isinstance(module, nn.Linear):
        raise TypeError("module must be torch.nn.Linear")
    return module.weight.detach()


def lstm_gate_weight(
    module: nn.LSTM,
    *,
    gate: LSTMGate = "input",
    layer: int = 0,
    reverse: bool = False,
) -> torch.Tensor:
    """Extract PyTorch LSTM input-to-hidden weights for one gate.

    PyTorch gate order is input, forget, cell/candidate, output (i, f, g, o).
    """
    if not isinstance(module, nn.LSTM):
        raise TypeError("module must be torch.nn.LSTM")
    if layer < 0 or layer >= module.num_layers:
        raise ValueError(f"layer must be in [0, {module.num_layers - 1}]")
    suffix = f"_l{layer}" + ("_reverse" if reverse else "")
    weight = getattr(module, f"weight_ih{suffix}").detach()
    chunks = weight.chunk(4, dim=0)
    index = {"input": 0, "forget": 1, "cell": 2, "output": 3}[gate]
    return chunks[index]


def multihead_attention_query_weight(module: nn.MultiheadAttention) -> torch.Tensor:
    """Extract the query projection matrix from torch.nn.MultiheadAttention."""
    if not isinstance(module, nn.MultiheadAttention):
        raise TypeError("module must be torch.nn.MultiheadAttention")
    if module.in_proj_weight is None:
        q_weight = getattr(module, "q_proj_weight", None)
        if q_weight is None:
            raise ValueError("attention module does not expose a query projection weight")
        return q_weight.detach()
    return module.in_proj_weight[: module.embed_dim].detach()
