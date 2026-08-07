from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import torch
from torch import nn


@dataclass
class Snapshot:
    inputs: torch.Tensor
    output: torch.Tensor


class ActivationRecorder:
    """Observational forward-hook recorder.

    Hooks return None and therefore do not replace module outputs or alter forward logic.
    Only the first tensor positional input and tensor output are recorded.
    """

    def __init__(self, model: nn.Module, module_types: tuple[type[nn.Module], ...] = (nn.Linear,)):
        self.model = model
        self.module_types = module_types
        self.snapshots: dict[str, Snapshot] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                return None
            self.snapshots[name] = Snapshot(inputs[0].detach().clone(), output.detach().clone())
            return None

        return hook

    def __enter__(self) -> ActivationRecorder:
        for name, module in self.model.named_modules():
            if name and isinstance(module, self.module_types):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
