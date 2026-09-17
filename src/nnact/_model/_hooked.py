from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookedModel(nn.Module):
    """Base wrapper for capturing raw layer activations."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model
        self._activations: dict[str, torch.Tensor] = {}

    def get_activation(self, layer_name: str) -> torch.Tensor:
        try:
            return self._activations[layer_name]
        except KeyError:
            raise RuntimeError(
                f"Activation for layer '{layer_name}' was not captured."
            ) from None

    @contextmanager
    def capture(self, layer_names: list[str]) -> Iterator[None]:
        self._activations.clear()

        handles: list[RemovableHandle] = []
        named_modules = dict(self._model.named_modules())

        try:
            for name in layer_names:
                if name not in named_modules:
                    raise ValueError(f"Layer '{name}' does not exist in model.")

                def make_hook(
                    layer_key: str,
                ) -> Callable[
                    [nn.Module, tuple[object, ...], object],
                    None,
                ]:
                    def hook(
                        _module: nn.Module,
                        _inputs: tuple[object, ...],
                        output: object,
                    ) -> None:
                        tensor = (
                            output
                            if isinstance(output, torch.Tensor)
                            else (output[0] if isinstance(output, tuple) else None)
                        )

                        if not isinstance(tensor, torch.Tensor):
                            raise TypeError(
                                f"Layer '{layer_key}' output "
                                f"is not a Tensor, got {type(output)}"
                            )

                        self._activations[layer_key] = tensor

                    return hook

                handles.append(
                    named_modules[name].register_forward_hook(make_hook(name))
                )

            yield

        finally:
            for handle in handles:
                handle.remove()
