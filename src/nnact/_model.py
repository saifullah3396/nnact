from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import final, override

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from nnact._types import LayerActivation


@final
class HookedModel(nn.Module):
    """Wraps any nn.Module to extract layer activations via forward hooks.

    Returns the model's raw output unchanged -- callers wanting derived
    per-sample/per-token fields (see `nnact.adapters`) apply their own
    transform to that raw output themselves, since only the caller has both
    the output and the batch's inputs in scope at the same time.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.activations: list[LayerActivation] = []

    def get_activation(self, layer_name: str) -> torch.Tensor:
        for act in self.activations:
            if act.layer_name == layer_name:
                return act.tensor
        raise RuntimeError(f"Activation for layer '{layer_name}' was not captured.")

    @contextmanager  # pyright: ignore[reportDeprecated]
    def capture(self, layer_names: list[str]) -> Iterator[None]:
        self.activations.clear()
        handles: list[RemovableHandle] = []
        named_modules: dict[str, nn.Module] = dict(self.model.named_modules())  # pyright: ignore[reportUnknownArgumentType]

        try:
            for name in layer_names:
                if name not in named_modules:
                    raise ValueError(f"Layer '{name}' does not exist in model.")

                def make_hook(
                    layer_key: str,
                ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
                    def hook(
                        _module: nn.Module,
                        _inputs: tuple[object, ...],
                        output: object,
                    ) -> None:
                        tensor = (
                            output
                            if isinstance(output, torch.Tensor)
                            else (
                                output[0] if isinstance(output, tuple) else None  # pyright: ignore[reportIndexIssue]
                            )
                        )
                        if not isinstance(tensor, torch.Tensor):
                            raise TypeError(
                                f"Layer '{layer_key}' output is not a Tensor, got {type(output)}"
                            )
                        self.activations.append(
                            LayerActivation(layer_name=layer_key, tensor=tensor)
                        )

                    return hook

                handle = named_modules[name].register_forward_hook(make_hook(name))
                handles.append(handle)

            yield
        finally:
            for h in handles:
                h.remove()

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.model(*args, **kwargs)  # pyright: ignore[reportAny]
