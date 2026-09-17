from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    import pandas as pd


class HookedModel(nn.Module):
    """Base wrapper for capturing raw layer activations."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model
        self._activations: dict[str, np.ndarray] = {}

    def forward(self, *args: object, **kwargs: object) -> object:
        return self._model(*args, **kwargs)

    def get_activation(self, layer_name: str) -> np.ndarray:
        try:
            return self._activations[layer_name]
        except KeyError:
            raise RuntimeError(
                f"Activation for layer '{layer_name}' was not captured."
            ) from None

    def available_layers(self, depth: int | None = None) -> list[str]:
        """List the layer names the model accepts, in definition order.

        Args:
            depth: Keep only names at most this many levels deep, counting
                dots — ``1`` gives top-level blocks such as ``layer4``, ``2``
                descends one level into them. ``None`` lists every module.

        Returns:
            Hookable module names. The model itself is excluded.
        """
        names = [name for name, _ in self._model.named_modules() if name]
        if depth is None:
            return names
        return [name for name in names if name.count(".") < depth]

    def parameter_count(self) -> int:
        """Total number of parameters in the wrapped model."""
        return sum(p.numel() for p in self._model.parameters())

    def summary(
        self, layer_names: str | list[str] | None = None, depth: int = 3
    ) -> pd.DataFrame:
        """Tabulate the hookable layers as a :class:`pandas.DataFrame`.

        Useful before extracting: it shows which names are hookable, what kind
        of module each one is, and how many parameters each holds. Being a
        DataFrame, it renders as a table in a notebook and can be filtered or
        sorted like any other.

        Args:
            layer_names: Mark these layers in a ``selected`` column. Omit to
                list every layer without marking.
            depth: Nesting depth to list, as in :meth:`available_layers`.

        Returns:
            One row per hookable layer, indexed by layer name, with columns
            ``module`` (the class name), ``parameters``, and — when
            ``layer_names`` is given — ``selected``.

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :meth:`available_layers` instead.
        """
        import pandas as pd

        modules = dict(self._model.named_modules())
        names = self.available_layers(depth=depth)

        frame = pd.DataFrame(
            {
                "module": [type(modules[name]).__name__ for name in names],
                "parameters": [
                    sum(p.numel() for p in modules[name].parameters()) for name in names
                ],
            },
            index=pd.Index(names, name="layer"),
        )

        if layer_names is not None:
            selected = set(
                [layer_names] if isinstance(layer_names, str) else list(layer_names)
            )
            frame["selected"] = [name in selected for name in names]
        return frame

    def check_layers(self, layer_names: list[str]) -> None:
        """Fail before the first forward pass if a requested layer is absent.

        Args:
            layer_names: The layers about to be captured.

        Raises:
            ValueError: If any is not a module of the model. The message lists
                the closest available names, since the usual cause is a typo or
                the wrong nesting depth.
        """
        available = self.available_layers()
        missing = [name for name in layer_names if name not in available]
        if not missing:
            return

        headline = (
            f"Layer(s) not found in {type(self._model).__name__}: {', '.join(missing)}"
        )
        lines = [headline]
        for name in missing:
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            siblings = [
                candidate
                for candidate in available
                if (candidate.rsplit(".", 1)[0] if "." in candidate else "") == parent
            ]
            if siblings:
                where = f"under '{parent}'" if parent else "at the top level"
                lines.append(f"  did you mean, {where}: {', '.join(siblings[:8])}")
        lines.append(
            f"Call summary() or available_layers() to list all "
            f"{len(available)} hookable layers."
        )
        raise ValueError("\n".join(lines))

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

                        self._activations[layer_key] = (
                            tensor.half().detach().cpu().numpy()
                        )

                    return hook

                handles.append(
                    named_modules[name].register_forward_hook(make_hook(name))
                )

            yield

        finally:
            for handle in handles:
                handle.remove()
