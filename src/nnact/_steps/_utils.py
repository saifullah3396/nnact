import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from nnact._model._hooked import tensor_to_numpy


def _assert_shape(
    name: str,
    tensor: torch.Tensor | None,
    shape: tuple[int | None, ...],
) -> None:
    """Assert ``tensor``'s shape matches ``shape`` exactly, dimension for dimension.

    A no-op when ``tensor`` is ``None``.

    Args:
        name: Label used in the raised message.
        tensor: The tensor to check, or ``None`` to skip.
        shape: Expected shape. ``None`` in any position accepts any size
            there.

    Raises:
        AssertionError: If ``tensor`` is not ``None`` and its shape doesn't
            match.
    """
    if tensor is None:
        return

    assert tensor.ndim == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(tensor.shape, shape)
    ), f"{name} must have shape {shape}, got {tuple(tensor.shape)}"


def _move_tensors(value: Any, device: torch.device | str) -> Any:
    """Move tensor leaves while preserving the caller's batch structure."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _move_tensors(
                    value=getattr(value, field.name), device=device
                )
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {
            key: _move_tensors(value=item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_tensors(value=item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(value=item, device=device) for item in value]
    return value


def _top_prediction(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a ``(..., vocab_size)`` logits tensor to its argmax and top softmax
    probability, discarding the vocab dimension.

    Logits are the single largest tensor a causal LM produces -- ``vocab_size``
    is commonly two orders of magnitude larger than the hidden size probing
    actually needs, so nothing here keeps the full tensor around once this
    returns.
    """
    top_probability, prediction = torch.softmax(logits, dim=-1).max(dim=-1)
    return tensor_to_numpy(tensor=prediction), tensor_to_numpy(tensor=top_probability)
