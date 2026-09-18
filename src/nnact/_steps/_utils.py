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
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
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
    return tensor_to_numpy(prediction), tensor_to_numpy(top_probability)
