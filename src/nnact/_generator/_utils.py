from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


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


def _describe_device(device: torch.device) -> str:
    """Return a readable description of a device."""
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return f"{torch.cuda.get_device_name(index)} (cuda:{index})"
    return str(device)


def _model_device(model: nn.Module, override: torch.device | str | None) -> str:
    """Describe the device the model's forward passes ran on."""
    try:
        return _describe_device(next(model.parameters()).device)
    except StopIteration:
        return _describe_device(torch.device(override)) if override else "cpu"


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


def _split_batch(batch: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Separate required sample IDs from keyword arguments for the model."""
    if "id" not in batch:
        raise ValueError('Each loader batch must contain an "id" field.')

    ids = batch["id"]
    if isinstance(ids, str) or not isinstance(ids, Iterable):
        raise TypeError('Batch "id" must be an iterable of sample ID strings.')

    ids = list(ids)
    if not all(isinstance(sample_id, str) for sample_id in ids):
        raise TypeError('Batch "id" must contain only strings.')

    model_inputs = {key: value for key, value in batch.items() if key != "id"}
    return ids, model_inputs
