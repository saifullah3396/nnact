from collections.abc import Mapping

import torch


def _as_list(layer_names: str | list[str]) -> list[str]:
    """Accept a single layer name or a list of them."""
    return [layer_names] if isinstance(layer_names, str) else list(layer_names)


def _move_tensors(value: object, device: torch.device | str) -> object:
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
