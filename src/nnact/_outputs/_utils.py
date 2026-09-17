from __future__ import annotations

import torch


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
