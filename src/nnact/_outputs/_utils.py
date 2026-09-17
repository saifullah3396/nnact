from __future__ import annotations

from typing import Any

import numpy as np


def _assert_shape(
    name: str,
    tensor: Any | None,
    shape: tuple[int | None, ...],
) -> None:
    if tensor is None:
        return

    assert tensor.ndim == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(tensor.shape, shape)
    ), f"{name} must have shape {shape}, got {tuple(tensor.shape)}"


def _softmax(array: np.ndarray, axis: int) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
