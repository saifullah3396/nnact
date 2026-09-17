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


def _assert_leading_shape(
    name: str,
    tensor: Any,
    shape: tuple[int | None, ...],
) -> None:
    """Assert only that ``tensor``'s first ``len(shape)`` dims match.

    Any further trailing dims (a layer's own feature shape) are unconstrained
    in both count and size.
    """
    assert tensor.ndim >= len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(tensor.shape, shape)
    ), f"{name} must start with shape {shape}, got {tuple(tensor.shape)}"


def _softmax(array: np.ndarray, axis: int) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
