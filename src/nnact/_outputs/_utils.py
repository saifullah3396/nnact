from __future__ import annotations

from typing import Any

import numpy as np


def _assert_shape(
    name: str,
    tensor: Any | None,
    shape: tuple[int | None, ...],
) -> None:
    """Assert ``tensor``'s shape matches ``shape`` exactly, dimension for dimension.

    A no-op when ``tensor`` is ``None`` -- callers use this to validate
    fields that are themselves optional.

    Args:
        name: Label used in the raised message, e.g. the field name.
        tensor: The array (or tensor) to check, or ``None`` to skip.
        shape: Expected shape. ``None`` in any position accepts any size
            there; the number of positions must still match ``tensor.ndim``.

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


def _assert_leading_shape(
    name: str,
    tensor: Any,
    shape: tuple[int | None, ...],
) -> None:
    """Assert only that ``tensor``'s first ``len(shape)`` dims match.

    Any further trailing dims (a layer's own feature shape) are unconstrained
    in both count and size.

    Args:
        name: Label used in the raised message, e.g. the field name.
        tensor: The array (or tensor) to check. Unlike :func:`_assert_shape`,
            this is never optional -- callers only use this for fields that
            are always present.
        shape: Expected leading shape. ``None`` in any position accepts any
            size there.

    Raises:
        AssertionError: If ``tensor``'s leading dimensions don't match.
    """
    assert tensor.ndim >= len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(tensor.shape, shape)
    ), f"{name} must start with shape {shape}, got {tuple(tensor.shape)}"


def _softmax(array: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable softmax along ``axis``.

    Args:
        array: Input values.
        axis: Axis to normalize over.

    Returns:
        An array the same shape as ``array``, summing to 1 along ``axis``.
    """
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
