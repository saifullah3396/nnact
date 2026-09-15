"""Activation tensors used as test input."""

import pytest
import torch

from tests.fixtures.types import ActivationFactory


@pytest.fixture
def identifiable() -> ActivationFactory:
    """Build a tensor whose row ``i`` is filled with the value ``start + i``.

    Ordering assertions then report a concrete index mismatch — ``expected 3,
    got 7`` — instead of an opaque comparison between random tensors. This is
    what lets a single test cover write ordering across batches.
    """

    def _make(n: int, dim: int = 4, start: int = 0) -> torch.Tensor:
        values = torch.arange(start, start + n, dtype=torch.float32)
        return values[:, None].expand(n, dim).contiguous()

    return _make
