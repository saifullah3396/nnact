"""Test doubles: minimal models and datasets.

Each model here exists to exercise one specific behaviour of activation
capture, and is kept as small as that purpose allows.
"""

from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from nnact._types import Sample

DatasetFactory = Callable[..., "ListDataset"]
"""Builds a dataset: ``(n=5, id_prefix="sample") -> ListDataset``."""


class TinyMLP(nn.Module):
    """Two-layer MLP with named layers to hook.

    The default model for tests that need a forward pass but do not care what
    the model does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _TupleLinear(nn.Module):
    """Linear layer returning a tuple, as recurrent modules do."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(x)
        return out, torch.zeros_like(out)


class TupleOutMLP(nn.Module):
    """Model whose hooked layer returns a tuple rather than a tensor.

    Exercises the branch in ``HookedModel`` that takes ``output[0]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = _TupleLinear(4, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.body(x)
        return self.head(out)


class TrainModeProbe(nn.Module):
    """Model whose output reveals whether it ran in train or eval mode.

    Dropout with ``p=1.0`` zeroes everything while training and is a no-op in
    eval, so a non-zero activation proves the mapper switched to eval.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.drop = nn.Dropout(p=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc(x))


class ListDataset(Dataset[Sample]):
    """Minimal in-memory Dataset over pre-built samples."""

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@pytest.fixture
def tiny_model() -> TinyMLP:
    """A fresh :class:`TinyMLP`, so weights never leak between tests."""
    return TinyMLP()


@pytest.fixture
def tiny_dataset() -> DatasetFactory:
    """Build a dataset of ``n`` samples with deterministic, distinct data.

    Seeded so a failure reproduces on the next run.
    """

    def _make(n: int = 5, id_prefix: str = "sample") -> ListDataset:
        generator = torch.Generator().manual_seed(0)
        return ListDataset(
            [
                Sample(id=f"{id_prefix}_{i}", data=torch.randn(4, generator=generator))
                for i in range(n)
            ]
        )

    return _make
