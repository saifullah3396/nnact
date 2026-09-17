from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final, override

import torch
from torch.utils.data import Dataset

from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput

if TYPE_CHECKING:
    import pandas as pd


class ActivationDataset(Dataset, ABC):
    """Base class for positional, read-only access to accumulated activations.

    Concrete subclasses hold either :class:`SequenceActivationOutput` or
    :class:`TokenActivationOutput` samples, built up in memory by an
    :class:`~nnact._steps._accumulator.ActivationAccumulator` as an
    :class:`~nnact._pipeline.ActivationPipeline` run completes.
    """

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        """Names of the layers held, in first-batch order."""

    @property
    @abstractmethod
    def activations(self) -> dict[str, torch.Tensor]:
        """Every layer's accumulated activations, stacked across samples."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples held."""

    def layer_shape(self, layer_name: str) -> tuple[int, ...]:
        """Per-sample activation shape for one layer, excluding the sample axis."""
        return tuple(self.activations[layer_name].shape[1:])

    def summary(self) -> pd.DataFrame:
        """Tabulate the accumulated layers as a :class:`pandas.DataFrame`.

        Returns:
            One row per layer, indexed by layer name, with columns ``shape``
            (per sample) and ``bytes`` (for every sample held, as float32).

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :attr:`layer_names` and :attr:`activations`
                instead.
        """
        import numpy as np
        import pandas as pd

        names = self.layer_names
        shapes = [self.layer_shape(name) for name in names]
        elements = [int(np.prod(shape, dtype=np.int64)) for shape in shapes]

        return pd.DataFrame(
            {
                "shape": shapes,
                "bytes": [count * len(self) * 4 for count in elements],
            },
            index=pd.Index(names, name="layer"),
        )


@final
class InMemorySequenceActivationDataset(ActivationDataset):
    """Activations accumulated in memory from :class:`SequenceActivationOutput` batches."""

    def __init__(self) -> None:
        self._logits: list[torch.Tensor] = []
        self._losses: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._activations: dict[str, list[torch.Tensor]] = {}

    def _add_batch(self, output: SequenceActivationOutput) -> None:
        self._logits.append(output.logits)
        if output.loss is not None:
            self._losses.append(output.loss)
        if output.labels is not None:
            self._labels.append(output.labels)
        for name, tensor in output.activations.items():
            self._activations.setdefault(name, []).append(tensor)

    @property
    @override
    def layer_names(self) -> list[str]:
        return list(self._activations)

    @property
    def logits(self) -> torch.Tensor:
        return torch.cat(self._logits, dim=0)

    @property
    def loss(self) -> torch.Tensor | None:
        return torch.cat(self._losses, dim=0) if self._losses else None

    @property
    def labels(self) -> torch.Tensor | None:
        return torch.cat(self._labels, dim=0) if self._labels else None

    @property
    @override
    def activations(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.cat(tensors, dim=0) for name, tensors in self._activations.items()
        }

    @override
    def __len__(self) -> int:
        return sum(logits.shape[0] for logits in self._logits)

    def __getitem__(self, idx: int) -> SequenceActivationOutput:
        activations = self.activations
        loss, labels = self.loss, self.labels
        return SequenceActivationOutput(
            logits=self.logits[idx].unsqueeze(0),
            loss=None if loss is None else loss[idx].unsqueeze(0),
            labels=None if labels is None else labels[idx].unsqueeze(0),
            activations={
                name: tensor[idx].unsqueeze(0) for name, tensor in activations.items()
            },
        )


@final
class InMemoryTokenActivationDataset(ActivationDataset):
    """Activations accumulated in memory from :class:`TokenActivationOutput` batches."""

    def __init__(self) -> None:
        self._offsets: list[torch.Tensor] = [torch.zeros(1, dtype=torch.long)]
        self._logits: list[torch.Tensor] = []
        self._losses: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._activations: dict[str, list[torch.Tensor]] = {}

    def _add_batch(self, output: TokenActivationOutput) -> None:
        base = self._offsets[-1][-1]
        self._offsets.append(output.offsets[1:] + base)
        self._logits.append(output.logits)
        if output.loss is not None:
            self._losses.append(output.loss)
        if output.labels is not None:
            self._labels.append(output.labels)
        for name, tensor in output.activations.items():
            self._activations.setdefault(name, []).append(tensor)

    @property
    @override
    def layer_names(self) -> list[str]:
        return list(self._activations)

    @property
    def offsets(self) -> torch.Tensor:
        return torch.cat(self._offsets, dim=0)

    @property
    def logits(self) -> torch.Tensor:
        return torch.cat(self._logits, dim=0)

    @property
    def loss(self) -> torch.Tensor | None:
        return torch.cat(self._losses, dim=0) if self._losses else None

    @property
    def labels(self) -> torch.Tensor | None:
        return torch.cat(self._labels, dim=0) if self._labels else None

    @property
    @override
    def activations(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.cat(tensors, dim=0) for name, tensors in self._activations.items()
        }

    @override
    def __len__(self) -> int:
        return max(self.offsets.numel() - 1, 0)

    def __getitem__(self, idx: int) -> TokenActivationOutput:
        offsets = self.offsets
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        activations = self.activations
        loss, labels = self.loss, self.labels
        return TokenActivationOutput(
            logits=self.logits[start:end],
            offsets=torch.tensor([0, end - start], dtype=torch.long),
            loss=None if loss is None else loss[start:end],
            labels=None if labels is None else labels[start:end],
            activations={name: tensor[start:end] for name, tensor in activations.items()},
        )
