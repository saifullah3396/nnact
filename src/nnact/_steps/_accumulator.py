from __future__ import annotations

from abc import ABC, abstractmethod
from typing import final, override

from ignite.engine import Engine, Events

from nnact._outputs._dataset import (
    ActivationDataset,
    InMemorySequenceActivationDataset,
    InMemoryTokenActivationDataset,
)
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


class ActivationAccumulator(ABC):
    """Ignite handler that builds an :class:`ActivationDataset` as batches complete.

    Attach with :meth:`attach`, run the pipeline, then read :attr:`dataset`.

    Example:
        >>> accumulator = InMemorySequenceActivationAccumulator()
        >>> engine, timer = pipeline.run(loader, handlers=[accumulator])  # doctest: +SKIP
        >>> accumulator.dataset.activations["fc1"].shape  # doctest: +SKIP
    """

    @property
    @abstractmethod
    def dataset(self) -> ActivationDataset:
        """The dataset built so far."""

    @abstractmethod
    def _add_batch(
        self, output: SequenceActivationOutput | TokenActivationOutput
    ) -> None:
        """Append one batch's output to the dataset being built."""

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        output = engine.state.output
        assert isinstance(output, (SequenceActivationOutput, TokenActivationOutput)), (
            f"Expected a SequenceActivationOutput or TokenActivationOutput, "
            f"got {type(output).__name__}."
        )
        self._add_batch(output)


@final
class InMemorySequenceActivationAccumulator(ActivationAccumulator):
    """Accumulates :class:`SequenceActivationOutput` batches in memory."""

    def __init__(self) -> None:
        self._dataset = InMemorySequenceActivationDataset()

    @property
    @override
    def dataset(self) -> InMemorySequenceActivationDataset:
        return self._dataset

    @override
    def _add_batch(
        self, output: SequenceActivationOutput | TokenActivationOutput
    ) -> None:
        assert isinstance(output, SequenceActivationOutput), (
            f"Expected SequenceActivationOutput, got {type(output).__name__}."
        )
        self._dataset._add_batch(output)


@final
class InMemoryTokenActivationAccumulator(ActivationAccumulator):
    """Accumulates :class:`TokenActivationOutput` batches in memory."""

    def __init__(self) -> None:
        self._dataset = InMemoryTokenActivationDataset()

    @property
    @override
    def dataset(self) -> InMemoryTokenActivationDataset:
        return self._dataset

    @override
    def _add_batch(
        self, output: SequenceActivationOutput | TokenActivationOutput
    ) -> None:
        assert isinstance(output, TokenActivationOutput), (
            f"Expected TokenActivationOutput, got {type(output).__name__}."
        )
        self._dataset._add_batch(output)
