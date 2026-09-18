from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, override

from ignite.engine import Engine, Events

from nnact._outputs._dataset import (
    ActivationDataset,
    InMemorySequenceActivationDataset,
    InMemoryTokenActivationDataset,
)
from nnact._outputs._h5_dataset import (
    H5SequenceActivationDataset,
    H5TokenActivationDataset,
)
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


class ActivationAccumulator(ABC):
    """Ignite handler that builds an :class:`ActivationDataset` as batches complete.

    Attach with :meth:`attach`, run the pipeline, then read :attr:`dataset`.
    :class:`~nnact._pipeline.ActivationPipeline` creates and attaches one of
    these itself; user code reads :attr:`dataset` from the
    :class:`~nnact._pipeline.RunResult` that
    :meth:`~nnact._pipeline.ActivationPipeline.run` returns, rather than
    constructing an accumulator directly.
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
        """Register this accumulator's callbacks on ``engine``.

        Args:
            engine: The Ignite engine running the activation steps.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.add_event_handler(Events.COMPLETED, self._on_completed)
        engine.add_event_handler(Events.EXCEPTION_RAISED, self._on_exception)

    def __call__(self, engine: Engine) -> None:
        """Ignite's ``ITERATION_COMPLETED`` callback: accumulate the latest output.

        Args:
            engine: The running engine; ``engine.state.output`` is this
                iteration's step result.

        Raises:
            AssertionError: If ``engine.state.output`` isn't a
                ``SequenceActivationOutput`` or ``TokenActivationOutput``.
        """
        output = engine.state.output
        assert isinstance(output, (SequenceActivationOutput, TokenActivationOutput)), (
            f"Expected a SequenceActivationOutput or TokenActivationOutput, "
            f"got {type(output).__name__}."
        )
        self._add_batch(output)

    def _on_completed(self, engine: Engine) -> None:
        """Called once the run finishes successfully. No-op unless overridden."""

    def _on_exception(self, engine: Engine, exc: BaseException) -> None:
        """Called if the run fails. Re-raises by default; override to clean up first."""
        raise exc


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


class _H5ActivationAccumulator(ActivationAccumulator):
    """Shared run-cleanup for the HDF5-backed accumulators.

    The backing file is removed if the run raises, so a failed run never
    leaves a partial ``.h5`` file behind.
    """

    @property
    @override
    @abstractmethod
    def dataset(self) -> H5SequenceActivationDataset | H5TokenActivationDataset: ...

    @override
    def _on_exception(self, engine: Engine, exc: BaseException) -> None:
        self.dataset.close(delete=True)
        raise exc


@final
class H5SequenceActivationAccumulator(_H5ActivationAccumulator):
    """Streams :class:`SequenceActivationOutput` batches to an HDF5 file."""

    def __init__(self, path: str | Path) -> None:
        self._dataset = H5SequenceActivationDataset(path=path)

    @property
    @override
    def dataset(self) -> H5SequenceActivationDataset:
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
class H5TokenActivationAccumulator(_H5ActivationAccumulator):
    """Streams :class:`TokenActivationOutput` batches to an HDF5 file."""

    def __init__(self, path: str | Path) -> None:
        self._dataset = H5TokenActivationDataset(path=path)

    @property
    @override
    def dataset(self) -> H5TokenActivationDataset:
        return self._dataset

    @override
    def _add_batch(
        self, output: SequenceActivationOutput | TokenActivationOutput
    ) -> None:
        assert isinstance(output, TokenActivationOutput), (
            f"Expected TokenActivationOutput, got {type(output).__name__}."
        )
        self._dataset._add_batch(output)
