from __future__ import annotations

from typing import final

from ignite.engine import Engine, Events

from nnact._outputs._dataset import (
    ActivationDataset,
    SequenceActivationDataset,
    TokenActivationDataset,
)
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


@final
class ActivationAccumulator:
    """Ignite handler that builds an :class:`ActivationDataset` as batches complete.

    Attach with :meth:`attach`, run the pipeline, then read :attr:`dataset`.
    The dataset's concrete type is picked from the first batch's output and
    every later batch must match it.

    Example:
        >>> accumulator = ActivationAccumulator()
        >>> engine, timer = pipeline.run(loader, handlers=[accumulator])  # doctest: +SKIP
        >>> accumulator.dataset.activations["fc1"].shape  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._dataset: ActivationDataset | None = None

    @property
    def dataset(self) -> ActivationDataset:
        if self._dataset is None:
            raise RuntimeError("No batches were accumulated yet.")
        return self._dataset

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        output = engine.state.output
        if isinstance(output, SequenceActivationOutput):
            if self._dataset is None:
                self._dataset = SequenceActivationDataset()
            assert isinstance(self._dataset, SequenceActivationDataset), (
                "Pipeline output kind changed mid-run."
            )
            self._dataset._add_batch(output)
        elif isinstance(output, TokenActivationOutput):
            if self._dataset is None:
                self._dataset = TokenActivationDataset()
            assert isinstance(self._dataset, TokenActivationDataset), (
                "Pipeline output kind changed mid-run."
            )
            self._dataset._add_batch(output)
        else:
            raise TypeError(f"Unsupported activation output type: {type(output)}")
