from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, final

import torch
from ignite.engine import Engine
from ignite.handlers import Timer
from ignite.handlers.tqdm_logger import ProgressBar
from transformers import PreTrainedTokenizerBase

from nnact._model._hooked import HookedModel
from nnact._steps._accumulator import ActivationAccumulator
from nnact._steps._sequence import SequenceActivationStep
from nnact._steps._token import TokenActivationStep


@final
class ActivationStepRunner:
    """Drives an Ignite engine over a loader, one step call per batch.

    Builds the right step (:class:`~nnact._steps._sequence.SequenceActivationStep`
    or :class:`~nnact._steps._token.TokenActivationStep`) for ``output_type``,
    wraps it in an ``ignite.engine.Engine``, and attaches every handler in
    ``handlers`` plus an optional progress bar.
    """

    def __init__(
        self,
        output_type: Literal["sequence", "token"],
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str = "cpu",
        tokenizer: PreTrainedTokenizerBase | None = None,
        handlers: Iterable[ActivationAccumulator] = (),
        show_progress: bool = True,
    ) -> None:
        """Build the step and the engine that will run it.

        Args:
            output_type: ``"token"`` or ``"sequence"`` -- selects which
                step class to build.
            hooked_model: The model to run, already wrapped for activation
                capture.
            layer_names: Layers to capture on every batch.
            device: Device to run the model on.
            tokenizer: Passed through to a ``"token"`` step to decode token
                strings; ignored for ``"sequence"``.
            handlers: Attached to the engine so each one sees every
                completed batch -- typically an
                :class:`~nnact._steps._accumulator.ActivationAccumulator`.
            show_progress: Whether to attach a console progress bar.
        """
        self._step = self._build_step(
            output_type=output_type,
            hooked_model=hooked_model,
            layer_names=layer_names,
            device=device,
            tokenizer=tokenizer,
        )
        self._engine, self._timer = self._create_engine(
            handlers=handlers,
            show_progress=show_progress,
        )

    def _build_step(
        self,
        *,
        output_type: Literal["sequence", "token"],
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str = "cpu",
        tokenizer: PreTrainedTokenizerBase | None,
    ) -> SequenceActivationStep | TokenActivationStep:
        """Construct the step matching ``output_type``.

        Args:
            output_type: See :meth:`__init__`.
            hooked_model: See :meth:`__init__`.
            layer_names: See :meth:`__init__`.
            device: See :meth:`__init__`.
            tokenizer: See :meth:`__init__`.

        Returns:
            A ``SequenceActivationStep`` for ``"sequence"``, or a
            ``TokenActivationStep`` for ``"token"``.

        Raises:
            AssertionError: If ``output_type`` isn't ``"sequence"`` or
                ``"token"``.
        """
        assert output_type in ("sequence", "token"), (
            f"Unknown output_type '{output_type}', expected 'sequence' or 'token'."
        )

        if output_type == "sequence":
            return SequenceActivationStep(
                hooked_model=hooked_model, layer_names=layer_names, device=device
            )
        return TokenActivationStep(
            hooked_model=hooked_model,
            layer_names=layer_names,
            device=device,
            tokenizer=tokenizer,
        )

    def _create_engine(
        self,
        *,
        handlers: Iterable[ActivationAccumulator],
        show_progress: bool,
    ) -> tuple[Engine, Timer]:
        """Build the Ignite engine and attach run-level handlers.

        Args:
            handlers: See :meth:`__init__`.
            show_progress: See :meth:`__init__`.

        Returns:
            The engine (with every handler and, if requested, a progress
            bar attached) and a ``Timer`` measuring total run time.
        """
        engine = Engine(self._step)
        timer = Timer(average=False).attach(engine)

        for handler in handlers:
            handler.attach(engine=engine)

        if show_progress:
            ProgressBar(desc="activations").attach(engine)

        return engine, timer

    def run(self, loader: Any) -> tuple[Engine, Timer]:
        """Prepare the model and run one epoch over ``loader``.

        Args:
            loader: Yields batches for the step to consume -- typically a
                ``torch.utils.data.DataLoader`` built by
                :func:`~nnact._pipeline.activation_loader`.

        Returns:
            The engine that ran (its handlers have already seen every
            batch by the time this returns) and the run's ``Timer``.
        """
        self._step._prepare_model()
        self._engine.run(loader, max_epochs=1)
        return self._engine, self._timer
