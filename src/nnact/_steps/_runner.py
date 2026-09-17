from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, final

from ignite.engine import Engine
from ignite.handlers import Timer

from nnact._model._hooked import HookedModel
from nnact._steps._accumulator import ActivationAccumulator
from nnact._steps._sequence import SequenceActivationStep
from nnact._steps._token import TokenActivationStep


@final
class ActivationStepRunner:
    def __init__(
        self,
        output_type: Literal["sequence", "token"],
        hooked_model: HookedModel,
        layer_names: list[str],
        device: Any | None = None,
        handlers: Iterable[ActivationAccumulator] = (),
        show_progress: bool = True,
    ) -> None:
        self._step = self._build_step(
            output_type=output_type,
            hooked_model=hooked_model,
            layer_names=layer_names,
            device=device,
        )
        self._engine, self._timer = self._create_engine(
            handlers=handlers, show_progress=show_progress
        )

    def _build_step(
        self,
        *,
        output_type: Literal["sequence", "token"],
        hooked_model: HookedModel,
        layer_names: list[str],
        device: Any | None,
    ) -> SequenceActivationStep | TokenActivationStep:
        assert output_type in ("sequence", "token"), (
            f"Unknown output_type '{output_type}', expected 'sequence' or 'token'."
        )

        if output_type == "sequence":
            return SequenceActivationStep(
                hooked_model=hooked_model, layer_names=layer_names, device=device
            )
        return TokenActivationStep(
            hooked_model=hooked_model, layer_names=layer_names, device=device
        )

    def _create_engine(
        self, *, handlers: Iterable[ActivationAccumulator], show_progress: bool
    ) -> tuple[Engine, Timer]:
        """Build the Ignite engine and attach run-level handlers."""
        engine = Engine(self._step)
        timer = Timer(average=False).attach(engine)

        for handler in handlers:
            handler.attach(engine)

        if show_progress:
            from ignite.contrib.handlers import ProgressBar

            ProgressBar(desc="activations").attach(engine)

        return engine, timer

    def run(self, loader: Any) -> tuple[Engine, Timer]:
        self._engine.run(loader, max_epochs=1)
        return self._engine, self._timer
