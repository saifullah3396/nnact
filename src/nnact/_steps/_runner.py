from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, final

from ignite.engine import Engine
from ignite.handlers import Timer


@final
class ActivationStepRunner:
    def __init__(self, step: Callable[[Engine, Mapping[str, Any]], Any]) -> None:
        self._step = step

    def _create_engine(self, *, progress: bool) -> tuple[Engine, Timer]:
        """Build the Ignite engine and attach run-level handlers."""
        engine = Engine(self._step)
        timer = Timer(average=False).attach(engine)

        if progress:
            from ignite.contrib.handlers import ProgressBar

            ProgressBar(desc="activations").attach(engine)

        return engine, timer

    def run(self, loader: Any, progress: bool = True) -> tuple[Engine, Timer]:
        engine, timer = self._create_engine(progress=progress)
        engine.run(loader, max_epochs=1)
        return engine, timer
