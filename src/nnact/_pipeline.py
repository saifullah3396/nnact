from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, final

from torch import nn

from nnact._model._hooked import HookedModel
from nnact._steps._factory import ActivationStepFactory
from nnact._steps._runner import ActivationStepRunner


@final
class ActivationPipeline:
    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        device: Any | None = None,
    ) -> None:
        hooked_model = HookedModel(model=model)
        names = [layer_names] if isinstance(layer_names, str) else list(layer_names)
        step = ActivationStepFactory().resolve(model=model)
        self._runner = ActivationStepRunner(
            step=step(hooked_model=hooked_model, layer_names=names, device=device)
        )

    def run(self, loader: Iterable[Mapping[str, Any]], progress: bool = True):
        return self._runner.run(loader, progress=progress)
