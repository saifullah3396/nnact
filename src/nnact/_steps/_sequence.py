from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ignite.engine import Engine
from nnact._generator._outputs._protocols import ModelOutput
from nnact._generator._outputs._sequence import SequenceActivationOutput
from nnact._generator._utils import _move_tensors

from nnact._model._hooked import HookedModel


class SequenceActivationStep:
    def __init__(
        self,
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str | None,
    ) -> None:
        self._hooked_model = hooked_model
        self._layer_names = layer_names
        self._device = device
        self._hooked_model.check_layers(self._layer_names)

    @torch.no_grad()
    def __call__(
        self,
        engine: Engine,
        batch: Mapping[str, Any],
    ) -> SequenceActivationOutput:
        assert isinstance(batch, Mapping), (
            "batch passed to the generator must be a mapping."
        )

        if self._device is not None:
            batch = _move_tensors(batch, self._device)

        with self._hooked_model.capture(self._layer_names):
            raw_output: ModelOutput = self._hooked_model(**batch)

            activations = {
                name: self._hooked_model.get_activation(name)
                for name in self._layer_names
            }

        return SequenceActivationOutput(
            logits=raw_output.logits,
            loss=raw_output.loss,
            labels=batch.get("labels"),
            activations=activations,
        )
