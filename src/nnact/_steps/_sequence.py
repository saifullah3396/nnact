from __future__ import annotations

import numpy as np
import torch
from ignite.engine import Engine

from nnact._model._hooked import HookedModel, tensor_to_numpy
from nnact._outputs._protocols import ActivationBatch, ModelOutput
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._steps._utils import _move_tensors, _top_prediction


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

    def _prepare_model(self):
        self._hooked_model = self._hooked_model.to(self._device).eval()

    @torch.no_grad()
    def __call__(
        self,
        engine: Engine,
        batch: ActivationBatch,
    ) -> SequenceActivationOutput:
        assert isinstance(batch, ActivationBatch), (
            "batch passed to the generator must be an ActivationBatch."
        )

        if self._device is not None:
            batch = _move_tensors(batch, self._device)

        with self._hooked_model.capture(self._layer_names):
            raw_output: ModelOutput = self._hooked_model(
                **batch.model_input.as_model_kwargs()
            )

            activations = {
                name: self._hooked_model.get_activation(name)
                for name in self._layer_names
            }

        prediction, top_probability = _top_prediction(raw_output.logits)

        return SequenceActivationOutput(
            prediction=prediction,
            top_probability=top_probability,
            loss=(
                tensor_to_numpy(raw_output.loss)
                if raw_output.loss is not None
                else None
            ),
            labels=(
                np.asarray(batch.activation_labels)
                if batch.activation_labels is not None
                else None
            ),
            activations=activations,
        )
