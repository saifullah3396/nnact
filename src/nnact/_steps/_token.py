from __future__ import annotations

import numpy as np
import torch
from ignite.engine import Engine
from transformers import PreTrainedTokenizerBase

from nnact._model._hooked import HookedModel, tensor_to_numpy
from nnact._outputs._protocols import ActivationBatch, ModelOutput
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput
from nnact._steps._utils import _move_tensors, _top_prediction


class TokenActivationStep:
    def __init__(
        self,
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str = "cpu",
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self._hooked_model = hooked_model
        self._layer_names = layer_names
        self._device = device
        self._tokenizer = tokenizer
        self._hooked_model.check_layers(self._layer_names)

    def _prepare_model(self):
        self._hooked_model = self._hooked_model.to(self._device).eval()

    @torch.no_grad()
    def __call__(
        self,
        engine: Engine,
        batch: ActivationBatch,
    ) -> TokenActivationOutput:
        assert isinstance(batch, ActivationBatch), (
            "batch passed to the generator must be an ActivationBatch."
        )

        if self._device is not None:
            batch = _move_tensors(batch, self._device)

        token_ids = batch.model_input.input_ids
        attention_mask = batch.model_input.attention_mask

        with self._hooked_model.capture(self._layer_names):
            raw_output: ModelOutput = self._hooked_model(
                **batch.model_input.as_model_kwargs()
            )

            activations = {
                name: self._hooked_model.get_activation(name)
                for name in self._layer_names
            }

        prediction, top_probability = _top_prediction(raw_output.logits)

        sequence_output = SequenceActivationOutput(
            prediction=prediction,
            top_probability=top_probability,
            loss=(
                tensor_to_numpy(raw_output.loss)
                if raw_output.loss is not None
                else None
            ),
            activations=activations,
        )

        tokens = None
        if self._tokenizer is not None:
            tokens = [
                self._tokenizer.convert_ids_to_tokens(ids) for ids in token_ids.tolist()
            ]
            assert isinstance(tokens, list), f"List[str] expected, got {type(tokens)}"

        return TokenActivationOutput.from_sequence(
            sequence_output,
            mask=tensor_to_numpy(attention_mask),
            labels=(
                tensor_to_numpy(batch.activation_labels)
                if batch.activation_labels is not None
                else None
            ),
            token_ids=tensor_to_numpy(token_ids),
            tokens=np.asarray(tokens) if tokens is not None else None,
        )
