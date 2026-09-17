from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from ignite.engine import Engine
from transformers import PreTrainedTokenizerBase

from nnact._model._hooked import HookedModel
from nnact._outputs._protocols import ModelOutput
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput
from nnact._steps._utils import _move_tensors


class TokenActivationStep:
    def __init__(
        self,
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self._hooked_model = hooked_model
        self._layer_names = layer_names
        self._device = device
        self._tokenizer = tokenizer
        self._hooked_model.check_layers(self._layer_names)

    @torch.no_grad()
    def __call__(
        self,
        engine: Engine,
        batch: Mapping[str, Any],
    ) -> TokenActivationOutput:
        assert isinstance(batch, Mapping), (
            "batch passed to the generator must be a mapping."
        )

        if self._device is not None:
            batch = _move_tensors(batch, self._device)

        assert "attention_mask" in batch, "batch must contain an attention_mask"
        mask = batch["attention_mask"]
        assert isinstance(mask, torch.Tensor), "attention_mask must be a torch.Tensor"

        with self._hooked_model.capture(self._layer_names):
            raw_output: ModelOutput = self._hooked_model(**batch)

            activations = {
                name: self._hooked_model.get_activation(name)
                for name in self._layer_names
            }

        sequence_output = SequenceActivationOutput(
            logits=raw_output.logits,
            loss=raw_output.loss,
            activations=activations,
        )

        tokens = None
        if self._tokenizer is not None:
            token_ids = batch["token_ids"]
            tokens = self._tokenizer.convert_ids_to_tokens(
                token_ids[mask.bool()].tolist()
            )
            assert isinstance(tokens, list), f"List[str] expected, got {type(tokens)}"

        return TokenActivationOutput.from_sequence(
            sequence_output, mask=mask, labels=batch.get("labels"), token_ids=token_ids
        )
