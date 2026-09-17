from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
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

        assert "input_ids" in batch, "batch must contain an input_ids"
        assert "attention_mask" in batch, "batch must contain an attention_mask"
        token_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        assert isinstance(token_ids, torch.Tensor), "token_ids must be a torch.Tensor"
        assert isinstance(attention_mask, torch.Tensor), (
            "attention_mask must be a torch.Tensor"
        )

        with self._hooked_model.capture(self._layer_names):
            raw_output: ModelOutput = self._hooked_model(**batch)

            activations = {
                name: self._hooked_model.get_activation(name)
                for name in self._layer_names
            }

        sequence_output = SequenceActivationOutput(
            logits=raw_output.logits.half().detach().cpu().numpy(),
            loss=(
                raw_output.loss.half().detach().cpu().numpy()
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

        labels = batch.get("labels")
        return TokenActivationOutput.from_sequence(
            sequence_output,
            mask=attention_mask.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy() if labels is not None else None,
            token_ids=token_ids.detach().cpu().numpy(),
            tokens=np.asarray(tokens),
        )
