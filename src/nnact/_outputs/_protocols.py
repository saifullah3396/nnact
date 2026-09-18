from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

import torch
from torch.utils.data import default_collate


class ModelOutput(Protocol):
    logits: torch.Tensor
    loss: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class SequenceModelInputBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


@dataclass(frozen=True, kw_only=True)
class SequenceModelInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationSample:
    model_input: SequenceModelInput
    activation_labels: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationBatch:
    model_input: SequenceModelInputBatch
    activation_labels: list[list[str]] | None = None

    @classmethod
    def from_samples(cls, samples: list[ActivationSample]) -> ActivationBatch:
        model_inputs = [sample.model_input for sample in samples]
        return cls(
            model_input=SequenceModelInputBatch(
                input_ids=default_collate(
                    [model_input.input_ids for model_input in model_inputs]
                ),
                attention_mask=default_collate(
                    [model_input.attention_mask for model_input in model_inputs]
                ),
                token_type_ids=(
                    default_collate(
                        [model_input.token_type_ids for model_input in model_inputs]
                    )
                    if all(
                        model_input.token_type_ids is not None
                        for model_input in model_inputs
                    )
                    else None
                ),
                position_ids=(
                    default_collate(
                        [model_input.position_ids for model_input in model_inputs]
                    )
                    if all(
                        model_input.position_ids is not None
                        for model_input in model_inputs
                    )
                    else None
                ),
            ),
            activation_labels=(
                [sample.activation_labels for sample in samples]
                if all(sample.activation_labels is not None for sample in samples)
                else None
            ),
        )
