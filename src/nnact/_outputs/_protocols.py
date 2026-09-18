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
    """A batch of pre-tokenized model inputs, as an activation step passes it to the model."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """This batch's set fields, ready to pass as ``model(**kwargs)``.

        Filters out unset optional fields (e.g. ``token_type_ids``), so a
        step never hardcodes which fields exist -- adding a new common field
        to this class needs no change at any call site.
        """
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


@dataclass(frozen=True, kw_only=True)
class SequenceModelInput:
    """One unbatched pre-tokenized example, stacked by :meth:`ActivationBatch.from_samples`."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationSample:
    """One unbatched :class:`SequenceModelInput`, paired with its ground-truth labels."""

    model_input: SequenceModelInput
    activation_labels: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationBatch:
    """A batched :class:`SequenceModelInput`, paired with the probe's ground-truth labels.

    ``activation_labels`` is never a model input -- it never reaches
    ``self._hooked_model(...)`` -- so it lives here rather than on
    :class:`SequenceModelInputBatch` itself.
    """

    model_input: SequenceModelInputBatch
    activation_labels: torch.Tensor | None = None

    @classmethod
    def from_samples(cls, samples: list[ActivationSample]) -> ActivationBatch:
        """Stack a list of :class:`ActivationSample` into one batch."""
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
                default_collate(
                    [sample.activation_labels for sample in samples]
                )
                if all(sample.activation_labels is not None for sample in samples)
                else None
            ),
        )
