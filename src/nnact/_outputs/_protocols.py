from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

import torch
from torch.utils.data import default_collate


class ModelOutput(Protocol):
    """Structural type for whatever a wrapped model's forward pass returns.

    Any object with these two attributes satisfies this protocol -- most
    Hugging Face ``transformers`` model outputs already do, with no explicit
    subclassing needed.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class SequenceModelInputBatch:
    """A batch of pre-tokenized model inputs, ready to pass to a model's forward pass.

    Built by :meth:`ActivationBatch.from_samples`, one per batch; never
    constructed directly by user code.

    Attributes:
        input_ids: Token ids, shape ``(batch_size, sequence_length)``.
        attention_mask: 1 at real tokens, 0 at padding, same shape as
            ``input_ids``.
        token_type_ids: Segment ids, for models that use them (e.g. BERT's
            sentence-pair encoding). ``None`` if unused.
        position_ids: Explicit position overrides, for models or callers
            that need them (e.g. to correct for left-padding). ``None`` to
            let the model compute its own default positions.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """This batch's set fields, ready to pass as ``model(**kwargs)``.

        Returns:
            A dict with one entry per field that isn't ``None`` -- so an
            unused optional field (e.g. ``token_type_ids`` for a model that
            doesn't take one) is simply absent, rather than passed as
            ``None``.
        """
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


@dataclass(frozen=True, kw_only=True)
class SequenceModelInput:
    """One unbatched pre-tokenized example -- the per-sample counterpart of
    :class:`SequenceModelInputBatch`.

    A dataset's ``__getitem__`` returns this (wrapped in an
    :class:`ActivationSample`); :meth:`ActivationBatch.from_samples` stacks
    a list of these into one :class:`SequenceModelInputBatch`.

    Attributes:
        input_ids: Token ids, shape ``(sequence_length,)``.
        attention_mask: 1 at real tokens, 0 at padding, same shape as
            ``input_ids``.
        token_type_ids: Segment ids, for models that use them. ``None`` if
            unused.
        position_ids: Explicit position overrides. ``None`` to let the
            model compute its own default positions.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationSample:
    """One dataset item: a model input, paired with its ground-truth labels.

    This is the type a dataset feeding
    :meth:`~nnact._pipeline.ActivationPipeline.run` must yield from
    ``__getitem__``.

    Attributes:
        model_input: The pre-tokenized example to run through the model.
        activation_labels: One role/class name per token, for a
            token-level run, or a single name for the whole sample, for a
            sequence-level run. ``None`` if this run has no ground truth to
            attach (e.g. plain activation extraction with nothing to
            probe against). Never passed to the model itself -- see
            :class:`SequenceModelInput`.
    """

    model_input: SequenceModelInput
    activation_labels: list[str] | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationBatch:
    """A batch of :class:`ActivationSample`, as an activation step consumes it.

    Attributes:
        model_input: The batch's model inputs, ready for
            ``model(**model_input.as_model_kwargs())``.
        activation_labels: One entry per sample, in batch order, each the
            same shape as that sample's own ``activation_labels``. Kept as
            a plain list rather than stacked into an array, since it holds
            strings -- ``None`` unless every sample in the batch has labels.
    """

    model_input: SequenceModelInputBatch
    activation_labels: list[list[str]] | None = None

    @classmethod
    def from_samples(cls, samples: list[ActivationSample]) -> ActivationBatch:
        """Stack a list of :class:`ActivationSample` into one batch.

        Args:
            samples: The samples to combine, in the order they should
                appear in the batch. All must share the same fields set
                (e.g. all have ``token_type_ids`` or none do) -- an
                optional field is included in the result only when every
                sample in ``samples`` has it.

        Returns:
            One :class:`ActivationBatch` with each tensor field stacked
            along a new leading batch dimension, and ``activation_labels``
            collected into a plain list (not stacked, since it holds
            strings).
        """
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
