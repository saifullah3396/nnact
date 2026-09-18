from __future__ import annotations

from collections.abc import Sequence
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


def _stack_model_input(
    model_inputs: list[SequenceModelInput],
) -> SequenceModelInputBatch:
    """Stack a list of :class:`SequenceModelInput` into one :class:`SequenceModelInputBatch`.

    Args:
        model_inputs: The per-sample model inputs to combine, in the order
            they should appear in the batch.

    Returns:
        Every tensor field stacked along a new leading batch dimension. An
        optional field (``token_type_ids``, ``position_ids``) is stacked
        only when every sample in ``model_inputs`` has it; otherwise it's
        ``None`` on the result.
    """
    return SequenceModelInputBatch(
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
                model_input.token_type_ids is not None for model_input in model_inputs
            )
            else None
        ),
        position_ids=(
            default_collate([model_input.position_ids for model_input in model_inputs])
            if all(model_input.position_ids is not None for model_input in model_inputs)
            else None
        ),
    )


@dataclass(frozen=True, kw_only=True)
class ActivationSample:
    """One dataset item: a model input, with no activation-step-specific concept attached.

    This is the type a dataset feeding
    :meth:`~nnact._pipeline.ActivationPipeline.run` must yield from
    ``__getitem__`` -- in practice, always via one of its subclasses,
    :class:`TokenActivationSample` or :class:`SequenceActivationSample`,
    whichever matches the pipeline's ``output_type``.

    Attributes:
        model_input: The pre-tokenized example to run through the model.
    """

    model_input: SequenceModelInput


def _assert_token_metadata_length(
    metadata: dict[str, list] | None, num_tokens: int
) -> None:
    """Assert every value in ``metadata`` has exactly ``num_tokens`` entries.

    Args:
        metadata: A :class:`TokenActivationSample`'s ``metadata``, or
            ``None`` to skip validation entirely.
        num_tokens: The token count ``metadata``'s values must match --
            that sample's own ``len(model_input.input_ids)``.

    Raises:
        ValueError: If any value's length doesn't equal ``num_tokens``.
    """
    if metadata is None:
        return
    mismatched = {
        key: len(value) for key, value in metadata.items() if len(value) != num_tokens
    }
    if mismatched:
        raise ValueError(
            f"TokenActivationSample.metadata values must have one entry per "
            f"token ({num_tokens}), got lengths {mismatched}."
        )


@dataclass(frozen=True, kw_only=True)
class TokenActivationSample(ActivationSample):
    """An :class:`ActivationSample` for :class:`~nnact._steps._token.TokenActivationStep`.

    Attributes:
        metadata: Named per-token values a caller wants attached to this
            sample -- ``nnact`` never interprets the keys or values, and
            never passes them to the model itself (see
            :class:`SequenceModelInput`). Each value must have one entry
            per token in ``model_input`` -- it's masked the same way
            ``token_ids``/``tokens`` are, with padding positions dropped.
            ``None`` if this sample has no metadata to attach.
    """

    metadata: dict[str, list] | None = None

    def __post_init__(self) -> None:
        """Validate every ``metadata`` value has one entry per token.

        Raises:
            ValueError: If any value's length doesn't match
                ``len(model_input.input_ids)``.
        """
        _assert_token_metadata_length(
            metadata=self.metadata, num_tokens=len(self.model_input.input_ids)
        )


@dataclass(frozen=True, kw_only=True)
class SequenceActivationSample(ActivationSample):
    """An :class:`ActivationSample` for :class:`~nnact._steps._sequence.SequenceActivationStep`.

    Attributes:
        metadata: Named values a caller wants attached to this sample --
            ``nnact`` never interprets the keys or values, and never
            passes them to the model itself (see :class:`SequenceModelInput`).
            Unlike :class:`TokenActivationSample`, values are never
            validated against a token count -- a sequence-level output has
            no per-token rows to mask against, so any value is accepted as
            a single, opaque value for this whole sample. ``None`` if this
            sample has no metadata to attach.
    """

    metadata: dict[str, object] | None = None


@dataclass(frozen=True, kw_only=True)
class ActivationBatch:
    """A batch of :class:`ActivationSample`, with no activation-step-specific concept attached.

    In practice, always one of its subclasses, :class:`TokenActivationBatch`
    or :class:`SequenceActivationBatch`, whichever an activation step
    actually consumes.

    Attributes:
        model_input: The batch's model inputs, ready for
            ``model(**model_input.as_model_kwargs())``.
    """

    model_input: SequenceModelInputBatch


def _collate_metadata(samples: Sequence[ActivationSample]) -> dict[str, list] | None:
    """Collect each sample's own ``metadata`` dict into one per-key list.

    Args:
        samples: The samples being batched, each optionally a
            :class:`TokenActivationSample` or :class:`SequenceActivationSample`
            with its own ``metadata``.

    Returns:
        One list per key, each list's ``i``-th entry being sample ``i``'s
        own value for that key -- or ``None`` if no sample has any
        metadata. A key is included only when every sample has it.

    Raises:
        ValueError: If ``metadata`` keys aren't identical across every
            sample that has any metadata at all -- a key present on some
            samples but not others usually means a bug in the caller's
            dataset, not an intentionally sparse field.
    """
    sample_metadatas = [getattr(sample, "metadata", None) or {} for sample in samples]
    metadata_key_sets = {frozenset(m) for m in sample_metadatas}
    if len(metadata_key_sets) > 1:
        raise ValueError(
            f"Samples in this batch disagree on which metadata keys are "
            f"present: {sorted(metadata_key_sets, key=sorted)}."
        )
    metadata_keys = next(iter(metadata_key_sets), frozenset())
    if not metadata_keys:
        return None
    return {key: [m[key] for m in sample_metadatas] for key in metadata_keys}


def _assert_consistent_metadata_lengths(metadata: dict[str, list] | None) -> None:
    """Assert every sample's list for a key is the same length as every other's.

    Called after :func:`_collate_metadata` has already confirmed every
    sample shares the same metadata keys -- this additionally requires
    those per-sample lists to agree on length within each key, since a
    :class:`TokenActivationBatch`'s ``model_input`` is itself collated to a
    single, uniform ``sequence_length`` and its ``metadata`` must line up
    with it the same way.

    Args:
        metadata: A :class:`TokenActivationBatch`'s collated ``metadata``,
            or ``None`` to skip validation entirely.

    Raises:
        ValueError: If any key's per-sample lists aren't all the same length.
    """
    if metadata is None:
        return
    for key, per_sample_values in metadata.items():
        lengths = {len(value) for value in per_sample_values}
        if len(lengths) > 1:
            raise ValueError(
                f"TokenActivationBatch.metadata[{key!r}] has samples of "
                f"differing lengths: {sorted(lengths)}."
            )


@dataclass(frozen=True, kw_only=True)
class TokenActivationBatch(ActivationBatch):
    """An :class:`ActivationBatch` for :class:`~nnact._steps._token.TokenActivationStep`.

    Attributes:
        metadata: One list per key, each list's ``i``-th entry being sample
            ``i``'s own per-token list for that key. Kept as plain nested
            lists rather than stacked into an array here -- the step
            converts each list to an array itself.
    """

    metadata: dict[str, list] | None = None

    @classmethod
    def from_samples(cls, samples: list[TokenActivationSample]) -> TokenActivationBatch:
        """Stack a list of :class:`TokenActivationSample` into one batch.

        Args:
            samples: The samples to combine, in the order they should
                appear in the batch.

        Returns:
            One :class:`TokenActivationBatch` with ``model_input`` stacked
            and ``metadata`` collected per key (see
            :func:`_collate_metadata`).

        Raises:
            ValueError: If, for any metadata key, samples disagree on that
                key's per-token list length (see
                :func:`_assert_consistent_metadata_lengths`).
        """
        metadata = _collate_metadata(samples)
        _assert_consistent_metadata_lengths(metadata=metadata)
        return cls(
            model_input=_stack_model_input([sample.model_input for sample in samples]),
            metadata=metadata,
        )


@dataclass(frozen=True, kw_only=True)
class SequenceActivationBatch(ActivationBatch):
    """An :class:`ActivationBatch` for :class:`~nnact._steps._sequence.SequenceActivationStep`.

    Attributes:
        metadata: One list per key, each list's ``i``-th entry being sample
            ``i``'s own scalar value for that key.
    """

    metadata: dict[str, list] | None = None

    @classmethod
    def from_samples(
        cls, samples: list[SequenceActivationSample]
    ) -> SequenceActivationBatch:
        """Stack a list of :class:`SequenceActivationSample` into one batch.

        Args:
            samples: The samples to combine, in the order they should
                appear in the batch.

        Returns:
            One :class:`SequenceActivationBatch` with ``model_input``
            stacked and ``metadata`` collected per key (see
            :func:`_collate_metadata`).
        """
        return cls(
            model_input=_stack_model_input([sample.model_input for sample in samples]),
            metadata=_collate_metadata(samples),
        )
