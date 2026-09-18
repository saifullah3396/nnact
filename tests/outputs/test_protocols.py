from __future__ import annotations

import pytest
import torch

from nnact._outputs._protocols import (
    SequenceActivationBatch,
    SequenceActivationSample,
    TokenActivationBatch,
    TokenActivationSample,
    _stack_model_input,
)
from tests.conftest import make_model_input


def test_stack_model_input_stacks_required_fields() -> None:
    batch = _stack_model_input(
        model_inputs=[make_model_input(ids=[1, 2, 3]), make_model_input(ids=[4, 5, 6])]
    )
    assert batch.input_ids.shape == (2, 3)
    assert batch.attention_mask.shape == (2, 3)
    assert batch.token_type_ids is None
    assert batch.position_ids is None


def test_stack_model_input_keeps_optional_field_only_if_every_sample_has_it() -> None:
    with_ids = make_model_input(ids=[1, 2])
    without_ids = make_model_input(ids=[3, 4])
    object.__setattr__(with_ids, "token_type_ids", torch.zeros(2, dtype=torch.long))

    batch = _stack_model_input(model_inputs=[with_ids, without_ids])
    assert batch.token_type_ids is None


def test_token_activation_sample_accepts_matching_metadata_length() -> None:
    sample = TokenActivationSample(
        model_input=make_model_input(ids=[1, 2, 3]),
        metadata={"label": ["a", "b", "c"]},
    )
    assert sample.metadata is not None
    assert len(sample.metadata["label"]) == 3


def test_token_activation_sample_rejects_mismatched_metadata_length() -> None:
    with pytest.raises(ValueError, match="one entry per"):
        TokenActivationSample(
            model_input=make_model_input(ids=[1, 2, 3]),
            metadata={"label": ["a", "b"]},
        )


def test_sequence_activation_sample_accepts_any_metadata_value() -> None:
    sample = SequenceActivationSample(
        model_input=make_model_input(ids=[1, 2, 3]),
        metadata={"label": "user", "score": 0.5, "tags": ["x", "y"]},
    )
    assert sample.metadata == {"label": "user", "score": 0.5, "tags": ["x", "y"]}


def test_token_activation_batch_collates_metadata_per_key() -> None:
    samples = [
        TokenActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": ["a", "b"]}
        ),
        TokenActivationSample(
            model_input=make_model_input(ids=[3, 4]), metadata={"label": ["c", "d"]}
        ),
    ]
    batch = TokenActivationBatch.from_samples(samples=samples)
    assert batch.metadata == {"label": [["a", "b"], ["c", "d"]]}


def test_token_activation_batch_raises_on_mismatched_keys() -> None:
    samples = [
        TokenActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": ["a", "b"]}
        ),
        TokenActivationSample(model_input=make_model_input(ids=[3, 4]), metadata=None),
    ]
    with pytest.raises(ValueError, match="disagree"):
        TokenActivationBatch.from_samples(samples=samples)


def test_token_activation_batch_raises_on_inconsistent_lengths_across_samples() -> None:
    samples = [
        TokenActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": ["a", "b"]}
        ),
        TokenActivationSample(
            model_input=make_model_input(ids=[3, 4, 5]),
            metadata={"label": ["c", "d", "e"]},
        ),
    ]
    with pytest.raises(ValueError, match="differing lengths"):
        TokenActivationBatch.from_samples(samples=samples)


def test_sequence_activation_batch_collates_scalar_metadata() -> None:
    samples = [
        SequenceActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": "user"}
        ),
        SequenceActivationSample(
            model_input=make_model_input(ids=[3, 4]), metadata={"label": "assistant"}
        ),
    ]
    batch = SequenceActivationBatch.from_samples(samples=samples)
    assert batch.metadata == {"label": ["user", "assistant"]}


def test_sequence_activation_batch_raises_on_mismatched_keys() -> None:
    samples = [
        SequenceActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": "user"}
        ),
        SequenceActivationSample(
            model_input=make_model_input(ids=[3, 4]), metadata=None
        ),
    ]
    with pytest.raises(ValueError, match="disagree"):
        SequenceActivationBatch.from_samples(samples=samples)
