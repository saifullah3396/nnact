from __future__ import annotations

import numpy as np

from nnact._outputs._dataset import (
    InMemorySequenceActivationDataset,
    InMemoryTokenActivationDataset,
)
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


def test_sequence_dataset_accumulates_across_batches() -> None:
    dataset = InMemorySequenceActivationDataset()
    dataset._add_batch(
        output=SequenceActivationOutput(
            prediction=np.zeros((2, 3), dtype=np.int64),
            top_probability=np.ones((2, 3), dtype=np.float32),
            metadata={"label": np.array(["a", "b"])},
            activations={"layer0": np.zeros((2, 3, 4), dtype=np.float32)},
        )
    )
    dataset._add_batch(
        output=SequenceActivationOutput(
            prediction=np.ones((1, 3), dtype=np.int64),
            top_probability=np.ones((1, 3), dtype=np.float32),
            metadata={"label": np.array(["c"])},
            activations={"layer0": np.ones((1, 3, 4), dtype=np.float32)},
        )
    )

    assert len(dataset) == 3
    assert dataset.metadata is not None
    assert dataset.metadata["label"].tolist() == ["a", "b", "c"]
    assert dataset.activations["layer0"].shape == (3, 3, 4)


def test_sequence_dataset_metadata_is_none_when_never_provided() -> None:
    dataset = InMemorySequenceActivationDataset()
    dataset._add_batch(
        output=SequenceActivationOutput(
            prediction=np.zeros((1, 2), dtype=np.int64),
            top_probability=np.ones((1, 2), dtype=np.float32),
        )
    )
    assert dataset.metadata is None


def test_token_dataset_accumulates_offsets_and_derived_fields() -> None:
    dataset = InMemoryTokenActivationDataset()
    dataset._add_batch(
        output=TokenActivationOutput(
            prediction=np.array([1, 2]),
            top_probability=np.ones(2, dtype=np.float32),
            offsets=np.array([0, 2]),
            metadata={"label": np.array(["a", "b"])},
            activations={"layer0": np.zeros((2, 4), dtype=np.float32)},
        )
    )
    dataset._add_batch(
        output=TokenActivationOutput(
            prediction=np.array([3]),
            top_probability=np.ones(1, dtype=np.float32),
            offsets=np.array([0, 1]),
            metadata={"label": np.array(["c"])},
            activations={"layer0": np.zeros((1, 4), dtype=np.float32)},
        )
    )

    assert len(dataset) == 2
    assert dataset.offsets.tolist() == [0, 2, 3]
    assert dataset.sequence_lengths.tolist() == [2, 1]
    assert dataset.sample_of_token.tolist() == [0, 0, 1]
    assert dataset.metadata is not None
    assert dataset.metadata["label"].tolist() == ["a", "b", "c"]

    summary = dataset.summary()
    assert "label" in summary.columns
    assert "layer0_norm" in summary.columns
