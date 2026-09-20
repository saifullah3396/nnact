from __future__ import annotations

from pathlib import Path

import numpy as np

from nnact._outputs._h5_dataset import (
    H5SequenceActivationDataset,
    H5TokenActivationDataset,
)
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


def test_sequence_h5_round_trip(tmp_path: Path) -> None:
    dataset = H5SequenceActivationDataset(path=tmp_path / "seq.h5")
    assert not dataset.exists()

    dataset._add_batch(
        output=SequenceActivationOutput(
            prediction=np.zeros((2, 3), dtype=np.int64),
            top_probability=np.ones((2, 3), dtype=np.float32),
            metadata={"label": np.array(["user", "assistant"])},
            activations={"layer0": np.zeros((2, 3, 4), dtype=np.float32)},
        )
    )

    assert dataset.exists()
    assert len(dataset) == 2
    assert dataset.metadata is not None
    assert dataset.metadata["label"].tolist() == ["user", "assistant"]
    activations = dataset.activations("layer0")
    assert activations.shape == (2, 3, 4)
    assert dataset.activations("layer0") is activations

    run_metadata = {"model": "FakeModel", "layer_names": ["layer0"]}
    dataset._write_run_metadata(run_metadata)
    assert dataset.run_metadata == run_metadata

    dataset.close(delete=True)
    assert not dataset.exists()


def test_token_h5_round_trip(tmp_path: Path) -> None:
    dataset = H5TokenActivationDataset(path=tmp_path / "token.h5")

    dataset._add_batch(
        output=TokenActivationOutput(
            prediction=np.array([1, 2, 3]),
            top_probability=np.ones(3, dtype=np.float32),
            offsets=np.array([0, 2, 3]),
            metadata={"label": np.array(["a", "b", "c"])},
            activations={"layer0": np.zeros((3, 4), dtype=np.float32)},
            token_ids=np.array([10, 11, 12]),
            tokens=np.array(["tok_a", "tok_b", "tok_c"]),
        )
    )

    assert len(dataset) == 2
    assert dataset.offsets.tolist() == [0, 2, 3]
    assert dataset.metadata is not None
    assert dataset.metadata["label"].tolist() == ["a", "b", "c"]
    assert dataset.token_ids is not None
    assert dataset.token_ids.tolist() == [10, 11, 12]
    assert dataset.tokens is not None
    assert dataset.tokens.tolist() == ["tok_a", "tok_b", "tok_c"]

    summary = dataset.summary()
    assert "label" in summary.columns
    assert "loss" not in summary.columns
    assert "layer0_norm" not in summary.columns


def test_token_h5_metadata_none_when_never_written(tmp_path: Path) -> None:
    dataset = H5TokenActivationDataset(path=tmp_path / "token_no_meta.h5")
    dataset._add_batch(
        output=TokenActivationOutput(
            prediction=np.array([1]),
            top_probability=np.ones(1, dtype=np.float32),
            offsets=np.array([0, 1]),
            activations={"layer0": np.zeros((1, 4), dtype=np.float32)},
        )
    )
    assert dataset.metadata is None
