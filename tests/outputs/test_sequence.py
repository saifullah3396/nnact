from __future__ import annotations

import numpy as np
import pytest

from nnact._outputs._sequence import SequenceActivationOutput


def _make_output(**overrides: object) -> SequenceActivationOutput:
    defaults: dict[str, object] = {
        "prediction": np.zeros((2, 3), dtype=np.int64),
        "top_probability": np.ones((2, 3), dtype=np.float32),
    }
    defaults.update(overrides)
    return SequenceActivationOutput(**defaults)  # type: ignore[arg-type]


def test_accepts_scalar_per_sample_metadata() -> None:
    # Regression test: this shape (batch_size,) previously crashed when
    # ActivationSample had a single ambiguous field shared with token-level.
    output = _make_output(metadata={"label": np.array(["user", "assistant"])})
    assert output.metadata is not None
    assert output.metadata["label"].shape == (2,)


def test_rejects_metadata_with_wrong_shape() -> None:
    with pytest.raises(AssertionError, match="metadata\\['label'\\]"):
        _make_output(metadata={"label": np.zeros((2, 1))})


def test_rejects_top_probability_shape_mismatch() -> None:
    with pytest.raises(AssertionError, match="top_probability"):
        _make_output(top_probability=np.ones((2, 2), dtype=np.float32))


def test_activations_validated_against_leading_shape() -> None:
    with pytest.raises(AssertionError):
        _make_output(activations={"layer0": np.zeros((3, 3, 4))})


def test_batch_size_and_sequence_length() -> None:
    output = _make_output()
    assert output.batch_size == 2
    assert output.sequence_length == 3
