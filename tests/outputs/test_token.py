from __future__ import annotations

import numpy as np
import pytest

from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput


def _make_sequence_output(**overrides: object) -> SequenceActivationOutput:
    defaults: dict[str, object] = {
        "prediction": np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64),
        "top_probability": np.ones((2, 3), dtype=np.float32),
    }
    defaults.update(overrides)
    return SequenceActivationOutput(**defaults)  # type: ignore[arg-type]


def _mask() -> np.ndarray:
    # sample 0 has 2 real tokens, sample 1 has 1 real token
    return np.array([[1, 1, 0], [1, 0, 0]])


def test_from_sequence_masks_predictions_and_builds_offsets() -> None:
    output = TokenActivationOutput.from_sequence(
        output=_make_sequence_output(), mask=_mask()
    )
    assert output.prediction.tolist() == [1, 2, 3]
    assert output.offsets.tolist() == [0, 2, 3]
    assert output.batch_size == 2
    assert output.num_tokens == 3


def test_from_sequence_masks_metadata_per_key() -> None:
    metadata = {"label": np.array([["a", "b", "z"], ["c", "z", "z"]])}
    output = TokenActivationOutput.from_sequence(
        output=_make_sequence_output(), mask=_mask(), metadata=metadata
    )
    assert output.metadata is not None
    assert output.metadata["label"].tolist() == ["a", "b", "c"]


def test_from_sequence_masks_token_ids_and_tokens() -> None:
    token_ids = np.array([[10, 11, 0], [12, 0, 0]])
    tokens = np.array([["a", "b", "<pad>"], ["c", "<pad>", "<pad>"]])
    output = TokenActivationOutput.from_sequence(
        output=_make_sequence_output(),
        mask=_mask(),
        token_ids=token_ids,
        tokens=tokens,
    )
    assert output.token_ids is not None
    assert output.token_ids.tolist() == [10, 11, 12]
    assert output.tokens is not None
    assert output.tokens.tolist() == ["a", "b", "c"]


def test_post_init_rejects_wrong_shaped_metadata() -> None:
    with pytest.raises(AssertionError, match="metadata\\['label'\\]"):
        TokenActivationOutput(
            prediction=np.array([1, 2, 3]),
            top_probability=np.ones(3, dtype=np.float32),
            offsets=np.array([0, 3]),
            metadata={"label": np.zeros((3, 1))},
        )


def test_post_init_validates_offsets_invariants() -> None:
    with pytest.raises(AssertionError):
        TokenActivationOutput(
            prediction=np.array([1, 2, 3]),
            top_probability=np.ones(3, dtype=np.float32),
            offsets=np.array([1, 3]),
        )
