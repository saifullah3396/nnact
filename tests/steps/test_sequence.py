from __future__ import annotations

import pytest

from nnact._model._hooked import HookedModel
from nnact._outputs._protocols import SequenceActivationBatch, SequenceActivationSample
from nnact._steps._sequence import SequenceActivationStep
from tests.conftest import FakeModel, make_model_input


def _build_step() -> SequenceActivationStep:
    step = SequenceActivationStep(
        hooked_model=HookedModel(model=FakeModel()),
        layer_names=["linear"],
        device="cpu",
    )
    step._prepare_model()
    return step


def test_call_rejects_non_sequence_batch() -> None:
    step = _build_step()
    with pytest.raises(AssertionError, match="SequenceActivationBatch"):
        step(None, object())  # type: ignore[arg-type]


def test_call_produces_one_row_per_sample_with_scalar_metadata() -> None:
    # Regression test: this used to crash with a shape mismatch when
    # sequence-level and token-level samples shared one ActivationSample type.
    step = _build_step()
    samples = [
        SequenceActivationSample(
            model_input=make_model_input(ids=[1, 2]), metadata={"label": "user"}
        ),
        SequenceActivationSample(
            model_input=make_model_input(ids=[3, 4]), metadata={"label": "assistant"}
        ),
    ]
    batch = SequenceActivationBatch.from_samples(samples=samples)

    output = step(None, batch)  # type: ignore[arg-type]

    assert output.batch_size == 2
    assert output.activations["linear"].shape == (2, 2, 4)
    assert output.metadata is not None
    assert output.metadata["label"].tolist() == ["user", "assistant"]
    assert getattr(output, "loss", None) is None


def test_call_without_metadata_leaves_it_none() -> None:
    step = _build_step()
    samples = [SequenceActivationSample(model_input=make_model_input(ids=[1, 2]))]
    batch = SequenceActivationBatch.from_samples(samples=samples)

    output = step(None, batch)  # type: ignore[arg-type]

    assert output.metadata is None
