from __future__ import annotations

import pytest

from nnact._model._hooked import HookedModel
from nnact._outputs._protocols import TokenActivationBatch, TokenActivationSample
from nnact._steps._token import TokenActivationStep
from tests.conftest import FakeModel, make_model_input


def _build_step() -> TokenActivationStep:
    step = TokenActivationStep(
        hooked_model=HookedModel(model=FakeModel()),
        layer_names=["linear"],
        device="cpu",
    )
    step._prepare_model()
    return step


def test_call_rejects_non_token_batch() -> None:
    step = _build_step()
    with pytest.raises(AssertionError, match="TokenActivationBatch"):
        step(None, object())  # type: ignore[arg-type]


def test_call_produces_one_row_per_real_token_with_metadata() -> None:
    step = _build_step()
    samples = [
        TokenActivationSample(
            model_input=make_model_input(ids=[1, 2, 3]),
            metadata={"label": ["a", "b", "c"]},
        ),
        TokenActivationSample(
            model_input=make_model_input(ids=[4, 5, 6]),
            metadata={"label": ["d", "e", "f"]},
        ),
    ]
    batch = TokenActivationBatch.from_samples(samples=samples)

    output = step(None, batch)  # type: ignore[arg-type]

    assert output.num_tokens == 6
    assert output.activations["linear"].shape == (6, 4)
    assert output.metadata is not None
    assert output.metadata["label"].tolist() == ["a", "b", "c", "d", "e", "f"]
    assert getattr(output, "loss", None) is None


def test_call_without_metadata_leaves_it_none() -> None:
    step = _build_step()
    samples = [TokenActivationSample(model_input=make_model_input(ids=[1, 2]))]
    batch = TokenActivationBatch.from_samples(samples=samples)

    output = step(None, batch)  # type: ignore[arg-type]

    assert output.metadata is None
