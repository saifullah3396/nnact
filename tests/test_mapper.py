"""Hook capture and mapper orchestration, exercised end to end with a model."""

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from nnact._mapper import ActivationMapper
from nnact._model import HookedModel
from tests.fixtures.models import (
    DatasetFactory,
    TinyMLP,
    TrainModeProbe,
    TupleOutMLP,
)


def _collate(samples: list[torch.Tensor]) -> dict[str, Any]:
    return {"x": torch.stack(samples)}


def loader(dataset: Any, batch_size: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)


def test_matches_manual_forward(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """Captured activations must equal a hand-computed forward pass.

    This is the oracle for the whole pipeline: it checks the returned numbers
    are right, not merely that they have plausible shapes.
    """
    dataset = tiny_dataset(n=7)
    result = ActivationMapper(tiny_model).map(
        loader(dataset, 3), ["fc1", "fc2"], progress=False
    )

    stacked = torch.stack([dataset[i] for i in range(len(dataset))])
    tiny_model.eval()
    with torch.no_grad():
        expected_fc1 = tiny_model.fc1(stacked)
        expected_fc2 = tiny_model.fc2(torch.relu(expected_fc1))

    assert torch.allclose(result.activations["fc1"], expected_fc1, atol=1e-6)
    assert torch.allclose(result.activations["fc2"], expected_fc2, atol=1e-6)
    assert result.output == {}


def test_hooks_removed_after_capture(tiny_model: TinyMLP) -> None:
    """Handles must be released on the normal path and when forward raises.

    A leaked handle is invisible in the output but degrades every later pass,
    so both exits are checked.
    """
    hooked = HookedModel(tiny_model)
    assert len(tiny_model.fc1._forward_hooks) == 0

    with hooked.capture(["fc1"]):
        hooked(torch.randn(2, 4))
        assert len(tiny_model.fc1._forward_hooks) == 1
    assert len(tiny_model.fc1._forward_hooks) == 0

    with pytest.raises(RuntimeError), hooked.capture(["fc1"]):
        hooked(torch.randn(2, 99))
    assert len(tiny_model.fc1._forward_hooks) == 0


def test_unknown_layer_raises(tiny_model: TinyMLP) -> None:
    """Hooking a name absent from the model is reported clearly."""
    hooked = HookedModel(tiny_model)
    with (
        pytest.raises(ValueError, match="does not exist in model"),
        hooked.capture(["nonexistent"]),
    ):
        pass


def test_tuple_output_uses_first_element() -> None:
    """A layer returning a tuple contributes its first element."""
    model = TupleOutMLP()
    hooked = HookedModel(model)
    data = torch.randn(2, 4)

    with hooked.capture(["body"]):
        hooked(data)
        captured = hooked.get_activation("body")

    model.eval()
    with torch.no_grad():
        expected, _ = model.body(data)
    assert captured.shape == (2, 8)
    assert torch.allclose(captured, expected, atol=1e-6)


def test_map_returns_stacked_tensors(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """map() collects one stacked tensor per layer, across all batches."""
    result = ActivationMapper(tiny_model).map(
        loader(tiny_dataset(n=5)), "fc1", progress=False
    )

    assert result.activations["fc1"].shape == (5, 8)


def test_model_in_eval_and_no_grad(tiny_dataset: DatasetFactory) -> None:
    """The mapper must run in eval mode with gradients disabled.

    A model left in train mode yields wrong activations through dropout or
    batchnorm without raising, so this guards a silent-corruption path. The
    probe model uses dropout with p=1.0, which zeroes everything while training
    and is a no-op in eval.
    """
    model = TrainModeProbe()
    model.train()

    result = ActivationMapper(model).map(
        loader(tiny_dataset(n=4)), "drop", progress=False
    )

    activations = result.activations["drop"]
    assert activations.abs().sum() > 0, "activations are zero, model ran in train mode"
    assert not activations.requires_grad
    assert activations.grad_fn is None


def test_map_rejects_unknown_layer(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """A bad layer name fails before any forward pass, naming near misses."""
    mapper = ActivationMapper(tiny_model)

    with pytest.raises(ValueError, match="not found in TinyMLP") as excinfo:
        mapper.map(loader(tiny_dataset(n=4)), ["fc1", "fc3"], progress=False)

    message = str(excinfo.value)
    assert "fc3" in message
    assert "fc1" in message


def test_summary_lists_and_marks_layers(tiny_model: TinyMLP) -> None:
    """summary() reports hookable layers and flags ones absent from the model."""
    mapper = ActivationMapper(tiny_model)

    assert mapper.available_layers() == ["fc1", "fc2"]
    assert mapper.parameter_count() == sum(p.numel() for p in tiny_model.parameters())

    frame = mapper.summary(["fc1"])
    assert list(frame.index) == ["fc1", "fc2"]
    assert list(frame.columns) == ["module", "parameters", "selected"]
    assert frame.loc["fc1", "module"] == "Linear"
    assert frame.loc["fc1", "selected"]
    assert not frame.loc["fc2", "selected"]
    assert frame["parameters"].sum() == mapper.parameter_count()

    assert "selected" not in mapper.summary().columns


def test_loader_batch_structure_and_non_tensor_values_are_preserved() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Identity()
            self.seen_label = None

        def forward(self, inputs, metadata, label):
            self.seen_label = label
            return self.layer(inputs + metadata["offset"])

    model = Model()
    batches = [
        {
            "inputs": torch.tensor([[1.0], [2.0]]),
            "metadata": {"offset": torch.tensor([[3.0], [4.0]])},
            "label": ["left", "right"],
        }
    ]
    result = ActivationMapper(model).map(
        batches, "layer", device="cpu", progress=False
    )

    assert torch.equal(result.activations["layer"], torch.tensor([[4.0], [6.0]]))
    assert model.seen_label == ["left", "right"]


def test_output_transform_runs_in_same_pass(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """The model's own output is collected alongside layer activations.

    Passing ``output_transform`` should not require a second forward pass:
    the collected output must match what a manual forward pass produces.
    """
    dataset = tiny_dataset(n=5)

    def transform(output: object) -> dict[str, torch.Tensor]:
        assert isinstance(output, torch.Tensor)
        return {"logits": output}

    result = ActivationMapper(tiny_model).map(
        loader(dataset, 2), "fc1", output_transform=transform, progress=False
    )

    stacked = torch.stack([dataset[i] for i in range(len(dataset))])
    tiny_model.eval()
    with torch.no_grad():
        expected = tiny_model(stacked)

    assert torch.allclose(result.output["logits"], expected, atol=1e-6)
