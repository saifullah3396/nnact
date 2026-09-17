"""Hook capture and mapper orchestration, exercised end to end with a model.

Failures here point at activation capture or at the mapper's wiring, since the
storage layer is covered independently in ``test_store.py``.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from nnact._generator._steps._runner import ActivationGenerator
from nnact._types import RunMetadata
from torch.utils.data import DataLoader

from nnact._model._hooked import HookedModel
from nnact._store import (
    H5ActivationStore,
    H5ActivationWriter,
    MemoryActivationStore,
    MemoryActivationWriter,
)
from tests.fixtures.models import (
    DatasetFactory,
    TinyMLP,
    TrainModeProbe,
    TupleOutMLP,
)


def _collate_samples(samples: list[Any]) -> dict[str, Any]:
    return {
        "id": [sample.id for sample in samples],
        "x": torch.stack([sample.data for sample in samples]),
    }


def loader(dataset: Any, batch_size: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_samples,
    )


def test_matches_manual_forward(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """Captured activations must equal a hand-computed forward pass.

    This is the oracle for the whole pipeline: it checks the stored numbers are
    right, not merely that they have plausible shapes.
    """
    dataset = tiny_dataset(n=7)
    store = ActivationGenerator(tiny_model).map(loader(dataset, 3), ["fc1", "fc2"])

    stacked = torch.stack([dataset[i].data for i in range(len(dataset))])
    tiny_model.eval()
    with torch.no_grad():
        expected_fc1 = tiny_model.fc1(stacked)
        expected_fc2 = tiny_model.fc2(torch.relu(expected_fc1))

    assert store.sample_ids == [f"sample_{i}" for i in range(7)]
    for i in range(7):
        fc1, fc2 = store[i].activations
        assert fc1.layer_name == "fc1"
        assert fc2.layer_name == "fc2"
        assert torch.allclose(fc1.tensor, expected_fc1[i], atol=1e-6)
        assert torch.allclose(fc2.tensor, expected_fc2[i], atol=1e-6)


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


def test_defaults_to_memory(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory, h5_path: Path
) -> None:
    """With no path, activations stay in memory and no file is written."""
    store = ActivationGenerator(tiny_model).map(loader(tiny_dataset(n=5)), "fc1")

    assert isinstance(store, MemoryActivationStore)
    assert len(store) == 5
    assert store.activations["fc1"].shape == (5, 8)
    assert not h5_path.exists()


def test_path_selects_h5_and_explicit_writer_wins(
    tiny_model: TinyMLP,
    tiny_dataset: DatasetFactory,
    h5_path: Path,
    tmp_path: Path,
) -> None:
    """A path selects the HDF5 backend; an explicit writer overrides it."""
    dataset = tiny_dataset(n=4)
    mapper = ActivationGenerator(tiny_model)

    store = mapper.map(loader(dataset), "fc1", h5_path)
    assert isinstance(store, H5ActivationStore)
    assert h5_path.exists()
    store.close()

    unused_path = tmp_path / "ignored.h5"
    memory_store = mapper.map(
        loader(dataset), "fc1", unused_path, writer=MemoryActivationWriter()
    )
    assert isinstance(memory_store, MemoryActivationStore)
    assert not unused_path.exists()

    explicit_path = tmp_path / "explicit.h5"
    h5_store = mapper.map(
        loader(dataset), "fc1", writer=H5ActivationWriter(explicit_path)
    )
    assert isinstance(h5_store, H5ActivationStore)
    assert explicit_path.exists()
    h5_store.close()


def test_model_in_eval_and_no_grad(
    tiny_dataset: DatasetFactory,
) -> None:
    """The mapper must run in eval mode with gradients disabled.

    A model left in train mode yields wrong activations through dropout or
    batchnorm without raising, so this guards a silent-corruption path. The
    probe model uses dropout with p=1.0, which zeroes everything while training
    and is a no-op in eval.
    """
    model = TrainModeProbe()
    model.train()

    store = ActivationGenerator(model).map(loader(tiny_dataset(n=4)), "drop")

    activations = store.activations["drop"]
    assert activations.abs().sum() > 0, "activations are zero, model ran in train mode"
    assert not activations.requires_grad
    assert activations.grad_fn is None


def test_map_rejects_unknown_layer(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """A bad layer name fails before any forward pass, naming near misses."""
    mapper = ActivationGenerator(tiny_model)

    with pytest.raises(ValueError, match="not found in TinyMLP") as excinfo:
        mapper.map(loader(tiny_dataset(n=4)), ["fc1", "fc3"])

    message = str(excinfo.value)
    assert "fc3" in message
    assert "fc1" in message


def test_summary_lists_and_marks_layers(tiny_model: TinyMLP) -> None:
    """summary() reports hookable layers and flags ones absent from the model."""
    mapper = ActivationGenerator(tiny_model)

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


def test_map_records_run_metadata(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory, h5_path: Path
) -> None:
    """Metadata describing the run is stored with the activations."""
    store = ActivationGenerator(tiny_model).map(
        loader(tiny_dataset(n=6)), ["fc1", "fc2"], progress=False
    )

    metadata = store.metadata
    assert metadata.model == "TinyMLP"
    assert metadata.layers == ["fc1", "fc2"]
    assert metadata.samples == 6
    assert metadata.batch_size == 2
    assert metadata.parameters == sum(p.numel() for p in tiny_model.parameters())
    assert isinstance(metadata.seconds, float)
    assert metadata.created
    assert metadata.samples_per_second > 0


def test_h5_metadata_survives_roundtrip(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory, h5_path: Path
) -> None:
    """HDF5 persists the metadata, so a reloaded cache says how it was made."""
    written = ActivationGenerator(tiny_model).map(
        loader(tiny_dataset(n=4)), "fc1", h5_path, progress=False
    )
    expected = written.metadata
    written.close()

    reloaded = H5ActivationStore.load(h5_path, [f"sample_{i}" for i in range(4)])
    assert reloaded.metadata == expected
    assert reloaded.metadata.model == "TinyMLP"
    reloaded.close()


def test_metadata_repr_is_readable() -> None:
    """The repr lists one field per line, abbreviating counts and duration."""
    text = repr(
        RunMetadata(
            model="ResNet",
            parameters=11_689_512,
            layers=["layer3", "layer4"],
            samples=512,
            batch_size=64,
            device="cpu",
            seconds=2.804,
            created="2026-01-01T00:00:00+00:00",
        )
    )

    lines = text.splitlines()
    assert lines[0] == "RunMetadata("
    assert lines[-1] == ")"
    assert "    model      = ResNet" in lines
    assert "    parameters = 11.7M" in lines
    assert "    seconds    = 2.80s" in lines
    assert "11689512" not in text

    assert "seconds    = 420us" in repr(RunMetadata(seconds=0.00042))
    assert "extra" not in repr(RunMetadata(model="M"))


def test_metadata_reports_actual_device(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """The device recorded is the model's own, not the argument passed.

    A model already on an accelerator runs there whether or not ``device`` was
    given, so trusting the argument would misreport the run.
    """
    store = ActivationGenerator(tiny_model).map(
        loader(tiny_dataset(n=4)), "fc1", progress=False
    )

    assert store.metadata.device == "cpu"
    assert next(tiny_model.parameters()).device.type == "cpu"


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
            "id": ["a", "b"],
            "inputs": torch.tensor([[1.0], [2.0]]),
            "metadata": {"offset": torch.tensor([[3.0], [4.0]])},
            "label": ["left", "right"],
        }
    ]
    store = ActivationGenerator(model).map(
        batches, "layer", device="cpu", progress=False
    )

    assert store.sample_ids == ["a", "b"]
    assert torch.equal(store.activations["layer"], torch.tensor([[4.0], [6.0]]))
    assert model.seen_label == ["left", "right"]
