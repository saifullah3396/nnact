"""Hook capture and mapper orchestration, exercised end to end with a model.

Failures here point at activation capture or at the mapper's wiring, since the
storage layer is covered independently in ``test_store.py``.
"""

from pathlib import Path

import pytest
import torch

from nnact._mapper import ActivationMapper
from nnact._model import HookedModel
from nnact.store import (
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


def test_matches_manual_forward(
    tiny_model: TinyMLP, tiny_dataset: DatasetFactory
) -> None:
    """Captured activations must equal a hand-computed forward pass.

    This is the oracle for the whole pipeline: it checks the stored numbers are
    right, not merely that they have plausible shapes.
    """
    dataset = tiny_dataset(n=7)
    store = ActivationMapper(tiny_model, ["fc1", "fc2"], batch_size=3).map(dataset)

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
    store = ActivationMapper(tiny_model, "fc1", batch_size=2).map(tiny_dataset(n=5))

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
    mapper = ActivationMapper(tiny_model, "fc1", batch_size=2)

    store = mapper.map(dataset, h5_path)
    assert isinstance(store, H5ActivationStore)
    assert h5_path.exists()
    store.close()

    unused_path = tmp_path / "ignored.h5"
    memory_store = mapper.map(dataset, unused_path, writer=MemoryActivationWriter())
    assert isinstance(memory_store, MemoryActivationStore)
    assert not unused_path.exists()

    explicit_path = tmp_path / "explicit.h5"
    h5_store = mapper.map(dataset, writer=H5ActivationWriter(explicit_path))
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

    store = ActivationMapper(model, "drop", batch_size=2).map(tiny_dataset(n=4))

    assert isinstance(store, MemoryActivationStore)
    activations = store.activations["drop"]
    assert activations.abs().sum() > 0, "activations are zero, model ran in train mode"
    assert not activations.requires_grad
    assert activations.grad_fn is None
