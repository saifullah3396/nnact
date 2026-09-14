"""Tests for nnact — HDF5-backed activation dataset generation."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from nnact._mapper import ActivationMapper
from nnact._types import Sample
from nnact.store import H5ActivationStore, MemoryActivationStore, MemoryActivationWriter


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _TinyDataset(Dataset[Sample]):
    def __init__(self, n: int = 5, id_prefix: str = "sample") -> None:
        self._samples = [
            Sample(id=f"{id_prefix}_{i}", data=torch.randn(4)) for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@pytest.fixture()
def h5_path(tmp_path: Path) -> Path:
    return tmp_path / "acts.h5"


def test_single_layer_one_activation_per_sample(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=5)
    store = ActivationMapper(model, "fc1", batch_size=8).map(dataset, h5_path)

    assert len(store) == 5
    for i in range(5):
        s = store[i]
        assert len(s.activations) == 1
        assert s.activations[0].layer_name == "fc1"
        assert s.activations[0].tensor.shape == (8,)

    store.close()


def test_multiple_layers_activations_captured(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=4)
    store = ActivationMapper(model, ["fc1", "fc2"], batch_size=8).map(dataset, h5_path)

    assert len(store) == 4
    for i in range(4):
        s = store[i]
        assert len(s.activations) == 2
        assert [a.layer_name for a in s.activations] == ["fc1", "fc2"]
        assert s.activations[0].tensor.shape == (8,)
        assert s.activations[1].tensor.shape == (2,)

    store.close()


def test_sample_ids_stored_and_exposed(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=4, id_prefix="img")
    store = ActivationMapper(model, "fc1", batch_size=8).map(dataset, h5_path)

    assert store.sample_ids == [f"img_{i}" for i in range(4)]
    store.close()


def test_load_cache_matches_hash(h5_path: Path) -> None:
    model = _TinyMLP()
    n = 6
    dataset = _TinyDataset(n=n)
    ActivationMapper(model, "fc1", batch_size=8).map(dataset, h5_path).close()

    expected_ids = [f"sample_{i}" for i in range(n)]
    store = H5ActivationStore.load(h5_path, expected_ids)
    assert len(store) == n
    assert store.sample_ids == expected_ids
    assert store[0].activations[0].tensor.shape == (8,)
    store.close()


def test_load_cache_rejects_wrong_ids(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=5)
    ActivationMapper(model, "fc1", batch_size=8).map(dataset, h5_path).close()

    with pytest.raises(ValueError, match="Hash mismatch"):
        H5ActivationStore.load(h5_path, ["wrong_0", "wrong_1"])


def test_memory_store_is_default(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=5)
    store = ActivationMapper(model, "fc1", batch_size=2).map(dataset)

    assert isinstance(store, MemoryActivationStore)
    assert len(store) == 5
    assert store.sample_ids == [f"sample_{i}" for i in range(5)]
    assert store.layer_names == ["fc1"]
    assert store.activations["fc1"].shape == (5, 8)
    assert store[3].activations[0].tensor.shape == (8,)
    assert not h5_path.exists()


def test_memory_and_h5_agree(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=6)
    mapper = ActivationMapper(model, ["fc1", "fc2"], batch_size=4)

    mem = mapper.map(dataset)
    h5 = mapper.map(dataset, h5_path)

    assert mem.sample_ids == h5.sample_ids
    for i in range(6):
        for a, b in zip(mem[i].activations, h5[i].activations, strict=True):
            assert a.layer_name == b.layer_name
            assert torch.allclose(a.tensor, b.tensor)
    h5.close()


def test_memory_store_rejects_length_mismatch() -> None:
    with pytest.raises(AssertionError, match="but there are 3 sample ids"):
        MemoryActivationStore(
            activations={"fc1": torch.randn(5, 8)},
            sample_ids=["a", "b", "c"],
        )


def test_h5_store_rejects_length_mismatch(h5_path: Path) -> None:
    model = _TinyMLP()
    dataset = _TinyDataset(n=5)
    ActivationMapper(model, "fc1", batch_size=8).map(dataset, h5_path).close()

    with pytest.raises(AssertionError, match="but there are 2 sample ids"):
        H5ActivationStore(
            path=h5_path, layer_names=["fc1"], sample_ids=["sample_0", "sample_1"]
        )


def test_writer_rejects_inconsistent_layers() -> None:
    writer = MemoryActivationWriter()
    writer.write(["a", "b"], {"fc1": torch.randn(2, 8)})

    with pytest.raises(AssertionError, match="already holds"):
        writer.write(["c"], {"fc2": torch.randn(1, 4)})


def test_writer_rejects_batch_size_mismatch() -> None:
    writer = MemoryActivationWriter()
    with pytest.raises(ValueError, match="does not match 3 sample ids"):
        writer.write(["a", "b", "c"], {"fc1": torch.randn(2, 8)})
