"""Store and writer behaviour, exercised without a model or forward pass.

Tests here build activation tensors by hand, so a failure points at the storage
layer rather than at hook capture. Nothing in this module imports
``ActivationMapper``, ``HookedModel`` or ``torch.nn``.
"""

from pathlib import Path

import pytest
import torch

from nnact.store import (
    H5ActivationStore,
    H5ActivationWriter,
    MemoryActivationStore,
    MemoryActivationWriter,
)
from tests.fixtures.types import ActivationFactory, StoreFactory, WriterFactory


def test_roundtrip_preserves_order_and_values(
    make_writer: WriterFactory, identifiable: ActivationFactory
) -> None:
    """Rows must land at their write position across unequal batches.

    Batch sizes 3, 1 and 2 are deliberately unequal: a uniform size would hide
    stride and offset errors in the H5 resize path.
    """
    writer = make_writer()
    ids: list[str] = []
    start = 0
    for size in (3, 1, 2):
        batch_ids = [f"s{start + i}" for i in range(size)]
        writer.write(batch_ids, {"l": identifiable(size, 4, start)})
        ids.extend(batch_ids)
        start += size

    store = writer.close()

    assert len(store) == 6
    assert store.sample_ids == ids
    for i in range(6):
        tensor = store[i].activations[0].tensor
        assert torch.equal(tensor, torch.full((4,), float(i))), (
            f"position {i} holds {tensor[0].item()}"
        )
    store.close()


def test_getitem_returns_layers_in_declared_order(
    written_store: StoreFactory, identifiable: ActivationFactory
) -> None:
    """Layer order must be stable, since a swap silently mislabels results."""
    activations = {
        "first": identifiable(3, 4, start=0),
        "second": identifiable(3, 2, start=100),
    }
    store = written_store(activations, ["a", "b", "c"])

    assert store.layer_names == ["first", "second"]
    sample = store[1]
    assert [a.layer_name for a in sample.activations] == ["first", "second"]
    assert sample.activations[0].tensor[0].item() == 1.0
    assert sample.activations[1].tensor[0].item() == 101.0
    store.close()


def test_uneven_final_batch(
    make_writer: WriterFactory, identifiable: ActivationFactory
) -> None:
    """A batch size that does not divide the total leaves a short final batch."""
    n, batch = 10, 3
    writer = make_writer()
    ids: list[str] = []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        batch_ids = [f"s{start + i}" for i in range(size)]
        writer.write(batch_ids, {"l": identifiable(size, 4, start)})
        ids.extend(batch_ids)

    store = writer.close()

    assert len(store) == n
    assert store.sample_ids == ids
    assert store[n - 1].activations[0].tensor[0].item() == float(n - 1)
    store.close()


def test_many_batches_preserve_order(
    make_writer: WriterFactory, identifiable: ActivationFactory
) -> None:
    """Repeated growth must stay ordered over many batches.

    Exercises the H5 resize loop and the memory concatenate at realistic batch
    counts; correctness is checked by strided sampling rather than by reading
    every row, which would test the backend more than nnact.
    """
    n, batch = 2000, 20
    writer = make_writer()
    for start in range(0, n, batch):
        ids = [f"s{start + i}" for i in range(batch)]
        writer.write(ids, {"l": identifiable(batch, 8, start)})

    store = writer.close()

    assert len(store) == n
    for i in (0, 1, 999, 1000, n - 1):
        assert store[i].activations[0].tensor[0].item() == float(i)
    store.close()


def test_backends_agree(identifiable: ActivationFactory, h5_path: Path) -> None:
    """Memory and HDF5 backends must be indistinguishable to a reader."""
    activations = {"a": identifiable(5, 4), "b": identifiable(5, 3, start=50)}
    ids = [f"s{i}" for i in range(5)]

    mem_writer = MemoryActivationWriter()
    mem_writer.write(ids, activations)
    mem = mem_writer.close()

    h5_writer = H5ActivationWriter(h5_path)
    h5_writer.write(ids, activations)
    h5 = h5_writer.close()

    assert mem.sample_ids == h5.sample_ids
    assert mem.layer_names == h5.layer_names
    assert len(mem) == len(h5)
    for i in range(len(mem)):
        for left, right in zip(mem[i].activations, h5[i].activations, strict=True):
            assert left.layer_name == right.layer_name
            assert torch.equal(left.tensor, right.tensor)
    h5.close()


def test_rejects_batch_size_mismatch(
    make_writer: WriterFactory, identifiable: ActivationFactory
) -> None:
    """A tensor whose leading axis disagrees with the ID count is rejected."""
    writer = make_writer()
    with pytest.raises(ValueError, match="does not match 3 sample ids"):
        writer.write(["a", "b", "c"], {"l": identifiable(2, 4)})


def test_rejects_inconsistent_layers(
    make_writer: WriterFactory, identifiable: ActivationFactory
) -> None:
    """The layer set must not change mid-run, in either direction.

    A layer that appears or vanishes partway would leave its activations
    misaligned against the sample IDs.
    """
    writer = make_writer()
    writer.write(["a"], {"x": identifiable(1, 4), "y": identifiable(1, 4)})

    with pytest.raises(AssertionError, match="already holds"):
        writer.write(["b"], {"x": identifiable(1, 4)})

    with pytest.raises(AssertionError, match="already holds"):
        writer.write(
            ["c"],
            {"x": identifiable(1, 4), "y": identifiable(1, 4), "z": identifiable(1, 4)},
        )


def test_store_rejects_length_mismatch(h5_path: Path) -> None:
    """Both stores refuse to construct when a layer disagrees with the IDs."""
    with pytest.raises(AssertionError, match="but there are 3 sample ids"):
        MemoryActivationStore(
            activations={"l": torch.zeros(5, 4)}, sample_ids=["a", "b", "c"]
        )

    writer = H5ActivationWriter(h5_path)
    writer.write(["a", "b", "c"], {"l": torch.zeros(3, 4)})
    writer.close().close()

    with pytest.raises(AssertionError, match="but there are 2 sample ids"):
        H5ActivationStore(path=h5_path, layer_names=["l"], sample_ids=["a", "b"])


def test_load_accepts_matching_ids(
    h5_path: Path, identifiable: ActivationFactory
) -> None:
    """A cache written from the same IDs, in order, loads and reads back."""
    ids = [f"s{i}" for i in range(4)]
    writer = H5ActivationWriter(h5_path)
    writer.write(ids, {"l": identifiable(4, 4)})
    writer.close().close()

    store = H5ActivationStore.load(h5_path, ids)

    assert store.sample_ids == ids
    assert store.layer_names == ["l"]
    assert store[2].activations[0].tensor[0].item() == 2.0
    store.close()


def test_load_rejects_reordered_ids(
    h5_path: Path, identifiable: ActivationFactory
) -> None:
    """The same IDs in a different order must fail.

    Activations are stored positionally, so a reordered dataset would misalign
    with the cache while looking superficially valid.
    """
    ids = [f"s{i}" for i in range(4)]
    writer = H5ActivationWriter(h5_path)
    writer.write(ids, {"l": identifiable(4, 4)})
    writer.close().close()

    reordered = [ids[1], ids[0], ids[2], ids[3]]
    with pytest.raises(ValueError, match="Hash mismatch"):
        H5ActivationStore.load(h5_path, reordered)


def test_load_rejects_unclosed_file(
    h5_path: Path, identifiable: ActivationFactory
) -> None:
    """An interrupted run must not leave a cache that loads successfully.

    Sample IDs and their hash are written by close(), so a file whose writer
    never closed is missing them.
    """
    writer = H5ActivationWriter(h5_path)
    writer.write(["a", "b"], {"l": identifiable(2, 4)})

    with pytest.raises((KeyError, ValueError, OSError)):
        H5ActivationStore.load(h5_path, ["a", "b"])


def test_close_then_read_reopens(
    h5_path: Path, identifiable: ActivationFactory
) -> None:
    """close() is idempotent, and a later read transparently reopens."""
    writer = H5ActivationWriter(h5_path)
    writer.write(["a", "b"], {"l": identifiable(2, 4)})
    store = writer.close()

    store.close()
    store.close()

    assert store[1].activations[0].tensor[0].item() == 1.0
    store.close()


def test_activations_property_is_stacked(
    identifiable: ActivationFactory,
) -> None:
    """The memory store exposes whole stacked tensors for bulk analysis."""
    writer = MemoryActivationWriter()
    writer.write(["a", "b", "c"], {"l": identifiable(3, 4)})
    store = writer.close()

    stacked = store.activations["l"]
    assert stacked.shape == (3, 4)
    assert torch.equal(stacked[:, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_summary_tabulates_layers(written_store: StoreFactory) -> None:
    """summary() reports per-sample shape and total bytes for every layer."""
    store = written_store(
        {"a": torch.zeros(4, 8), "b": torch.zeros(4, 2, 3)}, ["w", "x", "y", "z"]
    )

    frame = store.summary()

    assert list(frame.index) == ["a", "b"]
    assert list(frame.columns) == ["shape", "elements", "bytes"]
    assert frame.loc["a", "shape"] == (8,)
    assert frame.loc["b", "shape"] == (2, 3)
    assert frame.loc["b", "elements"] == 6
    assert frame.loc["b", "bytes"] == 6 * 4 * 4
    assert store.layer_shape("a") == (8,)
    store.close()
