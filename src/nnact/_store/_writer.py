"""Incremental writers that accumulate activations and hand back a store.

Writers are fed one batch at a time via :meth:`ActivationWriter.write` and
finalised with :meth:`ActivationWriter.close`, which returns the matching
read-only store from :mod:`nnact.store._store`.

Neither implementation needs the sample count up front. Per-layer storage is
created from the shape of the first batch that mentions a layer, and sample IDs
accumulate as batches arrive, so a single pass over the data is enough.

Two implementations are provided:

* :class:`MemoryActivationWriter` buffers batches in RAM.
* :class:`H5ActivationWriter` streams batches into an HDF5 file.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, override

import h5py
import numpy as np
import torch

from nnact._store._keys import (
    HASH_KEY,
    IDS_KEY,
    LAYERS_GROUP,
    METADATA_KEY,
    STR_DTYPE,
    sample_id_hash,
)
from nnact._store._store import (
    ActivationStore,
    H5ActivationStore,
    MemoryActivationStore,
)
from nnact._types import RunMetadata


class ActivationWriter(ABC):
    """Base class for incremental activation writers.

    Handles batch validation and sample counts, delegating storage to
    subclasses through :meth:`_append_batch`. Activations are converted to
    ``float32`` before reaching the backend, so all stores share one dtype
    regardless of the precision the model ran at.

    Writers are single-pass and not thread-safe: batches must be written in the
    positional order the resulting store should expose, from one thread.
    """

    def __init__(self) -> None:
        """Initialise an empty writer."""
        self._sample_count = 0

    @property
    @abstractmethod
    def sample_ids(self) -> list[str]:
        """Sample IDs written so far, in order."""

    @property
    def sample_count(self) -> int:
        """Number of samples written so far."""
        return self._sample_count

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        """Names of the layers seen so far, in first-write order."""

    @abstractmethod
    def _append_batch(self, ids: list[str], activations: dict[str, np.ndarray]) -> None:
        """Append a batch's IDs and layer activations together."""

    def close(self, metadata: RunMetadata | None = None) -> ActivationStore:
        """Finalise the written activations and return a store over them.

        Args:
            metadata: Optional run metadata to associate with the finished
                store. The writer persists it using its backend.

        Returns:
            A read-only store exposing every sample written so far.
        """
        return self._close(metadata)

    @abstractmethod
    def _close(self, metadata: RunMetadata | None) -> ActivationStore:
        """Finalize this writer's backend and return its read-only store."""

    def write(self, ids: list[str], activations: dict[str, torch.Tensor]) -> None:
        """Append one batch of activations across all layers.

        Args:
            ids: Sample IDs for this batch, in positional order.
            activations: Maps layer name to a tensor of shape
                ``(len(ids), *act_shape)``. Tensors are detached, moved to CPU
                and cast to ``float32``; the originals are left untouched.

        Raises:
            AssertionError: If this batch's layer set differs from earlier
                batches. A layer appearing or vanishing mid-run would leave its
                activations misaligned with the sample IDs.
            ValueError: If a tensor's leading axis does not equal ``len(ids)``.

        Note:
            Validation completes before the batch is appended. A backend write
            failure may still leave partial output; discard the writer then.
        """
        assert set(activations) == set(self.layer_names) or not self._sample_count, (
            f"Batch has layers {sorted(activations)} but writer already holds "
            f"{sorted(self.layer_names)}."
        )
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in activations.items():
            act = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            if act.shape[0] != len(ids):
                raise ValueError(
                    f"Layer '{name}' batch size {act.shape[0]} does not match "
                    f"{len(ids)} sample ids."
                )
            arrays[name] = act
        self._append_batch(ids, arrays)
        self._sample_count += len(ids)


@final
class MemoryActivationWriter(ActivationWriter):
    """Writer that buffers activations in RAM.

    Batches are held as separate arrays and concatenated once, on
    :meth:`close`.

    Warning:
        Peak memory is roughly twice the final activation size, since the
        per-batch buffers are still alive while :func:`numpy.concatenate`
        builds the combined array. Use :class:`H5ActivationWriter` for runs
        that approach the memory limit.

    Example:
        >>> writer = MemoryActivationWriter()
        >>> writer.write(["a", "b"], {"fc1": torch.randn(2, 8)})
        >>> store = writer.close()
        >>> len(store)
        2
    """

    def __init__(self) -> None:
        """Initialise a writer with no buffered batches."""
        super().__init__()
        self._batches: list[tuple[list[str], dict[str, np.ndarray]]] = []

    @property
    @override
    def sample_ids(self) -> list[str]:
        return [sample_id for ids, _ in self._batches for sample_id in ids]

    @property
    @override
    def layer_names(self) -> list[str]:
        """Names of the layers seen so far, in first-write order."""
        return list(self._batches[0][1]) if self._batches else []

    @override
    def _append_batch(self, ids: list[str], activations: dict[str, np.ndarray]) -> None:
        self._batches.append((ids.copy(), activations))

    @override
    def _close(self, metadata: RunMetadata | None) -> MemoryActivationStore:
        """Concatenate the buffered batches into a store.

        The writer is not reset and should not be reused afterwards.

        Returns:
            A store holding one stacked tensor per layer.
        """
        activations = {
            name: torch.from_numpy(
                np.concatenate([batch[name] for _, batch in self._batches], axis=0)
            )
            for name in self.layer_names
        }
        return MemoryActivationStore(
            activations=activations,
            sample_ids=self.sample_ids,
            metadata=metadata,
        )


@final
class H5ActivationWriter(ActivationWriter):
    """Writer that streams activations into an HDF5 file.

    Each layer gets a resizable dataset created from the first batch's shape
    and extended in place as batches arrive, so memory use stays flat in the
    number of samples. The file is created — and truncated, if it already
    exists — on construction, and closed by :meth:`close`.

    Sample IDs are appended to the HDF5 file along with each batch. Their hash
    is written on :meth:`close`, so an interrupted run does not leave a cache
    that validates through :meth:`~nnact.store._store.H5ActivationStore.load`.

    Example:
        >>> writer = H5ActivationWriter(Path("acts.h5"))  # doctest: +SKIP
        >>> writer.write(["a", "b"], {"fc1": torch.randn(2, 8)})  # doctest: +SKIP
        >>> store = writer.close()  # doctest: +SKIP
    """

    def __init__(self, path: Path) -> None:
        """Create the backing file and its layer group.

        Args:
            path: Destination for the HDF5 file. Parent directories are created
                as needed, and an existing file at this path is truncated.

        Raises:
            OSError: If the directory cannot be created or the file cannot be
                opened for writing.
        """
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")
        self._layers = self._file.create_group(LAYERS_GROUP)
        self._datasets: dict[str, h5py.Dataset] = {}
        self._ids_dataset = self._file.create_dataset(
            IDS_KEY, shape=(0,), maxshape=(None,), dtype=STR_DTYPE
        )

    @property
    def path(self) -> Path:
        """Path of the file being written."""
        return self._path

    @property
    @override
    def layer_names(self) -> list[str]:
        """Names of the layers seen so far, in first-write order."""
        return list(self._datasets)

    @property
    @override
    def sample_ids(self) -> list[str]:
        """Read IDs already written to the HDF5 dataset."""
        return list(self._ids_dataset.asstr()[...])

    def _dataset(self, name: str, shape: tuple[int, ...]) -> h5py.Dataset:
        """Return the dataset for ``name``, creating it on first use.

        The dataset starts empty and is resizable along its leading axis, which
        is what lets the writer run without knowing the sample count up front.

        Args:
            name: Layer name.
            shape: Per-sample activation shape, excluding the batch axis.

        Returns:
            A resizable ``float32`` dataset of shape ``(n_written, *shape)``.
        """
        ds = self._datasets.get(name)
        if ds is None:
            ds = self._layers.create_dataset(
                f"{name}/activations",
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=np.float32,
            )
            self._datasets[name] = ds
        return ds

    @override
    def _append_batch(self, ids: list[str], activations: dict[str, np.ndarray]) -> None:
        for name, act in activations.items():
            ds = self._dataset(name, act.shape[1:])
            start = ds.shape[0]
            ds.resize(start + act.shape[0], axis=0)
            ds[start : start + act.shape[0]] = act

        start = self._ids_dataset.shape[0]
        self._ids_dataset.resize(start + len(ids), axis=0)
        self._ids_dataset[start : start + len(ids)] = ids

    @override
    def _close(self, metadata: RunMetadata | None) -> H5ActivationStore:
        """Write the sample ID hash, close the file, and return a store.

        The writer must not be used afterwards; the file handle is closed and
        further writes will fail.

        Returns:
            A store reading back the file just written.

        Raises:
            OSError: If the trailing metadata cannot be written or the file
                cannot be closed cleanly.
        """
        ids = self.sample_ids
        self._file.attrs[HASH_KEY] = sample_id_hash(ids)
        self._file.attrs[METADATA_KEY] = json.dumps(
            (metadata or RunMetadata()).to_dict(), default=str
        )
        layer_names = self.layer_names
        self._file.close()
        return H5ActivationStore(
            path=self._path, layer_names=layer_names, sample_ids=ids
        )
