"""Read-only views over captured activations.

A store is a :class:`torch.utils.data.Dataset` yielding one
:class:`~nnact._types.ActivatedSample` per index, so it can be fed straight
into a :class:`~torch.utils.data.DataLoader` for downstream probing.

Two implementations are provided:

* :class:`MemoryActivationStore` keeps every activation resident in RAM.
* :class:`H5ActivationStore` reads one sample at a time from an HDF5 file.

Both are produced by the corresponding writer in :mod:`nnact.store._writer`;
construct them directly only when you already hold the underlying data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, override

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from nnact._types import ActivatedSample, LayerActivation
from nnact.store._keys import HASH_KEY, IDS_KEY, LAYERS_GROUP, sample_id_hash


def _dataset(f: h5py.File, key: str) -> h5py.Dataset:
    """Fetch ``key`` from ``f``, narrowing the result to a dataset.

    ``h5py`` returns a union of group, dataset and datatype from ``__getitem__``.
    This helper collapses that union so callers can index the result, and turns
    a malformed file into a clear error rather than an attribute error raised
    somewhere deeper.

    Args:
        f: An open HDF5 file.
        key: Path of the node within the file.

    Returns:
        The node at ``key``.

    Raises:
        KeyError: If ``key`` is absent from the file.
        TypeError: If the node exists but is not a dataset.
    """
    node = f[key]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(
            f"'{key}' in {f.filename} is a {type(node).__name__}, not a Dataset."
        )
    return node


def _group(f: h5py.File, key: str) -> h5py.Group:
    """Fetch ``key`` from ``f``, narrowing the result to a group.

    The group counterpart of :func:`_dataset`; see that function for why the
    narrowing is needed.

    Args:
        f: An open HDF5 file.
        key: Path of the node within the file.

    Returns:
        The node at ``key``.

    Raises:
        KeyError: If ``key`` is absent from the file.
        TypeError: If the node exists but is not a group.
    """
    node = f[key]
    if not isinstance(node, h5py.Group):
        raise TypeError(
            f"'{key}' in {f.filename} is a {type(node).__name__}, not a Group."
        )
    return node


class ActivationStore(Dataset[ActivatedSample], ABC):
    """Base class for positional, read-only access to captured activations.

    Subclasses expose activations for ``n`` samples across one or more layers,
    indexed by position ``0..n-1``. Position ``i`` corresponds to
    ``sample_ids[i]``; the store itself does not index by ID.

    Implementations must guarantee that every layer holds exactly
    ``len(sample_ids)`` activations.
    """

    @property
    @abstractmethod
    def sample_ids(self) -> list[str]:
        """Sample IDs in positional order."""

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        """Names of the layers held by this store."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the store."""

    @abstractmethod
    def __getitem__(self, idx: int) -> ActivatedSample:
        """Return the activations for the sample at ``idx``.

        Args:
            idx: Zero-based position of the sample.

        Returns:
            One :class:`~nnact._types.ActivatedSample` carrying a
            :class:`~nnact._types.LayerActivation` per layer, ordered to match
            :attr:`layer_names`.
        """

    def close(self) -> None:
        """Release any resources held by the store.

        The base implementation is a no-op, which suits stores backed purely by
        memory. Subclasses holding file handles or other operating-system
        resources override this. Always safe to call, and safe to call twice.
        """


@final
class MemoryActivationStore(ActivationStore):
    """Activations held entirely in memory.

    Suitable when the full activation tensor comfortably fits in RAM. Indexing
    is a cheap tensor slice, and :attr:`activations` gives direct access to the
    whole stacked tensor per layer, which is usually what downstream analysis
    wants.

    Example:
        >>> store = MemoryActivationStore(
        ...     activations={"fc1": torch.randn(4, 8)},
        ...     sample_ids=["a", "b", "c", "d"],
        ... )
        >>> len(store)
        4
        >>> store[0].activations[0].tensor.shape
        torch.Size([8])
    """

    def __init__(
        self, activations: dict[str, torch.Tensor], sample_ids: list[str]
    ) -> None:
        """Initialise the store from stacked per-layer tensors.

        Args:
            activations: Maps layer name to a tensor of shape
                ``(n_samples, *act_shape)``. Not copied; the caller should not
                mutate it afterwards.
            sample_ids: Sample IDs in the same positional order as the leading
                axis of every tensor in ``activations``.

        Raises:
            AssertionError: If any layer's leading axis does not equal
                ``len(sample_ids)``.
        """
        for name, tensor in activations.items():
            assert tensor.shape[0] == len(sample_ids), (
                f"Layer '{name}' has {tensor.shape[0]} activations "
                f"but there are {len(sample_ids)} sample ids."
            )
        self._activations = activations
        self._sample_ids = sample_ids

    @property
    @override
    def sample_ids(self) -> list[str]:
        """Sample IDs in positional order."""
        return self._sample_ids

    @property
    @override
    def layer_names(self) -> list[str]:
        """Names of the layers held by this store, in insertion order."""
        return list(self._activations)

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        """The stacked activation tensor for each layer.

        Returns the live mapping rather than a copy, so treat it as read-only.
        """
        return self._activations

    @override
    def __len__(self) -> int:
        return len(self._sample_ids)

    @override
    def __getitem__(self, idx: int) -> ActivatedSample:
        return ActivatedSample(
            activations=[
                LayerActivation(layer_name=name, tensor=tensor[idx])
                for name, tensor in self._activations.items()
            ]
        )


@final
class H5ActivationStore(ActivationStore):
    """Activations read lazily from an HDF5 file.

    Each :meth:`__getitem__` reads a single sample's activations from disk, so
    the store scales to caches far larger than RAM. The file handle is opened
    on construction and reused for the store's lifetime; call :meth:`close`
    when finished.

    Prefer :meth:`load` over the constructor: it verifies that the cache on
    disk was built from the sample IDs you expect.

    Note:
        The open handle is not inherited safely across ``fork``. When used as
        the dataset of a :class:`~torch.utils.data.DataLoader` with
        ``num_workers > 0``, call :meth:`close` before iterating so each worker
        reopens the file lazily.
    """

    def __init__(
        self, path: Path, layer_names: list[str], sample_ids: list[str]
    ) -> None:
        """Open an existing activation file without verifying its sample IDs.

        Args:
            path: Path to an HDF5 file laid out as described in
                :mod:`nnact.store._keys`.
            layer_names: Layers to expose, which also fixes the order of
                activations returned by :meth:`__getitem__`.
            sample_ids: Sample IDs in positional order.

        Raises:
            AssertionError: If any layer's stored activation count does not
                equal ``len(sample_ids)``.
            TypeError: If a required node exists but has the wrong HDF5 type.
            KeyError: If a requested layer is absent from the file.
            OSError: If the file cannot be opened.
        """
        self._path = path
        self._layer_names = layer_names
        self._sample_ids = sample_ids
        self._file: h5py.File | None = None

        f = self._open()
        for name in layer_names:
            stored = _dataset(f, f"{LAYERS_GROUP}/{name}/activations").shape[0]
            assert stored == len(sample_ids), (
                f"Layer '{name}' has {stored} activations "
                f"but there are {len(sample_ids)} sample ids."
            )

    @property
    def path(self) -> Path:
        """Path of the backing HDF5 file."""
        return self._path

    @property
    @override
    def sample_ids(self) -> list[str]:
        """Sample IDs in positional order."""
        return self._sample_ids

    @property
    @override
    def layer_names(self) -> list[str]:
        """Names of the layers exposed by this store."""
        return self._layer_names

    def _open(self) -> h5py.File:
        """Return the cached file handle, reopening it if it has been closed.

        Returns:
            An HDF5 file handle open for reading.

        Raises:
            OSError: If the file cannot be opened.
        """
        if self._file is None or not self._file.id.valid:
            self._file = h5py.File(self._path, "r")
        return self._file

    @override
    def __len__(self) -> int:
        return len(self._sample_ids)

    @override
    def __getitem__(self, idx: int) -> ActivatedSample:
        f = self._open()
        activations = [
            LayerActivation(
                layer_name=name,
                tensor=torch.from_numpy(
                    np.asarray(_dataset(f, f"{LAYERS_GROUP}/{name}/activations")[idx])
                ),
            )
            for name in self._layer_names
        ]
        return ActivatedSample(activations=activations)

    @override
    def close(self) -> None:
        """Close the backing file handle.

        Idempotent. A later :meth:`__getitem__` transparently reopens the file,
        so closing a store does not make it unusable.
        """
        if self._file is not None and self._file.id.valid:
            self._file.close()
        self._file = None

    @classmethod
    def load(cls, path: Path, expected_ids: list[str]) -> "H5ActivationStore":
        """Open a cache and verify it matches the sample IDs you expect.

        Activations are stored by position, so a cache built from a different
        dataset — or the same dataset in a different order — would silently
        misalign. This checks the stored hash against ``expected_ids`` before
        handing back a store, making a stale cache a loud failure.

        Args:
            path: Path to an HDF5 file written by
                :class:`~nnact.store._writer.H5ActivationWriter`.
            expected_ids: Sample IDs, in order, of the dataset the cache will
                be used with.

        Returns:
            A store exposing every layer found in the file.

        Raises:
            ValueError: If the stored sample IDs differ from ``expected_ids``,
                in content or in order.
            TypeError: If a required node exists but has the wrong HDF5 type.
            KeyError: If the file is missing the sample ID dataset or the layer
                group.
            OSError: If the file cannot be opened.

        Example:
            >>> ids = [s.id for s in dataset]  # doctest: +SKIP
            >>> store = H5ActivationStore.load(Path("acts.h5"), ids)  # doctest: +SKIP
        """
        with h5py.File(path, "r") as f:
            stored_hash = str(f.attrs[HASH_KEY])
            stored_ids: list[str] = [
                s.decode() if isinstance(s, bytes) else s
                for s in _dataset(f, IDS_KEY)[:]
            ]
            layer_names: list[str] = list(_group(f, LAYERS_GROUP).keys())

        expected_hash = sample_id_hash(expected_ids)
        if stored_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch: stored={stored_hash[:8]}… expected={expected_hash[:8]}…\n"
                "The cache was built from a different set or ordering of sample IDs."
            )
        if stored_ids != expected_ids:
            raise ValueError(
                "Sample ID list mismatch despite matching hash (collision?)."
            )

        return cls(path=path, layer_names=layer_names, sample_ids=stored_ids)
