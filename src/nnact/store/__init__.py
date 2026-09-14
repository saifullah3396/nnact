"""Storage backends for captured activations.

This package separates *writing* activations from *reading* them back:

* A writer (:class:`ActivationWriter`) accumulates batches and returns a store
  when closed. It never needs the sample count up front.
* A store (:class:`ActivationStore`) is a read-only
  :class:`~torch.utils.data.Dataset` over the finished activations.

Each side has a memory-backed and an HDF5-backed implementation, paired so that
closing a writer yields the corresponding store:

======================== ===========================
Writer                   Store returned by ``close``
======================== ===========================
MemoryActivationWriter   MemoryActivationStore
H5ActivationWriter       H5ActivationStore
======================== ===========================

Most callers never construct these directly, since
:class:`~nnact.ActivationMapper` selects a writer and returns its store. Use
them directly to write activations from a source other than a torch model, or
to read back a cache written earlier::

    from nnact.store import H5ActivationStore

    store = H5ActivationStore.load(Path("acts.h5"), [s.id for s in dataset])

The on-disk HDF5 layout is documented in :mod:`nnact.store._keys`.
"""

from nnact.store._store import (
    ActivationStore,
    H5ActivationStore,
    MemoryActivationStore,
)
from nnact.store._writer import (
    ActivationWriter,
    H5ActivationWriter,
    MemoryActivationWriter,
)

__all__ = [
    "ActivationStore",
    "ActivationWriter",
    "H5ActivationStore",
    "H5ActivationWriter",
    "MemoryActivationStore",
    "MemoryActivationWriter",
]
