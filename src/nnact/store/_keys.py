"""HDF5 layout constants shared by the writer and store implementations.

This module is the single definition of the on-disk format. Both
:mod:`nnact.store._writer` and :mod:`nnact.store._store` import from here so
that the layout cannot drift between the code that produces a file and the
code that reads it back.

File layout::

    /                                   (root)
      .attrs["sample_id_hash"]          SHA-256 over the ordered sample IDs
      sample_ids                        1-D variable-length UTF-8 string dataset
      layers/                           (group)
        <layer_name>/
          activations                   float32, shape (n_samples, *act_shape)

The root ``sample_id_hash`` attribute and the ``sample_ids`` dataset together
let a reader verify that a cache was built from the same samples, in the same
order, as the dataset it is about to be used with.
"""

import hashlib

import h5py

HASH_KEY = "sample_id_hash"
"""Root attribute holding the SHA-256 digest of the ordered sample IDs."""

IDS_KEY = "sample_ids"
"""Root dataset holding the sample IDs in write order."""

LAYERS_GROUP = "layers"
"""Root group containing one subgroup per captured layer."""

METADATA_KEY = "metadata"
"""Root attribute holding run metadata as a JSON object."""

STR_DTYPE = h5py.string_dtype(encoding="utf-8")
"""Variable-length UTF-8 dtype used for the sample ID dataset."""


def sample_id_hash(ids: list[str]) -> str:
    """Compute a SHA-256 digest over an ordered sequence of sample IDs.

    The digest is order-sensitive: the same IDs in a different order produce a
    different hash. This is deliberate, since activations are stored by
    position and a reordered dataset would silently misalign with the cache.

    Args:
        ids: Sample IDs in the order they were written.

    Returns:
        The digest as a lowercase hexadecimal string.

    Example:
        >>> sample_id_hash(["a", "b"]) == sample_id_hash(["b", "a"])
        False
    """
    h = hashlib.sha256()
    for s in ids:
        h.update(s.encode())
    return h.hexdigest()
