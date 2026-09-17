import hashlib

import h5py

HASH_KEY = "sample_id_hash"
IDS_KEY = "sample_ids"
LAYERS_GROUP = "layers"
METADATA_KEY = "metadata"
STR_DTYPE = h5py.string_dtype(encoding="utf-8")


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
