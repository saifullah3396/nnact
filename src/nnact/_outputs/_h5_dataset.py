from __future__ import annotations

from pathlib import Path
from typing import final, override

import h5py
import numpy as np

from nnact._outputs._dataset import ActivationDataset
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput

LOGITS_KEY = "logits"
LOSS_KEY = "loss"
LABELS_KEY = "labels"
TOKEN_IDS_KEY = "token_ids"
TOKENS_KEY = "tokens"
OFFSETS_KEY = "offsets"
LAYERS_GROUP = "activations"
STR_DTYPE = h5py.string_dtype(encoding="utf-8")


class _H5ActivationDataset(ActivationDataset):
    """Shared file handling for the HDF5-backed activation datasets.

    Each batch opens the backing file, appends to it, and closes it again,
    so no file handle is held between batches. Memory use stays flat in the
    number of samples. Construction does not touch the file, so pointing
    this at a path from a previous run reuses that cache as-is; the run's
    :class:`~nnact._steps._accumulator.ActivationAccumulator` is responsible
    for removing it if the run fails (see :meth:`close`).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Path of the backing HDF5 file."""
        return self._path

    def _append_dataset(
        self, file: h5py.File, key: str, values: np.ndarray, dtype: object
    ) -> None:
        ds = file.get(key, None)
        if ds is None:
            ds = file.create_dataset(
                key, shape=(0, *values.shape[1:]), maxshape=(None, *values.shape[1:]), dtype=dtype
            )
        assert isinstance(ds, h5py.Dataset)
        start = ds.shape[0]
        ds.resize(start + values.shape[0], axis=0)
        ds[start : start + values.shape[0]] = values

    def _append_activations(
        self, file: h5py.File, activations: dict[str, np.ndarray]
    ) -> None:
        existing = set(file[LAYERS_GROUP].keys()) if LAYERS_GROUP in file else set()
        assert not existing or set(activations) == existing, (
            f"Batch has layers {sorted(activations)} but dataset already holds "
            f"{sorted(existing)}."
        )
        for name, array in activations.items():
            key = f"{LAYERS_GROUP}/{name}"
            self._append_dataset(
                file, key, array.astype(np.float32, copy=False), np.float32
            )

    def _read_dataset(self, key: str) -> np.ndarray | None:
        with h5py.File(self._path, "r") as file:
            if key not in file:
                return None
            return np.asarray(file[key][...])

    @property
    @override
    def layer_names(self) -> list[str]:
        if not self._path.exists():
            return []
        with h5py.File(self._path, "r") as file:
            group = file.get(LAYERS_GROUP)
            return list(group.keys()) if group is not None else []

    @property
    @override
    def activations(self) -> dict[str, np.ndarray]:
        with h5py.File(self._path, "r") as file:
            group = file.get(LAYERS_GROUP)
            if group is None:
                return {}
            return {name: np.asarray(ds[...]) for name, ds in group.items()}

    def print_cache_info(self) -> None:
        """Print the backing file's path and its current size on disk."""
        size = self._path.stat().st_size if self._path.exists() else 0
        print(f"cache: {self._path} ({size / 1024**2:.1f} MB)")

    def close(self, delete: bool = False) -> None:
        """Optionally remove the backing HDF5 file.

        No file handle is held between batches, so there is nothing to
        flush or close by default; this only matters when ``delete=True``,
        e.g. when the run that was writing the file failed partway through.
        """
        if delete:
            self._path.unlink(missing_ok=True)


@final
class H5SequenceActivationDataset(_H5ActivationDataset):
    """Activations streamed to an HDF5 file from :class:`SequenceActivationOutput` batches."""

    def _add_batch(self, output: SequenceActivationOutput) -> None:
        with h5py.File(self._path, "a") as file:
            self._append_dataset(
                file, LOGITS_KEY, output.logits.astype(np.float32, copy=False), np.float32
            )

            if output.loss is not None:
                self._append_dataset(
                    file, LOSS_KEY, output.loss.astype(np.float32, copy=False), np.float32
                )

            if output.labels is not None:
                self._append_dataset(
                    file, LABELS_KEY, output.labels, output.labels.dtype
                )

            self._append_activations(file, output.activations)

    @override
    def __len__(self) -> int:
        logits = self._read_dataset(LOGITS_KEY)
        return 0 if logits is None else logits.shape[0]

    @property
    def logits(self) -> np.ndarray:
        logits = self._read_dataset(LOGITS_KEY)
        assert logits is not None, "No batches written yet."
        return logits

    @property
    def loss(self) -> np.ndarray | None:
        return self._read_dataset(LOSS_KEY)

    @property
    def labels(self) -> np.ndarray | None:
        return self._read_dataset(LABELS_KEY)


@final
class H5TokenActivationDataset(_H5ActivationDataset):
    """Activations streamed to an HDF5 file from :class:`TokenActivationOutput` batches.

    Padding is masked out before writing, so tokens land in a flat
    ``(n_tokens, ...)`` layout per layer; :attr:`offsets` records where each
    sample's tokens begin and end within that layout.
    """

    def _add_batch(self, output: TokenActivationOutput) -> None:
        with h5py.File(self._path, "a") as file:
            self._append_dataset(
                file, LOGITS_KEY, output.logits.astype(np.float32, copy=False), np.float32
            )

            if output.loss is not None:
                self._append_dataset(
                    file, LOSS_KEY, output.loss.astype(np.float32, copy=False), np.float32
                )

            if output.labels is not None:
                self._append_dataset(
                    file, LABELS_KEY, output.labels, output.labels.dtype
                )

            if output.token_ids is not None:
                self._append_dataset(
                    file, TOKEN_IDS_KEY, output.token_ids, output.token_ids.dtype
                )

            if output.tokens is not None:
                self._append_dataset(file, TOKENS_KEY, output.tokens, STR_DTYPE)

            self._append_activations(file, output.activations)

            existing_offsets = file.get(OFFSETS_KEY)
            base = int(existing_offsets[-1]) if existing_offsets is not None and existing_offsets.shape[0] else 0
            new_offsets = (base + output.offsets[1:]).astype(np.int64)
            self._append_dataset(file, OFFSETS_KEY, new_offsets, np.int64)

    @property
    def offsets(self) -> np.ndarray:
        tail = self._read_dataset(OFFSETS_KEY)
        if tail is None:
            tail = np.zeros(0, dtype=np.int64)
        return np.concatenate([np.zeros(1, dtype=np.int64), tail])

    @override
    def __len__(self) -> int:
        tail = self._read_dataset(OFFSETS_KEY)
        return 0 if tail is None else tail.shape[0]

    @property
    def logits(self) -> np.ndarray:
        logits = self._read_dataset(LOGITS_KEY)
        assert logits is not None, "No batches written yet."
        return logits

    @property
    def loss(self) -> np.ndarray | None:
        return self._read_dataset(LOSS_KEY)

    @property
    def labels(self) -> np.ndarray | None:
        return self._read_dataset(LABELS_KEY)

    @property
    def token_ids(self) -> np.ndarray | None:
        return self._read_dataset(TOKEN_IDS_KEY)

    @property
    def tokens(self) -> np.ndarray | None:
        with h5py.File(self._path, "r") as file:
            if TOKENS_KEY not in file:
                return None
            return np.asarray(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in file[TOKENS_KEY].asstr()[...]
                ]
            )

    @property
    def prediction(self) -> np.ndarray:
        return self.logits.argmax(axis=-1)

    @property
    def sequence_lengths(self) -> np.ndarray:
        offsets = self.offsets
        return offsets[1:] - offsets[:-1]

    @property
    def sample_of_token(self) -> np.ndarray:
        """Which accumulated sample each flat token index belongs to."""
        return np.repeat(np.arange(len(self)), self.sequence_lengths)
