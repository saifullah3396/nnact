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
LAYERS_GROUP = "activations"
STR_DTYPE = h5py.string_dtype(encoding="utf-8")


class _H5ActivationDataset(ActivationDataset):
    """Shared file handling for the HDF5-backed activation datasets.

    Each batch opens the backing file, appends to it, and closes it again,
    so no file handle is held between batches. Memory use stays flat in the
    number of samples. The file is truncated on construction; the run's
    :class:`~nnact._steps._accumulator.ActivationAccumulator` is responsible
    for removing it if the run fails (see :meth:`close`).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self._path, "w"):
            pass
        self._layer_names: list[str] = []
        self._num_samples = 0

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
        assert not self._layer_names or set(activations) == set(self._layer_names), (
            f"Batch has layers {sorted(activations)} but dataset already holds "
            f"{sorted(self._layer_names)}."
        )
        for name, array in activations.items():
            key = f"{LAYERS_GROUP}/{name}"
            is_new = key not in file
            self._append_dataset(
                file, key, array.astype(np.float32, copy=False), np.float32
            )
            if is_new:
                self._layer_names.append(name)

    def _read_dataset(self, key: str) -> np.ndarray | None:
        with h5py.File(self._path, "r") as file:
            if key not in file:
                return None
            return np.asarray(file[key][...])

    @property
    @override
    def layer_names(self) -> list[str]:
        return list(self._layer_names)

    @property
    @override
    def activations(self) -> dict[str, np.ndarray]:
        with h5py.File(self._path, "r") as file:
            return {
                name: np.asarray(file[f"{LAYERS_GROUP}/{name}"][...])
                for name in self._layer_names
            }

    @override
    def __len__(self) -> int:
        return self._num_samples

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

        self._num_samples += output.logits.shape[0]

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

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._offsets: list[int] = [0]

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

        base = self._offsets[-1]
        self._offsets.extend((base + output.offsets[1:]).tolist())
        self._num_samples = len(self._offsets) - 1

    @property
    def offsets(self) -> np.ndarray:
        return np.asarray(self._offsets, dtype=np.int64)

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
