from __future__ import annotations

from pathlib import Path
from typing import final, override

import h5py
import numpy as np

from nnact._outputs._dataset import ActivationDataset
from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._token import TokenActivationOutput

PREDICTION_KEY = "prediction"
TOP_PROBABILITY_KEY = "top_probability"
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

    @override
    def exists(self) -> bool:
        return self._path.exists()

    def _append_dataset(
        self, file: h5py.File, key: str, values: np.ndarray, dtype: object
    ) -> None:
        """Append ``values`` to (creating, if needed) the dataset at ``key``.

        Args:
            file: The open HDF5 file to write into.
            key: Dataset path within ``file``, e.g. ``"labels"`` or
                ``"activations/layer0"``.
            values: Rows to append, along axis 0. A dataset that doesn't
                exist yet is created with this shape's trailing dimensions
                fixed and axis 0 resizable.
            dtype: h5py dtype for a newly created dataset -- typically
                ``np.float32``, ``np.int64``, or ``STR_DTYPE``. Ignored if
                the dataset already exists.
        """
        ds = file.get(key, None)
        if ds is None:
            ds = file.create_dataset(
                key,
                shape=(0, *values.shape[1:]),
                maxshape=(None, *values.shape[1:]),
                dtype=dtype,
            )
        assert isinstance(ds, h5py.Dataset)
        start = ds.shape[0]
        ds.resize(start + values.shape[0], axis=0)
        ds[start : start + values.shape[0]] = values

    def _append_activations(
        self, file: h5py.File, activations: dict[str, np.ndarray]
    ) -> None:
        """Append every layer's activations under ``activations/{layer}``.

        Args:
            file: The open HDF5 file to write into.
            activations: This batch's captured layers, keyed by name. Must
                match the file's existing layer set exactly once one batch
                has been written -- a run can't add or drop layers partway
                through.

        Raises:
            AssertionError: If ``activations``'s keys don't match the
                layers already present in ``file`` (only checked once the
                file already holds at least one batch).
        """
        existing = set(file[LAYERS_GROUP].keys()) if LAYERS_GROUP in file else set()
        assert not existing or set(activations) == existing, (
            f"Batch has layers {sorted(activations)} but dataset already holds "
            f"{sorted(existing)}."
        )
        for name, array in activations.items():
            key = f"{LAYERS_GROUP}/{name}"
            self._append_dataset(
                file=file,
                key=key,
                values=array.astype(np.float32, copy=False),
                dtype=np.float32,
            )

    def _read_dataset(self, key: str) -> np.ndarray | None:
        """Read the full contents of the dataset at ``key``, or ``None`` if absent."""
        with h5py.File(self._path, "r") as file:
            if key not in file:
                return None
            return np.asarray(file[key][...])

    @property
    @override
    def layer_names(self) -> list[str]:
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
        """Open the backing file, append one batch's fields, and close it again."""
        with h5py.File(self._path, "a") as file:
            self._append_dataset(
                file=file,
                key=PREDICTION_KEY,
                values=output.prediction.astype(np.int64, copy=False),
                dtype=np.int64,
            )
            self._append_dataset(
                file=file,
                key=TOP_PROBABILITY_KEY,
                values=output.top_probability.astype(np.float32, copy=False),
                dtype=np.float32,
            )

            if output.loss is not None:
                self._append_dataset(
                    file=file,
                    key=LOSS_KEY,
                    values=output.loss.astype(np.float32, copy=False),
                    dtype=np.float32,
                )

            if output.labels is not None:
                self._append_dataset(
                    file=file, key=LABELS_KEY, values=output.labels, dtype=STR_DTYPE
                )

            self._append_activations(file=file, activations=output.activations)

    @override
    def __len__(self) -> int:
        prediction = self._read_dataset(key=PREDICTION_KEY)
        return 0 if prediction is None else prediction.shape[0]

    @property
    def prediction(self) -> np.ndarray:
        """The model's own argmax prediction, one per sample."""
        prediction = self._read_dataset(key=PREDICTION_KEY)
        assert prediction is not None, "No batches written yet."
        return prediction

    @property
    def top_probability(self) -> np.ndarray:
        """Softmax probability of :attr:`prediction`, one per sample."""
        top_probability = self._read_dataset(key=TOP_PROBABILITY_KEY)
        assert top_probability is not None, "No batches written yet."
        return top_probability

    @property
    def loss(self) -> np.ndarray | None:
        """Per-sample loss, or ``None`` if no batch ever provided one."""
        return self._read_dataset(key=LOSS_KEY)

    @property
    def labels(self) -> np.ndarray | None:
        """Ground-truth label per sample, or ``None`` if none were provided."""
        with h5py.File(self._path, "r") as file:
            if LABELS_KEY not in file:
                return None
            return np.asarray(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in file[LABELS_KEY].asstr()[...]
                ]
            )


@final
class H5TokenActivationDataset(_H5ActivationDataset):
    """Activations streamed to an HDF5 file from :class:`TokenActivationOutput` batches.

    Padding is masked out before writing, so tokens land in a flat
    ``(n_tokens, ...)`` layout per layer; :attr:`offsets` records where each
    sample's tokens begin and end within that layout.
    """

    def _add_batch(self, output: TokenActivationOutput) -> None:
        """Open the backing file, append one batch's fields, and close it again."""
        with h5py.File(self._path, "a") as file:
            self._append_dataset(
                file=file,
                key=PREDICTION_KEY,
                values=output.prediction.astype(np.int64, copy=False),
                dtype=np.int64,
            )
            self._append_dataset(
                file=file,
                key=TOP_PROBABILITY_KEY,
                values=output.top_probability.astype(np.float32, copy=False),
                dtype=np.float32,
            )

            if output.loss is not None:
                self._append_dataset(
                    file=file,
                    key=LOSS_KEY,
                    values=output.loss.astype(np.float32, copy=False),
                    dtype=np.float32,
                )

            if output.labels is not None:
                self._append_dataset(
                    file=file, key=LABELS_KEY, values=output.labels, dtype=STR_DTYPE
                )

            if output.token_ids is not None:
                self._append_dataset(
                    file=file,
                    key=TOKEN_IDS_KEY,
                    values=output.token_ids,
                    dtype=output.token_ids.dtype,
                )

            if output.tokens is not None:
                self._append_dataset(
                    file=file, key=TOKENS_KEY, values=output.tokens, dtype=STR_DTYPE
                )

            self._append_activations(file=file, activations=output.activations)

            existing_offsets = file.get(OFFSETS_KEY)
            base = (
                int(existing_offsets[-1])
                if existing_offsets is not None and existing_offsets.shape[0]
                else 0
            )
            new_offsets = (base + output.offsets[1:]).astype(np.int64)
            self._append_dataset(
                file=file, key=OFFSETS_KEY, values=new_offsets, dtype=np.int64
            )

    @property
    def offsets(self) -> np.ndarray:
        """Sample boundaries into the flat token layout, shape ``(len(self) + 1,)``."""
        tail = self._read_dataset(key=OFFSETS_KEY)
        if tail is None:
            tail = np.zeros(0, dtype=np.int64)
        return np.concatenate([np.zeros(1, dtype=np.int64), tail])

    @override
    def __len__(self) -> int:
        tail = self._read_dataset(key=OFFSETS_KEY)
        return 0 if tail is None else tail.shape[0]

    @property
    def prediction(self) -> np.ndarray:
        """The model's own argmax prediction, one per real token."""
        prediction = self._read_dataset(key=PREDICTION_KEY)
        assert prediction is not None, "No batches written yet."
        return prediction

    @property
    def top_probability(self) -> np.ndarray:
        """Softmax probability of :attr:`prediction`, one per real token."""
        top_probability = self._read_dataset(key=TOP_PROBABILITY_KEY)
        assert top_probability is not None, "No batches written yet."
        return top_probability

    @property
    def loss(self) -> np.ndarray | None:
        """Per-token loss, or ``None`` if no batch ever provided one."""
        return self._read_dataset(key=LOSS_KEY)

    @property
    def labels(self) -> np.ndarray | None:
        """Ground-truth label per real token, or ``None`` if none were provided."""
        with h5py.File(self._path, "r") as file:
            if LABELS_KEY not in file:
                return None
            return np.asarray(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in file[LABELS_KEY].asstr()[...]
                ]
            )

    @property
    def token_ids(self) -> np.ndarray | None:
        """Input token id per real token, or ``None`` if none were provided."""
        return self._read_dataset(key=TOKEN_IDS_KEY)

    @property
    def tokens(self) -> np.ndarray | None:
        """Decoded token string per real token, or ``None`` if none were provided."""
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
    def sequence_lengths(self) -> np.ndarray:
        """Real token count per sample, shape ``(len(self),)``."""
        offsets = self.offsets
        return offsets[1:] - offsets[:-1]

    @property
    def sample_of_token(self) -> np.ndarray:
        """Which accumulated sample each flat token index belongs to."""
        return np.repeat(np.arange(len(self)), self.sequence_lengths)

    @override
    def summary(self) -> pd.DataFrame:
        """Tabulate every real token held, one row per token.

        Returns:
            A :class:`pandas.DataFrame` indexed by flat token position, with
            columns ``sample`` (which accumulated sample the token belongs
            to), ``token_id`` and ``token`` (present only when a tokenizer was
            given to the pipeline), ``predicted_id`` (the model's own argmax
            prediction for that token), ``predicted_probability``, ``label``,
            and one ``{layer}_norm`` column per layer holding that token's
            activation L2 norm.

        Raises:
            ImportError: If pandas is not installed. It is not a dependency of
                ``nnact``; use :attr:`token_ids`, :attr:`tokens`, and
                :attr:`activations` instead.
        """
        import pandas as pd

        offsets = self.offsets
        columns: dict[str, object] = {"sample": self.sample_of_token.tolist()}

        token_ids = self.token_ids
        if token_ids is not None:
            columns["token_id"] = token_ids.tolist()
        tokens = self.tokens
        if tokens is not None:
            columns["token"] = tokens.tolist()

        columns["predicted_id"] = self.prediction.tolist()
        columns["predicted_probability"] = self.top_probability.tolist()
        columns["label"] = self.labels.tolist()

        for name, tensor in self.activations.items():
            columns[f"{name}_norm"] = np.linalg.norm(tensor, axis=-1).tolist()

        return pd.DataFrame(
            columns, index=pd.RangeIndex(int(offsets[-1]), name="token")
        )
