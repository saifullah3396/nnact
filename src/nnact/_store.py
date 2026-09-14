import hashlib
from pathlib import Path
from typing import final, override

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nnact._model import HookedModel
from nnact._types import ActivatedSample, LayerActivation, Sample

_HASH_KEY = "sample_id_hash"
_IDS_KEY = "sample_ids"
_LAYERS_GROUP = "layers"

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


def _sample_id_hash(ids: list[str]) -> str:
    h = hashlib.sha256()
    for s in ids:
        h.update(s.encode())
    return h.hexdigest()


def _collate_samples(batch: list[Sample]) -> tuple[list[Sample], torch.Tensor]:
    stacked = torch.stack([s.data for s in batch])  # type: ignore[arg-type]
    return batch, stacked


@final
class ActivationStore(Dataset[ActivatedSample]):
    """Read-only view over an HDF5 activation cache.

    Indexed by position (0..N-1). Each __getitem__ reads one sample's activations
    from disk — the file stays open but no full-dataset RAM load occurs.
    """

    def __init__(self, path: Path, layer_names: list[str], sample_ids: list[str]) -> None:
        self._path = path
        self._layer_names = layer_names
        self._sample_ids = sample_ids
        self._file: h5py.File | None = None

    @property
    def sample_ids(self) -> list[str]:
        return self._sample_ids

    def _open(self) -> h5py.File:
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
                    np.array(f[f"{_LAYERS_GROUP}/{name}/activations"][idx])
                ),
            )
            for name in self._layer_names
        ]
        return ActivatedSample(activations=activations)

    def close(self) -> None:
        if self._file is not None and self._file.id.valid:
            self._file.close()
        self._file = None

    @classmethod
    def load(cls, path: Path, expected_ids: list[str]) -> "ActivationStore":
        """Open an existing store and verify its sample-ID hash matches expected_ids."""
        with h5py.File(path, "r") as f:
            stored_hash = str(f.attrs[_HASH_KEY])
            stored_ids: list[str] = [
                s.decode() if isinstance(s, bytes) else s for s in f[_IDS_KEY][:]  # type: ignore[index]
            ]
            layer_names: list[str] = list(f[_LAYERS_GROUP].keys())  # type: ignore[index]

        expected_hash = _sample_id_hash(expected_ids)
        if stored_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch: stored={stored_hash[:8]}… expected={expected_hash[:8]}…\n"
                "The cache was built from a different set or ordering of sample IDs."
            )
        if stored_ids != expected_ids:
            raise ValueError("Sample ID list mismatch despite matching hash (collision?).")

        return cls(path=path, layer_names=layer_names, sample_ids=stored_ids)


@final
class ActivationMapper:
    """Runs a model over a Dataset[Sample] and writes activations to an HDF5 file.

    Sample.id (str) values are collected in iteration order, hashed, and stored
    alongside activations so ActivationStore.load() can verify cache consistency.

    Usage:
        store = ActivationMapper(model, "layer4").map(dataset, Path("acts.h5"))
        # Later:
        ids = [dataset[i].id for i in range(len(dataset))]
        store = ActivationStore.load(Path("acts.h5"), ids)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self._hooked = HookedModel(model)
        self._layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._device = device

    def map(self, dataset: Dataset[Sample], path: Path) -> ActivationStore:
        """Run the model over dataset, write activations to path, return an ActivationStore."""
        n = len(dataset)  # type: ignore[arg-type]

        loader: DataLoader[Sample] = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=_collate_samples,  # type: ignore[arg-type]
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Probe one batch to learn per-layer activation shapes.
        act_shapes: dict[str, tuple[int, ...]] = {}
        self._hooked.eval()
        with torch.no_grad():
            first_originals, first_data = next(iter(loader))  # type: ignore[misc]
            if self._device is not None:
                first_data = first_data.to(self._device)
            with self._hooked.capture(self._layer_names):
                self._hooked(first_data)
                for name in self._layer_names:
                    act = self._hooked.get_activation(name).detach().cpu()
                    act_shapes[name] = tuple(act.shape[1:])

        # Collect sample IDs in iteration order — assert str on every Sample.
        all_ids: list[str] = []
        for pos, sample in enumerate(dataset):  # type: ignore[union-attr]
            assert isinstance(sample.id, str), (
                f"Sample.id must be str, got {type(sample.id)} at position {pos}"
            )
            all_ids.append(sample.id)

        with h5py.File(path, "w") as f:
            f.attrs[_HASH_KEY] = _sample_id_hash(all_ids)
            f.create_dataset(_IDS_KEY, data=np.array(all_ids, dtype=object), dtype=_STR_DTYPE)

            grp = f.create_group(_LAYERS_GROUP)
            ds: dict[str, h5py.Dataset] = {}
            for name in self._layer_names:
                shape = act_shapes[name]
                ds[name] = grp.create_dataset(
                    f"{name}/activations",
                    shape=(n, *shape),
                    dtype=np.float32,
                )

            written = 0
            with torch.no_grad():
                for batch_idx, (originals, data) in enumerate(loader):  # type: ignore[misc]
                    if batch_idx == 0:
                        originals = first_originals
                        data = first_data
                    elif self._device is not None:
                        data = data.to(self._device)

                    with self._hooked.capture(self._layer_names):
                        self._hooked(data)
                        for name in self._layer_names:
                            act = self._hooked.get_activation(name).detach().cpu().numpy()
                            bs = act.shape[0]
                            ds[name][written : written + bs] = act

                    written += len(originals)

        return ActivationStore(path=path, layer_names=self._layer_names, sample_ids=all_ids)
