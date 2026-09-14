from pathlib import Path
from typing import final

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nnact._model import HookedModel
from nnact._types import Sample
from nnact.store import (
    ActivationStore,
    ActivationWriter,
    H5ActivationWriter,
    MemoryActivationWriter,
)


def _collate_samples(batch: list[Sample]) -> tuple[list[str], torch.Tensor]:
    ids = [s.id for s in batch]
    stacked = torch.stack([s.data for s in batch])  # type: ignore[arg-type]
    return ids, stacked


@final
class ActivationMapper:
    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        batch_size: int = 256,
        num_workers: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self._hooked = HookedModel(model)
        self._layer_names = (
            [layer_names] if isinstance(layer_names, str) else layer_names
        )
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._device = device

    def map(
        self,
        dataset: Dataset[Sample],
        path: Path | None = None,
        writer: ActivationWriter | None = None,
    ) -> ActivationStore:
        loader: DataLoader[Sample] = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=_collate_samples,  # type: ignore[arg-type]
        )

        if writer is None:
            writer = (
                MemoryActivationWriter() if path is None else H5ActivationWriter(path)
            )

        self._hooked.eval()
        with torch.no_grad():
            for ids, data in loader:  # type: ignore[misc]
                if self._device is not None:
                    data = data.to(self._device)
                with self._hooked.capture(self._layer_names):
                    self._hooked(data)
                    activations = {
                        name: self._hooked.get_activation(name)
                        for name in self._layer_names
                    }
                    writer.write(ids, activations)

        return writer.close()
