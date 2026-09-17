"""DataLoader helpers for activation mapping."""

from typing import Any

from torch.utils.data import DataLoader, Dataset


def activation_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    **kwargs: Any,
) -> DataLoader:
    """Build ordered batches from a dataset yielding ``id`` and model inputs."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
