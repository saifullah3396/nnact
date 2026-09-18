"""DataLoader helpers for activation mapping."""

from typing import Any

from torch.utils.data import DataLoader, Dataset

from nnact._outputs._protocols import ActivationBatch


def activation_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    **kwargs: Any,
) -> DataLoader:
    """Build ordered batches from a dataset yielding ``ActivationSample`` items."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ActivationBatch.from_samples,
        **kwargs,
    )
