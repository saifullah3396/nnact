from __future__ import annotations

import platform
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, final

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from nnact._logging import get_logger
from nnact._model._hooked import HookedModel
from nnact._outputs._dataset import ActivationDataset
from nnact._outputs._protocols import (
    ActivationSample,
    SequenceActivationBatch,
    TokenActivationBatch,
)
from nnact._steps._accumulator import (
    H5SequenceActivationAccumulator,
    H5TokenActivationAccumulator,
    InMemorySequenceActivationAccumulator,
    InMemoryTokenActivationAccumulator,
)
from nnact._steps._runner import ActivationStepRunner

logger = get_logger(name=__name__)

CACHE_FILE_NAME = "activations.h5"


def activation_loader(
    dataset: Dataset[ActivationSample],
    *,
    output_type: Literal["sequence", "token"],
    batch_size: int,
) -> DataLoader:
    """Build ordered batches from a dataset yielding ``ActivationSample`` items.

    Args:
        dataset: A dataset whose ``__getitem__`` returns
            ``TokenActivationSample`` (for ``output_type="token"``) or
            ``SequenceActivationSample`` (for ``output_type="sequence"``)
            instances -- typically the same dataset passed to
            :meth:`ActivationPipeline.run`.
        output_type: Selects which sample type's ``from_samples`` collates
            each batch -- must match ``dataset``'s own item type.
        batch_size: Number of samples per batch.

    Returns:
        A ``DataLoader`` that never shuffles (activation order must match
        the input dataset's order).
    """
    collate_fn = (
        TokenActivationBatch.from_samples
        if output_type == "token"
        else SequenceActivationBatch.from_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """The outcome of one :meth:`ActivationPipeline.run` call.

    Attributes:
        dataset: The accumulated activations.
        metadata: Facts about the run -- ``output_type``, ``layer_names``, and
            ``num_samples`` are always present. When the run actually executed
            a forward pass, ``model``, ``started_at``,
            ``finished_at``, ``duration_seconds``, and ``env`` are present
            too. Cached runs persist and return this same metadata on later
            calls without loading the model again.
    """

    dataset: ActivationDataset
    metadata: dict[str, object] = field(default_factory=dict)


@final
class ActivationPipeline:
    """Runs a model over a dataset, capturing named layers' activations.

    Wraps a model in a :class:`~nnact._model._hooked.HookedModel`, drives it
    with an Ignite-based :class:`~nnact._steps._runner.ActivationStepRunner`,
    and accumulates the results -- either in memory or streamed to an HDF5
    file -- into an :class:`~nnact._outputs._dataset.ActivationDataset`.

    Example:
        >>> pipeline = ActivationPipeline(
        ...     layer_names=["transformer.h.0"],
        ...     output_type="token",
        ... )  # doctest: +SKIP
        >>> result = pipeline.run(  # doctest: +SKIP
        ...     dataset=dataset, batch_size=8, model_fn=lambda: model
        ... )
        >>> result.dataset.activations("transformer.h.0").shape  # doctest: +SKIP
    """

    def __init__(
        self,
        layer_names: str | list[str],
        output_type: Literal["sequence", "token"],
        cache_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        show_progress: bool = True,
        cache_outputs: bool = False,
    ) -> None:
        """Build a pipeline for one set of layers.

        Args:
            layer_names: One or more module names (as in
                ``model.named_modules()``) whose forward output to capture.
            output_type: ``"token"`` captures one row per real (non-padding)
                token; ``"sequence"`` captures one row per sample.
            cache_dir: Directory to hold the HDF5 cache file, created only
                if it doesn't already exist. Required when
                ``cache_outputs=True``; ignored otherwise, since an
                in-memory run never touches disk.
            device: Where to run the model. Defaults to CUDA if available,
                else CPU.
            tokenizer: Needed only to decode token strings for a ``"token"``
                run's ``tokens`` column; omit to skip that column.
            show_progress: Whether to attach a console progress bar.
            cache_outputs: If ``True``, stream activations to an HDF5 file
                under ``cache_dir`` instead of holding them in memory, and
                resume from that file on a later call if it already exists.

        Raises:
            ValueError: If ``output_type`` isn't ``"sequence"`` or
                ``"token"``, or if ``cache_outputs=True`` but
                ``cache_dir`` is ``None``.
        """
        if output_type not in ("sequence", "token"):
            raise ValueError(
                f"Unknown output_type '{output_type}', expected 'sequence' or 'token'."
            )
        if cache_outputs and cache_dir is None:
            raise ValueError("cache_dir is required when cache_outputs=True.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

        names = [layer_names] if isinstance(layer_names, str) else list(layer_names)

        self._output_type: Literal["sequence", "token"] = output_type
        self._layer_names = names
        self._device = device
        self._tokenizer = tokenizer
        self._show_progress = show_progress
        self._cache_outputs = cache_outputs

    def run(
        self,
        dataset: Dataset[ActivationSample],
        *,
        batch_size: int,
        model_fn: Callable[[], nn.Module],
    ) -> RunResult:
        """Run the model over ``dataset``, capturing the configured layers.

        If this pipeline was built with ``cache_outputs=True`` and its
        cache file already exists (e.g. from an earlier call), ``model_fn``
        is never called -- the existing file's contents are returned as-is.
        Delete the cache file first to force a fresh run and model load.

        Args:
            dataset: Yields ``ActivationSample`` items, in the order
                activations should be captured in.
            batch_size: Number of samples per forward pass.
            model_fn: Zero-argument callable that constructs the model to run.
                It is called only after establishing that no cached output
                exists.

        Returns:
            The accumulated activations and the metadata from the run that
            produced them. A cache hit returns the persisted original metadata.
        """
        if self._cache_outputs:
            assert self._cache_dir is not None  # enforced in __init__
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / CACHE_FILE_NAME
            accumulator = (
                H5SequenceActivationAccumulator(path=cache_path)
                if self._output_type == "sequence"
                else H5TokenActivationAccumulator(path=cache_path)
            )
        else:
            accumulator = (
                InMemorySequenceActivationAccumulator()
                if self._output_type == "sequence"
                else InMemoryTokenActivationAccumulator()
            )

        if accumulator.dataset.exists():
            resumed = accumulator.dataset
            metadata = accumulator.run_metadata
            if metadata is None:
                assert self._cache_dir is not None
                raise RuntimeError(
                    f"Cache at {self._cache_dir / CACHE_FILE_NAME} has no run metadata. "
                    "Delete it and rerun to create a complete cache."
                )
            logger.info(
                "Resuming from existing cache: %d samples already present.",
                len(resumed),
            )
            return RunResult(
                dataset=resumed,
                metadata=metadata,
            )

        model = model_fn()
        model_name = type(model).__name__
        runner = ActivationStepRunner(
            output_type=self._output_type,
            hooked_model=HookedModel(model=model),
            layer_names=self._layer_names,
            device=self._device,
            tokenizer=self._tokenizer,
            handlers=[accumulator],
            show_progress=self._show_progress,
        )

        started_at = datetime.now(UTC)
        logger.info(
            "Starting run: model=%s, output_type=%s",
            model_name,
            self._output_type,
        )

        loader = activation_loader(
            dataset=dataset, output_type=self._output_type, batch_size=batch_size
        )
        _, timer = runner.run(loader=loader)
        result = accumulator.dataset

        finished_at = datetime.now(UTC)
        logger.info("Run finished: %d samples in %.2fs", len(result), timer.value())

        metadata = {
            "model": model_name,
            "output_type": self._output_type,
            "layer_names": self._layer_names,
            "num_samples": len(result),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": timer.value(),
            "env": {
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
            },
        }
        accumulator.save_run_metadata(metadata=metadata)
        return RunResult(dataset=result, metadata=metadata)
