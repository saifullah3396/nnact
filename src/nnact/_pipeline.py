from __future__ import annotations

import platform
import socket
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
from nnact._outputs._protocols import ActivationBatch, ActivationSample
from nnact._steps._accumulator import (
    H5SequenceActivationAccumulator,
    H5TokenActivationAccumulator,
    InMemorySequenceActivationAccumulator,
    InMemoryTokenActivationAccumulator,
)
from nnact._steps._runner import ActivationStepRunner

logger = get_logger(name=__name__)

CACHE_FILE_NAME = "activations.h5"


def activation_loader(dataset: Dataset[ActivationSample], *, batch_size: int) -> DataLoader:
    """Build ordered batches from a dataset yielding ``ActivationSample`` items.

    Args:
        dataset: A dataset whose ``__getitem__`` returns ``ActivationSample``
            instances -- typically the same dataset passed to
            :meth:`ActivationPipeline.run`.
        batch_size: Number of samples per batch.

    Returns:
        A ``DataLoader`` that never shuffles (activation order must match
        the input dataset's order) and collates samples via
        :meth:`~nnact._outputs._protocols.ActivationBatch.from_samples`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ActivationBatch.from_samples,
    )


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """The outcome of one :meth:`ActivationPipeline.run` call.

    Attributes:
        dataset: The accumulated activations.
        metadata: Facts about the run -- ``model``, ``output_type``,
            ``layer_names``, and ``num_samples`` are always present. When the
            run actually executed a forward pass, ``started_at``,
            ``finished_at``, ``duration_seconds``, and ``env`` are present
            too; when :meth:`ActivationPipeline.run` instead resumed a
            previously cached ``dataset`` without running the model, those
            four keys are absent, since no fresh run produced them. Check
            for ``"duration_seconds"`` to tell the two cases apart.
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
        ...     model=model,
        ...     layer_names=["transformer.h.0"],
        ...     output_type="token",
        ... )  # doctest: +SKIP
        >>> result = pipeline.run(dataset=dataset, batch_size=8)  # doctest: +SKIP
        >>> result.dataset.activations["transformer.h.0"].shape  # doctest: +SKIP
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        output_type: Literal["sequence", "token"],
        cache_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        show_progress: bool = True,
        cache_outputs: bool = False,
    ) -> None:
        """Build a pipeline for one model and one set of layers.

        Args:
            model: The model to run. Wrapped, not mutated -- ``model``
                itself is left untouched.
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

        hooked_model = HookedModel(model=model)
        names = [layer_names] if isinstance(layer_names, str) else list(layer_names)

        self._model_name = type(model).__name__
        self._output_type = output_type
        self._layer_names = names

        self._accumulator = self._build_accumulator(
            output_type=output_type, cache_outputs=cache_outputs
        )
        self._runner = self._build_runner(
            output_type=output_type,
            hooked_model=hooked_model,
            layer_names=names,
            device=device,
            tokenizer=tokenizer,
            show_progress=show_progress,
        )

    def _build_accumulator(
        self, *, output_type: Literal["sequence", "token"], cache_outputs: bool
    ) -> (
        InMemorySequenceActivationAccumulator
        | InMemoryTokenActivationAccumulator
        | H5SequenceActivationAccumulator
        | H5TokenActivationAccumulator
    ):
        """Build the handler that turns per-batch step output into a dataset.

        Args:
            output_type: See :meth:`__init__`.
            cache_outputs: See :meth:`__init__`. When ``True``, creates
                ``self._cache_dir`` if it doesn't exist yet, since this is
                the only code path that actually needs a directory on disk.

        Returns:
            An in-memory accumulator, or an HDF5-backed one under
            ``self._cache_dir / "activations.h5"``.
        """
        if cache_outputs:
            assert self._cache_dir is not None  # enforced in __init__
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / CACHE_FILE_NAME
            match output_type:
                case "sequence":
                    return H5SequenceActivationAccumulator(path=path)
                case "token":
                    return H5TokenActivationAccumulator(path=path)

        match output_type:
            case "sequence":
                return InMemorySequenceActivationAccumulator()
            case "token":
                return InMemoryTokenActivationAccumulator()

    def _build_runner(
        self,
        *,
        output_type: Literal["sequence", "token"],
        hooked_model: HookedModel,
        layer_names: list[str],
        device: torch.device | str,
        tokenizer: PreTrainedTokenizerBase | None,
        show_progress: bool,
    ) -> ActivationStepRunner:
        """Build the Ignite-driven runner that executes the forward passes.

        Args:
            output_type: See :meth:`__init__`.
            hooked_model: The model wrapped for activation capture.
            layer_names: Layers to capture, as passed to :meth:`__init__`.
            device: Device to run the model on.
            tokenizer: See :meth:`__init__`.
            show_progress: See :meth:`__init__`.

        Returns:
            A runner with ``self._accumulator`` already attached as a
            handler, so every completed batch feeds the dataset being built.
        """
        return ActivationStepRunner(
            output_type=output_type,
            hooked_model=hooked_model,
            layer_names=layer_names,
            device=device,
            tokenizer=tokenizer,
            handlers=[self._accumulator],
            show_progress=show_progress,
            log_progress_to_file=False,
        )

    def run(
        self, dataset: Dataset[ActivationSample], *, batch_size: int
    ) -> RunResult:
        """Run the model over ``dataset``, capturing the configured layers.

        If this pipeline was built with ``cache_outputs=True`` and its
        cache file already exists (e.g. from an earlier call), the model is
        never run again -- the existing file's contents are returned as-is.
        Delete the cache file first to force a fresh run.

        Args:
            dataset: Yields ``ActivationSample`` items, in the order
                activations should be captured in.
            batch_size: Number of samples per forward pass.

        Returns:
            The accumulated activations and this run's metadata. See
            :class:`RunResult` for what ``metadata`` contains when this
            call resumed an existing cache instead of running fresh.
        """
        if self._accumulator.dataset.exists():
            resumed = self._accumulator.dataset
            logger.info(
                "Resuming from existing cache: %d samples already present.",
                len(resumed),
            )
            return RunResult(
                dataset=resumed,
                metadata={
                    "model": self._model_name,
                    "output_type": self._output_type,
                    "layer_names": self._layer_names,
                    "num_samples": len(resumed),
                },
            )

        started_at = datetime.now(UTC)
        logger.info(
            "Starting run: model=%s, output_type=%s",
            self._model_name,
            self._output_type,
        )

        loader = activation_loader(dataset=dataset, batch_size=batch_size)
        _, timer = self._runner.run(loader=loader)
        result = self._accumulator.dataset

        finished_at = datetime.now(UTC)
        logger.info("Run finished: %d samples in %.2fs", len(result), timer.value())

        metadata = {
            "model": self._model_name,
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
        return RunResult(dataset=result, metadata=metadata)
