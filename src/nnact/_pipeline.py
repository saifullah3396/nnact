from __future__ import annotations

import json
import platform
import socket
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, final

from torch import nn
from transformers import PreTrainedTokenizerBase

from nnact._logging import enable_file_logging, get_logger
from nnact._model._hooked import HookedModel
from nnact._outputs._dataset import ActivationDataset
from nnact._steps._accumulator import (
    H5SequenceActivationAccumulator,
    H5TokenActivationAccumulator,
    InMemorySequenceActivationAccumulator,
    InMemoryTokenActivationAccumulator,
)
from nnact._steps._runner import ActivationStepRunner

logger = get_logger(__name__)

RUN_LOG_NAME = "run.log"
RUN_METADATA_NAME = "run_metadata.json"
CACHE_FILE_NAME = "activations.h5"


@final
class ActivationPipeline:
    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        output_type: Literal["sequence", "token"],
        run_dir: str | Path,
        device: Any | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        show_progress: bool = True,
        cache_outputs: bool = False,
    ) -> None:
        assert output_type in ("sequence", "token"), (
            f"Unknown output_type '{output_type}', expected 'sequence' or 'token'."
        )

        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        enable_file_logging(self._run_dir / RUN_LOG_NAME)

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
        if cache_outputs:
            path = self._run_dir / CACHE_FILE_NAME
            match output_type:
                case "sequence":
                    return H5SequenceActivationAccumulator(path)
                case "token":
                    return H5TokenActivationAccumulator(path)

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
        device: Any | None,
        tokenizer: PreTrainedTokenizerBase | None,
        show_progress: bool,
    ) -> ActivationStepRunner:
        return ActivationStepRunner(
            output_type=output_type,
            hooked_model=hooked_model,
            layer_names=layer_names,
            device=device,
            tokenizer=tokenizer,
            handlers=[self._accumulator],
            show_progress=show_progress,
            log_progress_to_file=True,
        )

    def run(self, loader: Iterable[Mapping[str, Any]]) -> ActivationDataset:
        if len(self._accumulator.dataset) > 0:
            return self._accumulator.dataset

        started_at = datetime.now(UTC)
        logger.info(
            "Starting run: model=%s, output_type=%s",
            self._model_name,
            self._output_type,
        )

        _, timer = self._runner.run(loader)
        dataset = self._accumulator.dataset

        finished_at = datetime.now(UTC)
        logger.info("Run finished: %d samples in %.2fs", len(dataset), timer.value())

        self._write_run_metadata(
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=timer.value(),
            num_samples=len(dataset),
        )

        return dataset

    def _write_run_metadata(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        duration_seconds: float,
        num_samples: int,
    ) -> None:
        metadata = {
            "model": self._model_name,
            "output_type": self._output_type,
            "layer_names": self._layer_names,
            "num_samples": num_samples,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "env": {
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
            },
        }
        metadata_path = self._run_dir / RUN_METADATA_NAME
        metadata_path.write_text(json.dumps(metadata, indent=2))
