from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, final

from torch import nn
from transformers import PreTrainedTokenizerBase

from nnact._model._hooked import HookedModel
from nnact._outputs._dataset import ActivationDataset
from nnact._steps._accumulator import (
    H5SequenceActivationAccumulator,
    H5TokenActivationAccumulator,
    InMemorySequenceActivationAccumulator,
    InMemoryTokenActivationAccumulator,
)
from nnact._steps._runner import ActivationStepRunner


@final
class ActivationPipeline:
    def __init__(
        self,
        model: nn.Module,
        layer_names: str | list[str],
        output_type: Literal["sequence", "token"],
        device: Any | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        show_progress: bool = True,
        cache_outputs: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        assert output_type in ("sequence", "token"), (
            f"Unknown output_type '{output_type}', expected 'sequence' or 'token'."
        )

        hooked_model = HookedModel(model=model)
        names = [layer_names] if isinstance(layer_names, str) else list(layer_names)

        self._accumulator = self._build_accumulator(
            output_type=output_type, cache_outputs=cache_outputs, cache_dir=cache_dir
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
        self,
        *,
        output_type: Literal["sequence", "token"],
        cache_outputs: bool,
        cache_dir: str | Path | None,
    ) -> (
        InMemorySequenceActivationAccumulator
        | InMemoryTokenActivationAccumulator
        | H5SequenceActivationAccumulator
        | H5TokenActivationAccumulator
    ):
        if cache_outputs:
            assert cache_dir is not None, (
                "cache_dir is required when cache_outputs=True."
            )
            path = Path(cache_dir) / "activations.h5"
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
        )

    def run(self, loader: Iterable[Mapping[str, Any]]) -> ActivationDataset:
        self._runner.run(loader)
        return self._accumulator.dataset
