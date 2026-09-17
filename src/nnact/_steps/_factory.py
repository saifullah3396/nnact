from __future__ import annotations

from nnact._steps._sequence import SequenceActivationStep
from nnact._steps._token import TokenActivationStep
from torch import nn
from transformers import PreTrainedModel

StepType = type[SequenceActivationStep] | type[TokenActivationStep]


class ActivationStepFactory:
    """Picks the activation step matching a model's class."""

    def resolve(self, model: nn.Module) -> StepType:
        if isinstance(model, PreTrainedModel):
            return self._resolve_huggingface(model)

        raise TypeError(f"{type(model).__name__} type not supported.")

    def _resolve_huggingface(self, model: PreTrainedModel) -> StepType:
        _STEP_TYPES_BY_SUFFIX: dict[str, StepType] = {
            "ForTokenClassification": TokenActivationStep,
            "ForCausalLM": TokenActivationStep,
            "ForSequenceClassification": SequenceActivationStep,
        }

        name = type(model).__name__
        architectures = (
            getattr(getattr(model, "config", None), "architectures", None) or []
        )
        candidates = [name, *architectures]

        for suffix, step_type in _STEP_TYPES_BY_SUFFIX.items():
            if any(candidate.endswith(suffix) for candidate in candidates):
                return step_type

        raise TypeError(
            f"No activation step is registered for model class '{name}' "
            f"(architectures={architectures}). Supported suffixes: "
            f"{', '.join(_STEP_TYPES_BY_SUFFIX)}."
        )
