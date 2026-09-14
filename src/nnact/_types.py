from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class LayerActivation:
    layer_name: str
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class Sample:
    """A raw input sample for activation collection."""

    id: str
    data: Any


@dataclass(frozen=True, slots=True)
class ActivatedSample:
    """Activations captured from one or more layers for a single sample."""

    activations: list[LayerActivation]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: torch.Tensor
