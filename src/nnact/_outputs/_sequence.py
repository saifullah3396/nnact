from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import torch
from nnact._generator._utils import _assert_shape
from torch.nn import functional as F


@dataclass(frozen=True, kw_only=True)
class SequenceActivationOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        _assert_shape("logits", self.logits, (None, None, None))

        batch_size, sequence_length = self.logits.shape[:2]

        _assert_shape("loss", self.loss, (batch_size, sequence_length))
        _assert_shape("labels", self.labels, (batch_size,))

        for name, tensor in self.activations.items():
            _assert_shape(
                f"activation '{name}'",
                tensor,
                (batch_size, sequence_length, None),
            )

    @cached_property
    def prediction(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    @cached_property
    def probabilities(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    @cached_property
    def batch_size(self) -> int:
        return self.logits.shape[0]

    @cached_property
    def sequence_length(self) -> int:
        return self.logits.shape[1]

    @cached_property
    def num_classes(self) -> int:
        return self.logits.shape[2]
