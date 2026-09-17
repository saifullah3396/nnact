from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nnact._outputs._utils import _assert_leading_shape, _assert_shape, _softmax


@dataclass(frozen=True, kw_only=True)
class SequenceActivationOutput:
    logits: np.ndarray
    loss: np.ndarray | None = None
    labels: np.ndarray | None = None
    activations: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        _assert_shape("logits", self.logits, (None, None, None))

        batch_size, sequence_length = self.logits.shape[:2]

        _assert_shape("loss", self.loss, (batch_size, sequence_length))
        _assert_shape("labels", self.labels, (batch_size,))

        for name, tensor in self.activations.items():
            _assert_leading_shape(
                f"activation '{name}'",
                tensor,
                (batch_size, sequence_length),
            )

    @cached_property
    def prediction(self) -> np.ndarray:
        return self.logits.argmax(axis=-1)

    @cached_property
    def probabilities(self) -> np.ndarray:
        return _softmax(self.logits, axis=-1)

    @cached_property
    def batch_size(self) -> int:
        return self.logits.shape[0]

    @cached_property
    def sequence_length(self) -> int:
        return self.logits.shape[1]

    @cached_property
    def num_classes(self) -> int:
        return self.logits.shape[2]
