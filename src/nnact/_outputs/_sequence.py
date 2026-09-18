from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nnact._outputs._utils import _assert_leading_shape, _assert_shape


@dataclass(frozen=True, kw_only=True)
class SequenceActivationOutput:
    prediction: np.ndarray
    top_probability: np.ndarray
    loss: np.ndarray | None = None
    labels: np.ndarray | None = None
    activations: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        _assert_shape("prediction", self.prediction, (None, None))

        batch_size, sequence_length = self.prediction.shape[:2]

        _assert_shape("top_probability", self.top_probability, (batch_size, sequence_length))
        _assert_shape("loss", self.loss, (batch_size, sequence_length))
        _assert_shape("labels", self.labels, (batch_size,))

        for name, tensor in self.activations.items():
            _assert_leading_shape(
                f"activation '{name}'",
                tensor,
                (batch_size, sequence_length),
            )

    @cached_property
    def batch_size(self) -> int:
        return self.prediction.shape[0]

    @cached_property
    def sequence_length(self) -> int:
        return self.prediction.shape[1]
