from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nnact._outputs._utils import _assert_leading_shape, _assert_shape


@dataclass(frozen=True, kw_only=True)
class SequenceActivationOutput:
    """One batch's worth of activations, one row per sample.

    Returned by :class:`~nnact._steps._sequence.SequenceActivationStep` for
    each batch, and by
    :meth:`~nnact._outputs._token.TokenActivationOutput.from_sequence`'s
    ``output`` argument as the pre-masking intermediate a token-level batch
    is built from.

    Attributes:
        prediction: The model's own argmax prediction per sample, per
            position, shape ``(batch_size, sequence_length)``.
        top_probability: Softmax probability of ``prediction``, same shape.
        metadata: Named per-sample values a caller attached to the batch's
            samples (see :class:`~nnact._outputs._protocols.SequenceActivationSample`),
            each shaped ``(batch_size,)`` -- one value per sample, never
            masked. ``nnact`` never interprets the keys or values.
        activations: Captured layer outputs, keyed by layer name, each
            shaped ``(batch_size, sequence_length, *feature)``.
    """

    prediction: np.ndarray
    top_probability: np.ndarray
    metadata: dict[str, np.ndarray] | None = None
    activations: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        """Validate every field's shape is consistent with ``prediction``'s.

        Raises:
            AssertionError: If any field's shape doesn't match
                ``(batch_size, sequence_length)`` (or, for a metadata
                value, just ``(batch_size,)``).
        """
        _assert_shape(name="prediction", tensor=self.prediction, shape=(None, None))

        batch_size, sequence_length = self.prediction.shape[:2]

        _assert_shape(
            name="top_probability",
            tensor=self.top_probability,
            shape=(batch_size, sequence_length),
        )

        if self.metadata is not None:
            for key, value in self.metadata.items():
                _assert_shape(
                    name=f"metadata[{key!r}]", tensor=value, shape=(batch_size,)
                )

        for name, tensor in self.activations.items():
            _assert_leading_shape(
                name=f"activation '{name}'",
                tensor=tensor,
                shape=(batch_size, sequence_length),
            )

    @cached_property
    def batch_size(self) -> int:
        """Number of samples in this batch."""
        return self.prediction.shape[0]

    @cached_property
    def sequence_length(self) -> int:
        """Number of positions per sample, padding included."""
        return self.prediction.shape[1]
