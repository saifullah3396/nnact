from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nnact._outputs._sequence import SequenceActivationOutput
from nnact._outputs._utils import _assert_leading_shape, _assert_shape, _softmax


@dataclass(frozen=True, kw_only=True)
class TokenActivationOutput:
    logits: np.ndarray
    offsets: np.ndarray
    loss: np.ndarray | None = None
    labels: np.ndarray | None = None
    activations: dict[str, np.ndarray] = field(default_factory=dict)
    token_ids: np.ndarray | None = None
    tokens: np.ndarray | None = None

    def __post_init__(self):
        _assert_shape("logits", self.logits, (None, None))

        num_tokens = self.logits.shape[0]

        _assert_shape("loss", self.loss, (num_tokens,))
        _assert_shape("labels", self.labels, (num_tokens,))
        _assert_shape("offsets", self.offsets, (None,))
        _assert_shape("token_ids", self.token_ids, (num_tokens,))

        if self.tokens is not None:
            assert len(self.tokens) == num_tokens, (
                f"tokens must have {num_tokens} entries, got {len(self.tokens)}"
            )

        for name, tensor in self.activations.items():
            _assert_leading_shape(
                f"activation '{name}'",
                tensor,
                (num_tokens,),
            )

        # Offsets must contain integer indices.
        assert self.offsets.dtype.kind in "iu"

        # Offsets must contain at least the initial boundary.
        assert self.offsets.size >= 1

        # The first sequence must start at token index 0.
        assert self.offsets[0].item() == 0

        # Sequence boundaries must be nondecreasing.
        assert np.all(self.offsets[1:] >= self.offsets[:-1])

        # The final boundary must equal the total number of tokens.
        assert self.offsets[-1].item() == num_tokens

    @classmethod
    def from_sequence(
        cls,
        output: SequenceActivationOutput,
        mask: np.ndarray,
        labels: np.ndarray | None = None,
        token_ids: np.ndarray | None = None,
        tokens: np.ndarray | None = None,
    ) -> TokenActivationOutput:
        _assert_shape(
            "mask",
            mask,
            (output.batch_size, output.sequence_length),
        )
        _assert_shape(
            "labels",
            labels,
            (output.batch_size, output.sequence_length),
        )
        _assert_shape(
            "token_ids",
            token_ids,
            (output.batch_size, output.sequence_length),
        )

        mask = mask.astype(bool)
        lengths = mask.sum(axis=-1)
        offsets = np.concatenate([np.zeros(1, dtype=lengths.dtype), lengths.cumsum()])

        return cls(
            logits=output.logits[mask],
            loss=output.loss[mask] if output.loss is not None else None,
            labels=labels[mask] if labels is not None else None,
            offsets=offsets,
            activations={
                name: tensor[mask] for name, tensor in output.activations.items()
            },
            token_ids=token_ids[mask] if token_ids is not None else None,
            tokens=tokens[mask] if tokens is not None else None,
        )

    @cached_property
    def prediction(self) -> np.ndarray:
        return self.logits.argmax(axis=-1)

    @cached_property
    def probabilities(self) -> np.ndarray:
        return _softmax(self.logits, axis=-1)

    @cached_property
    def batch_size(self) -> int:
        return self.offsets.size - 1

    @cached_property
    def num_tokens(self) -> int:
        return self.logits.shape[0]

    @cached_property
    def sequence_lengths(self) -> np.ndarray:
        return self.offsets[1:] - self.offsets[:-1]

    @cached_property
    def num_classes(self) -> int:
        return self.logits.shape[1]

    def __len__(self) -> int:
        return self.batch_size
