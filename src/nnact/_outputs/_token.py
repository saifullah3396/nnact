from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import torch
from nnact._generator._outputs._sequence import SequenceActivationOutput
from nnact._generator._utils import _assert_shape
from torch.nn import functional as F


@dataclass(frozen=True, kw_only=True)
class TokenActivationOutput:
    logits: torch.Tensor
    offsets: torch.Tensor
    loss: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        _assert_shape("logits", self.logits, (None, None))

        num_tokens = self.logits.shape[0]

        _assert_shape("loss", self.loss, (num_tokens,))
        _assert_shape("labels", self.labels, (num_tokens,))
        _assert_shape("offsets", self.offsets, (None,))

        # Offsets must contain integer indices.
        assert self.offsets.dtype == torch.long

        # Offsets must contain at least the initial boundary.
        assert self.offsets.numel() >= 1

        # The first sequence must start at token index 0.
        assert self.offsets[0].item() == 0

        # Sequence boundaries must be nondecreasing.
        assert torch.all(self.offsets[1:] >= self.offsets[:-1]).item()

        # The final boundary must equal the total number of tokens.
        assert self.offsets[-1].item() == num_tokens

        for name, tensor in self.activations.items():
            _assert_shape(
                f"activation '{name}'",
                tensor,
                (num_tokens, None),
            )

    @classmethod
    def from_sequence(
        cls,
        output: SequenceActivationOutput,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
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

        mask = mask.bool()
        lengths = mask.sum(dim=-1)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])

        return cls(
            logits=output.logits[mask],
            loss=output.loss[mask] if output.loss is not None else None,
            labels=labels[mask] if labels is not None else None,
            offsets=offsets,
            activations={
                name: tensor[mask] for name, tensor in output.activations.items()
            },
        )

    @cached_property
    def prediction(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    @cached_property
    def probabilities(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=-1)

    @cached_property
    def batch_size(self) -> int:
        return self.offsets.numel() - 1

    @cached_property
    def num_tokens(self) -> int:
        return self.logits.shape[0]

    @cached_property
    def sequence_lengths(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @cached_property
    def num_classes(self) -> int:
        return self.logits.shape[1]

    def __len__(self) -> int:
        return self.batch_size
