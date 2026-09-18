from __future__ import annotations

import torch
from torch import nn

from nnact._outputs._protocols import SequenceModelInput

VOCAB_SIZE = 11
HIDDEN_SIZE = 4


class FakeOutput:
    """Minimal stand-in for a Hugging Face model output, satisfying ``ModelOutput``."""

    def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None = None) -> None:
        self.logits = logits
        self.loss = loss


class FakeModel(nn.Module):
    """Tiny deterministic model with one named layer (``linear``) to hook."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> FakeOutput:
        del attention_mask
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)
        hidden = self.linear(x)
        logits = hidden.sum(dim=-1, keepdim=True).expand(-1, -1, VOCAB_SIZE)
        return FakeOutput(logits=logits)


def make_model_input(ids: list[int]) -> SequenceModelInput:
    """Build a :class:`SequenceModelInput` for ``ids`` with an all-ones attention mask."""
    return SequenceModelInput(
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.ones(len(ids), dtype=torch.long),
    )
