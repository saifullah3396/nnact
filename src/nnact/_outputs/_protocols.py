from __future__ import annotations

from typing import Protocol

import torch


class ModelOutput(Protocol):
    logits: torch.Tensor
    loss: torch.Tensor | None
