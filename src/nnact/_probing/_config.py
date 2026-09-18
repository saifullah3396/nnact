from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """Hyperparameters for the probe :class:`~nnact._probing._trainer.ProbeTrainer` builds."""

    C: float = 1.0
    max_iter: int = 5_000
    add_scaling: bool = True
    linesearch_max_iter: int = 100
    test_size: float = 0.1
    seed: int = 123
