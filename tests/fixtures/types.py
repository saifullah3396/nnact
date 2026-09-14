"""Type aliases for the factory fixtures.

Several fixtures return callables so a test can vary sample count or shape.
Naming those signatures here keeps the annotations in test modules short and
gives one place to change them.
"""

from collections.abc import Callable

import torch

from nnact.store import ActivationStore, ActivationWriter

ActivationFactory = Callable[..., torch.Tensor]
"""Builds an activation tensor: ``(n, dim=4, start=0) -> Tensor``."""

WriterFactory = Callable[[], ActivationWriter]
"""Constructs a fresh writer for the backend under test."""

StoreFactory = Callable[[dict[str, torch.Tensor], list[str]], ActivationStore]
"""Writes one batch through a backend and returns the finished store."""
