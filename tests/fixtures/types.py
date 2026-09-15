"""Type aliases for the factory fixtures.

Several fixtures return callables so a test can vary sample count or shape.
Naming those signatures here keeps the annotations in test modules short and
gives one place to change them.
"""

from collections.abc import Callable

import torch

ActivationFactory = Callable[..., torch.Tensor]
"""Builds an activation tensor: ``(n, dim=4, start=0) -> Tensor``."""
