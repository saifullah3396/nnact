from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ProbeResult:
    """Outcome of fitting and evaluating one probe on a held-out split."""

    accuracy: float
    predictions: np.ndarray
    probabilities: np.ndarray
    y_true: np.ndarray
    num_train_rows: int
    num_test_rows: int
