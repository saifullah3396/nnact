from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    def save(self, path: str | Path) -> None:
        """Serialize to an ``.npz`` file, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            accuracy=self.accuracy,
            predictions=self.predictions,
            probabilities=self.probabilities,
            y_true=self.y_true,
            num_train_rows=self.num_train_rows,
            num_test_rows=self.num_test_rows,
        )

    @classmethod
    def load(cls, path: str | Path) -> ProbeResult:
        """Load a :class:`ProbeResult` previously written by :meth:`save`."""
        with np.load(path) as data:
            return cls(
                accuracy=float(data["accuracy"]),
                predictions=data["predictions"],
                probabilities=data["probabilities"],
                y_true=data["y_true"],
                num_train_rows=int(data["num_train_rows"]),
                num_test_rows=int(data["num_test_rows"]),
            )
