from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nnact._probing._metrics import ProbeMetrics, evaluate_predictions


@dataclass(frozen=True, kw_only=True)
class EvalResult:
    """Outcome of scoring a probe's predictions against ground truth.

    Returned by :meth:`~nnact._probing._trainer.ProbeTrainer.evaluate`.
    """

    predictions: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    classes_: np.ndarray

    @property
    def metrics(self) -> ProbeMetrics:
        return evaluate_predictions(
            self.targets, self.predictions, labels=np.arange(len(self.classes_))
        )


@dataclass(frozen=True, kw_only=True)
class TrainResult:
    """Outcome of fitting a probe, including the fitted estimator itself.

    Returned by :meth:`~nnact._probing._trainer.ProbeTrainer.train`. Unlike
    :class:`EvalResult`, this is cacheable: :meth:`save`/:meth:`load`
    persist the estimator alongside the predictions, so a later
    :meth:`~nnact._probing._trainer.ProbeTrainer.evaluate` can run against it
    without re-fitting.
    """

    predictions: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    estimator: Any
    classes_: np.ndarray

    @property
    def metrics(self) -> ProbeMetrics:
        return evaluate_predictions(
            self.targets, self.predictions, labels=np.arange(len(self.classes_))
        )

    def save(self, path: str | Path) -> None:
        """Pickle this result -- arrays, metrics inputs, and the fitted
        estimator alike -- to one file, creating parent directories as needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> TrainResult:
        """Load a :class:`TrainResult` previously written by :meth:`save`."""
        return pickle.loads(Path(path).read_bytes())
