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

    def dump(self) -> dict[str, float]:
        """Summarize this result as one flat row for a results table.

        Returns:
            ``acc``, ``balanced_acc``, ``nll``, ``roc_auc_macro``, and
            confidence stats (mean confidence overall, and split by whether
            the prediction was correct) -- everything derived from
            :attr:`y_true` and :attr:`probabilities` alone, so it needs no
            extra arguments.
        """
        from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

        confidence = self.probabilities[np.arange(len(self.y_true)), self.predictions]
        correct = self.predictions == self.y_true
        labels = np.arange(self.probabilities.shape[1])

        try:
            roc_auc_macro = roc_auc_score(
                self.y_true,
                self.probabilities,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        except ValueError:
            roc_auc_macro = float("nan")

        return {
            "acc": self.accuracy,
            "balanced_acc": float(balanced_accuracy_score(self.y_true, self.predictions)),
            "nll": float(log_loss(self.y_true, self.probabilities, labels=labels)),
            "roc_auc_macro": float(roc_auc_macro),
            "mean_confidence": float(confidence.mean()),
            "mean_confidence_correct": float(confidence[correct].mean())
            if correct.any()
            else float("nan"),
            "mean_confidence_incorrect": float(confidence[~correct].mean())
            if (~correct).any()
            else float("nan"),
        }
