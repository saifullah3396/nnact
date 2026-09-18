from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ProbeMetrics:
    """Standard classification metrics for one probe's held-out predictions."""

    accuracy: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    confusion_matrix: np.ndarray


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int] | None = None
) -> ProbeMetrics:
    """Score predictions against ground truth with standard sklearn metrics.

    Args:
        y_true: Ground-truth class labels, shape ``(num_rows,)``.
        y_pred: Predicted class labels, shape ``(num_rows,)`` -- typically
            ``probabilities.argmax(axis=-1)`` for a probe fit with
            :class:`~nnact._probing._trainer.ProbeTrainer`.
        labels: Class label order for ``precision``/``recall``/``f1``/
            ``confusion_matrix``. Defaults to the sorted union of labels
            seen in ``y_true`` and ``y_pred``.

    Returns:
        Per-class precision, recall, and F1, plus overall accuracy and the
        confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return ProbeMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
    )
