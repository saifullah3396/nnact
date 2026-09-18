from __future__ import annotations

from pathlib import Path
from typing import Protocol, final

import numpy as np

from nnact._logging import get_logger
from nnact._outputs._dataset import ActivationDataset
from nnact._outputs._utils import _assert_leading_shape
from nnact._probing._config import ProbeConfig
from nnact._probing._result import EvalResult, TrainResult

logger = get_logger(__name__)


class FilterFn(Protocol):
    def __call__(
        self,
        *,
        labels: np.ndarray,
        sample_of_row: np.ndarray,
        token_ids: np.ndarray | None,
    ) -> np.ndarray:
        """Return a boolean mask, shape ``(num_rows,)``, ``True`` to keep."""
        ...


class PoolFn(Protocol):
    def __call__(self, *, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reduce ``(num_rows, seq_len, *feature)`` to ``(num_rows, *feature)``."""
        ...


def _default_pool(*, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return activations[:, 0, ...]


@final
class ProbeTrainer:
    """Fits and evaluates one linear probe on an :class:`ActivationDataset`.

    Example:
        >>> trainer = ProbeTrainer(ProbeConfig(C=0.5))
        >>> result = trainer.train(dataset, "model.layers.12")  # doctest: +SKIP
    """

    def __init__(self, config: ProbeConfig) -> None:
        self._config = config
        self.estimator_: object | None = None

    def train(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> TrainResult:
        x, y, sample_of_row = self._select(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )
        train_mask, test_mask = self._split(sample_of_row)
        return self._fit(x[train_mask], y[train_mask], x[test_mask], y[test_mask])

    def evaluate(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> EvalResult:
        """Score an already-fitted probe against fresh examples.

        Unlike :meth:`train`, every selected row is scored -- there is no
        train/test split, since the probe was already fitted elsewhere
        (typically by an earlier :meth:`train` call on this same trainer).
        """
        assert self.estimator_ is not None, (
            "no fitted estimator; call train() before evaluate()."
        )

        x, y, _sample_of_row = self._select(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )
        predictions, probabilities = self._predict(x)
        return EvalResult(predictions=predictions, probabilities=probabilities, targets=y)

    def load(self, path: str | Path) -> TrainResult:
        """Restore the fitted estimator from a :class:`TrainResult` cached by
        :meth:`~nnact._probing._pipeline.ProbePipeline.train`, so
        :meth:`evaluate` can run against it without calling :meth:`train`
        again first.
        """
        result = TrainResult.load(path)
        self.estimator_ = result.estimator
        return result

    def _select(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        filter_fn: FilterFn | None,
        pool_fn: PoolFn | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = getattr(dataset, "labels", None)
        assert labels is not None, (
            "dataset has no labels; probing requires labels to be attached "
            "to the pipeline run that produced it."
        )

        is_token_level = hasattr(dataset, "sample_of_token")

        if is_token_level:
            assert pool_fn is None, (
                "pool_fn has no effect on a token-level dataset: a token is "
                "already a row, there is nothing to pool across seq_len."
            )
            x = dataset.activations[layer_name]
            y = labels
            sample_of_row = dataset.sample_of_token  # type: ignore[attr-defined]
            token_ids = getattr(dataset, "token_ids", None)

            mask = self._filter_mask(filter_fn, y, sample_of_row, token_ids)
            x, y, sample_of_row = x[mask], y[mask], sample_of_row[mask]
        else:
            raw = dataset.activations[layer_name]
            y = labels
            sample_of_row = np.arange(len(dataset))

            mask = self._filter_mask(filter_fn, y, sample_of_row, None)
            raw, y, sample_of_row = raw[mask], y[mask], sample_of_row[mask]

            pool = pool_fn or _default_pool
            x = pool(activations=raw, labels=y)
            _assert_leading_shape("pooled activations", x, (raw.shape[0],))

        return x, y, sample_of_row

    def _filter_mask(
        self,
        filter_fn: FilterFn | None,
        labels: np.ndarray,
        sample_of_row: np.ndarray,
        token_ids: np.ndarray | None,
    ) -> np.ndarray:
        if filter_fn is None:
            return np.ones(labels.shape[0], dtype=bool)
        mask = filter_fn(
            labels=labels, sample_of_row=sample_of_row, token_ids=token_ids
        )
        _assert_leading_shape("filter_fn mask", mask, (labels.shape[0],))
        return mask

    def _split(self, sample_of_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        unique_samples = np.unique(sample_of_row)
        train_samples, test_samples = train_test_split(
            unique_samples,
            test_size=self._config.test_size,
            random_state=self._config.seed,
        )
        train_mask = np.isin(sample_of_row, train_samples)
        test_mask = np.isin(sample_of_row, test_samples)
        return train_mask, test_mask

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainResult:
        import cuml
        import cuml.pipeline
        import cupy

        logger.info(
            "Fitting probe: %d train rows, %d test rows", len(y_train), len(y_test)
        )
        missing_train_labels = set(np.unique(y_test).tolist()) - set(
            np.unique(y_train).tolist()
        )
        if missing_train_labels:
            raise ValueError(
                f"Training split is missing label(s) {missing_train_labels} "
                "present in the test split -- check that filter_fn/labels "
                "actually include tokens for every role in this role space."
            )
        logger.info(
            "DEBUG fit features shape=%s dtype=%s; train_labels shape=%s dtype=%s "
            "unique=%s; test_labels shape=%s dtype=%s unique=%s",
            x_train.shape,
            x_train.dtype,
            y_train.shape,
            y_train.dtype,
            np.unique(y_train).tolist(),
            y_test.shape,
            y_test.dtype,
            np.unique(y_test).tolist(),
        )

        steps = []
        if self._config.add_scaling:
            steps.append(("scaler", cuml.preprocessing.StandardScaler()))
        steps.append(
            (
                "clf",
                cuml.linear_model.LogisticRegression(
                    C=self._config.C,
                    max_iter=self._config.max_iter,
                    linesearch_max_iter=self._config.linesearch_max_iter,
                ),
            )
        )
        estimator = cuml.pipeline.Pipeline(steps) if len(steps) > 1 else steps[0][1]

        cupy_x_train = cupy.asarray(x_train)
        cupy_y_train = cupy.asarray(y_train)

        estimator.fit(cupy_x_train, cupy_y_train)
        self.estimator_ = estimator

        predictions, probabilities = self._predict(x_test)
        return TrainResult(
            predictions=predictions,
            probabilities=probabilities,
            targets=y_test,
            estimator=estimator,
        )

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cupy

        estimator = self.estimator_
        cupy_x = cupy.asarray(x)

        predictions = cupy.asnumpy(estimator.predict(cupy_x))
        probabilities = cupy.asnumpy(estimator.predict_proba(cupy_x))
        return predictions, probabilities
