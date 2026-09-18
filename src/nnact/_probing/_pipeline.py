from __future__ import annotations

from pathlib import Path
from typing import final

from nnact._logging import get_logger
from nnact._outputs._dataset import ActivationDataset
from nnact._probing._result import EvalResult, TrainResult
from nnact._probing._trainer import FilterFn, PoolFn, ProbeTrainer

logger = get_logger(__name__)


@final
class ProbePipeline:
    """Caches :class:`~nnact._probing._trainer.ProbeTrainer` runs on disk.

    Knows nothing about what a caller is probing for -- role spaces, or any
    other grouping -- it only trains or evaluates whatever ``dataset`` and
    ``filter_fn`` it is handed, and skips the (expensive) fit/evaluate
    entirely when a result is already cached at ``cache_path``. Run the
    pipeline once per input to probe several things: each call is
    independent and keyed by its own ``cache_path``.
    """

    def __init__(self, trainer: ProbeTrainer) -> None:
        self._trainer = trainer

    def train(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        cache_path: str | Path | None = None,
        force_rerun: bool = False,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> TrainResult:
        path = Path(cache_path) if cache_path is not None else None

        if path is not None and path.exists() and not force_rerun:
            logger.info("Loading cached probe: %s", path)
            return self._trainer.load(path)

        result = self._trainer.train(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )

        if path is not None:
            result.save(path)
            logger.info("Saved probe cache: %s", path)

        return result

    def evaluate(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        cache_path: str | Path | None = None,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> EvalResult:
        """Score the trainer's already-fitted probe against fresh examples.

        Not cached: unlike :meth:`train`, this scores whatever ``dataset``
        it is given right now, so caching by path would silently return a
        stale result for different examples reusing the same path.

        Args:
            cache_path: The same ``cache_path`` a prior :meth:`train` call
                used. Loads that run's saved estimator first, so this works
                standalone -- e.g. in a fresh process where ``train`` was
                never called on this pipeline's trainer. Omit only when the
                trainer already holds a fitted estimator in memory.
        """
        if cache_path is not None:
            self._trainer.load(cache_path)

        return self._trainer.evaluate(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )
