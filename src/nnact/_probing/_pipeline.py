from __future__ import annotations

from pathlib import Path
from typing import final

from nnact._logging import get_logger
from nnact._outputs._dataset import ActivationDataset
from nnact._probing._result import ProbeResult
from nnact._probing._trainer import FilterFn, PoolFn, ProbeTrainer

logger = get_logger(__name__)


@final
class ProbePipeline:
    """Caches :class:`~nnact._probing._trainer.ProbeTrainer` runs on disk.

    Knows nothing about what a caller is probing for -- role spaces, or any
    other grouping -- it only fits and evaluates whatever ``dataset`` and
    ``filter_fn`` it is handed, and skips the (expensive) fit entirely when a
    result is already cached at ``cache_path``. Run the pipeline once per
    input to probe several things: each call is independent and keyed by its
    own ``cache_path``.
    """

    def __init__(self, trainer: ProbeTrainer) -> None:
        self._trainer = trainer

    def run(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        cache_path: str | Path | None = None,
        force_rerun: bool = False,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> ProbeResult:
        path = Path(cache_path) if cache_path is not None else None

        if path is not None and path.exists() and not force_rerun:
            logger.info("Loading cached probe: %s", path)
            return ProbeResult.load(path)

        result = self._trainer.fit(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )

        if path is not None:
            result.save(path)
            logger.info("Saved probe cache: %s", path)

        return result
