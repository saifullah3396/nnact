from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``nnact`` hierarchy.

    ``name`` is typically a module's ``__name__``, which for code inside the
    ``nnact`` package already starts with ``nnact.`` (or is exactly
    ``nnact``); it's only prefixed when it doesn't already live under that
    hierarchy, so callers can pass ``__name__`` directly either way.
    """
    if name != "nnact" and not name.startswith("nnact."):
        name = f"nnact.{name}"
    return logging.getLogger(name)


_file_handler: logging.FileHandler | None = None


def enable_file_logging(log_file: str | Path, level: int = logging.INFO) -> None:
    """Attach a file handler to the ``nnact`` root logger, replacing any
    previously attached one.

    All ``nnact.*`` module loggers propagate to it, so this captures every
    run's log output in one file. Each call (e.g. one per
    :class:`~nnact._pipeline.ActivationPipeline` built) replaces the
    previous handler, so logs always go to the most recently configured
    run's file rather than piling up across every past handler.
    """
    global _file_handler

    root = logging.getLogger("nnact")
    root.setLevel(level)

    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()

    _file_handler = logging.FileHandler(log_file)
    _file_handler.setLevel(level)
    _file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(_file_handler)


class TqdmToLogger:
    """File-like adapter that forwards tqdm's rendered output to a logger.

    tqdm writes plain text to a stream rather than emitting log records, so
    a file handler attached via :func:`enable_file_logging` never sees a
    plain progress bar's updates on its own. Pass this as ``file=`` (e.g. to
    Ignite's ``ProgressBar``, which forwards it straight to tqdm) to get the
    bar's rendered lines into the log file as well.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def write(self, message: str) -> None:
        stripped = message.strip()
        if stripped:
            self._logger.info(stripped)

    def flush(self) -> None:
        pass
