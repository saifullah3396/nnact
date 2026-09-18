from __future__ import annotations

import logging

LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"

_root = logging.getLogger("nnact")
_root.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
_root.addHandler(_console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``nnact`` hierarchy.

    ``name`` is typically a module's ``__name__``, which for code inside the
    ``nnact`` package already starts with ``nnact.`` (or is exactly
    ``nnact``); it's only prefixed when it doesn't already live under that
    hierarchy, so callers can pass ``__name__`` directly either way.

    The ``nnact`` root logger has a console handler attached on import, so
    library diagnostics are visible by default.
    """
    if name != "nnact" and not name.startswith("nnact."):
        name = f"nnact.{name}"
    return logging.getLogger(name)
