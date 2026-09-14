"""Parametrized storage backends.

Any test taking :func:`make_writer` or :func:`written_store` runs once per
backend, with the backend name in the test id. That is what stops a shared
guarantee from quietly covering only one implementation.
"""

from pathlib import Path

import pytest
import torch

from nnact.store import (
    ActivationStore,
    ActivationWriter,
    H5ActivationWriter,
    MemoryActivationWriter,
)
from tests.fixtures.types import StoreFactory, WriterFactory


@pytest.fixture(params=["memory", "h5"])
def backend(request: pytest.FixtureRequest) -> str:
    """Name of the storage backend under test."""
    return str(request.param)


@pytest.fixture
def make_writer(backend: str, tmp_path: Path) -> WriterFactory:
    """Construct a fresh writer for the parametrized backend.

    Each call gets its own file, so one test may open several writers without
    one truncating another's output.
    """
    counter = 0

    def _make() -> ActivationWriter:
        nonlocal counter
        counter += 1
        if backend == "memory":
            return MemoryActivationWriter()
        return H5ActivationWriter(tmp_path / f"acts_{counter}.h5")

    return _make


@pytest.fixture
def written_store(make_writer: WriterFactory) -> StoreFactory:
    """Write one batch through the parametrized backend and return the store."""

    def _write(activations: dict[str, torch.Tensor], ids: list[str]) -> ActivationStore:
        writer = make_writer()
        writer.write(ids, activations)
        return writer.close()

    return _write
