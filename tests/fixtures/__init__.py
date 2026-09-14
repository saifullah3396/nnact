"""Fixture modules for the nnact test suite.

Each module here is a focused pytest plugin, loaded by the root ``conftest``
via ``pytest_plugins``:

* :mod:`tests.fixtures.types` — type aliases for the factory fixtures.
* :mod:`tests.fixtures.models` — test doubles: models and datasets.
* :mod:`tests.fixtures.data` — activation tensors and sample IDs.
* :mod:`tests.fixtures.backends` — the parametrized writer/store backends.
"""
