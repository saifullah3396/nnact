"""Root test configuration.

Fixtures live in :mod:`tests.fixtures`, split by what they provide, and are
registered here as plugins so every test module sees them without importing
anything.
"""

pytest_plugins = [
    "tests.fixtures.models",
    "tests.fixtures.data",
    "tests.fixtures.backends",
]
