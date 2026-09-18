#!/usr/bin/env bash
set -euo pipefail

uv run coverage run -m pytest -v
uv run coverage report --fail-under=80
