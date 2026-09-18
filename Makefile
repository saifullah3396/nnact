.PHONY: test lint lint-fix format ci

test:
	./scripts/test.sh

lint:
	./scripts/lint.sh

lint-fix:
	uv run ruff check . --fix --unsafe-fixes

format:
	./scripts/format.sh

ci: lint format test
