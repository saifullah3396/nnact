.PHONY: test lint format ci

test:
	./scripts/test.sh

lint:
	./scripts/lint.sh

format:
	./scripts/format.sh

ci: lint format test
