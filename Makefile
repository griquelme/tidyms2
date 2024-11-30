.PHONY: dev-install
dev-install:
	type uv && uv run --all-extras python -c "" || >&2 echo "Error: uv not found in user PATH." && \
	uv run pre-commit install

.PHONY: clean
clean:
	rm -rf .venv .pytest_cache .ruff_cache htmlcov docs/_build
	find -iname "*.pyc" -delete

.PHONY: check-format
check-format:
	uv run ruff check

.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

.PHONY: check-types
check-types:
	uv run pyright src

.PHONY: unit-tests
unit-tests:
	uv run pytest

.PHONY: integration-tests
integration-tests:
	uv run pytest src/tests/integration

.PHONY: coverage
coverage:
	uv run pytest --cov=src/tidyms2 && uv run coverage html
