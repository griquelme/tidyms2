.venv:
	type uv && uv venv || >&2 echo "Error: uv not found in user PATH."

.PHONY: dev-install
dev-install: .venv
	uv sync --all-extras && uv run pre-commit install

.PHONY: clean
clean:
	rm -rf .venv .pytest_cache .ruff_cache htmlcov docs/_build
	find -iname "*.pyc" -delete

.PHONY: check-format
check-format:
	uv run ruff check

.PHONY: check-spell
check-spell:
	cspell docs src

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
	uv run pytest src/tests --cov=src/tidyms2 && uv run coverage html
