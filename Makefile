.DEFAULT_GOAL := qa
.PHONY: check-env loc clean
TESTS ?= tests
CORE_TESTS ?= packages/core/tests
PYTEST_ARGS ?=
SOURCE_DIRS ?= src tests tools/extraction/tests
export PYTHONNOUSERSITE := 1
PYTHON ?= .venv/bin/python

.PHONY: env lint typecheck test qa qa-core test-core test-extract env-convert qa-convert test-convert qualify-convert

RUN := $(PYTHON)

env:
	uv lock --check
	uv sync --locked --extra all --extra mcp

check-env:
	@command -v "$(PYTHON)" >/dev/null || { echo "Missing project Python: run make env explicitly, or set PYTHON to its existing interpreter" >&2; exit 1; }

lint: check-env
	$(RUN) -m ruff check src tests tools/extraction/tests

typecheck: check-env
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(PYTHON) -m pyright --project . --warnings --pythonpath "$(PYTHON)"
	$(RUN) -m ruff check --select TID251,ANN401 --ignore-noqa src tests typings tools/extraction/tests

test: check-env
	$(RUN) -m pytest $(TESTS) $(PYTEST_ARGS)

test-extract: check-env
	$(RUN) -m unittest discover -s tools/extraction/tests

qa: lint typecheck

qa-core: check-env
	cd packages/core && "$(abspath $(PYTHON))" -m ruff check .
	$(RUN) -m ruff check --config packages/core/pyproject.toml --select TID251,ANN401 --ignore-noqa packages/core/src packages/core/scripts packages/core/tests
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(PYTHON) -m pyright --project packages/core --pythonpath "$(PYTHON)" --warnings

test-core: check-env
	$(RUN) -m pytest -c packages/core/pyproject.toml $(CORE_TESTS) $(PYTEST_ARGS)

env-convert:
	$(MAKE) -C conversion env

qa-convert:
	$(MAKE) -C conversion qa

test-convert:
	$(MAKE) -C conversion test

qualify-convert:
	$(MAKE) -C conversion qualify

loc:
	@find $(wildcard $(SOURCE_DIRS)) -type f \( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 | xargs -0 wc -l

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name .venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
