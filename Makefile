.DEFAULT_GOAL := qa
.PHONY: check-env loc clean
TESTS ?= tests
CORE_TESTS ?= packages/core/tests
PYTEST_ARGS ?=
SOURCE_DIRS ?= src tests tools/extraction/tests
GENERATED_PATHS ?= .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
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
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	list=$$(mktemp); trap 'rm -f "$$list"' EXIT; \
	for dir in $(SOURCE_DIRS); do \
		if [ -e "$$dir" ]; then find "./$$dir" -type f \
			\( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 >> "$$list"; fi; \
	done; test -s "$$list" || { echo "No source files found"; exit 1; }; xargs -0 wc -l < "$$list"

clean:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths GENERATED_PATHS $(GENERATED_PATHS); \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	for path in $(GENERATED_PATHS); do \
		rm -rf -- "$$path"; \
	done; for dir in $(SOURCE_DIRS); do \
		if [ -d "$$dir" ]; then find "./$$dir" -type d -name __pycache__ -prune -exec rm -rf {} +; fi; \
	done
