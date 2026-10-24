.DEFAULT_GOAL := qa
.PHONY: loc clean
TESTS ?= tests
PYTEST_ARGS ?=

.PHONY: env setup bootstrap host-check env-macos \
	lint typecheck test qa qualify loc clean

ENV_ARGS ?=
ENV_RUN := UV_OFFLINE=1 bash environment/setup.sh $(ENV_ARGS) --run
SOURCE_DIRS ?= recipes src tests scripts
GENERATED_PATHS ?= .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build

env:
	@test -f environment/setup.sh || { echo "Missing environment/setup.sh"; exit 1; }
	bash environment/setup.sh $(ENV_ARGS)

setup bootstrap: env

host-check:
	@test -f environment/setup.sh || { echo "Missing environment/setup.sh"; exit 1; }
	bash environment/setup.sh --host-check

env-macos:
	bash environment/setup.sh $(ENV_ARGS) --target macos

lint:
	$(ENV_RUN) python -m ruff check .

typecheck:
	$(ENV_RUN) sh -c 'command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }; exec env PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 python -m pyright --project . --warnings --pythonpath "$$(command -v python)"'
	bash -o pipefail -c 'rg --files --hidden --no-ignore --glob "*.py" --glob "*.pyi" --null recipes src tests scripts | xargs -0 env $(ENV_RUN) python -m ruff check --config pyproject.toml --select TID251,ANN401 --ignore-noqa --no-force-exclude --'

test:
	$(ENV_RUN) python -m pytest $(TESTS) --ignore=tests/test_recipes.py $(PYTEST_ARGS)

qa: lint typecheck

qualify:
	$(ENV_RUN) python -m pytest tests/test_recipes.py $(PYTEST_ARGS)

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
