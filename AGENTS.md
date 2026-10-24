# MedMLX

One repository owns the Python API, CLI, MCP transport, shared runtime and
conversion tooling. Model architectures, weights and inference pipelines remain
in independent runtime repositories and register through `medmlx.models`.

- Python 3.13 on macOS arm64. Preserve Apache-2.0 and upstream notices.
- `make env` installs the locked development environment; `make qa` checks the
  frontend. `make qa-core` checks shared runtime changes. Model families are optional `segment`, `generate` and
  `vision` extras. MCP is the `mcp` extra.
- `packages/core` owns the actual `medmlx_core` implementation. Its distribution
  and imports remain stable for independent runtimes; it is not a compatibility
  shim. Inference depends on MLX and NumPy, never source-framework converters.
- `conversion` keeps its own lock and qualified Torch 2.13 environment. Combining
  it with the core oracle stack was rejected because the qualified pins differ.
  Use `make env-convert` and `make qa-convert` for that environment.
- Match validation to changed behavior. Full model exports are opt-in through
  `make qualify-convert`; never run them for a repository move, documentation,
  dependency bookkeeping, or as a routine completion gate. A relevant numerical
  change calls for the affected model's comparison, not every recipe.
- Published provider revisions retain immutable core source references. The
  workspace override selects this repository's core for every provider, avoiding
  two implementations in the unified environment.
- CLI and MCP adapters call `medmlx.api`; runs use supervised worker processes.
  Preserve cancellation, argument validation, explicit Metal checks and errors.
- Use the inline-documentation skill for changed owned source. Preserve numerical
  and data contracts; moving source does not qualify a model again.
- Use the maintained host Node for Pyright; never `pyright[nodejs]`.
- No artifact hashing or mandatory artifact manifests. Existing
  source records, ordinary files and Git revisions suffice. Functional cache,
  authentication, deterministic identifier and lockfile hashes are allowed.
- Preserve unrelated dirty work. Original checkouts remain intact until migration
  validation is complete.

## Validation scope

Choose the smallest check that covers the changed behavior. `make qa` runs
static checks in the existing environment; it does not run model tests or builds.
For behavior changes, run the affected tests with `make test TESTS=path/to/test.py`
(and `PYTEST_ARGS` for pytest options). Full suites, numerical parity, model
exports, benchmarks, live checks and package builds are explicit tasks, used
when the change or the user requires them. Documentation, file moves and metadata
edits do not require full QA. Do not synchronize environments, run sibling gates,
or repeat passed checks without a new relevant change. Mandatory full QA after
every edit was rejected because it adds unrelated work and delays feedback.

## Typing policy

Use Pyright strict mode over the maintained source, tests, scripts and local
stubs declared in `pyproject.toml`. Type-ignore comments stay disabled and
unnecessary ignores are errors. Use the maintained host Node with the pinned
Pyright dependency; do not install the `pyright[nodejs]` extra.

Explicit dynamic types remain prohibited. Use Pyright and Ruff for static
checks; custom AST typing scanners add duplicate code and maintenance.
Fix source contracts rather than adding suppressions or weakening diagnostics.

Validation calls the provisioned project Python directly. Using `uv run --no-sync`
as an environment guard was rejected because it can create an empty environment
and select development tools from the host PATH.

## Distribution contents

Source releases include the runtime package, build metadata, README, required
notices and build inputs through an explicit file list. Development scripts,
fixture recorders, tests and benchmarks remain in Git for checkout workflows.
Whole-checkout source releases were rejected because they can publish local
validation and development files; renaming a directory does not exclude it.
