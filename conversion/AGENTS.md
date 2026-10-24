# Repository policy

## Development velocity

- Development velocity is the number one engineering priority.
- Use the smallest correct change that delivers the requested result.
- Treat excessive documentation and excessive validation as engineering
  failures because they slow delivery and increase maintenance.
- Do not add repeated checksums, hashes, audits, tests, or manual checks after
  the required pipeline check passes.
- Trust a stable pipeline after its required check passes.
- If a pipeline cannot be trusted, fix the pipeline instead of adding repeated
  downstream checks.
- Safety, correctness, science, security, data integrity, and explicit user
  requirements remain mandatory.

## Identity

- Project: `medmlx-convert`
- Location: `conversion/` in the MedMLX repository
- Primary preset: `base`
- Supported setup targets: `macos` (Core AI needs macOS 27 to run `.aimodel`)
- Primary environment: `medmlx-conversion`
- License stance: Apache-2.0
- Canonical run command:
  `bash environment/setup.sh --run python recipes/<model>/export.py`

## Canonical commands

- Setup: `make env`
- Lint: `make lint`
- Typecheck: `make typecheck`
- Tests: `make test` (converter contract checks, excluding full model recipes)
- Static checks: `make qa` (lint and Pyright; no model execution)
- Explicit model qualification: `make qualify` (every recipe end to end)

Scope checks to changed behavior. Repository moves, documentation and dependency
bookkeeping do not justify re-exporting every model. Full model qualification is
opt-in; numerical changes need the affected recipe's comparison.

Keep these public commands stable. Record tool-specific configuration in
`pyproject.toml`.

Public validation targets must work from an unactivated shell. Run
project-owned tools through the canonical environment entrypoint; do not rely
on ambient `PATH` selecting the right executable.

## Environment contract

- macOS setup entrypoint: `environment/setup.sh`
- Target specification: `environment/macos/environment.yml`
- Python ownership: conda owns Python 3.13 and pip; uv owns every other
  package through `pyproject.toml` and `uv.lock`.
- Required external system tools: mamba or conda, uv.
- `torch` is pinned to 2.13 for its security fixes. Requalify it together with
  both pinned converters before changing this version. Retaining 2.8 was rejected
  because it leaves known vulnerabilities in the conversion environment.

## Standard solutions first

Before writing a custom algorithm, workflow, integration, framework
replacement, or infrastructure layer:

- inspect the repository, environment, declared dependencies, and existing
  command surface for a maintained solution;
- identify the established domain workflow from official documentation and,
  for scientific methods, primary methods literature;
- verify the installed or pinned version's CLI help, signatures, parameter
  semantics, and upstream source when documentation is ambiguous; and
- run the standard implementation on a representative real input before
  investing in a replacement.

Do not replace an established solution merely because it is GUI-based,
non-Python, awkward to package, or inconvenient to automate. Check its
supported CLI, headless, batch, plugin, container, or export path first. A
custom primary path requires both a documented contract-level incompatibility
and a direct benchmark showing that it preserves or improves task-relevant
acceptance evidence. Until then, keep custom output explicitly exploratory and
prevent it from carrying production or scientific claims.

## Repository-specific safety and ownership

- Checkpoints download to the torch hub cache; never commit weights, `.aimodel`,
  `.mlpackage`, medical images, or patient data.
- Derived output goes to `build/` (ignored).
- Weight licenses differ per model; check `docs/survey.md` before shipping an
  exported model in an app.

## Repository inspection and shared work

- Discover live paths with `rg --files` before opening files; do not rely on
  remembered or guessed paths.
- Preserve dirty and concurrent work. Re-read the exact local context before
  editing a shared file, apply the smallest ordered patch, and do not format,
  revert, or rewrite unrelated content.

## Project-specific constraints

- Native MLX capture and source translation live in `src/medmlx_convert/native/`.
  `medmlx-core` owns the canonical graph schema/codec, loader, Metal executor
  and shared geometry; it must not depend on this converter. Capture uses original
  upstream CPU FP32 models, strict `torch.export`, then
  `run_decompositions({})` for functionalization. Compare capture,
  functionalization and saved Metal execution independently to the original.
- The native path is an experimental fixed-shape inference pilot. Unknown
  operators and unsupported variants raise; no CPU model fallback. File I/O
  and source-reference execution are explicitly host operations.
- Method decision: Python source transpilation was rejected after observed
  layout, split-size, cast and nonpersistent-buffer failures in torch2mlx.
  ONNX is an optional comparison route, not a required intermediate: an
  execution provider that falls back to CPU does not satisfy this path's Metal
  contract. Neither alternative is claimed universally inferior. TensorFlow
  custom code still needs explicit adapters or an independently checked Keras
  backend bridge; geometry checks do not qualify a full SynthMorph model.
- The project/package identity is `medmlx-convert` / `medmlx_convert`.
  Do not add old-name import shims or duplicate runtime implementations. Graph
  schema 1 is retired; regenerate artifacts as schema 2 rather than translating
  them silently during inference. Source-framework operation names belong only
  to converter frontends. The local uv source uses `../packages/core` in this repository;
  wheel consumers require its matching 0.2 release.

- Export from the upstream PyTorch model and checkpoint, never from the MLX
  port. Neither coremltools nor coreai-torch reads MLX.
- One recipe per model: `recipes/<model>/export.py`. It pins the upstream
  package and checkpoint, exports once with `torch.export`, converts to both
  `.aimodel` and `.mlpackage`, compares each against PyTorch, and exits nonzero
  when a tolerance fails.
- Use fixed input shapes and FP32 unless the recipe documents why not.
- When a recipe lands or a verdict changes, update the status table in
  `README.md` and the entry in `docs/survey.md`.

## Required handoff evidence

- Report the smallest relevant check and its result; full QA is not mandatory
  for documentation, moves or metadata edits.
- For an explicitly requested model qualification, report the recipe's printed
  per-format max logit difference and label agreement.

## Test-value gate

Tests are long-lived production code, not a default deliverable. Add or keep a
test only when it protects a material user-facing, scientific, security,
data-integrity, or public-contract behavior; has a plausible regression path
that static typing, linting, or a cheaper check would miss; and exercises a
stable boundary likely to survive internal refactoring. Its expected signal
must justify its runtime, fixtures, mocks, and maintenance cost.

Do not test trivial accessors, data containers, constants, imports, constructor
wiring, mock call choreography, or private implementation details. Do not add
tests for coverage counts or mirror every branch and edge case. Prefer one
load-bearing contract or integration test over many narrow unit tests. For a
bug fix, add a regression test only when recurrence is plausible and the test
can pin the durable contract. Delete low-value tests that do not meet this bar
when they are in scope.

## Source documentation

For new or changed owned source code, document actual behavior and non-obvious
rationale in docstrings and inline comments, including scientific/data contracts;
verify claims against the implementation, and preserve native documentation
conventions. Avoid boilerplate and unrelated, upstream, or generated
documentation rewrites.

## Agent Communication

- Discover or contact Claude Code only when the user explicitly requests Claude
  communication in the current task. Shared files, environments and running
  processes do not authorize contact.
- Automatic coordination was rejected because shared use does not establish
  overlapping edits and unsolicited messages interrupt other sessions.
- For authorized Claude communication, use the configured `claude-code` MCP
  server with only `ListAgents` and `SendMessage`. List first, then send only the
  requested message to the exact listed recipient.
- Re-read shared files before editing and preserve other agents' changes.

## Node runtime for development

Use the maintained host Node for Pyright. Do not add the `pyright[nodejs]`
extra while its wheel release trails the Node security patches; environment
synchronization must not reinstall an older bundled runtime. Verify the selected
executable after changing this dependency.

## Conda environment updates

Update an existing prefix without `--prune`. With the current Mamba solver,
pruning this valid prefix makes its required CA package unsatisfiable; the same
prefix and specification resolve without pruning. Exact Python dependencies
remain controlled by `uv.lock`. Reintroducing pruning was rejected because it
prevents the canonical environment from receiving the security updates.

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
