# medmlx

MedMLX Python API, CLI, MCP server, shared runtime (`packages/core`) and
conversion tooling (`conversion`). Model runtimes live in independent
repositories and register through `medmlx.models`.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Refer to MedMLX as a Python package.
- CLI and MCP adapters call `medmlx.api`; runs use supervised worker
  processes. Preserve cancellation, argument validation and explicit Metal
  checks.
- `packages/core` owns `medmlx_core`; its distribution and imports stay stable
  for the independent runtimes.
- Never commit weights, medical images or patient data. Runtimes and weights
  keep their own terms.

## Checks

- `make env` creates the locked environment; `make qa` checks the `medmlx` package and
  `make qa-core` the shared runtime.
- `conversion` has its own environment: `make env-convert`, `make qa-convert`;
  full exports run only through `make qualify-convert`.
- `make test-extract` tests the extraction verifier in `tools/extraction`.
- `make test TESTS=tests/test_name.py` runs selected tests. There is no hosted
  CI.
