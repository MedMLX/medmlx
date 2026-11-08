# Extraction verifier

Check a standalone MedMLX model repository's status, runtime imports, tests
and lint using its existing environment. Failed checks return a nonzero exit code.

Part of [MedMLX](https://github.com/MedMLX).

[Quick start](#quick-start) · [Scripts](#scripts) · [Development](#development) · [License](#license)

## Requirements

The verifier runs on macOS or Linux with Bash, Python 3.13 or 3.14, pytest, Ruff, and the
selected repository's dependencies. Set the tool paths to its existing environment.

## Quick start

Run the verifier from `tools/extraction`, naming a runtime checkout beside MedMLX:

```bash
MEDMLX_PYTHON=/path/to/env/bin/python \
MEDMLX_RUFF=/path/to/env/bin/ruff \
bash verify.sh flexray-mlx
```

`MEDMLX_REPO_ROOT` selects the parent containing the checkouts, and
`MEDMLX_CORE_SRC` selects the shared core source. Defaults are the workspace containing this MedMLX checkout and its
`packages/core/src` directory. Without explicit tool paths, the verifier uses
the main checkout's `.venv/bin/python` and `.venv/bin/ruff` (created by `make env`).

## Scripts

| Script | Purpose |
| --- | --- |
| [verify.sh](verify.sh) | Report changes and runtime imports, then run tests and lint |

Model weights and medical data are excluded from this repository.

## Development

Run the verifier regression checks:

```bash
python -m unittest discover -s tests
```

## License

Apache-2.0; see [LICENSE](../../LICENSE). Extracted MedMLX code retains applicable
upstream notices and separate checkpoint terms.
