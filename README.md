# MedMLX

TotalSegmentator, nnU-Net, MedSAM2, SynthSeg and more, running natively in MLX.
Run any of them from Python, the command line or an MCP client. Each model runtime
remains in its own repository; MedMLX discovers the installed runners and executes
them in supervised worker processes.

Research use only; model outputs require human review.

![Outputs from MedMLX models across CT, PET/CT, radiographs, endoscopy and MRI](https://raw.githubusercontent.com/MedMLX/medmlx/main/docs/gallery.jpg)

## Install

Needs an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/). Install the API,
the MCP server and every model family:

```sh
git clone https://github.com/MedMLX/medmlx.git
cd medmlx
uv sync --locked --no-dev --extra all --extra mcp
source .venv/bin/activate
medmlx models
```

Optional extras are `segment`, `generate`, `vision`, `all` (all three families),
and `mcp`. The base installation contains the API and shared runtime. It does
not install model families, MCP, Torch, TensorFlow or conversion frameworks.
RadImageNet remains available in `vision`.

The locked source checkout installs the model families from immutable public Git
revisions in `tool.uv.sources`. Wheel extras use versioned package requirements;
those provider versions must be available on your package index or in a local
wheel directory. Use this checkout for the combined installation.

## Use

```sh
medmlx packages
medmlx describe totalsegmentator
medmlx readiness totalsegmentator
medmlx run totalsegmentator --arguments arguments.json --root /path/to/work
```

`arguments.json` contains the chosen runner's keyword arguments, shown by
`describe`. Relative paths resolve against `--root`. Each run returns its completed
job and result or raises an execution error. Interrupting the command stops the
worker and its descendants.

```python
import json
from pathlib import Path
import medmlx

models = medmlx.list_models()
description = medmlx.describe_model("totalsegmentator")
arguments = json.loads(Path("arguments.json").read_text())
result = medmlx.run("totalsegmentator", arguments=arguments, root="/path/to/work")
```

Use `medmlx serve --root /path/to/work` as an MCP client's stdio command. The
server offers model/package listing, descriptions, readiness, runtime information,
job submission, status and cancellation. Python, CLI and MCP use the same action
layer. [Runner declarations](https://github.com/MedMLX/medmlx/blob/main/docs/declaring-models.md) describe the plugin contract.

## Repository layout

| Path | Owns |
| --- | --- |
| `src/medmlx` | API, CLI, MCP, model discovery and supervised jobs |
| `packages/core` | Shared `medmlx_core` runtime, graph executor and geometry |
| `conversion` | Upstream model capture, conversion recipes and qualification |
| `tools/extraction` | Model repository verifier |

The separate family distributions and MCP repository are replaced by this
frontend. `nv-generate-mlx` remains an independent model runtime, alongside the
other model repositories. Core remains a small internal distribution so those
runtimes do not acquire frontend or conversion dependencies.

## Conversion and development

```sh
make env
make qa
make env-convert
medmlx convert totalsegmentator -- --help
make qa-convert
```

`make qa` runs frontend lint and Pyright in the existing environment. It never
runs tests, installs dependencies or builds models. Use `make qa-core` and
`make qa-convert` for static checks of those components. Run affected tests with
`make test TESTS=tests/path.py`, `make test-core CORE_TESTS=packages/core/tests/path.py`
or `make test-convert TESTS=tests/path.py`; extraction checks use `make test-extract`.
Full model exports are an explicit, expensive `make qualify-convert` operation.
Use `PYTEST_ARGS='-k model_name'` to qualify only an affected recipe. Choose
checks for the changed behavior; documentation and moves do not require full QA.

Conversion uses its own qualified environment and lockfile in this repository;
the command never silently installs or changes an environment. Conversion
recipes require the full Git source checkout. The frontend wheel and source
distribution do not bundle the separate core or conversion projects. See
[conversion documentation](https://github.com/MedMLX/medmlx/blob/main/conversion/README.md)
and [runtime documentation](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md).

`medmlx convert` runs with `conversion/` as its working directory; recipe-relative
input paths and the default `build/` output resolve there.

## License

Apache-2.0; see [LICENSE](https://github.com/MedMLX/medmlx/blob/main/LICENSE) and [THIRD_PARTY_NOTICES.md](https://github.com/MedMLX/medmlx/blob/main/THIRD_PARTY_NOTICES.md).
Each runtime, checkpoint and upstream dependency retains its own terms.
