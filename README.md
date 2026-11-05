# MedMLX

One Python API, command line and MCP server for native medical imaging on Apple Silicon.
Run TotalSegmentator, nnU-Net, MedSAM2, SynthSeg and more through their independent MLX runtimes.

[Models](https://github.com/MedMLX) · [Docs](https://github.com/MedMLX/medmlx/blob/main/docs/declaring-models.md) · [Website](https://medmlx.org)

![Outputs across CT, PET/CT, radiographs, endoscopy and MRI](https://raw.githubusercontent.com/MedMLX/medmlx/main/docs/gallery.jpg)

**Apple Silicon · Metal · Python 3.13** · [uv](https://docs.astral.sh/uv/)

This README describes the **0.4.3 release**. The
[PyPI documentation](https://pypi.org/project/medmlx/) describes the same release;
source-only features are tracked in the [release notes](https://github.com/MedMLX/medmlx/blob/main/docs/release-notes.md).

## Install

Install the latest published CLI and MCP server with all model families:

```sh
uv tool install --python 3.13 "medmlx[all,mcp]"
```

For a Python project, use `uv add "medmlx[all,mcp]"` instead.

Choose `segment`, `generate` or `vision` instead of `all` for a smaller install.
The segment extra includes SAM 3 image and video support; generate includes
NV-Reason-CT and the weight-free FDG PET adapter. Model weights download
automatically on first use and keep their own terms. MedGemma uses a provisional bridge to a
separate MedVLM environment; checkpoint inference and numerical accuracy are
unqualified. See the
[bridge setup and limits](https://github.com/MedMLX/medmlx/blob/main/docs/medvlm-bridge.md).

For unreleased runtime changes, use the locked source workspace described in
the [MCP setup guide](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md). The [dependency definitions](https://github.com/MedMLX/medmlx/blob/main/pyproject.toml)
record the runtime versions used by that workspace.

## Run a model

```sh
medmlx describe mlx-segment-ct
medmlx readiness mlx-segment-ct --arguments arguments.json --root /path/to/work
medmlx run mlx-segment-ct --arguments arguments.json --root /path/to/work
```

For models with automatic assets, omit `weights_path`: the runner downloads its
configured Hub repository or model files, reuses the cached folder afterwards,
and records its revision in the run details. For your own weights, pass a local
folder as `weights_path`. MedGemma and MRSegmentator currently require local
weights; multi-model workflows can require separate asset paths.

`describe` shows each model's arguments. `readiness` checks the runtime and Metal
without downloading weights. Optionally pre-download installed model assets for
offline use with `medmlx setup --all --progress`; workflows with multiple assets
require manual setup.
See [automatic weights and readiness](https://github.com/MedMLX/medmlx/blob/main/docs/setup.md).

```python
import json
from pathlib import Path
import medmlx

arguments = json.loads(Path("arguments.json").read_text())
result = medmlx.run("mlx-segment-ct", arguments=arguments, root="/path/to/work")
```

For an MCP client, use `medmlx serve --root /path/to/work` as its stdio command.
Disconnecting the client cancels its running jobs. See the
[MCP guide](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) for client
setup, tools and a complete job workflow. For checkpoint conversion and
development, see [conversion](https://github.com/MedMLX/medmlx/blob/main/conversion/README.md)
and [core](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md).

The package also includes
[Reason-CT with nodule tools](https://github.com/MedMLX/medmlx/blob/main/docs/reason-ct-tools.md):
model-selected detector and MedSAM2 calls, spatially grounded masks and a local
CT review page. It requires the `reason-tools` extra and model assets.

## License

[Apache-2.0](https://github.com/MedMLX/medmlx/blob/main/LICENSE). See [third-party notices](https://github.com/MedMLX/medmlx/blob/main/THIRD_PARTY_NOTICES.md).
Runtimes and model weights retain their own terms.

For research and education. Outputs require human review.
