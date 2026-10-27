# MedMLX

One Python API, command line and MCP server for native medical imaging on Apple Silicon.
Run TotalSegmentator, nnU-Net, MedSAM2, SynthSeg and more through their independent MLX runtimes.

[Models](https://github.com/MedMLX) · [Docs](docs/declaring-models.md) · [Website](https://medmlx.org)

![Outputs across CT, PET/CT, radiographs, endoscopy and MRI](docs/gallery.jpg)

**Apple Silicon · Metal · Python 3.13** · [uv](https://docs.astral.sh/uv/)

## Install

Install from the source checkout with all model families and MCP:

```sh
git clone https://github.com/MedMLX/medmlx.git
cd medmlx
uv sync --locked --no-dev --extra all --extra mcp
source .venv/bin/activate
medmlx models
```

Choose `segment`, `generate` or `vision` instead of `all` for a smaller install.
The segment extra includes SAM3 image/video support; generate includes NV-Reason-CT
and the weight-free FDG PET adapter. Model weights have their own download and access
requirements. MedGemma/Lingshu is not bundled: med-vlm-mlx currently pins MLX 0.32.3,
which conflicts with this workspace's MLX >=0.32.4 requirement.

## Run a model

```sh
medmlx describe totalsegmentator
medmlx readiness totalsegmentator --arguments arguments.json --root /path/to/work
medmlx checkpoints --model-id nnunet
medmlx run totalsegmentator --arguments arguments.json --root /path/to/work
```

`describe` includes the provider's full instructions and argument descriptions,
including documented units and ranges. Relative paths resolve against `--root`.
Readiness reports `runtime_ready` for imports/provider checks and Metal separately
from `ready`: `false` identifies a blocker; `null` means input/checkpoint contents
remain unverified. Passing arguments checks names, required arguments and local
input/weight path existence, without running inference or creating outputs. Existing
clients that used `ready` for runtime availability should use `runtime_ready`.

`checkpoints` lists five source references, including catalog-only liver and AutoPET
entries for the general nnU-Net provider. It neither stages assets nor qualifies a
checkpoint. The same action is available as `list_checkpoints` in Python and MCP.

```python
import json
from pathlib import Path
import medmlx

arguments = json.loads(Path("arguments.json").read_text())
result = medmlx.run("totalsegmentator", arguments=arguments, root="/path/to/work")
```

For an MCP client, use `medmlx serve --root /path/to/work` as its stdio command.
Jobs belong to that server session; disconnecting cancels workers and their child
processes. Client registration and persistent filesystem access are configured by
that client, not changed by this package. Some providers fetch default assets when
paths are omitted; supply explicit local checkpoints for offline execution.
For checkpoint conversion and development, see [conversion](conversion/README.md) and [core](packages/core/docs/technical.md).

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
Runtimes and model weights retain their own terms.

For research and education. Outputs require human review.
