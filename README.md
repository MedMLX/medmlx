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
Model weights have their own download and access requirements.

## Run a model

```sh
medmlx describe totalsegmentator
medmlx readiness totalsegmentator
medmlx run totalsegmentator --arguments arguments.json --root /path/to/work
```

`describe` lists the arguments to put in `arguments.json`. Relative paths resolve against `--root`.

```python
import json
from pathlib import Path
import medmlx

arguments = json.loads(Path("arguments.json").read_text())
result = medmlx.run("totalsegmentator", arguments=arguments, root="/path/to/work")
```

For an MCP client, use `medmlx serve --root /path/to/work` as its stdio command.
For checkpoint conversion and development, see [conversion](conversion/README.md) and [core](packages/core/docs/technical.md).

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md).
Runtimes and model weights retain their own terms.

For research and education. Outputs require human review.
