# Provisional MedVLM bridge

This guide includes an **unreleased** bridge update to `med-vlm-mlx==0.3.3`.
The published MedMLX 0.4.3 bridge prepares `med-vlm-mlx==0.2.3`, which is not
available on PyPI.

`medgemma` is a discoverable `medmlx` adapter outside the unified model families.
Its isolated checkpoint execution and numerical accuracy are **not qualified by
MedMLX**. CPU tests cover the process protocol with synthetic provider stubs;
they do not execute medical model tensors.

Run MedGemma through the same Python package, CLI or MCP interface as other
models. Supply `weights_path` as a local MedGemma folder; automatic model assets
are not currently configured for this adapter. Its first run prepares a private
Python 3.13 environment containing
`med-vlm-mlx==0.3.3` under `~/.cache/medmlx/.runtimes/med-vlm-mlx-0.3.3/`.
Setting `MEDMLX_ASSET_CACHE` replaces the `~/.cache/medmlx` base directory.
Interrupted environment installs resume on retry; subsequent runs reuse that
environment.

MedVLM's [v0.3.3 requirements](https://github.com/MedMLX/med-vlm-mlx/blob/v0.3.3/pyproject.toml)
require `mlx>=0.32.4,<0.33` and `transformers==5.19.0`; its dependency graph stays isolated from
the unified model runtimes. The bridge checks that environment's runtime
requirements before inference and rejects a conflicting upstream `mlx-vlm`
distribution. No dependency overrides are applied to the application environment.

For offline use, supply `weights_path` as a local folder after the isolated runtime
has been prepared. Advanced integrations can still select an existing compatible
Python using `MEDMLX_MEDVLM_PYTHON`; this is optional. The same behavior applies
with `medmlx serve`. MCP client registration remains a separate client setting.

Save this complete example as `arguments.json` in your current directory,
replacing the local paths with your image, checkpoint and output folder:

```json
{
  "image_path": "/local/images/image.png",
  "weights_path": "/local/models/medgemma",
  "prompt": "Describe the visible image features.",
  "output_dir": "/local/results/example",
  "max_new_tokens": 256,
  "timeout_seconds": 300,
  "overwrite": false
}
```

```sh
medmlx run medgemma --arguments arguments.json --root /path/to/work
```

An explicit `weights_path` folder must contain nonempty `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `preprocessor_config.json` and safetensors weights.
Any indexed weight shards must exist locally. MedGemma requires `model_type=gemma3`.
These checks establish declared architecture and file presence, not checkpoint
identity or tensor compatibility. The provider loads with
strict tensor matching. Once setup completes, the isolated inference helper uses
Hub/Transformers offline mode and local-only loading. Missing local assets fail
before it imports MLX.

Only one local PNG/JPEG image is accepted. NIfTI/DICOM volumes, audio, video and
conversation history are outside this bridge. Greedy generation uses temperature
0 and a prefill chunk of 64 tokens. Bridge limits are 1–2048 new tokens, a nonblank
prompt of at most 16384 Unicode characters, and 1–3600 seconds for helper startup,
loading and generation. Environment installation precedes that generation
timeout. These are enforced bridge limits, not model
context limits.

Successful runs atomically publish `report.json` with text, token counts, runtime
versions, `weights_revision` when the shared cache recorded a Hub commit, and
explicit provisional/unverified status. Output must be outside the
checkpoint; existing reports require `overwrite=true`. Input, dependency and model
errors remain failed jobs. Session cancellation and timeouts stop the helper and
its descendants; job-owned control files are cleaned with the session.

Readiness inspects existing isolated dependency metadata only; it never installs
software or downloads weights. Before the first run it reports that the runtime
will be prepared automatically. It does not import MLX,
probe that interpreter's Metal device, load weights or validate image contents.
`run_ready` therefore remains unknown unless a known blocker is found. A metadata
pass does not establish that a real checkpoint will execute successfully.
