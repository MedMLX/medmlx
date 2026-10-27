# Provisional MedVLM bridge

`medgemma` and `lingshu` are discoverable frontend adapters, outside the unified
model families. Their isolated MedGemma/Lingshu checkpoint execution and numerical
accuracy are **not qualified by MedMLX**. CPU tests cover the process protocol with
synthetic provider stubs; they do not execute medical model tensors.

Use an existing, separately installed `med-vlm-mlx==0.2.0` Python 3.13 environment.
The bridge checks its runtime dependency metadata and rejects a conflicting
upstream `mlx-vlm` installation. Nothing is installed or configured persistently.
Select the interpreter explicitly for each CLI/server process:

```sh
MEDMLX_MEDVLM_PYTHON=/absolute/med-vlm-environment/bin/python   medmlx run medgemma --arguments arguments.json --root /path/to/work
```

Use the same environment variable with `medmlx serve`. This does not register an
MCP client or grant filesystem permissions. For Lingshu, choose `lingshu` instead.

```json
{
  "model_dir": "/local/checkpoints/medgemma",
  "image_path": "/local/images/image.png",
  "prompt": "Describe the visible image features.",
  "output_dir": "/local/results/example",
  "max_new_tokens": 256,
  "timeout_seconds": 300,
  "overwrite": false
}
```

The checkpoint must contain nonempty `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `preprocessor_config.json` and safetensors weights.
Any indexed weight shards must exist locally. MedGemma requires `model_type=gemma3`;
Lingshu requires `qwen2_5_vl`. These checks establish declared architecture and file
presence, not checkpoint identity or tensor compatibility. The provider loads with
strict tensor matching. Missing assets fail before importing MLX; Hub/Transformers
offline mode and local-only loading prevent automatic asset downloads.

Only one local PNG/JPEG image is accepted. NIfTI/DICOM volumes, audio, video and
conversation history are outside this bridge. Greedy generation uses temperature
0 and a prefill chunk of 64 tokens. Bridge limits are 1–2048 new tokens, a nonblank
prompt of at most 16384 Unicode characters, and 1–3600 seconds for helper startup,
loading and generation. These are enforced bridge limits, not model context limits.

Successful runs atomically publish `report.json` with text, token counts, runtime
versions and explicit provisional/unverified status. Output must be outside the
checkpoint; existing reports require `overwrite=true`. Input, dependency and model
errors remain failed jobs. Session cancellation and timeouts stop the helper and
its descendants; job-owned control files are cleaned with the session.

Readiness inspects the isolated dependency metadata only. It does not import MLX,
probe that interpreter's Metal device, load weights or validate image contents.
`run_ready` therefore remains unknown unless a known blocker is found. A metadata
pass does not establish that a real checkpoint will execute successfully.
