# Run MedGemma with MedMLX

The `medgemma` adapter runs MedGemma through the MedMLX Python API, CLI or MCP
server. It accepts one PNG or JPEG image and a local MedGemma model folder.

## Run

Save this as `arguments.json`, replacing the paths with your own:

```json
{
  "image_path": "images/image.png",
  "weights_path": "models/medgemma",
  "prompt": "Describe the visible image features.",
  "output_dir": "results/example",
  "max_new_tokens": 2000,
  "timeout_seconds": 300,
  "overwrite": false
}
```

```sh
medmlx run medgemma --arguments arguments.json --root /path/to/work
```

On first use, MedMLX prepares a separate Python 3.13 environment with
`med-vlm-mlx==0.3.3`. Later runs reuse it. Downloads of MedGemma weights are
manual: pass a local folder as `weights_path`.

The folder must contain nonempty `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `preprocessor_config.json` and safetensors weights,
including every indexed shard. The configuration must declare
`model_type=gemma3`. Inference loads local files in offline mode with strict
tensor matching.

## Options and results

| Option | Accepted values |
| --- | --- |
| `image_path` | One local PNG or JPEG |
| `prompt` | 1–16384 Unicode characters, with nonblank text |
| `max_new_tokens` | 1–2048; 2000 by default, as in Google's model card examples |
| `timeout_seconds` | 1–3600 seconds for helper startup, loading and generation |
| `overwrite` | Set to `true` to replace an existing report |

Generation is greedy, with a prefill chunk of 64 tokens. Environment setup
happens before the generation timeout. The bridge accepts single-image prompts;
volumes, audio, video and conversation history are unsupported.

The output is `report.json`, containing text, token counts, runtime versions and
any known weight revision. Choose an output directory outside the checkpoint
folder. Cancellation and timeouts stop the helper and its child processes.

## Runtime setup

The environment lives at
`~/.cache/medmlx/.runtimes/med-vlm-mlx-0.3.3/`. `MEDMLX_ASSET_CACHE` changes the
cache base directory. Interrupted installs resume on retry. To use an existing
compatible interpreter, set `MEDMLX_MEDVLM_PYTHON`.

The bridge isolates MedVLM's [dependencies](https://github.com/MedMLX/med-vlm-mlx/blob/v0.3.3/pyproject.toml)
from the other runtimes and rejects a conflicting `mlx-vlm` installation.
Readiness checks installed dependency metadata without installing packages or
loading the model. Until execution, `run_ready` can remain unknown.

The bridge has synthetic process-protocol tests. Real-checkpoint execution and
numerical accuracy remain unqualified; reports record that status.
