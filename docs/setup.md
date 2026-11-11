# Automatic weights and readiness

## Install and run

This page describes the 0.4.7 release. See the
[PyPI documentation](https://pypi.org/project/medmlx/) for the published package
and the [release notes](release-notes.md) for source-only features.

Install the latest published CLI and MCP server with all model families:

```sh
uv tool install --python 3.13 "medmlx[all,mcp]"
```

For a Python project, use `uv add "medmlx[all,mcp]"`. Choose `segment`,
`generate` or `vision` instead of `all` for a smaller installation.
No runtime repositories need to be cloned.

Most models download their weights on first use to
`~/.cache/medmlx/<model_id>`. To use a local copy, pass its folder as
`weights_path`. Multi-model workflows may need a separate path per model.

Automatic assets are not currently configured for MedGemma or MRSegmentator;
those runners require a compatible local `weights_path` folder.

Downloads resume after interruptions and are cached for offline use. Concurrent
requests share the download. Set `MEDMLX_ASSET_CACHE` to move the cache.
Run details record the downloaded Hub commit as `weights_revision`; for local
folders, it is `null` unless known. Explicit paths are used as supplied.

Variants are model IDs, including `nnunet-liver`, `nnunet-autopet`,
`segresnet-renal-cect`, `mlx-generate-ctmr-ct`, `mlx-generate-ctmr-mr` and
`mlx-generate-ctmr-mr-brain`. No separate setup IDs or bundle arguments are needed.
Weight-free models, such as radiomics and FDG PET synthesis, do not download
weights; their actual image, mask and profile inputs remain explicit.

## Optional offline pre-download

```sh
medmlx setup --all --progress
medmlx setup mlx-segment-ct --progress
```

The first command pre-downloads installed models. The second pre-downloads one.
Normal runs use the same helper. Models that require manual asset setup are
reported as unsupported; `--all` records them as failures. It continues after
individual failures and exits nonzero for partial completion or discovery errors.
Successful caches remain usable on retry. Setup downloads weights without running the model.
To update or repair a cached model, run `medmlx setup MODEL_ID --refresh`.
Use `--all --refresh` to refresh every installed model. Explicit local folders
are never modified by these commands.

`medmlx setup --all --plan` describes the operation without downloading. With no
arguments, `medmlx setup` lists installed models and their source repositories.
The same calls are available in Python as `setup_all()`, `setup_model(model_id)`,
`list_setup_plans()` and `setup_plan(model_id)`.

## Describe, readiness and checkpoints

`describe` shows the runner's arguments, units, ranges and instructions.
Relative paths resolve against `--root`.

`readiness` reports `ready` (and the equivalent `runtime_ready`) for imports,
provider checks and Metal. Neither validates run inputs or weights. The separate
`run_ready` is `false` for a known blocker and `null` while input and checkpoint
contents are unverified; this check never returns `run_ready=true`. Passing
arguments checks names, required arguments and that local input and weight paths
exist, without running inference or creating outputs.

`checkpoints` lists known source references for the named nnU-Net models,
including liver and AutoPET. It does not download assets or validate a checkpoint.
The same action is available as `list_checkpoints` in Python and MCP.
