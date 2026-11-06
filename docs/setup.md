# Automatic weights and readiness

## Install and run

This page describes the 0.4.5 release. See the
[PyPI documentation](https://pypi.org/project/medmlx/) for the published package
and the [release notes](release-notes.md) for source-only features.

Install the latest published CLI and MCP server with all model families:

```sh
uv tool install --python 3.13 "medmlx[all,mcp]"
```

For a Python project, use `uv add "medmlx[all,mcp]"`. Choose `segment`,
`generate` or `vision` instead of `all` for a smaller installation.
No runtime repositories need to be cloned.

Most model runners need no setup command and accept an optional `weights_path`.
When omitted, the runtime calls the shared `medmlx_core.weights.weights_dir(model_id)`
helper. It downloads the registered Hub repository, or the selected model files,
as ordinary files under `~/.cache/medmlx/<model_id>`. The runtime finds its own
weight and auxiliary files inside that folder. Multi-model workflows can require
separate explicit asset paths; see their runner instructions.

Automatic assets are not currently configured for MedGemma or MRSegmentator;
those runners require a compatible local `weights_path` folder.

For an unpinned source, the first download resolves the current Hub revision to a
commit; a package-pinned source uses its declared commit. Run details record that
revision as `weights_revision`. Completed downloads work offline.
Partial downloads resume on the next attempt, including after cancellation.
Concurrent first-use calls share one download. `MEDMLX_ASSET_CACHE` overrides
the cache directory. Explicit `weights_path` folders are used as supplied, with
no download or link resolution; their revision is `null` unless known locally.

Variants are model IDs, including `nnunet-liver`, `nnunet-autopet`,
`segresnet-brats`, `segresnet-renal-cect`, `mlx-generate-ctmr-ct`, `mlx-generate-ctmr-mr`
and `mlx-generate-ctmr-mr-brain`. No separate setup IDs or bundle arguments are needed.
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
Successful caches remain usable on retry. It does not run or qualify inference.
To update or repair a cached model, run `medmlx setup MODEL_ID --refresh`.
Use `--all --refresh` to refresh every installed model. Explicit local folders
are never modified by these commands.

`medmlx setup --all --plan` describes the operation without downloading. With no
arguments, `medmlx setup` lists installed models and their source repositories.
The same calls are available in Python as `setup_all()`, `setup_model(model_id)`,
`list_setup_plans()` and `setup_plan(model_id)`.

## Describe, readiness and checkpoints

`describe` includes the provider's full instructions and argument descriptions,
including documented units and ranges. Descriptions reproduce provider
documentation; they do not establish validated constraints. Relative paths
resolve against `--root`.

`readiness` reports `ready` (and the equivalent `runtime_ready`) for imports,
provider checks and Metal. Neither validates run inputs or weights. The separate
`run_ready` is `false` for a known blocker and `null` while input and checkpoint
contents are unverified; this check never returns `run_ready=true`. Passing
arguments checks names, required arguments and that local input and weight paths
exist, without running inference or creating outputs.

`checkpoints` lists known source references for the named nnU-Net models,
including liver and AutoPET. It does not download assets or validate a checkpoint.
The same action is available as `list_checkpoints` in Python and MCP.
