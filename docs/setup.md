# Automatic weights and readiness

## Install and run

This page describes the published release. See the
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

## Describe, readiness, check and checkpoints

`describe` shows the runner's arguments, units, ranges and instructions.
Relative paths resolve against `--root`.

`readiness [MODEL_ID ...]` reports `runtime_ready` for imports, provider checks
and Metal; it does not validate run inputs or weights. `check MODEL_ID
KEY=VALUE ...` checks one run: argument names, required arguments, that local
input and weight paths exist, that weights are supplied or cached, and that NIfTI
inputs have readable headers, without running inference or creating outputs. Its
`run_ready` is `true` when all of these pass, `false` for a known blocker, and
`null` when something could not be checked, such as a non-NIfTI input or weights
the run would download first. `true` does not prove that the weights and images
are compatible. Python has the same checks as `model_readiness(model_ids)` and
`check_run(model_id, **arguments)`.

`checkpoints` lists known source references for the named nnU-Net models,
including liver and AutoPET. It does not download assets or validate a checkpoint.
The same action is available as `list_checkpoints` in Python and MCP.
