# Release notes

## 0.4.8

Fixes installation of the `all` and `vision` extras, which selected the obsolete
MLXRadiomics development release in 0.4.7. These extras now install
[`mlxradiomics` 1.0.0](https://github.com/MedMLX/mlx-radiomics/releases/tag/v1.0.0)
or a compatible 1.0.x update.

## 0.4.7

### Breaking changes

**`medsam2` propagation stops after 8 consecutive empty slices in each direction,
with [`medsam2-mlx`
0.3.0](https://github.com/MedMLX/medsam2-mlx/releases/tag/v0.3.0).** Pass
`"stop_after_empty": null` in the run arguments to track through the whole volume as
before.

### Fixes

**MedGemma runs from the documented 4-bit MedGemma 1.5 4B and MedGemma 4B
checkpoints.** Their folders hold one `model.safetensors` beside an index naming two
shards, so 0.4.6 refused them with "Missing checkpoint shard". The bridge now
refuses only a folder where some indexed shards are present and others are missing.

**`mlx-reason-ct` is no longer refused on Macs with about 15 to 23 GB of free
memory, with [`mlx-reason-ct`
0.4.1](https://github.com/MedMLX/mlx-reason-ct/releases/tag/v0.4.1).** It declared a
stale 22.7 GB peak; a chest or abdominal CT report peaks at 14.4 GB.

### New

**Most models now declare their peak memory, so `medmlx` refuses a run that would
not fit before it starts.** The README lists every model's peak. The runtime
releases also add memory reductions, faster BioMedParse and MedSAM2 runs, and safer
output writing; see each runtime's release notes.

## 0.4.6

### New

**Runtime updates now reach `medmlx` users without a new `medmlx` release.** Each
extra accepts later releases of a runtime up to its next minor version, such as
`nnunet-mlx>=0.7.0,<0.8`, instead of one exact version, and `medmlx` accepts
`medmlx-core>=0.2.6,<0.3`. Run `uv tool upgrade medmlx` to pick up new runtime
releases; pin exact versions in your own lockfile to hold a set fixed.

### Faster

**Reason-CT is faster and uses less memory with [`mlx-reason-ct`
0.4.0](https://github.com/MedMLX/mlx-reason-ct/releases/tag/v0.4.0).** A `float32`
chest report takes about 0.6x the time on an M5 Max with identical text, and
one-shot reports no longer balloon process memory. The new
`precision="bfloat16_fast"` is faster still, but its text can differ from
`float32`.

## 0.4.5

- **Breaking:** `nnunet-liver`, `nnunet-autopet`, `nnunet-pancreas`, `nnunet-brats2021`
  and `mrsegmentator` runs return the label map under the `mask` output role. Liver and
  AutoPET used `segmentation` before.
- **Breaking:** `nnunet-brats2021` averages all five released folds instead of fold 0.
  Pass `fold=0` for the previous result. A BraTS download from before 2026-11-02 holds
  only fold 0, so run `medmlx setup nnunet-brats2021 --refresh` first.
  `medmlx setup nnunet-autopet --refresh` likewise adds AutoPET's folds 1-4.
- Native 3-D nnU-Net inference is faster: on an M5 Max, a mirrored `nnunet-liver` tile
  takes 842 ms instead of 1,039 ms, with identical labels.
- Pins [`nnunet-mlx` 0.7.0](https://github.com/MedMLX/nnunet-mlx/releases/tag/v0.7.0)
  and [`totalsegmentator-mlx` 0.1.7](https://github.com/MedMLX/totalsegmentator-mlx/releases/tag/v0.1.7).

## 0.4.4

- The MedGemma bridge prepares `med-vlm-mlx==0.3.3`. The 0.4.2 and 0.4.3 bridge
  pinned `med-vlm-mlx==0.2.3`, which is not available on PyPI, so it could not
  prepare its runtime. 0.3.3 includes HTTP image and host security fixes. Its
  dependency graph stays isolated from the other runtimes; real-checkpoint
  execution and numerical accuracy remain unqualified.
- Otherwise identical to 0.4.3, and still requires `medmlx-core` 0.2.6.

## 0.4.3

- Pins `lesiondiffusion-mlx` 0.2.7. On its reference 160×160×128 crop, peak MLX
  memory falls from 14.49 GB to 3.46 GB in source mode and from 5.14 GB to 3.29 GB
  in conditioned mode, with bitwise-identical outputs. Its declared
  `peak_memory_gb` falls from 18.74 to 3.46, so the memory guard admits it on Macs
  with less free memory.
- Otherwise identical to 0.4.2, and still requires `medmlx-core` 0.2.6.

## 0.4.2

- Requires `medmlx-core` 0.2.6. Provider extras pin the current MedMLX runtime
  releases, which also require `medmlx-core` 0.2.6.
- 48-channel full-resolution convolutions use 24-column tiles.
- `medmlx setup` and `medmlx setup MODEL_ID --plan` report `cached` and `revision`
  without downloading weights. Weight-free and manual models report `null` for both.
- `medmlx_core.weights.cached_weights(model_id)` checks completed downloads
  without network access, locks or cache writes.
- `medmlx setup --progress` reports downloaded bytes and totals on stderr as
  JSON lines. Status messages keep their existing format.
- Python setup calls accept `download_progress` for typed byte events. MCP setup
  jobs include these events in their log tail.
- Workflows declare `weights_from`, mapping each weight argument to the model
  whose automatic weights it uses. `reason-ct-tools` uses `mlx-reason-ct`,
  `lung-nodule` and `medsam2`: omitted weight paths resolve to those models'
  downloads, and `medmlx setup reason-ct-tools` prepares all three. Setup plans
  list them in `weights_from`. Explicit paths are still used as given, and the
  detector and MedSAM2 paths may be a model folder or its checkpoint file.

## 0.4.1

- Refresh provider extras to the current MedMLX package releases.
- Regenerate shared-runtime reference parameters from their seeds instead of
  storing them in Git, while keeping the recorded comparison outputs.

## 0.4.0

- Runners with configured assets obtain weights automatically when `weights_path` is omitted.
  They share `medmlx_core.weights.weights_dir(model_id)`, which downloads the
  published Hub repository or selected model files as ordinary cached files.
  Runners find their own files; explicit local weight folders remain supported.
  MedGemma, MRSegmentator and workflows with separately configured assets still
  require local paths.
- Completed downloads work offline. Interrupted downloads resume, concurrent
  callers share one download, and run details record the downloaded commit.
- Variants become separate model IDs. The package's bundle table, setup IDs,
  per-provider argument mappings and setup-time identity checks are removed.
- `medmlx setup --all` is an optional pre-download for installed models; manual
  asset requirements and partial completion exit nonzero. Individual supported
  pre-downloads return `weights_path` only.
- Weight-free models declare `requires_weights=False` and skip model downloads.
- `medmlx-core==0.2.5` is the common runtime dependency. Family requirements use
  the coordinated release versions, both in the source workspace and on PyPI.
- nnU-Net named models and FDG PET registration now live in their independent
  runtimes. MedMLX retains only its managed, isolated MedGemma bridge.
- `medmlx setup MODEL_ID --refresh` repairs or updates a managed weight folder.
- The managed MedGemma environment ignores caller project configuration, keeping
  its dependency graph separate from application overrides.
- Lung detection uses stable NumPy top-k by default. Custom selector libraries
  remain optional for reproducing the original tied-score selection order.
- The complete first-use workflow still requires compatible assets for models
  without automatic downloads.

## 0.3.2

- Model-family pins use the published provider versions compatible with
  `medmlx-core==0.2.4`. The `segment`, `generate`, `vision` and `all` extras
  can be installed from PyPI; model weights retain their separate terms.

- The provisional `medgemma` adapter can call an explicitly selected
  isolated MedVLM environment. They require local assets and preserve dependency
  pins, offline execution, overwrite protection and supervised cancellation.
  CPU bridge tests are not real-checkpoint or numerical qualification; see the
  [bridge contract](medvlm-bridge.md).
- `ready` remains a boolean with its existing provider/import plus Metal meaning.
  `runtime_ready` is an explicit alias. New clients should inspect `run_ready`:
  `false` means a known runtime, argument or path blocker; `null` means input and
  checkpoint contents remain unverified. No preflight result establishes a
  validated run. An importable runner or existing path is insufficient evidence.
- Readiness accepts one model's arguments for signature and local path checks.
  Parameter descriptions now expose provider prose, including wrapped argument
  groups and paragraphs naming arguments in backticks. Descriptions are not
  machine-validated constraints or independent qualification evidence.
- `segment`/`all` install SAM3 with video support; `generate`/`all` install
  NV-Reason-CT and the standalone FDG PET adapter. FDG synthesis requires an
  explicit tissue/scanner profile and matching CT/MAISI grids in mm. It has no
  learned weights or built-in physiological profile.
- `list_checkpoints` exposes five existing source references, including liver and
  AutoPET II research sources for the general nnU-Net provider. It does not add
  model IDs for those sources, download assets or claim checkpoint qualification.
- Jobs retain their server-session lifetime and cancellation behavior. Client
  registration and persistent filesystem access remain client configuration.

### MedGemma compatibility

The published `med-vlm-mlx` 0.2.2 package has no `medmlx.models` entry points.
Its declared dependency set cannot join this locked `all` environment:

| Dependency | MedVLM requirement | MedMLX all requirement |
| --- | --- | --- |
| MLX | `==0.32.3` | core `>=0.32.4,<0.33`; NV-Reason-CT `==0.32.4` |
| Hugging Face Hub | Transformers 5.19.0 requires `>=1.31.0,<3.0` | NV-Reason-CT 0.3.3 requires `==1.23.0` |
| Tokenizers | Transformers 5.19.0 requires `>=0.23.1,<0.24.0` | NV-Reason-CT 0.3.3 requires `==0.22.2` |

These conflicts come from declared/installed distribution metadata, not a failed
model inference. Changing the MedVLM MLX pin alone cannot resolve them. A shared
environment requires provider-owned dependency changes, validation of
MedGemma/Gemma3 on the selected versions, and NV-Reason-CT
regression checks for any Hub/tokenizer update. Only then can provider-owned MCP
entry points join the unified environment without dependency overrides. The
provisional separate-environment adapters avoid combining these dependency sets.

MedVLM's existing checkpoint smoke script uses real 4-bit checkpoints but a
synthetic solid-color image and at most 24 generated tokens. Passing that check
would establish loading and image-to-text execution only. Numerical parity and
medical task accuracy are separate claims. This MCP change does not run those
checkpoints, change the provider repositories or relax their version requirements.
