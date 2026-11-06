# Release notes

## Unreleased

- Pin registered automatic weight bundles to the revisions documented by the
  latest runtime release tags, including their linked Hugging Face model cards.
  SAM3 retains its existing pin. Unlisted model IDs still resolve
  `MedMLX/<model_id>` at `main` on first use. Existing weight files whose hashes
  match the pin can be reused across commits; changed cards and other selected
  files still use the resumable download path.

  The nnU-Net pins preserve these `dataset.json` foreground labels (background is 0):

  | Model ID | Weight revision and label source | Labels |
  | --- | --- | --- |
  | `nnunet-autopet` | [459d23802ae289be0f6539d40ca70c7facb1a9de](https://huggingface.co/MedMLX/nnunet-autopet-mlx/blob/459d23802ae289be0f6539d40ca70c7facb1a9de/dataset.json) | 1 tumor |
  | `nnunet-brats2021` | [7632a1f3c74d350eb6f2daeb35b43b75ef7601bd](https://huggingface.co/MedMLX/nnunet-brats-mlx/blob/7632a1f3c74d350eb6f2daeb35b43b75ef7601bd/dataset.json) | 1 edema, 2 nonenhancing, 3 empty, 4 enhancing |
  | `nnunet-liver` | [574a9c601b9b1a9e906ce985fbd46b106c5b7caa](https://huggingface.co/MedMLX/nnunet-liver-mlx/blob/574a9c601b9b1a9e906ce985fbd46b106c5b7caa/dataset.json) | 1 liver, 2 cancer |
  | `nnunet-pancreas` | [b2d5f14bb31f9793e3d9ab2c24096427eea983ae](https://huggingface.co/MedMLX/nnunet-pancreas-mlx/blob/b2d5f14bb31f9793e3d9ab2c24096427eea983ae/dataset.json) | 1 pancreas, 2 cancer |

- Automatic weight setup verifies unmarked or outdated caches against the selected
  Hub commit's LFS SHA-256 or Git blob hashes before reusing them. Verified files
  are kept, including files from the older top-level layout, which are copied into
  the current layout without deleting or linking the originals; only missing or
  mismatched files are downloaded. `cached_weights` remains offline and read-only;
  `refresh=True` still forces a download.

## 0.4.5

Updates the nnU-Net and TotalSegmentator models to
[`nnunet-mlx` 0.7.0](https://github.com/MedMLX/nnunet-mlx/releases/tag/v0.7.0) and
[`totalsegmentator-mlx` 0.1.7](https://github.com/MedMLX/totalsegmentator-mlx/releases/tag/v0.1.7).

### Breaking changes

**nnU-Net runs return the label map as `mask`.** `nnunet-liver` and
`nnunet-autopet` returned it as `segmentation`; the other nnU-Net models already
used `mask`.

**`nnunet-brats2021` averages all five released folds by default.** Pass
`fold=0` for the old result. Downloads from before November 2 hold only fold 0,
so refresh them first:

```sh
medmlx setup nnunet-brats2021 --refresh
medmlx setup nnunet-autopet --refresh   # AutoPET's folds 1-4
```

### Fixes

**TotalSegmentator no longer writes probability outputs through an existing
symlink.** Outputs are staged privately and renamed into place, and a dangling
symlink at an output path counts as an existing file.

### Faster

**Native 3-D nnU-Net inference is faster.** A full liver CT (AMOS22, M5 Max)
takes 466 s instead of 634 s, with identical labels.

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
