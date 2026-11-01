# Release notes

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
