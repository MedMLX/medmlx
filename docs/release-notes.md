# Release notes

## Unreleased MCP completion

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

### MedGemma and Lingshu compatibility

The examined `med-vlm-mlx` 0.2.0 revision
`7a8e3778670760c039438b646ad40df0244631c9` has no `medmlx.models` entry points.
Its declared dependency set cannot join this locked `all` environment:

| Dependency | MedVLM requirement | MedMLX all requirement |
| --- | --- | --- |
| MLX | `==0.32.3` | core `>=0.32.4,<0.33`; NV-Reason-CT `==0.32.4` |
| Hugging Face Hub | Transformers 5.19.0 requires `>=1.31.0,<3.0` | NV-Reason-CT 0.3.2 requires `==1.23.0` |
| Tokenizers | Transformers 5.19.0 requires `>=0.23.1,<0.24.0` | NV-Reason-CT 0.3.2 requires `==0.22.2` |

These conflicts come from declared/installed distribution metadata, not a failed
model inference. Changing the MedVLM MLX pin alone cannot resolve them. A shared
environment requires provider-owned dependency changes, validation of both
MedGemma/Gemma3 and Lingshu/Qwen2.5-VL on the selected versions, and NV-Reason-CT
regression checks for any Hub/tokenizer update. Only then can provider-owned MCP
entry points be integrated without dependency overrides.

MedVLM's existing checkpoint smoke script uses real 4-bit checkpoints but a
synthetic solid-color image and at most 24 generated tokens. Passing that check
would establish loading and image-to-text execution only. Numerical parity and
medical task accuracy are separate claims. This MCP change does not run those
checkpoints, change the provider repositories or relax their version requirements.
