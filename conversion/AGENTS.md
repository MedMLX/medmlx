# medmlx-convert

Exports upstream PyTorch models to Core AI (`.aimodel`) and Core ML
(`.mlpackage`), and captures native MLX graphs for `medmlx-core`.

## Rules

- Export from the upstream PyTorch model and checkpoint, never from an MLX
  port.
- One recipe per model, `recipes/<model>/export.py`: it pins the upstream
  package and checkpoint, exports once with `torch.export`, converts to both
  formats, compares each with PyTorch and exits nonzero when a tolerance
  fails.
- Use fixed input shapes and FP32 unless the recipe documents why not. When a
  recipe lands or a verdict changes, update `README.md` and `docs/survey.md`.
- Native capture is an experimental fixed-shape pilot: unknown operators
  raise, with no CPU model fallback.
- `torch` is pinned to 2.13 for its security fixes; requalify both converters
  before changing it. Update the conda prefix without `--prune`.
- Never commit weights, `.aimodel`, `.mlpackage`, medical images or patient
  data; derived output goes to `build/`.

## Checks

- `make env` sets up the conversion environment (Core AI needs macOS 27).
- `make qa` runs lint and Pyright; `make test` runs converter contract checks.
- `make qualify` runs every recipe end to end and is opt-in.
