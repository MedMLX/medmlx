# Technical reference

## Runtime and checkpoint conversion

Only macOS arm64 with MLX on Metal is supported. `import_mlx()` admits the host
and selects the Metal GPU. Unsupported hosts are rejected before importing MLX;
unavailable Metal is an error. There are no backend extras or backend fallback.
The existing `require_mlx_device("mlx")` API checks explicit MLX readiness.
Python is `>=3.13,<3.15`; MLX is `>=0.32.4,<0.33`. Runtime admission rejects
older or unsupported MLX versions before inference.

The `conversion` extra supplies Torch for `load_torch_checkpoint()`. Safe
`weights_only=True` loading is the default. Explicit `weights_only=False` remains
available for trusted legacy payloads. Missing Torch reports `extra="conversion"`
and the corresponding install hint.

## Public API and references

Main helpers are re-exported from `medmlx_core`. Errors derive from the real
`MedmlxError` base and retain their built-in catch types: `InvalidInputError`
(`ValueError`), `MissingDependencyError` (`ImportError`), `ModelExecutionError`
(`RuntimeError`), and `AssetNotReadyError` (`FileNotFoundError`).

The canonical reference pins are in `tests/reference_spec.json`. MONAI is
1.6.1, revision [`7af492176792f9069b8f1253d099abd05c5e769d`](https://github.com/Project-MONAI/MONAI/tree/7af492176792f9069b8f1253d099abd05c5e769d).
PyTorch is 2.14.1, revision [`5c4886908584029761b579af026dcfb627c84070`](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070).
Development dependencies pin these package versions; generators reject a different
version or source revision. Reference frameworks are absent from runtime imports.

| Module | API / independent reference |
| --- | --- |
| `runtime.py` | Host readiness and Metal admission; direct platform/API contract tests |
| `errors.py` | Exception identity, messages and metadata; direct public contract tests |
| `checkpoints.py`, `conversion.py` | Duplicate-safe host mappings and optional Torch deserialization; original seeded arrays and Torch save/load round trips |
| `layout.py` | NCDHW/NDHWC, convolution-weight and token permutations; plain NumPy float64 reference |
| `ops.py` | SiLU, linear, conv3d, conv_transpose3d, GroupNorm, pooling, interpolation, padding and concatenation; PyTorch functional operations |
| `ops.py` split convolutions | Actual MONAI `MaisiConvolution`, including its overlap, integer size-ratio checks and crop semantics |
| `upsample.py` | Kernel=stride=2 transpose convolution and fused trilinear interpolation + skip addition; PyTorch functional operations |
| `precision.py` | Metal FP32 convolution and GroupNorm versus PyTorch; custom epsilon decomposition and Welford moments versus NumPy float64 |
| `sliding_window.py` | MONAI `inferers/utils.py` and `data/utils.py` |
| `channel_last.py` | PyTorch layers, MONAI Swish and `UnetResBlock`; individual layers and complete seeded 2D/3D graphs |

Inference helpers run on Metal. Model topology, checkpoint keys, label meanings and
pre/postprocessing belong to each model package. Channel-last references use the
core's MONAI 1.6.1 pin; they do not qualify consumer models pinned to another release.

See the [decoder note](transpose-convolution.md) for stride-2 transpose-convolution
dispatch and the upstream reproducer.

## Sliding-window inference

```python
from medmlx_core import sliding_window_inference

logits = sliding_window_inference(
    image,  # NumPy or MLX (N, C, *spatial), float32/float16
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    predictor=network,  # MLX array -> MLX array, same spatial resolution
    overlap=0.25,
    mode="constant",
    padding_mode="replicate",
)
```

The result is a host NumPy array with the input dtype and original spatial size;
output channels follow the predictor. Public helpers include `dense_patch_slices`
and `compute_importance_map`. Scalar/per-axis overlap and sigma, fallback ROI
components, all four Torch padding modes, supplied ROI maps, window batching
across images, predictor arguments, and `with_coord` are supported. Batch size
is used as supplied. The existing `sw_device` parameter accepts None, `"gpu"`
or an MLX GPU device; CPU prediction is rejected. `device` is None or `"cpu"`
because NumPy accumulation stays on the host. `import_mlx()` admits Metal before
the predictor is invoked.

Window stride is `max(int(roi * (1-overlap)), 1)`, with the final window shifted
onto the edge. Small inputs receive symmetric padding, with an extra voxel on
the right for odd deficits. Gaussian maps use FP32 separable factors, center
`(size-1)/2`, sigma `size*sigma_scale`, no peak normalization, and minimum 0.001.
Predictions are weighted in predictor dtype before ordered input-dtype accumulation.

Multiple/scaled outputs, positive `buffer_steps`, `process_fn`, progress bars,
and MetaTensor metadata are unsupported and fail explicitly where applicable.
No background threshold or class-ID mapping is applied by the shared stitcher.
Recorded postprocessing references use MONAI `AsDiscrete(argmax=True)` only.

## Reference fixtures and regeneration

Seven small upstream-reference archives are included under
`tests/fixtures/darwin-arm64/` (each below 80 KB). Inputs retain the original
seeds: 7081 for helpers and 20261007 for channel-last/window graphs. Coverage
includes input/layout preparation, every helper, padding/cropping, window logits,
coordinates, importance maps, argmax labels, and complete synthetic layer graphs.
Each archive has `schema_version=2`, upstream versions/revisions, seed, reference
device, NumPy version, Torch build and recording platform. Generators use Torch
CPU and NumPy, import neither MedMLX nor MLX, and are never imported by the
numerical tests.
The reference frameworks execute on the CPU of the recording Mac; the library
itself runs on MLX Metal. Tests load the one recording directory directly, without
backend selection. After explicitly provisioning the unified repository's development
environment, run regeneration from the component directory. Starting at the
unified repository root on an Apple Silicon Mac:

```sh
cd packages/core
PYTHON=../../.venv/bin/python

"$PYTHON" scripts/make_reference_fixtures.py
"$PYTHON" scripts/make_channel_last_fixtures.py
"$PYTHON" scripts/make_sliding_window_fixtures.py

"$PYTHON" -m pytest -q tests/test_reference_fixtures.py tests/test_numerical.py
```

Those tests check fixture metadata, coverage and numerical-budget parsing without
qualifying Metal arithmetic. For changed numerical operations, also select the
affected cases from `tests/test_equivalence.py`, `tests/test_channel_last.py` or
`tests/test_sliding_window.py`; the first two require Metal at collection.

Generators require macOS arm64. `--output-dir` permits regeneration in a temporary
directory for review. Missing references fail with a regeneration hint.

Numerical policy is versioned in `tests/numerical_contract.json`. Every comparison
checks shape, FP32 dtype and finite values. Copies, permutations, nearest-neighbor
replication and discrete outputs compare exactly. Arithmetic compares with
operation-specific absolute and relative bounds, accounting for reduction order,
FMA and exponential rounding. Each channel-last layer is checked on its upstream
stage input; complete graphs have a separate cumulative budget. The triage and
bound rationale are in `tests/NUMERICAL_PARITY.md`. The repeated 48-cubed decoder
test uses seeded weights and requires no external checkpoint.

Independent sliding-window tests retain exact comparison for representable
constant-blending inputs, maximum absolute error 1e-6 for Gaussian logits and
1e-7 for Gaussian maps because NumPy/Torch FP32 exponential kernels round differently.

Host checks and reference generation do not qualify MLX execution. Reference arrays
retain independently recorded upstream outputs and source metadata. Run `make qa-core`
from the unified repository root for the current static gate; passing runtime tests
alone does not establish static-check status.

Fixtures are synthetic helper references. Clinical accuracy, released model
weights and full-volume memory/performance are not qualified by these fixtures.
