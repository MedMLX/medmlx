# Declaring a model

A model package declares each model as an entry point in the `medmlx.models` group. The entry point names a mapping, or a callable that returns one. The package does not import this server.

```toml
[project.entry-points."medmlx.models"]
totalsegmentator = "totalsegmentator_mlx.medmlx:MODEL"
```

```python
MODEL = {
    "id": "totalsegmentator",                            # lowercase letters, digits, ".", "_", "-"
    "label": "TotalSegmentator",
    "task": "segmentation",
    "run": "totalsegmentator_mlx.medmlx:run",            # module:function
    "readiness": "totalsegmentator_mlx.medmlx:readiness",  # optional, module:function
    "summary": "Whole-body CT segmentation.",  # optional
}
```

- `run` receives the JSON keyword arguments from `run_model` and returns a JSON object (`dict`) containing JSON values. The runner converts paths to strings and arrays to lists before returning. Non-finite numbers and unsupported objects fail the run. Raise an exception to fail the run; its type and message reach the client.
- `readiness` takes no arguments and returns `{"ready": bool, "reason": str | None, "setup_hint": str | None}`. Keep it cheap, because it runs in the server process. Without one, a model is ready when its `run` function imports. The server separately reports a model as not ready when MLX itself is unusable, so packages do not repeat that check.
- A declaration with a missing or unknown field, a bad reference, or an import error is reported in `discovery_errors`. When two entry points declare the same id, neither is served.

## Conventions for registered MedMLX runners

Model packages that register a runner follow these rules so clients can drive
them consistently. Family membership alone does not register a model tool.

- Sub-packages target Apple Silicon only: macOS arm64 with MLX on Metal. Declare only that supported environment in `pyproject.toml`; do not provide backend extras. Reference fixtures are recorded on darwin-arm64 and loaded directly, without backend-keyed selection. Metal must be available for inference.
- The declaration and runner live in `src/<package>/medmlx.py`, exported as `MODEL` and `run`. The entry point name and `id` are the distribution name without its `-mlx` suffix, for example `totalsegmentator` for `totalsegmentator-mlx`.
- `task` is one of `segmentation`, `generation`, `detection`, or `feature-extraction`.
- `run` takes keyword-only arguments. File inputs end in `_path` and directory inputs in `_dir`. The primary image input is `image_path`, or `image_paths` when a model takes one image per channel; other inputs say what they are, such as `mask_path`. Weights are always explicit (`weights_path`, or `<part>_weights_path` when a model has several), never found by searching. Every model that writes files takes `output_dir` and `overwrite: bool = False`, and refuses to replace existing outputs unless `overwrite` is true.
- Volumes are read and written as NIfTI. Segmentation masks and label maps keep the input image's shape and affine unless the model natively produces another grid, which its docstring then states.
- `run` returns `{"outputs": {role: path}, "details": {...}}`, with absolute POSIX paths and only JSON values in `details`.
- Failures raise `medmlx_core` errors with an actionable message: `InvalidInputError` for bad inputs, `AssetNotReadyError` for missing or mismatched weights, `MissingDependencyError` for a missing optional package.
- MLX is imported through `medmlx_core.import_mlx()`; there is no device argument and no fallback. Runtime code does not import `torch` or `monai`.
- Importing `<package>.medmlx`, and so the package itself, does not import MLX: the package `__init__` loads its public names on first access and `run` loads the network. On a machine where MLX is broken, the model is then still listed and reported as not ready instead of disappearing into `discovery_errors`.

## Bundled named nnU-Net models

MedMLX declares `nnunet-pancreas`, `nnunet-brats2021` and `mrsegmentator` through
small adapters to the independent `nnunet-mlx` runtime, installed by `segment`.
Supply `model_dir` explicitly; package readiness does not validate those assets.

Pancreas requires the inference-staged, converted Dataset999_PancCTPretrain
checkpoint (`3d_fullres_bs4`), with background, pancreas and cancer labels.
BraTS requires converted Dataset002_BRATS19 (`3d_fullres`) and four co-registered
NIfTI paths. Its channel order is T1, T1ce, T2, FLAIR with the released BAMF label
map. Both accept a fold and a mirroring selection.

MRSegmentator requires converted 1.2 base weights. The adapter selects the
`mrsegmentator-1.2-base` profile and `torch-order-fp32` arithmetic, with mirroring
disabled. Its default uses all five folds; `fast=True` selects fold 0, and an
explicit `fold` takes precedence. Missing or incompatible model folders fail.

`totalsegmentator-mlx` owns its own CT pipeline and `totalsegmentator` entry point.
Its README documents converted model directories, modes and organ selection.
These adapters preserve each runtime's existing qualification scope.
