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
    "run": "totalsegmentator_mlx.medmlx:run",            # module:function, always named run
    "readiness": "totalsegmentator_mlx.medmlx:readiness",  # optional, module:function
    "summary": "Whole-body CT segmentation.",  # optional
}
```

`run` receives JSON keyword arguments and returns a JSON-compatible dictionary.
Convert paths to strings and arrays to lists; non-finite numbers are rejected.
Exceptions fail the job and send their type and message to the client.

Document units, coordinates, accepted values and errors in the runner docstring.
`describe_model` uses Google-style `Args` entries for parameter descriptions,
or introductory paragraphs that name arguments in backticks. The full docstring
is returned as `instructions`.

| Optional field | Purpose |
| --- | --- |
| `readiness` | No-argument function returning `ready`, `reason` and `setup_hint`. Keep it cheap; MedMLX adds the Metal check. Without it, only runner imports are checked. |
| `requires_weights` | Defaults to `True`; set to `False` for weight-free runtimes. |
| `setup_supported` | Defaults to `True`; set to `False` when assets must be supplied manually. |
| `labels` | Integer label names, such as `{"1": "spleen"}`, for output summaries. |
| `example_arguments` | One runnable argument object. |
| `peak_memory_gb` | Measured peak memory in GB; readiness reports fit without blocking runs. |
| `cohort` | The package's `load_model`, for prepared cohorts; see [Reusing a loaded model](#reusing-a-loaded-model). |
| `weights_from` | Map workflow weight arguments to model IDs, such as `{"detector_weights_path": "lung-nodule"}`. Setup prepares each model; resolve omitted paths with `medmlx_core.weights.weights_dir(model_id)`. |

Use the runtime's own labels, examples and measurements. Readiness checks runtime
availability; input and checkpoint validation happen during execution.

Invalid declarations and import failures appear in `discovery_errors`.
Duplicate IDs disable both declarations. Unknown fields appear in
`discovery_warnings`; MedMLX 0.4.1 and earlier reject them, so new fields need
a compatible MedMLX version.

## Reporting outputs

Return `outputs` as a map from role to the path of every file the run wrote, so
callers read `result["outputs"]["mask"]`. Describe a role in `artifacts` with
`medmlx_core.outputs.artifact`; every artifact role must also be an `outputs`
role.

```python
from medmlx_core.outputs import artifact

return {
    "outputs": {"mask": mask_path.as_posix(), "record": record_path.as_posix()},
    "artifacts": {"mask": artifact(
        kind="label_map", coordinate_space="image_voxel",
        labels=[{"value": 1, "name": "spleen"}], grid_of="image_path",
    )},
}
```

Use RadNN's output kinds and coordinate spaces. Name every nonzero label this
call can write. Mask kinds require `labels`. For prompted runs, use the names
from that call's prompts. `grid_of` names the input argument whose grid the
output shares. It and `units` default to `null`. Paths may be absolute or relative
to the run's working directory. Invalid `outputs` or `artifacts` add
`outputs_error` to the job. The run still completes and its result stays
unchanged.

## Runtime API

Every runtime exposes the same small API, so code written for one model works
for the next. `describe_model` shows an installed runner's actual signature.

### One `run` per package

Every runtime's entry point is a function named `run`, whatever the task; the
model ID already says what it does. `task` describes the outputs:

| `task` | Usual output roles |
| --- | --- |
| `segmentation` | `mask` |
| `detection` | `detections` |
| `generation` | `image`, `mask` |
| `feature-extraction` | `features` |
| `report-generation` | `report` |

- `run` is exported from the package root (`from unest_mlx import run`) and is
  the function the declaration's `run` names; discovery rejects any other name.
  Keep the root import free of MLX by exporting lazily.
- `workflow` declarations orchestrate other models and may name any runner.
- A package whose models share one set of arguments has one `run` with a
  required `model` keyword holding the model ID:
  `run(model="radimagenet-resnet50", ...)`. MedMLX passes the declaration's ID
  and hides `model` from callers.
- Models whose inputs or options differ each get a module with its own `run`,
  so `describe` lists exactly their arguments: `nnunet_mlx.named_models.liver.run`.
- Don't add aliases such as `segment()` or `detect()`: `run` is the only one-shot
  entry point.

### Arguments

All public arguments are keyword-only and accept `str` or `os.PathLike` paths.
Fixed choices are `Literal` types, so `describe_model` lists them.

| Argument | Meaning |
| --- | --- |
| `image_path` | The case's one input image. |
| `image_paths` | Several co-registered images of one case, as a map from the model's channel names to paths: `{"ct": "ct.nii.gz", "pet": "suv.nii.gz"}`. Keys must match exactly; order never matters. |
| `<role>_path` | Other inputs, named by role: `mask_path`, `video_path`. |
| `output_dir` | Required. Directory for the output files, created if missing. |
| `weights_path=None` | The model's weights folder, as published. `None` downloads the pinned folder on first use. |
| `overwrite=False` | Replace existing outputs only when true. |
| `precision` | The model's numerical mode, where it offers more than one. |
| `scratch_dir=None` | Parent folder for temporary memory-mapped files. |
| `seed` | Integer seed for stochastic models. |
| `prompt` | Text the model reads, such as a question or a target description. |
| `classes` | Output classes to produce or keep, by name or class ID. |
| `frame_index` | 0-based slice or frame for 2D prompts. |

Other arguments are model-specific. Check every argument, input and output target
before loading or downloading weights.

`medmlx_core.files` holds the shared checks: `input_file`, `input_channels` for
`image_paths`, `output_targets` for `output_dir` and `overwrite`, and `publish`,
which writes a run's files together and restores earlier ones on failure.
`medmlx_core.weights.model_folder(model_id, weights_path)` returns the caller's
folder or downloads the managed one; record `weights_revision(folder)` in
`details`. Use exactly the registered model ID so `medmlx setup` and runs share a folder.

### Results and errors

Return `outputs` and `artifacts` as described above, plus JSON-only `details`.
Name roles by content, not file format: `mask`, `image`, `probabilities`,
`detections`, `features`, `report`, `metadata`. Masks keep the input grid unless
the model natively produces another; document any shape, affine or orientation change.

Raise `medmlx_core` errors with an actionable message: `InvalidInputError` for bad
inputs, `AssetNotReadyError` for missing or mismatched weights,
`MissingDependencyError` for a missing package and `ModelExecutionError` for
failures during inference. Model-specific errors subclass one of these. `medmlx.run`
and `ModelRun.wait` raise `medmlx.ModelRunError`, whose `category` names the shared
error and whose `hint` carries its setup hint.

### Reusing a loaded model

A package whose model is worth keeping loaded across cases exports
`load_model(*, weights_path=None, ...)`, taking `model` too when it declares
several. It returns an object with a `run` method and `close()`, and
works as a context manager:

```python
from nnunet_mlx.named_models.liver import load_model

with load_model() as model:
    for case in ("a.nii.gz", "b.nii.gz"):
        model.run(image_path=case, output_dir=f"out/{case[:-7]}")
```

Arguments that change the loaded weights or compiled graph belong to `load_model`;
per-case arguments belong to the method. Declare it as `cohort` so
`medmlx.start_cohort` reuses one loaded model for a list of cases. The one-shot
`run` releases its memory before returning; don't keep hidden global caches.

### Platform

- Runtimes target macOS arm64 with MLX on Metal. Declare only that environment in
  `pyproject.toml`; there are no backend extras or CPU fallbacks for inference.
- Use `medmlx_core.import_mlx()` and keep source frameworks out of inference.
- Keep declarations and package imports light so discovery can report a model even
  when its runtime is unavailable.

## Models declared by runtimes

Each runtime registers its own models. `nnunet-mlx` declares `nnunet-pancreas`,
`nnunet-brats2021`, `mrsegmentator`, `nnunet-liver` and `nnunet-autopet`;
`totalsegmentator-mlx` declares `totalsegmentator`; `fdg-pet-ct-mlx` declares
`fdg-pet-ct`. Their READMEs document weights, inputs and options. `medmlx describe
<model>` shows the arguments of any installed model.

## Checkpoint discovery

`list_checkpoints` (CLI: `medmlx checkpoints`) lists source references and
channel order. `runner_installed` reports whether a runner was discovered;
`qualification=not_assessed_by_catalog` is the catalog's status.

Liver (Zenodo 18807203) and AutoPET II (Zenodo 8362371) correspond to
`nnunet-liver` and `nnunet-autopet`. AutoPET takes `image_paths={"ct": ..., "pet": ...}`.
Use `setup` to download supported assets.
