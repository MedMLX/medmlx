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
| `weights_from` | Map workflow weight arguments to model IDs, such as `{"detector_weights_path": "lung-nodule"}`. Setup prepares each model; resolve omitted paths with `medmlx_core.weights.weights_dir(model_id)`. |

Use the runtime's own labels, examples and measurements. Readiness checks runtime
availability; input and checkpoint validation happen during execution.

Invalid declarations and import failures appear in `discovery_errors`.
Duplicate IDs disable both declarations. Unknown fields appear in
`discovery_warnings`; MedMLX 0.4.1 and earlier reject them, so new fields need
a compatible MedMLX version.

## Reporting outputs

Return an `outputs` list describing every file the run wrote. Build each entry
with `medmlx_core.outputs.output_entry`.

```python
from medmlx_core.outputs import output_entry

return {"outputs": [output_entry(
    mask_path, kind="label_map", coordinate_space="image_voxel",
    labels=[{"value": 1, "name": "spleen"}], grid_of="image_path",
)]}
```

Use RadNN's output kinds and coordinate spaces. Name every nonzero label this
call can write. Mask kinds require `labels`. For prompted runs, use the names
from that call's prompts. `grid_of` names the input argument whose grid the
output shares. It and `units` default to `null`. Paths may be absolute or relative
to the run's working directory. Invalid metadata adds `outputs_error` to the job.
The run still completes and its result stays unchanged.

## Conventions for registered MedMLX runners

Registered learned-weight runners use the common optional model-folder contract
below. Image and output formats remain model-specific; `describe_model` exposes
an installed runner's actual signature. Family membership alone does not register
a model tool.

- Sub-packages target Apple Silicon only: macOS arm64 with MLX on Metal. Declare only that supported environment in `pyproject.toml`; do not provide backend extras. Reference fixtures are recorded on darwin-arm64 and loaded directly, without backend-keyed selection. Metal must be available for inference.
- The declaration and runner live in `src/<package>/medmlx.py`, exported as `MODEL` and `run`. Use the same name for the entry point, model `id`, and Hugging Face repository under `MedMLX`. Register each checkpoint variant as a separate ID.
- Model `task` is one of `segmentation`, `generation`, `detection`, or
  `feature-extraction`. Multi-model orchestration uses `workflow`.
- Prefer keyword-only arguments. File inputs should end in `_path` and directory inputs in `_dir`. Use `image_path` for one image or `image_paths` for ordered channels; name other inputs by role, such as `mask_path`. Use optional `weights_path` for the model folder. When omitted, call
  `medmlx_core.weights.weights_dir(model_id)` and locate files inside that folder.
  Use exactly the registered model ID as the helper key so pre-downloads and
  normal runs share the same folder and lock.
  Record `weights_revision(folder)` in result details. Use explicit paths as
  given without resolving links before suffix checks. Runners that write files should take `output_dir` and `overwrite: bool = False`, and refuse to replace existing outputs unless `overwrite` is true.
- For NIfTI volume workflows, document shape, affine and orientation changes. Segmentation masks and label maps should keep the input grid unless the model natively produces another grid. Other providers accept or produce raster images, arrays, video, or reports; state their formats explicitly.
- Return the `outputs` list described above. Extra result fields such as `details` must contain only JSON values.
- Failures raise `medmlx_core` errors with an actionable message: `InvalidInputError` for bad inputs, `AssetNotReadyError` for missing or mismatched weights, `MissingDependencyError` for a missing optional package.
- Native runners should use `medmlx_core.import_mlx()` and avoid source-framework imports in inference. Document any provider-specific dependency or isolated bridge instead of assuming a fallback device.
- Keep model declarations and package imports lightweight so discovery can report a model even when its runtime is unavailable. Runner imports and provider readiness checks can still fail; those failures are reported by discovery or readiness as applicable.

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
`nnunet-liver` and `nnunet-autopet`. AutoPET channels are CT, then SUV PET.
Use `setup` to download supported assets.
