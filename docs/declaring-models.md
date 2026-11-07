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
- `readiness` takes no arguments and returns `{"ready": bool, "reason": str | None, "setup_hint": str | None}`. Keep it cheap, because it runs in the server process. Without one, only runner importability is checked. MedMLX combines this with Metal availability in the historical `ready` boolean and its explicit alias `runtime_ready`. These fields do not validate run inputs or weights. The separate `run_ready` is `null` while input/checkpoint contents remain unverified, or `false` for a known blocker; the current check never returns `true`. Optional argument checks inspect signature binding and conventional input/weight paths without reading their contents. Packages do not repeat the Metal check.
- Write units, coordinate conventions, accepted values and failure conditions in the runner docstring. `describe_model` copies Google-style `Args` entries, including grouped names wrapped across lines. For other styles it uses introductory paragraphs that explicitly name an argument in backticks. Explicit `Args` entries take priority. Missing prose stays `null`; descriptions do not create validation rules. The full docstring remains in `instructions`.
- `requires_weights` is an optional boolean (default `True`). Set it to `False`
  for weight-free runtimes so optional pre-downloads skip them.
- `setup_supported` is an optional boolean (default `True`). Set it to `False`
  when a runner needs assets supplied by hand. Setup plans report it as
  unsupported, and setup reports a manual-setup error.
- `labels` optionally maps each integer output label to its name (`{"1": "spleen"}`); `summarize_output` uses the names. `example_arguments` optionally gives one JSON object of runner arguments, and `peak_memory_gb` the run's measured peak memory in GB; readiness reports when less memory is available but never blocks the run. Take each value from the package's own label map, docs or measurements.
- `weights_from` is an optional mapping for workflows that run other models. Each
  key is a weight argument of `run`; each value is the model ID whose automatic
  weights it uses, for example `{"detector_weights_path": "lung-nodule"}`.
  `medmlx setup` prepares those models and returns one folder per argument, and
  setup plans list the model IDs. Make the arguments optional and resolve an
  omitted one with `medmlx_core.weights.weights_dir(model_id)`.
- A declaration with a missing field, a bad reference, or an import error is reported in `discovery_errors`. When two entry points declare the same id, neither is served.
- Unknown fields are ignored and reported in `discovery_warnings`; the model still loads. MedMLX 0.4.1 and earlier reject them instead and skip the model, so a runtime adds a new field only in a release that the `medmlx` extras pin together with a MedMLX version that reads it.

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

`list_checkpoints` (CLI: `medmlx checkpoints`) returns source references separately
from installed model declarations. Its `runner_installed` flag means only that the
referenced runner was discovered. `qualification=not_assessed_by_catalog` preserves
that boundary even for named adapters. Liver (Zenodo 18807203) and AutoPET II
(Zenodo 8362371) correspond to the runtime-owned `nnunet-liver` and
`nnunet-autopet` models. AutoPET channels are CT, then SUV PET. The source
catalog does not download or qualify those checkpoints.
