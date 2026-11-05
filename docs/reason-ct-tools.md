# Reason-CT with nodule tools

`reason-ct-tools` connects Reason-CT, the lung-nodule detector and MedSAM2 in one
supervised workflow. It writes an interactive local CT review, source-grid
binary masks, measurements and the complete tool conversation.

Install the `reason-tools` extra. It provides Reason-CT 0.3.5, whose runner
accepts `tools`, `tool_history` and `response_choices`. From the MedMLX checkout,
in a prepared Python 3.13 environment:

```sh
uv pip install --python .venv/bin/python -e '.[reason-tools]'
.venv/bin/medmlx describe reason-ct-tools
```

`readiness` checks the Reason-CT API before running. The workflow uses the
weights of three models: `mlx-reason-ct`, `lung-nodule` and `medsam2`. Omit
their paths and each one comes from that model's automatic download, shared with
its own runner. To download all three ahead of time:

```sh
.venv/bin/medmlx setup reason-ct-tools --progress
```

To use your own weights, pass `reason_weights_path` (a Reason-CT folder),
`detector_weights_path` and `medsam_weights_path` (a model folder or its
`.safetensors` checkpoint). Explicit paths never trigger downloads. The published
`medmlx` 0.4.3 package requires all three paths.

Create `arguments.json`, using paths relative to the execution root:

```json
{
  "image_path": "chest.nii.gz",
  "output_dir": "outputs/nodule-review",
  "score_threshold": 0.25,
  "max_candidates": 5,
  "max_segments": 2
}
```

```sh
.venv/bin/medmlx readiness reason-ct-tools --arguments arguments.json
.venv/bin/medmlx run reason-ct-tools --arguments arguments.json
open outputs/nodule-review/review.html
```

The same ID and arguments work through Python `medmlx.run` and MCP `run_model`;
MCP returns a job to poll with `get_job`. The existing supervised worker owns
Metal admission, cancellation and its process group. All three providers execute
serially inside that worker. No nested detached jobs are created.

## Model decisions and execution

The controller first permits `detect_nodules`, then offers
`segment_candidate(candidate_id)` for the returned candidates. After one mask,
or when no candidate passes the cutoff, it permits `finish`. It enforces the
configured candidate, segmentation and decision limits. The model chooses among
the currently allowed calls using its logits; the first call and the required
minimum workflow are imposed by the controller. `max_segments` is a cap, not a
promise that every candidate will be segmented.

Native Qwen tool syntax is rendered by the checkpoint's own chat template.
LLGuidance constrains greedy token selection to complete allowed calls, including
the tokenizer's special tool tokens. Plain prompting alone is not assumed to
produce valid calls. Calls and tool replies are fed back into the next decision;
final interpretation is generated separately from the CT and measured evidence.
This does not establish that the checkpoint was trained or evaluated as a tool
agent. Its unconstrained interpretation can still be wrong.

The model cannot supply filenames, coordinates, executable code, model IDs or
runtime options. Only returned candidate IDs can reach MedSAM2. Malformed,
unknown or repeated requests receive explicit errors and consume a decision.
Provider failures stop execution; partial evidence remains in `run.json`.
After cancellation, the job state is authoritative; `run.json` may retain the
last running step because terminating a worker does not run Python cleanup.
Existing output directories are rejected.

## Spatial contract

- Input is a finite scalar 3D HU chest CT with consistent, coded NIfTI geometry.
  Unknown spatial units require the explicit `assume_mm=true` option.
- NiBabel normalizes physical units and reorients orthogonal inputs to RAS by
  permutation/flips. Oblique or sheared inputs are linearly resampled to an
  orthogonal RAS grid at the canonical input voxel sizes. All models receive
  that same saved CT. The trace records both grids and any resampling.
- Detector boxes are RAS centres and sizes in millimetres. All eight corners
  are transformed into working voxels. The controller derives the axial seed,
  XY box and physically padded propagation interval; the model supplies no
  spatial coordinates. Out-of-field boxes are excluded and counted; clipped
  boxes are flagged. Candidate IDs preserve their original detector row.
- MedSAM2 runs bidirectionally over the requested interval. Its mask must retain
  the working shape, affine and millimetre units, be nonempty and binary, remain
  within the interval, and intersect its seed prompt.
- Masks return to the original storage grid by exact inverse orientation, or
  nearest-neighbour resampling when the CT was oblique. Exact inverse orientation
  avoids losing boundary voxels to serialized-affine rounding. Output preserves
  the original affine and spatial units. Volume and centroid use source voxels
  and the determinant of the affine in millimetres, including anisotropy/shear.

`review.html` embeds actual CT and mask pixels with a full axial overview,
three detail views, slice browsing, orientation labels, mask visibility controls
and source-grid NIfTI downloads. CT and mask use the same display transform and
physical pixel aspect. It displays the cutoff, detector scores, segmentation
count and any CT resampling. It has no external assets or network requests.

Detector scores are not calibrated malignancy probabilities. The default cutoff
is a workflow parameter, not a validated clinical operating point. The displayed
diameter is the diameter of a sphere with the measured mask volume, not a RECIST
diameter. A low-score demo or an empty candidate list does not establish diagnosis
or absence of disease. These outputs require human review.

The released lung detector orders tied scores by original index. With tied
scores, its selection of overlapping boxes can differ from upstream MONAI; the
detector's recorded comparisons cover this limitation.

## Checks

```sh
make qa
make test TESTS='tests/test_reason_tools.py tests/test_api_cli.py tests/test_server.py'
```

Synthetic checks cover flips, axis permutations, units, oblique resampling,
boundary voxels, mask admission, candidate identity and bounded execution.
Real checkpoint execution and visual review are separate evidence; neither
establishes detection accuracy, segmentation accuracy or clinical validity.
