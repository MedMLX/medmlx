# Review CT nodules with Reason-CT

`reason-ct-tools` combines Reason-CT, lung-nodule detection and MedSAM2
segmentation. Give it a chest CT to get an interactive review page, masks and
measurements.

## Run

From the MedMLX checkout with Python 3.13:

```sh
uv pip install --python .venv/bin/python -e '.[reason-tools]'
.venv/bin/medmlx setup reason-ct-tools --progress
```

Setup downloads the three models: `mlx-reason-ct`, `lung-nodule` and `medsam2`.
Source-checkout runs also download missing weights automatically. With the
published `medmlx` 0.4.7 package, supply `reason_weights_path`,
`detector_weights_path` and `medsam_weights_path` explicitly. These take model
folders; the detector and MedSAM2 also accept `.safetensors` files.

Save this as `arguments.json`, with paths relative to your working directory:

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

Use a new output directory for each run. The same model ID and arguments work
with Python `medmlx.run` and MCP `run_model`. Poll MCP jobs with `get_job`.

## What happens

1. The detector finds candidates above `score_threshold`, up to `max_candidates`.
2. Reason-CT chooses which returned candidates to segment with MedSAM2.
3. After one segmentation, it can finish or continue up to `max_segments`.
   It finishes immediately if no candidate passes the cutoff.
4. Reason-CT generates an interpretation from the CT and measured results.

The controller requires detection first and at least one segmentation when
candidates exist. `max_segments` sets an upper limit. Tool calls are constrained
to the allowed actions and returned candidate IDs; the model cannot supply
coordinates, paths or executable code. Calls and results feed into its next
decision. This uses the checkpoint's chat template and LLGuidance.

## Inputs and geometry

Use a finite, scalar 3D chest CT in Hounsfield units, saved as NIfTI with
consistent coded geometry. Set `assume_mm=true` only if its spatial units are
unknown and you know they are millimetres.

All three models use the same CT in RAS orientation. Orthogonal images are
reoriented by flips and axis permutations. Oblique or sheared images are
resampled linearly to an orthogonal grid at the input voxel sizes.

Detector boxes use RAS millimetres. The workflow transforms all eight corners
to voxel coordinates and derives MedSAM2's seed box and propagation interval.
Out-of-field boxes are excluded; clipped boxes are flagged. Candidate IDs keep
the original detector row. MedSAM2 tracks in both directions. Accepted masks
are nonempty, binary, on the working grid, within the interval and intersect
the seed prompt.

Masks return to the original storage grid by inverse orientation, or
nearest-neighbour resampling for oblique CT. Outputs keep the source affine
and spatial units. Volume and centroid are measured from source voxels using
the affine in millimetres, including anisotropy and shear.

## Results

`review.html` shows CT slices, masks, detector scores and measurements, with
slice browsing and NIfTI downloads. It works offline. `run.json` records the
steps, grids, resampling and tool conversation. Provider failures stop the run
and leave partial results. After cancellation, check the job state:
`run.json` may still show the interrupted step as running.

Scores express detector confidence. The cutoff defaults to 0.25. Diameter is
calculated from mask volume as an equivalent sphere. Tied detector scores
keep their original order, which can select different overlapping boxes from
upstream MONAI. Detection accuracy, segmentation accuracy and tool-agent
performance have not been established for this combined workflow.

## Development checks

```sh
make qa
make test TESTS='tests/test_reason_tools.py tests/test_api_cli.py tests/test_server.py'
```
