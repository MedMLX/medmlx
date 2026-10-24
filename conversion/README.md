# MedMLX conversion

Convert upstream model graphs and weights to native MLX, Apple's Core AI
(`.aimodel`) and Core ML (`.mlpackage`), with comparisons against the source model.

- Core AI and Core ML exports from pinned upstream PyTorch models
- Numerical comparisons against the source model for both formats
- Per-model recipes and documented converter coverage
- Versioned numerical receipts beside the exported artifacts
- Replayable source-input/output fixtures and explicit Apple runtime profiling
- Source, graph and native numerical checks for every recipe, including PSNR

Part of [MedMLX](https://github.com/MedMLX).
[Model survey](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/survey.md) · [Toolchains](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/toolchains.md) ·
[Runtime tuning instructions](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/runtime-tuning.html)

The experimental [native MLX conversion pilot](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/native-conversion.html)
captures runtime graphs and tensor state together. It includes a released
BraTS fold-0 network recipe, independent source comparisons, and
TensorFlow-checked SynthMorph geometry adapters. Full SynthMorph model conversion
and catalog-wide native conversion remain open. Native inference imports neither
Torch nor Core AI/Core ML. Its graph format, loader, Metal execution and shared
geometry live in `medmlx-core`; model applications install core without this
converter's dependencies. Existing hand-written MLX ports remain separate.

Development setup uses `../packages/core` in this repository through uv. The
runtime contract is graph schema 2, provided by `medmlx-core` (pinned to `==0.2.3`
in `pyproject.toml`); regenerate schema-1 pilot bundles with the current recipes. The package import is
`medmlx_convert`; no old-name compatibility package is provided.

The MedMLX repos run their models in MLX. Neither Apple converter reads MLX, so
each recipe here starts from the upstream PyTorch checkpoint that the MLX port
was converted from. It captures graphs with `torch.export`, converts them to
both formats, and fails if either drifts from PyTorch beyond its tolerance.
Original Keras checkpoints use Keras's maintained Torch backend, with a separate
CPU TensorFlow reference check before conversion.

[Quick start](#quick-start) · [Status](#status) · [Adding a recipe](#adding-a-recipe) · [License](#license)

## Requirements

- An Apple silicon Mac on macOS 27 or later. Core AI models only load on 27.
- Python 3.13. The locked converter environment uses Torch 2.13.
- mamba or conda, and [uv](https://docs.astral.sh/uv/).

## Quick start

```bash
make env
bash environment/setup.sh --run python recipes/flexible_unet/export.py
```

Output goes to `build/<model>/`. Upstream checkpoints download to the Hugging Face
or torch hub cache on first run. TotalSegmentator uses a converter-owned source
cache because other ports can rewrite its shared plans. Explicit
`TOTALSEG_WEIGHTS_PATH` and `TOTALSEG_HOME_DIR` overrides remain supported and
must contain pristine upstream files. `make qa` runs lint and Pyright without
executing tests or exporting models. `make test` runs converter contract checks;
use `TESTS` and `PYTEST_ARGS` to select affected cases. `make qualify` explicitly
runs the full model recipes; it is not
needed for repository moves or documentation changes. Tests use an isolated system-temp workspace and remove generated assets
when the session exits. Manual recipe runs retain
`*.verification.json` under `build/<recipe>/`, with shapes, tolerance, per-format
error, label agreement where applicable, runtime policy, package versions and
source/checkpoint revisions and the explicit build precision.
Separate test workspaces prevent concurrent writers from overwriting native
assets and source fixtures during comparison.

Each backend's prepared model must meet its original absolute tolerance and
exact equality or peak-normalized PSNR above 70 dB before conversion. Capture,
decomposition and each manual graph-edit stage are replayed independently
against the source; each retains its actual numerical comparison and graph.
Both saved native assets are then reloaded and compared to the source. FP32
native outputs require their original tolerance and normalized PSNR at least
60 dB. The owner enforces these checks during export; numerical reports retain
their results. There is no separate policy report to re-admit them.
See [toolchains](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/toolchains.md#apple-guidance-checks) for the Apple sources,
precision criteria and deployment scope.

Recipes declare their upstream revisions, checkpoint and configuration paths
where they are loaded. The final numerical report records those identifiers,
the converter Git revision, dirty state and installed package versions.
Ensemble recipes select their declared members directly. Representative
application/device qualification and broader catalog coverage remain open.
The selected deployment target is an Apple Silicon Mac with FP16 builds. Every
maintained recipe accepts an explicit mixed-precision network profile:

```bash
bash environment/setup.sh --run python recipes/flexible_unet/export.py \
  --profile macos-fp16-network --source-device mps
```

FP16 output defaults to `build/<model>-fp16/`. Select `--source-device cpu`
explicitly when the upstream source cannot run on MPS; no implicit device
fallback is permitted. This controls source/graph replay, while native runtime
checks use Core AI default specialization. Core ML defaults to `ALL`; select
`--coreml-profile coreml-gpu` explicitly to convert and reload with `CPU_AND_GPU`.
These are separate checked routes. Swin UNETR's FP16 `ALL` attempt crashed in
Apple's ANECompiler; prolonged ANE compilation also interrupted other attempts.
The remaining Mac checks use the explicit GPU route, with earlier attempts
retained in the audit. This route establishes no ANE coverage.
Standard Torch normalization keeps original FP32 weights and computes in FP32;
module LogSoftmax uses FP32 `x-logsumexp`. Remaining floating parameters and
network inputs become FP16; public floating I/O stays FP32. Recipes can declare
exact custom normalization types for FP32 handling; MedSAM2 declares LayerNorm2d.
Other functional/custom operations are unchanged and can reject this strategy. Keras Torch
graph layers receive an explicit FP16 compute policy; their normalization keeps
upstream FP32 arithmetic with rounded weights. CPU FP16 eager and graph replay
share the explicit math attention kernel. See [the implementation choices](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/toolchains.md#apple-guidance-checks).

Source precision drift and saved-native outputs require exact equality or raw
Apple PSNR strictly above 50 dB against the independent CPU FP32 oracle. Every
graph transformation separately requires exact equality or normalized PSNR
above 70 dB against its FP16 eager oracle. Original absolute limits remain
diagnostic for FP16 and mandatory for FP32/backend rewrites. FleXray declares independent logit-zero decisions
agreement ≥0.999. These engineering budgets measure source drift, not accuracy.
Other models still need model-specific task acceptance budgets. The audit
retains every attempted route, incomplete builds and explicit CPU retries.

The local test panel matches its recorded manifest identities: five LUNA16
images, plus 47 original image/label files for AMOS22 CT, MSD brain MRI, paired
autoPET PET/CT and Montgomery radiographs. Model and pipeline checks on the
AMOS22, MSD brain, autoPET and Montgomery cases remain pending. The
maintained R231 volume checker records original preprocessing, each network
slice, labels, postprocessing, inverse crop/orientation and saved NIfTI geometry.
Its prospective Dice ≥0.999 and voxel agreement ≥0.9999 criteria compare with
CPU source predictions; the panel has no lung ground truth.
Each recipe also saves `*.runtime.npz` with the exact synthetic source inputs
and outputs. Benchmark admission requires a passing numerical report with the
declared build precision and a finite source fixture with matching names, shapes
and dtypes. The report also records the recipe, Git revision, dirty state,
checkpoint/configuration identifiers and package versions.
Generated artifacts and reports are not hashed or sealed into a second inventory.
Missing or incompatible measurements require a new recipe run. Runtime output checks enforce
the admitted precision profile and explicit label requirements for every first, warm-up and timed
call, plus the separate Core AI profiling call. Comparisons occur outside each
inference timing interval. A failure stops the run and retains its phase,
one-based iteration, build profile and error/comparisons in schema-4 benchmark JSON. Follow the
[runtime tuning procedure](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/runtime-tuning.html)
to measure loading, input preparation and completed inference under explicit
Core AI and Core ML policies without changing the model's behavior.

The same runtime owner now measures six actual MLX ports with the identical
admitted FP32 fixtures. The paired driver reverses process order and excludes
the entire model if any backend or process fails. The
[matched comparison procedure](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/runtime-tuning.html) documents the locked
environment, explicit source checkouts and host-to-host timing boundary.

## Status

Updated 2026-10-16. Coverage is per checkpoint, fold and component, as listed
below; **the entire MedMLX catalog is not yet exported**. This table describes
the FP32 reference profile. The recipes test fixed
synthetic network inputs against upstream CPU FP32. These results do not test
complete image preprocessing, sliding windows, generation loops, clinical
accuracy or Neural Engine utilization. The runtime policies are Core ML `ALL`
and Core AI default specialization.
[docs/survey.md](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/survey.md) has the details for each repo, and
[docs/toolchains.md](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/toolchains.md) covers the converters, op coverage and
Apple's resources.

A Hugging Face file-tree snapshot taken on 2026-10-16 covered 18 MedMLX
repositories containing MLX weights. No Core AI or Core ML assets were visible
in those inspected `main`-branch file trees. Later repository updates have not
had their file trees rechecked. Local converted artifacts and Hub
distribution coverage are separate: the observed FleXray upload has only
`flux0375`, and BraTS and AutoPET only `fold_0`.

| Repo | Core ML | Core AI | Recipe |
|---|---|---|---|
| flexible-unet-mlx | Verified | Verified | `recipes/flexible_unet` |
| flexray-mlx: all five members | Verified | Verified | `recipes/flexray` |
| nnunet-mlx: TotalSegmentator fast | Verified | Verified | `recipes/totalsegmentator` |
| nnunet-mlx: liver | Verified | Verified | `recipes/nnunet_liver` |
| nnunet-mlx: BraTS, all five folds | Verified | Verified | `recipes/nnunet_brats` |
| nnunet-mlx: PANTHER pancreas fold_all | Verified | Verified | `recipes/nnunet_pancreas` |
| nnunet-mlx: AutoPET II, all five folds | Verified | Verified | `recipes/nnunet_autopet` |
| segresnet-mlx | Verified | Verified | `recipes/segresnet` |
| radimagenet-mlx: ResNet50 features | Verified | Verified | `recipes/radimagenet` |
| swin-unetr-mlx | Verified | Verified | `recipes/swin_unetr` |
| unest-mlx | Verified | Verified | `recipes/unest` |
| lung-nodule-mlx | Verified | Verified | `recipes/lung_nodule` |
| mlx-segment-ct | Verified | Verified | `recipes/vista3d` |
| medsam2-mlx | Verified | Verified | `recipes/medsam2` |
| sam3-mlx | Verified | Verified | `recipes/sam3` |
| synthseg-mlx: 1.0 logits at 64³ | Verified | Verified | `recipes/synthseg` |
| synthseg-mlx: 2.0, robust, parcellation, QC | Not exported | Not exported | 2.0, parcellation and QC weights local; exports pending |
| MLX-Reason-CT | Vision probe passed | Vision probe passed | Language/cache/generation pending |
| nv-generate-mlx | Numerical gate failed | Numerical gate failed | CT denoiser probe; see survey |
| lesiondiffusion-mlx | Four historical component probes passed | Four historical component probes passed | Current probe skips VQ encoder; text/DDIM/full-size checks pending |
| biomedparse-mlx | Not exported | Not exported | Data-dependent decoder export blocked |
| med-vlm-mlx: MedGemma vision + projector | Numerical gate failed | Component probe passed | Language/cache/generation pending |
| med-vlm-mlx: Lingshu | Not exported | Not exported | Original weights downloaded; exports pending |

**Verified** means both native runtimes matched the declared recipe tolerance on
the recipe's fixed synthetic inputs. It applies only to the named variant and component. A failed or missing receipt
is not a passing export. Radiomics and the FDG-PET simulator have no neural
checkpoint to convert.

Manifest validation is not configured. The full recipe gate uses FP32
reference exports; FP16 architecture, full-volume and placement qualifications
remain separate.
The Python 3.13 migration gate passed all 85 tests and 17 recipes in 45 minutes
16 seconds, including all 35 checkpoint/component builds in both native formats.
Lint and strict Pyright passed; the original FP32 tolerances and converter pins
were retained. Lung-nodule's Core ML error was 6.58e-5 against its 1e-2 limit. An
earlier numerical failure of that recipe is recorded in
[docs/validation](https://github.com/MedMLX/medmlx/tree/main/conversion/docs/validation); it did not recur in the fresh exports.

A 2026-10-16 run of the guidance checks passed all 37 tests across 17 recipes,
producing 35 checkpoint/component guidance reports and 70 native artifacts.
Every automated guidance verdict passed; the lowest peak-normalized PSNR was
94.6 dB. A subsequent set of 26 non-recipe tests passed after additional fractional
interpolation and runtime-guidance admission cases were added.
Core ML deprecation warnings remain (six in the full run, 26 in the expanded
focused run). These runs cover the maintained recipes, not the pending catalog
entries above.

The 2026-10-13 closeout validation passed all 24 tests across 17 recipes.
Earlier publication checks on the same day reported Lung Nodule Core AI error
0.764 against 0.01 and SAM3 Core ML mask/semantic errors 19.6/1.22 against 0.1/0.01.
The current recipe comparisons pass, but the cause of the earlier failures remains
unverified. Both historical failures and current passing receipts are retained in
[docs/validation](https://github.com/MedMLX/medmlx/tree/main/conversion/docs/validation).
Additional component probes and remaining failures are listed in the survey. Separate
[ANEForge experiments](https://github.com/MedMLX/medmlx/blob/main/conversion/docs/aneforge-probes.md) do not change these FP32 verdicts.

## Adding a recipe

A recipe is a script, `recipes/<model>/export.py`, that builds the upstream model and
calls `export_and_check` from `src/medmlx_convert`. That function captures graphs with
`torch.export`, converts to both formats, reloads both saved assets, compares each output to PyTorch,
and returns whether all numerical checks pass. Start from
`recipes/flexible_unet/export.py` (2D) or `recipes/totalsegmentator/export.py` (3D). Then:

1. Pin the upstream package, source commit, and checkpoint revision.
   Declare the consumed checkpoint/configuration paths and ensemble members in the recipe.
2. Use the input shape the MLX port runs at.
3. Pass `COREAI_3D` and `COREML_3D` for 3D models; add model-specific rewrites as
   `ModuleSwap`s, and check each primitive against upstream. Capture wrapper
   references before changing the source; use `check_source_outputs` to retain
   those comparisons. The shared owner additionally checks the full prepared model,
   captured graph, decomposition result and any later manual graph-edit result
   separately against the source, before supplying the final graph to its converter.
4. Set a tolerance that fits the model's outputs.
5. Pass the checkpoint path, origin and configuration identifiers through
   `BuildSource.capture` to `export_and_check`. The numerical report retains
   the recipe, source revision and source declarations.
6. Update the table above and the model's entry in the survey.

Untyped upstream code goes through `medmlx_convert.boundary`, which checks each value it
returns. `make qualify` picks up the new recipe automatically; use
`make qualify PYTEST_ARGS='-k model_name'` for just the affected recipe.

## License

Original contributions are Apache-2.0. Adapted Apple export recipe portions retain
BSD-3-Clause terms; see [THIRD_PARTY_NOTICES.md](https://github.com/MedMLX/medmlx/blob/main/conversion/THIRD_PARTY_NOTICES.md).
Each model's source and weights carry their own terms; see the survey before
redistributing converted models.

## Implementation mapping

[Central implementation map](https://github.com/MedMLX/Med-MLX/blob/main/implementation-maps/medmlx.json) records upstream-to-native changes, source references and qualification scope.
