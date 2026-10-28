# Normalization overlap failure and verified tiled-reduction workaround

The converter now uses bounded reductions for GroupNorm and input-statistics
InstanceNorm3d in both Apple formats. The full BraTS, pancreas, and lung detector
exports pass their original 0.01 tolerance. This avoids the observed failing execution path;
the underlying Apple kernel or runtime defect has not been identified.

## Original failure

On the M5 test machine (macOS 27.2, Core AI from the OS, coreai-core 1.0.0b3,
coreai-torch 0.4.3, coremltools 9.0 and PyTorch 2.13.0), BraTS SegResNet and
PANTHER pancreas produced intermittent errors on the saved synthetic FP32
fixtures. These are fixed-shape network checks, not clinical validation.

- The original BraTS Core AI asset matched PyTorch to 6.68e-6 on 20 consecutive
  calls in one isolated run. Two overlapping processes then produced errors
  above 2.0 on identical inputs. Overlapping calls in one session also failed.
- A GPU-preferred BraTS session also failed under overlapping workloads, so
  preferring the GPU is not a demonstrated fix. A later single call failed too;
  overlap increases the opportunity to observe the error but has not been shown
  to be necessary.
- Pancreas Core AI calls also alternated between an error of 2.65e-4 and errors
  above 1.5 during competing work. The source input array remained unchanged.
- A BraTS debug trace showed unchanged early activations and divergence at a
  fused convolution, group-normalization and ReLU boundary. In a separately
  converted probe using the released weights and full-size source input, the
  convolution stayed within 5.25e-6 while its following GroupNorm differed by
  0.120. A comparable pancreas probe kept its convolution within 9.54e-7 while
  InstanceNorm differed by 1.124. Separately expressed mean and variance
  reductions also yielded incorrect values during overlapping execution.
- The BraTS Core ML package also produced errors above 2.9 while Core AI stress
  was running. Five paired Core ML runs without that competing stress passed;
  this does not rule out intermittent Core ML failures in other conditions.
- Three BraTS Core AI CPU-only calls each differed by 7.04e-4 while a competing
  default-specialized process alternated between passing and failing results.
  This is a limited control, not a qualification of CPU-only execution.

These observations localize the error to, or before, normalization results in
the Apple execution stack. They do not identify the failing kernel or prove the
original failed export comparisons had the same cause: those runs retained only
summary metrics, not their actual Core AI output tensors. A debug-specialized
Core AI compute plan reported GPU placement, but that plan is not proof of the
placement of the original non-debug calls. No tolerance or default compute
policy has been changed to conceal these failures.

## Isolation and implementation

Using the same full-size BraTS activation and released GroupNorm weights,
24 calls per candidate ran while a separate original BraTS session competed:

| Normalization expression | Failed calls (>0.01) | Maximum error |
|---|---:|---:|
| Standard GroupNorm | 6/24 | 3.14025 |
| Explicit mean and centered variance | 5/24 | 3.55821 |
| Same formula with one flattened reduction axis | 2/24 | 4.13692 |
| Standard LayerNorm on each flattened group | 2/24 | 4.15440 |
| Tiled mean and centered variance | 0/24 | 9.54e-7 |

This rules out a simple replacement with another standard normalization or a
flatten-only rewrite. The tiled candidate's median paired-call duration was
1.20 seconds versus 1.17 seconds for standard GroupNorm in this competing
workload; these host-observed durations are not isolated kernel benchmarks.
Separate synthetic mean/variance-only probes passed, even with competing work.
The failure therefore remains sensitive to graph structure and execution;
it is not a claim that every large mean operation fails.

`TiledGroupNorm` and `TiledInstanceNorm3d` in `src/medmlx_convert/rewrites.py`
reduce at most 1,024 values per stage, including the partial statistics.
Zero-padded tails are corrected by their original element count. Variance
remains the population mean of squared centered values, with the original
epsilon and channel affine parameters. `precision.py` recognizes both types
so FP16 authoring retains FP32 normalization weights and computation.
Core AI still uses default specialization and Core ML uses ALL compute units.
No tolerance, source checkpoint, input fixture, or output interpretation changed.

## Full-model evidence

Both full-size models passed rewrite, capture, decomposition, and saved-runtime
comparisons against their original CPU FP32 models:

| Model | Core AI max logit error | Core ML max logit error | Argmax agreement, both |
|---|---:|---:|---:|
| BraTS SegResNet, 240×240×160 | 7.15256e-6 | 7.15256e-6 | 1.0 |
| PANTHER pancreas, 192³ | 2.10762e-4 | 2.10524e-4 | 1.0 |

Each model then ran in three competing processes: the original Core AI export,
the corrected Core AI export, and the corrected Core ML export. Each process
made 10 rounds of two overlapping calls in one session:

| Model | Original Core AI failures | Corrected Core AI failures | Corrected Core ML failures |
|---|---:|---:|---:|
| BraTS | 18/20 (maximum error 3.03653) | 0/20 | 0/20 |
| Pancreas | 3/20 (maximum error 2.39936) | 0/20 | 0/20 |

All 80 corrected calls retained the same maximum errors as the table above and
argmax agreement 1.0. Failed original tensors are retained under
`build/norm-investigation/{brats,pancreas}-peer/first-failure.npz`, with every
call's metrics in the corresponding `report.json`. The corrected reports are
in `{brats,pancreas}-fixed-{coreai,coreml}/report.json` beneath that directory.
These results qualify the tested fixtures and workload on this host; they do
not establish correctness across every input, device, or future OS version.

Focused regression checks cover unequal/padded reduction lengths (including
three stages), independent batches and groups, constant/near-constant input,
affine parameters, FP16 parameter preservation, and both native converters.
The three new checks passed; `make qa` passed Ruff and strict Pyright.
Other recipes sharing the 3D rewrite lists were not requalified in that initial run;
the subsequent lung detector qualification is recorded below.
The corrected assets, fixtures, and verification records are installed at the
usual `build/segresnet/brats_mri_segmentation` and
`build/nnunet_pancreas/nnunet_pancreas_fold_all` stems. Input and source-oracle
arrays were exactly equal to the original fixtures. Original exports are
preserved under `build/norm-original/segresnet/` and
`build/norm-original/nnunet_pancreas/` for further diagnosis.

The source postprocessing rules were also compared on both corrected formats:
independent sigmoid thresholds of 0.5 for BraTS, and multiclass argmax for
pancreas. Both formats changed zero of 27,648,000 BraTS binary decisions and
zero of 7,077,888 pancreas labels. These synthetic fixtures contain no predicted
BraTS tumor regions, 43 predicted pancreas voxels, and no predicted cancer.
This confirms agreement on these inputs, not unchanged lesion accuracy on real
scans. Results are in `build/norm-investigation/segmentation-decisions.json`.

## Lung detector follow-up

The M5 lung detector asset still contained the original GroupNorm lowering.
A fresh check reproduced a Core AI failure on the fifth timed call:
`box_regression_0` differed by 0.24204 against its original 0.01 tolerance.
The earlier passing export comparison therefore did not resolve the repeated-call
failure. The failed check is retained in
`build/status-investigation-20261027/lung-coreai.json`.

Re-exporting the existing `recipes/lung_nodule` recipe applies the shared tiled
GroupNorm rewrite to both Apple formats. The input and all six CPU reference
arrays are exactly equal to the previous fixture. Both formats pass every
rewrite, capture, decomposition, and saved-runtime comparison. The maximum
errors across the six detector heads are 6.19888e-5 for Core AI and 6.00815e-5
for Core ML, with the unchanged 0.01 tolerance.

Two Core AI GPU processes and one Core ML CPU_AND_GPU process then ran
simultaneously, each making one initial call, three warmups, and sixteen further
calls. Every call checked all six outputs:

| Corrected export | Passing calls | Maximum error |
|---|---:|---:|
| Core AI, process 1 | 20/20 | 6.19888e-5 |
| Core AI, process 2 | 20/20 | 6.19888e-5 |
| Core ML | 20/20 | 6.00815e-5 |

The per-call records are
`build/status-investigation-20261027/lung-tiled-coreai-{1,2}.json` and
`build/status-investigation-20261027/lung-tiled-coreml.json`. These concurrent
checks establish output agreement on the saved network fixture; the controlled
five-process timing campaign is separate. Detector decoding and NMS remain on
the host and were not changed.

The fix is committed as `cd69c0c`. A fresh export from that clean revision is
recorded in [the saved comparison](lung-nodule-20261028.verification.json).
It is installed at `build/lung_nodule/lung_nodule_retinanet`; the previous assets
are preserved in `build/norm-original/lung_nodule/`. Repeating the original
runtime check against the installed Core AI export passed all fourteen calls
(one initial call, three warmups, ten timed calls), with maximum error
6.19888e-5. That check is saved in
`build/status-investigation-20261027/lung-installed-coreai.json`.

## Upstream investigation

The failure also reproduces without MedMLX, medical checkpoints, or MLX imports.
The standalone model has two Conv3d/GroupNorm/ReLU blocks, seeded random FP32
weights, and input shape `[1, 4, 240, 240, 160]`. It was converted using
`apple/coreai-torch` main at `7a18c005deb3e791ac35f201a3d955b77034e947` with the
OS Core AI runtime. The host is an M5 Max with 128 GiB, macOS 27.2 (26B123),
Python 3.13.18, PyTorch 2.13.0, NumPy 2.4.6, and coreai-core 1.0.0b3.

| Synthetic workload | Failures above 0.01 | Maximum error |
|---|---:|---:|
| Core AI, two processes, initial harness | 21/40 | 1.45323 |
| Core AI, two calls in one function, alongside MLX | 6/20 | 0.381106 |
| MLX 0.32.4, alongside the preceding Core AI workload | 0/20 | 7.39098e-6 |
| Core AI CPU-only, two calls in one function | 0/6 | 3.48091e-5 |
| Core AI, two processes, standalone 63-line reproduction | 14/40 | 1.58126 |

Passing default Core AI calls differed by 6.19888e-6. The MLX control uses the
same weights and inputs, channels-last convolution weights, and GroupNorm with
`pytorch_compatible=True`; every call evaluates a new graph. A smaller `128³`
Core AI case passed 20 calls, so that shape is not a demonstrated reproduction.
These controls narrow the affected execution path without identifying its
internal cause or qualifying all MLX or CPU workloads.

Source inspection found mathematically consistent mean and centered-variance
lowerings in Core AI. MLX implements its normalization through separate Metal
kernels. No relevant existing fix was found in the searched upstream issues
and pull requests. The evidence supports reporting through `apple/coreai-torch`
for runtime triage; it does not justify an MLX code change.

The official MLX repository is `ml-explore/mlx`. Its
[agent rules](https://github.com/ml-explore/mlx/blob/main/AGENTS.md) prohibit
automated PR submission and writing PR descriptions; its issue template also
prohibits AI-written issues. Core AI's
[contribution rules](https://github.com/apple/coreai-torch/blob/main/CONTRIBUTING.md)
require a proposal and maintainer response before changes beyond a
straightforward bug fix. A general normalization workaround should be discussed
there before changing the upstream converter.

With user approval, the standalone reproduction and controls were submitted as
[apple/coreai-torch#140](https://github.com/apple/coreai-torch/issues/140).
The report includes the tested code, environment, and failure counts.
No upstream pull request was opened.

Local upstream checkouts and the reproduction, control scripts, logs, and
metrics are in `apple-contributions/`, beside the `MedMLX/` workspace.
`investigation/coreai_overlap_minimal.py` needs no downloaded models or data:
run `USE_OS_COREAI=1 python investigation/coreai_overlap_minimal.py` with the
tested Core AI environment. It saves the first failed output and exits nonzero
on error. Both upstream checkouts remain unmodified.

## Reproduction

To check the corrected assets using concurrent calls in one Core AI session:

```sh
USE_OS_COREAI=1 python scripts/reproduce_coreai_overlap.py \
  build/segresnet/brats_mri_segmentation --rounds 10 --concurrency 2
USE_OS_COREAI=1 python scripts/reproduce_coreai_overlap.py \
  build/nnunet_pancreas/nnunet_pancreas_fold_all --rounds 10 --concurrency 2
```

Run with the provisioned conversion Python. Add `--backend coreml` for the
Core ML control; its synchronous predictions run on separate threads.
`--output-dir build/<run-name>` selects the evidence directory; otherwise each
run creates a unique directory beneath the export directory's `overlap/`.
Use the corresponding `build/norm-original/` stem to exercise the old lowering
alongside a corrected model in a separate process.
Every round records error, argmax agreement, and call duration. The first
failing input/reference/output is retained in `first-failure.npz`. Nonfinite
outputs are saved and rejected. The command exits nonzero if any output exceeds
the original 0.01 absolute tolerance.

Probe scripts, raw logs, compiled diagnostic models, and stress reports from
this investigation are local ignored artifacts in `build/norm-investigation/`.
