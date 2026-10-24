# Core AI and Core ML toolchains

Pinned in this repo: coreai-torch 0.4.3 (with coreai-core 1.0.0b3), coremltools
9.0, torch 2.13.0. Checked on macOS 27.0.1 on 2026-10-12. The op-coverage row
below that names torch 2.8 has not been re-checked against 2.13 here.

Both converters start from a PyTorch model. Neither reads MLX, so every recipe
loads the upstream checkpoint that the MLX port was converted from.
Keras checkpoints use Keras 3.15's Torch backend on CPU, independently compared
with frozen TensorFlow outputs. The exporter never consumes MLX weights.

## Core AI (`.aimodel`)

Core AI ships with macOS and iOS 27 (`CoreAI.framework`). Models are converted
in Python and loaded in Swift with `AIModel(contentsOf:)`, which specializes the
asset for the device on first load and caches the result.

```python
import asyncio
import coreai_torch
import torch
from coreai.runtime import AIModel, NDArray

exported = torch.export.export(model, (example,))
program = (
    coreai_torch.TorchConverter()
    .add_exported_program(
        exported.run_decompositions(coreai_torch.get_decomp_table()),  # required
        input_names=["image"],
        output_names=["logits"],
    )
    .to_coreai()
)
program.save_asset(Path("model.aimodel"))  # a Path, not a str


async def run(path, x):
    function = (await AIModel.load(path)).load_function("main")
    return (await function({"image": NDArray(x)}))["logits"].numpy()
```

Things to know in 0.4.3:

- `AIProgram` has no `optimize()`, unlike the WWDC26 sample code. Optimization
  passes run inside `to_coreai()`.
- `save_asset` needs a `pathlib.Path`.
- `AIModel.load` and calling an `InferenceFunction` are both coroutines.

Other features:

- **Dynamic shapes:** pass `dynamic_shapes=` with `torch.export.Dim` at export.
  Xcode shows a dynamic dimension as `?`.
- **State (KV caches):** buffers made with `register_buffer` become states via
  the converter's `state_names`.
- **Missing ops:** `TorchConverter.register_torch_lowering` adds or overrides an
  op lowering. `register_custom_kernels` embeds Metal 4 source in the asset.
- **Compression:** `coreai-opt` (Python below 3.14) does config-driven int4/int8
  quantization and palettization. Apple compressed SAM 3 from 3 GB to 430 MB.
- **Inspection:** the Core AI Debugger app inspects a model and compares its
  outputs across compute units.

## Core ML (`.mlpackage`)

```python
import coremltools as ct

mlmodel = ct.convert(
    exported.run_decompositions({}),
    inputs=[ct.TensorType(name="image", shape=SHAPE)],
    outputs=[ct.TensorType(name="logits")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,  # the default is FP16
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)
mlmodel.save("model.mlpackage")
loaded = ct.models.MLModel("model.mlpackage", compute_units=ct.ComputeUnit.ALL)
loaded.predict({"image": x})  # macOS only; checks the saved package
```

`ct.RangeDim` and `ct.EnumeratedShapes` give flexible input shapes. coremltools
also converts TensorFlow/Keras, which matters for radimagenet and synthseg.

## Precision and compute units

The default reference profile requests FP32 and declares per-output tolerances.
The explicit mixed FP16 profile has separate admission criteria below. Runtime selection is
explicit: Core ML defaults to `ALL`, Core AI uses default specialization.
`--coreml-profile coreml-gpu` selects `CPU_AND_GPU` for both conversion and saved
package loading; the final numerical receipt records that choice. These settings permit
the framework to schedule work; a passing receipt does not establish which
compute unit executed each operation. ANE profiling and deployment qualification
are separate work. Changing precision requires a new numerical check.

The FP16 Swin UNETR `ALL` attempt crashed inside Apple's ANECompiler. SegResNet
was sampled in ANE validation after more than 30 minutes; its attempt, AutoPET
and SynthSeg were deliberately interrupted. Their partial assets and logs are
retained. Remaining network checks use a new, explicitly selected `CPU_AND_GPU`
attempt. Automatic retry under another compute policy was rejected because it
would hide an unqualified route. A GPU-route pass does not qualify `ALL` or ANE.
The sanitized crash frames and interruption record are linked in the audit.

## Op coverage for MedMLX networks

"Converted" means a recipe here ran it through the converter and runtime. The
other cells come from the coremltools 9.0 PyTorch frontend source and the
coreai-torch 0.4.3 supported-op list.

| PyTorch op | Core ML (coremltools 9.0) | Core AI (coreai-torch 0.4.3) | MedMLX users |
|---|---|---|---|
| Conv3d | converted | converted | all 3D models |
| ConvTranspose3d | converted | **fails**: "Transposed conv3d is not yet supported"; rewrite | nnunet, swin-unetr, unest, segresnet, vista3d |
| MaxPool3d | converted | **fails**: no `max_pool3d_with_indices`; rewrite | unest, lung-nodule, synthseg |
| AvgPool3d | yes | yes | nnunet |
| InstanceNorm3d | **fails at load**: runtime takes rank 3 or 4; rewrite | converted | nnunet, swin-unetr, unest |
| GroupNorm on 3D | **fails**: lowered through a rank-6 reshape; rewrite | converted | segresnet, lung-nodule |
| 3D trilinear upsample (module) | **no**; rewrite | not listed; rewritten too | segresnet (BraTS) |
| 3D nearest upsample (functional) | **no**; PyTorch decomposition to index ops | converted (decomposed by its table) | lung-nodule |
| Any tensor above rank 5 | **fails**, except reshape-transpose-reshape chains it folds (Swin windows convert) | converted | unest attention |
| 2D bilinear / nearest upsample | converted | converted | flexray, flexible-unet |
| `roll` | converted | converted, though not listed | swin-unetr |
| `roi_align` | **no** | not listed | sam3 |
| `squeeze(dim)` on a non-singleton axis | converted (no-op) | **fails**: lowered to shrink_dims; graph pass | vista3d |
| 2D bicubic upsample | **no** in 9.0 | converted | medsam2 (Hiera position embedding) |
| ConvTranspose3d with `output_padding` | **fails**; zero-insertion rewrite | (rewritten anyway) | vista3d |
| Boolean-mask writes (`x[mask] = ...`) | converted | **fails**: data-dependent shape; `torch.where` | medsam2 prompt encoder |
| Indexing with two index tensors | converted | **fails**: lowered to the wrong rank; `torch.gather` | medsam2 multimask selection |
| `torch.where` with a scalar branch | **fails**; decomposed to a 0-d tensor | converted | medsam2 |
| Non-contiguous graph inputs | **fails**; inputs made contiguous | converted | medsam2 |
| `new_ones`, `new_full`, tensor `&` / `\|` | **fails**; decomposed | converted | sam3 (transformers) |
| Shapes read from tensor values | data-dependent in torch 2.8 export; pass constants | same | sam3 DETR grid |
| ELU | yes | not listed | synthseg |
| Scaled dot-product attention | yes | yes (kept as a composite) | transformers |
| complex tensors | limited | `view_as_complex`, `polar` | sam3 RoPE, medsam2 memory attention |
| top-k, nonzero, scatter | yes | yes | detection post-processing |

## Exact rewrites

All in `src/medmlx_convert/rewrites.py`. Each is a module swap applied to a copy
of the model before export for one format, and each matches PyTorch's own op to
float rounding (checked on random tensors). Swaps match the exact class, so a
subclass with its own `forward` is left alone.

| Rewrite | Replaces | For | How |
|---|---|---|---|
| `StridedTransposedConv3d` | ConvTranspose3d, kernel == stride, no padding | Core AI | matmul over channels, reshape into output blocks |
| `ZeroInsertTransposedConv3d` | any other ConvTranspose3d; and for Core ML, any with `output_padding` | both | insert stride - 1 zeros (through rank-3 views), pad, Conv3d with the flipped kernel |
| `SeparableMaxPool3d` | MaxPool3d | Core AI | 2D max pool over (H, W), then over D |
| `FlatInstanceNorm3d` | InstanceNorm3d | Core ML | InstanceNorm1d over flattened spatial dims |
| `FlatGroupNorm` | GroupNorm | Core ML | the same GroupNorm on flattened spatial dims |
| `SeparableTrilinearUpsample` | trilinear `nn.Upsample` | both | bilinear over (H, W), then over D |

The trilinear rewrite preserves `scale_factor`, `align_corners` and
`recompute_scale_factor` in both passes, including fractional scales and
downsampling. Replacing scale factors with rounded output sizes changes the
sampling coordinates when scale recomputation is disabled.
Core ML tools 9 rejects nonintegral output sizes with both corner alignment
and scale recomputation disabled. That configuration fails conversion and does
not receive a passing numerical build result. Enabling recomputation implicitly is rejected
because it would change the source model's coordinates.

`COREAI_3D` and `COREML_3D` bundle these per format. Before every Core AI
conversion, `export_and_check` also drops non-singleton axes from
`squeeze(dims)`, which PyTorch ignores but coreai-torch 0.4.3 fails on. Functional 3D nearest
upsampling can't be swapped as a module, so the Core ML export runs PyTorch's
own decomposition for it. Model-specific rewrites live in their recipe: UNesT's attention
in rank 4 for Core ML, and MedSAM2's real-arithmetic RoPE, `torch.where` point
embedding, `torch.gather` multimask selection, and precomputed Hiera position
embedding. Each recipe checks the rewritten model against upstream PyTorch.

## Apple guidance checks

`make qualify` explicitly runs every recipe and both native runtimes; select
one with `PYTEST_ARGS='-k model_name'`. `make qa` runs static checks only. The shared exporter
enforces numerical checks at their source, graph and native-output boundaries;
`*.verification.json` retains the actual comparisons and explicit build precision.
The thresholds follow [Apple's output verification guidance](https://github.com/apple/coreai-models/blob/bd61eb182a02182b3bd550521e7130c5ef2bbee4/skills/skills/working-with-coreai/SKILL.md#verifying-outputs-psnr)
and [graph verification gates](https://github.com/apple/coreai-models/blob/bd61eb182a02182b3bd550521e7130c5ef2bbee4/skills/skills/model-authoring/SKILL.md#verification-gates).
They are implemented directly in `guidance.py`, without a separate policy catalog
or build-decision sidecar.

Before capture, each backend's copied/module-rewritten eager model must match
the original reference within the recipe's absolute tolerance and either be
exact or exceed 70 dB peak-normalized PSNR. Saved native assets are then reloaded
and compared separately. SAM3, MedSAM2, VISTA3D and UNesT also retain bounded
source-wrapper or learned-weight primitive checks. Common 3D rewrite properties
exercise convolution, normalization, pooling and interpolation. The same source
outputs also gate replay after capture, after decomposition, and after Core AI's
manual graph-edit pass. Core ML has no manual graph-edit pass in this path. Each
stage retains per-output comparisons and a textual graph snapshot, including a
graph whose comparison fails. The final replayed `ExportedProgram` is the object
supplied to the converter. Checking only the final graph was rejected because a
later transformation could conceal an earlier error. Each required stage executes
in the shared exporter before conversion can continue.
The shared Apple exporter calls `torch.export.export(..., strict=False)` with
fixed examples and no `dynamic_shapes` declaration. Replay verifies those
examples; it does not qualify unexecuted Python branches or other input shapes.
The separate experimental native MLX path uses strict capture and has its own
admission evidence.
Arithmetic, scalar selection, boolean
logic, and singleton/no-op squeeze cases have native round-trip properties.
Integer bitwise operations are explicitly rejected: Core AI lacks the scalar
overloads, and this Core ML path has no integer bitwise lowering. Treating them
as boolean logic would change the source meaning. These fixed cases do not
establish every custom branch.

Runtime measurement checks every completed first, warm-up and timed call under
the admitted precision profile and declared label requirements; Core AI's
separate profiling call is also checked. Comparisons execute between inference
timing intervals. Schema-4 run reports retain the build profile and each call's
phase, one-based iteration, timing and verdict.
Malformed/nonfinite outputs retain a failure message. The first failed call
stops the run, so a later correct output cannot overwrite it. Performance
comparisons must admit only passing runs and use the same checking protocol
across runtimes, since between-call validation changes the invocation cadence.

Recipes declare the pinned source/checkpoint revision and configuration paths
at the loaders that consume them. Ensemble recipes select their declared folds
or members explicitly at the declared trained-model folder. Archive directory
names are stable cache keys.

The final schema-3 `*.verification.json` includes the recipe, Git revision and
dirty state, consumed checkpoint path, declared origin/configuration and installed
package versions. Benchmarks require passing numerical comparisons, an explicit
precision profile and finite fixtures with the declared tensor names, shapes and
dtypes. Every completed runtime output is compared with the saved source oracle.
Older build reports require a fresh recipe run.

The TensorFlow reference recorder records its supplied checkpoint/source path,
source declaration, Git revision, dirty state and package versions in schema-3
fixtures. Generation remains CPU TensorFlow/Keras; the subsequent Torch bridge
and native exports retain independent numerical checks. Historical reference
tensors remain unchanged when their metadata inventory is simplified.

The FP32 reference profile checks evaluation mode, contiguous finite input
tensors, exact int64-to-int32 admission, supported graph capture, output
shape/dtype/finite values, every recipe tolerance and PSNR. PSNR uses Apple's
Core AI test definition (peak reference energy over mean squared error, with
the same regularization constants), computed in FP64 to avoid overflow. The raw
metric is retained; our gate first normalizes both signals by the reference
peak. This avoids a roughly 50 dB regularization ceiling on tiny outputs, observed
in MedSAM2 IoU scores despite errors near 1e-12. Exact outputs pass; nonexact
all-zero references require review because normalized PSNR is undefined. Below
60 dB requires investigation and fails the build; above 70 dB is the expected
FP32 range. PSNR supplements maximum error and never overrides it.

This checks a bounded interpretation of the cited guidance, not every Apple
recommendation. FP32 is a documented exception to the recommended FP16 deployment
default. Device memory, actual execution placement, compressed builds and full
application/pipeline behavior need separate evidence and remain untested here.
A passing fixed-shape network comparison does not establish device or full
application qualification.

All maintained recipes implement `--profile macos-fp16-network`. The independent
reference remains CPU FP32. Backend eager rewrites first pass the original
absolute limits and normalized PSNR >70 dB. Mixed precision is authored on a
copy: standard Torch normalization uses original unrounded FP32 weights and FP32
computation, module LogSoftmax uses FP32 `x-logsumexp`, and remaining floating
parameters/network inputs become FP16. Public floating I/O remains FP32.
Recipes can explicitly extend FP32 handling to exact custom normalization types;
MedSAM2 declares its upstream LayerNorm2d. Other custom normalization and
functional softmax are unchanged; unsupported behavior
fails explicitly. `--source-device mps` is the FP16 default; select `cpu`
explicitly for an upstream source incompatibility. Implicit MPS CPU fallback
is rejected.

Keras Torch graphs require a separate dtype-policy action after copying and
converting their tensor storage. Every graph layer receives Keras's FP16 compute
policy through its maintained layer traversal; Torch's module traversal exposes
the parameter dictionary rather than those graph layers. Keras normalization
keeps its upstream internal FP32 arithmetic with rounded FP16 weight storage.
Both Keras bridges still require their independent frozen TensorFlow comparison
before the separate mixed-precision and graph checks.

CPU FP16 attention explicitly uses Torch's math SDPA kernel for eager source,
capture, decomposition and graph replay. SAM2's Python kernel-selection context
does not survive capture; unrestricted replay selected CPU flash attention and
failed the graph budget. Holding the kernel choice fixed separates graph loss
from a source-kernel change. This context does not select the native runtime's
device or relax numerical acceptance.

For MPS FP16, both backends additionally use the checked transposed-convolution,
separable max-pool and trilinear rewrites. Average pooling factors through two
2D passes, preserving padding, ceil-mode counts and any fixed divisor; CPU
properties and a real MPS/Core AI/Core ML volume round trip cover that contract.
These replacements are admitted in FP32 before precision authoring.
Core ML FP16 module preparation also separates 2D convolution and transposed
convolution bias before precision authoring: Core ML Tools 9 promotes
input/weight types but omits bias promotion. This wrapper first passes FP32
source equivalence; its extra FP16 rounding boundary is then measured as source
precision loss, before capture. Performing that split only in decomposition
failed MedSAM2 memory's graph budget, so that failure and the corrected authoring
attempt are retained. Functional convolution still needs the equivalent
decomposition, whose independent graph check remains mandatory. The graph
threshold was not lowered to admit the earlier failed route.

Precision-authored source and saved native outputs require exact equality or
raw Apple PSNR strictly above 50 dB against CPU FP32, finite matching output
contracts, and any declared per-output label requirement. Nonexact all-zero
references fail. Captured/decomposed/edited graphs independently require exact
equality or normalized PSNR >70 dB against their own FP16 eager oracle. Every
comparison retains `absolute_verdict` and its original tolerance; those limits
remain mandatory for the FP32 reference and backend rewrites. Reusing FP32
absolute limits as the FP16 gate was rejected because it conflates intended
precision loss with compiler loss. Matching argmax alone was rejected because
labels can survive severe logit distortion. Neither PSNR nor label agreement establishes task accuracy.

Core ML receives an explicitly mixed graph with `compute_precision=FLOAT32`,
which preserves its authored types and prevents the converter adding a second
automatic downcast. This differs from supplying a FP32 graph with the converter's
FP16 transform, another route requiring separate source/native qualification.
Current schema-3 numerical reports record the profile; older build reports must be regenerated.
Each numerical report separates the profile verdict from its absolute-error
diagnostic. Its scope identifies a fixed-shape network comparison; deployment
qualification requires separate device/application evidence.

FleXray declares independent raw-logit-zero decisions agreement ≥0.999.
Other models have no declared task budget yet.

The Core ML guide recommends TorchScript tracing while also supporting
`torch.export`. This repository uses the latter to share a supported frontend
with Core AI and retain explicit shape constraints. Changing frontend is an
explicit model-specific choice and needs both runtime comparisons; it is not a
silent fallback after a failed build. Installed API signatures take precedence
over snippets from a newer guide. For example, the pinned Core AI 0.4.3 runtime
accepts an input dictionary, despite a contradictory keyword-argument example
in the downloaded common-issues guidance.

## Apple resources

- [apple/coreai-models](https://github.com/apple/coreai-models): export recipes
  for 32 models, including `sam3`, `sam3_video`, `efficient-sam`, `vlm`
  (Qwen3-VL) and `qwen3`. There are no 3D or medical models. It also ships a
  `coreai-skills` agent plugin with three skills: `working-with-coreai`,
  `model-authoring` and `model-compression-exploration`. Installation steps are in
  that repository's README.
- [coreai-torch documentation](https://apple.github.io/coreai-torch/main/), including
  the [supported ATen ops](https://apple.github.io/coreai-torch/main/api/supported-aten-ops.html).
- WWDC26 sessions [324](https://developer.apple.com/videos/play/wwdc2026/324)
  (Core AI overview) and [325](https://developer.apple.com/videos/play/wwdc2026/325)
  (conversion, custom kernels, compression).
- Apple's Core ML SAM 2.1 packages, for example
  [apple/coreml-sam2.1-tiny](https://huggingface.co/apple/coreml-sam2.1-tiny):
  image encoder, prompt encoder and mask decoder as separate FP16 models.
