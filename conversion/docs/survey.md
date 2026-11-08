# MedMLX export survey

Updated 2026-10-16 against the MedMLX model repositories and their local checkouts.
Recipes use coremltools 9.0 and coreai-torch 0.4.3 (see [toolchains.md](toolchains.md)).
The README lists executed coverage; manual recipe receipts in `build/<recipe>/`
are the measured per-output evidence. Historical QA runs are under
`build/qa/<run>/build/<recipe>/`; new test sessions remove their system-temp outputs
on exit. Counts of recipes, checkpoint folds and pipeline components
are different. A network export does not verify its host pipeline.

The shared exporter runs source, graph and native numerical checks for every
maintained recipe. Final `*.verification.json` reports retain the declared
FP32/FP16 precision and actual comparisons; device/application qualification is
separate. See [the check scope](toolchains.md#apple-guidance-checks).
Recipes declare checkpoint, configuration and source revisions at their loaders,
and select ensemble members explicitly. Historical exploratory passes do not
qualify a changed implementation; the LesionDiffusion VQ probe presently skips
its historically passing encoder.
An earlier 2026-10-16 full gate passed 64 tests in 55 minutes 5 seconds: all 35
checkpoint/component builds across 17 recipes passed in both native formats,
with 35 fresh guidance and source reports and 70 native artifacts. The gate also
covers input admission, graph-stage rejection, rewrite/native round trips and
every-call runtime qualification. It reported 106 Core ML deprecation warnings;
the generic manifest validator remains unconfigured.
That run predates the current compact numerical reports and explicit Core ML GPU policy. The
earlier full gate passed all 84 tests in 58 minutes 28 seconds, including all
35 FP32 checkpoint/component builds in both native formats. Lint and strict
Pyright passed; 151 Core ML deprecation warnings remain. FP16 network checks run
separately and remain pending until their current receipts are complete.
The Python 3.13 refresh now passes all 85 tests and 17 recipes in 45 minutes
16 seconds, with all 35 checkpoint/component builds passing both native formats.
Lint and strict Pyright passed with the original FP32 tolerances and converter
pins.

The mixed FP16 profile is implemented in every maintained recipe as
`--profile macos-fp16-network`, with explicit MPS or CPU source selection.
Source drift and saved native outputs require raw PSNR >50 dB versus CPU FP32;
each graph stage independently requires normalized >70 dB versus its FP16
eager oracle. Original absolute thresholds remain visible diagnostics for
FP16. Standard Torch normalization retains original FP32 weights/computation;
module LogSoftmax uses stable FP32 arithmetic. Both public floating inputs and
outputs stay FP32. Unsupported source operations, dtype contracts and numerical
failures remain failures.
Keras layers receive the explicit FP16 compute policy while retaining upstream
normalization arithmetic. CPU FP16 eager and graph execution use the same math
attention kernel; their independent CPU FP32 oracle remains unchanged.
Core ML defaults to `ALL`; `--coreml-profile coreml-gpu` explicitly selects
`CPU_AND_GPU` for conversion and saved-model execution. The FP16 Swin UNETR
`ALL` attempt crashed in ANECompiler, and prolonged ANE compilation interrupted
other attempts. Separate GPU-route results retain those earlier diagnostics;
they do not establish `ALL` or ANE qualification.

Nineteen cases are available locally as 47 manifest-admitted original
image/label files: AMOS22 CT, MSD brain MRI, paired autoPET PET/CT and
Montgomery radiographs. Their model/pipeline checks remain pending.

The nnunet conversion found two gaps that the op lists don't show. Core AI
rejects 3D transposed convolution, and the Core ML runtime rejects 3D
InstanceNorm. Both have exact rewrites in `recipes/totalsegmentator`, and the
verdicts below account for them.

- **Verified:** a recipe here exports it and matches PyTorch.
- **Not exported:** no passing pair of artifacts; this is not a claim that
  conversion is impossible.
- **Numerical gate failed:** artifacts execute, but their measured error exceeds
  the declared tolerance. The failure must be resolved before deployment.

Code outside the network (resampling, sliding windows, post-processing) is listed
under "Host" for each repo. An app needs it ported to Swift.

## Repos without a network (3)

medmlx (API, CLI, MCP server, shared MLX core and conversion tooling),
medmlx.org (website), mlxradiomics (radiomics features).

## Model coverage

### flexible-unet-mlx: Core ML verified, Core AI verified
- Recipe: `recipes/flexible_unet`. Max difference 5.3e-5 in both formats; no rewrites.
- MONAI bundle `endoscopic_tool_segmentation@f0416a27` (Apache-2.0 weights). The
  bundle also ships a TorchScript `models/model.ts`.
- 2D, fixed 3×736×480. EfficientNet-B2 encoder, 11.5M params.
- Host: bilinear resize in and out, min-max scaling, argmax.

### flexray-mlx: Core ML verified, Core AI verified
- Recipe: `recipes/flexray` exports all five released members: `flux000`,
  `flux025`, `flux0375`, `flux050`, `flux075`. Outputs are independent multilabel
  logits; argmax class agreement is not an appropriate measure for this output.
- `fxr.models.UNet`, HF `VictorButoi/flexray@6b63ddf0`. Weights are
  **CC-BY-NC-4.0**, so they can't ship in a commercial app.
- 2D 1×256×256 to 61 labels. About 101M params per member, 5-member ensemble.
- Host: pad to square, resize, percentile scaling, sigmoid, ensemble mean.

### nnunet-mlx: verified; PANTHER uses tiled normalization
- PANTHER's intermittent M5 normalization failures are avoided by tiled
  InstanceNorm reductions in both formats. Full-size maximum logit differences
  are 2.10762e-4 (Core AI) and 2.10524e-4 (Core ML), with argmax agreement 1.0.
  All 40 corrected overlapping calls passed while the original Core AI export
  failed alongside them. See [the investigation](validation/coreai-overlap-20261027.md).
- nnU-Net v2.8.1 PlainConvUNet and ResidualEncoderUNet (BasicBlockD). Checkpoints
  are user-supplied model folders (TotalSegmentator v2, MRSegmentator, and
  others), with per-model licenses.
- Recipe: `recipes/totalsegmentator` exports TotalSegmentator's 3 mm `--fast`
  model (Dataset297, 117 classes, Apache-2.0) at its 112×112×128 patch. Max
  logit difference against PyTorch is 1.1e-3 in both formats.
- Its default source cache is converter-owned because other ports can rewrite
  TotalSegmentator's shared plans. Explicit `TOTALSEG_WEIGHTS_PATH` and
  `TOTALSEG_HOME_DIR` overrides retain the same upstream-input admission checks.
- Core AI needs the transposed-conv rewrite; both formats use tiled InstanceNorm
  reductions. Other nnU-Net models follow the same recipe, with one export per
  plans, configuration and fold.
- Added `recipes/nnunet_liver` (Zenodo 18807203), `recipes/nnunet_brats`
  (11582627, all five folds), and `recipes/nnunet_pancreas` (17212158,
  `fold_all`). Each uses its original planned patch and input channel count.
  Max differences were 1.54e-4 (liver), 1.72e-4 (BraTS), and 3.18e-4 (pancreas).
  All measured argmax agreements were 1.0.
- `recipes/nnunet_autopet` covers all five AutoPET II folds from Zenodo 8362371.
  Both runtimes passed: maximum error 1.27e-4, all argmax agreements 1.0.
- PANTHER's original `nnUNetTrainer1e3` inherits the ordinary inference builder
  and changes training learning rate. Its pinned source is
  `MIC-DKFZ/panther@b4da2ff2afe6edffd9fb6be331f32b306953d0cc`, file
  `nnUNetTrainer_finetuning.py`. The recipe explicitly resolves that one trainer
  to the inherited builder, rejects other names, and leaves checkpoint bytes
  unchanged. The checkpoint's trainer metadata is not rewritten because
  an explicit inference-only lookup preserves source identity.
- Host: crop to nonzero, normalization, resampling, Gaussian sliding window,
  mirroring (up to 8 passes in 3D), fold averaging, resample back.

### segresnet-mlx: renal verified; BraTS verified with tiled normalization
- Tiled GroupNorm reductions avoid the intermittent M5 failure in both Apple
  formats. Full-size maximum logit difference is 7.15256e-6 in each format,
  with argmax agreement 1.0. All 40 corrected overlapping calls passed while
  the original Core AI export failed alongside them. See
  [the investigation](validation/coreai-overlap-20261027.md).
- Recipe: `recipes/segresnet`, both bundles. Max difference 8e-4 (renal) and 8e-6
  (BraTS, through the trilinear rewrite). Core AI uses the transposed-conv rewrite
  and both formats use tiled GroupNorm reductions.
- MONAI bundles `renalStructures_CECT_segmentation` (transposed-conv decoder,
  19M params) and `brats_mri_segmentation` (trilinear decoder, 4.7M params).
- BraTS needs its 3D trilinear upsample rewritten before export. The renal
  decoder's 2×2×2 transposed conv needs the Core AI rewrite.
- The tiled GroupNorm rewrite also keeps intermediate rank within Core ML's limit.
- Host: MONAI orientation, spacing and inverse transforms. Sliding window of 96³
  (renal) or 240×240×160 (BraTS).

### radimagenet-mlx: Core ML verified, Core AI verified
- Original `RadImageNet-ResNet50_notop.h5`, Keras 3.15 Torch CPU backend.
  Independent TensorFlow 2.20 / Keras 3.15 CPU outputs gate the bridge at 1e-3.
  The observed bridge error was at most 2.98e-6. Apple runtime errors were at
  most 4.41e-6 (Core AI) and 2.98e-4 (Core ML).
- Recipe: `recipes/radimagenet`; NHWC 1×224×224×3 to pooled and 7×7×2048 features.
- A custom PyTorch reimplementation was rejected in favor of maintained Keras
  backend support. Core ML needs the exact `torch.subtract` alias decomposition.

### swin-unetr-mlx: Core ML verified, Core AI verified
- Recipe: `recipes/swin_unetr`. Max difference 5.3e-5. `roll` converted in Core AI
  without a rewrite. MONAI 1.6 dropped the bundle's `img_size` argument, which only
  validated sizes, so the recipe removes it.
- MONAI 1.4 SwinUNETR, HF `MONAI/swin_unetr_btcv_segmentation@e3e994c2`. Fixed
  1×1×96³, 62M params.
- coreai-torch doesn't list `roll`, which the shifted windows use. The UNETR
  decoder's 2×2×2 transposed convs need the Core AI rewrite.
- The decoder's InstanceNorm3d needs the Core ML rewrite.
- Host: resample to 1.5×1.5×2 mm, sliding window 96³ with overlap 0.5.

### unest-mlx: Core ML verified, Core AI verified
- Recipe: `recipes/unest`. Max difference 4e-5. Core AI uses the transposed-conv and
  max-pool rewrites. Core ML uses the norm rewrites plus a rank-4 version of the
  block attention, which upstream writes in rank 6.
- HF `MONAI/renalStructures_UNEST_segmentation` 0.2.7. Fixed 96³, 87M params.
- coreai-torch doesn't list `max_pool3d` or `prelu`, and the decoder's
  transposed convs need the Core AI rewrite.
- The decoder's InstanceNorm3d needs the Core ML rewrite.
- Host: resample to 1 mm, sliding window 96³ with overlap 0.5.

### lung-nodule-mlx: verified with tiled normalization

- Re-exporting with tiled GroupNorm passes all six heads on the M5 Max, with
  maximum differences 6.19888e-5 (Core AI) and 6.00815e-5 (Core ML) against
  the unchanged 0.01 limit. All 60 calls across three competing processes
  passed. The previous export failed a fresh Core AI check with a
  `box_regression_0` error of 0.24204. See
  [the investigation](validation/coreai-overlap-20261027.md#lung-detector-follow-up)
  and [export comparison](validation/lung-nodule-20261028.verification.json).
- The historical 2026-10-13 closeout rerun passed all six outputs in both formats, with
  maximum error 6.58e-5. Receipt:
  `docs/validation/lung-nodule-20261013-closeout.verification.json`.
  The earlier Core AI `classification_0` error of 0.764 against 0.01 remains
  recorded in `docs/validation/lung-nodule-20261013.verification.json`;
  its cause is unverified.
- Recipe: `recipes/lung_nodule` exports the six head maps at a 256×256×128 window.
  Core AI uses the max-pool rewrite; Core ML decomposes the FPN's functional
  nearest upsample. Both formats now use tiled GroupNorm reductions.
- MONAI `lung_nodule_ct_detection` 0.6.10. 3D RetinaNet, 20.9M params.
- The FPN's 3D nearest upsample needs a rewrite for both converters. Core AI
  also lacks `max_pool3d`. The upstream sliding window is 512×512×192; the
  exported fixed input is the smaller 256×256×128 window stated above.
- Host: windowing, anchors, box decoding, top-k, 3D NMS.

### mlx-segment-ct: Core ML verified, Core AI verified
- Recipe: `recipes/vista3d` exports the class-prompted path (image encoder and class
  head) for 117 classes at a 128³ patch. Max difference 7e-4. The checkpoint stores the
  network under a `network.` prefix; the recipe strips it.
- VISTA-3D (MONAI 1.6.0, NVIDIA `models/model.pt`, NVIDIA Open Model License).
  About 196M params, 128³ windows.
- The output channel count depends on how many classes are prompted (up to 117),
  so export with a fixed count. The point-correction decoder is a separate
  model.
- Its transposed conv (kernel 3, stride 2, output padding 1) overlaps, so both
  formats use the zero-insertion rewrite; Core ML also rejects its output padding.
- Host: resample to 1.5 mm, crop foreground, invert, sliding window, connected
  components.

### medsam2-mlx: Core ML verified, Core AI verified
- Recipe: `recipes/medsam2` exports the image encoder, prompt + mask heads
  (box prompt), memory encoder, and memory attention, plus a combined prompted-slice
  encoder and mask-head graph for network timing. Max difference ≤1e-4 for the four
  component exports.
- Memory attention runs at a fixed 7 memories + 16 pointers with a validity mask;
  padded + masked matches upstream on the real slots to 3e-6. RoPE is rewritten in
  real arithmetic, point embedding and multimask selection without data-dependent
  indexing, and Hiera's position embedding is precomputed.
- MedSAM2 `MedSAM2_CTLesion.pt`, SAM 2.1 Hiera-T at 512.
- The image path (encoder, prompt encoder, mask decoder) follows Apple's SAM 2.1
  Core ML split. Memory attention takes a variable number of memories, so export
  it with fixed slots and a mask.
- Host: frame loop, per-object memory bank, hole filling.

### sam3-mlx: Core ML and Core AI verified

- The 2026-10-13 closeout receipt reports numerical agreement for five outputs,
  but does not record a checkpoint loading report. With transformers 5.10.1,
  loading this video checkpoint directly as an image model leaves detector weights
  missing. The prior receipts cannot establish a fully loaded source model.
- The 2026-10-27 text-only Transformers rerun loaded all 1,797 tensors from the pinned video checkpoint,
  then extracted the detector. All five outputs passed for both runtimes.
  Maximum mask-logit errors were 0.04086 (Core AI) and 0.01643 (Core ML);
  all other outputs were within 0.00061. The [verification receipt](validation/sam3-20261027.verification.json)
  records the converter revision and environment. Prompt-padding and fixed-grid
  source checks also passed.
- The current recipe includes the original image model’s empty-geometry CLS token,
  which the Transformers text-only call omits. It checks this path against the
  Transformers geometry encoder with a fully masked box. The 2026-10-27
  [aligned export](validation/sam3-geometry-aligned-20261027.verification.json)
  passed on both runtimes, with maximum mask-logit errors of 0.03560 (Core AI)
  and 0.03335 (Core ML); all other outputs were within 0.00021. The
  [empty-geometry source check](validation/sam3-empty-geometry-20261027.verification.json),
  prompt-padding and fixed-grid checks passed as well.
- Recipe: `recipes/sam3`, following apple/coreai-models' SAM 3 `--full` export: the
  image detector of the Hugging Face `Sam3VideoModel` from gated
  `facebook/sam3@3c879f39` at 1008×1008, with a 32-token text prompt and
  attention mask. It rejects any missing, unexpected, mismatched or failed
  checkpoint loads before converting the detector.
- The DETR decoder reads its feature grid from a tensor, which torch.export treats as
  data-dependent; the recipe records the 72×72 grid and passes it as integers.
- Apple's lite bundle (336 px, int4/int6 palettized, for iOS) is produced by Apple's own
  `uv run models/sam3/export.py` in apple/coreai-models; it isn't checked here.
- SAM License weights; downloading them needs `hf auth login` with access to `facebook/sam3`.

### synthseg-mlx: 1.0 verified; 2.0 and auxiliary networks not exported
- `recipes/synthseg` exports original 1.0 weights and 32-channel logits at 64³.
  Its graph comes from upstream `BBillot/SynthSeg@2a2aa3bb`, unchanged learned
  widths, with the final softmax removed to expose logits. An independent
  TF-Keras 2.20.1 CPU reference checks the Keras Torch bridge at 16³.
- Core AI rejected ELU and 3D max pooling. It now uses Torch's own ELU
  decomposition and exact separable 2D max pooling. Core ML rejected the
  rank-six repeat-interleave intermediate; nearest 3D interpolation preserves
  the upsampling values. Both runtime errors were 3.91e-5 at tolerance 1e-3.
- The 2026-10-13 source-fetch run received HTTP 403 from the registry's
  SharePoint URL and HTTP 404 from the upstream README's folder URL. The original
  2.0, parcellation and QC H5 files are now available locally.
  Export and independent source-reference checks remain pending; no 2.0,
  robust, parcellation or QC export is claimed. References and regeneration
  instructions are in `tests/fixtures/README.md`.
- Host: normalization, resampling, flips, smoothing, topology cleanup and QC.

### MLX-Reason-CT: vision component probe passed
- NV-Reason-CT: a 3D ViT (13,824 tokens) feeding a Qwen3.5 hybrid decoder (Gated
  DeltaNet). 4.35B params, 17.4 GB in FP32.
- The released checkpoint's original `Vision3D` at 192³ passed both runtimes,
  max error 2.03e-6 at tolerance 1e-3.
- Prefill, stateful language decoding and the generation loop remain unexported.
  This probe is additional to the maintained recipe suite; it does not qualify
  the complete 4.35B model.

### nv-generate-mlx: initial CT denoiser numerical gate failed
- MAISI (NV-Generate-CTMR): five networks, 476M params, 3D attention,
  30-step rflow and 1000-step DDPM loops, trilinear upsampling.
- Original pinned NV-Generate CT/MR/MR-Brain checkpoints were downloaded.
  The CT denoiser at a 16³ latent input ran in both Apple runtimes, but failed
  the initial 1e-3 max-absolute gate: 5.32e-3 Core AI and 3.24e-3 Core ML.
  Reference was upstream MONAI CPU FP32, flash attention explicitly disabled.
  Full denoising-loop error,
  other networks and production spatial shapes remain unqualified.
- An explicit CPU-only diagnostic also failed: Core AI 3.06e-3 and Core ML
  3.24e-3. Changing device selection did not resolve the numerical gate.

### lesiondiffusion-mlx: four component probes passed
- Two 3D latent-diffusion U-Nets, a VQ autoencoder and a BiomedCLIP/BERT text
  encoder, with a 10–200-step DDIM loop. Weight license not established.
- Released EMA LMNet and LINet weights passed both runtimes at 16³ test inputs:
  max errors 2.62e-6 and 1.37e-4. VQ encoding of a 32³ input and decoding its
  latent also passed, at 9.54e-6 and 4.41e-6. All gates were 1e-3, FP32.
- The VQ decoder needed exact `argmin` via `min(...).indices` for Core AI and
  dtype-preserving scalar scatter for Core ML. These probes are additional to the
  maintained recipe suite. Text encoding, production spatial sizes and the DDIM loop
  remain unqualified.

### biomedparse-mlx: export blocked in the decoder
- Original released checkpoint loads into pinned upstream `f571d5cc`.
  `torch.export` fails first on prompt-count Python min/max handling. Fixing
  the boundary to one prompt then reaches a data-dependent NaN assertion in
  `boltzformer_cls_decoder.py:306`. The decoder also uses Boltzmann random
  attention masks, which require an explicit replay/random-input contract for
  meaningful parity. Neither Apple artifact is qualified.

### med-vlm-mlx: MedGemma vision Core AI passed; Core ML numerical gate failed
- Original MedGemma 1.5 4B weights were downloaded from
  `google/medgemma-1.5-4b-it@91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
  Its full 896-square Siglip vision encoder and projector ran in both formats:
  Core AI max error 1.24e-5 passed the 1e-3 gate; Core ML max error 2.43e-2 failed.
  This is outside the maintained recipe suite.
- MedGemma still needs language prefill, explicit cache-state/decode exports
  and generation checks. Existing MLX quantized checkpoints were not used as
  upstream FP32 export sources.

## Notes that apply to every model

- A separate experimental [native MLX pilot](native-conversion.html) now
  converts the upstream Torch graph and tensor state for nnU-Net BraTS
  fold 0. The BraTS run uses a 64-cube diagnostic patch, not its
  planned image pipeline. SynthMorph has independently checked dense-warp and
  velocity-integration adapters only; its learned affine/joint networks remain
  unconverted. These results do not expand the Core AI/Core ML status table.

- The MLX ports' custom FP32 Metal kernels (medmlx-core `Float32Operators`,
  nnunet's torch-order mode) exist to follow PyTorch's accumulation order as
  closely as possible. Exported models are compared within a tolerance instead.
- Core ML `ALL` permits multiple compute units; device profiling identifies
  which units execute. Current comparisons use fixed synthetic inputs.
