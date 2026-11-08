# MedMLX conversion

Export upstream models to Apple's Core AI (`.aimodel`) and Core ML
(`.mlpackage`). Each recipe converts a pinned PyTorch model and compares the
saved exports with its source outputs.

[Model survey](docs/survey.md) · [Toolchains and numerical criteria](docs/toolchains.md) ·
[Runtime measurements](docs/runtime-tuning.html) · [Native MLX pilot](docs/native-conversion.html)

## Requirements

Apple Silicon, macOS 27 or later, Python 3.13, mamba or conda, and
[uv](https://docs.astral.sh/uv/). The locked environment uses Torch 2.13.

## Quick start

```bash
make env
bash environment/setup.sh --run python recipes/flexible_unet/export.py
```

Outputs go to `build/<model>/`. Checkpoints download to the Hugging Face or
Torch cache on first use. TotalSegmentator uses a separate source cache;
`TOTALSEG_WEIGHTS_PATH` and `TOTALSEG_HOME_DIR` can point to pristine upstream
files instead.

| Command | Purpose |
| --- | --- |
| `make qa` | Lint and type checks |
| `make test` | Converter contract tests; select cases with `TESTS` and `PYTEST_ARGS` |
| `make qualify` | Run all export recipes end to end |

Recipes write verification JSON and `*.runtime.npz` source fixtures alongside
the exports. These record source revisions, package versions, precision,
tolerances and output differences. The runtime tools use them to check outputs
before and during measurements.

## Precision and runtime

The default export profile is FP32. Each recipe also accepts
`--profile macos-fp16-network`; FP16 outputs default to `build/<model>-fp16/`.
Use `--source-device cpu` when source execution needs CPU. For Core ML GPU
execution, select `--coreml-profile coreml-gpu`; the default is `ALL`.

Prepared FP32 graphs must meet their original tolerances and normalized PSNR
above 70 dB; reloaded native exports require at least 60 dB. FP16 source drift
and native exports require Apple PSNR above 50 dB, with graph transformations
above 70 dB. Exact equality passes these checks. Model-specific decision checks
also apply. See [toolchains](docs/toolchains.md#apple-guidance-checks) for the
full criteria and runtime policies.

## Status

The table records FP32 comparisons on fixed synthetic inputs, as of
2026-10-16. Coverage is specific to each checkpoint, fold and component.
Full-volume pipelines and device placement need separate comparisons.

| Repo | Core ML | Core AI | Recipe |
|---|---|---|---|
| flexible-unet-mlx | Verified | Verified | `recipes/flexible_unet` |
| flexray-mlx: all five members | Verified | Verified | `recipes/flexray` |
| nnunet-mlx: TotalSegmentator fast | Verified | Verified | `recipes/totalsegmentator` |
| nnunet-mlx: liver | Verified | Verified | `recipes/nnunet_liver` |
| nnunet-mlx: BraTS, all five folds | Verified | Verified | `recipes/nnunet_brats` |
| nnunet-mlx: PANTHER pancreas fold_all | [Verified with tiled norms](docs/validation/coreai-overlap-20261027.md) | Verified with tiled norms | `recipes/nnunet_pancreas` |
| nnunet-mlx: AutoPET II, all five folds | Verified | Verified | `recipes/nnunet_autopet` |
| segresnet-mlx | [Verified; BraTS tiled norms](docs/validation/coreai-overlap-20261027.md) | Verified; BraTS tiled norms | `recipes/segresnet` |
| radimagenet-mlx: ResNet50 features | Verified | Verified | `recipes/radimagenet` |
| swin-unetr-mlx | Verified | Verified | `recipes/swin_unetr` |
| unest-mlx | Verified | Verified | `recipes/unest` |
| lung-nodule-mlx | [Verified with tiled norms](docs/validation/coreai-overlap-20261027.md#lung-detector-follow-up) | Verified with tiled norms | `recipes/lung_nodule` |
| mlx-segment-ct | Verified | Verified | `recipes/vista3d` |
| medsam2-mlx | Verified | Verified | `recipes/medsam2` |
| sam3-mlx | Verified | Verified | `recipes/sam3`; text geometry aligned |
| synthseg-mlx: 1.0 logits at 64³ | Verified | Verified | `recipes/synthseg` |
| synthseg-mlx: 2.0, robust, parcellation, QC | Not exported | Not exported | 2.0, parcellation and QC weights local; exports pending |
| MLX-Reason-CT | Vision probe passed | Vision probe passed | Language/cache/generation pending |
| mlx-generate | Numerical gate failed | Numerical gate failed | CT denoiser probe; see survey |
| lesiondiffusion-mlx | Four historical component probes passed | Four historical component probes passed | Current probe skips VQ encoder; text/DDIM/full-size checks pending |
| biomedparse-mlx | Not exported | Not exported | Data-dependent decoder export blocked |
| med-vlm-mlx: MedGemma vision + projector | Numerical gate failed | Component probe passed | Language/cache/generation pending |

**Verified** means both saved native exports met the recipe's tolerance.
The Python 3.13 migration passed 85 tests and 17 recipes, covering 35
checkpoint/component builds in both formats. Earlier Lung Nodule and SAM3
failures and subsequent passing runs are recorded in
[validation](docs/validation/). Their original cause remains unresolved.

The native MLX pilot uses graph schema 2 from `medmlx-core`; regenerate older
bundles with the current recipes. It includes a BraTS fold-0 recipe and
TensorFlow comparisons for SynthMorph geometry. Full SynthMorph and catalog-wide
conversion remain unfinished. Runtime loading and execution live in
`packages/core`; converters are separate from inference dependencies.

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

Untyped upstream code goes through `medmlx_convert.boundary`. Its typed helpers
check the kinds they document; raw attribute and item access is unvalidated. `make qualify` picks up the new recipe automatically; use
`make qualify PYTEST_ARGS='-k model_name'` for just the affected recipe.

## License

Original contributions: Apache-2.0. Adapted Apple export recipes retain
BSD-3-Clause terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
Model source and weights retain their own terms, listed in the survey.
