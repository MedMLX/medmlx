# MedMLX

One Python API, command line and MCP server for native medical imaging on Apple Silicon.
Run TotalSegmentator, nnU-Net, MedSAM2, SynthSeg and more through their independent MLX runtimes.

[Models](#models) · [Setup guide](https://github.com/MedMLX/medmlx/blob/main/docs/setup.md) · [MCP guide](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) · [Website](https://medmlx.org)

![Outputs across CT, PET/CT, radiographs, endoscopy and MRI](https://raw.githubusercontent.com/MedMLX/medmlx/main/docs/gallery.jpg)

**Apple Silicon · Metal · Python 3.13** · [uv](https://docs.astral.sh/uv/)

## Install

```sh
uv tool install --python 3.13 "medmlx[all,mcp]"
```

For a Python project, use `uv add "medmlx[all,mcp]"`. Replace `all` with
`segment`, `generate` or `vision` for a smaller install, and run
`uv tool upgrade medmlx` to update MedMLX and its models. Model weights download
on first use and keep their own terms.

## Quickstart

List the installed models, then show a model's arguments:

```sh
medmlx models
medmlx describe mlx-segment-ct
```

Put a CT volume in a working folder and write the run's arguments next to it.
Relative paths resolve against `--root`:

```sh
mkdir -p ~/medmlx-work && cp /path/to/ct.nii.gz ~/medmlx-work/
cat > ~/medmlx-work/arguments.json <<'EOF'
{"image_path": "ct.nii.gz", "output_dir": "segment-ct"}
EOF
medmlx run mlx-segment-ct --arguments ~/medmlx-work/arguments.json --root ~/medmlx-work
```

The first run downloads the `MedMLX/vista3d-ct-mlx` weights; later runs reuse the
cache. The mask is written to `segment-ct/mlx_segment_ct_mask.nii.gz`. Use
`medmlx readiness` with the same arguments to check Metal and inputs without
downloading weights, and `medmlx setup --all --progress` to pre-download weights
for offline use. See [automatic weights and readiness](https://github.com/MedMLX/medmlx/blob/main/docs/setup.md).

## Models

Model IDs are what `medmlx run` and `medmlx describe` take.

| Model ID | Task | Input | Extra |
| :--- | :--- | :--- | :--- |
| `totalsegmentator` | Whole-body CT segmentation | CT | segment |
| `mlx-segment-ct`, `mlx-segment-ctmr` | Automatic and point-prompted segmentation (VISTA-3D) | CT, MRI | segment |
| `nnunet-liver`, `nnunet-pancreas`, `nnunet-autopet`, `nnunet-brats2021` | Organ and tumor segmentation (nnU-Net v2) | CT, PET/CT, brain MRI | segment |
| `mrsegmentator` | Multi-organ MRI segmentation | MRI | segment |
| `synthseg` | Brain segmentation and parcellation | Brain MRI, CT | segment |
| `medsam2` | Box-, point- or mask-prompted masks with propagation | NIfTI slices or volumes, frames | segment |
| `sam3` | Text-, box- or point-prompted image segmentation | 2D images, video frames | segment |
| `biomedparse-v2` | Text-prompted segmentation | 3D volumes, 2D images | segment |
| `swin-unetr` | 13 abdominal organs | Contrast CT | segment |
| `segresnet-brats`, `segresnet-renal-cect` | Brain tumor and renal structures | Brain MRI, multiphase CT | segment |
| `unest` | Kidney cortex, medulla and collecting system | CT | segment |
| `flexray` | Radiograph anatomy | Radiographs | segment |
| `flexible-unet` | Surgical tools | Endoscopy frames | segment |
| `lung-nodule` | Lung nodule detection | Chest CT | vision |
| `radimagenet-*` | Radiology image features | 2D images | vision |
| `mlxradiomics` | Radiomics features and voxel maps | Image and mask | vision |
| `mlx-reason-ct` | CT reports and question answering | CT | generate |
| `reason-ct-tools` | Reason-CT with nodule detection and MedSAM2 masks | Chest CT | reason-tools |
| `mlx-generate-ctmr-ct`, `-mr`, `-mr-brain` | Synthetic CT and MRI | Label maps or anatomy | generate |
| `lesiondiffusion` | Synthetic CT lesions | CT and organ mask | generate |
| `fdg-pet-ct` | Synthetic FDG PET (weight-free) | CT and labels | generate |
| `medgemma` | Image-to-text through a separate environment | 2D images | [bridge](https://github.com/MedMLX/medmlx/blob/main/docs/medvlm-bridge.md) |

Each model has its own repository in the [MedMLX organization](https://github.com/MedMLX).
`medgemma` and `mrsegmentator` require local weights.

### Memory

MedMLX runs one model at a time by default and refuses a run whose peak memory
exceeds the memory available, so a 16 GB or 32 GB Mac cannot run the largest
generators. Declared peaks:

| Model | Peak MLX memory |
| :--- | ---: |
| `mlx-generate-ctmr-ct` | 30 GB |
| `mlx-generate-ctmr-mr-brain` | 26 GB |
| `mlx-reason-ct` | 23 GB |
| `sam3` | 12 GB |
| `lesiondiffusion` | 3.5 GB |

Models without a declared peak record their measured peak after each run, and
`medmlx describe` reports it.

## MCP

`medmlx serve` runs a stdio MCP server that lets an assistant list models, check
inputs and run jobs in a working folder. For Claude Code:

```sh
mkdir -p ~/medmlx-work
claude mcp add --scope user medmlx -- "$(uv tool dir --bin)/medmlx" serve --root "$HOME/medmlx-work"
```

For clients configured with JSON, use absolute paths:

```json
{
  "mcpServers": {
    "medmlx": {
      "command": "/absolute/path/to/medmlx",
      "args": ["serve", "--root", "/absolute/path/to/medmlx-work"]
    }
  }
}
```

Disconnecting the client cancels its running jobs. The
[MCP guide](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) covers Codex,
other clients, tools and a complete job workflow.

## Python

```python
import medmlx

result = medmlx.run(
    "mlx-segment-ct",
    arguments={"image_path": "ct.nii.gz", "output_dir": "segment-ct"},
    root="/path/to/medmlx-work",
)
```

## More

- [Reason-CT with nodule tools](https://github.com/MedMLX/medmlx/blob/main/docs/reason-ct-tools.md):
  model-selected detector and MedSAM2 calls, spatially grounded masks and a local
  CT review page.
- [MedGemma bridge](https://github.com/MedMLX/medmlx/blob/main/docs/medvlm-bridge.md):
  a provisional bridge to a separate MedVLM environment. MedMLX has not yet
  qualified its checkpoint inference or numerical accuracy.
- This README describes the 0.4.6 release on [PyPI](https://pypi.org/project/medmlx/).
  Unreleased changes are in the [release notes](https://github.com/MedMLX/medmlx/blob/main/docs/release-notes.md);
  to use them, install from the locked source workspace described in the
  [MCP guide](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md).
- Development: [declaring models](https://github.com/MedMLX/medmlx/blob/main/docs/declaring-models.md),
  [checkpoint conversion](https://github.com/MedMLX/medmlx/blob/main/conversion/README.md)
  and the [shared core runtime](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md).

## License

[Apache-2.0](https://github.com/MedMLX/medmlx/blob/main/LICENSE). See [third-party notices](https://github.com/MedMLX/medmlx/blob/main/THIRD_PARTY_NOTICES.md).
Runtimes and model weights retain their own terms.

For research and education. Outputs require human review.
