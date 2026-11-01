"""Export MONAI SwinUNETR (BTCV) to Core AI and Core ML, then check both against PyTorch."""

import sys
from pathlib import Path

import torch

from medmlx_convert import export_and_check, load_bundle_network, recipe_args
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

options = recipe_args("swin_unetr")
out_dir = options.out
# MONAI 1.6 dropped `img_size`; it only validated input sizes and does not change the network.
bundle = load_bundle_network(
    "MONAI/swin_unetr_btcv_segmentation",
    "e3e994c2b1438f84f431162a6b319f2819500c65",
    drop_args=("img_size",),
)
model = bundle.model
source = BuildSource.capture(Path(__file__), bundle.checkpoint,
                             origin=bundle.origin, configuration=bundle.configuration)
# One 96^3 sliding-window patch after the bundle's intensity scaling to [0, 1].
image = torch.rand((1, 1, 96, 96, 96), generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    model,
    {"image": image},
    ["logits"],
    out_dir / "swin_unetr_btcv",
    source=source, tolerance=1e-2,
    coreai_swaps=COREAI_3D,
    coreml_swaps=COREML_3D,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
