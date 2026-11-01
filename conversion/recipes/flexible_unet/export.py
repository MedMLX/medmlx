"""Export MONAI FlexibleUNet (endoscopic tools) to Core AI and Core ML, then check both."""

import sys
from pathlib import Path

import torch

from medmlx_convert import export_and_check, load_bundle_network, recipe_args
from medmlx_convert.provenance import BuildSource

options = recipe_args("flexible_unet")
out_dir = options.out
bundle = load_bundle_network(
    "MONAI/endoscopic_tool_segmentation", "f0416a27d59516d9c1b6bec5181c9c33002c84fa"
)
model = bundle.model
source = BuildSource.capture(Path(__file__), bundle.checkpoint,
                             origin=bundle.origin, configuration=bundle.configuration)
# flexible-unet-mlx runs one 736x480 RGB frame scaled to [0, 1].
image = torch.rand((1, 3, 736, 480), generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    model, {"image": image}, ["logits"], out_dir / "flexible_unet_endoscopic",
    source=source, tolerance=1e-3,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
