"""Export both MONAI SegResNet bundles to Core AI and Core ML, then check against PyTorch."""

import sys
from pathlib import Path

import torch

from medmlx_convert import export_and_check, load_bundle_network, recipe_args
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

# (bundle, revision, sliding-window patch with input channels)
BUNDLES = [
    (
        "renalStructures_CECT_segmentation",
        "8ccbcd9573c004dc435d1913c0459ccd1a110ed1",
        (1, 3, 96, 96, 96),
    ),
    ("brats_mri_segmentation", "370f7f9d062745fbac445e7fe6d6616d35df04ec", (1, 4, 240, 240, 160)),
]

options = recipe_args("segresnet")
out_dir = options.out
passed = True
for bundle, revision, shape in BUNDLES:
    print(f"== {bundle}")
    image = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    loaded = load_bundle_network(f"MONAI/{bundle}", revision)
    passed &= export_and_check(
        loaded.model,
        {"image": image},
        ["logits"],
        out_dir / bundle,
        tolerance=1e-2, source=BuildSource.capture(
            Path(__file__), loaded.checkpoint, origin=loaded.origin,
            configuration=loaded.configuration,
        ),
        coreai_swaps=COREAI_3D,
        coreml_swaps=COREML_3D,
        profile=options.profile, source_device=options.source_device,
        coreml_profile=options.coreml_profile,
    )
sys.exit(0 if passed else 1)
