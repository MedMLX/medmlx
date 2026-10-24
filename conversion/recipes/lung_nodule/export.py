"""Export the MONAI 3D RetinaNet lung nodule network to Core AI and Core ML, then check both."""

import sys
from pathlib import Path

import torch

from medmlx_convert import export_and_check, load_bundle_network, recipe_args
from medmlx_convert.boundary import Module, as_tensors
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

LEVELS = 3
OUTPUTS = [f"classification_{i}" for i in range(LEVELS)] + [
    f"box_regression_{i}" for i in range(LEVELS)
]


class HeadOutputs(Module):
    """The network's per-level head maps; anchors, decoding and NMS stay on the host."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return as_tensors(self.model(image), "RetinaNet head outputs")


options = recipe_args("lung_nodule")
out_dir = options.out
# use_list_output returns the head maps as a flat list instead of a dict of lists.
bundle = load_bundle_network(
    "MONAI/lung_nodule_ct_detection",
    "9d6622fda0e52be0a4155fed2e0afb9c17bcd80b",
    set_args={"use_list_output": True},
)
model = bundle.model
source = BuildSource.capture(Path(__file__), bundle.checkpoint,
                             origin=bundle.origin, configuration=bundle.configuration)
# One window divisible by the bundle's (16, 16, 8), after HU scaling to [0, 1]. The bundle's
# own sliding window is 512x512x192; the heads are fully convolutional at any such size.
image = torch.rand((1, 1, 256, 256, 128), generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    HeadOutputs(model),
    {"image": image},
    OUTPUTS,
    out_dir / "lung_nodule_retinanet",
    source=source, tolerance=1e-2,
    coreai_swaps=COREAI_3D,
    coreml_swaps=COREML_3D,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
