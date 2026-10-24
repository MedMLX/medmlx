"""Export VISTA-3D automatic segmentation (mlx-segment-ct) to Core AI and Core ML, then check."""

import sys
from pathlib import Path

import torch
from monai.networks.nets.vista3d import vista3d132

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    Module,
    as_sequence,
    as_tensor,
    as_tensors,
    hf_download,
    load_safetensors,
    module,
)
from medmlx_convert.check import check_source_outputs
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D

REPO = "nvidia/NV-Segment-CT"
REVISION = "afb51518689f71e6abb367ee6301b2cd0225c66a"
# mlx-segment-ct prompts up to 117 classes at once; the count is fixed at export.
NUM_CLASSES = 117


class AutomaticSegmentation(Module):
    """VISTA3D's class-prompted path: image encoder, then the class head.

    This is the branch VISTA3D.forward takes with ``class_vector`` and no point prompts.
    Point correction is a separate model.
    """

    def __init__(self, vista: torch.nn.Module) -> None:
        super().__init__()
        self.vista = vista

    def forward(self, image: torch.Tensor, class_vector: torch.Tensor) -> torch.Tensor:
        encoded = module(self.vista, "image_encoder")(image, with_point=False, with_label=True)
        # (point-branch features, class-branch features); only the class branch is used.
        features = as_tensor(as_sequence(encoded, "image encoder outputs")[1], "class features")
        logits = as_tensors(module(self.vista, "class_head")(features, class_vector))[0]
        return logits.transpose(1, 0)


options = recipe_args("vista3d")
out_dir = options.out
vista = vista3d132().eval()
weights = hf_download(REPO, "vista3d_pretrained_model/model.safetensors", REVISION)
# The checkpoint stores the network under a `network.` prefix.
state = load_safetensors(weights)
vista.load_state_dict({key.removeprefix("network."): value for key, value in state.items()})

# One 128^3 sliding-window patch after the bundle's HU scaling to [0, 1].
image = torch.rand((1, 1, 128, 128, 128), generator=torch.Generator().manual_seed(0))
class_vector = torch.arange(1, NUM_CLASSES + 1).reshape(-1, 1)
wrapper = AutomaticSegmentation(vista).eval()
with torch.no_grad():
    original = as_tensor(vista(image, class_vector=class_vector, transpose=True))
    automatic = wrapper(image, class_vector)
check_source_outputs(
    out_dir / "vista3d_automatic_source", {"image": image, "class_vector": class_vector},
    (automatic,), (original,), ["logits"], {"logits": 1e-4}, "source_automatic_branch",
)
passed = export_and_check(
    wrapper,
    {"image": image, "class_vector": class_vector},
    ["logits"],
    out_dir / "vista3d_auto",
    source=BuildSource.capture(Path(__file__), weights, origin=f"{REPO}@{REVISION}",
                               configuration={"num_classes": str(NUM_CLASSES)}), tolerance=1e-2,
    coreai_swaps=COREAI_3D,
    coreml_swaps=COREML_3D,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
