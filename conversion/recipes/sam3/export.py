"""Export SAM 3 text-prompted detection (Apple's full recipe) to Core AI and Core ML, then check.

Follows apple/coreai-models' SAM 3 `--full` export: the Hugging Face Sam3Model at 1008x1008,
returning masks, boxes, logits, presence and semantic segmentation. The text prompt is padded
to a fixed length with an attention mask, so one export serves any prompt that fits.
"""

import sys
from pathlib import Path
from types import MethodType

import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    Module,
    as_module,
    as_sequence,
    as_tensor,
    attribute,
    class_,
    function,
    hf_snapshot,
    item,
    library,
    module,
)
from medmlx_convert.check import check_source_outputs
from medmlx_convert.provenance import BuildSource

# Gated: accept the SAM License on Hugging Face and log in (`hf auth login`) first.
REPO = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
IMAGE_SIZE = 1008
TEXT_TOKENS = 32  # apple/coreai-models' default static text length
PROMPT = "liver"
OUTPUTS = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg"]


class Sam3Full(Module):
    """Apple's _Sam3FullModule, plus the text attention mask as an input."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        return tuple(as_tensor(attribute(outputs, name), name) for name in OUTPUTS)


def tokenize(tokenizer: object, padding: bool | str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = function(tokenizer, "__call__")(
        [PROMPT], return_tensors="pt", padding=padding, max_length=TEXT_TOKENS, truncation=True
    )
    input_ids = as_tensor(item(encoded, "input_ids"), "input_ids").to(torch.int32)
    return input_ids, as_tensor(item(encoded, "attention_mask"), "attention_mask").to(torch.int32)



options = recipe_args("sam3")
out_dir = options.out
transformers = library("transformers")
snapshot = hf_snapshot(REPO, REVISION, ["model.safetensors", "*.json", "*.txt"])
model = as_module(
    function(class_(transformers, "Sam3Model"), "from_pretrained")(
        str(snapshot), local_files_only=True
    ),
    "Sam3Model",
)
processor = function(class_(transformers, "Sam3Processor"), "from_pretrained")(
    str(snapshot), local_files_only=True
)
tokenizer = attribute(processor, "tokenizer")
wrapped = Sam3Full(model).eval()

# The DETR decoder reads the feature grid size out of a tensor, which torch.export treats as
# data-dependent. For a fixed image size the grid is constant: record it on an eager pass,
# then hand the decoder's position-bias function plain integers.
decoder = module(model, "detr_decoder")
rpb_matrix = function(decoder, "_get_rpb_matrix")
grids: set[tuple[int, int]] = set()


def recording_rpb(boxes: torch.Tensor, spatial_shape: object) -> torch.Tensor:
    height, width = as_sequence(spatial_shape, "spatial shape")
    grid = (int(as_tensor(height, "height")), int(as_tensor(width, "width")))
    grids.add(grid)
    return as_tensor(rpb_matrix(boxes, grid), "position bias")


# A random image after SAM 3's normalization, as Apple's export uses, and one text prompt.
pixel_values = torch.randn(
    (1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=torch.Generator().manual_seed(0)
)
input_ids, attention_mask = tokenize(tokenizer, "max_length")

# The fixed-length, masked prompt must give the same result as the prompt alone.
short_ids, short_mask = tokenize(tokenizer, False)
with torch.no_grad():
    original = wrapped(pixel_values, input_ids, attention_mask)
    vars(decoder)["_get_rpb_matrix"] = recording_rpb
    padded = wrapped(pixel_values, input_ids, attention_mask)
    unpadded = wrapped(pixel_values, short_ids, short_mask)
inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
# These are source-transform checks, distinct from either native converter's error.
check_source_outputs(out_dir / "sam3_prompt_padding_source", inputs, padded, unpadded,
                     OUTPUTS, dict.fromkeys(OUTPUTS, 1e-2), "source_prompt_padding")
(feature_grid,) = grids


def static_rpb(owner: torch.nn.Module, boxes: torch.Tensor,
               spatial_shape: object) -> torch.Tensor:
    """Use the fixed grid with the copied decoder's own weights and device."""
    original_method = function(type(owner), "_get_rpb_matrix")
    return as_tensor(original_method(owner, boxes, feature_grid), "position bias")


# A bound method follows deepcopy; closing over the original CPU decoder does not.
vars(decoder)["_get_rpb_matrix"] = MethodType(static_rpb, decoder)
with torch.no_grad():
    static = wrapped(pixel_values, input_ids, attention_mask)
check_source_outputs(out_dir / "sam3_fixed_grid_source", inputs, static, original,
                     OUTPUTS, dict.fromkeys(OUTPUTS, 1e-3), "source_fixed_grid")

passed = export_and_check(
    wrapped,
    inputs,
    OUTPUTS,
    out_dir / "sam3_full_1008",
    # Mask logits reach ~100 after a 32-layer ViT-L; 0.1 is 1e-3 of that range. The exports
    # land at 2e-2 (Core AI) and 5e-2 (Core ML), ordinary FP32 drift at this depth.
    source=BuildSource.capture(
        Path(__file__), snapshot / "model.safetensors", origin=f"{REPO}@{REVISION}",
        configuration={"model": "config.json", "processor": "processor_config.json",
                       "prompt": PROMPT, "text_tokens": str(TEXT_TOKENS),
                       "image_size": str(IMAGE_SIZE)},
    ),
    tolerance={**dict.fromkeys(OUTPUTS, 1e-2), "pred_masks": 0.1},
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
