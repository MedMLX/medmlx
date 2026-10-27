"""Load the image detector from the pinned SAM 3 video checkpoint."""
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from medmlx_convert.boundary import (
    Module,
    as_module,
    as_sequence,
    as_tensor,
    as_tensors,
    attribute,
    class_,
    function,
    item,
    library,
    module,
)

OUTPUTS = ("pred_masks", "pred_boxes", "pred_logits", "presence_logits", "semantic_seg")


def load_detector(snapshot: Path) -> torch.nn.Module:
    """Require a fully loaded video checkpoint and return its image detector.

    The official snapshot contains a video model with detector and tracker
    submodels. Loading it directly as `Sam3Model` leaves detector parameters
    uninitialized, so reject every incomplete loading report before export.
    """
    transformers = library("transformers")
    loaded = as_sequence(
        function(class_(transformers, "Sam3VideoModel"), "from_pretrained")(
            str(snapshot), local_files_only=True, output_loading_info=True,
        ),
        "SAM 3 video model and loading report",
    )
    if len(loaded) != 2:
        raise ValueError("Expected a SAM 3 video model and loading report")
    video, info = as_module(loaded[0], "Sam3VideoModel"), loaded[1]
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        entries = item(info, key)
        if not isinstance(entries, (list, set, tuple)) or entries:
            raise RuntimeError(f"SAM 3 checkpoint loading {key}: {entries}")
    return module(video, "detector_model")


class Sam3TextWithGeometry(Module):
    """Match the original SAM 3 text path's empty-geometry CLS token.

    Transformers skips geometry for a text-only call, whereas the original
    image model encodes a geometry CLS token even when there are no boxes.
    Reproduce that single token without ROI pooling; the recipe checks it
    against the Transformers path with a fully masked box before export.
    """

    def __init__(self, detector: torch.nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        vision = function(self.detector, "get_vision_features")(pixel_values=pixel_values)
        text = function(self.detector, "get_text_features")(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        feature = as_tensors(attribute(vision, "fpn_hidden_states"))[-2].flatten(2).transpose(1, 2)
        position = as_tensors(attribute(vision, "fpn_position_encoding"))[-2]
        position = position.flatten(2).transpose(1, 2)
        geometry = module(self.detector, "geometry_encoder")
        cls = as_tensor(attribute(module(geometry, "cls_embed"), "weight")).reshape(1, 1, -1)
        cls = cls.expand(pixel_values.shape[0], -1, -1)
        cls = as_tensor(module(geometry, "final_proj")(cls))
        cls = as_tensor(module(geometry, "prompt_layer_norm")(cls))
        for layer in module(geometry, "layers").children():
            cls = as_tensor(layer(prompt_feats=cls, vision_feats=feature,
                                  vision_pos_encoding=position, prompt_mask=None))
        cls = as_tensor(module(geometry, "output_layer_norm")(cls))
        text_features = as_tensor(attribute(text, "pooler_output"))
        prompt = BaseModelOutputWithPooling()
        prompt["pooler_output"] = torch.cat((text_features, cls), dim=1)
        mask = torch.cat((attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1)
        result = self.detector(vision_embeds=vision, text_embeds=prompt, attention_mask=mask)
        return tuple(as_tensor(attribute(result, name), name) for name in OUTPUTS)
