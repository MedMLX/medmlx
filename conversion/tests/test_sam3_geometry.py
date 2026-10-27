"""Compare the text export with the source's masked-geometry path."""

import torch
from transformers import Sam3Config, Sam3Model

from medmlx_convert.boundary import function
from medmlx_convert.sam3 import OUTPUTS, Sam3TextWithGeometry


def test_text_export_matches_a_fully_masked_box() -> None:
    function(torch, "manual_seed")(7)
    config = Sam3Config(
        vision_config={
            "backbone_config": {
                "image_size": 56, "patch_size": 14, "pretrain_image_size": 56,
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "window_size": 4, "global_attn_indexes": [1],
            },
            "fpn_hidden_size": 32, "backbone_feature_sizes": [[16, 16], [8, 8], [4, 4]],
        },
        text_config={
            "vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "max_position_embeddings": 32,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        geometry_encoder_config={
            "hidden_size": 32, "num_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 64, "roi_size": 3,
        },
        detr_encoder_config={
            "hidden_size": 32, "num_layers": 2, "num_attention_heads": 4, "intermediate_size": 64,
        },
        detr_decoder_config={
            "hidden_size": 32, "num_layers": 2, "num_queries": 5,
            "num_attention_heads": 4, "intermediate_size": 64,
        },
        mask_decoder_config={"hidden_size": 32, "num_attention_heads": 4},
    )
    detector = Sam3Model(config).eval()
    wrapped = Sam3TextWithGeometry(detector).eval()
    image = torch.randn((1, 3, 56, 56))
    ids = torch.tensor([[1, 3, 2, *([0] * 29)]])
    mask = torch.tensor([[1, 1, 1, *([0] * 29)]])
    with torch.no_grad():
        actual = wrapped(image, ids, mask)
        expected = detector(
            pixel_values=image, input_ids=ids, attention_mask=mask,
            input_boxes=torch.zeros((1, 1, 4)),
            input_boxes_labels=torch.full((1, 1), -10),
        )
    for name, value in zip(OUTPUTS, actual, strict=True):
        torch.testing.assert_close(value, getattr(expected, name), rtol=1e-4, atol=1e-5)
