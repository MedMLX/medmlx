"""Export MedSAM2's image encoder, prompt + mask heads, memory encoder, and memory attention.

Each part is checked against PyTorch. Memory attention runs at a fixed size, with a validity
mask over the memory slots, so neither converter needs dynamic shapes.
"""

import math
import sys
from importlib import import_module
from pathlib import Path
from types import MethodType

import torch

from medmlx_convert import export_and_check, recipe_args
from medmlx_convert.boundary import (
    Module,
    as_float,
    as_int,
    as_int_pair,
    as_module,
    as_sequence,
    as_tensor,
    as_tensors,
    attribute,
    function,
    hf_download,
    item,
    module,
    module_class,
    tensor,
)
from medmlx_convert.check import check_source_outputs
from medmlx_convert.provenance import BuildSource
from medmlx_convert.sources import github_source

SOURCE_REPO = "bowang-lab/MedSAM2"
SOURCE_REVISION = "332f30d420f1d1b08e2a79b3ae6a602458808383"
WEIGHTS_REPO = "wanglab/MedSAM2"
WEIGHTS_REVISION = "e4a6f35edd7e091619cbc0750f462f1574e23955"
CHECKPOINT = "MedSAM2_CTLesion.pt"
CONFIG = "configs/sam2.1_hiera_t512.yaml"
IMAGE_SIZE = 512
# MedSAM2 keeps up to 7 slice memories of 32x32 tokens and 16 object pointers of 4 tokens each.
MEMORY_SLOTS = 7
POINTER_SLOTS = 16


def keep_source(path: str) -> str | None:
    """The sam2 and efficient_track_anything Python and YAML files; never the prebuilt _C.so."""
    wanted = path.startswith(("sam2/", "efficient_track_anything/"))
    return path if wanted and path.endswith((".py", ".yaml")) else None


class ImageEncoder(Module):
    """Hiera + FPN with the decoder's high-res projections, as SAM2Base.forward_image runs them."""

    def __init__(self, sam: torch.nn.Module) -> None:
        super().__init__()
        self.sam = sam
        # Hiera bicubic-resizes its learned position embedding to the patch grid on every
        # call. It doesn't depend on the image, and coremltools 9.0 has no bicubic upsample,
        # so compute it once here.
        trunk = module(module(sam, "image_encoder"), "trunk")
        with torch.no_grad():
            blank = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            patches = as_tensor(module(trunk, "patch_embed")(blank), "patch embedding")
            grid = (patches.shape[1], patches.shape[2])
            cached = as_tensor(function(trunk, "_get_pos_embed")(grid), "position embedding")

        # Buffers and bound methods follow deepcopy/.to/.half. A closure over
        # the original CPU tensor would survive a copied FP16/MPS model.
        trunk.register_buffer("medmlx_cached_position", cached, persistent=False)

        def cached_pos_embed(owner: torch.nn.Module, hw: object) -> torch.Tensor:
            if as_int_pair(hw, "patch grid") != grid:
                raise ValueError(f"position embedding cached for {grid}, got {hw}")
            return tensor(owner, "medmlx_cached_position")

        vars(trunk)["_get_pos_embed"] = MethodType(cached_pos_embed, trunk)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = function(self.sam, "forward_image")(image)
        feat_s0, feat_s1, features = as_tensors(item(out, "backbone_fpn"), "backbone_fpn")
        positions = as_tensors(item(out, "vision_pos_enc"), "vision_pos_enc")
        # The first prompted slice has no memory; SAM2 adds no_mem_embed instead.
        no_memory = features + tensor(self.sam, "no_mem_embed").reshape(1, -1, 1, 1)
        return feat_s0, feat_s1, features, no_memory, positions[-1]


def stable_position_encoding(owner: torch.nn.Module, coords: torch.Tensor) -> torch.Tensor:
    """Encode normalized xy coordinates in FP32, then restore the buffer dtype.

    Upstream forces coordinates to FP32; its frequency buffer follows .half(),
    producing an invalid matmul without this explicit arithmetic boundary.
    Frequency rounding remains part of the measured FP16 precision loss.
    """
    matrix = tensor(owner, "positional_encoding_gaussian_matrix")
    angles = ((2 * coords.float() - 1) @ matrix.float()) * (2 * math.pi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(matrix.dtype)


def typed_prompt_embeddings(
    owner: torch.nn.Module,
    points: tuple[torch.Tensor, torch.Tensor] | None,
    boxes: torch.Tensor | None,
    masks: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Restore learned embedding dtype after upstream's empty FP32 concatenation.

    PromptEncoder.forward creates its empty sparse tensor without a dtype, so
    concatenating FP16 point embeddings promotes the result to FP32. Restore
    the explicit network dtype before the decoder's FP16 linear layers.
    """
    outputs = as_tensors(function(type(owner), "forward")(owner, points, boxes, masks))
    dtype = tensor(module(owner, "not_a_point_embed"), "weight").dtype
    return tuple(value.to(dtype) for value in outputs)


class SamHeads(Module):
    """Prompt encoder and mask decoder for one box prompt (two corner points)."""

    def __init__(self, sam: torch.nn.Module) -> None:
        super().__init__()
        self.sam = sam
        decoder = module(sam, "sam_mask_decoder")
        def gather_stability(
            owner: torch.nn.Module, all_mask_logits: torch.Tensor, all_iou_scores: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """MaskDecoder._dynamic_multimask_via_stability, selecting with torch.gather.

            Upstream picks the best multimask output with two index tensors, which
            coreai-torch 0.4.3 lowers to the wrong rank. Same argmax, same selection.
            """
            multimask_logits = all_mask_logits[:, 1:]
            multimask_iou_scores = all_iou_scores[:, 1:]
            best = torch.argmax(multimask_iou_scores, dim=-1, keepdim=True)
            best_logits = torch.gather(
                multimask_logits,
                1,
                best[..., None, None].expand(-1, -1, *multimask_logits.shape[-2:]),
            )
            best_iou = torch.gather(multimask_iou_scores, 1, best)
            single_logits = all_mask_logits[:, 0:1]
            single_iou = all_iou_scores[:, 0:1]
            stability_scores = function(owner, "_get_stability_scores")
            threshold = as_float(attribute(owner, "dynamic_multimask_stability_thresh"))
            is_stable = as_tensor(stability_scores(single_logits), "stability") >= threshold
            return (
                torch.where(
                    is_stable[..., None, None].expand_as(single_logits), single_logits, best_logits
                ),
                torch.where(is_stable.expand_as(single_iou), single_iou, best_iou),
            )

        vars(decoder)["_dynamic_multimask_via_stability"] = MethodType(gather_stability, decoder)

        encoder = module(sam, "sam_prompt_encoder")
        vars(encoder)["forward"] = MethodType(typed_prompt_embeddings, encoder)
        positions = module(encoder, "pe_layer")
        vars(positions)["_pe_encoding"] = MethodType(stable_position_encoding, positions)

        def where_embed_points(
            owner: torch.nn.Module, points: torch.Tensor, labels: torch.Tensor, pad: bool
        ) -> torch.Tensor:
            """PromptEncoder._embed_points with torch.where instead of boolean-mask writes.

            Upstream writes through ``embedding[labels == k]``, a data-dependent shape that
            coreai-torch can't lower. The same per-label updates, as selects and masked adds.
            """
            points = points + 0.5
            if pad:
                points = torch.cat([points, torch.zeros_like(points[:, :1])], dim=1)
                labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], dim=1)
            forward_with_coords = function(module(owner, "pe_layer"), "forward_with_coords")
            embedding = as_tensor(forward_with_coords(points, attribute(owner, "input_image_size")),
                                  "point encoding")
            label = labels[..., None]
            embedding = torch.where(label == -1,
                                    tensor(module(owner, "not_a_point_embed"), "weight"), embedding)
            for k, table in enumerate(module(owner, "point_embeddings").children()):
                embedding = embedding + (label == k).to(embedding.dtype) * tensor(table, "weight")
            return embedding

        vars(encoder)["_embed_points"] = MethodType(where_embed_points, encoder)

    def forward(
        self,
        features: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        feat_s0: torch.Tensor,
        feat_s1: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = function(self.sam, "_forward_sam_heads")(
            features,
            point_inputs={"point_coords": point_coords, "point_labels": point_labels},
            high_res_features=[feat_s0, feat_s1],
            multimask_output=False,
        )
        _, _, ious, low_res, high_res, obj_ptr, object_score = as_tensors(outputs, "SAM heads")
        return low_res, high_res, ious, obj_ptr, object_score


class MemoryEncoder(Module):
    """SAM2Base._encode_new_memory on the sigmoid path.

    For a prompted slice, upstream binarizes the mask instead; pass the mask as +/-1e4 there,
    which the float32 sigmoid maps to exactly 1 and 0.
    """

    def __init__(self, sam: torch.nn.Module) -> None:
        super().__init__()
        self.sam = sam

    def forward(
        self, features: torch.Tensor, high_res_masks: torch.Tensor, object_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = features.shape[-2:]
        encoded = function(self.sam, "_encode_new_memory")(
            [features.flatten(2).permute(2, 0, 1)], [(h, w)], high_res_masks, object_score, False
        )
        memory, positions = as_sequence(encoded, "encoded memory")
        return as_tensor(memory, "memory"), as_tensors(positions, "memory positions")[-1]


class MaskedRealRoPEAttention(Module):
    """SAM2's RoPEAttention with real cos/sin rotation and an optional key mask.

    Upstream rotates with complex tensors, which Core ML can't run; rotating each (even, odd)
    pair by the same angle in real arithmetic is the same math. ``key_mask`` is an additive
    mask over keys for padded memory slots; with it unset this matches upstream exactly.
    Built through ``masked_real_rope`` so it also subclasses upstream's RoPEAttention, whose
    isinstance checks decide whether pointer tokens skip the rotation.
    """

    def __init__(self, attention: torch.nn.Module) -> None:
        super().__init__()
        self.q_proj, self.k_proj = module(attention, "q_proj"), module(attention, "k_proj")
        self.v_proj, self.out_proj = module(attention, "v_proj"), module(attention, "out_proj")
        self.separate_heads = function(attention, "_separate_heads")
        self.recombine_heads = function(attention, "_recombine_heads")
        self.heads = as_int(attribute(attention, "num_heads"), "num_heads")
        freqs = tensor(attention, "freqs_cis")
        self.register_buffer("cos", freqs.real.float().contiguous(), persistent=False)
        self.register_buffer("sin", freqs.imag.float().contiguous(), persistent=False)
        self.key_mask: torch.Tensor | None = None

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        return torch.stack([real * cos - imag * sin, real * sin + imag * cos], -1).flatten(-2)

    def _heads(self, projection: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return as_tensor(self.separate_heads(as_tensor(projection(x)), self.heads), "heads")

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0
    ) -> torch.Tensor:
        cos, sin = tensor(self, "cos"), tensor(self, "sin")
        q = self._heads(self.q_proj, q)
        k = self._heads(self.k_proj, k)
        v = self._heads(self.v_proj, v)
        if cos.shape[0] != q.shape[-2]:
            raise ValueError(f"queries must be {cos.shape[0]} tokens, got {q.shape[-2]}")
        q = self._rotate(q, cos, sin)
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        repeat = num_k_rope // q.shape[-2]
        rotated = self._rotate(k[:, :, :num_k_rope], cos.repeat(repeat, 1), sin.repeat(repeat, 1))
        k = torch.cat([rotated, k[:, :, num_k_rope:]], dim=-2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.key_mask)
        return as_tensor(self.out_proj(as_tensor(self.recombine_heads(out))), "attention output")


def masked_real_rope(attention: torch.nn.Module, rope_class: type) -> MaskedRealRoPEAttention:
    combined = type("MaskedRealRoPEAttention", (MaskedRealRoPEAttention, rope_class), {})
    instance = combined(attention)
    if not isinstance(instance, rope_class):
        raise TypeError("expected a MaskedRealRoPEAttention that is also a RoPEAttention")
    return instance


class MemoryAttention(Module):
    """Condition a slice's features on fixed memory slots, as SAM2 does for tracked slices.

    The host fills ``memory`` and ``memory_pos`` with up to MEMORY_SLOTS slice memories (with
    temporal encodings added, as upstream does) followed by POINTER_SLOTS object pointers,
    and marks real slots in ``memory_valid``.
    """

    def __init__(self, sam: torch.nn.Module, rope_class: type) -> None:
        super().__init__()
        self.attention = module(sam, "memory_attention")
        self.rope: list[MaskedRealRoPEAttention] = []
        for layer in module(self.attention, "layers").children():
            layer.self_attn = masked_real_rope(module(layer, "self_attn"), rope_class)
            cross = masked_real_rope(module(layer, "cross_attn_image"), rope_class)
            layer.cross_attn_image = cross
            self.rope.append(cross)

    def forward(
        self,
        features: torch.Tensor,
        vision_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        # A float32 softmax maps a -1e4 offset to exactly zero weight.
        mask = (memory_valid - 1.0).reshape(1, 1, 1, -1) * 1e4
        for attention in self.rope:
            attention.key_mask = mask
        out = self.attention(
            curr=[features.flatten(2).permute(2, 0, 1)],
            curr_pos=[vision_pos.flatten(2).permute(2, 0, 1)],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=POINTER_SLOTS * 4,
        )
        return as_tensor(out, "memory attention").permute(1, 2, 0).reshape(features.shape)


def pad(t: torch.Tensor, slots: int) -> torch.Tensor:
    """Append ``slots`` zero tokens along the first axis."""
    return torch.cat([t, torch.zeros_like(t)[:1].expand(slots, -1, -1)])



options = recipe_args("medsam2")
out_dir = options.out
source_root = github_source(SOURCE_REPO, SOURCE_REVISION, keep_source)
sys.path.insert(0, str(source_root))
build = function(import_module("sam2.build_sam"), "build_sam2_video_predictor_npz")
rope_class = module_class(import_module("sam2.modeling.sam.transformer"), "RoPEAttention")
# The memory encoder's custom channel normalization must retain FP32 statistics
# and original affine weights before the next FP16 convolution.
fp32_norm_types = (module_class(import_module("sam2.modeling.sam2_utils"), "LayerNorm2d"),)

checkpoint = hf_download(WEIGHTS_REPO, CHECKPOINT, WEIGHTS_REVISION)
source = BuildSource.capture(
    Path(__file__), checkpoint, origin=f"{WEIGHTS_REPO}@{WEIGHTS_REVISION}",
    configuration={"source": f"{SOURCE_REPO}@{SOURCE_REVISION}",
                   "config": CONFIG},
)
sam = as_module(build(CONFIG, str(checkpoint), device="cpu"), "MedSAM2").eval()

# One slice as medsam2-mlx feeds it: RGB in [0, 1], then ImageNet mean/std.
generator = torch.Generator().manual_seed(0)
image = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
image = (image - mean) / std

with torch.no_grad():
    original_image = function(sam, "forward_image")(image)
    original_fpn = as_tensors(item(original_image, "backbone_fpn"))
    original_positions = as_tensors(item(original_image, "vision_pos_enc"))
    original_features = original_fpn[-1]
    original_no_memory = original_features + tensor(sam, "no_mem_embed").reshape(1, -1, 1, 1)
encoder = ImageEncoder(sam).eval()
with torch.no_grad():
    encoded = encoder(image)
check_source_outputs(
    out_dir / "medsam2_image_encoder_source", {"image": image}, encoded,
    (*original_fpn, original_no_memory, original_positions[-1]),
    ["feat_s0", "feat_s1", "features", "features_no_memory", "vision_pos"],
    dict.fromkeys(["feat_s0", "feat_s1", "features", "features_no_memory", "vision_pos"], 1e-3),
    "source_position_cache",
)
passed = export_and_check(
    encoder,
    {"image": image},
    ["feat_s0", "feat_s1", "features", "features_no_memory", "vision_pos"],
    out_dir / "medsam2_image_encoder",
    source=source, tolerance=1e-2,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
    fp32_norm_types=fp32_norm_types,
)
with torch.no_grad():
    feat_s0, feat_s1, features, no_memory, vision_pos = encoder(image)

# A box prompt in model pixels: labels 2 and 3 mark the top-left and bottom-right corners.
box = torch.tensor([[[150.0, 150.0], [350.0, 350.0]]])
box_labels = torch.tensor([[2, 3]], dtype=torch.int32)
original_embed = function(module(sam, "sam_prompt_encoder"), "_embed_points")
original_select = function(module(sam, "sam_mask_decoder"), "_dynamic_multimask_via_stability")
points = torch.rand((1, 5, 2), generator=generator) * IMAGE_SIZE
labels = torch.tensor([[-1, 0, 1, 2, 3]], dtype=torch.int32)
mask_logits = torch.randn((2, 4, 16, 16), generator=generator)
# Exercise both stability branches and every supported point label before replacement.
mask_logits[0, 0] = 2.0
mask_logits[1, 0] = 0.0
iou_scores = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.1, 0.2, 0.4, 0.3]])
with torch.no_grad():
    point_reference = tuple(as_tensor(original_embed(points, labels, pad)) for pad in (False, True))
    selection_reference = as_tensors(original_select(mask_logits, iou_scores))
    head_reference = as_tensors(function(sam, "_forward_sam_heads")(
        no_memory, point_inputs={"point_coords": box, "point_labels": box_labels},
        high_res_features=[feat_s0, feat_s1], multimask_output=False,
    ))
heads = SamHeads(sam).eval()
with torch.no_grad():
    embed = function(module(sam, "sam_prompt_encoder"), "_embed_points")
    select = function(module(sam, "sam_mask_decoder"), "_dynamic_multimask_via_stability")
    point_actual = tuple(as_tensor(embed(points, labels, pad)) for pad in (False, True))
    selection_actual = as_tensors(select(mask_logits, iou_scores))
    head_actual = heads(no_memory, box, box_labels, feat_s0, feat_s1)
check_source_outputs(
    out_dir / "medsam2_point_embedding_source", {"points": points, "labels": labels},
    point_actual, point_reference, ["unpadded", "padded"],
    {"unpadded": 1e-5, "padded": 1e-5}, "source_point_embedding",
)
check_source_outputs(
    out_dir / "medsam2_stability_selection_source",
    {"masks": mask_logits, "ious": iou_scores}, selection_actual, selection_reference,
    ["masks", "ious"], {"masks": 0, "ious": 0}, "source_stability_selection",
)
head_names = ["low_res_masks", "high_res_masks", "ious", "obj_ptr", "object_score"]
check_source_outputs(
    out_dir / "medsam2_sam_heads_source", {"features": no_memory, "point_coords": box},
    head_actual, tuple(head_reference[index] for index in (3, 4, 2, 5, 6)),
    head_names, dict.fromkeys(head_names, 1e-3), "source_sam_heads",
)
passed &= export_and_check(
    heads,
    {
        "features": no_memory,
        "point_coords": box,
        "point_labels": box_labels,
        "feat_s0": feat_s0,
        "feat_s1": feat_s1,
    },
    ["low_res_masks", "high_res_masks", "ious", "obj_ptr", "object_score"],
    out_dir / "medsam2_sam_heads",
    source=source, tolerance=1e-2,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
    fp32_norm_types=fp32_norm_types,
)
with torch.no_grad():
    _, high_res, _, obj_ptr, object_score = heads(no_memory, box, box_labels, feat_s0, feat_s1)

memory_encoder = MemoryEncoder(sam)
passed &= export_and_check(
    memory_encoder,
    {"features": features, "high_res_masks": high_res, "object_score": object_score},
    ["memory", "memory_pos"],
    out_dir / "medsam2_memory_encoder",
    source=source, tolerance=1e-2,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
    fp32_norm_types=fp32_norm_types,
)

# Memory attention. Build the padded inputs from this slice's own memory and pointer, check
# in PyTorch that padding + mask matches upstream on the real slots only, then export.
with torch.no_grad():
    memory, memory_pos = memory_encoder(features, high_res, object_score)
spatial = memory.flatten(2).permute(2, 0, 1)  # (1024, 1, 64)
spatial_pos = memory_pos.flatten(2).permute(2, 0, 1)
pointer = obj_ptr.reshape(1, 4, 64).permute(1, 0, 2)  # 256-d pointer as 4 tokens of 64
used_memories, used_pointers = 2, 3
real_memory = torch.cat([spatial] * used_memories + [pointer] * used_pointers)
real_pos = torch.cat([spatial_pos] * used_memories + [torch.zeros_like(pointer)] * used_pointers)
memory_in = torch.cat([
    pad(torch.cat([spatial] * used_memories), (MEMORY_SLOTS - used_memories) * 1024),
    pad(torch.cat([pointer] * used_pointers), (POINTER_SLOTS - used_pointers) * 4),
])
pos_in = torch.cat([
    pad(torch.cat([spatial_pos] * used_memories), (MEMORY_SLOTS - used_memories) * 1024),
    torch.zeros(POINTER_SLOTS * 4, 1, 64),
])
valid = torch.cat([
    torch.ones(used_memories * 1024), torch.zeros((MEMORY_SLOTS - used_memories) * 1024),
    torch.ones(used_pointers * 4), torch.zeros((POINTER_SLOTS - used_pointers) * 4),
])
with torch.no_grad():
    upstream_out = module(sam, "memory_attention")(
        curr=[features.flatten(2).permute(2, 0, 1)],
        curr_pos=[vision_pos.flatten(2).permute(2, 0, 1)],
        memory=real_memory,
        memory_pos=real_pos,
        num_obj_ptr_tokens=used_pointers * 4,
    )
    upstream = as_tensor(upstream_out, "upstream memory attention")
    upstream = upstream.permute(1, 2, 0).reshape(features.shape)
    attention = MemoryAttention(sam, rope_class)
    padded = attention(features, vision_pos, memory_in, pos_in, valid)
check_source_outputs(
    out_dir / "medsam2_memory_attention_source", {"memory_valid": valid},
    (padded,), (upstream,), ["conditioned_features"], {"conditioned_features": 1e-4},
    "source_masked_memory_attention",
)
passed &= export_and_check(
    attention,
    {
        "features": features,
        "vision_pos": vision_pos,
        "memory": memory_in,
        "memory_pos": pos_in,
        "memory_valid": valid,
    },
    ["conditioned_features"],
    out_dir / "medsam2_memory_attention",
    source=source, tolerance=1e-2,
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
    fp32_norm_types=fp32_norm_types,
)
sys.exit(0 if passed else 1)
