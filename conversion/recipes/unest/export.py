"""Export MONAI UNesT (renal structures) to Core AI and Core ML, then check both."""

import sys
from importlib import import_module
from pathlib import Path

import torch

from medmlx_convert import export_and_check, load_bundle_network, recipe_args
from medmlx_convert.boundary import Module, as_tensor, attribute, module, module_class, tensor
from medmlx_convert.check import check_source_outputs
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D, ModuleSwap


class LowRankBlockAttention(Module):
    """NesT block attention with tensors of rank 4 or less; Core ML caps tensor rank at 5.

    Same math as the bundle's Attention, including its output channel order, which puts
    the head index last (channel = head_dim_index * heads + head).
    """

    def __init__(self, attention: torch.nn.Module) -> None:
        super().__init__()
        heads, scale = attribute(attention, "num_heads"), attribute(attention, "scale")
        if not isinstance(heads, int) or not isinstance(scale, float):
            raise TypeError("expected the bundle's Attention with int num_heads, float scale")
        self.heads, self.scale = heads, scale
        self.qkv, self.proj = module(attention, "qkv"), module(attention, "proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        qkv = as_tensor(self.qkv(x), "qkv").reshape(b * t, n, 3 * c).chunk(3, dim=-1)
        q, k, v = (
            part.reshape(b * t, n, self.heads, c // self.heads).permute(0, 2, 1, 3) for part in qkv
        )
        weights = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (weights @ v).permute(0, 2, 3, 1).reshape(b, t, n, c)
        return as_tensor(self.proj(out), "proj")


options = recipe_args("unest")
out_dir = options.out
bundle = load_bundle_network(
    "MONAI/renalStructures_UNEST_segmentation", "f15df38db78f38865c8d49cbc4631f7a7286ef3b"
)
model = bundle.model
source = BuildSource.capture(Path(__file__), bundle.checkpoint,
                             origin=bundle.origin, configuration=bundle.configuration)
# Importable once the bundle's scripts directory is on the path.
Attention = module_class(import_module("scripts.networks.nest_transformer_3D"), "Attention")

# Exercise the layout rewrite independently with every source attention's learned weights.
attention_inputs: dict[str, torch.Tensor] = {}
source_outputs: list[torch.Tensor] = []
rewritten_outputs: list[torch.Tensor] = []
with torch.no_grad():
    for attention in model.modules():
        if type(attention) is not Attention:
            continue
        name = f"attention_{len(attention_inputs)}"
        channels = tensor(module(attention, "qkv"), "weight").shape[1]
        features = torch.randn((1, 3, 5, channels), generator=torch.Generator().manual_seed(0))
        attention_inputs[name] = features
        source_outputs.append(as_tensor(attention(features)))
        rewritten_outputs.append(LowRankBlockAttention(attention).eval()(features))
if not attention_inputs:
    raise ValueError("The pinned UNesT model contains no source Attention modules")
check_source_outputs(
    out_dir / "unest_attention_source", attention_inputs, tuple(rewritten_outputs),
    tuple(source_outputs), list(attention_inputs), dict.fromkeys(attention_inputs, 1e-4),
    "source_low_rank_attention",
)

# One 96^3 sliding-window patch after the bundle's intensity scaling to [0, 1].
image = torch.rand((1, 1, 96, 96, 96), generator=torch.Generator().manual_seed(0))
passed = export_and_check(
    model,
    {"image": image},
    ["logits"],
    out_dir / "unest_renal",
    source=source, tolerance=1e-2,
    coreai_swaps=COREAI_3D,
    coreml_swaps=[*COREML_3D, ModuleSwap(Attention, LowRankBlockAttention)],
    profile=options.profile, source_device=options.source_device,
    coreml_profile=options.coreml_profile,
)
sys.exit(0 if passed else 1)
