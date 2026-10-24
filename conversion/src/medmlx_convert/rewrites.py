"""Exact module swaps for ops a converter or runtime rejects. Each keeps the same math."""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from medmlx_convert.boundary import Module, as_int, interpolate_bilinear


class Swap(Protocol):
    def apply(self, module: torch.nn.Module) -> torch.nn.Module | None:
        """The replacement for ``module``, or None when this swap doesn't match it."""
        ...


@dataclass(frozen=True)
class ModuleSwap[M: torch.nn.Module]:
    """Replace modules of exactly ``module_type`` (not subclasses) with ``replacement(module)``."""

    module_type: type[M]
    replacement: Callable[[M], torch.nn.Module]

    def apply(self, module: torch.nn.Module) -> torch.nn.Module | None:
        if type(module) is self.module_type and isinstance(module, self.module_type):
            return self.replacement(module)
        return None


class SeparateBias2d(Module):
    """Author a 2D convolution and bias add before choosing network precision.

    Core ML Tools 9 can promote input/weight without bias. Splitting bias only
    after FP16 capture adds a rounding boundary absent from its eager oracle.
    This wrapper makes that boundary explicit in both FP32 rewrite checks and
    FP16 source-precision checks before graph capture. Default output size and
    all convolution parameters remain owned by the copied Torch module.
    """

    def __init__(self, source: torch.nn.Conv2d | torch.nn.ConvTranspose2d) -> None:
        super().__init__()
        self.convolution = copy.deepcopy(source)
        self.bias = (None if source.bias is None
                     else torch.nn.Parameter(source.bias.detach().clone()))
        self.convolution.bias = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.convolution(value)
        return output if self.bias is None else output + self.bias.reshape(1, -1, 1, 1)


class StridedTransposedConv3d(Module):
    """ConvTranspose3d with kernel == stride and no padding, as a matmul and reshape.

    coreai-torch 0.4.3 rejects 5D transposed convolutions. In this non-overlapping case
    each input voxel expands to one output block, so the rewrite is exact.
    """

    def __init__(self, conv: torch.nn.ConvTranspose3d) -> None:
        super().__init__()
        if (
            conv.kernel_size != conv.stride
            or conv.padding != (0, 0, 0)
            or conv.output_padding != (0, 0, 0)
            or conv.dilation != (1, 1, 1)
            or conv.groups != 1
        ):
            raise ValueError(f"only kernel == stride transposed convs can be rewritten: {conv}")
        self.kernel = conv.kernel_size
        self.out_channels = conv.out_channels
        self.weight = torch.nn.Parameter(conv.weight.detach().reshape(conv.in_channels, -1))
        self.bias = None if conv.bias is None else torch.nn.Parameter(conv.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, d, h, w = x.shape
        kd, kh, kw = self.kernel
        y = (x.permute(0, 2, 3, 4, 1) @ self.weight).reshape(
            n, d, h, w, self.out_channels, kd, kh, kw
        )
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(n, self.out_channels, d * kd, h * kh, w * kw)
        return y if self.bias is None else y + self.bias.view(1, -1, 1, 1, 1)


class ZeroInsertTransposedConv3d(Module):
    """Any ConvTranspose3d as zero insertion, padding, and a stride-1 Conv3d.

    This is the textbook equivalence: insert stride - 1 zeros between input voxels, pad by
    kernel - 1 - padding, and convolve with the spatially flipped, in/out-swapped kernel.
    """

    def __init__(self, conv: torch.nn.ConvTranspose3d) -> None:
        super().__init__()
        if conv.dilation != (1, 1, 1) or conv.groups != 1:
            raise ValueError(f"dilated or grouped transposed convs are not rewritten: {conv}")
        if isinstance(conv.padding, str):
            raise ValueError(f"string padding is not rewritten: {conv}")
        if any(k - 1 - p < 0 for k, p in zip(conv.kernel_size, conv.padding, strict=True)):
            raise ValueError(f"padding larger than kernel - 1 is not rewritten: {conv}")
        self.stride = conv.stride
        pads = [
            (k - 1 - p, k - 1 - p + op)
            for k, p, op in zip(conv.kernel_size, conv.padding, conv.output_padding, strict=True)
        ]
        # F.pad orders the last dimension first.
        self.pad = [amount for pair in reversed(pads) for amount in pair]
        weight = conv.weight.detach().flip(2, 3, 4).transpose(0, 1).contiguous()
        self.weight = torch.nn.Parameter(weight)
        self.bias = None if conv.bias is None else torch.nn.Parameter(conv.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for axis, stride in zip((2, 3, 4), self.stride, strict=True):
            if stride == 1:
                continue
            # Interleave zeros through a rank-3 view (dims up to the axis, stride, the rest),
            # which stays inside Core ML's rank-5 limit.
            shape = list(x.shape)
            leading = x.reshape(math.prod(shape[: axis + 1]), 1, math.prod(shape[axis + 1 :]))
            zeros = leading.new_zeros(leading.shape[0], stride - 1, leading.shape[2])
            x = torch.cat([leading, zeros], dim=1)
            x = x.reshape(shape[:axis] + [shape[axis] * stride] + shape[axis + 1 :])
            x = x.narrow(axis, 0, (shape[axis] - 1) * stride + 1)
        x = torch.nn.functional.pad(x, self.pad)
        return torch.nn.functional.conv3d(x, self.weight, self.bias)


def _output_padded(conv: torch.nn.ConvTranspose3d) -> torch.nn.Module:
    """coremltools rejects output_padding on 3D transposed convs; other cases convert."""
    if conv.output_padding == (0, 0, 0):
        return conv
    return ZeroInsertTransposedConv3d(conv)


def _transposed(conv: torch.nn.ConvTranspose3d) -> torch.nn.Module:
    """The cheaper matmul form when blocks don't overlap, zero insertion otherwise."""
    if conv.kernel_size == conv.stride and conv.padding == (0, 0, 0):
        if conv.output_padding == (0, 0, 0) and conv.dilation == (1, 1, 1) and conv.groups == 1:
            return StridedTransposedConv3d(conv)
    return ZeroInsertTransposedConv3d(conv)


class FlatInstanceNorm3d(Module):
    """InstanceNorm3d over the flattened spatial dims, which is the same normalization.

    coremltools converts InstanceNorm3d, but the Core ML runtime's instance_norm accepts
    only rank 3 or 4 inputs.
    """

    def __init__(self, norm: torch.nn.InstanceNorm3d) -> None:
        super().__init__()
        if norm.track_running_stats:
            raise ValueError(f"running statistics are not supported: {norm}")
        self.norm = torch.nn.InstanceNorm1d(norm.num_features, eps=norm.eps, affine=norm.affine)
        self.norm.load_state_dict(norm.state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.flatten(2)).view_as(x)


class FlatGroupNorm(Module):
    """GroupNorm over the flattened spatial dims, which is the same normalization.

    coremltools lowers 5D GroupNorm through a rank-6 reshape, past Core ML's rank-5 limit.
    """

    def __init__(self, norm: torch.nn.GroupNorm) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.flatten(2)).view_as(x)


def _pool_triple(value: int | tuple[int, ...]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    if len(value) != 3:
        raise ValueError(f"expected three values, got {value}")
    return value[0], value[1], value[2]


class SeparableMaxPool3d(Module):
    """MaxPool3d as a 2D max pool over (H, W) and then one over D.

    A max over a box is the max of maxes along each axis, and both passes pad with -inf,
    so this is the same result. coreai-torch 0.4.3 has no 3D max pooling.
    Request the indexed 2D operation and discard its indices: Torch 2.13 MPS
    export retains the nonindexed operation even after decomposition.
    """

    def __init__(self, pool: torch.nn.MaxPool3d) -> None:
        super().__init__()
        if pool.dilation not in (1, (1, 1, 1)) or pool.ceil_mode or pool.return_indices:
            raise ValueError(f"only plain max pooling is rewritten: {pool}")

        # MaxPool3d stores stride = kernel_size when constructed without one.
        self.kernel = _pool_triple(pool.kernel_size)
        self.stride = _pool_triple(pool.stride)
        self.padding = _pool_triple(pool.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        y = torch.nn.functional.max_pool2d_with_indices(
            x.reshape(n, c * d, h, w), self.kernel[1:], self.stride[1:], self.padding[1:],
            return_indices=True,
        )[0]
        oh, ow = y.shape[-2:]
        y = torch.nn.functional.max_pool2d_with_indices(
            y.reshape(n, c, d, oh * ow),
            (self.kernel[0], 1),
            (self.stride[0], 1),
            (self.padding[0], 0),
            return_indices=True,
        )[0]
        return y.reshape(n, c, -1, oh, ow)


class SeparableAvgPool3d(Module):
    """Average NCDHW boxes through two supported 2D pooling passes.

    Torch MPS 2.8 lacks AvgPool3d. Rectangular valid/padded counts factor by
    axis, including ceil-mode partial windows. A fixed divisor is applied once
    over H/W and the depth pass then sums, preserving divisor_override.
    """

    def __init__(self, pool: torch.nn.AvgPool3d) -> None:
        super().__init__()
        self.kernel = _pool_triple(pool.kernel_size)
        self.stride = _pool_triple(pool.stride)
        self.padding = _pool_triple(pool.padding)
        self.ceil_mode = pool.ceil_mode
        self.count_include_pad = pool.count_include_pad
        self.divisor = (None if pool.divisor_override is None
                        else as_int(pool.divisor_override))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("SeparableAvgPool3d requires NCDHW input")
        n, c, d, h, w = x.shape
        y = torch.nn.functional.avg_pool2d(
            x.reshape(n, c * d, h, w), self.kernel[1:], self.stride[1:], self.padding[1:],
            self.ceil_mode, self.count_include_pad, self.divisor,
        )
        oh, ow = y.shape[-2:]
        y = torch.nn.functional.avg_pool2d(
            y.reshape(n, c, d, oh * ow), (self.kernel[0], 1),
            (self.stride[0], 1), (self.padding[0], 0), self.ceil_mode,
            self.count_include_pad, None if self.divisor is None else 1,
        )
        return y.reshape(n, c, -1, oh, ow)


class SeparableTrilinearUpsample(Module):
    """Trilinear nn.Upsample as a bilinear pass over (H, W) and then one over D.

    Preserve scale factors, corner alignment and scale recomputation in both passes.
    Passing rounded output sizes instead would change fractional-scale coordinates.
    """

    def __init__(self, upsample: torch.nn.Upsample) -> None:
        super().__init__()
        if upsample.mode != "trilinear" or upsample.size is not None:
            raise ValueError(f"only scale-factor trilinear upsampling is rewritten: {upsample}")
        scale = upsample.scale_factor
        assert scale is not None
        self.scale: tuple[float, ...] = (
            tuple(scale) if isinstance(scale, tuple) else (scale, scale, scale)
        )
        self.align_corners = upsample.align_corners
        self.recompute_scale_factor = upsample.recompute_scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        sd, sh, sw = self.scale
        y = interpolate_bilinear(x.reshape(n, c * d, h, w), (sh, sw),
                                 self.align_corners, self.recompute_scale_factor)
        oh, ow = y.shape[-2:]
        y = interpolate_bilinear(y.reshape(n, c, d, oh * ow), (sd, 1.0),
                                 self.align_corners, self.recompute_scale_factor)
        return y.reshape(n, c, y.shape[2], oh, ow)


def _trilinear(upsample: torch.nn.Upsample) -> torch.nn.Module:
    return SeparableTrilinearUpsample(upsample) if upsample.mode == "trilinear" else upsample


COREAI_3D: list[Swap] = [
    ModuleSwap(torch.nn.ConvTranspose3d, _transposed),
    ModuleSwap(torch.nn.MaxPool3d, SeparableMaxPool3d),
    ModuleSwap(torch.nn.Upsample, _trilinear),
]
COREML_3D: list[Swap] = [
    ModuleSwap(torch.nn.ConvTranspose3d, _output_padded),
    ModuleSwap(torch.nn.InstanceNorm3d, FlatInstanceNorm3d),
    ModuleSwap(torch.nn.GroupNorm, FlatGroupNorm),
    ModuleSwap(torch.nn.Upsample, _trilinear),
]
# MPS also lacks half ConvTranspose3d and 3D pooling. Apply these explicit
# compatibility rewrites before precision authoring, with full FP32 parity.
MPS_FP16: list[Swap] = [*COREAI_3D, ModuleSwap(torch.nn.AvgPool3d, SeparableAvgPool3d)]
COREML_FP16_2D: list[Swap] = [
    ModuleSwap(torch.nn.Conv2d, SeparateBias2d),
    ModuleSwap(torch.nn.ConvTranspose2d, SeparateBias2d),
]


def replace_modules(model: torch.nn.Module, swaps: list[Swap]) -> torch.nn.Module:
    """Return a copy of ``model`` with every module of each swapped type replaced.

    Matching is by exact type, so a subclass with its own ``forward`` is never swapped.
    """
    rewritten = copy.deepcopy(model)
    for parent in list(rewritten.modules()):
        for name, child in parent.named_children():
            for swap in swaps:
                replacement = swap.apply(child)
                if replacement is not None:
                    setattr(parent, name, replacement)
                    break
    return rewritten
