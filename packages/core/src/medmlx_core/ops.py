"""Shared MLX primitives for NCDHW volumes, written for the MAISI autoencoder.

Tensors are float32 ``(N, C, D, H, W)`` between operations; each operation moves to
MLX's channel-last layout internally and back. Weights keep the Torch layout.
Importing this module does not import MLX.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.layout import (
    conv3d_weight_to_mlx,
    conv_transpose3d_weight_to_mlx,
    require_ncdhw,
    to_ncdhw,
    to_ndhwc,
)
from medmlx_core.typing import ArrayFactory, HostArray, MlxRuntime
from medmlx_core.upsample import deconv2x_ncdhw


def as_fp32(array: Array | HostArray, mx: MlxRuntime) -> Array:
    """Copy a host or device array to MLX as float32."""

    make_array = cast(ArrayFactory, mx.array)
    return make_array(array, dtype=mx.float32)


def silu(array: Array, mx: MlxRuntime) -> Array:
    """SiLU, matching ``torch.nn.SiLU`` / MONAI ``Swish(alpha=1)``."""

    return array * mx.sigmoid(array)


def linear(array: Array, weight: Array, bias: Array | None, mx: MlxRuntime) -> Array:
    """``x @ W.T + b`` with Torch/MLX layout ``[out, in]``."""

    output = array @ mx.swapaxes(weight, 0, 1)
    if bias is not None:
        output = output + bias
    return output


def group_norm_ncdhw(
    array: Array,
    weight: Array,
    bias: Array,
    *,
    num_groups: int,
    eps: float,
    mx: MlxRuntime,
) -> Array:
    """NCDHW GroupNorm matching ``mlx.nn.GroupNorm(..., pytorch_compatible=True)``.

    Args:
        array: ``(N, C, D, H, W)`` input.
        weight: Per-channel scale of shape ``(C,)``.
        bias: Per-channel shift of shape ``(C,)``.
        num_groups: Number of channel groups; must divide ``C``.
        eps: Variance epsilon.

    Raises:
        ValueError: ``array`` is not rank 5 or ``C`` is not divisible by ``num_groups``.
    """

    require_ncdhw(array, name="group_norm input")
    spatial = to_ndhwc(array, mx)
    batch = int(spatial.shape[0])
    channels = int(spatial.shape[-1])
    if channels % num_groups != 0:
        raise ValueError(
            f"group_norm channels {channels} must be divisible by num_groups {num_groups}"
        )
    group_size = channels // num_groups
    grouped = mx.reshape(spatial, (batch, -1, num_groups, group_size))
    grouped = mx.transpose(grouped, (0, 2, 1, 3))
    grouped = mx.reshape(grouped, (batch, num_groups, -1))
    normalized = mx.fast.layer_norm(grouped, eps=eps, weight=None, bias=None)
    restored = mx.reshape(normalized, (batch, num_groups, -1, group_size))
    restored = mx.transpose(restored, (0, 2, 1, 3))
    restored = mx.reshape(restored, spatial.shape)
    return to_ncdhw(restored * weight + bias, mx)


def conv3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    mx: MlxRuntime,
    stride: int = 1,
) -> Array:
    """NCDHW conv3d with a Torch ``[O, I, KD, KH, KW]`` weight and optional ``(O,)`` bias.

    ``padding`` and ``stride`` are one int applied to all three spatial axes; there
    is no dilation or grouping.
    """

    require_ncdhw(array, name="conv3d input")
    output = mx.conv3d(
        to_ndhwc(array, mx),
        conv3d_weight_to_mlx(weight, mx),
        stride=stride,
        padding=padding,
    )
    if bias is not None:
        output = output + bias
    return to_ncdhw(output, mx)


def conv_transpose3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    output_padding: int,
    mx: MlxRuntime,
    stride: int = 1,
) -> Array:
    """NCDHW conv_transpose3d with a Torch ``[I, O, KD, KH, KW]`` weight.

    The stride 2, kernel 2, zero padding, all-float32 case is routed to the fused
    ``deconv2x_ncdhw`` kernel; every other case uses ``mx.conv_transpose3d``.
    """

    require_ncdhw(array, name="conv_transpose3d input")
    if (
        stride == 2
        and padding == output_padding == 0
        and tuple(weight.shape[2:]) == (2, 2, 2)
        and array.dtype == weight.dtype == mx.float32
        and (bias is None or bias.dtype == mx.float32)
    ):
        # MLX's stock transposed conv marks phase views of the public output as
        # temporaries, which loses cross-encoder synchronization when a lazy
        # decoder consumes it; confirmed still reproducible on mlx 0.32.4, the
        # pinned floor (see test_transpose_convolution.py), so this custom path
        # stays in place regardless of patch version.
        return deconv2x_ncdhw(array, weight, bias, mx=mx)
    output = mx.conv_transpose3d(
        to_ndhwc(array, mx),
        conv_transpose3d_weight_to_mlx(weight, mx),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    if bias is not None:
        output = output + bias
    return to_ncdhw(output, mx)


def split_conv3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    mx: MlxRuntime,
    stride: int = 1,
    num_splits: int = 1,
    dim_split: int = 1,
    convolution: Callable[..., Array] = conv3d_ncdhw,
) -> Array:
    """Convolve in overlapping chunks along one spatial axis, as MAISI's ``MaisiConvolution`` does.

    Splitting bounds peak memory. With ``num_splits <= 1`` this is a single call to
    ``convolution``; ``dim_split`` is then unused.

    Args:
        array: ``(N, C, D, H, W)`` input.
        weight: Torch ``[O, I, KD, KH, KW]`` weight.
        bias: Optional ``(O,)`` bias.
        padding: Symmetric padding on every spatial axis.
        mx: MLX module.
        stride: Stride on every spatial axis.
        num_splits: Number of chunks.
        dim_split: Spatial axis to split: 0, 1 or 2 for D, H or W.
        convolution: Callable with the ``conv3d_ncdhw`` signature applied per chunk.

    Raises:
        ValueError: ``array`` is not rank 5, ``dim_split`` is not 0, 1 or 2, the
            axis is shorter than ``num_splits``, or the axis length is not evenly
            divisible by ``num_splits``.
    """

    return _split_convolution_ncdhw(
        array,
        apply=lambda chunk: convolution(chunk, weight, bias, padding=padding, stride=stride, mx=mx),
        stride=stride,
        num_splits=num_splits,
        dim_split=dim_split,
        name="split_conv3d",
        mx=mx,
    )


def split_conv_transpose3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    output_padding: int,
    mx: MlxRuntime,
    stride: int = 1,
    num_splits: int = 1,
    dim_split: int = 1,
) -> Array:
    """Transposed counterpart of ``split_conv3d_ncdhw``, with the same chunking rules.

    Raises:
        ValueError: As for ``split_conv3d_ncdhw``.
    """

    return _split_convolution_ncdhw(
        array,
        apply=lambda chunk: conv_transpose3d_ncdhw(
            chunk,
            weight,
            bias,
            padding=padding,
            output_padding=output_padding,
            stride=stride,
            mx=mx,
        ),
        stride=stride,
        num_splits=num_splits,
        dim_split=dim_split,
        name="split_conv_transpose3d",
        mx=mx,
    )


def _split_convolution_ncdhw(
    array: Array,
    *,
    apply: Callable[[Array], Array],
    stride: int,
    num_splits: int,
    dim_split: int,
    name: str,
    mx: MlxRuntime,
) -> Array:
    require_ncdhw(array, name=f"{name} input")
    if num_splits <= 1:
        return apply(array)
    if dim_split not in (0, 1, 2):
        raise ValueError(f"dim_split must be 0, 1, or 2; got {dim_split}")
    axis = dim_split + 2
    length = int(array.shape[axis])
    split_size = length // num_splits
    if split_size < 1:
        raise ValueError(f"{name} length {length} is smaller than num_splits {num_splits}")
    if length % num_splits != 0:
        # MaisiConvolution's own crop takes a fixed split_size from every chunk,
        # including the last, so a non-divisible length silently drops the
        # remainder there too (verified against the pinned upstream source).
        # Refuse instead of reproducing that silent truncation.
        raise ValueError(f"{name} length {length} is not divisible by num_splits {num_splits}")
    # Chunks share 3 voxels of context, the reach of a 3-wide kernel, raised to a
    # multiple of the stride so chunk starts stay aligned with strided outputs.
    # This mirrors MaisiConvolution and does not adapt to other kernel sizes.
    overlap = 3
    if overlap % stride > 0:
        overlap = (overlap // stride + 1) * stride
    chunks: list[Array] = []
    for index in range(num_splits):
        start = 0 if index == 0 else index * split_size - overlap
        extra = 0 if index == num_splits - 1 else overlap
        end = (index + 1) * split_size + extra
        chunks.append(_slice_axis(array, axis, start, end))
    outputs = [apply(chunk) for chunk in chunks]
    other = (dim_split + 1 if dim_split < 2 else 0) + 2
    in_other = int(chunks[0].shape[other])
    out_other = int(outputs[0].shape[other])
    out_split = split_size
    out_overlap = overlap
    # MaisiConvolution checks integer // == 2, not exact 2x growth. A
    # trailing-pad stride-2 downsample maps S+1 to S//2, for example 16 -> 17 -> 8.
    if in_other > 0 and out_other // in_other == 2:
        out_split *= 2
        out_overlap *= 2
    elif out_other > 0 and in_other // out_other == 2:
        out_split //= 2
        out_overlap //= 2
    cropped: list[Array] = []
    for index, output in enumerate(outputs):
        if index == 0:
            cropped.append(_slice_axis(output, axis, 0, out_split))
        else:
            cropped.append(_slice_axis(output, axis, out_overlap, out_overlap + out_split))
    return mx.concatenate(cropped, axis=axis)


def _slice_axis(array: Array, axis: int, start: int, end: int) -> Array:
    slices: list[slice] = [slice(None)] * 5
    slices[axis] = slice(start, end)
    return array[tuple(slices)]


def avg_pool3d_ncdhw(
    array: Array, *, mx: MlxRuntime, kernel_size: int = 2, stride: int = 2
) -> Array:
    """NCDHW AvgPool3d matching ``mlx.nn.AvgPool3d`` / Torch ``AvgPool3d(2)``.

    The kernel and stride are one int applied to every spatial axis.
    """

    require_ncdhw(array, name="avg pool input")
    spatial = to_ndhwc(array, mx)
    mlx_nn = importlib.import_module("mlx.nn")

    pool = cast(
        "Callable[[Array], Array]", mlx_nn.AvgPool3d(kernel_size=kernel_size, stride=stride)
    )
    pooled = pool(spatial)
    return to_ncdhw(pooled, mx)


def upsample_nearest_ncdhw(array: Array, *, mx: MlxRuntime, scale: int = 2) -> Array:
    """Upsample NCDHW by an integer ``scale`` per spatial axis with nearest-neighbor repeats.

    Raises:
        ValueError: ``array`` is not rank 5 or ``scale`` is below 1.
    """

    require_ncdhw(array, name="nearest upsample input")
    if scale < 1:
        raise ValueError(f"upsample scale must be >= 1; got {scale}")
    output = array
    for axis in (2, 3, 4):
        output = mx.repeat(output, scale, axis=axis)
    return output


def upsample_trilinear_ncdhw(array: Array, *, mx: MlxRuntime, scale: int = 2) -> Array:
    """Upsample NCDHW by ``scale`` per spatial axis, trilinear with ``align_corners=False``.

    ``align_corners=False`` is the ``torch.nn.functional.interpolate`` default.
    """

    require_ncdhw(array, name="trilinear upsample input")
    mlx_nn = importlib.import_module("mlx.nn")

    upsample = cast(
        "Callable[[Array], Array]",
        mlx_nn.Upsample(scale_factor=scale, mode="linear", align_corners=False),
    )
    upsampled = upsample(to_ndhwc(array, mx))
    return to_ncdhw(upsampled, mx)


def pad_spatial_trailing_ncdhw(array: Array, mx: MlxRuntime) -> Array:
    """``F.pad(x, (0, 1) * 3)``: add one voxel on the high side of D, H, W."""

    require_ncdhw(array, name="spatial pad input")
    return mx.pad(array, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)])


def concat_channels_ncdhw(left: Array, right: Array, mx: MlxRuntime) -> Array:
    """Skip-concat on NCDHW channel axis 1."""

    require_ncdhw(left, name="skip concat left")
    require_ncdhw(right, name="skip concat right")
    return mx.concatenate([left, right], axis=1)
