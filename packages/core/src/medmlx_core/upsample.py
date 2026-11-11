"""Float32 2x transposed convolution for NCDHW decoders on MLX Metal.

``deconv2x_ncdhw`` computes a kernel=stride=2 transposed convolution with an
ordered FP32 projection. It covers the pinned SegResNet decoder configuration only.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.layout import require_ncdhw
from medmlx_core.precision import Float32Operators as _Float32Operators
from medmlx_core.typing import MlxRuntime


@cache
def _deconv_projection(mx: MlxRuntime) -> _Float32Operators:
    """Reuse the ordered convolution kernel for the eight decoder phases.

    Its explicit float32 accumulation avoids the reduced-precision path that
    ordinary MLX matmul may select on M5. GroupNorm is unused by this projection.
    """
    return _Float32Operators(mx, groups=1, eps=0.0)


def deconv2x_ncdhw(values: Array, weight: Array, bias: Array | None, *, mx: MlxRuntime) -> Array:
    """2x NCDHW transposed convolution with all eight phases projected in full FP32.

    This covers the pinned SegResNet decoder configuration only: groups=1,
    kernel=stride=2, padding=output_padding=0.  Keeping the checkpoint's
    Torch layout avoids the generic MLX ConvTranspose3d fast path here. A single
    ordered 1x1x1 convolution computes all phase/channel outputs; bias is added
    after the reduction, matching the original transposed convolution.

    Args:
        values: Float32 ``(N, C, D, H, W)`` input.
        weight: Float32 Torch ConvTranspose3d weight of shape ``(C, O, 2, 2, 2)``.
        bias: Optional float32 ``(O,)`` bias.
        mx: MLX module.

    Returns:
        ``(N, O, 2D, 2H, 2W)`` array. It is a transposed view of channel-last
        storage, so a following channel-last op can use it without a copy.

    Raises:
        TypeError: ``values`` or ``weight`` is not float32.
        ValueError: ``values`` is not rank 5, or ``weight`` or ``bias`` has the
            wrong shape or dtype.
    """

    require_ncdhw(values, name="decoder deconvolution input")
    if values.dtype != mx.float32 or weight.dtype != mx.float32:
        raise TypeError("SegResNet decoder deconvolution requires float32 inputs")
    if len(weight.shape) != 5:
        raise ValueError("SegResNet decoder deconvolution weight must be rank 5")
    n, channels, depth, height, width = (int(size) for size in values.shape)
    out_channels = int(weight.shape[1])
    expected_weight_shape = (channels, out_channels, 2, 2, 2)
    if tuple(int(size) for size in weight.shape) != expected_weight_shape:
        raise ValueError(
            "SegResNet decoder deconvolution weight has shape "
            f"{tuple(weight.shape)}; expected {expected_weight_shape}"
        )
    if bias is not None and (
        bias.dtype != mx.float32 or tuple(int(size) for size in bias.shape) != (out_channels,)
    ):
        raise ValueError(
            "SegResNet decoder deconvolution bias must have shape "
            f"({out_channels},) and dtype float32"
        )

    # Pack phases before output channels, retaining the original D/H/W phase order.
    # An ordinary float32 matmul can use reduced-precision products on M5 even
    # with float32 inputs and outputs; this path must honor the FP32 contract
    # independently of the process-wide MLX_ENABLE_TF32 setting.
    projection = mx.reshape(
        mx.transpose(weight, (2, 3, 4, 1, 0)), (8 * out_channels, channels, 1, 1, 1)
    )
    projected = _deconv_projection(mx).conv(values, projection, None, padding=0, stride=1)
    packed = mx.reshape(
        mx.transpose(projected, (0, 2, 3, 4, 1)),
        (n, depth, height, width, 2, 2, 2, out_channels),
    )
    # Keep storage channel-last: channel-last consumers can recover it with a view.
    output = mx.reshape(
        mx.transpose(packed, (0, 1, 4, 2, 5, 3, 6, 7)),
        (n, 2 * depth, 2 * height, 2 * width, out_channels),
    )
    if bias is not None:
        output = output + mx.reshape(bias, (1, 1, 1, 1, out_channels))
    return mx.transpose(output, (0, 4, 1, 2, 3))
