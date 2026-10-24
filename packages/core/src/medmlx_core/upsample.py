"""Float32 decoder fusions for NCDHW SegResNet inference on MLX Metal.

``upsample_add_ncdhw`` fuses 2x trilinear upsampling with the skip addition in one
Metal kernel, and ``deconv2x_ncdhw`` computes a 2x transposed convolution as a
single matmul. Both cover the pinned SegResNet decoder configuration only.
"""

from __future__ import annotations

from functools import cache
from math import prod
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.layout import require_ncdhw
from medmlx_core.typing import MetalKernel, MlxRuntime

# Half-pixel coordinates implement PyTorch align_corners=False interpolation.
_SOURCE = """
uint idx = thread_position_in_grid.x;
if (idx >= N * C * OD * OH * OW) return;
uint n, c, oz, oy, ox;
if (CHANNELS_LAST) {
    c = idx % C;
    uint v = idx / C;
    ox = v % OW; oy = (v / OW) % OH; oz = (v / (OW * OH)) % OD;
    n = v / (OW * OH * OD);
} else {
    ox = idx % OW; oy = (idx / OW) % OH; oz = (idx / (OW * OH)) % OD;
    c = (idx / (OW * OH * OD)) % C;
    n = idx / (OW * OH * OD * C);
}
int x0 = int(ox / 2) - ((ox & 1) == 0 ? 1 : 0);
int y0 = int(oy / 2) - ((oy & 1) == 0 ? 1 : 0);
int z0 = int(oz / 2) - ((oz & 1) == 0 ? 1 : 0);
float tx = (ox & 1) ? 0.25f : 0.75f;
float ty = (oy & 1) ? 0.25f : 0.75f;
float tz = (oz & 1) ? 0.25f : 0.75f;
// Clamp the coordinate before weighting, including singleton input axes.
if (x0 < 0 || x0 >= W - 1) tx = 0.0f;
if (y0 < 0 || y0 >= H - 1) ty = 0.0f;
if (z0 < 0 || z0 >= D - 1) tz = 0.0f;
x0 = clamp(x0, 0, W - 1); y0 = clamp(y0, 0, H - 1); z0 = clamp(z0, 0, D - 1);
int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1), z1 = min(z0 + 1, D - 1);
int ca = CHANNELS_LAST ? 4 : 1;
int za = CHANNELS_LAST ? 1 : 2;
int ya = CHANNELS_LAST ? 2 : 3;
int xa = CHANNELS_LAST ? 3 : 4;
long base = long(n) * x_strides[0] + long(c) * x_strides[ca];
long a000 = base + long(z0)*x_strides[za] + long(y0)*x_strides[ya] + long(x0)*x_strides[xa];
long a001 = base + long(z0)*x_strides[za] + long(y0)*x_strides[ya] + long(x1)*x_strides[xa];
long a010 = base + long(z0)*x_strides[za] + long(y1)*x_strides[ya] + long(x0)*x_strides[xa];
long a011 = base + long(z0)*x_strides[za] + long(y1)*x_strides[ya] + long(x1)*x_strides[xa];
long a100 = base + long(z1)*x_strides[za] + long(y0)*x_strides[ya] + long(x0)*x_strides[xa];
long a101 = base + long(z1)*x_strides[za] + long(y0)*x_strides[ya] + long(x1)*x_strides[xa];
long a110 = base + long(z1)*x_strides[za] + long(y1)*x_strides[ya] + long(x0)*x_strides[xa];
long a111 = base + long(z1)*x_strides[za] + long(y1)*x_strides[ya] + long(x1)*x_strides[xa];
float l00 = (1.0f-tx)*x[a000] + tx*x[a001];
float l01 = (1.0f-tx)*x[a010] + tx*x[a011];
float l10 = (1.0f-tx)*x[a100] + tx*x[a101];
float l11 = (1.0f-tx)*x[a110] + tx*x[a111];
float l0 = (1.0f-ty)*l00 + ty*l01;
float l1 = (1.0f-ty)*l10 + ty*l11;
float value = (1.0f-tz)*l0 + tz*l1;
long s = long(n)*skip_strides[0] + long(c)*skip_strides[ca]
       + long(oz)*skip_strides[za] + long(oy)*skip_strides[ya] + long(ox)*skip_strides[xa];
out[idx] = value + skip[s];
"""


@cache
def _kernel(mx: MlxRuntime) -> MetalKernel:
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="medmlx_segresnet_trilinear2x_add",
            input_names=["x", "skip"],
            output_names=["out"],
            source=_SOURCE,
            header="#pragma clang fp contract(off)\n",
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        ),
    )


def upsample_add_ncdhw(values: Array, skip: Array, *, mx: MlxRuntime) -> Array:
    """Fuse align_corners=False 2x trilinear interpolation and skip addition on Metal.

    Args:
        values: Float32 ``(N, C, D, H, W)`` decoder input.
        skip: Float32 ``(N, C, 2D, 2H, 2W)`` encoder skip tensor.
        mx: MLX module.

    Returns:
        Float32 ``(N, C, 2D, 2H, 2W)`` array equal to the interpolated ``values``
        plus ``skip``.

    Raises:
        TypeError: An input is not float32.
        ValueError: An input is not rank 5, ``skip`` has the wrong shape, or the
            output element count exceeds the 32-bit Metal index limit.
    """
    require_ncdhw(values, name="decoder input")
    require_ncdhw(skip, name="decoder skip")
    if values.dtype != mx.float32 or skip.dtype != mx.float32:
        raise TypeError("SegResNet decoder fusion requires float32 inputs")
    n, c, d, h, w = values.shape
    shape = (n, c, 2 * d, 2 * h, 2 * w)
    if skip.shape != shape:
        raise ValueError(f"Decoder skip shape {skip.shape} must equal {shape}")
    count = prod(shape)
    if count > 2**31 - 1:
        raise ValueError("SegResNet decoder fusion exceeds the Metal index limit")
    return _kernel(mx)(
        inputs=[values, skip],
        template=list(
            dict(N=n, C=c, D=d, H=h, W=w, OD=2 * d, OH=2 * h, OW=2 * w, CHANNELS_LAST=False).items()
        ),
        grid=(((count + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]


def deconv2x_ncdhw(values: Array, weight: Array, bias: Array | None, *, mx: MlxRuntime) -> Array:
    """2x NCDHW transposed convolution with all eight phases in one MLX matmul.

    This covers the pinned SegResNet decoder configuration only: groups=1,
    kernel=stride=2, padding=output_padding=0.  Keeping the checkpoint's
    Torch layout avoids the generic MLX ConvTranspose3d fast path here.

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

    source = mx.transpose(values, (0, 2, 3, 4, 1))
    # Pack phases before output channels, retaining the original D/H/W phase order.
    projection = mx.reshape(mx.transpose(weight, (0, 2, 3, 4, 1)), (channels, 8 * out_channels))
    packed = mx.reshape(source @ projection, (n, depth, height, width, 2, 2, 2, out_channels))
    # Keep storage channel-last: channel-last consumers can recover it with a view.
    output = mx.reshape(
        mx.transpose(packed, (0, 1, 4, 2, 5, 3, 6, 7)),
        (n, 2 * depth, 2 * height, 2 * width, out_channels),
    )
    if bias is not None:
        output = output + mx.reshape(bias, (1, 1, 1, 1, out_channels))
    return mx.transpose(output, (0, 4, 1, 2, 3))
