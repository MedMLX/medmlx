"""Compensated Metal operators and paired hidden layers for accuracy profiles.

The standard operators retain residuals internally and round boundaries to
float32. The paired profile carries both words across its supported hidden
layers. The package tests compare both with float64 reference arithmetic on
small synthetic inputs; that is not a general error bound for other shapes or
data.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from itertools import count
from typing import TYPE_CHECKING, cast

from medmlx_core.layout import require_ncdhw
from medmlx_core.typing import MetalKernel, MlxRuntime

if TYPE_CHECKING:
    from mlx.core import array as Array

_NAMESPACES = count()

_HEADER = r"""
#include <metal_stdlib>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline float2 ds_add(float2 a, float2 b) {
    volatile float s = a.x + b.x;
    volatile float v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v);
    float t = e + (a.y + b.y);
    volatile float h = s + t;
    return float2(h, t - (h - s));
}
inline float2 ds_neg(float2 a) { return float2(-a.x, -a.y); }
inline float2 ds_sub(float2 a, float2 b) { return ds_add(a, ds_neg(b)); }
inline float2 ds_mul(float2 a, float2 b) {
    volatile float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e += a.x * b.y + a.y * b.x;
    e += a.y * b.y;
    return ds_add(float2(p, 0), float2(e, 0));
}
inline float2 ds_div(float2 a, float b) {
    float q = a.x / b;
    float2 r = ds_sub(a, ds_mul(float2(q, 0), float2(b, 0)));
    return ds_add(float2(q, 0), float2((r.x + r.y) / b, 0));
}
inline float2 ds_rsqrt(float2 a) {
    float2 r = float2(rsqrt(a.x), 0);
    for (uint iteration = 0; iteration < 2; ++iteration) {
        float2 e = ds_sub(float2(1, 0), ds_mul(a, ds_mul(r, r)));
        r = ds_add(r, ds_mul(ds_mul(float2(0.5f, 0), r), e));
    }
    return r;
}

constant float2 DS_LN2 = float2(0.69314718246459961f, -1.9046542121259336e-09f);
constant float2 DS_EXP_COEFFICIENTS[14] = {
    float2(1.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.5f, 0.0f),
    float2(0.1666666716337204f, -4.9670538793122887e-09f),
    float2(0.041666667908430099f, -1.2417634698280722e-09f),
    float2(0.0083333337679505348f, -4.3461720333759502e-10f),
    float2(0.0013888889225199819f, -3.3631094437103215e-11f),
    float2(0.00019841270113829523f, -2.7255968749334558e-12f),
    float2(2.4801587642286904e-05f, -3.4069960936668198e-13f),
    float2(2.7557318844628753e-06f, 3.7935712242972291e-14f),
    float2(2.755731998149713e-07f, -7.5751122090511949e-15f),
    float2(2.5052107943679403e-08f, 4.4176230446483665e-16f),
    float2(2.0876755879584152e-09f, 1.1082839809204342e-16f),
    float2(1.6059044372074283e-10f, -5.3525265115627256e-18f)
};

inline float2 ds_exp_nonpositive(float2 x) {
    // Below this bound even exp(x)/1 rounds to zero in a float32 softmax.
    if (x.x < -104.0f) return float2(0, 0);
    float exponent = rint(x.x * 1.4426950408889634f);
    float2 r = ds_sub(x, ds_mul(float2(exponent, 0), DS_LN2));
    float2 p = DS_EXP_COEFFICIENTS[13];
    for (int index = 12; index >= 0; --index)
        p = ds_add(DS_EXP_COEFFICIENTS[index], ds_mul(r, p));
    return float2(ldexp(p.x, int(exponent)), ldexp(p.y, int(exponent)));
}
inline float2 ds_ratio(float2 a, float2 b) {
    float q = a.x / b.x;
    float2 r = ds_sub(a, ds_mul(float2(q, 0), b));
    return ds_add(float2(q, 0), float2((r.x + r.y) / b.x, 0));
}
inline float2 ds_load(const device float* x, uint i, uint size, bool carry) {
    return float2(x[i], carry ? x[i + size] : 0.0f);
}
inline float2 ds_load(const constant float* x, uint i, uint size, bool carry) {
    return float2(x[i], carry ? x[i + size] : 0.0f);
}
inline void ds_store(device float* y, uint i, uint size, float2 value, bool carry) {
    y[i] = carry ? value.x : value.x + value.y;
    if (carry) y[i + size] = value.y;
}
"""

_CONV = r"""
uint i = thread_position_in_grid.x;
if (i >= BATCH * CO * D * H * S) return;
uint x = i % S, y = (i / S) % H, z = (i / (S * H)) % D;
uint oc = (i / (D * H * S)) % CO, b = i / (CO * D * H * S);
float2 result = float2(BIAS[oc], 0);
for (uint ic = 0; ic < CI; ++ic)
 for (uint kz = 0; kz < KZ; ++kz)
  for (uint ky = 0; ky < KY; ++ky)
   for (uint kx = 0; kx < KX; ++kx) {
    int iz = int(z * SZ + kz) - PZ;
    int iy = int(y * SY + ky) - PY;
    int ix = int(x * SX + kx) - PX;
    if (iz < 0 || iz >= DI || iy < 0 || iy >= HI || ix < 0 || ix >= SI) continue;
    uint position = (((b * CI + ic) * DI + iz) * HI + iy) * SI + ix;
    float a = X[position];
    float w = W[(((oc * CI + ic) * KZ + kz) * KY + ky) * KX + kx];
    volatile float product = a * w;
    result = ds_add(result, CARRY ? ds_mul(ds_load(X, position, INPUT_SIZE, true), float2(w, 0))
                                : float2(product, fma(a, w, -product)));
   }
ds_store(Y, i, OUTPUT_SIZE, result, CARRY);
"""

_DECONV = r"""
uint i = thread_position_in_grid.x;
if (i >= BATCH * CO * D * H * S) return;
uint x = i % S, y = (i / S) % H, z = (i / (S * H)) % D;
uint oc = (i / (D * H * S)) % CO, b = i / (CO * D * H * S);
uint kz = z % 2, ky = y % 2, kx = x % 2;
float2 result = float2(BIAS[oc], 0);
for (uint ic = 0; ic < CI; ++ic) {
    uint position = (((b * CI + ic) * DI + z / 2) * HI + y / 2) * SI + x / 2;
    float a = X[position];
    float w = W[(((ic * CO + oc) * 2 + kz) * 2 + ky) * 2 + kx];
    volatile float product = a * w;
    result = ds_add(result, CARRY ? ds_mul(ds_load(X, position, INPUT_SIZE, true), float2(w, 0))
                                : float2(product, fma(a, w, -product)));
}
ds_store(Y, i, OUTPUT_SIZE, result, CARRY);
"""

_NORM = r"""
uint tid = thread_position_in_threadgroup.x;
uint g = threadgroup_position_in_grid.x;
threadgroup float sh[256], sl[256];
float2 sum = float2(0, 0);
for (uint i = tid; i < N; i += 256) sum = ds_add(sum, ds_load(X, g * N + i, INPUT_SIZE, CARRY));
sh[tid] = sum.x; sl[tid] = sum.y;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride /= 2) {
 if (tid < stride) {
    float2 s = ds_add(float2(sh[tid], sl[tid]), float2(sh[tid + stride], sl[tid + stride]));
    sh[tid] = s.x; sl[tid] = s.y;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
float2 mean = ds_div(float2(sh[0], sl[0]), float(N));
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = float2(0, 0);
for (uint i = tid; i < N; i += 256) {
 float2 centered = ds_sub(ds_load(X, g * N + i, INPUT_SIZE, CARRY), mean);
 sum = ds_add(sum, ds_mul(centered, centered));
}
sh[tid] = sum.x; sl[tid] = sum.y;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride /= 2) {
 if (tid < stride) {
    float2 s = ds_add(float2(sh[tid], sl[tid]), float2(sh[tid + stride], sl[tid + stride]));
    sh[tid] = s.x; sl[tid] = s.y;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
float2 variance = ds_div(float2(sh[0], sl[0]), float(N));
float2 inverse = ds_rsqrt(ds_add(variance, float2(EPS[0], EPS[1])));
uint previous = 0xffffffff;
float2 scale, offset;
for (uint i = tid; i < N; i += 256) {
 uint channel = (g % GROUPS) * (CHANNELS / GROUPS) + i / SPATIAL;
 if (channel != previous) {
    scale = ds_mul(inverse, float2(W[channel], 0));
    offset = ds_sub(float2(B[channel], 0), ds_mul(scale, mean));
    previous = channel;
 }
 float2 v = ds_add(ds_mul(scale, ds_load(X, g * N + i, INPUT_SIZE, CARRY)), offset);
 ds_store(Y, g * N + i, OUTPUT_SIZE, v, CARRY);
}
"""


_BN = r"""
uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
uint c = i % C;
float2 v = ds_add(float2(VAR[c], 0), float2(EPS[0], EPS[1]));
float2 scale = ds_mul(ds_rsqrt(v), float2(W[c], 0));
float2 offset = ds_sub(float2(B[c], 0), ds_mul(float2(MEAN[c], 0), scale));
float2 y = ds_add(ds_mul(float2(X[i], 0), scale), offset);
Y[i] = y.x + y.y;
"""

_ELU = r"""
uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
float x = X[i];
if (x >= 0) { Y[i] = x; return; }
// exp(x)-1 rounds to -1 below -18; retain small negative inputs via float pairs.
if (x < -18.0f) { Y[i] = -1; return; }
float2 y = ds_sub(ds_exp_nonpositive(float2(x, 0)), float2(1, 0));
Y[i] = y.x + y.y;
"""

_SOFTMAX = r"""
uint row = thread_position_in_grid.x;
if (row >= ROWS) return;
float maximum = X[row * C];
for (uint c = 1; c < C; ++c) maximum = fmax(maximum, X[row * C + c]);
float2 sum = float2(0, 0);
for (uint c = 0; c < C; ++c) {
    float2 difference = ds_sub(float2(X[row * C + c], 0), float2(maximum, 0));
    sum = ds_add(sum, ds_exp_nonpositive(difference));
}
for (uint c = 0; c < C; ++c) {
    float2 difference = ds_sub(float2(X[row * C + c], 0), float2(maximum, 0));
    float2 y = ds_ratio(ds_exp_nonpositive(difference), sum);
    Y[row * C + c] = y.x + y.y;
}
"""

_GAUSSIAN = r"""
uint i = thread_position_in_grid.x;
if (i >= BATCH * D * H * S * C) return;
uint c = i % C, x = (i / C) % S, y = (i / (C * S)) % H;
uint z = (i / (C * S * H)) % D, b = i / (C * S * H * D);
float2 sum = float2(0, 0);
for (uint kz = 0; kz < K; ++kz)
 for (uint ky = 0; ky < K; ++ky)
  for (uint kx = 0; kx < K; ++kx) {
    int iz = int(z + kz) - K / 2;
    int iy = int(y + ky) - K / 2;
    int ix = int(x + kx) - K / 2;
    if (iz < 0 || iz >= D || iy < 0 || iy >= H || ix < 0 || ix >= S) continue;
    float a = X[(((b * D + iz) * H + iy) * S + ix) * C + c];
    float w = W[(kz * K + ky) * K + kx];
    volatile float product = a * w;
    sum = ds_add(sum, float2(product, fma(a, w, -product)));
  }
Y[i] = sum.x + sum.y;
"""


class CompensatedFloat32Operators:
    """Accuracy-oriented NCDHW operators using compensated FP32 arithmetic.

    Convolution supports dense kernels without dilation. Transpose convolution
    supports the non-overlapping 2-cubed, stride-two decoder. GroupNorm uses
    two centered passes and admits at most 2**24 values per group, so the group
    size has an exact float32 representation. All kernels execute on Metal.
    """

    def __init__(self, mx: MlxRuntime, groups: int, eps: float) -> None:
        """Compile the kernels for one GroupNorm configuration.

        Args:
            mx: MLX module.
            groups: GroupNorm group count used by ``norm``.
            eps: Variance epsilon, carried as a float32 high part plus a float32
                remainder so the configured double value is not rounded away.
        """
        self.mx = mx
        self.groups = groups
        self._carry = False
        high = struct.unpack("f", struct.pack("f", eps))[0]
        self.eps = mx.array([high, eps - high], dtype=mx.float32)
        namespace = next(_NAMESPACES)
        self.kernels = {
            name: cast(
                MetalKernel,
                mx.fast.metal_kernel(
                    name=f"medmlx_compensated_{namespace}_{name}",
                    input_names=inputs,
                    output_names=["Y"],
                    source=source,
                    header=_HEADER,
                ),
            )
            for name, inputs, source in [
                ("conv", ["X", "W", "BIAS"], _CONV),
                ("deconv", ["X", "W", "BIAS"], _DECONV),
                ("norm", ["X", "W", "B", "EPS"], _NORM),
                ("batch_norm", ["X", "MEAN", "VAR", "W", "B", "EPS"], _BN),
                ("elu", ["X"], _ELU),
                ("softmax", ["X"], _SOFTMAX),
                ("gaussian", ["X", "W"], _GAUSSIAN),
            ]
        }

    def _run(
        self,
        name: str,
        inputs: list[Array],
        shape: tuple[int, ...],
        *,
        workgroups: int | None = None,
        **constants: int,
    ) -> Array:
        if self._carry and name not in {"conv", "deconv", "norm", "add", "relu"}:
            raise ValueError(f"Paired hidden-layer precision does not support {name}")
        if any(value.dtype != self.mx.float32 for value in inputs):
            raise ValueError("Compensated operators require float32 tensors")
        factor = 2 if self._carry else 1
        size = math.prod(shape) // factor
        threads = workgroups * 256 if workgroups is not None else ((size + 255) // 256) * 256
        return self.kernels[name](
            inputs=inputs,
            template=[
                *constants.items(),
                ("CARRY", self._carry),
                ("INPUT_SIZE", math.prod(inputs[0].shape) // factor),
                ("OUTPUT_SIZE", size),
            ],
            grid=(threads, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[self.mx.float32],
            stream=self.mx.gpu,
        )[0]

    def conv(
        self,
        values: Array,
        weight: Array,
        bias: Array | None,
        *,
        padding: int | Sequence[int],
        stride: int | Sequence[int],
    ) -> Array:
        """Accumulate each dense kernel product and its FMA residual."""
        require_ncdhw(values, name="compensated convolution input")
        pads = (padding,) * 3 if isinstance(padding, int) else padding
        steps = (stride,) * 3 if isinstance(stride, int) else stride
        batch, ci, di, hi, si = (int(size) for size in values.shape)
        co, _, kz, ky, kx = (int(size) for size in weight.shape)
        spatial = tuple(
            (int(n) + 2 * p - int(k)) // s + 1
            for n, p, k, s in zip(values.shape[2:], pads, weight.shape[2:], steps, strict=True)
        )
        initial = self.mx.zeros((co,), dtype=self.mx.float32) if bias is None else bias
        return self._run(
            "conv",
            [values, weight, initial],
            (batch, co, *spatial),
            BATCH=batch // (2 if self._carry else 1),
            CI=ci,
            CO=co,
            DI=di,
            HI=hi,
            SI=si,
            D=spatial[0],
            H=spatial[1],
            S=spatial[2],
            KZ=kz,
            KY=ky,
            KX=kx,
            PZ=pads[0],
            PY=pads[1],
            PX=pads[2],
            SZ=steps[0],
            SY=steps[1],
            SX=steps[2],
        )

    def conv_transpose3d(self, values: Array, weight: Array, bias: Array) -> Array:
        """Non-overlapping 2x decoder with compensated channel accumulation."""
        require_ncdhw(values, name="compensated transpose-convolution input")
        if tuple(weight.shape[2:]) != (2, 2, 2):
            raise ValueError("Compensated transpose convolution requires a 2x2x2 kernel")
        batch, ci, di, hi, si = (int(size) for size in values.shape)
        co = int(weight.shape[1])
        shape = (batch, co, di * 2, hi * 2, si * 2)
        return self._run(
            "deconv",
            [values, weight, bias],
            shape,
            BATCH=batch // (2 if self._carry else 1),
            CI=ci,
            CO=co,
            DI=di,
            HI=hi,
            SI=si,
            D=shape[2],
            H=shape[3],
            S=shape[4],
        )

    def norm(self, values: Array, weight: Array, bias: Array) -> Array:
        """Two-pass centered moments and compensated reciprocal-root affine."""
        require_ncdhw(values, name="compensated GroupNorm input")
        batch, channels = (int(size) for size in values.shape[:2])
        if channels % self.groups:
            raise ValueError("Compensated GroupNorm channels must be divisible by groups")
        workgroups = batch // (2 if self._carry else 1) * self.groups
        size = int(values.size) // (batch * self.groups)
        if size > 2**24:
            raise ValueError("Compensated GroupNorm admits at most 2**24 values per group")
        return self._run(
            "norm",
            [values, weight, bias, self.eps],
            tuple(values.shape),
            workgroups=workgroups,
            N=size,
            GROUPS=self.groups,
            CHANNELS=channels,
            SPATIAL=math.prod(values.shape[2:]),
        )

    def batch_norm_ndhwc(
        self, values: Array, mean: Array, variance: Array, weight: Array, bias: Array
    ) -> Array:
        """Frozen channels-last BatchNorm using the configured epsilon pair."""
        return self._run(
            "batch_norm",
            [values, mean, variance, weight, bias, self.eps],
            tuple(values.shape),
            SIZE=int(values.size),
            C=int(values.shape[-1]),
        )

    def elu(self, values: Array) -> Array:
        """Float-pair ELU with range-reduced exponential, rounded once to FP32."""
        return self._run("elu", [values], tuple(values.shape), SIZE=int(values.size))

    def softmax_last(self, values: Array) -> Array:
        """Max-shifted, compensated exponent/sum/division over the last axis."""
        channels = int(values.shape[-1])
        rows = int(values.size) // channels
        return self._run(
            "softmax",
            [values],
            tuple(values.shape),
            workgroups=(rows + 255) // 256,
            ROWS=rows,
            C=channels,
        )

    def gaussian_ndhwc(self, values: Array, kernel: Array) -> Array:
        """Channelwise cubic Gaussian stencil with zero padding and FP32 outputs."""
        batch, depth, height, width, channels = (int(size) for size in values.shape)
        size = int(kernel.shape[0])
        if tuple(kernel.shape) != (size, size, size) or size % 2 != 1:
            raise ValueError("Compensated Gaussian convolution requires an odd cubic kernel")
        return self._run(
            "gaussian",
            [values, kernel],
            tuple(values.shape),
            BATCH=batch,
            D=depth,
            H=height,
            S=width,
            C=channels,
            K=size,
        )


_PAIRED_ADD = r"""
uint i = thread_position_in_grid.x;
if (i >= OUTPUT_SIZE) return;
float2 value = ds_add(ds_load(X, i, INPUT_SIZE, true), ds_load(OTHER, i, INPUT_SIZE, true));
ds_store(Y, i, OUTPUT_SIZE, value, true);
"""

_PAIRED_RELU = r"""
uint i = thread_position_in_grid.x;
if (i >= OUTPUT_SIZE) return;
float2 value = ds_load(X, i, INPUT_SIZE, true);
if (value.x < 0 || (value.x == 0 && value.y < 0)) value = float2(0, 0);
ds_store(Y, i, OUTPUT_SIZE, value, true);
"""


class CompensatedFloat48Operators(CompensatedFloat32Operators):
    """Carry both FP32 words between linear, normalization and residual layers.

    The leading batch dimension packs all high words followed by all low words.
    No float32 rounding occurs at hidden-layer boundaries. Only the declared
    convolution, GroupNorm, transpose-convolution, addition and ReLU are supported.
    """

    def __init__(self, mx: MlxRuntime, groups: int, eps: float) -> None:
        super().__init__(mx, groups, eps)
        self._carry = True
        namespace = next(_NAMESPACES)
        for name, inputs, source in [
            ("add", ["X", "OTHER"], _PAIRED_ADD),
            ("relu", ["X"], _PAIRED_RELU),
        ]:
            self.kernels[name] = cast(
                MetalKernel,
                mx.fast.metal_kernel(
                    name=f"medmlx_paired_{namespace}_{name}",
                    input_names=inputs,
                    output_names=["Y"],
                    source=source,
                    header=_HEADER,
                ),
            )

    def pack(self, values: Array) -> Array:
        """Represent an FP32 input exactly with a zero low word."""
        if values.dtype != self.mx.float32:
            raise ValueError("Paired network inputs must be float32")
        return self.mx.concatenate(
            [values, self.mx.zeros(values.shape, dtype=self.mx.float32)], axis=0
        )

    def round(self, values: Array) -> Array:
        """Round paired logits to one FP32 tensor at the public network boundary."""
        batch = int(values.shape[0]) // 2
        return values[:batch] + values[batch:]

    def add(self, values: Array, other: Array) -> Array:
        """Add two paired tensors of identical shape, keeping both words.

        Raises:
            ValueError: The shapes differ.
        """
        if values.shape != other.shape:
            raise ValueError("Paired residual addition requires matching tensor shapes")
        return self._run("add", [values, other], tuple(values.shape))

    def relu(self, values: Array) -> Array:
        """ReLU on a paired tensor, zeroing both words where the pair is negative."""
        return self._run("relu", [values], tuple(values.shape))
