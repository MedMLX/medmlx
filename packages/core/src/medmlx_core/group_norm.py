"""Native NCDHW group normalization with an optional fused SiLU.

Conditioned mode does not need the source-ordered Welford cascade in
``_source_half``. In NCDHW each ``(batch, group)`` is one contiguous run of
``C / G * D * H * W`` values, so statistics take three Metal dispatches:

1. Every threadgroup sums a ``CHUNK``-value slice of one group, shifted by the
   group's first value, accumulating the sum and sum of squares in FP32.
2. One thread per group converts each chunk to a mean and M2, merges chunks
   with Chan's parallel update, and writes each channel's FP32 scale and offset.
3. A vectorized pass applies ``scale * x + offset`` and, optionally,
   ``y / (1 + exp(-y))``, rounding once to the input dtype.

Shifting by a sample of the group keeps the per-chunk sums well conditioned;
the chunk merge avoids the global ``E[x^2] - E[x]^2`` cancellation. Results
match FP32 statistics to FP32 rounding. For FP16 inputs the single rounding
after SiLU differs from normalizing, rounding to FP16 and then applying SiLU.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

from medmlx_core.typing import MetalKernel, MlxRuntime

if TYPE_CHECKING:
    from mlx.core import array

_CHUNK = 16384
_THREADS = 256


class _GroupNormRuntime(MlxRuntime, Protocol):
    """The shared runtime plus the contiguous operation used by normalization."""

    def contiguous(self, a: array, /) -> array: ...


_PARTIAL = r"""
const uint chunk = threadgroup_position_in_grid.x, bg = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const ulong base = ulong(bg) * SIZE;
const float shift = float(x[base]);
const uint start = chunk * CHUNK, stop = min(start + CHUNK, uint(SIZE));
float s = 0.0f, ss = 0.0f;
for (uint i = start + tid * 4; i < stop; i += TG * 4) {
  const float4 v = float4(*reinterpret_cast<const device vec<T, 4>*>(x + base + i)) - shift;
  s += v.x + v.y + v.z + v.w;
  ss = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, ss))));
}
threadgroup float reduced[2][TG / 32];
s = simd_sum(s);
ss = simd_sum(ss);
if (thread_index_in_simdgroup == 0) {
  reduced[0][simdgroup_index_in_threadgroup] = s;
  reduced[1][simdgroup_index_in_threadgroup] = ss;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
  float a = 0.0f, b = 0.0f;
  for (uint i = 0; i < TG / 32; i++) { a += reduced[0][i]; b += reduced[1][i]; }
  partials[(bg * CHUNKS + chunk) * 2] = a;
  partials[(bg * CHUNKS + chunk) * 2 + 1] = b;
}
"""

_AFFINE = r"""
const uint bg = thread_position_in_grid.x;
if (bg >= B * G) return;
float count = 0.0f, mean = 0.0f, m2 = 0.0f;
for (uint c = 0; c < CHUNKS; c++) {
  const float n = float(min(uint(CHUNK), uint(SIZE) - c * CHUNK));
  const float s = partials[(bg * CHUNKS + c) * 2], ss = partials[(bg * CHUNKS + c) * 2 + 1];
  const float chunk_mean = s / n;
  const float total = count + n;
  const float delta = chunk_mean - mean;
  mean += delta * (n / total);
  m2 += max(ss - s * chunk_mean, 0.0f) + delta * delta * (count * n / total);
  count = total;
}
mean += float(x[ulong(bg) * SIZE]);
const float rstd = metal::precise::rsqrt(m2 / count + epsilon[0]);
const uint g = bg % G, b = bg / G;
for (uint j = 0; j < C / G; j++) {
  const uint c = g * (C / G) + j;
  const float scale = rstd * float(weight[c]);
  affine[(b * C + c) * 2] = scale;
  affine[(b * C + c) * 2 + 1] = float(bias[c]) - mean * scale;
}
"""

_APPLY = r"""
const uint vector = thread_position_in_grid.x;
const uint bc = thread_position_in_grid.y;
if (vector * 4 >= SPATIAL) return;
const ulong index = ulong(bc) * SPATIAL + vector * 4;
const float scale = affine[bc * 2], offset = affine[bc * 2 + 1];
float4 y = float4(*reinterpret_cast<const device vec<T, 4>*>(x + index)) * scale + offset;
if (SILU) y = y / (1.0f + metal::precise::exp(-y));
*reinterpret_cast<device vec<T, 4>*>(out + index) = vec<T, 4>(y);
"""


@lru_cache(maxsize=1)
def _kernels(mx: MlxRuntime) -> tuple[MetalKernel, MetalKernel, MetalKernel]:
    return (
        cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="medmlx_group_norm_partial",
                input_names=["x"],
                output_names=["partials"],
                source=_PARTIAL,
            ),
        ),
        cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="medmlx_group_norm_affine",
                input_names=["x", "partials", "weight", "bias", "epsilon"],
                output_names=["affine"],
                source=_AFFINE,
            ),
        ),
        cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="medmlx_group_norm_apply",
                input_names=["x", "affine"],
                output_names=["out"],
                source=_APPLY,
            ),
        ),
    )


def native_group_norm_eligible(x: array, groups: int) -> bool:
    """Whether NCDHW ``x`` fits the vectorized kernels: rank 5, ``groups | C``, ``4 | DHW``."""
    return (
        x.ndim == 5
        and int(x.shape[1]) % groups == 0
        and math.prod(int(size) for size in x.shape[2:]) % 4 == 0
    )


def group_norm_affine(
    x: array, weight: array, bias: array, *, groups: int, eps: float, mx: MlxRuntime
) -> array:
    """Fold NCDHW group statistics and the affine into per-channel FP32 pairs.

    Args:
        x: Contiguous ``(B, C, D, H, W)`` float16 or float32 input satisfying
            ``native_group_norm_eligible``.
        weight: Per-channel scale ``(C,)``.
        bias: Per-channel shift ``(C,)``.
        groups: Number of groups dividing ``C``.
        eps: Variance epsilon.
        mx: The imported ``mlx.core`` module.

    Returns:
        FP32 ``(B * C * 2)`` array; entries ``2 i`` and ``2 i + 1`` are the
        scale and offset that normalize channel ``i = b * C + c`` as
        ``x * scale + offset``.
    """
    batch, channels = int(x.shape[0]), int(x.shape[1])
    spatial = math.prod(int(size) for size in x.shape[2:])
    size = channels // groups * spatial
    chunks = -(-size // _CHUNK)
    partial_kernel, affine_kernel, _ = _kernels(mx)
    partials = partial_kernel(
        inputs=[x],
        template=[
            ("T", x.dtype),
            ("SIZE", size),
            ("CHUNK", _CHUNK),
            ("CHUNKS", chunks),
            ("TG", _THREADS),
        ],
        grid=(_THREADS * chunks, batch * groups, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(batch * groups * chunks * 2,)],
        output_dtypes=[mx.float32],
    )[0]
    epsilon = mx.array([eps], dtype=mx.float32)
    return affine_kernel(
        inputs=[x, partials, weight, bias, epsilon],
        template=[
            ("B", batch),
            ("G", groups),
            ("C", channels),
            ("SIZE", size),
            ("CHUNK", _CHUNK),
            ("CHUNKS", chunks),
        ],
        grid=(batch * groups, 1, 1),
        threadgroup=(min(32, batch * groups), 1, 1),
        output_shapes=[(batch * channels * 2,)],
        output_dtypes=[mx.float32],
    )[0]


def native_group_norm(
    x: array, weight: array, bias: array, *, groups: int, eps: float, silu: bool,
    mx: _GroupNormRuntime,
) -> array:
    """Group-normalize NCDHW ``x`` with per-channel affine, then optionally SiLU.

    Args:
        x: ``(B, C, D, H, W)`` float16 or float32 input satisfying
            ``native_group_norm_eligible``.
        weight: Per-channel scale ``(C,)``.
        bias: Per-channel shift ``(C,)``.
        groups: Number of groups dividing ``C``.
        eps: Variance epsilon.
        silu: Apply SiLU to the FP32 normalized value before rounding.
        mx: The imported ``mlx.core`` module.

    Returns:
        Contiguous array with ``x``'s shape and dtype.
    """
    # Every kernel reads x; one shared contiguous copy avoids three.
    x = mx.contiguous(x)
    affine = group_norm_affine(x, weight, bias, groups=groups, eps=eps, mx=mx)
    spatial = math.prod(int(size) for size in x.shape[2:])
    return _kernels(mx)[2](
        inputs=[x, affine],
        template=[("T", x.dtype), ("SPATIAL", spatial), ("SILU", silu)],
        grid=(spatial // 4, int(x.shape[0]) * int(x.shape[1]), 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[tuple(x.shape)],
        output_dtypes=[x.dtype],
    )[0]
