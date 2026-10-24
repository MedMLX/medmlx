"""Float32 convolution and normalization arithmetic shared by MLX engines."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from itertools import count
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array as Array

from medmlx_core.typing import MetalKernel, MlxRuntime

_KERNEL_NAMESPACES = count()

# Ordered FP32 convolution, derived from the Apache-2.0 register-blocked kernel in
# MedMLX/nnunet-mlx (src/nnunet_mlx/kernels/metal_fused_ordered_conv.py, 816e8e2).
# Rows are output positions (b, z, y, x; x fastest) and columns output channels.
# Each simdgroup owns TM x TN 8x8 fragments and gathers its A operands straight
# from the padded NCDHW input: row r and reduction index k = ch * KT + tap read
# X[rowbase(r) + offset(k)], with no im2col tile, threadgroup staging or barriers.
# Fragments stay in float2 registers using MLX's 8x8 lane layout. Every output
# keeps the arithmetic of the earlier K32-staged kernel: bias + 0 first, then one
# simdgroup MMA per K8 chunk in ascending k over the same zero-padded chunk count.
_CONV_SOURCE = """
constexpr int KT = KZ * KY * KX;
constexpr int K = CI * KT;
constexpr int CS = DP * HP * XP;
constexpr int SPATIAL = D * H * S;
constexpr int ROWS = BATCH * SPATIAL;
constexpr int CHUNKS = ((K + 31) / 32) * 4;
constexpr int FULL_CHUNKS = K / 8;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint qid = lane / 4;
uint fm = (qid & 4) + ((lane / 2) % 4);
uint fn = (qid & 2) * 2 + (lane % 2) * 2;
uint row0 = threadgroup_position_in_grid.y * (WM * TM * 8) + sg * (TM * 8);
uint col0 = threadgroup_position_in_grid.z * (TN * 8);
uint rowbase[TM];
bool rowok[TM];
#pragma unroll
for (int i = 0; i < TM; ++i) {
    uint row = row0 + i * 8 + fm;
    rowok[i] = row < ROWS;
    // Tail rows read a valid position; rows never mix in an MMA and are not stored.
    row = rowok[i] ? row : 0;
    uint x = row % S, y = (row / S) % H, z = (row / (S * H)) % D;
    uint b = row / SPATIAL;
    rowbase[i] = ((b * CI * DP + z * SZ) * HP + y * SY) * XP + x * SX;
}
float2 acc[TM][TN];
#pragma unroll
for (int j = 0; j < TN; ++j) {
    uint oc = col0 + j * 8 + fn;
    float2 init = float2(
        (FULL_OUTPUT || oc < CO) ? BIAS[oc] + 0.0f : 0.0f,
        (FULL_OUTPUT || oc + 1 < CO) ? BIAS[oc + 1] + 0.0f : 0.0f);
    #pragma unroll
    for (int i = 0; i < TM; ++i) acc[i][j] = init;
}
auto offset_of = [](int k) -> uint {
    uint ch = uint(k) / KT, tap = uint(k) % KT;
    return ch * CS + (tap / (KY * KX)) * (HP * XP) + ((tap / KX) % KY) * XP + tap % KX;
};
// Each lane advances its two reduction offsets by 8 through per-tap tables:
// moving from tap t to tap (t + 8) % KT adds the spatial delta plus channel wraps.
threadgroup int delta_table[KT];
threadgroup ushort next_table[KT];
for (uint t = thread_position_in_threadgroup.x; t < uint(KT); t += 32 * WM) {
    auto tap_offset = [](uint tap) -> int {
        return int((tap / (KY * KX)) * (HP * XP) + ((tap / KX) % KY) * XP + tap % KX);
    };
    uint next = (t + 8) % KT;
    delta_table[t] = int(((t + 8) / KT) * CS) + tap_offset(next) - tap_offset(t);
    next_table[t] = ushort(next);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint offset0 = offset_of(fn), offset1 = offset_of(fn + 1);
uint tap0 = fn % KT, tap1 = (fn + 1) % KT;
for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    int k0 = chunk * 8;
    float2 a[TM], w[TN];
    if (chunk < FULL_CHUNKS) {
        uint o0 = offset0, o1 = offset1;
        offset0 += delta_table[tap0];
        offset1 += delta_table[tap1];
        tap0 = next_table[tap0];
        tap1 = next_table[tap1];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            a[i] = float2(X[rowbase[i] + o0], X[rowbase[i] + o1]);
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint oc = col0 + j * 8 + fn;
            w[j] = float2(
                (FULL_OUTPUT || oc < CO) ? W[(k0 + fm) * CO + oc] : 0.0f,
                (FULL_OUTPUT || oc + 1 < CO) ? W[(k0 + fm) * CO + oc + 1] : 0.0f);
        }
    } else {
        // K tail: both operands are exact zeros past K, as in the staged kernel.
        int ka = k0 + fn, kb = ka + 1, kw = k0 + fm;
        uint o0 = offset_of(min(ka, K - 1)), o1 = offset_of(min(kb, K - 1));
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            a[i] = float2(ka < K ? X[rowbase[i] + o0] : 0.0f,
                          kb < K ? X[rowbase[i] + o1] : 0.0f);
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint oc = col0 + j * 8 + fn;
            w[j] = float2(
                (kw < K && (FULL_OUTPUT || oc < CO)) ? W[kw * CO + oc] : 0.0f,
                (kw < K && (FULL_OUTPUT || oc + 1 < CO)) ? W[kw * CO + oc + 1] : 0.0f);
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            metal::simdgroup_float8x8 am, wm, cm;
            am.thread_elements()[0] = a[i].x;
            am.thread_elements()[1] = a[i].y;
            wm.thread_elements()[0] = w[j].x;
            wm.thread_elements()[1] = w[j].y;
            cm.thread_elements()[0] = acc[i][j].x;
            cm.thread_elements()[1] = acc[i][j].y;
            simdgroup_multiply_accumulate(cm, am, wm, cm);
            acc[i][j] = float2(cm.thread_elements()[0], cm.thread_elements()[1]);
        }
    }
}
// NCDHW output: channel planes of SPATIAL positions per batch item.
#pragma unroll
for (int i = 0; i < TM; ++i) {
    uint row = row0 + i * 8 + fm;
    uint offset = (row / SPATIAL) * CO * SPATIAL + row % SPATIAL;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        uint oc = col0 + j * 8 + fn;
        if (rowok[i] && (FULL_OUTPUT || oc < CO)) Y[offset + oc * SPATIAL] = acc[i][j].x;
        if (rowok[i] && (FULL_OUTPUT || oc + 1 < CO)) Y[offset + (oc + 1) * SPATIAL] = acc[i][j].y;
    }
}
"""


_SOURCES = {
    "tile": (
        ["X"],
        ["M", "V"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*TILES*4) return;
        uint g=i/(TILES*4), t=i/4%TILES, lane=i%4;
        float m=0, v=0;
        uint count=min(16u, uint(N/4)-min(uint(N/4), t*16));
        for (uint j=0; j<count; ++j) {
          float x=X[g*N+t*64+j*4+lane], delta=x-m;
          m=fma(1.0f/float(j+1), delta, m);
          v=fma(delta, x-m, v);
        }
        M[i]=m; V[i]=v;
        """,
    ),
    "merge": (
        ["M", "V"],
        ["MO", "VO"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*PAIRS*4) return;
        uint g=i/(PAIRS*4), t=i/4%PAIRS, lane=i%4;
        uint a=(g*TILES+2*t)*4+lane, b=a+4;
        uint left=min(uint(COUNT), uint(NV)-2*t*COUNT);
        uint right=min(uint(COUNT), uint(NV)-(2*t+1)*COUNT);
        float delta=M[b]-M[a], cd=float(right)/float(left+right)*delta;
        MO[i]=M[a]+cd;
        VO[i]=fma(delta*float(left), cd, V[a]+V[b]);
        """,
    ),
    "join": (
        ["M", "V", "MA", "VA"],
        ["MO", "VO"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=GROUPS*4) return;
        float c=float(ADD)/float(COUNT+ADD), delta=MA[i]-M[i];
        float cd=c*delta;
        MO[i]=M[i]+cd;
        VO[i]=fma(delta*float(COUNT), cd, V[i]+VA[i]);
        """,
    ),
    "finish": (
        ["X", "M", "V"],
        ["MEAN", "VAR"],
        """
        uint g=thread_position_in_grid.x;
        if (g>=GROUPS) return;
        float m=0, v=0;
        uint count=0;
        for (uint i=NV*4; i<N; ++i) {
          float x=X[g*N+i], delta=x-m;
          ++count; m+=delta/float(count); v=fma(delta, x-m, v);
        }
        for (uint lane=0; lane<4; ++lane) {
          float c=float(NV)/float(count+NV), delta=M[g*4+lane]-m;
          m=fma(c, delta, m);
          float dc=delta*delta*c;
          v+=fma(dc, float(count), V[g*4+lane]);
          count+=NV;
        }
        MEAN[g]=m; VAR[g]=v/float(N);
        """,
    ),
    "coeff": (
        ["M", "V", "W", "B", "EPS"],
        ["S", "BIAS"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=BATCH*CHANNELS) return;
        uint c=i%CHANNELS, g=i/(CHANNELS/GROUPS);
        // Retain the low part of the configured epsilon sum.
        float v=V[g], hi=v+EPS[0], t=hi-v;
        float lo=(v-(hi-t))+(EPS[0]-t)+EPS[1];
        // Correct the reciprocal root using its float32 FMA residual.
        float r=rsqrt(hi), q=hi*r, qe=fma(hi,r,-q);
        float error=fma(-q,r,1.0f)-qe*r-lo*r*r;
        r=fma(0.5f*r,error,r);
        float s=r*W[c]; S[i]=s; BIAS[i]=fma(-s,M[g],B[c]);
        """,
    ),
    "affine": (
        ["X", "S", "B"],
        ["Y"],
        """
        uint i=thread_position_in_grid.x;
        if (i>=SIZE) return;
        uint c=i/SPATIAL;
        Y[i]=fma(S[c],X[i],B[c]);
        """,
    ),
}


class Float32Operators:
    """Metal convolution and GroupNorm kernels with a fixed float32 accumulation order.

    MLX's standard convolutions and reductions do not expose their accumulation
    order, so these kernels fix it: the convolution accumulates in ascending
    reduction index with one simdgroup MMA per 8-wide chunk, and the group
    statistics use four-lane Welford moments merged in a fixed tree. The
    constructor compiles the kernels, and arrays are float32 NCDHW. Inputs are
    not validated here, so callers must pass float32 arrays of matching shapes.
    """

    def __init__(self, mx: MlxRuntime, groups: int, eps: float) -> None:
        """Compile the kernels for one GroupNorm configuration.

        Args:
            mx: MLX module.
            groups: GroupNorm group count used by ``norm``; it must divide the
                channel count of the arrays passed to ``norm``.
            eps: Variance epsilon, carried as a float32 high part plus a float32
                remainder so the configured double value is not rounded away.
        """
        self.mx = mx
        self.groups = groups
        namespace = next(_KERNEL_NAMESPACES)
        # Preserve the configured epsilon while keeping all tensor math float32.
        high = struct.unpack("f", struct.pack("f", eps))[0]
        self.eps = mx.array([high, eps - high], dtype=mx.float32)
        self.kernels = {
            name: cast(
                MetalKernel,
                mx.fast.metal_kernel(
                    name=f"medmlx_float32_packed_{namespace}_{name}",
                    input_names=inputs,
                    output_names=outputs,
                    source=source,
                    header="#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\n",
                ),
            )
            for name, (inputs, outputs, source) in _SOURCES.items()
        }
        self.kernels["conv"] = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name=f"medmlx_float32_packed_{namespace}_conv",
                input_names=["X", "W", "BIAS"],
                output_names=["Y"],
                source=_CONV_SOURCE,
                header="#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\n",
                # Safe math keeps bias + 0 from dropping to bias for a negative-zero bias.
                compile_options={"math_mode": "safe"},
            ),
        )

    def _run(
        self, name: str, inputs: list[Array], shape: tuple[int, ...], **constants: int
    ) -> list[Array]:
        return self.kernels[name](
            inputs=inputs,
            template=list(constants.items()),
            grid=(math.prod(shape), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape] * len(_SOURCES[name][1]),
            output_dtypes=[self.mx.float32] * len(_SOURCES[name][1]),
        )

    def conv(
        self,
        values: Array,
        weight: Array,
        bias: Array | None,
        *,
        padding: int | Sequence[int],
        stride: int | Sequence[int],
    ) -> Array:
        """Convolve NCDHW ``values`` with a Torch ``[O, I, KD, KH, KW]`` weight.

        Args:
            values: Float32 ``(N, C, D, H, W)`` input.
            weight: Float32 weight with ``I == C``.
            bias: Optional float32 ``(O,)`` bias; None starts the sum from zero.
            padding: Zero padding, one int or a ``(D, H, W)`` sequence.
            stride: Stride, one int or a ``(D, H, W)`` sequence.

        Returns:
            Float32 ``(N, O, D', H', W')`` array. Dilation and grouping are not
            supported.
        """
        mx = self.mx
        pads = (padding,) * 3 if isinstance(padding, int) else padding
        steps = (stride,) * 3 if isinstance(stride, int) else stride
        padded = mx.pad(values, [(0, 0), (0, 0), *((p, p) for p in pads)])
        kernel = tuple(int(n) for n in weight.shape[2:])
        spatial = tuple(
            (int(n) + 2 * p - k) // s + 1
            for n, p, k, s in zip(values.shape[2:], pads, kernel, steps, strict=True)
        )
        batch, channels = map(int, values.shape[:2])
        out_channels = int(weight.shape[0])
        initial = mx.zeros((out_channels,), dtype=mx.float32) if bias is None else bias
        positions = batch * math.prod(spatial)
        # 4x4 fragments per simdgroup fit in registers; small maps use 2x4 so
        # enough threadgroups stay resident. Tiling does not change arithmetic.
        row_fragments = 4 if positions > 512 else 2
        column_fragments = min(4, (out_channels + 7) // 8)
        simdgroups = 4 if positions > 512 else 2
        block_rows, block_columns = simdgroups * row_fragments * 8, column_fragments * 8
        constants = dict(
            FULL_OUTPUT=out_channels % block_columns == 0,
            TM=row_fragments,
            TN=column_fragments,
            WM=simdgroups,
            BATCH=batch,
            CI=channels,
            CO=out_channels,
            D=spatial[0],
            H=spatial[1],
            S=spatial[2],
            KZ=kernel[0],
            KY=kernel[1],
            KX=kernel[2],
            SZ=steps[0],
            SY=steps[1],
            SX=steps[2],
            DP=int(padded.shape[2]),
            HP=int(padded.shape[3]),
            XP=int(padded.shape[4]),
        )
        return self.kernels["conv"](
            inputs=[padded, mx.transpose(weight, (1, 2, 3, 4, 0)), initial],
            template=list(constants.items()),
            grid=(
                32 * simdgroups,
                (positions + block_rows - 1) // block_rows,
                (out_channels + block_columns - 1) // block_columns,
            ),
            threadgroup=(32 * simdgroups, 1, 1),
            output_shapes=[(batch, out_channels, *spatial)],
            output_dtypes=[mx.float32],
        )[0]

    def _moments(self, values: Array) -> tuple[Array, Array]:
        groups = int(values.shape[0]) * self.groups
        size = int(values.size) // groups
        vectors = size // 4
        tiles = max(1, (vectors + 15) // 16)
        mean, variance = self._run(
            "tile", [values], (groups, tiles, 4), GROUPS=groups, TILES=tiles, N=size
        )
        count = 16
        remainders: list[tuple[Array, Array, int]] = []
        # Keep the rightmost partial blocks in ascending order, as in MONAI's CPU reference.
        while tiles > 1:
            if tiles % 2:
                remainders.append(
                    (mean[:, -1, :], variance[:, -1, :], min(count, vectors - (tiles - 1) * count))
                )
            pairs = tiles // 2
            mean, variance = self._run(
                "merge",
                [mean, variance],
                (groups, pairs, 4),
                GROUPS=groups,
                TILES=tiles,
                PAIRS=pairs,
                COUNT=count,
                NV=vectors,
            )
            count *= 2
            tiles = pairs
        remainders.append((mean[:, 0, :], variance[:, 0, :], min(count, vectors)))
        mean, variance, count = remainders[0]
        for added_mean, added_variance, added_count in remainders[1:]:
            mean, variance = self._run(
                "join",
                [mean, variance, added_mean, added_variance],
                (groups, 4),
                GROUPS=groups,
                COUNT=count,
                ADD=added_count,
            )
            count += added_count
        mean, variance = self._run(
            "finish", [values, mean, variance], (groups,), GROUPS=groups, N=size, NV=vectors
        )
        return mean, variance

    def norm(self, values: Array, weight: Array, bias: Array) -> Array:
        """GroupNorm over ``self.groups`` channel groups of NCDHW ``values``.

        ``weight`` and ``bias`` are per-channel ``(C,)`` scale and shift. The result
        has the shape of ``values``.
        """
        mean, variance = self._moments(values)
        batch, channels = map(int, values.shape[:2])
        scale, offset = self._run(
            "coeff",
            [mean, variance, weight, bias, self.eps],
            (batch * channels,),
            BATCH=batch,
            CHANNELS=channels,
            GROUPS=self.groups,
        )
        return self._run(
            "affine",
            [values, scale, offset],
            tuple(values.shape),
            SIZE=int(values.size),
            SPATIAL=math.prod(values.shape[2:]),
        )[0]
