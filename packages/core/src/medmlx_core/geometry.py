"""Shared NumPy affine and sampling-grid geometry for MONAI model ports.

The Torch reference is the CPU build of PyTorch 2.14.1 for macOS arm64 (Accelerate
BLAS, Clang with floating-point contraction). Its kernels fuse ``a * b + c`` into one
rounding wherever the source writes it as one expression, and the helpers here
reproduce those roundings in float64 NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import cast

import numpy as np
import numpy.typing as npt

type Float64Array = npt.NDArray[np.float64]
AFFINE_TOL = 1e-3
# 2**27 + 1, Veltkamp's splitting constant for float64.
_SPLIT = 134217729.0

# The underscored names are imported directly by the SegResNet, Swin UNETR and
# MuscleMap ports; ``torch_mm``, ``torch_bmm`` and ``linspace_from_neg_one`` are
# their public names.
__all__ = [
    "AFFINE_TOL", "Float64Array", "_bmm", "_fma", "_linspace_from_neg_one", "_mm",
    "affine_to_spacing", "compute_shape_offset", "fma", "linspace_from_neg_one",
    "to_affine_nd", "torch_bmm", "torch_mm", "zoom_affine",
]


def to_affine_nd(rank: int, affine: npt.ArrayLike) -> Float64Array:
    """``monai.data.utils.to_affine_nd`` for an integer rank."""
    source = np.array(affine, dtype=np.float64, copy=True)
    result = np.eye(rank + 1, dtype=np.float64)
    size = max(min(len(result) - 1, len(source) - 1), 1)
    result[:size, :size] = source[:size, :size]
    if size > 1:
        result[:size, -1] = source[:size, -1]
    return result


def affine_to_spacing(affine: Float64Array, rank: int) -> Float64Array:
    """``monai.data.utils.affine_to_spacing`` with zeros suppressed to one."""
    part = np.asarray(affine[:rank, :rank], dtype=np.float64)
    spacing = np.sqrt(np.sum(part * part, axis=0))
    spacing[spacing == 0] = 1.0
    return spacing


def zoom_affine(affine: Float64Array, scale: Float64Array) -> Float64Array:
    """``monai.data.utils.zoom_affine`` with ``diagonal=False``."""
    affine = np.array(affine, dtype=float, copy=True)
    scale_np = np.array(scale, dtype=float, copy=True)
    size = len(affine) - 1
    norm = affine_to_spacing(affine, size)
    if len(scale_np) < size:
        scale_np = np.append(scale_np, norm[len(scale_np) :])
    scale_np = scale_np[:size]
    scale_np = np.where(scale_np > 0, scale_np, norm)
    scale_np[scale_np == 0] = 1.0
    rzs = affine[:-1, :-1]
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    signed = np.sign(np.diag(zs)) * np.abs(scale_np)
    result = np.eye(len(affine))
    result[:-1, :-1] = rotation @ np.diag(signed)
    return result


def compute_shape_offset(
    spatial_shape: Sequence[int],
    in_affine: Float64Array,
    out_affine: Float64Array,
) -> tuple[npt.NDArray[np.int_], Float64Array]:
    """``monai.data.utils.compute_shape_offset`` with ``scale_extent=False``."""
    shape = np.array(tuple(spatial_shape), copy=True, dtype=float)
    rank = len(shape)
    in_affine_ = to_affine_nd(rank, in_affine)
    out_affine_ = to_affine_nd(rank, out_affine)
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners: Float64Array = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((rank, -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners_out = np.linalg.solve(out_affine_, in_affine_) @ corners
    corners = in_affine_ @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(cast(Float64Array, np.ptp(corners_out, axis=1)) + 1.0)
    offset = None
    for index in range(corners.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, index : index + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners[:-1, index]
            break
    if offset is None:
        offset = (
            in_affine_[:-1, :-1] @ (shape / 2.0)
            + in_affine_[:-1, -1]
            - out_affine_[:-1, :-1] @ (out_shape / 2.0)
        )
    return out_shape.astype(int, copy=False), offset


def torch_mm(left: Float64Array, right: Float64Array) -> Float64Array:
    """A MetaTensor affine update, ``torch.mm`` of two 4x4 float64 matrices.

    Accelerate computes each entry as ``fma(a2, b2, a0 * b0) + fma(a3, b3, a1 * b1)``;
    NumPy's ``@`` on the same host rounds identically. Other shapes use ``@``.
    """
    if left.shape != (4, 4) or right.shape != (4, 4):
        return left @ right
    result = np.empty((4, 4), dtype=np.float64)
    for row in range(4):
        a = [float(value) for value in left[row]]
        for column in range(4):
            b = [float(value) for value in right[:, column]]
            result[row, column] = _fma(a[2], b[2], a[0] * b[0]) + _fma(a[3], b[3], a[1] * b[1])
    return result


def torch_bmm(left: Float64Array, right: Float64Array) -> Float64Array:
    """ATen's small-matrix ``bmm`` kernel (below 400 multiply-adds) for one float64 batch.

    Each entry accumulates ``acc = fma(a[k], b[k], acc)`` for ascending ``k`` from
    ``a[0] * b[0]``. MONAI's ``to_norm_affine`` multiplies its 4x4 matrices this way.
    """
    rows, inner = left.shape
    columns = right.shape[1]
    result = np.empty((rows, columns), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            total = float(left[row, 0]) * float(right[0, column])
            for index in range(1, inner):
                total = _fma(float(left[row, index]), float(right[index, column]), total)
            result[row, column] = total
    return result


def _bmm(left: Float64Array, right: Float64Array) -> Float64Array:
    """Unfused sequential float64 accumulation of a small matrix product.

    This is the order the SegResNet and Swin UNETR ports were recorded with. It agrees
    with ``torch_bmm`` whenever each entry has at most one inexact product, as for
    axis-aligned resampling, and can differ in the last bit otherwise.
    """
    rows, inner = left.shape
    columns = right.shape[1]
    result = np.empty((rows, columns), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            total = 0.0
            for index in range(inner):
                total += float(left[row, index]) * float(right[index, column])
            result[row, column] = total
    return result


def _fma(left: float, right: float, addend: float) -> float:
    """Correctly rounded ``left * right + addend``."""
    return float(Fraction(left) * Fraction(right) + Fraction(addend))


def fma(left: npt.ArrayLike, right: npt.ArrayLike, addend: npt.ArrayLike) -> Float64Array:
    """Elementwise float64 ``left * right + addend`` rounded once to nearest even.

    Dekker's exact product and Knuth's two-sum carry the exact value; Boldo and
    Melquiond's rounding to odd (IEEE Trans. Computers 57(4), 2008) makes the final
    addition round correctly. Exact for finite operands whose partial products
    neither overflow nor fall into the subnormal range.
    """
    a, b, c = np.broadcast_arrays(
        np.asarray(left, np.float64), np.asarray(right, np.float64), np.asarray(addend, np.float64)
    )
    product = a * b
    a_high, a_low = _split(a)
    b_high, b_low = _split(b)
    product_error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    total, total_error = _two_sum(c, product)
    tail, tail_error = _two_sum(total_error, product_error)
    return total + _round_to_odd(tail, tail_error)


def _split(value: Float64Array) -> tuple[Float64Array, Float64Array]:
    """Veltkamp's split of float64 values into 26- and 27-bit halves."""
    scaled = _SPLIT * value
    high = scaled - (scaled - value)
    return high, value - high


def _two_sum(left: Float64Array, right: Float64Array) -> tuple[Float64Array, Float64Array]:
    """Knuth's error-free sum: ``(s, e)`` with ``s = fl(left + right)`` and ``s + e`` exact."""
    total = left + right
    shifted = total - left
    return total, (left - (total - shifted)) + (right - shifted)


def _round_to_odd(value: Float64Array, error: Float64Array) -> Float64Array:
    """``value + error`` rounded to odd, given ``value = fl(value + error)``.

    An inexact ``value`` with an even last bit moves one unit toward ``error``.
    """
    bits = value.view(np.int64).copy()
    move = (error != 0) & ((bits & 1) == 0)
    if move.any():
        # Sign-magnitude encoding: one more unit is one step away from zero.
        bits += np.where((error > 0) == (value > 0), 1, -1) * move
    return bits.view(np.float64)


def linspace_from_neg_one(steps: int) -> Float64Array:
    """ATen ``linspace(-1, 1, steps) * (steps - 1) / steps`` in float64.

    This is ``affine_grid``'s base axis for ``align_corners=False``. ATen evaluates
    ``start + step * i`` below the midpoint and ``end - step * (steps - i - 1)`` above
    it, each as one fused multiply-add.
    """
    if steps <= 1:
        return np.zeros(1, dtype=np.float64)
    step = (1.0 - (-1.0)) / (steps - 1)
    index = np.arange(steps, dtype=np.float64)
    values = np.where(
        index < steps // 2, fma(step, index, -1.0), fma(-step, (steps - 1) - index, 1.0)
    )
    return values * (steps - 1) / steps


_mm = torch_mm
_linspace_from_neg_one = linspace_from_neg_one
