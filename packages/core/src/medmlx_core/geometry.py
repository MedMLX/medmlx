"""Shared NumPy affine and sampling-grid geometry for MONAI model ports."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import cast

import numpy as np
import numpy.typing as npt

type Float64Array = npt.NDArray[np.float64]
AFFINE_TOL = 1e-3

# These helpers are imported directly by the SegResNet and Swin UNETR ports.
__all__ = [
    "AFFINE_TOL", "Float64Array", "_bmm", "_fma", "_linspace_from_neg_one", "_mm",
    "affine_to_spacing", "compute_shape_offset", "to_affine_nd", "zoom_affine",
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


def _mm(left: Float64Array, right: Float64Array) -> Float64Array:
    """A MetaTensor affine update, ``torch.mm`` of two 4x4 float64 matrices.

    The reference Torch build's MKL computes each entry as
    ``fma(a2, b2, a0 * b0) + fma(a3, b3, a1 * b1)``.
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


def _bmm(left: Float64Array, right: Float64Array) -> Float64Array:
    """Torch's small-matrix ``bmm`` kernel: unfused sequential float64 accumulation."""
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


def _linspace_from_neg_one(steps: int) -> Float64Array:
    """ATen ``linspace(-1, 1, steps) * (steps - 1) / steps`` in float64.

    ATen evaluates ``start + step * i`` below the midpoint and
    ``end - step * (steps - i - 1)`` above it, each as one fused multiply-add.
    """
    if steps <= 1:
        return np.zeros(1, dtype=np.float64)
    step = (1.0 - (-1.0)) / (steps - 1)
    halfway = steps // 2
    values = np.array(
        [
            _fma(step, index, -1.0) if index < halfway else _fma(-step, steps - index - 1, 1.0)
            for index in range(steps)
        ],
        dtype=np.float64,
    )
    return values * (steps - 1) / steps
