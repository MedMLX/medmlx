"""Host NumPy ports of the MONAI 1.6 transforms that MedMLX model bundles share.

These functions reproduce, without Torch, MONAI or nibabel:

* the NIfTI reader's header correction (``correct_nifti_header_if_necessary``);
* ``Orientation`` to axis codes, and its inverse through ``axis_codes``;
* ``Spacing`` and ``SpatialResample`` with ``mode="bilinear"`` or ``"nearest"``,
  ``padding_mode="border"``, ``align_corners=False`` and ``dtype=float64`` (the
  bundle defaults). ``SpatialResample.inverse``, which ``Invertd`` replays, is the
  same resampling back to the recorded affine and shape;
* ``ScaleIntensityRange(clip=True)``;
* ``CropForeground``'s box (``select_fn=x > 0``, ``allow_smaller=True``);
* ``AsDiscrete(argmax=True)``.

Conventions: images are channel first, ``(C, D, H, W)``, whose spatial axes are the
array's storage axes (``D``, ``H``, ``W`` are Torch's names for axes 1, 2 and 3).
Affines are float64 4x4 voxel-to-world matrices; MONAI ignores units and so do these
functions, which the runtimes use in millimeters. Resampled images are float32.

Exactness: the reference is MONAI 1.6.1 with PyTorch 2.14.1 on macOS arm64 (CPU,
Accelerate BLAS). Orientation, scaling, boxes, argmax and header correction are
bit-identical to it. Resampled voxels and affines are bit-identical for every grid,
axis-aligned or oblique, with one exception measured on this host: on rare oblique
output grids (seen at 945 and 1,377 voxels), Accelerate rounds ``affine_grid``'s last
grid row in a different order, which can move one voxel's float64 sampling position
by one unit in the last place.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from medmlx_core.errors import InvalidInputError
from medmlx_core.geometry import (
    AFFINE_TOL,
    affine_to_spacing,
    compute_shape_offset,
    fma,
    linspace_from_neg_one,
    to_affine_nd,
    torch_bmm,
    torch_mm,
    zoom_affine,
)

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
type Shape3 = tuple[int, int, int]
type Mode = Literal["bilinear", "nearest"]
type HeaderFloat = type[np.float32] | type[np.float64]

__all__ = [
    "AFFINE_TOL",
    "Mode",
    "ResamplePlan",
    "axis_codes",
    "channel_argmax",
    "correct_nifti_header",
    "foreground_box",
    "orient",
    "orientation_transform",
    "resample_plan",
    "scale_intensity_range",
    "spacing",
    "spacing_grid",
    "spatial_resample",
]

_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))
_THREADS = min(8, os.cpu_count() or 1)
# Output values (voxels times channels) per resampling block. Each block in flight
# holds, independent of the volume, about 45 bytes per value on an axis-aligned grid
# and, with its per-voxel coordinate tables, 85 (three channels) to 190 (one channel)
# on an oblique grid. Argmax blocks are larger: their result keeps one channel of
# all, and their per-block reductions cost less over more voxels.
_BLOCK_VALUES = 1 << 18
_ARGMAX_BLOCK_VALUES = 1 << 20
_SPLIT = 134217729.0  # 2**27 + 1, Veltkamp's splitting constant for float64
# ATen's corner order (tnw, tne, tsw, tse, bnw, bne, bsw, bse) as (z, y, x) sides.
_CORNERS = tuple(product((0, 1), repeat=3))


# --- NIfTI header correction ---------------------------------------------------------


def correct_nifti_header(
    affine: npt.ArrayLike,
    zooms: Sequence[float],
    sform: npt.ArrayLike,
    sform_code: int,
    qform: npt.ArrayLike,
    qform_code: int,
    *,
    header_float: HeaderFloat = np.float32,
) -> tuple[Float64Array, Float64Array]:
    """MONAI's ``correct_nifti_header_if_necessary`` for a 3D NIfTI, on plain arrays.

    Args:
        affine: The image affine nibabel selected (``img.affine``), 4x4.
        zooms: ``header.get_zooms()`` for the three spatial axes.
        sform, qform: ``header.get_sform()`` and ``header.get_qform()``, whatever
            their codes, in the header's own units.
        sform_code, qform_code: The header codes.
        header_float: The header's storage type, ``np.float32`` for NIfTI-1 and
            ``np.float64`` for NIfTI-2. It rounds the fields nibabel writes.

    Returns:
        ``(affine, zooms)`` after the correction: the ``img.affine`` and
        ``header.get_zooms()`` MONAI's nibabel reader continues with. When the zooms
        disagree with the affine (``np.allclose``), a form whose spacing matches the
        zooms replaces the other (the sform when ``sform_code`` is nonzero, else the
        qform when ``qform_code`` is nonzero), following nibabel's ``set_sform`` and
        ``set_qform`` storage; otherwise the zooms are reset to the affine spacing.

    Raises:
        InvalidInputError: The sform cannot be stored as a qform quaternion.
    """
    current = np.array(affine, dtype=np.float64, copy=True)
    pixdim = np.asarray(zooms, dtype=np.float64)[:3]
    if np.allclose(pixdim, affine_to_spacing(current, 3)):
        return current, pixdim.copy()
    sform_ = np.asarray(sform, dtype=np.float64)
    qform_ = np.asarray(qform, dtype=np.float64)
    sform_matches = np.allclose(affine_to_spacing(sform_, 3), pixdim)
    qform_matches = np.allclose(affine_to_spacing(qform_, 3), pixdim)
    if sform_code != 0:
        if sform_matches:
            return current, pixdim.copy()
        if qform_matches:
            # ``set_sform(qform)`` stores the rows in the header's float type and the
            # sform stays the best affine.
            stored = np.eye(4, dtype=np.float64)
            stored[:3] = qform_[:3].astype(header_float)
            return stored, pixdim.copy()
    if qform_code != 0:
        if qform_matches:
            return current, pixdim.copy()
        if sform_matches:
            # Reached only with sform_code 0, so the new qform is the best affine.
            return _stored_qform(sform_, header_float)
    # ``header.set_zooms`` leaves ``img.affine`` unchanged.
    return current, affine_to_spacing(current, 3).astype(header_float).astype(np.float64)


def _stored_qform(
    affine: Float64Array, header_float: HeaderFloat
) -> tuple[Float64Array, Float64Array]:
    """nibabel ``Nifti1Header.set_qform(affine)`` followed by ``get_qform()``.

    Returns the qform affine and the zooms ``set_qform`` writes. The operations and
    their types follow nibabel 5.4, whose quaternion math uses its widest float type,
    float64 on macOS arm64, so the products take NumPy's float64 BLAS path.
    """
    translation = affine[:3, 3]
    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    rotation = np.asarray(rzs / zooms, dtype=np.float64)
    if np.linalg.det(rotation) > 0:
        qfac = 1.0
    else:
        qfac = -1.0
        rotation[:, -1] *= -1
    left, _, right = np.linalg.svd(rotation)
    quaternion = _mat2quat(np.dot(left, right))
    offsets = translation.astype(header_float)
    pixdim = np.array([qfac, *zooms], dtype=header_float)
    bcd = quaternion[1:].astype(header_float)
    eps = np.finfo(np.float32).eps if header_float is np.float32 else np.finfo(np.float64).eps
    threshold = 3 * float(eps)
    xyz = np.asarray(bcd, dtype=np.float64)
    w2 = 1.0 - xyz @ xyz
    if np.abs(w2) < threshold:
        w = np.float64(0)
    elif w2 < 0:
        raise InvalidInputError(f"NIfTI sform cannot be stored as a qform quaternion: {affine}")
    else:
        w = np.sqrt(w2)
    matrix = _quat2mat(np.concatenate((np.asarray([w], dtype=np.float64), xyz)))
    voxel = pixdim[1:4].copy()
    voxel[-1] *= pixdim[0]
    qform = np.eye(4, dtype=np.float64)
    qform[:3, :3] = np.dot(matrix, np.diag(voxel))
    qform[:3, 3] = offsets
    return qform, pixdim[1:4].astype(np.float64)


def _quat2mat(quaternion: Float64Array) -> Float64Array:
    """nibabel ``quaternions.quat2mat``."""
    w, x, y, z = (quaternion[index] for index in range(4))
    norm = w * w + x * x + y * y + z * z
    if norm < np.finfo(np.float64).eps:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / norm
    big_x, big_y, big_z = x * s, y * s, z * s
    w_x, w_y, w_z = w * big_x, w * big_y, w * big_z
    x_x, x_y, x_z = x * big_x, x * big_y, x * big_z
    y_y, y_z, z_z = y * big_y, y * big_z, z * big_z
    return np.array(
        [
            [1.0 - (y_y + z_z), x_y - w_z, x_z + w_y],
            [x_y + w_z, 1.0 - (x_x + z_z), y_z - w_x],
            [x_z - w_y, y_z + w_x, 1.0 - (x_x + y_y)],
        ],
        dtype=np.float64,
    )


def _mat2quat(matrix: Float64Array) -> Float64Array:
    """nibabel ``quaternions.mat2quat``: the largest eigenvector, with ``w >= 0``."""
    qxx, qyx, qzx, qxy, qyy, qzy, qxz, qyz, qzz = (float(value) for value in matrix.flat)
    k = (
        np.array(
            [
                [qxx - qyy - qzz, 0, 0, 0],
                [qyx + qxy, qyy - qxx - qzz, 0, 0],
                [qzx + qxz, qzy + qyz, qzz - qxx - qyy, 0],
                [qyz - qzy, qzx - qxz, qxy - qyx, qxx + qyy + qzz],
            ]
        )
        / 3.0
    )
    values, vectors = np.linalg.eigh(k)
    quaternion = vectors[[3, 0, 1, 2], np.argmax(values)]
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


# --- Orientation ---------------------------------------------------------------------


def _io_orientation(affine: Float64Array) -> Float64Array:
    """nibabel ``io_orientation``: per voxel axis, the closest world axis and its sign."""
    rows, columns = affine.shape[0] - 1, affine.shape[1] - 1
    rzs = affine[:rows, :columns]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    zooms[zooms == 0] = 1
    rs = np.asarray(rzs / zooms, dtype=np.float64)
    left, singular, right = np.linalg.svd(rs, full_matrices=False)
    tolerance = singular.max() * max(rs.shape) * np.finfo(np.float64).eps
    keep = singular > tolerance
    rotation = np.dot(left[:, keep], right[keep])
    ornt = np.full((columns, 2), np.nan)
    for in_axis in range(columns):
        column = rotation[:, in_axis]
        if not np.allclose(column, 0):
            out_axis = int(np.argmax(np.abs(column)))
            ornt[in_axis] = (out_axis, -1 if column[out_axis] < 0 else 1)
            rotation[out_axis, :] = 0
    return ornt


def axis_codes(affine: npt.ArrayLike) -> str:
    """nibabel ``aff2axcodes`` with labels L/R, P/A and I/S, for example ``"RAS"``.

    ``Orientation.inverse`` reorients to ``axis_codes`` of the affine it recorded.

    Raises:
        InvalidInputError: A voxel axis has no world direction (singular affine).
    """
    ornt = _io_orientation(np.asarray(affine, dtype=np.float64))
    if np.isnan(ornt).any():
        raise InvalidInputError(f"Affine has a degenerate voxel axis: {affine}")
    return "".join(
        _LABELS[int(axis)][0 if flip == -1 else 1] for axis, flip in ornt.tolist()
    )


def orientation_transform(affine: npt.ArrayLike, axcodes: str) -> IntArray:
    """nibabel ``ornt_transform(io_orientation(affine), axcodes2ornt(axcodes))``.

    Returns an int64 ``(3, 2)`` array: for each input voxel axis, the output axis it
    becomes and ``-1`` when it is flipped.

    Raises:
        InvalidInputError: ``axcodes`` are not three distinct L/R, P/A, I/S codes, or
            the affine has a degenerate voxel axis.
    """
    source = _io_orientation(to_affine_nd(3, affine))
    if np.isnan(source).any():
        raise InvalidInputError(f"Affine has a degenerate voxel axis: {affine}")
    codes = axcodes[:3]
    target: list[tuple[int, int]] = []
    for code in codes:
        matches = [
            (axis, -1 if code == pair[0] else 1)
            for axis, pair in enumerate(_LABELS)
            if code in pair
        ]
        if not matches:
            raise InvalidInputError(f"Unknown axis code {code!r} in {axcodes!r}")
        target.append(matches[0])
    if len(target) != 3 or len({axis for axis, _ in target}) != 3:
        raise InvalidInputError(f"Axis codes must name three distinct axes, got {axcodes!r}")
    result = np.empty((3, 2), dtype=np.int64)
    for target_axis, (world_axis, target_flip) in enumerate(target):
        source_axis = int(np.flatnonzero(source[:, 0] == world_axis)[0])
        result[source_axis] = (target_axis, 1 if source[source_axis, 1] == target_flip else -1)
    return result


def orient[S: np.generic](
    data: npt.NDArray[S], affine: npt.ArrayLike, axcodes: str
) -> tuple[npt.NDArray[S], Float64Array]:
    """``Orientation(axcodes)`` on a channel-first ``(C, D, H, W)`` array of any dtype.

    Flips and permutes the spatial axes so their world directions follow ``axcodes``
    (``labels`` L/R, P/A, I/S); voxel values are not changed. Returns a C-contiguous
    array (``data`` itself when nothing moves) and the reoriented affine,
    ``affine @ inv_ornt_aff``, rounded as MONAI's NumPy product.

    Raises:
        InvalidInputError: ``data`` is not channel-first 3D, or as
            ``orientation_transform``.
    """
    _require_volume(data)
    affine_np = np.array(affine, dtype=np.float64, copy=True)
    ornt = orientation_transform(affine_np, axcodes)
    shape = np.asarray(data.shape[1:], dtype=np.float64)
    # nibabel ``inv_ornt_aff``: undo the reorder, then undo each flip about the center.
    undo_reorder = np.eye(4)[[*(int(axis) for axis in ornt[:, 0]), 3], :]
    undo_flip = np.diag([*(float(flip) for flip in ornt[:, 1]), 1.0])
    center = -(shape - 1) / 2.0
    undo_flip[:3, 3] = ornt[:, 1] * center - center
    xform = np.dot(undo_flip, undo_reorder)
    flips = tuple(axis + 1 for axis in range(3) if ornt[axis, 1] == -1)
    result = np.flip(data, flips) if flips else data
    order = [0, *(int(axis) + 1 for axis in np.argsort(ornt[:, 0]))]
    if order != [0, 1, 2, 3]:
        result = result.transpose(order)
    return np.ascontiguousarray(result), torch_mm(affine_np, to_affine_nd(3, xform))


# --- Spacing and SpatialResample ------------------------------------------------------


def spacing_grid(
    affine: npt.ArrayLike, spatial_shape: Sequence[int], pixdim: Sequence[float]
) -> tuple[Float64Array, Shape3]:
    """``Spacing(pixdim, diagonal=False, scale_extent=False)``'s output affine and shape.

    ``pixdim`` shorter than three is padded with its last value; an axis whose current
    spacing is within ``AFFINE_TOL`` of the target keeps it, and a nonpositive target
    keeps the current spacing. Rotation and axis signs are kept and shear is removed.
    """
    affine_ = to_affine_nd(3, affine)
    target = np.array(tuple(pixdim), dtype=np.float64)[:3].copy()
    if target.size < 3:
        target = np.append(target, [target[-1]] * (3 - target.size))
    current = affine_to_spacing(affine_, 3)
    for index, value in enumerate(current):
        wanted = target[index]
        target[index] = value if (wanted - AFFINE_TOL) <= value <= (wanted + AFFINE_TOL) else wanted
    new_affine = zoom_affine(affine_, target)
    shape, offset = compute_shape_offset(tuple(spatial_shape), affine_, new_affine)
    new_affine[:3, -1] = offset[:3]
    return new_affine, _shape3(shape.tolist())


@dataclass(frozen=True, slots=True)
class ResamplePlan:
    """How ``SpatialResample`` samples ``in_shape`` voxels onto ``out_shape`` voxels.

    Attributes:
        theta: ``(3, 4)`` float64 ``affine_grid`` theta. Rows and columns are in
            ``grid_sample``'s ``(x, y, z)`` order, input and output axes 2, 1, 0.
        affine: The output affine MONAI records, ``src_affine @ xform``.
        in_shape: Input spatial shape ``(D, H, W)``.
        out_shape: Output spatial shape.
    """

    theta: Float64Array
    affine: Float64Array
    in_shape: Shape3
    out_shape: Shape3

    @property
    def separable(self) -> bool:
        """Whether each sampling coordinate depends on one output axis (no rotation)."""
        return all(self.theta[row, column] == 0 for row in range(3) for column in range(3)
                   if row != column)

    def axis_indices(self) -> tuple[Float64Array, Float64Array, Float64Array]:
        """Border-clipped continuous input indices along axes 0, 1 and 2 for each output
        index along the same axis; exact for a separable plan.

        Raises:
            InvalidInputError: The plan is not separable.
        """
        if not self.separable:
            raise InvalidInputError("Per-axis sampling indices need an axis-aligned plan")
        indices = [
            # With zero off-diagonal terms the fused grid sum is ``base * t + offset``.
            _source_index(
                linspace_from_neg_one(self.out_shape[axis]) * self.theta[2 - axis, 2 - axis]
                + self.theta[2 - axis, 3],
                self.in_shape[axis],
            )
            for axis in range(3)
        ]
        return indices[0], indices[1], indices[2]

    def source_indices(
        self, d: npt.ArrayLike, h: npt.ArrayLike, w: npt.ArrayLike
    ) -> tuple[Float64Array, Float64Array, Float64Array]:
        """Border-clipped continuous input indices ``(i0, i1, i2)`` of output voxels.

        ``d``, ``h`` and ``w`` are broadcastable integer index arrays along output axes
        0, 1 and 2. The values are ``grid_sample``'s source indices before ``floor``
        (bilinear) or ``nearbyint`` (nearest).
        """
        bases = (
            linspace_from_neg_one(self.out_shape[2])[np.asarray(w)],
            linspace_from_neg_one(self.out_shape[1])[np.asarray(h)],
            linspace_from_neg_one(self.out_shape[0])[np.asarray(d)],
        )
        x, y, z = (
            _source_index(_grid_row(self.theta[row], *bases), self.in_shape[2 - row])
            for row in range(3)
        )
        return z, y, x


def _grid_row(
    theta: Float64Array, x: Float64Array, y: Float64Array, z: Float64Array
) -> Float64Array:
    """One ``affine_grid`` coordinate: ATen's ``bmm`` sums ``x, y, z, 1`` with fused
    multiply-adds in that order."""
    return fma(z, theta[2], fma(y, theta[1], x * theta[0])) + theta[3]


def _source_index(coordinate: Float64Array, size: int) -> Float64Array:
    """``grid_sampler_compute_source_index``: ``align_corners=False``, border padding.

    Clang contracts the unnormalization ``(coordinate + 1) * size - 1`` into one
    fused multiply-add before the halving.
    """
    index = fma(coordinate + 1.0, float(size), -1.0) / 2
    return np.minimum(np.float64(size - 1), np.maximum(index, 0.0))


def resample_plan(
    src_affine: npt.ArrayLike,
    dst_affine: npt.ArrayLike,
    in_shape: Sequence[int],
    out_shape: Sequence[int],
) -> ResamplePlan | None:
    """MONAI ``spatial_resample``'s transform from ``dst_affine`` voxels to ``src_affine``
    voxels, or ``None`` when MONAI returns the input unresampled.

    MONAI skips resampling when the shapes are ``np.allclose`` and either the affines
    or their relative transform and the identity are ``np.allclose`` with
    ``atol=AFFINE_TOL``.

    Raises:
        InvalidInputError: ``src_affine`` is singular or a shape is not three positive
            integers.
    """
    source_shape, target_shape = _shape3(in_shape), _shape3(out_shape)
    source = to_affine_nd(3, src_affine)
    target = to_affine_nd(3, dst_affine)
    try:
        xform = to_affine_nd(3, np.linalg.solve(source, target))
    except np.linalg.LinAlgError as error:
        raise InvalidInputError(f"Source affine is not invertible: {source}") from error
    same_size = np.allclose(target_shape, source_shape)
    if (np.allclose(source, target, atol=AFFINE_TOL) and same_size) or (
        np.allclose(xform, np.eye(4), atol=AFFINE_TOL) and same_size
    ):
        return None
    full = np.array(src_affine, dtype=np.float64, copy=True)
    affine = torch_mm(full, to_affine_nd(len(full) - 1, xform))
    # ``to_norm_affine``: Torch batched products of the normalizing transforms.
    theta = torch_bmm(
        torch_bmm(_normalize_transform(source_shape), xform),
        np.asarray(np.linalg.inv(_normalize_transform(target_shape)), dtype=np.float64),
    )
    reverse = [2, 1, 0]
    theta[:3] = theta[reverse]
    theta[:, :3] = theta[:, reverse]
    return ResamplePlan(theta[:3].copy(), affine, source_shape, target_shape)


def _normalize_transform(shape: Shape3) -> Float64Array:
    """``monai.networks.utils.normalize_transform`` for ``align_corners=False``."""
    size = np.asarray(shape, dtype=np.float64)
    norm = size.copy()
    norm[norm <= 0.0] = 2.0
    result = np.diag(np.concatenate((2.0 / norm, np.ones(1))))
    result[:-1, -1] = 1.0 / size - 1.0
    return result


def spatial_resample(
    data: Float32Array,
    src_affine: npt.ArrayLike,
    dst_affine: npt.ArrayLike,
    spatial_size: Sequence[int],
    *,
    mode: Mode = "bilinear",
    argmax: bool = False,
) -> tuple[Float32Array, Float64Array]:
    """MONAI ``SpatialResample`` (border padding, ``align_corners=False``, float64).

    Args:
        data: Float32 ``(C, D, H, W)`` image on the ``src_affine`` grid; any memory
            layout, including strided views.
        src_affine: The image affine (``MetaTensor.affine``).
        dst_affine: The output grid's affine.
        spatial_size: The output spatial shape.
        mode: ``"bilinear"`` (trilinear) or ``"nearest"`` (round half to even).
        argmax: Return ``channel_argmax`` of the float32 result instead, computed block
            by block so the resampled channels are never stored together.

    Returns:
        Float32 ``(C, *spatial_size)`` values (``(1, *spatial_size)`` labels with
        ``argmax``) and the output affine MONAI records. When MONAI skips resampling
        (see ``resample_plan``) the values are ``data`` itself and the affine a copy of
        ``src_affine``. The inverse of a resampling is this function back to the
        recorded source affine and shape.

    Raises:
        InvalidInputError: ``data`` is not float32 channel-first 3D, ``mode`` is not
            supported, or as ``resample_plan``.
    """
    _require_volume(data)
    if data.dtype != np.float32:
        raise InvalidInputError(f"Resampling takes float32 images, got {data.dtype}")
    if mode not in ("bilinear", "nearest"):
        raise InvalidInputError(f"Unsupported resampling mode: {mode!r}")
    plan = resample_plan(src_affine, dst_affine, data.shape[1:], spatial_size)
    if plan is None:
        values = channel_argmax(data) if argmax else data
        return values, np.array(src_affine, dtype=np.float64, copy=True)
    return _grid_sample(data, plan, mode, argmax), plan.affine


def spacing(
    data: Float32Array,
    affine: npt.ArrayLike,
    pixdim: Sequence[float],
    *,
    mode: Mode = "bilinear",
    spatial_size: Sequence[int] | None = None,
) -> tuple[Float32Array, Float64Array]:
    """``Spacing(pixdim, mode)``: ``spacing_grid`` then ``spatial_resample``.

    ``spatial_size`` overrides the computed output shape, as ``Spacing``'s
    ``output_spatial_shape`` does. The inverse is ``spatial_resample`` back to
    ``affine`` and ``data.shape[1:]``.
    """
    _require_volume(data)
    target, shape = spacing_grid(affine, data.shape[1:], pixdim)
    return spatial_resample(
        data, affine, target, shape if spatial_size is None else spatial_size, mode=mode
    )


# --- grid_sample ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Axis:
    """Sampling of one axis at each output position: corners, weights, inside flags."""

    low: IntArray
    high: IntArray  # clamped to the input; ``outside`` marks where that clamp applied
    outside: BoolArray
    weights: tuple[Float64Array, Float64Array]  # ATen's (high - i, i - low)
    nearest: IntArray

    def select(self, index: IntArray | tuple[IntArray, ...]) -> _Axis:
        """The same tables at the positions ``index``."""
        return _Axis(
            self.low[index],
            self.high[index],
            self.outside[index],
            (self.weights[0][index], self.weights[1][index]),
            self.nearest[index],
        )


def _axis(index: Float64Array, size: int) -> _Axis:
    """Corner tables for border-clipped continuous indices ``index`` on ``size`` voxels."""
    low = np.floor(index)
    high = low + 1
    nearest = np.rint(index).astype(np.int64)  # std::nearbyint, ties to even
    return _Axis(
        low.astype(np.int64),
        np.minimum(high, size - 1).astype(np.int64),
        high >= size,
        (high - index, index - low),
        nearest,
    )


def _grid_sample(data: Float32Array, plan: ResamplePlan, mode: Mode, argmax: bool) -> Float32Array:
    """``grid_sample(input.double(), affine_grid(theta))`` rounded to float32.

    The output is split into blocks of whole rows that run on ``_THREADS`` threads;
    NumPy releases the GIL and every voxel's arithmetic is independent of the split,
    so the result does not depend on the block size or the thread count.
    """
    channels = data.shape[0]
    depth, height, width = plan.out_shape
    result = np.empty((1 if argmax else channels, depth, height, width), dtype=np.float32)
    row_values = max(1, width * channels)
    budget = _ARGMAX_BLOCK_VALUES if argmax else _BLOCK_VALUES
    planes = max(1, budget // (height * row_values))
    rows = height if planes > 1 else min(height, max(1, budget // row_values))
    sample: Callable[[slice, slice], Float32Array]
    if plan.separable:
        axes = tuple(
            _axis(index, size)
            for index, size in zip(plan.axis_indices(), plan.in_shape, strict=True)
        )
        # The largest |value| of each channel's input planes, for the certification.
        plane_bound = np.maximum(data.max(axis=(2, 3)), -data.min(axis=(2, 3)))

        def separable(planes_: slice, rows_: slice) -> Float32Array:
            return _sample_separable(
                data, plane_bound, (axes[0], axes[1], axes[2]), planes_, rows_, mode, argmax
            )

        sample = separable
    else:
        sample = partial(_GeneralGrid(plan).sample, data, mode=mode, argmax=argmax)

    def fill(start: int, top: int) -> None:
        block = (slice(start, min(depth, start + planes)), slice(top, min(height, top + rows)))
        # Non-finite input gives NaN sums and certification ends; ``_certify`` resums
        # those lanes exactly.
        with np.errstate(invalid="ignore"):
            result[(slice(None), *block)] = sample(*block)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        blocks = [
            pool.submit(fill, start, top)
            for start in range(0, depth, planes)
            for top in range(0, height, rows)
        ]
        for block in blocks:
            block.result()
    return result


def _sample_separable(
    data: Float32Array,
    plane_bound: Float32Array,
    axes: tuple[_Axis, _Axis, _Axis],
    planes: slice,
    rows: slice,
    mode: Mode,
    argmax: bool,
) -> Float32Array:
    """One output block of an axis-aligned grid from per-axis tables, or its argmax.

    For each z side, the block's depth planes are gathered once and then reduced along
    the axis (y or x) whose gather leaves the smaller array; each corner then takes
    whole rows or columns of that. The z sides run one after the other, as ATen's
    corner order does, so one side's gathers are live at a time.
    """
    z_axis, y_axis, x_axis = axes
    if mode == "nearest":
        region, y_start = _rows(data, y_axis.nearest[rows], y_axis.nearest[rows])
        slab = region[:, z_axis.nearest[planes]]
        picked = slab[:, :, y_axis.nearest[rows] - y_start][..., x_axis.nearest]
        return channel_argmax(picked) if argmax else picked
    region, y_start = _rows(data, y_axis.low[rows], y_axis.high[rows])
    row_index = (y_axis.low[rows] - y_start, y_axis.high[rows] - y_start)
    y_outside = y_axis.outside[rows]
    x_outside = x_axis.outside
    y_weights = (y_axis.weights[0][rows], y_axis.weights[1][rows])
    z_weights = (z_axis.weights[0][planes], z_axis.weights[1][planes])
    # (x * y) * z, left to right, as in grid_sampler_3d_cpu_impl.
    plane_weights = {
        (side_y, side_x): x_axis.weights[side_x][None, :] * y_weights[side_y][:, None]
        for side_y in (0, 1)
        for side_x in (0, 1)
    }
    shape = (data.shape[0], len(z_weights[0]), len(y_weights[0]), len(x_axis.low))
    x_first = region.shape[2] * shape[3] <= shape[2] * region.shape[3]
    total = np.empty(shape, dtype=np.float64)
    term = np.empty(shape, dtype=np.float64)
    values = np.empty(shape, dtype=np.float32)
    weight = np.empty(shape[1:], dtype=np.float64)
    for side_z, z_index in enumerate((z_axis.low[planes], z_axis.high[planes])):
        slab = region[:, z_index]
        if side_z:
            # Corners beyond the input have weight zero; ATen skips them. A zero
            # value adds an exact zero, which leaves every sum unchanged.
            slab[:, z_axis.outside[planes]] = 0
        if x_first:
            gathered = [slab[..., x_axis.low], slab[..., x_axis.high]]
            gathered[1][..., x_outside] = 0
        else:
            gathered = [slab[:, :, row_index[0]], slab[:, :, row_index[1]]]
            gathered[1][:, :, y_outside] = 0
        del slab
        for _, side_y, side_x in _CORNERS[:4]:
            if x_first:
                np.take(gathered[side_x], row_index[side_y], axis=2, out=values)
                if side_y:
                    values[:, :, y_outside] = 0
            else:
                np.take(gathered[side_y], (x_axis.low, x_axis.high)[side_x], axis=3, out=values)
                if side_x:
                    values[..., x_outside] = 0
            np.multiply(
                plane_weights[side_y, side_x][None], z_weights[side_z][:, None, None], out=weight
            )
            # Float32 values widen exactly: this is the product on MONAI's float64 copy.
            first = not (side_z or side_y or side_x)
            np.multiply(values, weight, out=total if first else term)
            if not first:
                total += term
        del gathered
    # The certification bound: the largest |value| of the input planes a block reads.
    bound = np.maximum(plane_bound[:, z_axis.low[planes]], plane_bound[:, z_axis.high[planes]])

    def exact(lanes: tuple[IntArray, ...]) -> Float32Array:
        channel, d, h, w = lanes
        return _fused_lanes(
            data,
            channel,
            (
                z_axis.select(d + (planes.start or 0)),
                y_axis.select(h + (rows.start or 0)),
                x_axis.select(w),
            ),
        )

    del values
    return _certify(total, term, bound[:, :, None, None], exact, argmax)


def _rows(data: Float32Array, low: IntArray, high: IntArray) -> tuple[Float32Array, int]:
    """The input rows a block reads, as a view, and the first row's index.

    Gathers from the view copy only the block's rows, so a strided ``data`` (logits
    cropped from a padded window) is never copied in full.
    """
    start = int(min(low.min(), high.min()))
    stop = int(max(low.max(), high.max())) + 1
    return data[:, :, start:stop], start


class _GeneralGrid:
    """Sampling coordinates of an oblique grid, built block by block.

    Each coordinate row sums ``x * t0``, ``y * t1``, ``z * t2`` and ``t3`` with ATen's
    fused multiply-adds; the ``x`` and ``y`` part is shared by every output plane.
    """

    def __init__(self, plan: ResamplePlan) -> None:
        """Precompute the per-plane part of each coordinate row."""
        depth, height, width = plan.out_shape
        self._plan = plan
        self._z = linspace_from_neg_one(depth)
        x = linspace_from_neg_one(width)
        y = linspace_from_neg_one(height)
        self._xy = [
            fma(y[:, None], plan.theta[row, 1], (x * plan.theta[row, 0])[None, :])
            for row in range(3)
        ]

    def sample(
        self, data: Float32Array, planes: slice, rows: slice, mode: Mode, argmax: bool
    ) -> Float32Array:
        """One output block ``(C, planes, rows, W)``, or its argmax."""
        plan = self._plan
        z = self._z[planes][:, None, None]
        x, y, depth = (
            _source_index(
                fma(z, plan.theta[row, 2], self._xy[row][None, rows]) + plan.theta[row, 3],
                plan.in_shape[2 - row],
            )
            for row in range(3)
        )
        size_d, size_h, size_w = plan.in_shape
        if mode == "nearest":
            picked = _gather(data, *(np.rint(value).astype(np.int64) for value in (depth, y, x)))
            return channel_argmax(picked) if argmax else picked
        corner_axes = (_axis(depth, size_d), _axis(y, size_h), _axis(x, size_w))
        shape = (data.shape[0], *x.shape)
        total = np.empty(shape, dtype=np.float64)
        term = np.empty(shape, dtype=np.float64)
        bound = np.zeros(shape, dtype=np.float32)
        weight = np.empty(x.shape, dtype=np.float64)
        for corner, sides in enumerate(_CORNERS):
            values = _corner_values(data, shape, corner_axes, sides)
            np.maximum(bound, np.abs(values), out=bound)
            _corner_weight(corner_axes, sides, out=weight)
            np.multiply(values, weight, out=term if corner else total)
            if corner:
                total += term

        def exact(lanes: tuple[IntArray, ...]) -> Float32Array:
            channel, *position = lanes
            index = (position[0], position[1], position[2])
            selected = tuple(axis.select(index) for axis in corner_axes)
            return _fused_lanes(data, channel, (selected[0], selected[1], selected[2]))

        return _certify(total, term, bound, exact, argmax)


def _corner_values(
    data: Float32Array,
    shape: tuple[int, ...],
    axes: tuple[_Axis, _Axis, _Axis],
    sides: tuple[int, ...],
) -> Float32Array:
    """Input values at one corner; corners beyond the input are zero (ATen skips them)."""
    z, y, x = (axis.high if side else axis.low for axis, side in zip(axes, sides, strict=True))
    values = _gather(data, z, y, x)
    outside = [axis.outside for axis, side in zip(axes, sides, strict=True) if side]
    if outside:
        mask = outside[0] if len(outside) == 1 else np.logical_or.reduce(outside)
        if mask.any():
            np.copyto(values, np.float32(0), where=np.broadcast_to(mask, shape))
    return values


def _corner_weight(
    axes: tuple[_Axis, _Axis, _Axis], sides: tuple[int, ...], *, out: Float64Array
) -> None:
    """``(wx * wy) * wz``, left to right, as in grid_sampler_3d_cpu_impl."""
    z, y, x = (axis.weights[side] for axis, side in zip(axes, sides, strict=True))
    np.multiply(x, y, out=out)
    out *= z


def _gather(data: Float32Array, z: IntArray, y: IntArray, x: IntArray) -> Float32Array:
    """``data[:, z, y, x]`` for broadcast index arrays, flat ``take`` when contiguous."""
    if data.flags.c_contiguous:
        _, depth, height, width = data.shape
        flat = (z * height + y) * width + x
        flat_data = data.reshape(data.shape[0], depth * height * width)
        return cast(Float32Array, flat_data.take(flat, axis=1))
    picked: Float32Array = data[:, z, y, x]
    return picked


# ATen accumulates each output as ``out += value * weight``, contracted by Clang into
# one rounding per corner. The blocks sum the rounded products instead, which needs
# far fewer operations, and certify the result. With u = 2**-53, the unfused sum of
# eight corners is within 15 u of the exact sum and the fused one within 8 u, both
# relative to the sum of |value * weight|, which is at most the largest corner |value|
# because the weights are nonnegative and sum to one. The fused float32 result is
# therefore known wherever both ends of an interval of 64 u times that value round to
# the same float32. Only the remaining lanes, near float32 rounding boundaries, are
# summed again with fused updates.
_CERTIFY = 2.0**-47


def _certify(
    total: Float64Array,
    scratch: Float64Array,
    bound: Float32Array,
    exact: Callable[[tuple[IntArray, ...]], Float32Array],
    argmax: bool,
) -> Float32Array:
    """Float32 ATen sums from the unfused ``total``, or their ``channel_argmax``.

    ``bound`` broadcasts to ``total`` and bounds each lane's corner magnitudes.
    ``exact`` returns the fused sums of the lanes given as index arrays. ``scratch``,
    a free array of ``total``'s shape, holds the two float32 ends.
    """
    radius = bound.astype(np.float64) * _CERTIFY
    # Each end is rounded to float64, then to float32 on the way out.
    ends = scratch.reshape(-1).view(np.float32).reshape(2, *total.shape)
    low, high = ends[0], ends[1]
    np.subtract(total, radius, out=low, casting="same_kind")
    np.add(total, radius, out=high, casting="same_kind")
    uncertain = np.flatnonzero(low != high)
    if not argmax:
        if uncertain.size:
            lanes = np.unravel_index(uncertain, high.shape)
            high[lanes] = exact(lanes)
        return high
    labels = np.argmax(high, axis=0)
    if uncertain.size:
        # Each fused sum lies in [low, high]. The low of the channel with the largest
        # high bounds its voxel's maximum from below; a channel whose high is under it
        # cannot be or change the argmax.
        # A NaN end (a non-finite corner in the lane's bound) fails ``<``, so such
        # lanes, and every lane of a voxel whose floor is NaN, are summed again.
        voxel = uncertain % labels.size
        floor = low.reshape(low.shape[0], -1)[labels.reshape(-1)[voxel], voxel]
        uncertain = uncertain[~(high.reshape(-1)[uncertain] < floor)]
    if uncertain.size:
        lanes = np.unravel_index(uncertain, high.shape)
        high[lanes] = exact(lanes)
        voxels = np.unravel_index(np.unique(uncertain % labels.size), labels.shape)
        labels[voxels] = np.argmax(high[(slice(None), *voxels)], axis=0)
    return labels[None].astype(np.float32)


def _fused_lanes(
    data: Float32Array, channel: IntArray, axes: tuple[_Axis, _Axis, _Axis]
) -> Float32Array:
    """ATen's fused trilinear sums ``out = fma(value, weight, out)`` for single lanes.

    ``channel`` and the tables in ``axes`` are 1-D, one entry per lane. Corners beyond
    the input read zero, which leaves the sum unchanged as ATen's skip does.
    """
    sides = np.array(_CORNERS, dtype=np.bool_)[:, :, None]
    picks = [
        (
            np.where(side, axis.high, axis.low),
            side & axis.outside,
            np.where(side, axis.weights[1], axis.weights[0]),
        )
        for axis, side in zip(axes, (sides[:, 0], sides[:, 1], sides[:, 2]), strict=True)
    ]
    (z, z_out, z_weight), (y, y_out, y_weight), (x, x_out, x_weight) = picks
    values = data[channel, z, y, x].astype(np.float64)
    values[z_out | y_out | x_out] = 0
    # (x * y) * z, left to right, as in grid_sampler_3d_cpu_impl.
    weights = (x_weight * y_weight) * z_weight
    # Exact products: float32 values have 24 significant bits, so only the weight
    # needs Veltkamp's split for Dekker's error term.
    products = values * weights
    scaled = weights * _SPLIT
    weight_high = scaled - (scaled - weights)
    errors = (values * weight_high - products) + values * (weights - weight_high)
    total = products[0]
    for corner in range(1, len(_CORNERS)):
        product, error = products[corner], errors[corner]
        # Two-sums of total + product and of their tail + error; then the tail sum
        # is rounded to odd so that the final addition rounds once (see ``fma``).
        added = total + product
        shift = added - total
        tail = (total - (added - shift)) + (product - shift)
        odd = tail + error
        shift = odd - tail
        remainder = (tail - (odd - shift)) + (error - shift)
        inexact = remainder != 0
        # An inexact sum is truncated toward zero, one unit less in magnitude when the
        # remainder has the other sign, then its last bit is set.
        bits = odd.view(np.int64)
        np.subtract(bits, 1, out=bits, where=inexact & (np.signbit(remainder) != np.signbit(odd)))
        np.bitwise_or(bits, 1, out=bits, where=inexact)
        total = added + odd
    # With a non-finite product ATen's sum is NaN or an infinity whatever the rounding,
    # and the plain sum gives it; the error terms above turn an infinity into NaN.
    special = ~np.isfinite(products).all(axis=0)
    if special.any():
        total[special] = products[:, special].sum(axis=0)
    return total.astype(np.float32)


# --- Intensity, foreground and labels ------------------------------------------------


def scale_intensity_range(
    data: npt.ArrayLike, a_min: float, a_max: float, b_min: float, b_max: float
) -> Float32Array:
    """``ScaleIntensityRange(a_min, a_max, b_min, b_max, clip=True)``, returned as float32.

    Torch's operation order: ``(x - a_min) / (a_max - a_min)``, times ``b_max - b_min``,
    plus ``b_min``, clipped to ``[b_min, b_max]``; each difference of bounds is a Python
    float. Float64 input is computed in float64 and rounded to float32 at the end, any
    other input in float32 with the bounds rounded to float32. With ``a_max == a_min``
    MONAI returns ``x - a_min + b_min`` unclipped, in the input dtype where this returns
    float32. Returns a new array; ``data`` is unchanged.
    """
    values = np.asarray(data)
    scalar = np.float64 if values.dtype == np.float64 else np.float32
    result = np.subtract(values, scalar(a_min), dtype=scalar)
    if a_max - a_min == 0.0:
        result += scalar(b_min)
    else:
        result /= scalar(a_max - a_min)
        result *= scalar(b_max - b_min)
        result += scalar(b_min)
        np.clip(result, scalar(b_min), scalar(b_max), out=result)
    return result.astype(np.float32, copy=False)


def foreground_box(
    data: npt.NDArray[np.number] | npt.NDArray[np.bool_], margin: int | Sequence[int] = 0
) -> tuple[Shape3, Shape3]:
    """``CropForeground``'s box: ``generate_spatial_bounding_box(select_fn=x > 0,
    allow_smaller=True, margin)`` with ``k_divisible=1``.

    ``data`` is channel-first ``(C, D, H, W)``; a voxel is foreground when any channel
    is above zero. Returns ``(start, end)`` per spatial axis, ``end`` exclusive, the
    margin clipped to the volume; ``((0, 0, 0), (0, 0, 0))`` without foreground.
    """
    _require_volume(data)
    margins = (margin,) * 3 if isinstance(margin, int) else tuple(margin)
    if len(margins) != 3 or any(value < 0 for value in margins):
        raise InvalidInputError(f"Foreground margin must be three nonnegative values: {margin}")
    mask = cast(BoolArray, np.greater(data, 0).any(axis=0))
    start: list[int] = []
    end: list[int] = []
    for axis in range(3):
        profile = mask.any(axis=tuple(other for other in range(3) if other != axis))
        hits = np.flatnonzero(profile)
        if hits.size == 0:
            return (0, 0, 0), (0, 0, 0)
        start.append(max(int(hits[0]) - margins[axis], 0))
        end.append(min(int(hits[-1]) + margins[axis] + 1, mask.shape[axis]))
    return (start[0], start[1], start[2]), (end[0], end[1], end[2])


def channel_argmax(data: npt.NDArray[np.generic]) -> Float32Array:
    """``AsDiscrete(argmax=True)``: the first maximal channel as float32 ``(1, ...)``."""
    return np.argmax(data, axis=0)[None].astype(np.float32)


# --- helpers --------------------------------------------------------------------------


def _require_volume(data: npt.NDArray[np.generic]) -> None:
    """Reject arrays that are not channel-first with three nonempty spatial axes."""
    if data.ndim != 4 or min(data.shape) < 1:
        raise InvalidInputError(f"Expected a nonempty channel-first 3D array, got {data.shape}")


def _shape3(shape: Sequence[int] | npt.ArrayLike) -> Shape3:
    """Three positive integer sizes."""
    values = [int(value) for value in cast(Sequence[int], np.asarray(shape).tolist())]
    if len(values) != 3 or min(values) < 1:
        raise InvalidInputError(f"Expected three positive spatial sizes, got {shape}")
    return values[0], values[1], values[2]
