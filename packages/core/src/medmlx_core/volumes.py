"""Read 3D NIfTI volumes under one admission policy and write masks on their grid.

``read_volume`` is the single reader for model inputs: it rejects anything that is
not a real, finite, three-dimensional NIfTI-1 or NIfTI-2 volume on a valid
millimeter grid, and reports the header geometry that callers need for MONAI-style
header correction, resampling and writing. ``write_volume`` writes a
new NIfTI-1 file and ``write_like`` writes on a reference volume's grid with its
header. Both writers write straight to ``path``; the runner passes a
private staging path from ``medmlx_core.files.publish``, which handles overwrite
and atomic replacement.

nibabel is imported inside the functions, so importing this module needs neither
nibabel nor MLX. The typed protocols below are the only nibabel typing any runtime
needs.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from medmlx_core.errors import InvalidInputError, MissingDependencyError
from medmlx_core.files import PathLike, input_file
from medmlx_core.typing import HostArray

type Float64Array = NDArray[np.float64]
# Stored voxel types the reader admits: real integers and floating point.
type StoredScalar = np.integer | np.floating

# NIfTI spatial unit code (low three bits of ``xyzt_units``) -> millimeters per unit.
# 0 is unspecified, taken as millimeters; 1 is meter, 2 millimeter, 3 micron.
_UNIT_TO_MM = {0: 1.0, 1: 1000.0, 2: 1.0, 3: 0.001}

@runtime_checkable
class NiftiHeader(Protocol):
    """The nibabel NIfTI header calls used here; treat an instance as read-only.

    ``header[name]`` returns the named NIfTI field (``"descrip"``, ``"intent_code"``,
    ``"pixdim"``, ...) as an array, in the header's own units.
    """

    def __getitem__(self, key: str) -> HostArray:
        """Return a named header field."""
        ...

    def __setitem__(self, key: str, value: HostArray) -> None:
        """Set a named header field."""
        ...

    def copy(self) -> NiftiHeader:
        """Return an independent copy with every field and extension."""
        ...

    def get_data_shape(self) -> tuple[int, ...]:
        """Return the stored array shape."""
        ...

    def get_data_dtype(self) -> np.dtype[np.generic]:
        """Return the stored voxel dtype."""
        ...

    def set_data_dtype(self, datatype: type[np.generic]) -> None:
        """Set the dtype voxels are written with."""
        ...

    def get_zooms(self) -> tuple[float, ...]:
        """Return the voxel sizes in the header's spatial unit."""
        ...

    def get_sform(self, coded: bool = False) -> Float64Array:
        """Return the sform matrix, whatever its code."""
        ...

    def get_qform(self, coded: bool = False) -> Float64Array:
        """Return the qform matrix, whatever its code."""
        ...

    def get_best_affine(self) -> Float64Array:
        """Return the sform if coded, else the qform if coded, else the zoom grid."""
        ...

    def set_sform(self, affine: Float64Array, code: int) -> None:
        """Store an sform and its code."""
        ...

    def set_qform(self, affine: Float64Array, code: int) -> None:
        """Store a qform (shear removed) and its code."""
        ...

    def set_xyzt_units(self, xyz: str) -> None:
        """Set the spatial unit by name."""
        ...


class _ArrayProxy(Protocol):
    def __array__(self, dtype: None = None, copy: None = None) -> HostArray:
        """Decode the voxels with the header's slope and intercept applied."""
        ...


@runtime_checkable
class _Image(Protocol):
    header: NiftiHeader
    dataobj: _ArrayProxy
    affine: Float64Array

    def __init__(
        self, dataobj: HostArray, affine: Float64Array, header: NiftiHeader | None = None
    ) -> None:
        """Wrap voxels and an affine, optionally with a header."""
        ...


class _Nibabel(Protocol):
    Nifti1Image: type[_Image]
    Nifti2Image: type[_Image]
    Nifti2Header: type[NiftiHeader]

    def load(self, filename: str) -> object:
        """Load an image whose class follows the file contents."""
        ...

    def save(self, img: _Image, filename: str) -> None:
        """Write an image; a ``.gz`` suffix selects gzip compression."""
        ...


@dataclass(frozen=True, eq=False)
class Volume[S: np.generic]:
    """A 3D NIfTI volume with the geometry its header carries.

    Attributes:
        data: Voxels in file axis order ``(i, j, k)`` with the header's slope and
            intercept applied. No reorientation or resampling.
        affine: Float64 4x4 voxel-to-world matrix in millimeters, as nibabel selects
            it (sform if its code is nonzero, else qform if its code is nonzero,
            else the zoom grid), with its first three rows scaled when the header
            spatial unit is meter or micron.
        path: The file read.
        zooms: Voxel sizes ``(i, j, k)`` in millimeters, finite and positive.
        sform: The header's sform, 4x4, in millimeters, whatever its code.
        sform_code: NIfTI ``sform_code``; 0 means the sform is unset.
        qform: The header's qform, 4x4, in millimeters, whatever its code.
        qform_code: NIfTI ``qform_code``; 0 means the qform is unset.
        header: The source header in its own units, for ``write_like`` and for
            fields not listed here. Do not modify it.
    """

    data: NDArray[S]
    affine: Float64Array
    path: Path
    zooms: tuple[float, float, float]
    sform: Float64Array
    sform_code: int
    qform: Float64Array
    qform_code: int
    header: NiftiHeader


@overload
def read_volume(path: PathLike, *, role: str = ...) -> Volume[np.float32]: ...
@overload
def read_volume(
    path: PathLike, *, role: str = ..., dtype: type[np.float32]
) -> Volume[np.float32]: ...
@overload
def read_volume(
    path: PathLike, *, role: str = ..., dtype: type[np.float64]
) -> Volume[np.float64]: ...
@overload
def read_volume(path: PathLike, *, role: str = ..., dtype: None) -> Volume[StoredScalar]: ...
def read_volume(
    path: PathLike,
    *,
    role: str = "image_path",
    dtype: type[np.float32] | type[np.float64] | None = np.float32,
) -> Volume[np.float32] | Volume[np.float64] | Volume[StoredScalar]:
    """Read a 3D NIfTI-1 or NIfTI-2 volume, admitting only valid model input.

    Args:
        path: The ``.nii`` or ``.nii.gz`` file.
        role: Names the input in every error message, for example ``"image_path"``.
        dtype: ``np.float32`` (default) or ``np.float64`` returns a writable,
            C-contiguous array of that type. ``None`` keeps the stored dtype (for
            label maps, or to avoid a float64 copy of integer data) and returns the
            array as nibabel decodes it, in any memory order. Values are decoded once
            and converted once; no further full-volume copy is made.

    Returns:
        The voxels and header geometry; see ``Volume``. The affine and zooms are in
        millimeters: spatial unit meter or micron is converted and unspecified is
        taken as millimeter.

    Raises:
        MissingDependencyError: nibabel is not installed.
        InvalidInputError: The file is missing or unreadable, is not NIfTI-1 or
            NIfTI-2, is not a nonempty 3D volume of real integer or floating-point
            voxels, contains NaN or infinity, has voxels outside the float32 range
            when ``dtype`` is ``np.float32``, has a spatial unit other than
            unspecified, meter, millimeter or micron, has an affine that is not
            finite with last row ``0 0 0 1`` and an invertible 3x3 block, or has
            zooms that are not finite and positive. Every message starts with
            ``role``.
    """
    nib = _nibabel()
    from nibabel.filebasedimages import ImageFileError
    from nibabel.spatialimages import HeaderDataError

    read_errors = (OSError, EOFError, OverflowError, ValueError, zlib.error, ImageFileError)
    file = input_file(role, path)
    try:
        image = nib.load(file.as_posix())
    except (*read_errors, HeaderDataError) as error:
        raise InvalidInputError(f"{role} is not a readable NIfTI file: {file}: {error}") from error
    if not isinstance(image, nib.Nifti1Image):
        raise InvalidInputError(f"{role} must be a NIfTI-1 or NIfTI-2 file: {file}")
    header = image.header
    shape = header.get_data_shape()
    if len(shape) != 3 or min(shape) < 1:
        raise InvalidInputError(f"{role} must be a nonempty 3D volume; got shape {shape}")
    if header.get_data_dtype().kind not in "iuf":
        raise InvalidInputError(f"{role} must contain real integer or floating-point voxels")
    unit_code = int(header["xyzt_units"]) & 0x07
    if unit_code not in _UNIT_TO_MM:
        raise InvalidInputError(
            f"{role} spatial unit code {unit_code} is not unspecified, meter, millimeter or micron"
        )
    scale = _UNIT_TO_MM[unit_code]

    affine = np.array(image.affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.array_equal(affine[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise InvalidInputError(f"{role} affine must be finite with last row 0 0 0 1")
    affine[:3] *= scale
    if np.linalg.matrix_rank(affine[:3, :3]) < 3:
        raise InvalidInputError(f"{role} affine must be invertible")
    zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
    if zooms.shape != (3,) or not np.isfinite(zooms).all() or (zooms <= 0.0).any():
        raise InvalidInputError(f"{role} voxel sizes must be finite and positive")

    try:
        data = np.asanyarray(image.dataobj)
    except read_errors as error:
        raise InvalidInputError(f"{role} voxels cannot be read: {file}: {error}") from error
    if data.dtype.kind == "f":
        low, high = float(data.min()), float(data.max())
        if not (np.isfinite(low) and np.isfinite(high)):
            raise InvalidInputError(f"{role} contains non-finite voxels")
        if dtype is np.float32:
            limit = float(np.finfo(np.float32).max)
            if low < -limit or high > limit:
                raise InvalidInputError(f"{role} contains voxels outside the float32 range")
    # A memmap would keep the file mapped; copy it out (the one required copy).
    detach = isinstance(data, np.memmap)
    if dtype is None:
        # The dtype kind was checked above, so the stored scalars are integer or floating.
        values = cast(NDArray[StoredScalar], np.array(data) if detach else data)
    else:
        values = np.array(data, dtype=dtype, order="C", copy=True if detach else None)

    sform = header.get_sform(coded=False).astype(np.float64)
    qform = header.get_qform(coded=False).astype(np.float64)
    sform[:3] *= scale
    qform[:3] *= scale
    zooms *= scale
    return Volume(
        data=values,
        affine=affine,
        path=file,
        zooms=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
        sform=sform,
        sform_code=int(header["sform_code"]),
        qform=qform,
        qform_code=int(header["qform_code"]),
        header=header,
    )


def write_volume(
    path: PathLike, data: HostArray, affine: Float64Array, *, dtype: type[np.generic]
) -> None:
    """Write a new NIfTI-1 file from a 3D array and a millimeter affine.

    The voxels are cast to ``dtype`` without scaling. The affine (4x4, voxel to
    world in millimeters) is written as sform and as qform, both with code 1
    (scanner anatomical); the qform cannot represent shear, which nibabel removes
    from it. The spatial unit is millimeter and the time unit unspecified. A
    ``.gz`` suffix selects gzip compression. ``path`` is written directly; use a
    staging path from ``medmlx_core.files.publish`` to replace a file atomically.

    Raises:
        MissingDependencyError: nibabel is not installed.
        InvalidInputError: ``data`` is not 3D.
    """
    nib = _nibabel()
    array = _cast_volume(data, dtype)
    matrix = np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(array, matrix)
    image.header.set_xyzt_units("mm")
    image.header.set_sform(matrix, 1)
    image.header.set_qform(matrix, 1)
    nib.save(image, str(path))


def write_like[S: np.generic](
    path: PathLike, data: HostArray, reference: Volume[S], *, dtype: type[np.generic]
) -> None:
    """Write ``data`` on the grid of ``reference`` with the reference file's header.

    ``data`` must be 3D with the shape of ``reference.data`` and is cast to ``dtype``
    without scaling. The file takes a copy of the reference header: voxel sizes,
    spatial and temporal units, the qform and sform with their codes, description,
    intent and extensions are those of the source file, in its own units and NIfTI
    version. Only the voxel dtype and the intensity scaling are replaced. ``path`` is
    written directly; use a staging path from ``medmlx_core.files.publish`` to
    replace a file atomically.

    Raises:
        MissingDependencyError: nibabel is not installed.
        InvalidInputError: ``data`` is not 3D or its shape differs from the
            reference's.
    """
    nib = _nibabel()
    array = _cast_volume(data, dtype)
    if array.shape != reference.data.shape:
        raise InvalidInputError(
            f"Output shape {array.shape} differs from the reference shape {reference.data.shape}"
        )
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image_class = nib.Nifti2Image if isinstance(header, nib.Nifti2Header) else nib.Nifti1Image
    # The image affine must be the header's own, or nibabel rewrites the forms from it.
    image = image_class(array, header.get_best_affine(), header)
    nib.save(image, str(path))


def _cast_volume(data: HostArray, dtype: type[np.generic]) -> HostArray:
    array = np.asarray(data)
    if array.ndim != 3:
        raise InvalidInputError(f"Output volume must be 3D; got shape {array.shape}")
    return array.astype(dtype, copy=False)


def _nibabel() -> _Nibabel:
    try:
        return cast(_Nibabel, import_module("nibabel"))
    except ImportError as error:
        raise MissingDependencyError(
            "Reading and writing NIfTI volumes requires nibabel",
            extra="nifti",
            hint="Install nibabel, for example with medmlx-core[nifti].",
        ) from error


__all__ = ["NiftiHeader", "StoredScalar", "Volume", "read_volume", "write_like", "write_volume"]
