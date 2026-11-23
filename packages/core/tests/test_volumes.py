"""NIfTI admission and writing: what a model runtime sees at the public boundary.

Fixture files are built byte by byte, so header fields reach the reader exactly as
a scanner or converter would write them.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("nibabel")

from medmlx_core.errors import InvalidInputError
from medmlx_core.volumes import read_volume, write_like, write_volume

# NIfTI-1 datatype code and bits per voxel.
_DATATYPES: dict[type[np.generic], tuple[int, int]] = {
    np.uint8: (2, 8),
    np.int16: (4, 16),
    np.float32: (16, 32),
    np.float64: (64, 64),
    np.complex64: (32, 64),
}
_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Header fields that place voxels in world space, as (offset, struct format).
_GEOMETRY = {
    "pixdim": (76, "<8f"),
    "xyzt_units": (123, "<b"),
    "qform_code": (252, "<h"),
    "quaternion": (256, "<3f"),
    "qoffset": (268, "<3f"),
    "sform_code": (254, "<h"),
    "srow": (280, "<12f"),
}


def write_nifti(
    path: Path,
    data: NDArray[np.generic],
    *,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    unit: int = 2,
    srow: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    sform_code: int = 1,
    qform_code: int = 0,
    qoffset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    slope: float = 1.0,
    intercept: float = 0.0,
    dim: tuple[int, ...] | None = None,
) -> None:
    shape = dim if dim is not None else data.shape
    code, bits = _DATATYPES[data.dtype.type]
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, len(shape), *shape, *([1] * (7 - len(shape))))
    struct.pack_into("<hh", header, 70, code, bits)
    struct.pack_into("<8f", header, 76, 1.0, *zooms, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<fff", header, 108, 352.0, slope, intercept)
    struct.pack_into("<b", header, 123, unit)
    struct.pack_into("<hh", header, 252, qform_code, sform_code)
    struct.pack_into("<3f", header, 256, 0.0, 0.0, 0.0)
    struct.pack_into("<3f", header, 268, *qoffset)
    struct.pack_into("<12f", header, 280, *srow)
    header[344:348] = b"n+1\0"
    path.write_bytes(bytes(header) + bytes(4) + data.tobytes(order="F"))


def raw_geometry(path: Path) -> dict[str, tuple[float, ...]]:
    blob = path.read_bytes()
    return {
        name: struct.unpack_from(fmt, blob, offset) for name, (offset, fmt) in _GEOMETRY.items()
    }


def raw_datatype(path: Path) -> int:
    return int(struct.unpack_from("<h", path.read_bytes(), 70)[0])


def test_read_applies_scaling_and_reports_the_header_geometry(tmp_path: Path) -> None:
    stored = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    file = tmp_path / "ct.nii"
    write_nifti(
        file,
        stored,
        zooms=(2.0, 3.0, 4.0),
        srow=(2.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0, 20.0, 0.0, 0.0, 4.0, 30.0),
        slope=2.0,
        intercept=-10.0,
    )

    volume = read_volume(file)

    assert volume.data.dtype == np.float32
    assert volume.data.flags.c_contiguous
    np.testing.assert_array_equal(volume.data, stored.astype(np.float32) * 2.0 - 10.0)
    np.testing.assert_allclose(
        volume.affine, [[2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]]
    )
    assert volume.affine.dtype == np.float64
    assert volume.zooms == (2.0, 3.0, 4.0)
    assert (volume.sform_code, volume.qform_code) == (1, 0)
    np.testing.assert_allclose(volume.sform, volume.affine)
    assert volume.path == file


@pytest.mark.parametrize(("unit", "millimeters"), [(0, 1.0), (2, 1.0), (1, 1000.0), (3, 0.001)])
def test_read_converts_spatial_units_to_millimeters(
    tmp_path: Path, unit: int, millimeters: float
) -> None:
    file = tmp_path / "volume.nii"
    write_nifti(
        file,
        np.zeros((2, 3, 4), np.float32),
        zooms=(1.0, 2.0, 3.0),
        unit=unit,
        srow=(1.0, 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 6.0, 0.0, 0.0, 3.0, 7.0),
    )

    volume = read_volume(file)

    expected = np.array([[1, 0, 0, 5], [0, 2, 0, 6], [0, 0, 3, 7], [0, 0, 0, 1 / millimeters]])
    np.testing.assert_allclose(volume.affine, expected * millimeters)
    np.testing.assert_allclose(volume.sform, expected * millimeters)
    np.testing.assert_allclose(volume.zooms, np.array([1.0, 2.0, 3.0]) * millimeters)


def test_read_keeps_the_stored_dtype_without_a_dtype(tmp_path: Path) -> None:
    file = tmp_path / "labels.nii"
    labels = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    write_nifti(file, labels)

    volume = read_volume(file, dtype=None)

    assert volume.data.dtype == np.int16
    np.testing.assert_array_equal(volume.data, labels)


def test_read_float64_admits_values_beyond_float32(tmp_path: Path) -> None:
    file = tmp_path / "wide.nii"
    wide = np.full((2, 2, 2), 1e300)
    write_nifti(file, wide)

    volume = read_volume(file, dtype=np.float64)

    assert volume.data.dtype == np.float64
    np.testing.assert_array_equal(volume.data, wide)


def _nan_volume() -> NDArray[np.generic]:
    voxels = np.zeros((2, 2, 2), np.float32)
    voxels[0, 0, 0] = np.nan
    return voxels


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "not-nifti",
        "four-dimensional",
        "nan-voxels",
        "beyond-float32",
        "complex",
        "singular-affine",
        "non-finite-affine",
        "unit-code",
        "non-finite-zoom",
    ],
)
def test_read_rejects_inputs_a_model_cannot_use(tmp_path: Path, case: str) -> None:
    file = tmp_path / "input.nii"
    data = np.zeros((2, 2, 2), np.float32)
    if case == "not-nifti":
        file.write_text("not a NIfTI file" * 40)
    elif case == "four-dimensional":
        write_nifti(file, np.zeros((2, 2, 2, 2), np.float32), dim=(2, 2, 2, 2))
    elif case == "nan-voxels":
        write_nifti(file, _nan_volume())
    elif case == "beyond-float32":
        write_nifti(file, np.full((2, 2, 2), 1e300))
    elif case == "complex":
        write_nifti(file, data.astype(np.complex64))
    elif case == "singular-affine":
        write_nifti(file, data, srow=(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
    elif case == "non-finite-affine":
        write_nifti(file, data, srow=(float("nan"), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))
    elif case == "unit-code":
        write_nifti(file, data, unit=4)
    elif case == "non-finite-zoom":
        write_nifti(file, data, zooms=(1.0, float("nan"), 1.0))

    with pytest.raises(InvalidInputError) as raised:
        read_volume(file, role="mask_path")

    assert str(raised.value).startswith("mask_path")


def test_write_volume_reads_back_with_the_same_voxels_and_millimeter_grid(
    tmp_path: Path,
) -> None:
    labels = (np.arange(24, dtype=np.int16).reshape(2, 3, 4) % 5).astype(np.uint8)
    affine = np.array(
        [[0.7, 0.0, 0.0, -50.0], [0.0, 0.8, 0.0, 12.0], [0.0, 0.0, 2.5, 3.0], [0, 0, 0, 1]]
    )
    file = tmp_path / "mask.nii"

    write_volume(file, labels, affine, dtype=np.uint8)
    volume = read_volume(file, dtype=None)

    assert volume.data.dtype == np.uint8
    np.testing.assert_array_equal(volume.data, labels)
    np.testing.assert_allclose(volume.affine, affine, atol=1e-5)
    assert (volume.sform_code, volume.qform_code) == (1, 1)
    assert int(raw_geometry(file)["xyzt_units"][0]) & 0x07 == 2
    assert raw_datatype(file) == _DATATYPES[np.uint8][0]


def test_write_like_keeps_the_reference_header_and_units(tmp_path: Path) -> None:
    reference_file = tmp_path / "reference.nii"
    write_nifti(
        reference_file,
        np.zeros((3, 4, 5), np.int16),
        zooms=(0.0007, 0.0008, 0.0025),
        unit=1,
        srow=(0.0007, 0.0, 0.0, -0.05, 0.0, 0.0008, 0.0, 0.012, 0.0, 0.0, 0.0025, 0.003),
        sform_code=2,
        qform_code=1,
        qoffset=(-0.049, 0.0121, 0.0031),
    )
    with reference_file.open("r+b") as stream:  # descrip, a field outside the geometry
        stream.seek(148)
        stream.write(b"reference scan")
    reference = read_volume(reference_file, dtype=None)
    mask = (np.arange(60).reshape(3, 4, 5) % 3).astype(np.uint8)
    file = tmp_path / "mask.nii"

    write_like(file, mask, reference, dtype=np.uint8)

    assert raw_geometry(file) == raw_geometry(reference_file)
    assert file.read_bytes()[148:228] == reference_file.read_bytes()[148:228]
    assert raw_datatype(file) == _DATATYPES[np.uint8][0]
    written = read_volume(file, dtype=None)
    np.testing.assert_array_equal(written.data, mask)
    np.testing.assert_array_equal(written.affine, reference.affine)
    np.testing.assert_array_equal(written.qform, reference.qform)
    assert (written.sform_code, written.qform_code) == (2, 1)

    with pytest.raises(InvalidInputError):
        write_like(tmp_path / "bad.nii", mask[:2], reference, dtype=np.uint8)
