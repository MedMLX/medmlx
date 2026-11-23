"""Bit-exact parity of ``medmlx_core.transforms`` with MONAI 1.6.1 and PyTorch."""

from collections.abc import Callable
from types import ModuleType
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from real_images import real_image

from medmlx_core import transforms
from medmlx_core.errors import InvalidInputError

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]


class _Tensor(Protocol):
    def numpy(self) -> npt.NDArray[np.generic]: ...


class _MetaTensor(_Tensor, Protocol):
    @property
    def affine(self) -> _Tensor: ...


class _Transform(Protocol):
    def __call__(self, data: _MetaTensor, **options: object) -> _MetaTensor: ...

    def inverse(self, data: _MetaTensor) -> _MetaTensor: ...


class _Monai:
    """The MONAI 1.6.1 and Torch entry points the tests compare against."""

    def __init__(self) -> None:
        torch = pytest.importorskip("torch")
        pytest.importorskip("monai")
        import monai.data
        import monai.data.utils
        import monai.transforms
        import monai.transforms.utils

        cast(Callable[[int], None], vars(torch)["set_num_threads"])(1)
        self._torch: ModuleType = torch
        self._data = vars(monai.data)
        self._utils = {**vars(monai.data.utils), **vars(monai.transforms.utils)}
        self._transforms = vars(monai.transforms)

    def array(self, data: npt.NDArray[np.generic]) -> object:
        from_numpy = vars(self._torch)["from_numpy"]
        return cast(Callable[[npt.NDArray[np.generic]], object], from_numpy)(data.copy())

    def tensor(self, data: npt.NDArray[np.generic], affine: Float64Array) -> _MetaTensor:
        meta = cast(Callable[..., _MetaTensor], self._data["MetaTensor"])
        return meta(self.array(data), affine=self.array(affine))

    def transform(self, name: str, **options: object) -> _Transform:
        return cast(Callable[..., _Transform], self._transforms[name])(**options)

    def utility(self, name: str) -> Callable[..., object]:
        return cast(Callable[..., object], self._utils[name])


@pytest.fixture(scope="module")
def monai() -> _Monai:
    return _Monai()


def _affine(
    spacing: tuple[float, float, float],
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
    flips: tuple[int, int, int] = (1, 1, 1),
    shear: float = 0.0,
) -> Float64Array:
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rx @ ry @ rz @ np.diag(np.multiply(spacing, flips))
    result[0, 1] += shear
    result[:3, 3] = (10.3, -20.7, 5.1)
    return result


# Real CT crops on awkward grids: anisotropic, flipped, oblique, sheared, odd sizes,
# and a target that keeps one axis (nonpositive pixdim) or matches within AFFINE_TOL.
GRIDS = {
    "anisotropic": ((11, 13, 9), _affine((0.7, 0.8, 2.5)), (1.5, 1.5, 2.0)),
    "flipped": ((12, 9, 10), _affine((0.9, 1.1, 3.0), flips=(-1, 1, -1)), (1.0, 1.0, 1.0)),
    "oblique": ((10, 11, 12), _affine((0.8, 0.9, 1.7), angles=(0.3, -0.2, 0.5)), (1.2, 1.1, 1.3)),
    "sheared": (
        (9, 10, 11),
        _affine((1.0, 1.0, 2.0), angles=(0.1, 0.05, -0.2), shear=0.2),
        (1.5, 1.5, 1.5),
    ),
    "odd": ((7, 5, 3), _affine((2.3, 0.6, 1.9), flips=(1, -1, 1)), (1.0, 2.0, 0.7)),
    "kept axis": ((16, 12, 8), _affine((1.5, 1.5, 4.0)), (1.5, 1.0, -1.0)),
}


def _image(shape: tuple[int, int, int], channels: int = 2) -> Float32Array:
    return real_image((1, channels, *shape))[0] * np.float32(1024)


def _assert_identical(actual: npt.NDArray[np.generic], expected: npt.NDArray[np.generic]) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_spacing_and_inverse_match_monai(monai: _Monai, grid: str, mode: str) -> None:
    shape, affine, pixdim = GRIDS[grid]
    data = _image(shape)
    spacing = monai.transform("Spacing", pixdim=pixdim, mode=mode, padding_mode="border")
    expected = spacing(monai.tensor(data, affine))
    resample_mode: transforms.Mode = "nearest" if mode == "nearest" else "bilinear"
    values, out_affine = transforms.spacing(data, affine, pixdim, mode=resample_mode)
    _assert_identical(values, cast(Float32Array, expected.numpy()))
    _assert_identical(out_affine, cast(Float64Array, expected.affine.numpy()))
    # Invertd replays SpatialResample back to the recorded affine and shape.
    restored = spacing.inverse(expected)
    back, back_affine = transforms.spatial_resample(
        values, out_affine, affine, shape, mode=resample_mode
    )
    _assert_identical(back, cast(Float32Array, restored.numpy()))
    _assert_identical(back_affine, cast(Float64Array, restored.affine.numpy()))


@pytest.mark.parametrize("grid", ["anisotropic", "oblique"])
def test_resampled_argmax_matches_monai(monai: _Monai, grid: str) -> None:
    shape, affine, pixdim = GRIDS[grid]
    target, out_shape = transforms.spacing_grid(affine, shape, pixdim)
    # Few distinct values make many channel ties and float32 rounding ties.
    data = np.rint(_image(shape, channels=5) / np.float32(256))
    resample = monai.transform("SpatialResample", mode="bilinear", padding_mode="border")
    sampled = resample(
        monai.tensor(data, affine), dst_affine=monai.array(target), spatial_size=out_shape
    )
    expected = monai.transform("AsDiscrete", argmax=True)(sampled)
    labels, _ = transforms.spatial_resample(data, affine, target, out_shape, argmax=True)
    _assert_identical(labels, cast(Float32Array, expected.numpy()))
    _assert_identical(transforms.channel_argmax(cast(Float32Array, sampled.numpy())), labels)


@pytest.mark.parametrize(
    "special", [np.nan, np.inf, -np.inf, None], ids=["nan", "inf", "-inf", "mixed"]
)
def test_non_finite_inputs_resample_like_monai(monai: _Monai, special: float | None) -> None:
    """NaN and infinite voxels among finite ones reach values and labels as in ATen."""
    rng = np.random.default_rng(0)
    for grid in ("anisotropic", "oblique"):
        shape, affine, pixdim = GRIDS[grid]
        data = _image(shape, channels=3)
        for index in range(6):
            value = (np.nan, np.inf, -np.inf)[index % 3] if special is None else special
            data[(index % 3, *(int(rng.integers(1, size - 1)) for size in shape))] = value
        spacing = monai.transform("Spacing", pixdim=pixdim, padding_mode="border")
        expected = spacing(monai.tensor(data, affine))
        values, out_affine = transforms.spacing(data, affine, pixdim)
        _assert_identical(values, cast(Float32Array, expected.numpy()))
        back, _ = transforms.spatial_resample(values, out_affine, affine, shape)
        _assert_identical(back, cast(Float32Array, spacing.inverse(expected).numpy()))
        labels = monai.transform("AsDiscrete", argmax=True)(expected)
        target, out_shape = transforms.spacing_grid(affine, shape, pixdim)
        _assert_identical(
            transforms.spatial_resample(data, affine, target, out_shape, argmax=True)[0],
            cast(Float32Array, labels.numpy()),
        )


def test_near_ties_resample_exactly(monai: _Monai) -> None:
    """Halving the spacing samples midpoints of integers: exact float32 rounding ties."""
    shape = (9, 8, 7)
    affine = _affine((2.0, 2.0, 2.0))
    data = np.rint(_image(shape, channels=3) * np.float32(4096))
    spacing = monai.transform("Spacing", pixdim=(1.0, 1.0, 1.0), padding_mode="border")
    expected = spacing(monai.tensor(data, affine))
    values, _ = transforms.spacing(data, affine, (1.0, 1.0, 1.0))
    _assert_identical(values, cast(Float32Array, expected.numpy()))


def test_unchanged_grid_returns_input(monai: _Monai) -> None:
    shape, affine, _ = GRIDS["anisotropic"]
    data = _image(shape)
    values, out_affine = transforms.spatial_resample(data, affine, affine + 1e-4, shape)
    assert values is data
    _assert_identical(out_affine, affine)
    expected = monai.transform("SpatialResample")(
        monai.tensor(data, affine), dst_affine=monai.array(affine + 1e-4), spatial_size=shape
    )
    _assert_identical(values, cast(Float32Array, expected.numpy()))


@pytest.mark.parametrize("axcodes", ["RAS", "LPS", "PIR", "SAL"])
def test_orientation_and_inverse_match_monai(monai: _Monai, axcodes: str) -> None:
    affine = _affine((0.7, 0.8, 2.5), angles=(0.1, 0.4, -0.3), flips=(-1, 1, -1))
    data = _image((5, 6, 7))
    expected = monai.transform(
        "Orientation", axcodes=axcodes, labels=(("L", "R"), ("P", "A"), ("I", "S"))
    )(monai.tensor(data, affine))
    oriented, oriented_affine = transforms.orient(data, affine, axcodes)
    _assert_identical(oriented, cast(Float32Array, expected.numpy()))
    _assert_identical(oriented_affine, cast(Float64Array, expected.affine.numpy()))
    assert transforms.axis_codes(oriented_affine) == axcodes
    restored, _ = transforms.orient(oriented, oriented_affine, transforms.axis_codes(affine))
    _assert_identical(restored, data)


@pytest.mark.parametrize(
    "bounds", [(-175.0, 250.0, 0.0, 1.0), (-963.8247715525971, 1053.678477684517, 0.0, 1.0),
               (-1000.0, 1000.0, -1.0, 2.0), (40.0, 40.0, 0.5, 1.0)]
)
def test_scale_intensity_range_matches_monai(
    monai: _Monai, bounds: tuple[float, float, float, float]
) -> None:
    a_min, a_max, b_min, b_max = bounds
    data = _image((6, 7, 8), channels=1)
    expected = monai.transform(
        "ScaleIntensityRange", a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
    )(monai.tensor(data, np.eye(4)))
    _assert_identical(
        transforms.scale_intensity_range(data, a_min, a_max, b_min, b_max),
        cast(Float32Array, expected.numpy()),
    )


def test_scale_intensity_range_computes_float64_input_in_float64(monai: _Monai) -> None:
    data = np.random.default_rng(0).uniform(-1100.0, 1500.0, (1, 6, 7, 8))
    for a_min, a_max, b_min, b_max in ((-963.8247715525971, 1053.678477684517, 0.0, 1.0),
                                       (40.0, 40.0, 0.5, 1.0)):
        expected = monai.transform(
            "ScaleIntensityRange", a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
        )(monai.tensor(data, np.eye(4)))
        # MONAI keeps float64 for ``a_min == a_max``; the contract is float32 output.
        _assert_identical(
            transforms.scale_intensity_range(data, a_min, a_max, b_min, b_max),
            expected.numpy().astype(np.float32),
        )


@pytest.mark.parametrize("margin", [0, 3, 20])
def test_foreground_box_matches_monai(monai: _Monai, margin: int) -> None:
    data = _image((12, 10, 9))
    data[:, :3] = -1
    data[:, :, -2:] = 0
    box = monai.utility("generate_spatial_bounding_box")
    start, end = cast(tuple[list[int], list[int]], box(data, margin=margin, allow_smaller=True))
    assert transforms.foreground_box(data, margin) == (tuple(start), tuple(end))
    assert transforms.foreground_box(np.zeros_like(data), margin) == ((0, 0, 0), (0, 0, 0))


class _Header(Protocol):
    def __getitem__(self, key: str) -> npt.NDArray[np.generic]: ...

    def get_zooms(self) -> tuple[float, ...]: ...

    def set_zooms(self, zooms: tuple[float, ...]) -> None: ...


class _Nifti(Protocol):
    @property
    def affine(self) -> Float64Array: ...

    @property
    def header(self) -> _Header: ...

    def get_sform(self) -> Float64Array: ...

    def get_qform(self) -> Float64Array: ...

    def set_sform(self, affine: Float64Array, code: int) -> None: ...

    def set_qform(self, affine: Float64Array, code: int) -> None: ...


def _nifti(
    version: int,
    sform: Float64Array,
    sform_code: int,
    qform: Float64Array,
    qform_code: int,
    zooms: tuple[float, float, float],
) -> _Nifti:
    image_type = cast(
        Callable[[Float32Array, None], _Nifti], nib.Nifti2Image if version == 2 else nib.Nifti1Image
    )
    image = image_type(np.zeros((4, 5, 6), dtype=np.float32), None)
    image.set_qform(qform, code=qform_code)
    image.set_sform(sform, code=sform_code)
    image.header.set_zooms(zooms)
    return image


# nibabel's get_qform() is built from pixdim, so after set_zooms the qform always
# matches the zooms: a mismatching sform with a nonzero code takes the qform.
HEADERS = {
    "consistent": (1, 1, (0.7, 0.8, 2.5)),
    "sform replaced by qform": (1, 1, (1.0, 1.0, 1.0)),
    "qform is the affine": (0, 1, (1.0, 1.0, 1.0)),
    "no codes": (0, 0, (1.0, 1.0, 1.0)),
}


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("case", sorted(HEADERS))
def test_header_correction_matches_monai(monai: _Monai, version: int, case: str) -> None:
    sform_code, qform_code, zooms = HEADERS[case]
    sform = _affine((0.7, 0.8, 2.5), angles=(0.2, -0.1, 0.3), flips=(-1, 1, 1))
    qform = _affine((0.9, 1.1, 3.0), angles=(0.0, 0.2, 0.0))
    image = _nifti(version, sform, sform_code, qform, qform_code, zooms)
    header = image.header
    arrays = (
        np.array(image.affine), header.get_zooms(), image.get_sform(),
        int(header["sform_code"]), image.get_qform(), int(header["qform_code"]),
    )
    corrected = cast(_Nifti, monai.utility("correct_nifti_header_if_necessary")(image))
    affine, out_zooms = transforms.correct_nifti_header(
        *arrays, header_float=np.float64 if version == 2 else np.float32
    )
    _assert_identical(affine, np.asarray(corrected.affine, dtype=np.float64))
    _assert_identical(out_zooms, np.asarray(corrected.header.get_zooms()[:3], dtype=np.float64))


def test_unsupported_inputs_raise() -> None:
    data = _image((4, 5, 6))
    with pytest.raises(InvalidInputError):
        wrong = cast(Float32Array, data.astype(np.float64))
        transforms.spatial_resample(wrong, np.eye(4), np.eye(4) * 2, (4, 5, 6))
    with pytest.raises(InvalidInputError):
        transforms.spacing(data, np.eye(4), (2.0,), mode=cast(transforms.Mode, "bicubic"))
    with pytest.raises(InvalidInputError):
        transforms.orient(data, np.eye(4), "RAR")
