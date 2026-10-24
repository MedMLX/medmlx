"""Independent MONAI 1.6.1 references for placement, padding, and FP32 blending."""

from collections.abc import Callable, Sequence
from typing import Literal, Never, TypedDict, cast

import mlx.core as mx
import numpy as np
import pytest
from fixture_cases import HostArray, load_fixture
from numpy.typing import DTypeLike

from medmlx_core.sliding_window import FloatArray
from medmlx_core.typing import ArrayEvaluator

pytest.importorskip("torch")
import torch

pytest.importorskip("monai")
from monai.data.utils import compute_importance_map as monai_importance
from monai.data.utils import dense_patch_slices as monai_slices
from monai.inferers import utils as monai_utils

monai_interval = cast(
    Callable[[Sequence[int], Sequence[int], int, Sequence[float]], tuple[int, ...]],
    vars(monai_utils)["_get_scan_interval"],
)
monai_inference = cast(
    Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]],
    vars(monai_utils)["sliding_window_inference"],
)
from torch.nn import functional as F  # noqa: E402

from medmlx_core import sliding_window as window_module  # noqa: E402
from medmlx_core.sliding_window import (  # noqa: E402
    compute_importance_map,
    dense_patch_slices,
    sliding_window_inference,
)

_get_scan_interval = cast(
    Callable[[Sequence[int], Sequence[int], int, Sequence[float]], tuple[int, ...]],
    vars(window_module)["_get_scan_interval"],
)
_pad_input = cast(
    Callable[[FloatArray, Sequence[int], str, float], tuple[FloatArray, tuple[slice, ...]]],
    vars(window_module)["_pad_input"],
)
eval_arrays = cast(ArrayEvaluator, vars(mx)["eval"])

# Torch's NumPy adapter has an unannotated input in the installed stub.
from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])

type Coordinates = list[list[slice]]


class RejectedOptions(TypedDict, total=False):
    device: str
    buffer_steps: int
    process_fn: Callable[..., object]
    progress: bool
    sw_device: mx.Device | mx.DeviceType | str


class WindowOptions(TypedDict, total=False):
    overlap: float
    mode: str
    padding_mode: str
    cval: float
    sigma_scale: float
    device: str
    buffer_steps: int | None


# Representative ROI/batch combinations; each is compared directly with MONAI.
WINDOW_SETTINGS = [
    pytest.param((240, 240, 160), 1, 0.5, "constant", id="large_anisotropic"),
    pytest.param((96, 96, 96), 4, 0.25, "constant", id="batch4_quarter"),
    pytest.param((96, 96, 96), 4, 0.5, "constant", id="batch4_half"),
    pytest.param((128, 128, 128), 1, 0.25, "replicate", id="replicate"),
]


def torch_predict(patch: torch.Tensor) -> torch.Tensor:
    anchor = patch[(slice(None), slice(0, 1), *(slice(0, 1),) * (patch.ndim - 2))]
    return torch.cat((patch * 0.5 + anchor * 0.25, patch * -0.25 + 0.125), dim=1)


def test_float64_host_windows_preserve_sub_float32_values_against_monai() -> None:
    values = np.arange(77, dtype=np.float64).reshape(1, 1, 11, 7) + 1e8

    def host_prediction(patch: HostArray) -> HostArray:
        anchor = patch[:, :, :1, :1]
        return np.concatenate([patch * 0.5 + anchor * 0.25, patch * -0.25 + 0.125], axis=1)

    expected = cast(
        torch.Tensor, monai_inference(from_numpy(values), (6, 5), 3, torch_predict, overlap=0.5)
    )
    actual = sliding_window_inference(
        values, (6, 5), 3, host_prediction, overlap=0.5, predictor_input="numpy"
    )
    np.testing.assert_array_equal(actual, expected.numpy())


def mlx_predict(patch: mx.array) -> mx.array:
    anchor = patch[(slice(None), slice(0, 1), *(slice(0, 1),) * (patch.ndim - 2))]
    left, offset, right = patch * 0.5, anchor * 0.25, patch * -0.25
    eval_arrays(left, offset, right)  # Separate multiply/add, like Torch eager execution.
    return mx.concatenate([left + offset, right + 0.125], axis=1)


def assert_parity(actual: HostArray, expected: HostArray, *, exact: bool = False) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if exact:
        assert np.array_equal(actual, expected)
    else:
        error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        # NumPy/Torch FP32 exp kernels differ by a rounding unit; weighting,
        # ordered addition, and division propagate that small Gaussian error.
        assert error <= 1e-6, f"max abs difference {error} exceeds 1e-6"


@pytest.mark.parametrize("shape,roi", [((11, 9, 7), (8, 6, 4)), ((3, 5, 2), (8, 8, 6))])
@pytest.mark.parametrize("overlap", [0.25, 0.5])
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
@pytest.mark.parametrize("sw_batch_size", [1, 3])
def test_patch_dependent_blending_matches_monai(
    shape: tuple[int, ...],
    roi: tuple[int, ...],
    overlap: float,
    mode: Literal["constant", "gaussian", "reflect", "replicate", "circular"],
    sw_batch_size: int,
) -> None:
    # Two images expose batch-crossing window groups; two channels expose channel placement.
    inputs = np.random.default_rng(1203).integers(-64, 65, (2, 2, *shape)).astype(np.float32) / 64
    original = inputs.copy()
    options = WindowOptions(overlap=overlap, mode=mode, padding_mode="constant", cval=-0.25)
    expected = monai_inference(from_numpy(inputs), roi, sw_batch_size, torch_predict, **options)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, roi, sw_batch_size, mlx_predict, **options)
    assert_parity(actual, expected.numpy(), exact=mode == "constant")
    assert np.array_equal(inputs, original)


@pytest.mark.parametrize(
    "mode,cval", [("constant", -0.75), ("reflect", 0), ("replicate", 0), ("circular", 0)]
)
def test_padding_values_and_crop_match_torch(
    mode: Literal["constant", "gaussian", "reflect", "replicate", "circular"], cval: float
) -> None:
    inputs = np.arange(60, dtype=np.float32).reshape(1, 2, 3, 5, 2).swapaxes(2, 3)
    roi = (8, 6, 4)
    padded, crop = _pad_input(inputs, roi, mode, cval)
    expected = F.pad(from_numpy(inputs), (1, 1, 1, 2, 1, 2), mode=mode, value=cval)
    assert np.array_equal(padded, expected.numpy())
    assert crop == (slice(1, 6), slice(1, 4), slice(1, 3))
    assert np.array_equal(padded[(slice(None), slice(None), *crop)], inputs)
    expected_logits = monai_inference(
        from_numpy(inputs / 64), roi, 1, torch_predict, padding_mode=mode, cval=cval
    )
    assert isinstance(expected_logits, torch.Tensor)
    actual = sliding_window_inference(
        inputs / 64, roi, 1, mlx_predict, padding_mode=mode, cval=cval
    )
    assert_parity(actual, expected_logits.numpy(), exact=True)


@pytest.mark.parametrize("overlap,axis0", [(0.25, [0, 6, 11]), (0.5, [0, 4, 8, 11])])
def test_coordinates_match_monai_edge_shift_and_order(overlap: float, axis0: list[int]) -> None:
    image, roi = (19, 11, 13), (8, 8, 8)
    amounts = (overlap,) * 3
    interval = _get_scan_interval(image, roi, 3, amounts)
    assert interval == monai_interval(image, roi, 3, amounts)
    windows = dense_patch_slices(image, roi, interval)
    assert windows == monai_slices(image, roi, interval)
    assert list(dict.fromkeys(window[0].start for window in windows)) == axis0
    assert windows[-1] == (slice(11, 19), slice(3, 11), slice(5, 13))
    assert dense_patch_slices(image, roi, interval, False) == monai_slices(
        image, roi, interval, False
    )
    assert _get_scan_interval((2, 9), (2, 3), 2, (0.5, 0.99)) == (2, 1)


@pytest.mark.parametrize(
    "patch,scale",
    [((8, 8, 6), 0.125), ((7, 5, 3), (0.2, 0.125, 0.25)), ((1, 4, 5), 0.125), ((8, 6, 4), 0.01)],
)
@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_importance_map_matches_monai_center_clamp_and_dtype(
    patch: tuple[int, ...],
    scale: Sequence[float] | float,
    mode: Literal["constant", "gaussian", "reflect", "replicate", "circular"],
) -> None:
    expected = monai_importance(patch, mode, scale).numpy()
    actual = compute_importance_map(patch, mode, scale)
    assert actual.dtype == np.float32
    if mode == "constant" or scale == 0.01:
        # A sufficiently narrow Gaussian is entirely clamped to 1e-3.
        assert np.array_equal(actual, expected)
    else:
        # Same FP32 separable operations; exp differs between the independent kernels.
        assert float(np.max(np.abs(actual - expected))) <= 1e-7
        assert actual.min() >= np.float32(1e-3)
        assert np.array_equal(actual, np.flip(actual))
    expected_half = monai_importance(patch, mode, scale, dtype=torch.float16).numpy()
    assert np.array_equal(compute_importance_map(patch, mode, scale, np.float16), expected_half)


@pytest.mark.parametrize("full_roi,sw_batch_size,overlap,padding_mode", WINDOW_SETTINGS)
@pytest.mark.parametrize("smaller_than_roi", [False, True], ids=["odd_overlap", "padding"])
def test_scaled_window_settings_match_monai(
    full_roi: tuple[int, ...],
    sw_batch_size: int,
    overlap: float,
    padding_mode: str,
    smaller_than_roi: bool,
) -> None:
    roi = tuple(size // 16 for size in full_roi)
    shape = tuple(size - 3 if smaller_than_roi else size * 2 + 1 for size in roi)
    inputs = np.random.default_rng(703).integers(-32, 33, (1, 1, *shape)).astype(np.float32) / 32
    options = WindowOptions(
        overlap=overlap,
        mode="constant",
        sigma_scale=0.125,
        padding_mode=padding_mode,
        cval=0,
        device="cpu",
        buffer_steps=None,
    )
    expected_tensor = monai_inference(
        from_numpy(inputs), roi, sw_batch_size, torch_predict, **options
    )
    assert isinstance(expected_tensor, torch.Tensor)
    expected = expected_tensor.numpy()
    actual = sliding_window_inference(inputs, roi, sw_batch_size, mlx_predict, **options)
    assert_parity(actual, expected, exact=True)


@pytest.mark.parametrize("roi,sw_batch_size,overlap,padding_mode", WINDOW_SETTINGS)
def test_full_roi_window_settings_match_monai(
    roi: tuple[int, ...], sw_batch_size: int, overlap: float, padding_mode: str
) -> None:
    # Exercise the real patch dimensions and padding without a costly network.
    shape = tuple(size - 1 for size in roi)
    inputs = np.random.default_rng(704).integers(-32, 33, (1, 1, *shape), dtype=np.int8)
    inputs = inputs.astype(np.float32) / 32

    def predict_torch(patch: torch.Tensor) -> torch.Tensor:
        assert patch.shape == (1, 1, *roi)
        return patch * 0.5 + 0.125

    def predict_mlx(patch: mx.array) -> mx.array:
        assert patch.shape == (1, 1, *roi)
        result = patch * 0.5
        eval_arrays(result)
        return result + 0.125

    options = WindowOptions(overlap=overlap, padding_mode=padding_mode, device="cpu")
    expected_tensor = monai_inference(
        from_numpy(inputs), roi, sw_batch_size, predict_torch, **options
    )
    assert isinstance(expected_tensor, torch.Tensor)
    expected = expected_tensor.numpy()
    actual = sliding_window_inference(inputs, roi, sw_batch_size, predict_mlx, **options)
    assert_parity(actual, expected, exact=True)


@pytest.mark.parametrize("overlap", [0.25, 0.5], ids=["quarter", "half"])
def test_single_window_batch_matches_monai(overlap: float) -> None:
    inputs = np.random.default_rng(705).integers(-32, 33, (1, 1, 13, 11, 5))
    inputs = inputs.astype(np.float32) / 32
    options = WindowOptions(overlap=overlap, device="cpu")
    expected = monai_inference(from_numpy(inputs), (6, 6, 6), 1, torch_predict, **options)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, (6, 6, 6), 1, mlx_predict, **options)
    assert_parity(actual, expected.numpy(), exact=True)


def test_cached_gaussian_map_and_coordinate_predictor_are_bitwise_equal() -> None:
    inputs = np.random.default_rng(11).integers(-32, 33, (2, 1, 9, 7)).astype(np.float32) / 32
    weight = monai_importance((6, 4), "gaussian", (0.2, 0.15)).numpy()
    expected_coords: Coordinates = []
    actual_coords: Coordinates = []

    def predict_torch(
        patch: torch.Tensor, coords: Coordinates, scale: float, *, bias: float
    ) -> torch.Tensor:
        expected_coords.extend(coords)
        # MONAI starts are np.int64, so these expressions are np.float64.
        # Without dtype=patch.dtype, Torch infers FP64 offsets/predictions and
        # weights in FP64 before casting to the FP32 accumulator. MLX uses FP32.
        offsets = torch.tensor(
            [int(c[2].start) / 32 + int(c[3].start) / 64 for c in coords], dtype=patch.dtype
        )[:, None, None, None]
        return patch * scale + offsets + bias

    def predict_mlx(patch: mx.array, coords: Coordinates, scale: float, *, bias: float) -> mx.array:
        actual_coords.extend(coords)
        offsets = mx.array([int(c[2].start) / 32 + int(c[3].start) / 64 for c in coords])[
            :, None, None, None
        ]
        result = patch * scale
        eval_arrays(result)
        return result + offsets + bias

    expected = monai_inference(
        from_numpy(inputs),
        (6, 4),
        4,
        predict_torch,
        overlap=(0.5, 0.25),
        roi_weight_map=from_numpy(weight),
        with_coord=True,
        scale=0.5,
        bias=0.125,
    )
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(
        mx.array(inputs),
        (6, 4),
        4,
        predict_mlx,
        overlap=(0.5, 0.25),
        roi_weight_map=weight,
        with_coord=True,
        scale=0.5,
        bias=0.125,
        device="cpu",
    )
    assert actual_coords == expected_coords
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize(
    "input_dtype,prediction_dtype", [(np.float32, np.float16), (np.float16, np.float32)]
)
def test_predictor_weighting_precedes_accumulator_dtype_conversion(
    input_dtype: DTypeLike, prediction_dtype: DTypeLike
) -> None:
    inputs = np.random.default_rng(41).integers(-32, 33, (1, 1, 9, 7)).astype(input_dtype) / 32
    weight = monai_importance((6, 4), "gaussian").numpy()
    torch_dtype = torch.float16 if prediction_dtype == np.float16 else torch.float32
    mlx_dtype = mx.float16 if prediction_dtype == np.float16 else mx.float32

    def torch_half(x: torch.Tensor) -> torch.Tensor:
        return torch_predict(x).to(torch_dtype)

    def mlx_half(x: mx.array) -> mx.array:
        return mlx_predict(x).astype(mlx_dtype)

    expected = monai_inference(
        from_numpy(inputs),
        (6, 4),
        1,
        torch_half,
        roi_weight_map=from_numpy(weight),
    )
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, (6, 4), 1, mlx_half, roi_weight_map=weight)
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize("roi", [4, (None, -1)])
def test_scalar_and_fallback_roi_match_monai(roi: Sequence[int | None] | int) -> None:
    inputs = np.arange(35, dtype=np.float32).reshape(1, 1, 5, 7) / 64
    expected = monai_inference(from_numpy(inputs), roi, 2, torch_predict, overlap=0.5)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, roi, 2, mlx_predict, overlap=0.5)
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_recorded_monai_scores_coordinates_and_labels(
    mode: Literal["constant", "gaussian", "reflect", "replicate", "circular"],
) -> None:
    _, reference = load_fixture(f"sliding_window_{mode}")
    inputs, weights, bias = reference["inputs"], reference["weights"], reference["bias"]
    coordinates: list[list[int]] = []

    def predict(patch: mx.array, coords: Coordinates) -> mx.array:
        coordinates.extend([[int(s.start) for s in coord[2:]] for coord in coords])
        result = patch * mx.array(weights)
        eval_arrays(result)  # Torch's eager multiplication precedes addition.
        return result + mx.array(bias)

    actual = sliding_window_inference(
        inputs,
        tuple(map(int, reference["roi_size"])),
        1,
        predict,
        mode=mode,
        padding_mode="replicate",
        with_coord=True,
    )
    assert_parity(actual, reference["scores"], exact=mode == "constant")
    np.testing.assert_array_equal(coordinates, reference["coordinates"])
    # MONAI AsDiscrete(argmax=True): channel indices, with first-index ties.
    # Background thresholds and external class-id remaps belong to model packages.
    labels = np.argmax(actual, axis=1)[:, None].astype(np.uint8)
    np.testing.assert_array_equal(labels, reference["labels"])


@pytest.mark.parametrize("sw_device", [None, "gpu", mx.gpu, mx.Device(mx.gpu)])
def test_gpu_device_forms_are_explicitly_admitted(
    sw_device: mx.Device | mx.DeviceType | str | None,
) -> None:
    inputs = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)

    def predict(patch: mx.array) -> mx.array:
        assert mx.default_device().type == mx.gpu
        return patch

    actual = sliding_window_inference(inputs, (2, 3), 2, predict, sw_device=sw_device)
    assert np.array_equal(actual, inputs)


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"overlap": 1}, ValueError, "overlap must"),
        ({"buffer_steps": 2}, NotImplementedError, "buffer_steps"),
        ({"device": "gpu"}, ValueError, "device must"),
        ({"sw_device": "cpu"}, ValueError, "sw_device must"),
        ({"sw_device": mx.cpu}, ValueError, "sw_device must"),
        ({"sw_device": mx.Device(mx.cpu)}, ValueError, "sw_device must"),
    ],
)
def test_invalid_or_unimplemented_options_fail_before_prediction(
    options: RejectedOptions, error: type[Exception], message: str
) -> None:
    def unexpected_call(_: mx.array) -> Never:
        pytest.fail("invalid request reached predictor")

    with pytest.raises(error, match=message):
        sliding_window_inference(
            np.zeros((1, 1, 3, 3), np.float32), 4, 1, unexpected_call, **options
        )


def multiple_outputs(x: mx.array) -> tuple[mx.array, mx.array]:
    return x, x


def scaled_output(x: mx.array) -> mx.array:
    return x[:, :, ::2, ::2]


@pytest.mark.parametrize(
    "predictor,error,message",
    [
        (multiple_outputs, TypeError, "one MLX array"),
        (scaled_output, ValueError, "ROI spatial shape"),
    ],
)
def test_unsupported_predictor_outputs_fail_explicitly(
    predictor: Callable[[mx.array], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        sliding_window_inference(np.zeros((1, 1, 4, 4), np.float32), 4, 1, predictor)
