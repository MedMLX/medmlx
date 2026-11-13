"""Independent MONAI 1.6.1 references for placement, padding, and FP32 blending."""

from collections.abc import Callable, Sequence
from typing import Literal, Never, TypedDict, cast

import mlx.core as mx
import numpy as np
import pytest
from fixture_cases import HostArray, load_fixture
from numpy.typing import DTypeLike
from real_images import real_image

from medmlx_core.sliding_window import FloatArray
from medmlx_core.typing import ArrayEvaluator
from medmlx_core.volume_workspace import VolumeWorkspace

pytest.importorskip("torch")
import torch

pytest.importorskip("monai")
from monai.data.utils import compute_importance_map as monai_importance
from monai.inferers import utils as monai_utils

monai_inference = cast(
    Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]],
    vars(monai_utils)["sliding_window_inference"],
)
from medmlx_core.sliding_window import (  # noqa: E402
    compute_importance_map,
    sliding_window_inference,
)

eval_arrays = cast(ArrayEvaluator, vars(mx)["eval"])

# Torch's NumPy adapter has an unannotated input in the installed stub.
from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])

type Coordinates = list[list[slice]]
type Mode = Literal["constant", "gaussian", "reflect", "replicate", "circular"]


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


# Representative full-size ROI/batch combinations; each is compared directly with MONAI.
WINDOW_SETTINGS = [
    pytest.param((240, 240, 160), 1, 0.5, "constant", id="large_anisotropic"),
    pytest.param((96, 96, 96), 4, 0.25, "replicate", id="batch4_replicate"),
]


def torch_predict(patch: torch.Tensor) -> torch.Tensor:
    anchor = patch[(slice(None), slice(0, 1), *(slice(0, 1),) * (patch.ndim - 2))]
    return torch.cat((patch * 0.5 + anchor * 0.25, patch * -0.25 + 0.125), dim=1)


def test_float64_host_windows_preserve_sub_float32_values_against_monai() -> None:
    values = real_image((1, 1, 11, 7)).astype(np.float64) + 1e8

    def host_prediction(patch: HostArray) -> HostArray:
        anchor = patch[:, :, :1, :1]
        return np.concatenate([patch * 0.5 + anchor * 0.25, patch * -0.25 + 0.125], axis=1)

    expected = cast(
        torch.Tensor, monai_inference(from_numpy(values), (6, 5), 3, torch_predict, overlap=0.5)
    )
    actual = sliding_window_inference(values, (6, 5), 3, host_prediction, overlap=0.5,
                                      predictor_input="numpy")
    np.testing.assert_array_equal(actual, expected.numpy())
    with VolumeWorkspace() as workspace:
        streamed = sliding_window_inference(values, (6, 5), 3, host_prediction, overlap=0.5,
                                            predictor_input="numpy", workspace=workspace)
        np.testing.assert_array_equal(streamed, actual)


@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_single_window_views_preserve_input_strides_without_copying(layout: str) -> None:
    values = real_image((2, 2, 6, 7, 8))
    if layout == "transposed":
        values = values.swapaxes(2, 4)
    calls = 0

    def predict(window: FloatArray, coordinates: Coordinates) -> FloatArray:
        nonlocal calls
        calls += 1
        assert len(coordinates) == 1
        assert window.strides == values.strides
        assert np.shares_memory(window, values)
        np.testing.assert_array_equal(window, values[tuple(coordinates[0])])
        return window

    result = sliding_window_inference(
        values, (4, 4, 4), 1, predict, overlap=0.5, with_coord=True,
        predictor_input="numpy", preserve_window_strides=True,
    )
    assert calls > 1
    np.testing.assert_array_equal(result, values)


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


@pytest.mark.parametrize(
    "shape,roi,overlap,mode,sw_batch_size",
    [
        ((11, 9, 7), (8, 6, 4), 0.25, "constant", 1),
        ((11, 9, 7), (8, 6, 4), 0.5, "gaussian", 3),
        ((3, 5, 2), (8, 8, 6), 0.5, "constant", 3),
        ((3, 5, 2), (8, 8, 6), 0.25, "gaussian", 1),
    ],
)
def test_patch_dependent_blending_matches_monai(
    shape: tuple[int, ...],
    roi: tuple[int, ...],
    overlap: float,
    mode: Mode,
    sw_batch_size: int,
) -> None:
    # Two images expose batch-crossing window groups; two channels expose channel placement.
    inputs = real_image((2, 2, *shape))
    original = inputs.copy()
    options = WindowOptions(overlap=overlap, mode=mode, padding_mode="constant", cval=-0.25)
    expected = monai_inference(from_numpy(inputs), roi, sw_batch_size, torch_predict, **options)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, roi, sw_batch_size, mlx_predict, **options)
    assert_parity(actual, expected.numpy(), exact=mode == "constant")
    with VolumeWorkspace() as workspace:
        streamed = sliding_window_inference(
            inputs, roi, sw_batch_size, mlx_predict, **options, workspace=workspace
        )
        np.testing.assert_array_equal(streamed, actual)
    assert np.array_equal(inputs, original)


@pytest.mark.parametrize(
    "mode,cval", [("constant", -0.75), ("reflect", 0), ("replicate", 0), ("circular", 0)]
)
def test_padding_modes_match_monai(
    mode: Mode, cval: float
) -> None:
    inputs = real_image((1, 2, 3, 5, 2)).swapaxes(2, 3)
    roi = (8, 6, 4)
    expected = monai_inference(from_numpy(inputs / 64), roi, 1, torch_predict,
                               padding_mode=mode, cval=cval)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs / 64, roi, 1, mlx_predict,
                                      padding_mode=mode, cval=cval)
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize(
    "patch,scale,mode",
    [
        ((8, 8, 6), 0.125, "constant"),
        ((7, 5, 3), (0.2, 0.125, 0.25), "gaussian"),
        ((1, 4, 5), 0.125, "gaussian"),
        ((8, 6, 4), 0.01, "gaussian"),
    ],
)
def test_importance_map_matches_monai_center_clamp_and_dtype(
    patch: tuple[int, ...],
    scale: Sequence[float] | float,
    mode: Mode,
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


@pytest.mark.parametrize("roi,sw_batch_size,overlap,padding_mode", WINDOW_SETTINGS)
def test_full_roi_window_settings_match_monai(
    roi: tuple[int, ...], sw_batch_size: int, overlap: float, padding_mode: str
) -> None:
    # Exercise the real patch dimensions and padding without a costly network.
    shape = tuple(size - 1 for size in roi)
    inputs = real_image((1, 1, *shape))

    def predict_torch(patch: torch.Tensor) -> torch.Tensor:
        assert patch.shape == (1, 1, *roi)
        return patch * 0.5 + 0.125

    def predict_mlx(patch: mx.array) -> mx.array:
        assert patch.shape == (1, 1, *roi)
        result = patch * 0.5
        eval_arrays(result)
        return result + 0.125

    options = WindowOptions(overlap=overlap, padding_mode=padding_mode, device="cpu")
    expected = monai_inference(from_numpy(inputs), roi, sw_batch_size, predict_torch, **options)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, roi, sw_batch_size, predict_mlx, **options)
    assert_parity(actual, expected.numpy(), exact=True)


def test_cached_gaussian_map_and_coordinate_predictor_are_bitwise_equal() -> None:
    inputs = real_image((2, 1, 9, 7))
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

    expected = monai_inference(from_numpy(inputs), (6, 4), 4, predict_torch, overlap=(0.5, 0.25),
                               roi_weight_map=from_numpy(weight), with_coord=True, scale=0.5,
                               bias=0.125)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(mx.array(inputs), (6, 4), 4, predict_mlx, overlap=(0.5, 0.25),
                                      roi_weight_map=weight, with_coord=True, scale=0.5,
                                      bias=0.125, device="cpu")
    assert actual_coords == expected_coords
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize(
    "input_dtype,prediction_dtype", [(np.float32, np.float16), (np.float16, np.float32)]
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("overlap_prediction", [False, True])
def test_predictor_weighting_precedes_accumulator_dtype_conversion(
    input_dtype: DTypeLike, prediction_dtype: DTypeLike, batch_size: int, overlap_prediction: bool,
) -> None:
    inputs = real_image((1, 1, 11, 8)).astype(input_dtype)
    inputs = inputs[..., ::-1]
    original = inputs.copy()
    weight = monai_importance((6, 4), "gaussian").numpy()
    torch_dtype = torch.float16 if prediction_dtype == np.float16 else torch.float32
    mlx_dtype = mx.float16 if prediction_dtype == np.float16 else mx.float32

    def torch_half(x: torch.Tensor) -> torch.Tensor:
        return torch_predict(x).to(torch_dtype)

    def mlx_half(x: mx.array) -> mx.array:
        return mlx_predict(x).astype(mlx_dtype)

    expected = monai_inference(from_numpy(np.ascontiguousarray(inputs)), (6, 4), batch_size,
                               torch_half, roi_weight_map=from_numpy(weight))
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, (6, 4), batch_size, mlx_half, roi_weight_map=weight,
                                      overlap_prediction=overlap_prediction)
    assert_parity(actual, expected.numpy(), exact=True)
    with VolumeWorkspace() as workspace:
        streamed = sliding_window_inference(
            inputs, (6, 4), batch_size, mlx_half, roi_weight_map=weight,
            overlap_prediction=overlap_prediction, workspace=workspace,
        )
        np.testing.assert_array_equal(streamed, actual)
    np.testing.assert_array_equal(inputs, original)


@pytest.mark.parametrize("roi", [4, (None, -1)])
def test_scalar_and_fallback_roi_match_monai(roi: Sequence[int | None] | int) -> None:
    inputs = real_image((1, 1, 5, 7))
    expected = monai_inference(from_numpy(inputs), roi, 2, torch_predict, overlap=0.5)
    assert isinstance(expected, torch.Tensor)
    actual = sliding_window_inference(inputs, roi, 2, mlx_predict, overlap=0.5)
    assert_parity(actual, expected.numpy(), exact=True)


@pytest.mark.parametrize("mode", ["constant", "gaussian"])
def test_recorded_monai_scores_coordinates_and_labels(
    mode: Mode,
) -> None:
    _, reference = load_fixture(f"sliding_window_{mode}")
    inputs, weights, bias = reference["inputs"], reference["weights"], reference["bias"]
    coordinates: list[list[int]] = []

    def predict(patch: mx.array, coords: Coordinates) -> mx.array:
        coordinates.extend([[int(s.start) for s in coord[2:]] for coord in coords])
        result = patch * mx.array(weights)
        eval_arrays(result)  # Torch's eager multiplication precedes addition.
        return result + mx.array(bias)

    roi = tuple(map(int, reference["roi_size"]))
    actual = sliding_window_inference(inputs, roi, 1, predict, mode=mode,
                                      padding_mode="replicate", with_coord=True)
    assert_parity(actual, reference["scores"], exact=mode == "constant")
    np.testing.assert_array_equal(coordinates, reference["coordinates"])
    # MONAI AsDiscrete(argmax=True): channel indices, with first-index ties.
    # Background thresholds and external class-id remaps belong to model packages.
    labels = np.argmax(actual, axis=1)[:, None].astype(np.uint8)
    np.testing.assert_array_equal(labels, reference["labels"])


@pytest.mark.parametrize("sw_device", [None, "gpu", mx.Device(mx.gpu)])
def test_gpu_device_forms_are_explicitly_admitted(
    sw_device: mx.Device | mx.DeviceType | str | None,
) -> None:
    inputs = real_image((1, 1, 3, 4))

    def predict(patch: mx.array) -> mx.array:
        assert mx.default_device().type == mx.gpu
        return patch

    actual = sliding_window_inference(inputs, (2, 3), 2, predict, sw_device=sw_device)
    assert np.array_equal(actual, inputs)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"overlap": 1}, ValueError),
        ({"buffer_steps": 2}, NotImplementedError),
        ({"device": "gpu"}, ValueError),
        ({"sw_device": mx.cpu}, ValueError),
    ],
)
def test_invalid_or_unimplemented_options_fail_before_prediction(
    options: RejectedOptions, error: type[Exception]
) -> None:
    def unexpected_call(_: mx.array) -> Never:
        pytest.fail("invalid request reached predictor")

    with pytest.raises(error):
        sliding_window_inference(
            real_image((1, 1, 3, 3)), 4, 1, unexpected_call, **options
        )


def multiple_outputs(x: mx.array) -> tuple[mx.array, mx.array]:
    return x, x


def scaled_output(x: mx.array) -> mx.array:
    return x[:, :, ::2, ::2]


@pytest.mark.parametrize(
    "predictor,error", [(multiple_outputs, TypeError), (scaled_output, ValueError)],
)
@pytest.mark.parametrize("overlap_prediction", [False, True])
def test_unsupported_predictor_outputs_fail_explicitly(
    predictor: Callable[[mx.array], object], error: type[Exception], overlap_prediction: bool,
) -> None:
    with pytest.raises(error):
        sliding_window_inference(real_image((1, 1, 4, 4)), 3, 1, predictor,
                                  overlap_prediction=overlap_prediction)
