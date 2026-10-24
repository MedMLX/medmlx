# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under the Apache License, Version 2.0.
# See the project-root THIRD_PARTY_NOTICES.md for attribution and full license terms.

"""MONAI 1.6.0 window placement and host stitching for MLX predictors.

Only NumPy and MLX are required. Inputs and outputs are channel first with a
batch dimension. Stitching uses the input dtype and MONAI's sequential patch
order; only the predictor runs on the MLX device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Literal, cast, overload

import mlx.core as mx
import numpy as np
from numpy.typing import DTypeLike, NDArray

from medmlx_core.runtime import import_mlx
from medmlx_core.typing import ArrayFactory, HostArray, MlxRuntime

type FloatArray = NDArray[np.floating]

__all__ = ["compute_importance_map", "dense_patch_slices", "sliding_window_inference"]


def _tuple_rep[T: int | float | None](value: T | Sequence[T], dimensions: int) -> tuple[T, ...]:
    if np.isscalar(value) or value is None:
        return (cast(T, value),) * dimensions
    result = tuple(value)
    if len(result) != dimensions:
        raise ValueError(f"Expected {dimensions} values, got {len(result)}")
    return result


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: Sequence[float],
) -> tuple[int, ...]:
    """Use integer strides, with a minimum of one voxel, as MONAI does."""
    if len(image_size) != num_spatial_dims or len(roi_size) != num_spatial_dims:
        raise ValueError("image_size and roi_size must match num_spatial_dims")
    return tuple(
        int(roi) if roi == size else max(int(roi * (1 - amount)), 1)
        for size, roi, amount in zip(image_size, roi_size, overlap, strict=True)
    )


@overload
def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: Literal[True] = True,
) -> list[tuple[slice, ...]]: ...
@overload
def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: Literal[False],
) -> list[tuple[tuple[int, int], ...]]: ...
@overload
def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool,
) -> list[tuple[slice, ...]] | list[tuple[tuple[int, int], ...]]: ...
def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> list[tuple[slice, ...]] | list[tuple[tuple[int, int], ...]]:
    """Enumerate MONAI's row-major windows, shifting the last to the edge."""
    patch = tuple(
        min(size, roi) if roi else size for size, roi in zip(image_size, patch_size, strict=True)
    )
    starts: list[list[int]] = []
    for size, roi, step in zip(image_size, patch, scan_interval, strict=True):
        if size < 1 or roi < 1 or step < 0:
            raise ValueError("Image/patch sizes must be positive and strides nonnegative")
        count = (size - roi + step - 1) // step + 1 if step else 1
        starts.append([min(index * step, size - roi) for index in range(count)])
    if return_slice:
        return [
            tuple(slice(start, start + roi) for start, roi in zip(origin, patch, strict=True))
            for origin in product(*starts)
        ]
    return [
        tuple((start, start + roi) for start, roi in zip(origin, patch, strict=True))
        for origin in product(*starts)
    ]


@overload
def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: type[np.float32] = np.float32,
) -> NDArray[np.float32]: ...
@overload
def compute_importance_map[T: np.generic](
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    *,
    dtype: type[T] | np.dtype[T],
) -> NDArray[T]: ...
@overload
def compute_importance_map[T: np.generic](
    patch_size: Sequence[int],
    mode: str,
    sigma_scale: Sequence[float] | float,
    dtype: type[T] | np.dtype[T],
) -> NDArray[T]: ...
@overload
def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: DTypeLike = np.float32,
) -> HostArray: ...
def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: DTypeLike = np.float32,
) -> HostArray:
    """Return MONAI's separable FP32 Gaussian, clamped at a minimum of 1e-3.

    The Gaussian is centered at (size - 1) / 2, without peak normalization.
    NumPy and Torch exp kernels can differ by an FP32 rounding unit.
    """
    patch = tuple(patch_size)
    if not patch or any(size < 1 for size in patch):
        raise ValueError("patch_size must contain positive sizes")
    if mode == "constant":
        importance = np.ones(patch, dtype=np.float32)
    elif mode == "gaussian":
        importance = np.ones((), dtype=np.float32)
        scales = _tuple_rep(sigma_scale, len(patch))
        for axis, (size, scale) in enumerate(zip(patch, scales, strict=True)):
            sigma = size * scale
            x = np.arange(size, dtype=np.float32) - np.float32((size - 1) / 2)
            x = np.exp(x**2 / np.float32(-2 * sigma**2))
            importance = importance[..., None] * x[(None,) * axis] if axis else x
    else:
        raise ValueError(f"Unsupported mode: {mode}; expected constant or gaussian")
    minimum = max(float(importance.min()), 1e-3)
    return np.maximum(importance, np.float32(minimum)).astype(dtype)


def _pad_input(
    inputs: FloatArray, roi_size: Sequence[int], padding_mode: str, cval: float
) -> tuple[FloatArray, tuple[slice, ...]]:
    padding = [(0, 0), (0, 0)]
    crop: list[slice] = []
    for size, roi in zip(inputs.shape[2:], roi_size, strict=True):
        extra = max(roi - size, 0)
        before = extra // 2
        padding.append((before, extra - before))
        crop.append(slice(before, before + size))
    if not any(before or after for before, after in padding):
        return inputs, tuple(crop)
    modes: dict[str, Literal["constant", "reflect", "edge", "wrap"]] = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if padding_mode not in modes:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    if padding_mode != "constant" and cval != 0:
        raise ValueError(f"Padding mode {padding_mode} does not accept a nonzero cval")
    for size, (before, after) in zip(inputs.shape[2:], padding[2:], strict=True):
        if padding_mode == "reflect" and max(before, after) >= size:
            raise ValueError("Reflect padding must be smaller than the input dimension")
        if padding_mode == "circular" and max(before, after) > size:
            raise ValueError("Circular padding cannot wrap more than once")
    if padding_mode == "constant":
        padded = np.pad(inputs, padding, mode="constant", constant_values=cval)
    else:
        padded = np.pad(inputs, padding, mode=modes[padding_mode])
    return padded, tuple(crop)


def _positive_batch_size(value: object) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError("sw_batch_size must be a positive integer")
    return value


def sliding_window_inference(
    inputs: HostArray | mx.array,
    roi_size: Sequence[int | None] | int,
    sw_batch_size: int,
    predictor: Callable[..., object],
    overlap: Sequence[float] | float = 0.25,
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    sw_device: mx.Device | mx.DeviceType | str | None = None,
    device: str | None = None,
    progress: bool = False,
    roi_weight_map: FloatArray | mx.array | None = None,
    process_fn: Callable[..., object] | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    *args: object,
    predictor_input: Literal["mlx", "numpy"] = "mlx",
    **kwargs: object,
) -> FloatArray:
    """Predict and blend 1D/2D/3D windows, returning host NumPy logits.

    The predictor receives MLX ``(N, C, *roi_size)`` arrays and must return one
    MLX array at the same spatial resolution (output channels may differ).
    Output dtype follows the input, including when predictor dtype differs.
    ``predictor_input='numpy'`` explicitly passes/receives host NumPy windows,
    allowing float64 stitching around a predictor that exports paired Metal words.
    ``with_coord`` passes MONAI's list of batch/channel/spatial slice lists.

    ``sw_device`` is None, 'gpu', mx.gpu or an MLX GPU device; prediction requires Metal.
    ``device`` is None or 'cpu' because stitching is on the host. Active buffering, process_fn,
    progress bars, multiple outputs, and scaled outputs are unsupported.
    None/nonpositive ROI components use the corresponding input size.

    Args:
        inputs: ``(N, C, *spatial)`` array with 1 to 3 spatial axes, float16 or
            float32 (also float64 with ``predictor_input='numpy'``).
        roi_size: Window size per spatial axis, or one int for all axes.
        sw_batch_size: Maximum number of windows per predictor call, counted
            across images.
        predictor: Called as ``predictor(windows, *args, **kwargs)``, or
            ``predictor(windows, coordinates, *args, **kwargs)`` with ``with_coord``.
        overlap: Fractional window overlap in ``[0, 1)``, per axis or shared.
        mode: Window weighting, ``'constant'`` or ``'gaussian'``.
        sigma_scale: Gaussian sigma as a fraction of the window size.
        padding_mode: Padding for images smaller than the ROI: ``'constant'``,
            ``'reflect'``, ``'replicate'`` or ``'circular'``.
        cval: Fill value for ``'constant'`` padding; other modes require 0.
        roi_weight_map: Window weights replacing the ``mode`` map, shaped
            ``roi_size`` or ``(1, 1, *roi_size)``.
        *args, **kwargs: Passed through to ``predictor``.

    Returns:
        Host array ``(N, C_out, *spatial)`` of window predictions averaged with
        the importance weights, cropped back to the unpadded input size.

    Raises:
        ValueError: An argument or the predictor output has an invalid shape or value.
        TypeError: ``inputs`` has an unsupported dtype, or the predictor returns
            the wrong array type.
        NotImplementedError: ``buffer_steps``, ``process_fn`` or ``progress`` is used.
        MissingDependencyError: Metal is unavailable.
    """
    if device not in (None, "cpu"):
        raise ValueError("device must be None or 'cpu' for NumPy stitching")
    if buffer_steps is not None and buffer_steps > 0:
        raise NotImplementedError("Positive buffer_steps is not supported")
    if process_fn is not None or progress:
        raise NotImplementedError("process_fn and progress bars are not supported")
    if (
        sw_device is not None
        and sw_device != "gpu"
        and sw_device != mx.gpu
        and (not isinstance(sw_device, mx.Device) or sw_device.type != mx.gpu)
    ):
        raise ValueError("sw_device must be None, 'gpu', or an MLX GPU device")
    if isinstance(sw_device, str):
        sw_device = mx.gpu
    raw_host: HostArray = np.asarray(inputs)
    spatial = raw_host.shape[2:]
    if not 1 <= len(spatial) <= 3 or any(size < 1 for size in raw_host.shape):
        raise ValueError("inputs must be a nonempty (N, C, *spatial) 1D/2D/3D array")
    if predictor_input not in {"mlx", "numpy"}:
        raise ValueError("predictor_input must be 'mlx' or 'numpy'")
    allowed_dtypes = (np.dtype(np.float16), np.dtype(np.float32))
    if predictor_input == "numpy":
        allowed_dtypes = (*allowed_dtypes, np.dtype(np.float64))
    if raw_host.dtype not in allowed_dtypes:
        raise TypeError(
            "inputs must have float16 or float32 dtype; NumPy predictors also admit float64"
        )
    host = cast(FloatArray, raw_host)
    sw_batch_size = _positive_batch_size(sw_batch_size)
    roi = tuple(
        int(value) if value is not None and value > 0 else size
        for value, size in zip(_tuple_rep(roi_size, len(spatial)), spatial, strict=True)
    )
    amounts = _tuple_rep(overlap, len(spatial))
    if any(not 0 <= amount < 1 for amount in amounts):
        raise ValueError(f"overlap must be >= 0 and < 1, got {amounts}")
    padded, crop = _pad_input(host, roi, padding_mode, cval)
    image_size = padded.shape[2:]
    intervals = _get_scan_interval(image_size, roi, len(spatial), amounts)
    windows = dense_patch_slices(image_size, roi, intervals)
    importance = (
        compute_importance_map(roi, mode, sigma_scale, host.dtype)
        if roi_weight_map is None
        else np.asarray(roi_weight_map, dtype=host.dtype)
    )
    if importance.shape not in (roi, (1, 1, *roi)):
        raise ValueError("roi_weight_map must have shape roi_size or (1, 1, *roi_size)")
    importance = importance.reshape(1, 1, *roi)
    # Admit Metal explicitly before invoking the predictor.
    runtime = cast(MlxRuntime, import_mlx())
    make_array = cast(ArrayFactory, runtime.array)
    output = None
    counts = np.zeros((1, 1, *image_size), dtype=host.dtype)
    # MONAI builds this once per spatial window, independently of image batch.
    for window in windows:
        counts[(slice(None), slice(None), *window)] += importance
    num_windows = len(windows)
    for start in range(0, num_windows * host.shape[0], sw_batch_size):
        coordinates = [
            [
                slice(index // num_windows, index // num_windows + 1),
                slice(None),
                *windows[index % num_windows],
            ]
            for index in range(start, min(start + sw_batch_size, num_windows * host.shape[0]))
        ]
        patches = np.concatenate([padded[tuple(coord)] for coord in coordinates], axis=0)
        with mx.stream(sw_device if sw_device is not None else mx.default_device()):
            batch = patches if predictor_input == "numpy" else make_array(patches)
            prediction = (
                predictor(batch, coordinates, *args, **kwargs)
                if with_coord
                else predictor(batch, *args, **kwargs)
            )
            if predictor_input == "mlx":
                if not isinstance(prediction, mx.array):
                    raise TypeError("predictor must return one MLX array")
                runtime.eval(prediction)
            elif not isinstance(prediction, np.ndarray):
                raise TypeError("NumPy predictor must return one NumPy array")
            predicted = cast(
                FloatArray, np.array(cast(HostArray | mx.array, prediction), copy=True)
            )
        if (
            predicted.ndim != host.ndim
            or predicted.shape[0] != len(coordinates)
            or predicted.shape[2:] != roi
            or predicted.shape[1] < 1
        ):
            raise ValueError("predictor output must match the window batch and ROI spatial shape")
        if output is None:
            output = np.zeros((host.shape[0], predicted.shape[1], *image_size), dtype=host.dtype)
        elif predicted.shape[1] != output.shape[1]:
            raise ValueError("predictor output channels must stay constant across windows")
        # MONAI's in-place multiplication rounds to predictor dtype BEFORE the
        # addition into the input-dtype accumulator. Keep these separate.
        predicted *= importance
        for coord, patch in zip(coordinates, predicted, strict=True):
            output[tuple(coord)] += patch
    assert output is not None  # Nonempty images always produce at least one window.
    output /= counts
    return output[(slice(None), slice(None), *crop)]
