"""Independent dense blending and restoration contracts for bounded host storage."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from medmlx_core.volume_workspace import (
    FloatArray,
    SlabAccumulator,
    VolumeWorkspace,
    restored_argmax,
)


@pytest.mark.parametrize(
    "dtype,shape,roi",
    [(np.float16, (9,), (4,)), (np.float32, (9, 7, 6), (4, 5, 3)), (np.float64, (9, 7), (4, 5))],
)
def test_slabs_match_sequential_dense_blending(
    dtype: type[np.floating], shape: tuple[int, ...], roi: tuple[int, ...], tmp_path: Path
) -> None:
    rng = np.random.default_rng(851)
    crop = tuple(slice(1, size - 1) for size in shape)
    result_shape = (2, 3, *(size - 2 for size in shape))
    expected = np.zeros((2, 3, *shape), dtype=dtype)
    counts = np.zeros(shape, dtype=dtype)
    importance = rng.uniform(0.1, 1, size=roi).astype(dtype)
    axes = [
        sorted({*range(0, size - width + 1, 2), size - width})
        for size, width in zip(shape, roi, strict=True)
    ]
    windows = [
        tuple(slice(start, start + width) for start, width in zip(origin, roi, strict=True))
        for origin in product(*axes)
    ]
    for window in windows:
        counts[window] += importance
    with VolumeWorkspace(tmp_path) as workspace:
        actual = workspace.empty(result_shape, dtype)

        def sink(image: int, start: int, values: FloatArray) -> None:
            actual[image, :, start : start + values.shape[1]] = values

        accumulator = SlabAccumulator(shape, roi, crop, 3, dtype, importance, sink)
        assert accumulator.scores.shape == (3, roi[0], *shape[1:])
        for image in range(2):
            for window in windows:
                # Prediction dtype differs from the FP16/FP64 accumulator as admitted by MONAI.
                prediction = rng.normal(size=(3, *roi)).astype(np.float32)
                prediction *= importance
                expected[(image, slice(None), *window)] += prediction
                accumulator.add(image, window, prediction)
        accumulator.finish()
        accumulator.finish()
        expected /= counts
        np.testing.assert_array_equal(actual, expected[(slice(None), slice(None), *crop)])
    assert not list(tmp_path.iterdir())


def test_workspace_cleanup_on_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), VolumeWorkspace(tmp_path) as workspace:
        workspace.empty((2, 3, 4))[:] = 1
        raise RuntimeError("cancelled")
    assert not list(tmp_path.iterdir())


def test_restore_probabilities_before_argmax_and_keep_first_tie() -> None:
    probabilities = np.array([[0.51, 0.0, 0.5], [0.49, 1.0, 0.5]], dtype=np.float32)

    def interpolate(channel: np.ndarray[tuple[int, ...], np.dtype[np.float32]]):
        return np.stack(((channel[:, 0] + channel[:, 1]) * 0.5, channel[:, 2]), axis=1)

    np.testing.assert_array_equal(restored_argmax(probabilities, interpolate), [1.0, 0.0])


def test_blocked_channel_mapping_keeps_global_eight_lane_tail(tmp_path: Path) -> None:
    values = np.arange(3 * 39, dtype=np.float32).reshape(3, 3, 13)

    def source_softmax(block: np.ndarray[tuple[int, ...], np.dtype[np.float32]]):
        shifted = np.exp(block - block.max(axis=0, keepdims=True))
        return shifted / shifted.sum(axis=0, keepdims=True)

    with VolumeWorkspace(tmp_path) as workspace:
        actual = workspace.map_channels(values, source_softmax, block_voxels=16)
        expected = source_softmax(values.reshape(3, -1)).reshape(values.shape)
        np.testing.assert_array_equal(actual, expected)
