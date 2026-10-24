"""Split convolution matches the unsplit result and refuses non-divisible splits.

MAISI's ``MaisiConvolution`` crops every chunk's output to a fixed size, including
the last, so a split axis whose length is not evenly divisible by ``num_splits``
silently drops the remainder voxels (verified against the pinned upstream MONAI
source: ``_concatenate_tensors`` never uses the extra context it slices into the
last input chunk). ``split_conv3d_ncdhw``/``split_conv_transpose3d_ncdhw`` refuse
that input instead of reproducing the silent truncation.
"""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from medmlx_core.ops import (
    conv3d_ncdhw,
    conv_transpose3d_ncdhw,
    split_conv3d_ncdhw,
    split_conv_transpose3d_ncdhw,
)
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime


@pytest.mark.parametrize("dim_split", [0, 1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_split_conv3d_matches_unsplit(dim_split: int, stride: int) -> None:
    """A divisible split axis reconstructs the same result as one unsplit call."""

    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(2024)
    # Every spatial axis is divisible by num_splits=2.
    values = rng.standard_normal((1, 2, 8, 8, 8)).astype(np.float32)
    weight = rng.standard_normal((3, 2, 3, 3, 3)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)

    x = mx.array(values)
    w = mx.array(weight)
    b = mx.array(bias)

    unsplit = conv3d_ncdhw(x, w, b, padding=1, stride=stride, mx=mx)
    split = split_conv3d_ncdhw(
        x, w, b, padding=1, stride=stride, num_splits=2, dim_split=dim_split, mx=mx
    )
    mx.eval(unsplit, split)
    np.testing.assert_array_equal(
        cast("NDArray[np.float32]", np.asarray(split)),
        cast("NDArray[np.float32]", np.asarray(unsplit)),
    )


@pytest.mark.parametrize("dim_split", [0, 1, 2])
def test_split_conv_transpose3d_matches_unsplit(dim_split: int) -> None:
    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(2025)
    values = rng.standard_normal((1, 2, 8, 8, 8)).astype(np.float32)
    weight = rng.standard_normal((2, 3, 3, 3, 3)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)

    x = mx.array(values)
    w = mx.array(weight)
    b = mx.array(bias)

    unsplit = conv_transpose3d_ncdhw(x, w, b, padding=1, output_padding=0, stride=1, mx=mx)
    split = split_conv_transpose3d_ncdhw(
        x, w, b, padding=1, output_padding=0, stride=1, num_splits=2, dim_split=dim_split, mx=mx
    )
    mx.eval(unsplit, split)
    np.testing.assert_array_equal(
        cast("NDArray[np.float32]", np.asarray(split)),
        cast("NDArray[np.float32]", np.asarray(unsplit)),
    )


def test_split_conv3d_rejects_non_divisible_length() -> None:
    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(2026)
    # W axis has length 4, not divisible by num_splits=3.
    values = rng.standard_normal((1, 2, 8, 6, 4)).astype(np.float32)
    weight = rng.standard_normal((3, 2, 3, 3, 3)).astype(np.float32)
    x, w = mx.array(values), mx.array(weight)

    with pytest.raises(ValueError, match="split_conv3d length 4 is not divisible by num_splits 3"):
        split_conv3d_ncdhw(x, w, None, padding=1, num_splits=3, dim_split=2, mx=mx)


def test_split_conv_transpose3d_rejects_non_divisible_length() -> None:
    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(2027)
    values = rng.standard_normal((1, 2, 8, 6, 4)).astype(np.float32)
    weight = rng.standard_normal((2, 3, 3, 3, 3)).astype(np.float32)
    x, w = mx.array(values), mx.array(weight)

    with pytest.raises(
        ValueError, match="split_conv_transpose3d length 4 is not divisible by num_splits 3"
    ):
        split_conv_transpose3d_ncdhw(
            x, w, None, padding=1, output_padding=0, num_splits=3, dim_split=2, mx=mx
        )
