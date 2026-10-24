"""Independent Torch decoder oracles, including the repeated 48-cubed operation."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from fixture_cases import HostArray
from numerical import assert_reference

from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime


@pytest.mark.parametrize(
    ("input_shape", "out_channels", "with_bias", "strided"),
    [
        ((1, 4, 48, 48, 48), 3, True, False),
        ((2, 5, 3, 4, 6), 7, False, True),
        ((2, 5, 3, 4, 6), 7, True, True),
    ],
)
def test_deconv2x_matches_torch_repeated(
    input_shape: tuple[int, int, int, int, int],
    out_channels: int,
    with_bias: bool,
    strided: bool,
) -> None:
    """Exercise the lazy phase-view-sensitive 48-cubed decoder operation."""

    from medmlx_core.upsample import deconv2x_ncdhw

    mx = cast(MlxRuntime, import_mlx())
    pytest.importorskip("torch")
    import torch

    from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])
    rng = np.random.default_rng(617)
    values = rng.normal(size=input_shape).astype(np.float32)
    weight = rng.normal(size=(input_shape[1], out_channels, 2, 2, 2)).astype(np.float32)
    bias = rng.normal(size=out_channels).astype(np.float32) if with_bias else None
    expected = torch.nn.functional.conv_transpose3d(
        from_numpy(values),
        weight=from_numpy(weight),
        bias=from_numpy(bias) if bias is not None else None,
        stride=2,
    ).numpy()

    native_values, native_weight = mx.array(values), mx.array(weight)
    if strided:
        native_values, native_weight = [
            mx.array(v.swapaxes(1, 4).copy()).swapaxes(1, 4) for v in (values, weight)
        ]
    native_bias = mx.array(bias) if bias is not None else None
    for _ in range(3):
        actual = deconv2x_ncdhw(native_values, native_weight, native_bias, mx=mx)
        mx.eval(actual)
        host = cast(HostArray, np.asarray(actual))
        assert_reference(host, expected, budget="dot")


@pytest.mark.parametrize("strided", [False, True])
def test_decoder_fusion_matches_torch_at_borders_and_singleton_axes(strided: bool) -> None:
    from medmlx_core.upsample import upsample_add_ncdhw

    mx = cast(MlxRuntime, import_mlx())
    pytest.importorskip("torch")
    import torch

    from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2, 3, 1, 3, 5)).astype(np.float32)
    skip = rng.normal(size=(2, 3, 2, 6, 10)).astype(np.float32)
    expected = (
        torch.nn.functional.interpolate(
            from_numpy(values.copy()), scale_factor=2, mode="trilinear", align_corners=False
        ).numpy()
        + skip
    )
    inputs = [mx.array(v) for v in (values, skip)]
    if strided:
        inputs = [mx.array(v.swapaxes(1, 4).copy()).swapaxes(1, 4) for v in (values, skip)]
    actual = upsample_add_ncdhw(*inputs, mx=mx)
    mx.eval(actual)
    assert_reference(cast(HostArray, np.asarray(actual)), expected, budget="interpolation")
