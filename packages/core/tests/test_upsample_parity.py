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
        pytest.param((1, 64, 12, 12, 12), 64, True, False, id="renal-64-channels"),
    ],
)
def test_deconv2x_matches_torch_repeated(
    input_shape: tuple[int, int, int, int, int],
    out_channels: int,
    with_bias: bool,
    strided: bool,
) -> None:
    """Check phase views and renal channel counts that allow reduced-precision matmul."""

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
