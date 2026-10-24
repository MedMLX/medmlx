"""Ordered float32 convolution against independent float64 arithmetic."""

from itertools import product
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from medmlx_core.precision import Float32Operators
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime

type Float64Array = npt.NDArray[np.float64]


def _reference(
    values: Float64Array, weight: Float64Array, bias: Float64Array, padding: int, stride: int
) -> Float64Array:
    padded = np.pad(values, [(0, 0), (0, 0), *[(padding, padding)] * 3])
    kernel = weight.shape[2:]
    shape = [(n - k) // stride + 1 for n, k in zip(padded.shape[2:], kernel, strict=True)]
    result = np.zeros((values.shape[0], weight.shape[0], *shape)) + bias[None, :, None, None, None]
    for z, y, x in product(*(range(k) for k in kernel)):
        window = padded[
            :,
            :,
            z : z + stride * shape[0] : stride,
            y : y + stride * shape[1] : stride,
            x : x + stride * shape[2] : stride,
        ]
        result += np.einsum("bidhw,oi->bodhw", window, weight[:, :, z, y, x])
    return result


@pytest.mark.parametrize(
    ("batch", "inputs", "outputs", "spatial", "kernel", "stride", "padding"),
    [
        pytest.param(2, 5, 7, (5, 6, 7), 3, 1, 1, id="batched-channel-tails"),
        pytest.param(1, 4, 6, (6, 5, 7), 5, 1, 2, id="more-taps-than-threads"),
        pytest.param(1, 48, 40, (9, 10, 11), 3, 2, 1, id="strided-large-map"),
        pytest.param(1, 3, 13, (4, 3, 5), 1, 1, 0, id="pointwise"),
    ],
)
def test_conv_matches_float64_reference(
    batch: int,
    inputs: int,
    outputs: int,
    spatial: tuple[int, int, int],
    kernel: int,
    stride: int,
    padding: int,
) -> None:
    mx = cast(MlxRuntime, import_mlx())
    rng = np.random.default_rng(kernel * 100 + outputs)
    values = rng.normal(size=(batch, inputs, *spatial)).astype(np.float32)
    weight = rng.normal(size=(outputs, inputs, kernel, kernel, kernel)).astype(np.float32)
    bias = rng.normal(size=(outputs,)).astype(np.float32)

    actual = Float32Operators(mx, 1, 1e-5).conv(
        mx.array(values), mx.array(weight), mx.array(bias), padding=padding, stride=stride
    )
    mx.eval(actual)

    expected = _reference(
        values.astype(np.float64),
        weight.astype(np.float64),
        bias.astype(np.float64),
        padding,
        stride,
    )
    np.testing.assert_allclose(np.array(actual, dtype=np.float32), expected, rtol=1e-5, atol=1e-4)
