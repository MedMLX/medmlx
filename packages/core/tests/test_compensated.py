"""Higher-accuracy operator boundaries against independent float64 arithmetic."""

from typing import cast

import mlx.core as mlx
import numpy as np
import numpy.typing as npt

from medmlx_core.compensated import CompensatedFloat32Operators, CompensatedFloat48Operators
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime

mx = cast(MlxRuntime, import_mlx())


def _host(actual: mlx.array) -> npt.NDArray[np.float32]:
    mx.eval(actual)
    assert actual.dtype == mx.float32
    return np.array(actual, dtype=np.float32)


def _ops(groups: int = 1, eps: float = 1e-3) -> CompensatedFloat32Operators:
    return CompensatedFloat32Operators(mx, groups, eps)


def _group_norm(values: npt.NDArray[np.float64], groups: int) -> npt.NDArray[np.float64]:
    grouped = values.reshape(values.shape[0], groups, -1)
    centered = grouped - grouped.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
    return normalized.reshape(values.shape)


def test_paired_graph_retains_terms_across_hidden_layer_boundaries() -> None:
    ops = CompensatedFloat48Operators(mx, 1, 1e-5)
    values = np.array([[1e8, 1], [-1e8, 3]], dtype=np.float32).reshape(2, 2, 1, 1, 1)
    hidden = ops.conv(ops.pack(mx.array(values)),
                      mx.array(np.ones((1, 2, 1, 1, 1), np.float32)), None, padding=0, stride=1)
    offset = mx.array(np.array([-1e8, 1e8], np.float32).reshape(2, 1, 1, 1, 1))
    result = ops.round(ops.relu(ops.add(hidden, ops.pack(offset))))
    np.testing.assert_array_equal(_host(result).ravel(), [1, 3])


def test_paired_group_norm_keeps_the_input_low_word() -> None:
    ops = CompensatedFloat48Operators(mx, 2, 1e-5)
    source = np.random.default_rng(112).normal(size=(2, 4, 3, 5, 7)) + 1e4
    high = source.astype(np.float32)
    low = (source - high.astype(np.float64)).astype(np.float32)
    expected = _group_norm(high.astype(np.float64) + low.astype(np.float64), 2)
    actual = ops.norm(mx.array(np.concatenate([high, low])),
                      mx.array(np.ones(4, np.float32)), mx.array(np.zeros(4, np.float32)))
    packed = _host(actual)
    combined = packed[:2].astype(np.float64) + packed[2:].astype(np.float64)
    np.testing.assert_allclose(combined, expected, rtol=1e-9, atol=1e-9)


def test_compensated_convolution_retains_cancelled_terms() -> None:
    mx.set_default_device(mx.gpu)
    values = np.array([1e8, 1, -1e8], dtype=np.float32).reshape(1, 3, 1, 1, 1)
    expected = values.astype(np.float64).sum(axis=1, keepdims=True).astype(np.float32)
    actual = _ops(eps=1e-5).conv(mx.array(values), mx.array(np.ones((1, 3, 1, 1, 1), np.float32)),
                                 None, padding=0, stride=1)
    np.testing.assert_array_equal(_host(actual), expected)


def test_compensated_group_norm_ragged_large_offset() -> None:
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(719)
    values = (rng.normal(size=(2, 4, 7, 5, 9)) + 1e4).astype(np.float32)
    weight, bias = rng.normal(size=4).astype(np.float32), rng.normal(size=4).astype(np.float32)
    expected = (_group_norm(values.astype(np.float64), 2) * weight.reshape(1, 4, 1, 1, 1)
                + bias.reshape(1, 4, 1, 1, 1)).astype(np.float32)
    actual = _ops(2, 1e-5).norm(mx.array(values), mx.array(weight), mx.array(bias))
    np.testing.assert_allclose(_host(actual), expected, rtol=1e-6, atol=1e-6)


def test_compensated_decoder_retains_cancelled_channels() -> None:
    mx.set_default_device(mx.gpu)
    values = np.broadcast_to(
        np.array([1e8, 1, -1e8], dtype=np.float32).reshape(1, 3, 1, 1, 1), (1, 3, 2, 2, 2)
    ).copy()
    bias = np.array([4, -2], dtype=np.float32)
    source_sum = values.astype(np.float64).sum(axis=1, keepdims=True)
    repeated = source_sum.repeat(2, axis=2).repeat(2, axis=3).repeat(2, axis=4)
    expected = (np.broadcast_to(repeated, (1, 2, 4, 4, 4))
                + bias.reshape(1, 2, 1, 1, 1)).astype(np.float32)
    actual = _ops(eps=1e-5).conv_transpose3d(
        mx.array(values), mx.array(np.ones((3, 2, 2, 2, 2), dtype=np.float32)), mx.array(bias))
    np.testing.assert_array_equal(_host(actual), expected)


def test_compensated_elu_preserves_small_negative_values() -> None:
    values = np.concatenate([
        np.array([-100, -18, -1e-12, -1e-8, 0, 1], dtype=np.float32),
        np.random.default_rng(23).uniform(-17, 3, 1000).astype(np.float32),
    ])
    promoted = values.astype(np.float64)
    expected = np.where(promoted > 0, promoted, np.expm1(promoted)).astype(np.float32)
    np.testing.assert_allclose(_host(_ops().elu(mx.array(values))), expected, rtol=1e-6, atol=0)


def test_compensated_softmax_keeps_near_tied_decisions() -> None:
    values = np.random.default_rng(47).uniform(-20, 20, (2, 7, 5, 33)).astype(np.float32)
    values[0, 0, 0] = -100
    values[0, 0, 0, :2] = [0, 2**-22]
    shifted = values.astype(np.float64) - values.astype(np.float64).max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    expected = (exponentials / exponentials.sum(axis=-1, keepdims=True)).astype(np.float32)
    host = _host(_ops().softmax_last(mx.array(values)))
    np.testing.assert_allclose(host, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(host.argmax(axis=-1), expected.argmax(axis=-1))


def test_compensated_frozen_batch_norm_retains_centered_detail() -> None:
    rng = np.random.default_rng(719)
    values = (rng.normal(size=(2, 7, 5, 9, 4)) + 1e4).astype(np.float32)
    mean = (rng.normal(size=4) + 1e4).astype(np.float32)
    variance = rng.uniform(0.5, 2, 4).astype(np.float32)
    weight, bias = rng.normal(size=(2, 4)).astype(np.float32)
    expected = ((values.astype(np.float64) - mean) / np.sqrt(variance.astype(np.float64) + 1e-3)
                * weight + bias).astype(np.float32)
    actual = _ops().batch_norm_ndhwc(mx.array(values), mx.array(mean), mx.array(variance),
                                     mx.array(weight), mx.array(bias))
    np.testing.assert_allclose(_host(actual), expected, rtol=1e-6, atol=1e-6)


def test_compensated_gaussian_matches_double_stencil_at_borders() -> None:
    rng = np.random.default_rng(719)
    values = rng.normal(size=(2, 5, 7, 9, 3)).astype(np.float32)
    kernel = rng.uniform(size=(3, 3, 3)).astype(np.float32)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(windows, 3, axis=2)
    windows = np.lib.stride_tricks.sliding_window_view(windows, 3, axis=3)
    expected = np.einsum("bdhwcijk,ijk->bdhwc", windows,
                         kernel.astype(np.float64)).astype(np.float32)
    actual = _ops().gaussian_ndhwc(mx.array(values), mx.array(kernel))
    np.testing.assert_allclose(_host(actual), expected, rtol=1e-6, atol=1e-6)
