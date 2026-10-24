"""Pinned MONAI/PyTorch references for each layer and complete seeded graphs."""

from typing import cast

import mlx.core as mlx
import numpy as np
import pytest
from fixture_cases import HostArray, load_fixture
from numerical import assert_reference

from medmlx_core.channel_last import ChannelLastGraph
from medmlx_core.runtime import import_mlx
from medmlx_core.typing import ArrayFactory, MlxRuntime

mx: MlxRuntime = import_mlx()
make_array = cast(ArrayFactory, mx.array)


@pytest.fixture
def graph_and_arrays() -> tuple[ChannelLastGraph, dict[str, HostArray]]:
    _, arrays = load_fixture("channel_last")
    weights = {
        key: make_array(value)
        for key, value in arrays.items()
        if key.endswith((".weight", ".bias", ".running_mean", ".running_var"))
    }
    return ChannelLastGraph(weights, mx=mx), arrays


def assert_stage(actual: mlx.array, expected: HostArray, *, budget: str = "graph") -> None:
    mx.eval(actual)
    host = cast(HostArray, np.asarray(actual))
    assert_reference(host, expected, budget=budget)


@pytest.mark.parametrize("case", ["2d", "3d"])
def test_channel_last_seeded_graph_matches_upstream(
    graph_and_arrays: tuple[ChannelLastGraph, dict[str, HostArray]], case: str
) -> None:
    graph, arrays = graph_and_arrays
    x = make_array(arrays[f"input_{case}"])
    if case == "2d":
        x = graph.conv(x, "conv", padding=1, groups=2)
        x = graph.batch_norm(x, "bn", eps=1e-3)
        x = graph.swish(x)
        x = graph.layer_norm(x)
    else:
        x = graph.deconv3d(x, "deconv")
        x = graph.instance_norm(x)
    assert_stage(x, arrays[f"expected_{case}"])


@pytest.mark.parametrize(
    "layer,input_key,expected_key,budget",
    [
        ("linear", "input_2d", "expected_linear", "dot"),
        ("affine_ln", "input_2d", "expected_affine_ln", "normalization"),
        ("volume", "input_3d", "expected_volume", "dot"),
        ("relu", "input_2d", "expected_relu", "exact"),
        ("batch_norm", "expected_conv", "expected_bn", "normalization"),
        ("swish", "expected_bn", "expected_swish", "pointwise"),
        ("layer_norm", "expected_swish", "expected_ln", "normalization"),
        ("instance_norm", "expected_deconv", "expected_3d", "normalization"),
        ("deconv", "input_3d", "expected_deconv", "dot"),
    ],
    ids=[
        "linear",
        "affine_ln",
        "volume",
        "relu",
        "batch_norm",
        "swish",
        "layer_norm",
        "instance_norm",
        "deconv",
    ],
)
def test_individual_layers_match_upstream(
    graph_and_arrays: tuple[ChannelLastGraph, dict[str, HostArray]],
    layer: str,
    input_key: str,
    expected_key: str,
    budget: str,
) -> None:
    graph, arrays = graph_and_arrays
    # Feed each layer the upstream stage input, isolating its own rounding from
    # errors propagated by earlier layers. Complete graphs are checked above.
    x = make_array(arrays[input_key])
    if layer == "linear":
        actual = graph.linear(x, "projection")
    elif layer == "affine_ln":
        actual = graph.layer_norm(x, "ln")
    elif layer == "volume":
        actual = graph.conv(x, "volume", padding=1)
    elif layer == "relu":
        actual = graph.relu(x)
    elif layer == "batch_norm":
        actual = graph.batch_norm(x, "bn", eps=1e-3)
    elif layer == "deconv":
        actual = graph.deconv3d(x, "deconv")
    else:
        actual = getattr(graph, layer)(x)
    assert_stage(actual, arrays[expected_key], budget=budget)


@pytest.mark.parametrize("name", ["residual_same", "residual_project"])
def test_residual_blocks_match_monai(
    graph_and_arrays: tuple[ChannelLastGraph, dict[str, HostArray]], name: str
) -> None:
    graph, arrays = graph_and_arrays
    assert_stage(graph.residual3d(make_array(arrays["input_3d"]), name), arrays[f"expected_{name}"])


def test_input_contract(graph_and_arrays: tuple[ChannelLastGraph, dict[str, HostArray]]) -> None:
    graph, arrays = graph_and_arrays
    x = make_array(arrays["input_2d"])
    graph.require_input(x, (5, 7, 4))
    with pytest.raises(ValueError, match="Expected Bx"):
        graph.require_input(x, (4, 7, 4))
    with pytest.raises(ValueError, match="float32"):
        graph.require_input(x.astype(mx.float16), (5, 7, 4))
