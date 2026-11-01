"""Portable graph serialization and saved Metal execution without a converter."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from medmlx_core.graph._mlx import invoke
from medmlx_core.graph.codec import dumps, loads
from medmlx_core.graph.runtime import Session
from medmlx_core.graph.schema import (
    Argument,
    Call,
    DType,
    Graph,
    Op,
    Producer,
    Ref,
    Role,
    Tensor,
)


def _graph(channels: int) -> Graph:
    return Graph((Tensor("x", "image", Role.INPUT, (1, channels), DType.FLOAT32),
                  Tensor("offset", "buffer", Role.BUFFER, (channels,), DType.FLOAT32)),
                 (Call("sum", Op.ADD, (Argument("input", Ref("x")),
                                       Argument("other", Ref("offset")), Argument("alpha", 1))),),
                 (Ref("sum"),), Producer("independent-test", "1"))


def test_graph_roundtrip_rejects_source_operations_and_retired_schema() -> None:
    graph = _graph(3)
    assert loads(dumps(graph)) == graph
    with pytest.raises(ValueError):
        loads(dumps(graph).replace('"add"', '"aten.add.Tensor"'))
    with pytest.raises(ValueError):
        loads(dumps(graph).replace('"schema_version": 2', '"schema_version": 1'))


def test_saved_graph_runs_from_core_and_checks_inputs_and_state(tmp_path: Path) -> None:
    graph = _graph(3)
    (tmp_path / "graph.json").write_text(dumps(graph))
    invoke(mx, "save_safetensors", tmp_path / "weights.safetensors",
           {"offset": mx.array([100, 101, 102], dtype=mx.float32)})
    session = Session.load(tmp_path)
    output, = session(mx.array([[1, 2, 3]], dtype=mx.float32))
    np.testing.assert_array_equal(np.asarray(output, dtype=np.float32), [[101, 103, 105]])
    with pytest.raises(ValueError):
        session(mx.ones((2, 3)))
    with pytest.raises(ValueError):
        Session(graph, {})


@pytest.mark.parametrize("dtype", ["float64", None, 3, ["float32"]])
def test_cast_dtype_is_parsed_before_runtime(dtype: object) -> None:
    """Reject malformed cast artifacts before weights or Metal execution are needed."""
    base = _graph(3)
    call = Call("sum", Op.CAST, (Argument("input", Ref("x")), Argument("dtype", DType.FLOAT32)))
    graph = Graph(base.tensors, (call,), base.outputs, base.producer)
    assert loads(dumps(graph)) == graph
    with pytest.raises(ValueError):
        loads(dumps(graph).replace('"value": "float32"', f'"value": {json.dumps(dtype)}'))
