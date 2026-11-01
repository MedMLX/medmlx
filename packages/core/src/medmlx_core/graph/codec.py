"""Single JSON boundary for the native graph ontology; never executes input code."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import cast

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
    Value,
)


def _mapping(raw: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(raw, dict) or set(cast(dict[object, object], raw)) != fields:
        raise ValueError(f"Expected fields {sorted(fields)}")
    return cast(dict[str, object], raw)


def _sequence(raw: object) -> tuple[object, ...]:
    if not isinstance(raw, list):
        raise ValueError("Expected JSON list")
    return tuple(cast(list[object], raw))


def _text(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError("Expected nonempty string")
    return raw


def _value(raw: object) -> Value:
    if isinstance(raw, dict):
        return Ref(_text(_mapping(cast(object, raw), {"id"})["id"]))
    if isinstance(raw, list):
        return tuple(_value(child) for child in cast(list[object], raw))
    if raw is None or isinstance(raw, int | float | bool | str):
        return raw
    raise ValueError("Unsupported graph argument")


def _tensor(raw: object) -> Tensor:
    row = _mapping(raw, {"id", "source_name", "role", "shape", "dtype"})
    shape = _sequence(row["shape"])
    if any(type(dim) is not int for dim in shape):
        raise ValueError("Tensor dimensions must be integers")
    return Tensor(_text(row["id"]), _text(row["source_name"]), Role(_text(row["role"])),
                  cast(tuple[int, ...], shape), DType(_text(row["dtype"])))


def _call(raw: object) -> Call:
    row = _mapping(raw, {"id", "op", "arguments"})
    args: list[Argument] = []
    for entry in _sequence(row["arguments"]):
        arg = _mapping(entry, {"name", "value"})
        args.append(Argument(_text(arg["name"]), _value(arg["value"])))
    return Call(_text(row["id"]), Op(_text(row["op"])), tuple(args))


def loads(payload: str) -> Graph:
    """Parse schema 2 or raise ValueError for malformed/unsupported graph data."""
    row = _mapping(json.loads(payload),
                   {"schema_version", "tensors", "calls", "outputs", "producer"})
    version = row["schema_version"]
    if type(version) is not int:
        raise ValueError("schema_version must be an integer")
    outputs = tuple(_value(raw) for raw in _sequence(row["outputs"]))
    if not all(isinstance(output, Ref) for output in outputs):
        raise ValueError("Outputs must be tensor references")
    producer = _mapping(row["producer"], {"name", "version"})
    return Graph(tuple(_tensor(raw) for raw in _sequence(row["tensors"])),
                 tuple(_call(raw) for raw in _sequence(row["calls"])),
                 cast(tuple[Ref, ...], outputs),
                 Producer(_text(producer["name"]), _text(producer["version"])), version)


def dumps(graph: Graph) -> str:
    """Serialize a constructed graph with stable keys and no nonfinite numbers."""
    return json.dumps(asdict(graph), indent=2, sort_keys=True, allow_nan=False) + "\n"
