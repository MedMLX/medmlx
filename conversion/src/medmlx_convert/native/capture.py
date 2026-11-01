"""Convert original Torch inference graphs and all consumed tensor state.

Capture uses strict torch.export with static inputs, then Torch's functionalizing
pass with an empty decomposition table. Nonmutating alias references share their
source tensor in the serialized value graph. No precision rewrites are made.
Parameters, persistent buffers,
nonpersistent buffers and lifted constants share the same storage boundary.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from medmlx_core.graph.codec import dumps
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
from torch.export.graph_signature import InputKind, OutputKind, TensorArgument
from torch.fx import Node

from medmlx_convert.boundary import as_sequence, as_tensors, attribute, function
from medmlx_convert.native.torch_ops import OPERATORS, lower_call


@dataclass(frozen=True)
class Conversion:
    """Saved graph plus outputs replayed from that exact captured Torch program."""

    graph: Graph
    captured_outputs: tuple[torch.Tensor, ...]
    functional_outputs: tuple[torch.Tensor, ...]


def _is_alias(node: Node) -> bool:
    """Identify Torch's storage alias, whose value is unchanged in an immutable graph."""
    return node.op == "call_function" and node.target == torch.ops.aten.alias.default


def _value(raw: object) -> Value:
    if isinstance(raw, Node):
        if _is_alias(raw):
            return _value(raw.args[0])
        return Ref(raw.name)
    if isinstance(raw, tuple | list):
        return tuple(_value(item) for item in cast(tuple[object, ...] | list[object], raw))
    if isinstance(raw, torch.dtype):
        return DType(str(raw).removeprefix("torch.")).value
    if isinstance(raw, torch.device | torch.layout):
        return str(raw).removeprefix("torch.")
    if raw is None or isinstance(raw, int | float | bool | str):
        return raw
    raise ValueError(f"Unsupported captured argument: {type(raw).__name__}: {raw}")


def _call(node: Node) -> Call:
    target = node.target
    if target is operator.getitem:
        return Call(node.name, Op.GETITEM,
                    (Argument("value", _value(node.args[0])),
                     Argument("index", _value(node.args[1]))))
    if not callable(target):
        raise ValueError(f"Unsupported graph target: {target}")
    args: list[Argument] = []
    # Bind defaults from the installed source schema, never a copied signature.
    schema = attribute(target, "_schema")
    for index, arg in enumerate(as_sequence(attribute(schema, "arguments"))):
        name = attribute(arg, "name")
        if not isinstance(name, str):
            raise ValueError("ATen schema argument requires a name")
        if name in node.kwargs:
            value = node.kwargs[name]
        elif index < len(node.args):
            value = node.args[index]
        elif function(arg, "has_default_value")():
            value = attribute(arg, "default_value")
        else:
            raise ValueError(f"Missing {name} at {node.name}")
        args.append(Argument(name, _value(value)))
    return lower_call(node.name, str(target), tuple(args))


def _state(program: torch.export.ExportedProgram,
           inputs: tuple[torch.Tensor, ...]) -> tuple[tuple[Tensor, ...], dict[str, np.ndarray]]:
    specs: list[Tensor] = []
    arrays: dict[str, np.ndarray] = {}
    user_inputs = iter(inputs)
    nodes = {node.name: node for node in program.graph.nodes}
    roles = {InputKind.PARAMETER: Role.PARAMETER, InputKind.BUFFER: Role.BUFFER,
             InputKind.CONSTANT_TENSOR: Role.CONSTANT}
    for entry in program.graph_signature.input_specs:
        if not isinstance(entry.arg, TensorArgument):
            raise ValueError("Only flat tensor inputs and captured tensor state are supported")
        name = entry.arg.name
        if entry.kind == InputKind.USER_INPUT:
            value, role, source = next(user_inputs), Role.INPUT, name
        else:
            if not nodes[name].users:
                continue
            if entry.kind not in roles or entry.target is None:
                raise ValueError(f"Unsupported captured input: {entry}")
            role, source = roles[entry.kind], entry.target
            storage = program.constants if entry.persistent is False else program.state_dict
            if entry.kind == InputKind.CONSTANT_TENSOR:
                storage = program.constants
            value = storage[source]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"{source}: capture requires CPU tensors")
        dtype = DType(str(value.dtype).removeprefix("torch."))
        specs.append(Tensor(name, source, role, tuple(value.shape), dtype))
        if role != Role.INPUT:
            array = value.detach().numpy().copy()
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite captured tensor: {source}")
            arrays[name] = array
    return tuple(specs), arrays


def convert(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], folder: Path) -> Conversion:
    """Save a static graph and Safetensors to a new relative directory.

    The model must already be in eval mode and all inputs/state must be CPU
    float32, int32 or bool. Unsupported operators, output mutation and dynamic
    objects fail before creating the destination. This is conversion, not a
    parity verdict; callers must compare the saved runtime against the source.
    """
    if folder.is_absolute() or ".." in folder.parts:
        raise ValueError("Conversion output must be repository-relative without parent traversal")
    if folder.exists():
        raise FileExistsError(f"Refusing to replace conversion output: {folder}")
    if not inputs or any(module.training for module in model.modules()):
        raise ValueError("Conversion requires tensor inputs and an eval-mode model")
    program = torch.export.export(model, inputs, strict=True)
    raw_program = program
    functional = function(program, "run_decompositions")({})
    if not isinstance(functional, torch.export.ExportedProgram):
        raise TypeError("Torch functionalization did not return an ExportedProgram")
    program = functional
    if any(entry.kind != OutputKind.USER_OUTPUT for entry in program.graph_signature.output_specs):
        raise ValueError("State/input mutation and training outputs are unsupported")
    with torch.no_grad():
        raw = raw_program.module()(*inputs)
    captured_outputs = (raw,) if isinstance(raw, torch.Tensor) else as_tensors(raw)
    unsupported = sorted({str(node.target) for node in program.graph.nodes
                          if node.op == "call_function" and node.target is not operator.getitem
                          and not _is_alias(node)
                          and str(node.target) not in OPERATORS})
    if unsupported:
        raise ValueError(f"Unsupported native MLX operators: {unsupported}")
    tensors, arrays = _state(program, inputs)
    calls: list[Call] = []
    outputs: tuple[Ref, ...] = ()
    for node in program.graph.nodes:
        if _is_alias(node):
            continue
        if node.op == "call_function":
            calls.append(_call(node))
        elif node.op == "output":
            value = _value(node.args[0])
            if not isinstance(value, tuple) or not all(isinstance(v, Ref) for v in value):
                raise ValueError("Only flat tensor outputs are supported")
            outputs = cast(tuple[Ref, ...], value)
        elif node.op != "placeholder":
            raise ValueError(f"Unsupported graph node: {node.op}")
    graph = Graph(tensors, tuple(calls), outputs, Producer("torch", str(torch.__version__)))
    with torch.no_grad():
        replayed = program.module()(*inputs)
    functional_outputs = (replayed,) if isinstance(replayed, torch.Tensor) else as_tensors(replayed)
    import mlx.core as mx

    folder.mkdir(parents=True)
    function(mx, "save_safetensors")(folder / "weights.safetensors",
                                     {name: mx.array(value) for name, value in arrays.items()})
    (folder / "graph.json").write_text(dumps(graph))
    return Conversion(graph, captured_outputs, functional_outputs)
