"""Canonical, versioned ontology of fixed-shape inference graphs.

Tensor axes retain their source ordering. Operators own any local layout change;
references are explicit graph edges rather than tensor-name conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Op(StrEnum):
    """Portable inference operations; source-framework names belong to converters."""

    CONV2 = "conv2d"
    CONV3 = "conv3d"
    DECONV2 = "conv_transpose2d"
    DECONV3 = "conv_transpose3d"
    BATCH_NORM = "batch_norm"
    EVAL_BATCH_NORM = "batch_norm_with_aux"
    INSTANCE_NORM = "instance_norm"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    AVG_POOL2 = "avg_pool2d"
    BILINEAR = "resize_bilinear2d"
    CAT = "concatenate"
    SLICE = "slice"
    LOG_SOFTMAX = "log_softmax"
    ADD = "add"
    SUB = "subtract"
    MUL = "multiply"
    DIV = "divide"
    RESHAPE = "reshape"
    SPLIT = "split_chunks"
    CAST = "cast"
    ASSERT_META = "assert_tensor"
    GETITEM = "tuple_item"

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the complete named-argument contract for this operation."""
        return _PARAMETERS[self]


_CONV = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
_NORM = ("input", "weight", "bias", "running_mean", "running_var", "eps")
_PARAMETERS = {
    Op.CONV2: _CONV,
    Op.CONV3: _CONV,
    Op.DECONV2: (*_CONV, "output_padding"),
    Op.DECONV3: (*_CONV, "output_padding"),
    Op.BATCH_NORM: (*_NORM, "training"),
    Op.EVAL_BATCH_NORM: _NORM,
    Op.INSTANCE_NORM: (*_NORM, "use_input_stats"),
    Op.RELU: ("input",),
    Op.LEAKY_RELU: ("input", "negative_slope"),
    Op.AVG_POOL2: ("input", "kernel_size", "stride", "padding", "ceil_mode", "divisor_override"),
    Op.BILINEAR: ("input", "scale_factors", "output_size", "align_corners"),
    Op.CAT: ("tensors", "axis"),
    Op.SLICE: ("input", "axis", "start", "end", "step"),
    Op.LOG_SOFTMAX: ("input", "axis", "dtype"),
    Op.ADD: ("input", "other", "alpha"),
    Op.SUB: ("input", "other", "alpha"),
    Op.MUL: ("input", "other"),
    Op.DIV: ("input", "other"),
    Op.RESHAPE: ("input", "shape"),
    Op.SPLIT: ("input", "chunk_size", "axis"),
    Op.CAST: ("input", "dtype"),
    Op.ASSERT_META: ("input", "shape", "dtype"),
    Op.GETITEM: ("value", "index"),
}


class Role(StrEnum):
    """Where a tensor's value comes from: the caller, or stored graph state."""

    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"


class DType(StrEnum):
    """Element types the graph runtime supports."""

    FLOAT32 = "float32"
    INT32 = "int32"
    BOOL = "bool"


@dataclass(frozen=True)
class Ref:
    """Directed edge to an input tensor or an earlier operator result."""

    id: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Graph references require a nonempty identifier")


type Scalar = int | float | bool | str | None
type Value = Scalar | Ref | tuple[Value, ...]


def references(value: Value) -> tuple[str, ...]:
    """Return the ids of every ``Ref`` nested in ``value``, in order.

    Raises:
        ValueError: A float constant is NaN or infinite.
    """
    if isinstance(value, Ref):
        return (value.id,)
    if isinstance(value, tuple):
        return tuple(ref for child in value for ref in references(child))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Graph scalar constants must be finite")
    return ()


@dataclass(frozen=True)
class Argument:
    """Named operand of a ``Call``: a constant, a ``Ref``, or a tuple of them."""

    name: str
    value: Value

    def __post_init__(self) -> None:
        Ref(self.name)
        references(self.value)


@dataclass(frozen=True)
class Tensor:
    """Static declaration of a graph input or stored state tensor.

    ``id`` is the graph-local key (also the Safetensors key for non-input
    tensors). ``source_name`` is the name in the source framework, kept for
    error messages. ``shape`` is fully static; zero-sized dimensions are allowed.
    """

    id: str
    source_name: str
    role: Role
    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        Ref(self.id)
        Ref(self.source_name)
        if any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ValueError("Only static nonnegative tensor dimensions are supported")


@dataclass(frozen=True)
class Call:
    """Operation with canonical parameter names and a supported cast dtype."""

    id: str
    op: Op
    arguments: tuple[Argument, ...]

    def __post_init__(self) -> None:
        Ref(self.id)
        if len({arg.name for arg in self.arguments}) != len(self.arguments):
            raise ValueError(f"Duplicate arguments at {self.id}")
        if {arg.name for arg in self.arguments} != set(self.op.parameters):
            raise ValueError(f"{self.op} requires arguments {self.op.parameters}")
        if self.op == Op.CAST:
            dtype = next(arg.value for arg in self.arguments if arg.name == "dtype")
            if not isinstance(dtype, str):
                raise ValueError(f"{self.id}: cast dtype must name a supported DType")
            DType(dtype)


@dataclass(frozen=True)
class Producer:
    """Source frontend identity, retained as provenance and never imported by runtime."""

    name: str
    version: str

    def __post_init__(self) -> None:
        Ref(self.name)
        Ref(self.version)


@dataclass(frozen=True)
class Graph:
    """Pure inference graph with ordered flat tensor inputs and outputs.

    Dynamic shapes, tensor mutation, training, arbitrary Python objects and
    output pytree reconstruction are outside schema version 2. Schema 1 used
    source-specific operation names and must be regenerated by its converter.
    """

    tensors: tuple[Tensor, ...]
    calls: tuple[Call, ...]
    outputs: tuple[Ref, ...]
    producer: Producer
    schema_version: int = 2

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Unsupported graph schema_version; regenerate with medmlx-convert")
        seen: set[str] = set()
        for item in (*self.tensors, *self.calls):
            if item.id in seen:
                raise ValueError(f"Duplicate graph identifier: {item.id}")
            if isinstance(item, Call):
                required = {r for a in item.arguments for r in references(a.value)}
                if not required <= seen:
                    missing = required - seen
                    raise ValueError(f"Forward or missing reference at {item.id}: {missing}")
            seen.add(item.id)
        if not self.outputs or any(out.id not in seen for out in self.outputs):
            raise ValueError("Graph outputs must reference existing values")
