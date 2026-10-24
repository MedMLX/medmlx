"""Standalone Metal inference from converted graph and Safetensors.

This module imports neither Torch nor Core ML/Core AI. Inputs and outputs are
flat ordered tensors with the source axes/dtypes. Host transfers belong to the
caller. Runtime operations use MLX Metal explicitly, with no CPU fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import mlx.core as mx

from medmlx_core.graph._mlx import invoke
from medmlx_core.graph.codec import loads
from medmlx_core.graph.operators import Result, execute
from medmlx_core.graph.schema import Graph, Ref, Role, Tensor, Value


def _resolve(value: Value, values: dict[str, Result]) -> Result:
    if isinstance(value, Ref):
        return values[value.id]
    if isinstance(value, tuple):
        return tuple(_resolve(child, values) for child in value)
    return value


def _check(value: mx.array, spec: Tensor) -> None:
    if tuple(value.shape) != spec.shape or str(value.dtype).removeprefix("mlx.core.") != spec.dtype:
        raise ValueError(f"{spec.source_name}: expected {spec.shape}/{spec.dtype}, "
                         f"got {value.shape}/{value.dtype}")


class Session:
    """Executable graph plus its stored tensors; each call keeps its own intermediates."""

    def __init__(self, graph: Graph, weights: dict[str, mx.array]) -> None:
        """Bind weights to a graph and check them against the declared tensors.

        Args:
            graph: Validated graph.
            weights: Tensor id to array for every non-input tensor, with the
                declared shape and dtype.

        Raises:
            RuntimeError: Metal is unavailable.
            ValueError: The keys differ from the graph's non-input tensors, or an
                array's shape or dtype differs from its declaration.
        """
        if not mx.metal.is_available():
            raise RuntimeError("Native MLX conversion requires Metal; no CPU fallback")
        specs = tuple(tensor for tensor in graph.tensors if tensor.role != Role.INPUT)
        if set(weights) != {spec.id for spec in specs}:
            raise ValueError("Converted tensor keys differ from graph state")
        for spec in specs:
            _check(weights[spec.id], spec)
        self.graph = graph
        self._weights = dict(weights)
        self.inputs = tuple(tensor for tensor in graph.tensors if tensor.role == Role.INPUT)

    @classmethod
    def load(cls, folder: Path) -> Session:
        """Load ``graph.json`` and ``weights.safetensors`` from ``folder``.

        The weights file is keyed by tensor id. The schema, keys, shapes and dtypes
        are checked strictly, and no source code is executed.

        Raises:
            FileNotFoundError: A file is missing.
            RuntimeError: Metal is unavailable.
            ValueError: The graph or weights fail validation.
        """
        graph = loads((folder / "graph.json").read_text())
        # File loading has no Metal implementation; materialize host storage
        # before scheduling model arithmetic on the explicitly selected GPU.
        weights = mx.load(folder / "weights.safetensors", stream=mx.cpu)
        if not isinstance(weights, dict):
            raise ValueError("Safetensors must contain named arrays")
        invoke(mx, "eval", *weights.values())
        return cls(graph, weights)

    def __call__(self, *inputs: mx.array) -> tuple[mx.array, ...]:
        """Run the graph on Metal and return its flat outputs, evaluated.

        Inputs are positional, in the graph's input order, and must match the
        declared shape and dtype exactly.

        Raises:
            ValueError: The input count, shape or dtype differs from the graph's
                declaration, or an output is not a tensor.
        """
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} input tensors, got {len(inputs)}")
        values: dict[str, Result] = dict(self._weights)
        for value, spec in zip(inputs, self.inputs, strict=True):
            _check(value, spec)
            values[spec.id] = value
        with mx.stream(mx.gpu):
            for call in self.graph.calls:
                arguments = {arg.name: _resolve(arg.value, values) for arg in call.arguments}
                values[call.id] = execute(call.op, arguments)
            outputs = tuple(values[ref.id] for ref in self.graph.outputs)
            if not all(isinstance(value, mx.array) for value in outputs):
                raise ValueError("Graph produced a non-tensor output")
            invoke(mx, "eval", *outputs)
        return cast(tuple[mx.array, ...], outputs)
