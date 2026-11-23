"""Type-check contract between ``medmlx_core.typing`` and the installed MLX stubs.

Pyright checks this module; nothing runs it. It shows how a runtime obtains each protocol
from the real modules and calls each member whose stub is wrong or incomplete, so an MLX
upgrade that breaks the contract fails type checking.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    import mlx.core as mx
    import mlx.utils as mlx_utils
    import numpy as np

    from medmlx_core.typing import (
        ArrayConstructor,
        ArrayEvaluator,
        FastRuntime,
        HostArray,
        IndexedUpdates,
        MetalKernel,
        MetalRuntime,
        MlxRuntime,
        NeuralRuntime,
        RandomRuntime,
        UtilsRuntime,
    )

    # Namespaces whose stubs match the protocol are assigned directly.
    metal_runtime: MetalRuntime = mx.metal
    random_runtime: RandomRuntime = mx.random

    # ``mx.fast.metal_kernel`` is stubbed as returning ``object``.
    fast_runtime = cast(FastRuntime, mx.fast)

    # ``mx.eval`` and ``mx.async_eval`` take untyped ``*args`` and ``mx.array`` rejects NumPy
    # input, so the runtime is a cast. ``mlx.nn`` has no stubs, so it is imported by name.
    runtime = cast(MlxRuntime, mx)
    neural_runtime = cast(NeuralRuntime, import_module("mlx.nn"))
    utils_runtime = cast(UtilsRuntime, mlx_utils)

    def stub_gaps(
        host: HostArray, weights: dict[str, mx.array], path: Path, scale: np.float32
    ) -> None:
        """Call each member that the installed stubs mistype."""
        # array: NumPy input and nested host sequences.
        make_array = cast(ArrayConstructor, mx.array)
        values = make_array(host, mx.float32)
        nested = make_array([[1.0, 2.0], [3.0, 4.0]])

        # eval and async_eval: arrays, sequences and trees of arrays.
        evaluate: ArrayEvaluator = runtime.eval
        evaluate(values, [values, nested], {"a": values, "b": [nested]})
        runtime.eval(values, (nested, values))
        runtime.async_eval({"a": values})

        # save_safetensors returns None and takes any path or file object.
        saved: None = runtime.save_safetensors(path, weights, {"format": "mlx"})

        # load: safetensors with and without metadata, and the format-dependent fallback.
        tensors: dict[str, mx.array] = runtime.load(path, "safetensors")
        with_metadata: tuple[dict[str, mx.array], dict[str, str]] = runtime.load(
            path, "safetensors", return_metadata=True
        )
        anything = runtime.load(path)

        # fast.metal_kernel returns a typed kernel callable.
        kernel: MetalKernel = runtime.fast.metal_kernel(
            "copy", ["x"], ["y"], "y[0] = x[0];", compile_options={"math_mode": "fast"}
        )
        launched: list[mx.array] = kernel(
            inputs=[values],
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )

        # array.at[...]: scatter updates with array and host values.
        updates = cast(IndexedUpdates, values.at)
        added: mx.array = updates[nested].add(values)
        added_host: mx.array = updates[nested].add(host)

        # split returns a list of arrays, not one array.
        parts: list[mx.array] = runtime.split(values, 2)

        # roll accepts a sequence of shifts and axes.
        rolled: mx.array = runtime.roll(values, (1, 2), (0, 1))

        _ = (saved, tensors, with_metadata, anything, launched, added, added_host, parts, rolled)
        _ = (nested, scale)
