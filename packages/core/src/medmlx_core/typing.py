"""Typed MLX boundaries without importing the optional runtime at package import.

Each protocol states the parts of one MLX namespace that MedMLX code calls, with the
installed stubs' parameter names, defaults and overloads. Where the stubs are wrong or
incomplete (``eval`` and ``async_eval`` arguments, ``load`` and ``save_safetensors``
results, ``fast.metal_kernel``, ``array`` from NumPy, ``array.at``, ``split``, ``roll``),
the protocol states the behavior the runtime has. A module obtains a protocol with
``cast``, for example ``cast(MlxRuntime, import_mlx())`` or ``cast(NeuralRuntime, nn)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import (
        Device,
        DeviceType,
        Dtype,
        Stream,
        StreamContext,
        StreamOrDevice,
        ThreadLocalStream,
        array,
    )

type HostArray = NDArray[np.generic]
type Scalar = int | float | bool | complex
type ArrayInput = Scalar | array | HostArray | Sequence[ArrayInput]
type ArrayOperand = Scalar | array
type Tree[T] = T | Sequence[Tree[T]] | Mapping[str, Tree[T]]
type ArrayTree = Tree[array]


class ArrayConstructor(Protocol):
    """Callable that constructs an MLX array from host data, as ``mx.array`` does."""

    def __call__(self, val: ArrayInput, /, dtype: Dtype | None = None) -> array:
        """Build an array from host data, converting when a dtype is requested."""
        ...


ArrayFactory = ArrayConstructor


class ArrayEvaluator(Protocol):
    """Callable that materializes lazy arrays on the compute device."""

    def __call__(self, *args: ArrayTree) -> None:
        """Schedule every argument, and its dependencies, for evaluation."""
        ...


class MetalKernel(Protocol):
    """Callable returned by ``mx.fast.metal_kernel``; ``grid`` counts threads, not groups."""

    def __call__(
        self,
        *,
        inputs: Sequence[ArrayOperand],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[tuple[str, Dtype | int | bool]] | None = None,
        init_value: float | None = None,
        verbose: bool = False,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Launch the kernel and return one array per declared output."""
        ...


class ArrayLayer(Protocol):
    """Callable that maps one MLX array to another, such as an activation or a pooling layer."""

    def __call__(self, x: array, /) -> array:
        """Apply the layer to ``x``."""
        ...


class IndexedUpdate(Protocol):
    """The scatter operations of ``array.at[indices]``; each returns a new array."""

    def add(self, value: ArrayOperand | HostArray) -> array:
        """Add ``value`` at the selected indices, accumulating repeated indices."""
        ...

    def subtract(self, value: ArrayOperand | HostArray) -> array:
        """Subtract ``value`` at the selected indices, accumulating repeated indices."""
        ...

    def multiply(self, value: ArrayOperand | HostArray) -> array:
        """Multiply by ``value`` at the selected indices, accumulating repeated indices."""
        ...

    def divide(self, value: ArrayOperand | HostArray) -> array:
        """Divide by ``value`` at the selected indices, accumulating repeated indices."""
        ...

    def maximum(self, value: ArrayOperand | HostArray) -> array:
        """Take the elementwise maximum with ``value`` at the selected indices."""
        ...

    def minimum(self, value: ArrayOperand | HostArray) -> array:
        """Take the elementwise minimum with ``value`` at the selected indices."""
        ...


class IndexedUpdates(Protocol):
    """The object ``array.at``; indexing it selects the elements an update applies to."""

    def __getitem__(self, indices: object, /) -> IndexedUpdate:
        """Select the elements to update."""
        ...


class MetalRuntime(Protocol):
    """The part of ``mx.metal`` used to check backend availability."""

    def is_available(self) -> bool:
        """Return whether the Metal backend is present on this host."""
        ...


class RandomRuntime(Protocol):
    """The part of ``mx.random`` used for seeded sampling."""

    def key(self, seed: int) -> array:
        """Create an explicit random key from a seed."""
        ...

    def seed(self, seed: int) -> None:
        """Set the default random seed."""
        ...

    def uniform(
        self,
        low: Scalar | array = 0,
        high: Scalar | array = 1,
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Draw uniformly distributed values in ``[low, high)``."""
        ...

    def normal(
        self,
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        loc: Scalar | array | None = None,
        scale: Scalar | array | None = None,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Draw normally distributed values with the given mean and standard deviation."""
        ...


class FastRuntime(Protocol):
    """The part of ``mx.fast`` used by MedMLX code."""

    def layer_norm(
        self,
        x: array,
        weight: array | None,
        bias: array | None,
        eps: float,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Normalize the last axis, then scale and shift with optional parameters."""
        ...

    def scaled_dot_product_attention(
        self,
        q: array,
        k: array,
        v: array,
        *,
        scale: float,
        mask: str | array | None = None,
        sinks: array | None = None,
        force_fused: bool = False,
        stream: StreamOrDevice = None,
    ) -> array:
        """Compute fused attention over queries, keys and values."""
        ...

    def metal_kernel(
        self,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
        compile_options: Mapping[str, str] | None = None,
    ) -> MetalKernel:
        """Compile Metal source into a kernel callable with the declared arguments."""
        ...


class NeuralRuntime(Protocol):
    """The part of ``mlx.nn`` used at inference: pooling and activation functions.

    The layer classes (``Module``, ``Conv3d`` and others) stay with the runtime that
    subclasses them: core ships no stubs for ``mlx.nn``.
    """

    def MaxPool2d(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | None = 0,
    ) -> ArrayLayer:
        """Build a 2-D max pooling layer over channels-last input."""
        ...

    def MaxPool3d(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] | None = 0,
    ) -> ArrayLayer:
        """Build a 3-D max pooling layer over channels-last input."""
        ...

    def relu(self, x: array) -> array:
        """Apply the rectified linear unit."""
        ...

    def leaky_relu(self, x: array, negative_slope: float = 0.01) -> array:
        """Apply leaky ReLU with the selected negative slope."""
        ...

    def gelu(self, x: array) -> array:
        """Apply the exact Gaussian error linear unit."""
        ...

    def elu(self, x: array, alpha: float = 1.0) -> array:
        """Apply the exponential linear unit."""
        ...

    def silu(self, x: array) -> array:
        """Apply the sigmoid linear unit."""
        ...


class UtilsRuntime(Protocol):
    """The part of ``mlx.utils`` that flattens and rebuilds parameter trees."""

    @overload
    def tree_flatten[T](
        self,
        tree: Tree[T],
        prefix: str = "",
        is_leaf: Callable[[object], bool] | None = None,
        destination: None = None,
    ) -> list[tuple[str, T]]:
        """Flatten a tree into ``(dotted path, leaf)`` pairs."""
        ...

    @overload
    def tree_flatten[T](
        self,
        tree: Tree[T],
        prefix: str,
        is_leaf: Callable[[object], bool] | None,
        destination: dict[str, T],
    ) -> dict[str, T]:
        """Flatten a tree into the given dictionary keyed by dotted path."""
        ...

    def tree_unflatten[T](self, tree: list[tuple[str, T]] | dict[str, T]) -> Tree[T]:
        """Rebuild the nested tree from dotted-path leaves."""
        ...

    def tree_map_with_path[T, R](
        self,
        fn: Callable[..., R],
        tree: Tree[T],
        *rest: Tree[T],
        is_leaf: Callable[[object], bool] | None = None,
        path: str | None = None,
    ) -> Tree[R]:
        """Apply ``fn(path, leaf, *rest_leaves)`` to every leaf of the tree."""
        ...


class MlxRuntime(Protocol):
    """The part of ``mlx.core`` used by MedMLX code, so modules take an injected ``mx``."""

    @property
    def __version__(self) -> str:
        """Return the installed MLX version."""
        ...

    @property
    def __file__(self) -> str:
        """Return the path of the MLX extension module."""
        ...

    @property
    def array(self) -> ArrayConstructor:
        """Return the MLX array constructor, which also accepts NumPy arrays."""
        ...

    @property
    def float32(self) -> Dtype:
        """Return the single-precision floating dtype."""
        ...

    @property
    def float16(self) -> Dtype:
        """Return the half-precision floating dtype."""
        ...

    @property
    def bfloat16(self) -> Dtype:
        """Return the bfloat16 dtype."""
        ...

    @property
    def complex64(self) -> Dtype:
        """Return the 64-bit complex dtype."""
        ...

    @property
    def int32(self) -> Dtype:
        """Return the signed 32-bit integer dtype."""
        ...

    @property
    def int64(self) -> Dtype:
        """Return the signed 64-bit integer dtype."""
        ...

    @property
    def uint16(self) -> Dtype:
        """Return the unsigned 16-bit integer dtype."""
        ...

    @property
    def uint32(self) -> Dtype:
        """Return the unsigned 32-bit integer dtype."""
        ...

    @property
    def bool_(self) -> Dtype:
        """Return the boolean dtype."""
        ...

    @property
    def gpu(self) -> DeviceType:
        """Return the GPU device kind accepted by ``set_default_device``."""
        ...

    @property
    def fast(self) -> FastRuntime:
        """Return the fused-operation namespace."""
        ...

    @property
    def metal(self) -> MetalRuntime:
        """Return the Metal backend namespace."""
        ...

    @property
    def random(self) -> RandomRuntime:
        """Return the random-number namespace."""
        ...

    def eval(self, *args: ArrayTree) -> None:
        """Evaluate the given arrays and their dependencies in place."""
        ...

    def async_eval(self, *args: ArrayTree) -> None:
        """Schedule arrays without waiting for their Metal work to finish."""
        ...

    def synchronize(self, stream: StreamOrDevice = None) -> None:
        """Wait for the work queued on a stream, or on the default stream."""
        ...

    def stream(self, s: Stream | ThreadLocalStream | Device | DeviceType) -> StreamContext:
        """Return a context manager that runs operations on the given stream or device."""
        ...

    def new_thread_local_stream(self, device: Device | DeviceType) -> ThreadLocalStream:
        """Create a stream that is unique to each thread on the given device."""
        ...

    def compile[**P, R](
        self,
        fun: Callable[P, R],
        inputs: object | None = None,
        outputs: object | None = None,
        shapeless: bool = False,
    ) -> Callable[P, R]:
        """Compile a function, capturing the given state and shape options."""
        ...

    @overload
    def load(
        self,
        file: str | Path | BinaryIO,
        /,
        format: Literal["safetensors"],
        return_metadata: Literal[False] = False,
        *,
        stream: StreamOrDevice = None,
    ) -> dict[str, array]:
        """Load the arrays of a safetensors file."""
        ...

    @overload
    def load(
        self,
        file: str | Path | BinaryIO,
        /,
        format: Literal["safetensors"],
        return_metadata: Literal[True],
        *,
        stream: StreamOrDevice = None,
    ) -> tuple[dict[str, array], dict[str, str]]:
        """Load the arrays and the string metadata of a safetensors file."""
        ...

    @overload
    def load(
        self,
        file: str | Path | BinaryIO,
        /,
        format: str | None = None,
        return_metadata: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array | dict[str, array] | tuple[dict[str, array], dict[str, str]]:
        """Load an array, an archive of arrays, or arrays with metadata, by file format."""
        ...

    def save_safetensors(
        self,
        file: str | Path | BinaryIO,
        arrays: dict[str, array],
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Write named arrays and optional string metadata to a safetensors file."""
        ...

    def reset_peak_memory(self) -> None:
        """Reset the backend's peak memory counter."""
        ...

    def get_peak_memory(self) -> int:
        """Return the peak memory use recorded since the last reset, in bytes."""
        ...

    def get_active_memory(self) -> int:
        """Return the memory held by live arrays, in bytes."""
        ...

    def get_cache_memory(self) -> int:
        """Return the memory held by the allocator cache, in bytes."""
        ...

    def get_memory_limit(self) -> int:
        """Return the allocator's memory limit, in bytes."""
        ...

    def set_memory_limit(self, limit: int) -> int:
        """Set the allocator's memory limit in bytes and return the previous limit."""
        ...

    def set_cache_limit(self, limit: int) -> int:
        """Set the allocator cache limit in bytes and return the previous limit."""
        ...

    def clear_cache(self) -> None:
        """Release the allocator's cached buffers."""
        ...

    def device_info(self, d: Device | DeviceType | None = None) -> dict[str, str | int]:
        """Return the properties of a device, or of the default device."""
        ...

    def default_device(self) -> Device:
        """Return the device that new arrays are placed on."""
        ...

    def set_default_device(self, device: Device | DeviceType) -> None:
        """Place new arrays on the given device."""
        ...

    def reshape(self, a: array, /, shape: Sequence[int], *, stream: StreamOrDevice = None) -> array:
        """Return ``a`` with its elements reinterpreted in ``shape``."""
        ...

    def abs(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise absolute value."""
        ...

    def cos(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise cosine."""
        ...

    def sin(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise sine."""
        ...

    def exp(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise exponential."""
        ...

    def expm1(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise ``exp(a) - 1``."""
        ...

    def log(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise natural logarithm."""
        ...

    def log1p(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise ``log(1 + a)``."""
        ...

    def erf(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise Gauss error function."""
        ...

    def reciprocal(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise reciprocal."""
        ...

    def rsqrt(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise reciprocal square root."""
        ...

    def sqrt(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise square root."""
        ...

    def square(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise square."""
        ...

    def sigmoid(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise logistic function."""
        ...

    def floor(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise greatest integer not above each value."""
        ...

    def ceil(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise least integer not below each value."""
        ...

    def isfinite(self, a: array, stream: StreamOrDevice = None) -> array:
        """Return a boolean array marking elements that are neither infinite nor NaN."""
        ...

    def softmax(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        *,
        precise: bool = False,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply softmax along the given axes."""
        ...

    @overload
    def arange(
        self,
        start: int | float,
        stop: int | float | None,
        step: int | float | None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return evenly spaced values from ``start`` up to ``stop``."""
        ...

    @overload
    def arange(
        self,
        stop: int | float,
        step: int | float | None = None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return evenly spaced values from zero up to ``stop``."""
        ...

    def take(
        self,
        a: array,
        /,
        indices: int | array,
        axis: int | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Gather entries at ``indices`` along ``axis``, or from the flattened array."""
        ...

    def zeros(
        self,
        shape: int | Sequence[int],
        dtype: Dtype | None = ...,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return a zero-filled array of the given shape and dtype."""
        ...

    def ones(
        self,
        shape: int | Sequence[int],
        dtype: Dtype | None = ...,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return a one-filled array of the given shape and dtype."""
        ...

    def zeros_like(
        self, a: array, /, dtype: Dtype | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Return a zero-filled array shaped like ``a``."""
        ...

    def ones_like(
        self, a: array, /, dtype: Dtype | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Return a one-filled array shaped like ``a``."""
        ...

    def full(
        self,
        shape: int | Sequence[int],
        vals: ArrayOperand,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return an array of the given shape filled with ``vals``."""
        ...

    def eye(
        self,
        n: int,
        m: int | None = None,
        k: int = 0,
        dtype: Dtype | None = ...,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return a matrix with ones on the ``k``-th diagonal and zeros elsewhere."""
        ...

    def minimum(
        self, a: ArrayOperand, b: ArrayOperand, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise smaller of two operands."""
        ...

    def maximum(
        self, a: ArrayOperand, b: ArrayOperand, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise larger of two operands."""
        ...

    def clip(
        self,
        a: array,
        /,
        a_min: ArrayOperand | None,
        a_max: ArrayOperand | None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Limit each element to ``[a_min, a_max]``, leaving an omitted bound open."""
        ...

    def bitwise_and(self, a: ArrayOperand, b: ArrayOperand, stream: StreamOrDevice = None) -> array:
        """Return the elementwise bitwise AND of two integer operands."""
        ...

    def left_shift(self, a: ArrayOperand, b: ArrayOperand, stream: StreamOrDevice = None) -> array:
        """Shift each element of ``a`` left by the matching element of ``b``."""
        ...

    def logical_or(self, a: array, b: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise logical OR of two arrays."""
        ...

    def logical_not(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise logical NOT of an array."""
        ...

    def where(
        self,
        condition: ArrayOperand,
        x: ArrayOperand,
        y: ArrayOperand,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Select ``x`` where ``condition`` is true and ``y`` elsewhere."""
        ...

    def matmul(self, a: array, b: array, /, *, stream: StreamOrDevice = None) -> array:
        """Multiply two arrays with matrix multiplication semantics."""
        ...

    def addmm(
        self,
        c: array,
        a: array,
        b: array,
        /,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return ``beta * c + alpha * (a @ b)``."""
        ...

    def all(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return whether every element is true over the given axes."""
        ...

    def any(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return whether any element is true over the given axes."""
        ...

    def sum(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the sum over the given axes."""
        ...

    def min(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the minimum over the given axes."""
        ...

    def max(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the maximum over the given axes."""
        ...

    def argmax(
        self,
        a: array,
        /,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the index of the largest value along ``axis``, or in the flattened array."""
        ...

    def argmin(
        self,
        a: array,
        /,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the index of the smallest value along ``axis``, or in the flattened array."""
        ...

    def mean(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the arithmetic mean over the given axes."""
        ...

    def var(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        ddof: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the variance over the given axes with ``ddof`` correction."""
        ...

    def swapaxes(
        self, a: array, /, axis1: int, axis2: int, *, stream: StreamOrDevice = None
    ) -> array:
        """Return ``a`` with two axes exchanged."""
        ...

    def moveaxis(
        self, a: array, /, source: int, destination: int, *, stream: StreamOrDevice = None
    ) -> array:
        """Return ``a`` with one axis moved to a new position."""
        ...

    def transpose(
        self,
        a: array,
        /,
        axes: Sequence[int] | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return ``a`` with its axes permuted, reversing them when ``axes`` is omitted."""
        ...

    def squeeze(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Remove axes of length one."""
        ...

    def expand_dims(
        self, a: array, /, axis: int | Sequence[int], *, stream: StreamOrDevice = None
    ) -> array:
        """Insert axes of length one at the given positions."""
        ...

    def broadcast_to(
        self, a: ArrayOperand, /, shape: Sequence[int], *, stream: StreamOrDevice = None
    ) -> array:
        """Broadcast ``a`` to ``shape``."""
        ...

    def tile(
        self, a: array, reps: int | Sequence[int], /, *, stream: StreamOrDevice = None
    ) -> array:
        """Repeat ``a`` as a block ``reps`` times along each axis."""
        ...

    def flip(
        self,
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reverse the elements along the given axes."""
        ...

    def roll(
        self,
        a: array,
        shift: int | Sequence[int],
        axis: int | Sequence[int] | None = None,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Shift elements circularly along the given axes."""
        ...

    def contiguous(
        self, a: array, /, allow_col_major: bool = False, *, stream: StreamOrDevice = None
    ) -> array:
        """Return an array with a row-contiguous memory layout."""
        ...

    def concatenate(
        self, arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along an existing axis."""
        ...

    def stack(
        self, arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join equally shaped arrays along a new axis."""
        ...

    def split(
        self,
        a: array,
        /,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Split ``a`` into equal sections or at the given indices along ``axis``."""
        ...

    def meshgrid(
        self,
        *arrays: array,
        sparse: bool | None = False,
        indexing: str | None = "xy",
        stream: StreamOrDevice = None,
    ) -> tuple[array, ...]:
        """Return coordinate arrays from one-dimensional coordinate vectors."""
        ...

    def repeat(
        self,
        array: array,
        repeats: int,
        axis: int | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Repeat each element ``repeats`` times along ``axis``, or in flattened order."""
        ...

    def slice_update(
        self,
        a: array,
        update: array,
        start_indices: array,
        axes: Sequence[int],
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return ``a`` with a slice starting at ``start_indices`` replaced by ``update``."""
        ...

    def pad(
        self,
        a: array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        constant_values: ArrayOperand = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Pad each axis to the requested widths using the given boundary rule."""
        ...

    def conv1d(
        self,
        input: array,
        weight: array,
        /,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Convolve NLC input with an OLI filter, optionally in depthwise groups."""
        ...

    def conv2d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Convolve NHWC input with an OHWI filter, optionally in depthwise groups."""
        ...

    def conv3d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Convolve NDHWC input with an ODHWI filter, optionally in depthwise groups."""
        ...

    def conv_transpose2d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Transpose-convolve NHWC input with an OHWI filter, growing each spatial axis."""
        ...

    def conv_transpose3d(
        self,
        input: array,
        weight: array,
        /,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Transpose-convolve NDHWC input with an ODHWI filter, growing each spatial axis."""
        ...
