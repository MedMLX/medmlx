"""Typed MLX boundaries without importing the optional runtime at package import."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import Device, DeviceType, Dtype, StreamOrDevice, array

type HostArray = NDArray[np.generic]
type Scalar = int | float | bool


class ArrayFactory(Protocol):
    """Callable that constructs an MLX array from host data."""

    def __call__(
        self, val: array | HostArray | list[float], dtype: Dtype | None = None
    ) -> array:
        """Build an array from host data, converting when a dtype is requested."""
        ...


class ArrayEvaluator(Protocol):
    """Callable that materializes lazy arrays on the compute device."""

    def __call__(self, *args: array | Sequence[array]) -> None:
        """Schedule every argument, and its dependencies, for evaluation."""
        ...


class MetalKernel(Protocol):
    """Callable returned by ``mx.fast.metal_kernel``; ``grid`` counts threads, not groups."""

    def __call__(
        self,
        *,
        inputs: list[array],
        template: Sequence[tuple[str, Dtype | int | bool]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Dtype],
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Launch the kernel and return one array per declared output."""
        ...


class MetalRuntime(Protocol):
    """The part of ``mx.metal`` used to check backend availability."""

    def is_available(self) -> bool:
        """Return whether the Metal backend is present on this host."""
        ...


class FastRuntime(Protocol):
    """The part of ``mx.fast`` used by this package."""

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

    def metal_kernel(
        self,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
        compile_options: object | None = None,
    ) -> object:
        """Compile Metal source into a kernel callable with the declared arguments."""
        ...


class MlxRuntime(Protocol):
    """The part of ``mlx.core`` used by this package, so modules take an injected ``mx``."""

    @property
    def array(self) -> type[array]:
        """Return the MLX array constructor."""
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
    def int32(self) -> Dtype:
        """Return the signed 32-bit integer dtype."""
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

    def eval(self, *args: array | Sequence[array]) -> None:
        """Evaluate the given arrays and their dependencies in place."""
        ...

    def reset_peak_memory(self) -> None:
        """Reset the backend's peak memory counter."""
        ...

    def get_peak_memory(self) -> int:
        """Return the peak memory use recorded since the last reset, in bytes."""
        ...

    def default_device(self) -> Device:
        """Return the device that new arrays are placed on."""
        ...

    def set_default_device(self, device: Device | DeviceType) -> None:
        """Place new arrays on the given device."""
        ...

    def reshape(
        self, a: array, /, shape: Sequence[int], *, stream: StreamOrDevice = None
    ) -> array:
        """Return ``a`` with its elements reinterpreted in ``shape``."""
        ...

    def rsqrt(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise reciprocal square root."""
        ...

    def sigmoid(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise logistic function."""
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
    ) -> array: ...
    @overload
    def arange(
        self,
        stop: int | float,
        step: int | float | None = None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array: ...
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
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return a zero-filled array of the given shape and dtype."""
        ...

    def minimum(
        self, a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise smaller of two operands."""
        ...

    def maximum(
        self, a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise larger of two operands."""
        ...

    def floor(self, a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Return the elementwise greatest integer not above each value."""
        ...

    def swapaxes(
        self, a: array, /, axis1: int, axis2: int, *, stream: StreamOrDevice = None
    ) -> array:
        """Return ``a`` with two axes exchanged."""
        ...

    def transpose(
        self, a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Return ``a`` with its axes permuted, reversing them when ``axes`` is omitted."""
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

    def concatenate(
        self, arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along an existing axis."""
        ...

    def stack(
        self, arrays: list[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join equally shaped arrays along a new axis."""
        ...

    def repeat(
        self, array: array, repeats: int, axis: int | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Repeat each element ``repeats`` times along ``axis``, or in flattened order."""
        ...

    def pad(
        self,
        a: array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        constant_values: Scalar | array = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Pad each axis to the requested widths using the given boundary rule."""
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

    def where(
        self,
        condition: Scalar | array,
        x: Scalar | array,
        y: Scalar | array,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Select ``x`` where ``condition`` is true and ``y`` elsewhere."""
        ...
