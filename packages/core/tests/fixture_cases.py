"""Canonical seeded cases and versioned upstream-reference fixture boundary."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import array as Array

    from medmlx_core.typing import ArrayFactory, MlxRuntime

type HostArray = NDArray[np.number]
type CaseValue = int | float | str | list[int] | tuple[int, ...]


class CaseOptions(TypedDict, total=False):
    name: str
    spatial: list[int] | tuple[int, ...]
    padding: int | list[int] | tuple[int, int, int]
    stride: int | list[int] | tuple[int, int, int]
    output_padding: int
    num_splits: int
    dim_split: int
    num_groups: int
    eps: float
    scale: int
    kernel_size: int


RECORDING_PLATFORM = "darwin-arm64"
FIXTURES = Path(__file__).parent / "fixtures" / RECORDING_PLATFORM


@dataclass(frozen=True, slots=True)
class ArrayCase:
    """One call of a package function on named seeded arrays.

    ``error`` marks cases expected to raise; they are not recorded as upstream
    fixtures.
    """

    id: str
    function: str
    args: tuple[str | None, ...]
    kwargs: tuple[tuple[str, CaseValue], ...]
    error: type[Exception] | None = None

    @property
    def options(self) -> CaseOptions:
        # Cases are built from the declared fixture options below, never external payloads.
        return cast(CaseOptions, dict(self.kwargs))


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    schema_version: int
    monai_version: str
    monai_revision: str
    torch_version: str
    torch_revision: str


@dataclass(frozen=True, slots=True)
class RecordedCase:
    id: str
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixtureMetadata:
    """Provenance stored as JSON under ``metadata`` in each recorded .npz fixture."""

    upstream: ReferenceSpec
    cases: tuple[RecordedCase, ...]

    @classmethod
    def from_json(cls, value: str) -> FixtureMetadata:
        payload = json.loads(value)
        if payload.get("schema_version") != 2 or payload.get("platform") != RECORDING_PLATFORM:
            raise ValueError("Fixtures must use schema 2 and be recorded on darwin-arm64")
        cases = tuple(RecordedCase(case["id"], tuple(case["outputs"]))
                      for case in payload.get("cases", []))
        return cls(ReferenceSpec(**payload["upstream"]), cases)


REFERENCE = ReferenceSpec(**json.loads((Path(__file__).parent / "reference_spec.json").read_text()))


class ChannelFixtureRecorder(Protocol):
    """Operations that rebuild seeded channel-last test inputs."""

    def seeded_arrays(self) -> tuple[np.random.Generator, dict[str, HostArray]]:
        """Recreate the initial inputs and advance the generator."""
        ...

    def record_extra_layers(self, rng: np.random.Generator, arrays: dict[str, HostArray]) -> None:
        """Recreate parameters for additional operators."""
        ...

    def record_residuals(self, rng: np.random.Generator, arrays: dict[str, HostArray]) -> None:
        """Recreate the reduced residual-block parameters."""
        ...


def _restore_seeded_inputs(name: str, arrays: dict[str, HostArray]) -> None:
    """Recreate test inputs and parameters while retaining recorded source outputs."""
    if name in {"layout", "ops", "upsample", "precision"}:
        inputs, _ = array_cases()
        arrays.update(inputs)
        if name == "layout":
            arrays["conv3d_weight_to_mlx_4__output_0"] = inputs["w"].transpose(0, 2, 3, 4, 1)
            arrays["conv_transpose3d_weight_to_mlx_5__output_0"] = inputs["wt"].transpose(
                1, 2, 3, 4, 0
            )
    elif name == "channel_last":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        recorder = cast(ChannelFixtureRecorder, import_module("make_channel_last_fixtures"))
        import torch

        rng, inputs = recorder.seeded_arrays()
        with torch.no_grad():
            recorder.record_extra_layers(rng, inputs)
            recorder.record_residuals(rng, inputs)
        arrays.update(
            {key: value for key, value in inputs.items() if not key.startswith("expected_")}
        )
    elif name in {"sliding_window_constant", "sliding_window_gaussian"}:
        rng = np.random.default_rng(20261007)
        rng.integers(-64, 65, (1, 1, 11, 9, 7))
        arrays["weights"] = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 8
        arrays["bias"] = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 16


def load_fixture(name: str) -> tuple[FixtureMetadata, dict[str, HostArray]]:
    """Return the metadata and arrays of the recorded fixture ``name``.

    Raises ``FileNotFoundError`` if it is missing and ``ValueError`` if it was not
    recorded against the pinned upstream revisions.
    """
    path = FIXTURES / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream reference: {path}; run the README fixture commands"
        )
    with np.load(path, allow_pickle=False) as archive:
        metadata = FixtureMetadata.from_json(str(archive["metadata"]))
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    if metadata.upstream != REFERENCE:
        raise ValueError(f"Fixture does not match the pinned upstream source: {path}")
    _restore_seeded_inputs(name, arrays)
    return metadata, arrays


def array_cases() -> tuple[dict[str, HostArray], dict[str, list[ArrayCase]]]:
    """Return the seeded float32 input arrays and the case lists per module.

    Case ids are ``<function>_<index>``, so the order within each module must stay
    stable once fixtures are recorded.
    """
    rng = np.random.default_rng(7081)

    def random(shape: tuple[int, ...]) -> HostArray:
        return rng.standard_normal(shape).astype(np.float32)

    arrays = {
        "x": random((1, 2, 8, 6, 4)),
        "small": random((1, 2, 1, 2, 3)),
        "w": random((3, 2, 3, 3, 3)),
        "wt": random((2, 3, 3, 3, 3)),
        "wd": random((2, 3, 2, 2, 2)),
        "bias": random((3,)),
        "norm_weight": random((2,)),
        "norm_bias": random((2,)),
        "linear_x": random((2, 4)),
        "linear_w": random((3, 4)),
        "bad_rank": random((2, 3)),
        # No case reads "skip"; drawing it keeps the later seeded arrays unchanged.
        "skip": random((1, 2, 2, 4, 6)),
        "half": random((1, 2, 1, 2, 3)).astype(np.float16),
    }
    cases: dict[str, list[ArrayCase]] = {}

    def add(module: str, function: str, args: list[str | None], *,
            error: type[Exception] | None = None, **kwargs: CaseValue) -> None:
        items = cases.setdefault(module, [])
        items.append(ArrayCase(f"{function}_{len(items)}", function, tuple(args),
                               tuple(kwargs.items()), error))

    _add_layout_cases(add, arrays, random)
    _add_basic_cases(add)
    _add_split_cases(add)
    _add_resize_cases(add)
    # Appended after every other "ops" case so recorded fixture ids never shift.
    _add_split_divisibility_cases(add)
    _add_decoder_cases(add)
    _add_precision_cases(add, arrays, random)
    return arrays, cases


type AddCase = Callable[..., None]
type RandomArray = Callable[[tuple[int, ...]], HostArray]


def _add_layout_cases(add: AddCase, arrays: dict[str, HostArray], random: RandomArray) -> None:
    add("layout", "require_ncdhw", ["x"], name="input")
    add("layout", "require_ncdhw", ["bad_rank"], name="input", error=ValueError)
    add("layout", "to_ndhwc", ["x"])
    add("layout", "to_ncdhw", ["x"])
    add("layout", "conv3d_weight_to_mlx", ["w"])
    add("layout", "conv_transpose3d_weight_to_mlx", ["wt"])
    add("layout", "conv3d_weight_to_mlx", ["bad_rank"], error=ValueError)
    add("layout", "conv_transpose3d_weight_to_mlx", ["bad_rank"], error=ValueError)
    add("layout", "tokens_from_ncdhw", ["x"])
    # Use a fixed independently stored tokens input for the inverse mapping.
    arrays["tokens"] = random((1, 192, 2))
    add("layout", "tokens_to_ncdhw", ["tokens"], spatial=[1, 2, 8, 6, 4])


def _add_basic_cases(add: AddCase) -> None:
    add("ops", "as_fp32", ["half"])
    add("ops", "silu", ["x"])
    for bias in ("bias", None):
        add("ops", "linear", ["linear_x", "linear_w", bias])
        add("ops", "conv3d_ncdhw", ["x", "w", bias], padding=1)
        add("ops", "conv_transpose3d_ncdhw", ["small", "wt", bias],
            padding=1, stride=2, output_padding=1)
        add("upsample", "deconv2x_ncdhw", ["small", "wd", bias])
    for groups in (1, 2, 3):
        add("ops", "group_norm_ncdhw", ["x", "norm_weight", "norm_bias"],
            num_groups=groups, eps=1e-5, error=ValueError if groups == 3 else None)


def _add_split_cases(add: AddCase) -> None:
    for dim in (0, 1, 2):
        add("ops", "split_conv3d_ncdhw", ["x", "w", "bias"], padding=1, num_splits=2, dim_split=dim)
        add("ops", "split_conv_transpose3d_ncdhw", ["x", "wt", "bias"],
            padding=1, stride=2, output_padding=1, num_splits=2, dim_split=dim)
    add("ops", "split_conv3d_ncdhw", ["x", "w", None], padding=1)
    add("ops", "split_conv_transpose3d_ncdhw", ["x", "wt", None], padding=1, output_padding=0)
    add("ops", "split_conv3d_ncdhw", ["x", "w", None],
        padding=1, stride=2, num_splits=2, dim_split=0)
    add("ops", "split_conv3d_ncdhw", ["x", "w", None],
        padding=1, num_splits=2, dim_split=3, error=ValueError)
    add("ops", "split_conv3d_ncdhw", ["x", "w", None],
        padding=1, num_splits=20, dim_split=0, error=ValueError)


def _add_resize_cases(add: AddCase) -> None:
    add("ops", "avg_pool3d_ncdhw", ["x"])
    add("ops", "avg_pool3d_ncdhw", ["x"], kernel_size=3, stride=1)
    for scale in (1, 2, 0):
        add("ops", "upsample_nearest_ncdhw", ["small"],
            scale=scale, error=ValueError if scale == 0 else None)
    add("ops", "upsample_trilinear_ncdhw", ["small"])
    add("ops", "pad_spatial_trailing_ncdhw", ["small"])
    add("ops", "concat_channels_ncdhw", ["x", "x"])


def _add_split_divisibility_cases(add: AddCase) -> None:
    # "x" is (1, 2, 8, 6, 4); dim_split=2 is the W axis (length 4), not divisible
    # by num_splits=3. MaisiConvolution's crop drops the remainder silently;
    # medmlx-core refuses instead (see split_conv3d_ncdhw's docstring).
    add("ops", "split_conv3d_ncdhw", ["x", "w", None],
        padding=1, num_splits=3, dim_split=2, error=ValueError)
    add("ops", "split_conv_transpose3d_ncdhw", ["x", "wt", None],
        padding=1, output_padding=0, num_splits=3, dim_split=2, error=ValueError)


def _add_decoder_cases(add: AddCase) -> None:
    add("upsample", "deconv2x_ncdhw", ["half", "wd", None], error=TypeError)
    add("upsample", "deconv2x_ncdhw", ["small", "bad_rank", None], error=ValueError)
    add("upsample", "deconv2x_ncdhw", ["small", "w", None], error=ValueError)
    add("upsample", "deconv2x_ncdhw", ["small", "wd", "norm_bias"], error=ValueError)


def _add_precision_cases(add: AddCase, arrays: dict[str, HostArray], random: RandomArray) -> None:
    add("precision", "eps", [])
    add("precision", "conv", ["small", "w", "bias"], padding=1, stride=1)
    add("precision", "conv", ["small", "w", None], padding=[1, 1, 1], stride=[1, 1, 1])
    add("precision", "_moments", ["x"])
    add("precision", "norm", ["x", "norm_weight", "norm_bias"])
    for length in (3, 4, 64, 68, 132, 260):
        key = f"moments_{length}"
        arrays[key] = random((1, 2, 1, 1, length))
        add("precision", "_moments", [key])


def run_array_case(
    module: ModuleType, case: ArrayCase, arrays: dict[str, HostArray], mx: MlxRuntime
) -> Array | tuple[Array, ...] | None:
    """Call the case's function on MLX copies of its arrays.

    Precision cases go through ``Float32Operators`` (groups=1, eps=1e-5); other
    cases receive ``mx`` as a keyword, except ``require_ncdhw``.
    """
    make_array = cast("ArrayFactory", mx.array)
    args = [None if key is None else make_array(arrays[key]) for key in case.args]
    kwargs: dict[str, CaseValue | MlxRuntime] = dict(case.kwargs)
    if "spatial" in kwargs:
        kwargs["spatial"] = tuple(cast(list[int] | tuple[int, ...], kwargs["spatial"]))
    if hasattr(module, "Float32Operators"):
        from medmlx_core.precision import Float32Operators

        instance = Float32Operators(mx, groups=1, eps=1e-5)
        if case.function == "eps":
            return instance.eps
        return cast("Callable[..., Array | tuple[Array, ...]]", getattr(instance, case.function))(
            *args, **kwargs
        )
    if case.function != "require_ncdhw":
        kwargs["mx"] = mx
    return cast("Callable[..., Array | None]", getattr(module, case.function))(*args, **kwargs)
