"""Verify arithmetic and interpolation contracts through both actual Apple runtimes."""

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from medmlx_convert import export_and_check
from medmlx_convert import runtime_benchmark as benchmark
from medmlx_convert.boundary import Array, Module, as_tensor, function
from medmlx_convert.evidence import Verdict
from medmlx_convert.provenance import BuildSource
from medmlx_convert.rewrites import COREAI_3D, COREML_3D
from medmlx_convert.runtime_benchmark import CallPhase, Profile, Workload, inspect_coreai, measure


class Subtract(Module):
    """Exercise subtraction aliases introduced by Torch 2.13 graph capture."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        difference = as_tensor(function(torch.ops.aten.alias, "default")(torch.subtract(x, y)))
        return difference, torch.nn.functional.elu(difference)


@settings(max_examples=3, deadline=None)
@given(
    x=arrays(np.float32, (2, 3), elements=st.floats(-10, 10, width=32, allow_subnormal=False)),
    y=arrays(np.float32, (2, 3), elements=st.floats(-10, 10, width=32, allow_subnormal=False)),
)
def test_export_roundtrip(x: Array, y: Array) -> None:
    out = Path("build/roundtrip")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {name: as_tensor(function(torch, "from_numpy")(a)) for name, a in (("x", x), ("y", y))}
    assert export_and_check(
        Subtract(), inputs, ["difference", "activation"], out / "keras_ops", tolerance=1e-5,
        source=BuildSource.capture(Path(__file__), synthetic=True),
    )
    stem = out / "keras_ops"

    async def check_calls(profile: Profile) -> None:
        workload = Workload.load(stem, stem.with_suffix(".runtime.npz"), profile.backend)
        report, session = await measure(workload, profile, runs=2, warmup=1)
        assert report.verdict == Verdict.PASS
        assert [call.phase for call in report.calls] == [
            CallPhase.FIRST, CallPhase.WARMUP, CallPhase.TIMED, CallPhase.TIMED,
        ]
        if profile.backend == "coreai":
            _, inspection = await inspect_coreai(session, workload)
            assert inspection.phase == CallPhase.INSPECTION and inspection.verdict == Verdict.PASS

    for profile in (Profile.COREAI_DEFAULT, Profile.COREML_ALL):
        asyncio.run(check_calls(profile))


def test_failed_profiling_call_invalidates_the_persisted_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a post-timing inspection failure out of passing benchmark evidence."""
    stem = Path("build/stage_gate/profile_failure")
    stem.parent.mkdir(parents=True, exist_ok=True)
    assert export_and_check(
        Subtract(), {"x": torch.ones(2, 3), "y": torch.zeros(2, 3)},
        ["difference", "activation"], stem, tolerance=1e-5,
        source=BuildSource.capture(Path(__file__), synthetic=True),
    )
    original = benchmark.inspect_coreai

    async def corrupted_inspection(session: benchmark.Session, workload: Workload
                                    ) -> tuple[list[benchmark.ProfileEvent], benchmark.CallCheck]:
        events, checked = await original(session, workload)
        corrupted: dict[str, Array] = {
            name: reference.astype(np.float32) + np.float32(1)
            for name, reference in workload.references.items()
        }
        comparisons = workload.compare("coreai", corrupted)
        return events, replace(checked, comparisons=comparisons, verdict=Verdict.FAIL,
                               error="Injected test-only profiling output corruption")

    monkeypatch.setattr(benchmark, "inspect_coreai", corrupted_inspection)
    out = stem.parent / "profile_failure_run.json"
    monkeypatch.setattr(sys, "argv", [
        "runtime_benchmark", str(stem), str(stem.with_suffix(".runtime.npz")),
        "coreai-default", str(out), "--runs", "1", "--warmup", "0", "--inspect",
    ])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 1
    persisted = json.loads(out.read_text())
    assert persisted["verdict"] == "fail"
    assert [call["phase"] for call in persisted["calls"]] == ["first", "timed", "inspection"]
    assert persisted["calls"][-1]["verdict"] == "fail"
    assert all(call["verdict"] == "pass" for call in persisted["calls"][:-1])
    assert json.loads(out.with_suffix(".inspection.json").read_text())["call"]["verdict"] == "fail"


class FractionalUpsample(Module):
    def __init__(self, recompute: bool | None, align: bool) -> None:
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=1.5, mode="trilinear",
                                         align_corners=align, recompute_scale_factor=recompute)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return as_tensor(self.upsample(image))


class ScalarAndSqueeze(Module):
    """Exercise scalar where branches, boolean logic and no-op squeezing."""

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        positive, negative = image > 0, image < 0
        return (
            torch.where(positive, image, -2.0).squeeze(2),
            torch.where(negative, 3.0, image).squeeze(2),
            ((positive & ~negative) | negative).to(torch.float32),
        )


@pytest.mark.parametrize("width", [1, 5])
@settings(max_examples=3, deadline=None)
@given(seed=st.integers(0, 65535))
def test_lowering_semantics_survive_both_native_formats(width: int, seed: int) -> None:
    """Replay graph edits and execute both saved artifacts over varied tensor values."""
    out = Path("build/lowering_roundtrip")
    out.mkdir(parents=True, exist_ok=True)
    image = torch.randn((1, 2, width), generator=torch.Generator().manual_seed(seed))
    assert export_and_check(
        ScalarAndSqueeze(), {"image": image}, ["other", "self", "boolean"],
        out / f"width_{width}", tolerance=1e-5,
        source=BuildSource.capture(Path(__file__), synthetic=True),
    )


@pytest.mark.parametrize(("recompute", "align"), [(None, False), (False, False),
                                                (True, False), (None, True)])
def test_fractional_trilinear_native_export_contract(recompute: bool | None, align: bool) -> None:
    """Verify native parity or explicit converter rejection without a passing numerical result."""
    out = Path("build/trilinear_roundtrip")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"recompute_{recompute}_align_{align}"
    image = torch.randn((1, 2, 5, 5, 5), generator=torch.Generator().manual_seed(0))

    def run_export() -> bool:
        return export_and_check(
            FractionalUpsample(recompute, align), {"image": image}, ["features"], stem,
            tolerance=1e-4, source=BuildSource.capture(Path(__file__), synthetic=True),
            coreai_swaps=COREAI_3D, coreml_swaps=COREML_3D,
        )

    if not align and recompute is not True:
        # coremltools 9 cannot preserve these coordinates for nonintegral output sizes.
        with pytest.raises(NotImplementedError, match="float output size is not supported"):
            run_export()
        return
    assert run_export()
    for backend in ("coreai", "coreml"):
        Workload.load(stem, stem.with_suffix(".runtime.npz"), backend)


class ScalarBitwise(Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ((image.to(torch.int32) & 3) | 1).to(torch.float32)


def test_unsupported_scalar_bitwise_cannot_receive_passing_evidence() -> None:
    """Keep the observed Core AI scalar-overload rejection explicit."""
    stem = Path("build/lowering_roundtrip/scalar_rejection")
    stem.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="unsupported ATen ops:.*bitwise_and.Scalar"):
        export_and_check(
            ScalarBitwise(), {"image": torch.ones((1, 2, 5))}, ["features"],
            stem, tolerance=0, source=BuildSource.capture(Path(__file__), synthetic=True),
        )


class TensorBitwise(Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        integers = image.to(torch.int32)
        return ((integers & integers.new_full((), 3)) | integers.new_full((), 1)).to(torch.float32)


def test_coreml_integer_bitwise_cannot_be_replaced_by_boolean_logic() -> None:
    """Reject integer semantics rather than silently treating every nonzero value as true."""
    stem = Path("build/lowering_roundtrip/integer_rejection")
    stem.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(NotImplementedError, match="Core ML integer bitwise"):
        export_and_check(
            TensorBitwise(), {"image": torch.ones((1, 2, 5))}, ["features"],
            stem, tolerance=0, source=BuildSource.capture(Path(__file__), synthetic=True),
        )
