"""An intermittent bad inference must invalidate its entire benchmark run."""

import asyncio
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from medmlx_convert import runtime_benchmark as benchmark
from medmlx_convert.boundary import Array
from medmlx_convert.evidence import InputTensor, Verdict


@dataclass(frozen=True)
class ScriptedSession:
    calls: Iterator[benchmark.Call]
    model: object = None

    async def infer(self) -> benchmark.Call:
        return next(self.calls)


def run_calls(calls: list[benchmark.Call], warmup: int, runs: int,
              profile: benchmark.Profile) -> benchmark.Run:
    """Exercise the public measurement boundary with deterministic completed outputs."""
    reference = np.ones(8, dtype=np.float32)
    workload = benchmark.Workload(
        Path("build/network"), Path("build/network.runtime.npz"),
        (InputTensor("image", reference.shape, "float32"),),
        {"image": reference}, {"features": reference}, {"features": 0.1},
    )
    session = ScriptedSession(iter(calls))

    def load_coreml(workload: benchmark.Workload, profile: benchmark.Profile) -> ScriptedSession:
        return session

    async def load_coreai(workload: benchmark.Workload, profile: benchmark.Profile
                          ) -> tuple[ScriptedSession, float]:
        return session, 0.0

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(benchmark, "load_coreml", load_coreml)
        patch.setattr(benchmark, "load_coreai", load_coreai)
        report, _ = asyncio.run(benchmark.measure(workload, profile, runs, warmup))
    return report


@pytest.mark.parametrize("profile", [benchmark.Profile.COREML_ALL,
                                     benchmark.Profile.COREAI_DEFAULT])
@pytest.mark.parametrize("failure", ["absolute_error", "psnr", "nonfinite",
                                     "shape", "dtype", "missing"])
@settings(max_examples=20, deadline=None)
@example(warmup=2, runs=4, fault=2)
@example(warmup=2, runs=4, fault=4)
@given(warmup=st.integers(0, 3), runs=st.integers(1, 5), fault=st.integers(0, 8))
def test_any_failed_call_disqualifies_the_run(
    profile: benchmark.Profile, failure: str, warmup: int, runs: int, fault: int,
) -> None:
    """Retain the first failing phase, including failures surrounded by correct calls."""
    reference = np.ones(8, dtype=np.float32)
    values: dict[str, dict[str, Array]] = {
        "absolute_error": {"features": reference + np.float32(0.2)},
        "psnr": {"features": reference + np.float32(0.01)},
        "nonfinite": {"features": reference * np.float32(np.nan)},
        "shape": {"features": reference.reshape(2, 4)},
        "dtype": {"features": reference.astype(np.float64)},
        "missing": {},
    }
    phases = ([(benchmark.CallPhase.FIRST, 1)]
              + [(benchmark.CallPhase.WARMUP, i) for i in range(1, warmup + 1)]
              + [(benchmark.CallPhase.TIMED, i) for i in range(1, runs + 1)])
    fault %= len(phases)
    timing = benchmark.Timing(0.01, 0.02, 0.03, 0.06)
    calls = [benchmark.Call(timing, values[failure] if i == fault else {"features": reference})
             for i in range(len(phases))]
    report = run_calls(calls, warmup, runs, profile)
    assert report.verdict == Verdict.FAIL
    assert len(report.calls) == fault + 1
    assert (report.calls[-1].phase, report.calls[-1].iteration) == phases[fault]
    assert report.calls[-1].verdict == Verdict.FAIL
    assert all(call.verdict == Verdict.PASS for call in report.calls[:-1])
    # Malformed/nonfinite tensors remain serializable failure evidence.
    persisted = json.loads(json.dumps(asdict(report), allow_nan=False))
    assert persisted["verdict"] == "fail" and persisted["schema_version"] == 4


@given(warmup=st.integers(0, 3), runs=st.integers(1, 5))
def test_passing_run_retains_every_call_outside_its_timing_interval(warmup: int, runs: int) -> None:
    reference = np.ones(8, dtype=np.float32)
    timing = benchmark.Timing(0.01, 0.02, 0.03, 0.06)
    calls = [benchmark.Call(timing, {"features": reference}) for _ in range(1 + warmup + runs)]
    report = run_calls(calls, warmup, runs, benchmark.Profile.COREML_ALL)
    assert report.verdict == Verdict.PASS
    assert len(report.calls) == 1 + warmup + runs
    assert all(call.timing == timing and call.verdict == Verdict.PASS for call in report.calls)
    assert sum(call.phase == benchmark.CallPhase.WARMUP for call in report.calls) == warmup
    assert sum(call.phase == benchmark.CallPhase.TIMED for call in report.calls) == runs
