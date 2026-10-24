"""Completed network calls under the admitted precision and Apple runtime policies."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import numpy as np

from medmlx_convert.boundary import Array, as_array, attribute, function, item, library
from medmlx_convert.evidence import (
    EXPORT_EVIDENCE_SCHEMA_VERSION,
    RUNTIME_FIXTURE_SCHEMA_VERSION,
    InputTensor,
    LabelRequirement,
    OutputComparison,
    Verdict,
    compare_output,
)
from medmlx_convert.guidance import (
    BuildProfile,
    native_comparison_verdict,
)
from medmlx_convert.provenance import SourceRevision

RUNTIME_RUN_SCHEMA_VERSION = 4


class Profile(StrEnum):
    COREML_ALL = "coreml-all"
    COREML_FAST = "coreml-fast"
    COREML_GPU = "coreml-gpu"
    COREML_NE = "coreml-ne"
    COREAI_DEFAULT = "coreai-default"
    COREAI_GPU = "coreai-gpu"
    COREAI_NE = "coreai-ne"
    COREAI_REUSE_BYTES = "coreai-reuse-bytes"
    COREAI_REUSE_METAL = "coreai-reuse-metal"
    COREAI_REUSE_IOSURFACE = "coreai-reuse-iosurface"
    MLX_GPU = "mlx-gpu"

    @property
    def backend(self) -> str:
        return self.value.split("-", 1)[0]

    @property
    def reuses_inputs(self) -> bool:
        return self in (
            Profile.COREAI_REUSE_BYTES, Profile.COREAI_REUSE_METAL,
            Profile.COREAI_REUSE_IOSURFACE,
        )

    @property
    def coreml_compute_unit(self) -> str:
        """Resolve the installed Core ML compute-unit name for an explicit profile."""
        if self.backend != "coreml":
            raise ValueError(f"Not a Core ML runtime profile: {self}")
        return {Profile.COREML_GPU: "CPU_AND_GPU", Profile.COREML_NE: "CPU_AND_NE"}.get(
            self, "ALL",
        )


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    entries = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in entries):
        raise TypeError("Expected a JSON object with string keys")
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("Expected a JSON list")
    return cast(list[object], value)


def text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected a string")
    return value


def integer(value: object) -> int:
    if type(value) is not int:
        raise TypeError("Expected an integer")
    return value


def number(value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError("Expected a number")
    result = float(cast(int | float, value))
    if not np.isfinite(result) or result < 0:
        raise ValueError("Expected a finite nonnegative number")
    return result


@dataclass(frozen=True)
class Workload:
    stem: Path
    fixture: Path
    inputs: tuple[InputTensor, ...]
    arrays: dict[str, Array]
    references: dict[str, Array]
    tolerances: dict[str, float]
    build_profile: BuildProfile = BuildProfile.FP32
    output_requirements: tuple[LabelRequirement, ...] = ()

    @classmethod
    def load(cls, stem: Path, fixture: Path, backend: str) -> Workload:
        """Load a passing numerical report and its matching finite source fixture."""
        if stem.is_absolute() or fixture.is_absolute():
            raise ValueError("Use paths relative to the repository root")
        if fixture != stem.with_suffix(".runtime.npz"):
            raise ValueError("Fixture does not match the build")
        # MLX shares the native source oracle and admits its checkpoint before loading.
        native_backend = "coreml" if backend == "mlx" else backend
        receipt = mapping(json.loads(stem.with_suffix(".verification.json").read_text()))
        if integer(receipt["schema_version"]) != EXPORT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Unsupported export receipt schema")
        if text(receipt["model"]) != stem.name:
            raise ValueError("Export evidence model does not match the build")
        build_profile = BuildProfile(text(receipt["build_profile"]))
        inputs: list[InputTensor] = []
        for raw in sequence(receipt["inputs"]):
            spec = mapping(raw)
            shape = tuple(integer(dim) for dim in sequence(spec["shape"]))
            if not shape or any(dim < 1 for dim in shape):
                raise ValueError("Input dimensions must be positive")
            dtype = text(spec["dtype"])
            if dtype not in ("float32", "int32", "int64"):
                raise ValueError(f"FP32 or integer inputs required; found {dtype}")
            inputs.append(InputTensor(text(spec["name"]), shape,
                                      "int32" if dtype == "int64" else dtype))
        tolerances: dict[str, float] = {}
        requirements: list[LabelRequirement] = []
        for raw in sequence(receipt["comparisons"]):
            comparison = mapping(raw)
            if Verdict(text(comparison["verdict"])) != Verdict.PASS:
                raise ValueError("Refusing to benchmark a failed export")
            if text(comparison["framework"]) != native_backend:
                continue
            name = text(comparison["output"])
            if name in tolerances:
                raise ValueError(f"Duplicate output comparison: {name}")
            tolerances[name] = number(comparison["tolerance"])
            raw_requirement = comparison["label_requirement"]
            if raw_requirement is not None:
                requirement = LabelRequirement.parse(raw_requirement)
                if requirement.output != name:
                    raise ValueError("Output requirement relationship differs from comparison")
                requirements.append(requirement)
        if not inputs or not tolerances or len({spec.name for spec in inputs}) != len(inputs):
            raise ValueError("Receipt requires unique inputs and passing backend outputs")
        with np.load(fixture, allow_pickle=False) as archive:
            schema = as_array(archive["schema_version"])
            if (schema.shape != () or schema.dtype.kind not in ("i", "u")
                    or schema != RUNTIME_FIXTURE_SCHEMA_VERSION):
                raise ValueError("Unsupported runtime fixture schema")
            arrays = {spec.name: as_array(archive[f"input/{spec.name}"]) for spec in inputs}
            refs = {name: as_array(archive[f"output/{name}"]) for name in tolerances}
        for spec in inputs:
            value = arrays[spec.name]
            if value.shape != spec.shape or str(value.dtype) != spec.dtype:
                raise ValueError(f"Fixture does not match receipt input: {spec.name}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite input: {spec.name}")
        if any(value.dtype != np.float32 or not np.isfinite(value).all()
               for value in refs.values()):
            raise ValueError("Reference outputs must be finite FP32 tensors")
        return cls(stem, fixture, tuple(inputs), arrays, refs, tolerances, build_profile,
                   tuple(requirements))

    def compare(self, backend: str, outputs: dict[str, Array]) -> tuple[OutputComparison, ...]:
        """Recheck completed outputs against the recipe tolerance and build PSNR policy."""
        comparisons: list[OutputComparison] = []
        for name, reference in self.references.items():
            if outputs[name].dtype != reference.dtype:
                raise ValueError(f"Output precision differs: {name}")
            comparison = compare_output(
                backend, name, outputs[name], reference, self.tolerances[name],
                next((value for value in self.output_requirements if value.output == name), None),
            )
            comparisons.append(replace(comparison, verdict=native_comparison_verdict(
                comparison, self.build_profile)))
        return tuple(comparisons)


@dataclass(frozen=True)
class Timing:
    prepare_s: float
    execute_s: float
    materialize_s: float
    total_s: float


@dataclass(frozen=True)
class Call:
    timing: Timing
    outputs: dict[str, Array]


class Session(Protocol):
    @property
    def model(self) -> object: ...

    async def infer(self) -> Call: ...


@dataclass(frozen=True)
class CoreMLSession:
    model: object
    workload: Workload

    async def infer(self) -> Call:
        started = perf_counter()
        result = function(self.model, "predict")(self.workload.arrays)
        executed = perf_counter()
        outputs = {name: as_array(item(result, name)) for name in self.workload.references}
        finished = perf_counter()
        return Call(Timing(0, executed - started, finished - executed, finished - started),
                    outputs)


@dataclass(frozen=True)
class CoreAISession:
    model: object
    workload: Workload
    invoke: Callable[[dict[str, object]], Awaitable[object]]
    array_constructor: Callable[..., object]
    backing: object
    prepared: dict[str, object] | None

    async def infer(self) -> Call:
        started = perf_counter()
        inputs = self.prepared
        if inputs is None:
            inputs = {name: self.array_constructor(value, backing=self.backing)
                      for name, value in self.workload.arrays.items()}
        prepared = perf_counter()
        result = await self.invoke(inputs)
        executed = perf_counter()
        outputs = {name: as_array(function(item(result, name), "numpy")())
                   for name in self.workload.references}
        finished = perf_counter()
        return Call(Timing(prepared - started, executed - prepared,
                           finished - executed, finished - started), outputs)


def load_coreml(workload: Workload, profile: Profile) -> CoreMLSession:
    ct = library("coremltools")
    units = attribute(ct, "ComputeUnit")
    unit = profile.coreml_compute_unit
    hints = None
    if profile == Profile.COREML_FAST:
        hints = {"specializationStrategy": attribute(
            attribute(ct, "SpecializationStrategy"), "FastPrediction",
        )}
    model = function(attribute(ct, "models"), "MLModel")(
        str(workload.stem.with_suffix(".mlpackage")), compute_units=attribute(units, unit),
        optimization_hints=hints,
    )
    return CoreMLSession(model, workload)


async def load_coreai(workload: Workload, profile: Profile) -> tuple[CoreAISession, float]:
    if "USE_LOCAL_COREAI" in os.environ:
        raise RuntimeError("Unset USE_LOCAL_COREAI; this benchmark requires OS Core AI")
    runtime = library("coreai.runtime")
    options_class = attribute(runtime, "SpecializationOptions")
    if function(options_class, "is_supported")() is not True:
        raise RuntimeError("OS Core AI required; set USE_OS_COREAI=1, unset USE_LOCAL_COREAI")
    options = function(options_class, "default")()
    if profile in (Profile.COREAI_GPU, Profile.COREAI_NE):
        name = "gpu" if profile == Profile.COREAI_GPU else "neural_engine"
        kind = function(attribute(runtime, "ComputeUnitKind"), name)()
        options = function(options_class, "from_preferred_compute_unit_kind")(kind)
    load = cast(Callable[..., Awaitable[object]], function(attribute(runtime, "AIModel"),
                                                        "load"))
    model = await load(workload.stem.with_suffix(".aimodel"), specialization_options=options)
    invoke = cast(Callable[[dict[str, object]], Awaitable[object]], function(
        function(model, "load_function")("main"), "__call__",
    ))
    backing_name = {Profile.COREAI_REUSE_METAL: "METAL",
                    Profile.COREAI_REUSE_IOSURFACE: "IO_SURFACE"}.get(profile, "BYTES")
    backing = attribute(attribute(runtime, "StorageKind"), backing_name)
    constructor = function(runtime, "NDArray")
    started = perf_counter()
    prepared = ({name: constructor(value, backing=backing)
                 for name, value in workload.arrays.items()} if profile.reuses_inputs else None)
    prepare_s = perf_counter() - started
    return CoreAISession(model, workload, invoke, constructor, backing, prepared), prepare_s


class CallPhase(StrEnum):
    FIRST = "first"
    WARMUP = "warmup"
    TIMED = "timed"
    INSPECTION = "inspection"


@dataclass(frozen=True)
class CallCheck:
    """One completed call's timing and source comparison, retained even on failure."""

    phase: CallPhase
    iteration: int
    timing: Timing
    comparisons: tuple[OutputComparison, ...]
    verdict: Verdict
    error: str | None = None


def _check_call(workload: Workload, backend: str, phase: CallPhase,
                iteration: int, call: Call) -> CallCheck:
    """Adjudicate host outputs after the session has stopped its inference timer."""
    try:
        comparisons = workload.compare(backend, call.outputs)
    except (ValueError, KeyError) as error:
        return CallCheck(phase, iteration, call.timing, (), Verdict.FAIL, str(error))
    verdict = (Verdict.PASS if comparisons and all(out.verdict == Verdict.PASS
                                                  for out in comparisons) else Verdict.FAIL)
    return CallCheck(phase, iteration, call.timing, comparisons, verdict)


@dataclass(frozen=True)
class Run:
    schema_version: int
    recorded_at: str
    model: str
    fixture: str
    profile: Profile
    load_s: float
    reused_input_setup_s: float
    requested_warmup: int
    requested_runs: int
    calls: tuple[CallCheck, ...]
    verdict: Verdict
    environment: dict[str, str]
    process_peak_rss_bytes: int
    build_profile: BuildProfile = BuildProfile.FP32
    scope: str = "Batch-one; host input to completed host output; no image pipeline"
    cache_state: str = "Fresh process; OS caches retained; no cache-hit claim"
    inspection: str = "Performed after timings when requested; never used to rank latency"
    implementation: tuple[SourceRevision, ...] = ()


async def measure(workload: Workload, profile: Profile, runs: int, warmup: int, *,
                  mlx_root: Path | None = None) -> tuple[Run, Session]:
    """Check every first, warm-up and timed call; stop at the first failed decision.

    Comparisons run between inference intervals and are excluded from their
    timings. Failed calls retain their phase, one-based iteration and evidence;
    later successes can never replace them in a passing run.
    """
    if runs < 1 or warmup < 0:
        raise ValueError("Require runs >= 1 and warmup >= 0")
    load_average = os.getloadavg()
    started = perf_counter()
    reused_input_setup = 0.0
    implementation: tuple[SourceRevision, ...] = ()
    if profile.backend == "coreml":
        session: Session = load_coreml(workload, profile)
    elif profile.backend == "mlx":
        from medmlx_convert.mlx_runtime import load_mlx

        if mlx_root is None:
            raise ValueError("MLX requires an explicit --mlx-root checkout directory")
        mlx_session = load_mlx(workload, mlx_root)
        session = mlx_session
        implementation = mlx_session.implementation
    else:
        session, reused_input_setup = await load_coreai(workload, profile)
    load_s = perf_counter() - started - reused_input_setup
    calls: list[CallCheck] = []
    for phase, count in ((CallPhase.FIRST, 1), (CallPhase.WARMUP, warmup),
                         (CallPhase.TIMED, runs)):
        for iteration in range(1, count + 1):
            call = await session.infer()
            checked = _check_call(workload, profile.backend, phase, iteration, call)
            calls.append(checked)
            del call
            if checked.verdict != Verdict.PASS:
                break
        if calls[-1].verdict != Verdict.PASS:
            break
    report = Run(
        RUNTIME_RUN_SCHEMA_VERSION, datetime.now(UTC).isoformat(),
        str(workload.stem), str(workload.fixture), profile,
        load_s, reused_input_setup,
        warmup, runs, tuple(calls), calls[-1].verdict,
        {"python": platform.python_version(), "macos": platform.mac_ver()[0],
         "machine": platform.machine(), "numpy": version("numpy"),
         "load_average_start": str(load_average), "load_average_end": str(os.getloadavg()),
         "coreai-core": version("coreai-core"), "coreai-torch": version("coreai-torch"),
         "coremltools": version("coremltools")},
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        build_profile=workload.build_profile,
        scope=("Repeated identical prepared input to completed host output; setup excluded"
               if profile.reuses_inputs else
               "Batch-one; host input to completed host output; no image pipeline"),
        implementation=implementation,
    )
    if profile.backend == "mlx":
        report = replace(report, environment={**report.environment, "mlx": version("mlx"),
                                              "mlx-metal": version("mlx-metal")})
    return report, session


@dataclass(frozen=True)
class ProfileEvent:
    timestamp: int
    event_id: str
    phase: str
    data: str
    kind: str
    interval: int | None


@dataclass(frozen=True)
class PlannedOperation:
    operator: str
    anticipated_device: str | None
    estimated_cost_weight: float | None


async def inspect_coreai(session: Session, workload: Workload
                         ) -> tuple[list[ProfileEvent], CallCheck]:
    """Profile a separate inference and retain its own source-parity decision."""
    if not isinstance(session, CoreAISession):
        raise TypeError("Core AI inspection requires a Core AI session")
    events: list[ProfileEvent] = []

    def record(event: object, kind: str, interval: int | None = None) -> None:
        events.append(ProfileEvent(integer(attribute(event, "timestamp")),
                                   text(attribute(event, "event_id")),
                                   text(attribute(event, "phase")), text(attribute(event, "data")),
                                   kind, interval))

    def on_event(event: object) -> None:
        record(event, "event")

    def on_begin(event: object) -> int:
        interval = len(events)
        record(event, "begin", interval)
        return interval

    def on_end(event: object, interval: int) -> None:
        record(event, "end", interval)

    profiler = function(library("coreai.runtime"), "Profiler")(
        on_log_event=on_event, on_log_event_begin=on_begin, on_log_event_end=on_end,
    )
    profiled = function(session.model, "load_function")("main", profiler=profiler)
    invoke = cast(Callable[[dict[str, object]], Awaitable[object]],
                  function(profiled, "__call__"))
    call = await replace(session, invoke=invoke).infer()
    return events, _check_call(workload, "coreai", CallPhase.INSPECTION, 1, call)


def inspect_coreml(session: Session, profile: Profile) -> list[PlannedOperation]:
    ct = library("coremltools")
    plan_class = attribute(attribute(attribute(ct, "models"), "compute_plan"), "MLComputePlan")
    unit_name = profile.coreml_compute_unit
    plan = function(plan_class, "load_from_path")(
        function(session.model, "get_compiled_model_path")(),
        compute_units=attribute(attribute(ct, "ComputeUnit"), unit_name),
    )
    program = attribute(attribute(plan, "model_structure"), "program")
    if program is None:
        raise ValueError("Expected an ML Program compute plan")
    main = item(attribute(program, "functions"), "main")
    operations = cast(list[object], attribute(attribute(main, "block"), "operations"))
    rows: list[PlannedOperation] = []
    for operation in operations:
        usage = function(plan, "get_compute_device_usage_for_mlprogram_operation")(operation)
        cost = function(plan, "get_estimated_cost_for_mlprogram_operation")(operation)
        device = (None if usage is None
                  else type(attribute(usage, "preferred_compute_device")).__name__)
        weight = None if cost is None else number(attribute(cost, "weight"))
        rows.append(PlannedOperation(text(attribute(operation, "operator_name")), device, weight))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", type=Path)
    parser.add_argument("fixture", type=Path, help="NPZ: input/<name> and output/<name> arrays")
    parser.add_argument("profile", type=Profile, choices=list(Profile))
    parser.add_argument("out", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--mlx-root", type=Path, help="Explicit sibling MLX source checkout root")
    parser.add_argument("--inspect", action="store_true",
                        help="Collect a separate diagnostic call/plan")
    args = parser.parse_args()
    if args.out.is_absolute():
        parser.error("Require a repository-relative output path")
    if args.profile.backend == "mlx" and args.inspect:
        parser.error("MLX does not expose the native Core AI profiler/Core ML compute-plan API")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Admission/runtime errors must not leave an old passing run at this path.
    args.out.unlink(missing_ok=True)
    args.out.with_suffix(".inspection.json").unlink(missing_ok=True)
    workload = Workload.load(args.stem, args.fixture, args.profile.backend)

    async def run() -> Run:
        report, session = await measure(workload, args.profile, args.runs, args.warmup,
                                        mlx_root=args.mlx_root)
        if args.inspect and report.verdict == Verdict.PASS:
            check = None
            if args.profile.backend == "coreai":
                diagnostic, check = await inspect_coreai(session, workload)
            else:
                diagnostic = inspect_coreml(session, args.profile)
            args.out.with_suffix(".inspection.json").write_text(json.dumps(
                {"schema_version": 2, "profile": args.profile, "data": [
                    asdict(value) for value in diagnostic],
                 "call": None if check is None else asdict(check)}, indent=2, allow_nan=False,
            ) + "\n")
            if check is not None:
                report = replace(report, calls=(*report.calls, check), verdict=check.verdict)
        args.out.write_text(json.dumps(asdict(report), indent=2, allow_nan=False) + "\n")
        return report

    report = asyncio.run(run())
    print(json.dumps({"profile": report.profile, "model": report.model,
                      "verdict": report.verdict}), flush=True)
    if report.verdict != Verdict.PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
