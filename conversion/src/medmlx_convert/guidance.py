"""Numerical gates for FP32 reference and mixed FP16 network builds."""

from enum import StrEnum

import numpy as np

from medmlx_convert.boundary import Array
from medmlx_convert.evidence import OutputComparison, Verdict


class BuildProfile(StrEnum):
    """Build precision: the FP32 reference, or mixed FP16 network (FP32 public I/O)."""

    FP32 = "macos-fp32-reference"
    FP16 = "macos-fp16-network"


class GraphStage(StrEnum):
    """Graph transformation boundary that is replayed against the source model."""

    CAPTURE = "capture"
    DECOMPOSITION = "decomposition"
    GRAPH = "graph"


def graph_stages(backend: str) -> tuple[GraphStage, ...]:
    """Declare graph transformation boundaries in each backend's converter order."""
    if backend == "coreai":
        return tuple(GraphStage)
    if backend == "coreml":
        return (GraphStage.CAPTURE, GraphStage.DECOMPOSITION)
    raise ValueError(f"Unknown graph backend: {backend}")


def runtime_inputs(inputs: dict[str, Array]) -> dict[str, Array]:
    """Admit identical runtime values for both backends without modifying inputs.

    Return C-contiguous FP32/int32 arrays. Int64 inputs must fit exactly in int32;
    reject scalar, empty, nonfinite and unsupported-dtype arrays with ValueError.
    """
    if not inputs:
        raise ValueError("A build requires named inputs")
    arrays: dict[str, Array] = {}
    for name, value in inputs.items():
        if not name or not value.shape or value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"Invalid or nonfinite build input: {name!r}")
        if value.dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("int64")):
            raise ValueError(f"{name}: FP32/int32 inputs required; found {value.dtype}")
        if value.dtype.kind == "i":
            limits = np.iinfo(np.int32)
            if value.min() < limits.min or value.max() > limits.max:
                raise ValueError(f"{name}: input cannot be represented exactly as int32")
            value = value.astype(np.int32)
        arrays[name] = np.ascontiguousarray(value)
    return arrays


def transformation_comparison_verdict(
    out: OutputComparison, profile: BuildProfile = BuildProfile.FP32, *, graph: bool = False,
) -> Verdict:
    """Require >70 dB normalized PSNR; rewrites and FP32 graphs retain absolute limits."""
    absolute_required = profile == BuildProfile.FP32 or not graph
    if ((absolute_required and out.absolute_verdict != Verdict.PASS)
            or (out.max_abs_diff != 0 and (out.normalized_psnr_db is None
                or out.normalized_psnr_db <= 70))):
        return Verdict.FAIL
    return Verdict.PASS


def require_transform_parity(
    comparisons: tuple[OutputComparison, ...], profile: BuildProfile = BuildProfile.FP32,
    *, graph: bool = False,
) -> None:
    """Reject inaccurate rewrites or graph stages against their independent eager oracle."""
    failed = [out for out in comparisons
              if transformation_comparison_verdict(out, profile, graph=graph) != Verdict.PASS]
    if not comparisons or failed:
        raise ValueError(f"Source transformation differs from the source: {failed}")


def native_comparison_verdict(comparison: OutputComparison,
                              profile: BuildProfile = BuildProfile.FP32) -> Verdict:
    """Apply absolute/PSNR and declared label gates to one completed source comparison.

    FP32 requires its original absolute tolerance and normalized PSNR >=60 dB;
    FP16 retains absolute errors as diagnostics and requires raw PSNR >50 dB.
    Exact tensors pass the PSNR check; a nonexact zero reference fails both profiles.
    """
    requirement = comparison.label_requirement
    if requirement is not None and (comparison.label_agreement is None
                                    or comparison.label_agreement < requirement.minimum_agreement):
        return Verdict.FAIL
    if profile == BuildProfile.FP32 and comparison.absolute_verdict != Verdict.PASS:
        return Verdict.FAIL
    if comparison.max_abs_diff != 0:
        if comparison.reference_max == 0:
            return Verdict.FAIL
        if profile == BuildProfile.FP16:
            if comparison.psnr_db <= 50:
                return Verdict.FAIL
        elif comparison.normalized_psnr_db is None or comparison.normalized_psnr_db < 60:
            return Verdict.FAIL
    return Verdict.PASS
