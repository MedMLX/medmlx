"""Numerical export decisions and their durable, versioned evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path

import numpy as np

from medmlx_convert.boundary import Array, as_str_dict
from medmlx_convert.provenance import BuildSource

RUNTIME_FIXTURE_SCHEMA_VERSION = 1
EXPORT_EVIDENCE_SCHEMA_VERSION = 3


class Verdict(StrEnum):
    """Outcome of a numerical check against its declared tolerance."""

    PASS = "pass"
    FAIL = "fail"


class LabelMode(StrEnum):
    """How label agreement is computed between native and source outputs.

    ``multiclass_argmax`` compares the argmax over the class axis (axis 1).
    ``independent_logit_zero`` compares each element's decision ``logit > 0``.
    """

    MULTICLASS = "multiclass_argmax"
    INDEPENDENT = "independent_logit_zero"


@dataclass(frozen=True)
class LabelRequirement:
    """A recipe-owned engineering drift budget, not a clinical accuracy target."""

    output: str
    mode: LabelMode
    minimum_agreement: float

    def __post_init__(self) -> None:
        if not self.output or not np.isfinite(self.minimum_agreement) \
                or not 0 <= self.minimum_agreement <= 1:
            raise ValueError("A label requirement needs an output and agreement in [0, 1]")

    @classmethod
    def parse(cls, value: object) -> LabelRequirement:
        """Rebuild a requirement from its JSON mapping.

        The mapping needs exactly the keys ``output``, ``mode`` and ``minimum_agreement``.

        Raises:
            TypeError: A field has the wrong type.
            ValueError: The mode is unknown or the values fail validation.
        """
        raw = as_str_dict(value)
        output, mode, minimum = raw["output"], raw["mode"], raw["minimum_agreement"]
        if (not isinstance(output, str) or not isinstance(mode, str)
                or type(minimum) not in (float, int)):
            raise TypeError("Invalid label requirement")
        assert isinstance(minimum, float | int)
        return cls(output, LabelMode(mode), float(minimum))

    def agreement(self, actual: Array, reference: Array) -> float:
        """Compare NCHW/NCDHW class decisions or each independent zero-logit decision."""
        if self.mode == LabelMode.INDEPENDENT:
            return float(((actual.astype(np.float64) > 0)
                          == (reference.astype(np.float64) > 0)).mean())
        if reference.ndim < 4 or reference.shape[1] < 2:
            raise ValueError("Multiclass agreement requires NCHW/NCDHW with >=2 classes")
        return float((actual.argmax(1) == reference.argmax(1)).mean())


@dataclass(frozen=True)
class OutputComparison:
    """Result of comparing one native output with its source reference.

    ``max_abs_diff`` and ``reference_max`` are in the output's own units and
    ``tolerance`` is the absolute limit. ``psnr_db`` is Apple's metric and
    ``normalized_psnr_db`` is the peak-normalized one (None for an all-zero
    reference). Both ``verdict`` and ``absolute_verdict`` are set from
    ``max_abs_diff <= tolerance``; the PSNR values are recorded, not used here.
    """

    framework: str
    output: str
    shape: tuple[int, ...]
    dtype: str
    max_abs_diff: float
    reference_max: float
    tolerance: float
    label_agreement: float | None
    verdict: Verdict
    psnr_db: float
    normalized_psnr_db: float | None
    absolute_verdict: Verdict
    label_requirement: LabelRequirement | None = None


@dataclass(frozen=True)
class InputTensor:
    """Name, static shape and dtype of one exported input."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ExportEvidence:
    """Contents of a ``*.verification.json`` report.

    Records the inputs, per-output comparisons, installed package versions, the
    reference and compute policy, and ``scope``, which states what the report does
    not cover. ``source`` carries the checkpoint and revision provenance.
    """

    schema_version: int
    recorded_at: str
    model: str
    inputs: tuple[InputTensor, ...]
    comparisons: tuple[OutputComparison, ...]
    packages: dict[str, str]
    reference: str = "PyTorch CPU FP32"
    compute_policy: str = "Core ML ALL; Core AI default specialization"
    scope: str = "Fixed-shape network tensors; no host pipeline or clinical test"
    source: BuildSource | None = None
    build_profile: str = "macos-fp32-reference"


def compare_output(
    framework: str, name: str, got: Array, want: Array, tolerance: float,
    label_requirement: LabelRequirement | None = None,
) -> OutputComparison:
    """Compare aligned outputs without broadcasting or output dtype coercion.

    Maximum error uses the source units. Both PSNR values are in decibels and
    use mean squared error across all tensor elements. The normalized metric
    first divides both signals by the reference peak; it is undefined (None)
    for an all-zero reference. The unscaled Apple example metric is retained.
    Reject empty, shape- or dtype-mismatched, or nonfinite tensors before making
    either decision.

    Label agreement is computed for an output named ``logits`` with at least two
    classes (argmax over axis 1), or by ``label_requirement`` when one is given.

    Raises:
        ValueError: The tolerance is invalid, the tensors are empty, differ in shape
            or dtype, or contain NaN/Inf, or ``label_requirement`` names another output.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"{name}: tolerance must be finite and nonnegative")
    if got.shape != want.shape or got.size == 0:
        raise ValueError(
            f"{framework} {name}: shapes differ or are empty: {got.shape}, {want.shape}"
        )
    if not np.isfinite(got).all() or not np.isfinite(want).all():
        raise ValueError(f"{framework} {name}: output or reference contains NaN/Inf")
    if got.dtype != want.dtype:
        raise ValueError(f"{framework} {name}: output dtype differs: {got.dtype}, {want.dtype}")
    actual, reference = got.astype(np.float64), want.astype(np.float64)
    noise = actual - reference
    diff = float(np.abs(noise).max())
    scale = float(np.abs(reference).max())
    # Apple's Core AI test metric: peak reference energy / mean noise energy,
    # regularized with 1e-5 and 0.99e-10. FP64 avoids overflow in finite FP32 outputs.
    # https://github.com/apple/coreai-models/blob/bd61eb182a02182b3bd550521e7130c5ef2bbee4/python/tests/_runner_infra/common/utils/test.py
    psnr = float(10 * np.log10((scale * scale + 1e-5) / (np.mean(noise * noise) + 0.99e-10)))
    # Peak normalization avoids the example's 50 dB ceiling for near-zero signals.
    normalized_psnr = (None if scale == 0 else float(10 * np.log10(
        (1 + 1e-5) / (np.mean((noise / scale) ** 2) + 0.99e-10)
    )))
    agreement = None
    if name == "logits" and want.ndim >= 4 and want.shape[1] > 1:
        agreement = float((got.argmax(1) == want.argmax(1)).mean())
    if label_requirement is not None:
        if label_requirement.output != name:
            raise ValueError(f"Label requirement does not match output: {name}")
        agreement = label_requirement.agreement(got, want)
    return OutputComparison(
        framework, name, got.shape, str(got.dtype), diff, scale, tolerance, agreement,
        Verdict.PASS if diff <= tolerance else Verdict.FAIL, psnr, normalized_psnr,
        Verdict.PASS if diff <= tolerance else Verdict.FAIL,
        label_requirement,
    )


def write_evidence(
    out_stem: Path, inputs: dict[str, Array], comparisons: list[OutputComparison],
    *, reference: str = "PyTorch CPU FP32",
    compute_policy: str = "Core ML ALL; Core AI default specialization",
    source: BuildSource | None = None, build_profile: str = "macos-fp32-reference",
) -> None:
    """Persist numerical decisions and their execution context at the output stem."""
    evidence = ExportEvidence(
        schema_version=EXPORT_EVIDENCE_SCHEMA_VERSION,
        recorded_at=datetime.now(UTC).isoformat(),
        model=out_stem.name,
        inputs=tuple(InputTensor(name, a.shape, str(a.dtype)) for name, a in inputs.items()),
        comparisons=tuple(comparisons),
        reference=reference,
        compute_policy=compute_policy,
        source=source,
        build_profile=build_profile,
        packages={
            name: version(name)
            for name in ("torch", "coreai-core", "coreai-torch", "coremltools", "numpy")
        },
    )
    out_stem.with_suffix(".verification.json").write_text(
        json.dumps(asdict(evidence), indent=2, allow_nan=False) + "\n"
    )


def write_runtime_fixture(
    out_stem: Path, inputs: dict[str, Array], references: dict[str, Array],
) -> None:
    """Retain the exact synthetic inputs and source outputs for runtime comparisons."""
    values = {f"input/{name}": value if value.dtype.kind == "f" else value.astype(np.int32)
              for name, value in inputs.items()}
    values.update({f"output/{name}": value for name, value in references.items()})
    np.savez(out_stem.with_suffix(".runtime.npz"),
             schema_version=np.array(RUNTIME_FIXTURE_SCHEMA_VERSION),
             allow_pickle=False, **values)
