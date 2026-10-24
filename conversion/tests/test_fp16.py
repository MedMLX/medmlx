"""FP16 admission must preserve FP32 diagnostics and reject misleading label parity."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from medmlx_convert import export_and_check
from medmlx_convert.boundary import Module, function
from medmlx_convert.evidence import LabelMode, LabelRequirement, Verdict, compare_output
from medmlx_convert.guidance import BuildProfile, graph_stages, native_comparison_verdict
from medmlx_convert.precision import SourceDevice
from medmlx_convert.provenance import BuildSource
from medmlx_convert.runtime_benchmark import Profile, Workload


@given(output=st.text(min_size=1, max_size=20), mode=st.sampled_from(list(LabelMode)),
       minimum=st.floats(0, 1, allow_nan=False, allow_infinity=False))
def test_label_requirement_roundtrip(output: str, mode: LabelMode, minimum: float) -> None:
    requirement = LabelRequirement(output, mode, minimum)
    assert LabelRequirement.parse(json.loads(json.dumps(asdict(requirement)))) == requirement
    invalid_values: tuple[object, ...] = (
        None, [], {**asdict(requirement), "mode": "unknown"},
        {**asdict(requirement), "minimum_agreement": float("nan")},
    )
    for invalid in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            LabelRequirement.parse(invalid)


def test_fp16_gate_preserves_absolute_failure_and_rejects_identical_bad_labels() -> None:
    reference = np.asarray([[-2, -1, -3]], dtype=np.float32).reshape(1, 3, 1, 1)
    near = compare_output("coreai", "logits", reference + np.float32(0.001), reference, 1e-5)
    assert near.absolute_verdict == Verdict.FAIL
    assert native_comparison_verdict(near, BuildProfile.FP32) == Verdict.FAIL
    assert native_comparison_verdict(near, BuildProfile.FP16) == Verdict.PASS
    misleading = compare_output("coreai", "logits", reference * np.float32(1000), reference, 1e-5)
    assert misleading.label_agreement == 1
    assert native_comparison_verdict(misleading, BuildProfile.FP16) == Verdict.FAIL
    zero = compare_output("coreai", "score", np.ones(2, dtype=np.float32) * np.float32(1e-12),
                          np.zeros(2, dtype=np.float32), 1e-5)
    assert native_comparison_verdict(zero, BuildProfile.FP16) == Verdict.FAIL
    ambiguous = np.asarray([[0.9999, 1.0001]], dtype=np.float32).reshape(1, 2, 1, 1)
    labels = compare_output("coreml", "logits", ambiguous[:, ::-1].copy(), ambiguous, 1,
                            LabelRequirement("logits", LabelMode.MULTICLASS, 0.999))
    assert labels.psnr_db > 50 and labels.label_agreement == 0
    assert native_comparison_verdict(labels, BuildProfile.FP16) == Verdict.FAIL


class TinyLungNetwork(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1)
        self.norm = torch.nn.BatchNorm2d(3)
        self.up = torch.nn.ConvTranspose2d(3, 3, 1)
        self.head = torch.nn.LogSoftmax(dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.up(self.norm(self.conv(image))))


def test_fp16_mps_3d_compatibility_runs_both_saved_native_formats() -> None:
    """Prevent unsupported MPS 3D ops from reappearing in either backend's source."""
    function(torch, "manual_seed")(7)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 2, 1), torch.nn.MaxPool3d(2),
        torch.nn.AvgPool3d(2), torch.nn.ConvTranspose3d(2, 3, 2, stride=2),
        torch.nn.LogSoftmax(dim=1),
    ).eval()
    image = torch.rand((1, 1, 8, 8, 8), generator=torch.Generator().manual_seed(8))
    stem = Path("build/fp16_roundtrip/volume")
    stem.parent.mkdir(parents=True, exist_ok=True)
    assert export_and_check(
        model, {"image": image}, ["logits"], stem, tolerance=1e-4,
        source=BuildSource.capture(Path(__file__), synthetic=True),
        profile=BuildProfile.FP16, source_device=SourceDevice.MPS,
    )


def test_fp16_saved_artifacts_and_runtime_use_the_admitted_precision_policy() -> None:
    """Exercise both real converters/runtimes, then reapply FP16 gates during replay."""
    function(torch, "manual_seed")(4)
    model = TinyLungNetwork().eval()
    image = torch.rand((1, 1, 16, 16), generator=torch.Generator().manual_seed(5))
    stem = Path("build/fp16_roundtrip/network")
    stem.parent.mkdir(parents=True, exist_ok=True)
    assert export_and_check(
        model, {"image": image}, ["logits"], stem, tolerance=1e-5,
        source=BuildSource.capture(Path(__file__), synthetic=True),
        profile=BuildProfile.FP16, source_device=SourceDevice.MPS,
        coreml_profile=Profile.COREML_GPU,
        output_requirements=(LabelRequirement("logits", LabelMode.MULTICLASS, 0.999),),
    )
    assert model.conv.weight.dtype == torch.float32 and model.norm.weight.dtype == torch.float32
    numerical = json.loads(stem.with_suffix(".verification.json").read_text())
    assert numerical["build_profile"] == BuildProfile.FP16
    assert numerical["compute_policy"] == "Core ML CPU_AND_GPU; Core AI default specialization"
    assert all(out["absolute_verdict"] == "fail" for out in numerical["comparisons"])
    for backend in ("coreai", "coreml"):
        for stage in graph_stages(backend):
            receipt = stem.with_name(f"{stem.name}_{backend}_{stage}")
            metadata = json.loads(receipt.with_suffix(".verification.json").read_text())
            assert metadata["reference"] == f"PyTorch {BuildProfile.FP16} eager on mps"
        workload = Workload.load(stem, stem.with_suffix(".runtime.npz"), backend)
        assert workload.build_profile == BuildProfile.FP16
        reference = workload.references["logits"].astype(np.float32)
        near = workload.compare(backend, {"logits": reference + np.float32(0.001)})[0]
        assert near.verdict == Verdict.PASS and near.absolute_verdict == Verdict.FAIL
        assert workload.compare(backend, {"logits": reference * np.float32(1000)})[0].verdict \
            == Verdict.FAIL
        with pytest.raises(ValueError, match="precision differs"):
            workload.compare(backend, {"logits": reference.astype(np.float16)})
    with pytest.raises(ValueError, match="oracle requires CPU"):
        export_and_check(
            model.to("mps"), {"image": image.to("mps")}, ["logits"],
            stem.parent / "mislabeled_oracle", tolerance=1e-5,
            source=BuildSource.capture(Path(__file__), synthetic=True),
        )
    with pytest.raises(ValueError, match="oracle requires FP32"):
        export_and_check(
            TinyLungNetwork().half(), {"image": image}, ["logits"],
            stem.parent / "rounded_oracle", tolerance=1e-5,
            source=BuildSource.capture(Path(__file__), synthetic=True),
        )
