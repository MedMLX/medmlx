"""Load passing numerical builds and replay their exact source tensors."""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from medmlx_convert.boundary import Array
from medmlx_convert.evidence import (
    Verdict,
    compare_output,
    write_evidence,
    write_runtime_fixture,
)
from medmlx_convert.guidance import native_comparison_verdict
from medmlx_convert.runtime_benchmark import Workload


# Fixture persistence performs disk I/O; this property has no latency contract.
@settings(deadline=None)
@given(width=st.integers(1, 8), labels=st.integers(2, 5),
       value=st.floats(-10, 10, allow_nan=False, allow_infinity=False, width=32))
def test_fixture_roundtrip_and_reject_incompatible_input(
    width: int, labels: int, value: float,
) -> None:
    Path("build").mkdir(exist_ok=True)
    with TemporaryDirectory(dir="build") as directory:
        stem = Path(directory).relative_to(Path.cwd()) / "network"
        image = np.full((1, labels, width, width), value, dtype=np.float32)
        classes = np.arange(labels, dtype=np.int64).reshape(-1, 1)
        inputs: dict[str, Array] = {"image": image, "class_vector": classes}
        decision = compare_output("coreai", "logits", image, image, 0.001)
        comparisons = [decision, replace(decision, framework="coreml")]
        write_evidence(stem, inputs, comparisons)
        write_runtime_fixture(stem, inputs, {"logits": image})
        fixture = stem.with_suffix(".runtime.npz")
        parsed = Workload.load(stem, fixture, "coreai")
        np.testing.assert_array_equal(parsed.arrays["image"], image)
        np.testing.assert_array_equal(parsed.references["logits"], image)
        assert parsed.arrays["class_vector"].dtype == np.int32
        assert parsed.compare("coreai", {"logits": image}) == (decision,)
        write_runtime_fixture(stem, {**inputs, "image": image.astype(np.float64)},
                              {"logits": image})
        with pytest.raises(ValueError, match="does not match receipt input"):
            Workload.load(stem, fixture, "coreai")
        write_runtime_fixture(stem, {**inputs, "image": image.reshape(-1)}, {"logits": image})
        with pytest.raises(ValueError, match="does not match receipt input"):
            Workload.load(stem, fixture, "coreai")


@pytest.mark.parametrize("backend", ["coreai", "coreml"])
def test_benchmark_rejects_failed_missing_or_incompatible_export(backend: str) -> None:
    Path("build").mkdir(exist_ok=True)
    with TemporaryDirectory(dir="build") as directory:
        stem = Path(directory).relative_to(Path.cwd()) / "network"
        reference = np.ones(8, dtype=np.float32)
        inputs: dict[str, Array] = {"image": reference}
        exact = [compare_output(name, "features", reference, reference, 0.1)
                 for name in ("coreai", "coreml")]
        fixture = stem.with_suffix(".runtime.npz")
        receipt = stem.with_suffix(".verification.json")
        write_runtime_fixture(stem, inputs, {"features": reference})
        low_psnr = [compare_output(name, "features", reference + np.float32(0.01), reference, 0.1)
                    for name in ("coreai", "coreml")]
        assert all(out.verdict == Verdict.PASS for out in low_psnr)
        for failed_backend in ("coreai", "coreml"):
            write_evidence(stem, inputs, [
                replace(out, verdict=native_comparison_verdict(out))
                if out.framework == failed_backend else passing
                for out, passing in zip(low_psnr, exact, strict=True)
            ])
            with pytest.raises(ValueError, match="failed export"):
                Workload.load(stem, fixture, backend)
        receipt.unlink()
        with pytest.raises(FileNotFoundError):
            Workload.load(stem, fixture, backend)
        write_evidence(stem, inputs, exact)
        workload = Workload.load(stem, fixture, backend)
        assert all(out.verdict == Verdict.PASS
                   for out in workload.compare(backend, {"features": reference}))
        # A passing build does not admit a later profile's low-PSNR outputs.
        assert all(out.verdict == Verdict.FAIL for out in workload.compare(
            backend, {"features": reference + np.float32(0.01)},
        ))
        passed = json.loads(receipt.read_text())
        for invalid in ({**passed, "schema_version": passed["schema_version"] + 1},
                        {**passed, "model": "other"}, {**passed, "build_profile": "other"}):
            receipt.write_text(json.dumps(invalid))
            with pytest.raises(ValueError, match="schema|does not match|other"):
                Workload.load(stem, fixture, backend)
