"""Numerical gates preserve admitted values and scale-independent signal errors."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from medmlx_convert.evidence import Verdict, compare_output
from medmlx_convert.guidance import native_comparison_verdict, runtime_inputs


@given(values=st.lists(st.integers(-(2**31), 2**31 - 1), min_size=1, max_size=20))
def test_runtime_integer_admission_is_exact(values: list[int]) -> None:
    original = np.asarray(values, dtype=np.int64)[::-1]
    actual = runtime_inputs({"tokens": original})["tokens"]
    np.testing.assert_array_equal(actual, original)
    assert actual.dtype == np.int32 and actual.flags.c_contiguous


@given(value=st.one_of(st.integers(2**31, 2**63 - 1),
                       st.integers(-(2**63), -(2**31) - 1)))
def test_runtime_integer_admission_rejects_loss(value: int) -> None:
    with pytest.raises(ValueError, match="represented exactly"):
        runtime_inputs({"tokens": np.asarray([value], dtype=np.int64)})


@given(width=st.integers(1, 32), noise=st.floats(0.001, 0.1, allow_nan=False))
def test_psnr_does_not_depend_on_replicating_the_same_signal(width: int, noise: float) -> None:
    reference = np.ones((1, width), dtype=np.float32)
    actual = reference + np.float32(noise)
    replicated = compare_output("coreai", "features", actual, reference, 1)
    scalar = compare_output("coreai", "features", actual[:, :1], reference[:, :1], 1)
    assert replicated.psnr_db == pytest.approx(scalar.psnr_db)


@given(scale=st.floats(1e-20, 1e-7, allow_nan=False, allow_subnormal=False))
def test_normalized_psnr_accepts_accurate_small_signals(scale: float) -> None:
    reference = np.full(4, scale, dtype=np.float32)
    actual = reference * np.float32(1.00001)
    outputs = [compare_output(backend, "score", actual, reference, float(scale) * 0.001)
               for backend in ("coreai", "coreml")]
    assert outputs[0].psnr_db < 60
    assert all(output.normalized_psnr_db is not None and output.normalized_psnr_db > 90
               for output in outputs)
    assert all(native_comparison_verdict(output) == Verdict.PASS for output in outputs)
