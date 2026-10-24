"""Protect the parity decision against broadcasting and non-finite output."""

import numpy as np
import pytest

from medmlx_convert.evidence import Verdict, compare_output


def test_parity_rejects_broadcastable_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compare_output("coreml", "logits", np.zeros((1, 3)), np.zeros((2, 3)), 1e-3)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_parity_rejects_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN/Inf"):
        compare_output("coreai", "logits", np.array([invalid]), np.zeros(1), 1e-3)


def test_error_gate_preserves_declared_tolerance() -> None:
    want = np.array([1.0, 2.0], dtype=np.float32)
    assert compare_output("coreai", "features", want, want, 0).verdict == Verdict.PASS
    got = want + np.float32(0.25)
    decision = compare_output("coreai", "features", got, want, 0.125)
    assert decision.verdict == Verdict.FAIL
    assert decision.max_abs_diff == 0.25


def test_parity_rejects_output_precision_change() -> None:
    with pytest.raises(ValueError, match="dtype differs"):
        compare_output("coreml", "features", np.ones(2, np.float16), np.ones(2, np.float32), 0.1)
