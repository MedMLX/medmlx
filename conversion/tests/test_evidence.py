"""Protect the parity decision against broadcasting, non-finite output and precision changes."""

import numpy as np
import pytest

from medmlx_convert.evidence import Verdict, compare_output


@pytest.mark.parametrize(
    ("got", "want"),
    [
        (np.zeros((1, 3)), np.zeros((2, 3))),
        (np.array([np.nan]), np.zeros(1)),
        (np.ones(2, np.float16), np.ones(2, np.float32)),
    ],
    ids=["broadcastable-shape", "nonfinite", "dtype"],
)
def test_parity_rejects_invalid_comparisons(got: np.ndarray, want: np.ndarray) -> None:
    with pytest.raises(ValueError):
        compare_output("coreai", "features", got, want, 0.1)


def test_error_gate_preserves_declared_tolerance() -> None:
    want = np.array([1.0, 2.0], dtype=np.float32)
    assert compare_output("coreai", "features", want, want, 0).verdict == Verdict.PASS
    got = want + np.float32(0.25)
    decision = compare_output("coreai", "features", got, want, 0.125)
    assert decision.verdict == Verdict.FAIL
    assert decision.max_abs_diff == 0.25
