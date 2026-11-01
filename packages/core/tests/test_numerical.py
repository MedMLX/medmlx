"""The shared numerical gate admits rounding but rejects wrong values and invalid outputs."""

import numpy as np
import pytest
from numerical import BUDGETS, assert_reference


@pytest.mark.parametrize("budget", BUDGETS)
def test_numerical_gate_allows_rounding_but_rejects_wrong_values(budget: str) -> None:
    expected = np.array([-1, 0, 1], dtype=np.float32)
    actual = expected.copy()
    assert_reference(actual, expected, budget=budget)
    if budget != "exact":
        actual[-1] += np.finfo(np.float32).eps
        assert_reference(actual, expected, budget=budget)
    with pytest.raises(AssertionError):
        assert_reference(actual + np.float32(0.01), expected, budget=budget)


def test_nonfinite_dtype_shape_and_layout_errors_are_rejected() -> None:
    nan = np.array([np.nan], dtype=np.float32)
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    for actual, reference in ((nan, nan.copy()), (expected.astype(np.float16), expected),
                              (expected.T, expected), (expected[:, ::-1], expected)):
        with pytest.raises(AssertionError):
            assert_reference(actual, reference, budget="graph")
