"""One declared numerical policy for independent FP32 helper references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fixture_cases import HostArray


@dataclass(frozen=True, slots=True)
class NumericalBudget:
    """Per-operation tolerances and the reason for them; zero for both means bit-exact."""

    rtol: float
    atol: float
    reason: str


_CONTRACT = json.loads((Path(__file__).parent / "numerical_contract.json").read_text())
BUDGETS = {
    name: NumericalBudget(float(rule["rtol"]), float(rule["atol"]), str(rule["reason"]))
    for name, rule in _CONTRACT["budgets"].items()
}


def assert_reference(actual: HostArray, expected: HostArray, *, budget: str) -> None:
    """Assert two float32 arrays of equal shape match under the named budget."""
    assert actual.dtype == expected.dtype == np.dtype(np.float32)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    rule = BUDGETS[budget]
    if rule.rtol == rule.atol == 0:
        np.testing.assert_array_equal(actual, expected)
        return
    # The explicit per-operation bound and its reason live in the versioned
    # numerical contract. Relative error alone is unstable around cancellation.
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rule.rtol,
        atol=rule.atol,
        equal_nan=False,
        strict=True,
        err_msg=f"{budget}: {rule.reason}",
    )


def helper_budget(module: str, function: str) -> str:
    """Name of the budget for a helper function.

    Layout and cast helpers are exact, contractions use ``dot``, the rest are
    chosen by operation family. Unmapped functions raise ``KeyError``.
    """
    if module == "layout" or function in {
        "eps",
        "as_fp32",
        "upsample_nearest_ncdhw",
        "pad_spatial_trailing_ncdhw",
        "concat_channels_ncdhw",
    }:
        return "exact"
    if function in {
        "linear",
        "conv3d_ncdhw",
        "conv_transpose3d_ncdhw",
        "split_conv3d_ncdhw",
        "split_conv_transpose3d_ncdhw",
        "deconv2x_ncdhw",
        "conv",
    }:
        return "dot"
    return {
        "silu": "pointwise",
        "group_norm_ncdhw": "normalization",
        "avg_pool3d_ncdhw": "normalization",
        "norm": "normalization",
        "_moments": "moments",
        "upsample_trilinear_ncdhw": "interpolation",
        "upsample_add_ncdhw": "interpolation",
    }[function]
