"""One declared numerical policy for independent FP32 helper references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np
from fixture_cases import HostArray, JSONValue


@dataclass(frozen=True, slots=True)
class NumericalBudget:
    rtol: float
    atol: float
    reason: str

    def __post_init__(self) -> None:
        for value in (self.rtol, self.atol):
            if type(value) not in (int, float) or value < 0:
                raise ValueError("Numerical bounds must be finite nonnegative numbers")
            try:
                finite = isfinite(value)
            except OverflowError as exc:
                raise ValueError("Numerical bound exceeds floating-point range") from exc
            if not finite:
                raise ValueError("Numerical bounds must be finite nonnegative numbers")
        object.__setattr__(self, "rtol", float(self.rtol))
        object.__setattr__(self, "atol", float(self.atol))
        if not _has_reason(self.reason):
            raise ValueError("Numerical bounds require a reason")


def _has_reason(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate numerical contract field: {key}")
        fields[key] = value
    return fields


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid numerical JSON constant: {value}")


def parse_contract(value: str) -> dict[str, NumericalBudget]:
    payload: JSONValue = json.loads(
        value, object_pairs_hook=_unique_fields, parse_constant=_reject_constant
    )
    boundary = {"schema_version": 1, "dtype": "float32", "shape": "reference"}
    fields = {"schema_version", "dtype", "shape", "allow_nan", "allow_inf", "budgets"}
    if (
        not isinstance(payload, dict)
        or payload.keys() != fields
        or type(payload.get("schema_version")) is not int
    ):
        raise ValueError("Invalid numerical contract")
    if any(payload.get(key) != expected for key, expected in boundary.items()):
        raise ValueError("Unsupported numerical boundary")
    if payload.get("allow_nan") is not False or payload.get("allow_inf") is not False:
        raise ValueError("Numerical reference outputs must be finite")
    budgets = payload.get("budgets")
    if not isinstance(budgets, dict) or not budgets:
        raise ValueError("Numerical contract must declare its budgets")
    if any(
        not isinstance(rule, dict) or rule.keys() != {"rtol", "atol", "reason"}
        for rule in budgets.values()
    ):
        raise ValueError("Each numerical budget must declare bounds and a reason")
    parsed: dict[str, NumericalBudget] = {}
    for name, rule in budgets.items():
        assert isinstance(rule, dict)  # The complete rule shape was checked above.
        rtol, atol, reason = rule["rtol"], rule["atol"], rule["reason"]
        if (
            type(rtol) not in (int, float)
            or type(atol) not in (int, float)
            or not isinstance(reason, str)
        ):
            raise ValueError("Numerical bounds must be finite nonnegative numbers with a reason")
        parsed[name] = NumericalBudget(cast(float, rtol), cast(float, atol), reason)
    return parsed


BUDGETS = parse_contract((Path(__file__).parent / "numerical_contract.json").read_text())


def assert_reference(actual: HostArray, expected: HostArray, *, budget: str) -> None:
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
