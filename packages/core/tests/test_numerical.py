"""Numerical gates admit rounding but reject semantic errors and invalid outputs."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numerical import BUDGETS, assert_reference, helper_budget, parse_contract

CONTRACT = (Path(__file__).parent / "numerical_contract.json").read_text()


@pytest.mark.parametrize("budget", BUDGETS)
def test_numerical_gate_allows_rounding_but_rejects_wrong_values(budget: str) -> None:
    expected = np.array([-1, 0, 1], dtype=np.float32)
    actual = expected.copy()
    assert_reference(actual, expected, budget=budget)
    if budget != "exact":
        actual[-1] += np.finfo(np.float32).eps
        assert_reference(actual, expected, budget=budget)
    actual += 0.01
    with pytest.raises(AssertionError):
        assert_reference(actual, expected, budget=budget)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_rejected_even_when_both_arrays_match(invalid: float) -> None:
    array = np.array([invalid], dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_reference(array, array.copy(), budget="graph")


def test_dtype_shape_and_layout_errors_are_rejected() -> None:
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    for actual in (expected.astype(np.float16), expected.T, expected[:, ::-1]):
        with pytest.raises(AssertionError):
            assert_reference(actual, expected, budget="graph")


@given(st.floats(min_value=0, max_value=1e-4, allow_nan=False, allow_infinity=False))
@settings(database=None)
def test_numerical_contract_round_trip(bound: float) -> None:
    payload = json.loads(CONTRACT)
    payload["budgets"]["moments"]["atol"] = bound
    parsed = parse_contract(json.dumps(payload))
    payload["budgets"] = {name: asdict(budget) for name, budget in parsed.items()}
    assert parse_contract(json.dumps(payload)) == parsed


@pytest.mark.parametrize("bound", [True, -1, 10**400, float("nan"), float("inf"), "0", None])
def test_invalid_numerical_bounds_fail_fast(bound: object) -> None:
    payload = json.loads(CONTRACT)
    payload["budgets"]["moments"]["atol"] = bound
    with pytest.raises(ValueError):
        parse_contract(json.dumps(payload))


def test_duplicate_contract_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate numerical contract field"):
        parse_contract('{"schema_version": 1, "schema_version": 1}')


@pytest.mark.parametrize("payload", ["null", "[]", "true", "NaN", "Infinity"])
def test_invalid_contract_payloads_raise_value_error(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_contract(payload)


@given(st.none() | st.booleans() | st.integers() | st.text() | st.lists(st.integers()))
@settings(database=None)
def test_budget_parser_constructs_valid_objects_or_raises_value_error(value: object) -> None:
    payload = json.loads(CONTRACT)
    payload["budgets"]["moments"]["atol"] = value
    try:
        budgets = parse_contract(json.dumps(payload))
    except ValueError:
        return
    assert np.isfinite(budgets["moments"].atol)
    assert budgets["moments"].atol >= 0


def test_every_helper_has_an_explicit_budget() -> None:
    from fixture_cases import array_cases

    _, cases = array_cases()
    for module, records in cases.items():
        for case in records:
            if case.error is None:
                assert helper_budget(module, case.function) in BUDGETS
