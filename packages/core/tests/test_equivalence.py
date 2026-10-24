"""Independent upstream outputs for shared helpers, without reference frameworks at runtime."""

import importlib
from typing import cast

import numpy as np
import pytest
from fixture_cases import ArrayCase, HostArray, array_cases, load_fixture, run_array_case
from numerical import assert_reference, helper_budget

from medmlx_core.runtime import import_mlx
from medmlx_core.typing import MlxRuntime

mx: MlxRuntime = import_mlx()

_, CASES = array_cases()
ARRAY_CASES = [
    (name, case)
    for name, cases in CASES.items()
    for case in cases
    if case.error is None or case.function == "group_norm_ncdhw"
]


@pytest.mark.parametrize(
    ("name", "case"), ARRAY_CASES, ids=[f"{name}.{case.id}" for name, case in ARRAY_CASES]
)
def test_upstream_outputs(name: str, case: ArrayCase) -> None:
    metadata, arrays = load_fixture(name)
    module = importlib.import_module(f"medmlx_core.{name}")
    if case.error is not None:
        with pytest.raises(case.error) as captured:
            run_array_case(module, case, arrays, mx)
        assert str(captured.value) == case.error_message
        return
    record = next(record for record in metadata.cases if record.id == case.id)
    value = run_array_case(module, case, arrays, mx)
    if value is None:
        assert record.outputs == ()
        return
    mx.eval(value)
    values = value if isinstance(value, tuple) else (value,)
    assert len(values) == len(record.outputs)
    for actual, key in zip(values, record.outputs, strict=True):
        assert_reference(
            cast(HostArray, np.asarray(actual)),
            arrays[key],
            budget=helper_budget(name, case.function),
        )
        if case.function == "_moments" and key == record.outputs[1]:
            assert (np.asarray(actual) >= 0).all(), "Population variance must be nonnegative"
