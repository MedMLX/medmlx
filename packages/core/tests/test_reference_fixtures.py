"""Seeded fixture coverage and independent reference dependency pins."""

import numpy as np
import pytest
from fixture_cases import (
    REFERENCE,
    array_cases,
    load_fixture,
)


@pytest.mark.parametrize("name", ["layout", "ops", "upsample", "precision"])
def test_recorded_helper_coverage_and_seeded_inputs(name: str) -> None:
    metadata, arrays = load_fixture(name)
    seeded, cases = array_cases()
    assert metadata.seed == 7081
    assert {record.id for record in metadata.cases} == {
        case.id for case in cases[name] if case.error is None
    }
    for key, value in seeded.items():
        np.testing.assert_array_equal(arrays[key], value)
    for record in metadata.cases:
        for key in record.outputs:
            assert key in arrays
            assert np.isfinite(arrays[key]).all()


def test_reference_pin_matches_development_dependencies() -> None:
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependencies = project["project"]["optional-dependencies"]["dev"]
    assert f"monai=={REFERENCE.monai_version}" in dependencies
    assert f"torch=={REFERENCE.torch_version}" in dependencies
