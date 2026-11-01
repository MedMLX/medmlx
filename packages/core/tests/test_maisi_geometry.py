"""Admission checks for the MAISI mask geometry contract."""

import pytest

from medmlx_core.maisi import validate_maisi_geometry


def test_geometry_rejects_nan_in_y_spacing() -> None:
    with pytest.raises(ValueError):
        validate_maisi_geometry((256, 256, 128), (1.0, float("nan"), 1.0))
