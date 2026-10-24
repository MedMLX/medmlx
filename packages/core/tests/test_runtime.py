import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fixture_cases import darwin_contracts

from medmlx_core import errors, runtime


def test_simulated_darwin_arm64_contract() -> None:
    contract = darwin_contracts(runtime)
    expected = {
        "available": True,
        "mlx_version": "0.32.4",
        "macos_version": "15.0",
        "apple_chip": "Apple M3",
        "memory_bytes": 17179869184,
        "python_version": "3.12.13",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "reason": None,
        "hint": None,
    }
    assert contract["probe"] == contract["required"] == contract["device"] == expected
    assert contract["selected"] == ["gpu"]
    assert contract["device_name"] == "Device(gpu, 0)"
    assert contract["invalid_device"]["kind"] == "error"
    assert contract["invalid_device"]["type"] == "MissingDependencyError"
    assert "explicit mlx device" in contract["invalid_device"]["message"]
    assert not contract["missing_metal"]["available"]
    assert contract["missing_metal"]["reason"] is not None
    assert "refusing a CPU or host fallback" in contract["missing_metal"]["reason"]
    assert contract["rejected"]["kind"] == "error"
    assert contract["rejected"]["type"] == "MissingDependencyError"


@pytest.mark.parametrize(
    ("system", "machine"), [("Darwin", "x86_64"), ("Linux", "x86_64"), ("Windows", "AMD64")]
)
def test_unsupported_hosts_are_rejected(system: str, machine: str) -> None:
    with (
        patch.object(runtime.platform, "system", return_value=system),
        patch.object(runtime.platform, "machine", return_value=machine),
        patch.object(runtime, "_load_mlx_core") as load_backend,
    ):
        report = runtime.probe_mlx_runtime()
        assert not report.available
        assert report.reason is not None
        assert f"{system} {machine}" in report.reason
        assert report.mlx_version is None
        with pytest.raises(errors.MissingDependencyError):
            runtime.require_mlx_runtime()
        load_backend.assert_not_called()


def test_missing_mlx_reports_unavailable() -> None:
    with patch.object(runtime, "_load_mlx_core", side_effect=ImportError("not installed")):
        report = runtime.probe_mlx_runtime()
        assert not report.available
        assert report.reason == "mlx is not installed"
        with pytest.raises(errors.MissingDependencyError, match="mlx is not installed"):
            runtime.import_mlx()


@pytest.mark.parametrize(
    "version", ["0.32.3", "0.32.2", "0.31.4", "0.33.0", "0.32.3.dev1", "unknown"]
)
def test_unsupported_mlx_versions_fail_before_backend_execution(version: str) -> None:
    backend = SimpleNamespace(__version__=version)
    with (
        patch.object(runtime.platform, "system", return_value="Darwin"),
        patch.object(runtime.platform, "machine", return_value="arm64"),
        patch.object(runtime, "_load_mlx_core", return_value=backend),
        patch.object(runtime, "_metal_available") as metal,
    ):
        report = runtime.probe_mlx_runtime()
        assert not report.available
        assert report.mlx_version == version
        assert report.reason is not None
        assert "MLX >=0.32.4,<0.33 is required" in report.reason
        with pytest.raises(errors.MissingDependencyError, match="Upgrade MLX before inference"):
            runtime.import_mlx()
        metal.assert_not_called()


def test_report_serializes_without_backend_objects() -> None:
    backend = SimpleNamespace(
        __version__="0.32.4", metal=SimpleNamespace(is_available=lambda: True)
    )
    with patch.object(runtime, "_load_mlx_core", return_value=backend):
        report = runtime.probe_mlx_runtime()
    assert json.loads(json.dumps(report.to_payload())) == report.to_payload()
