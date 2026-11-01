"""Host and device readiness contract, with Darwin arm64 simulated."""

import json
import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from medmlx_core import errors, runtime


@contextmanager
def _darwin_arm64(backend: SimpleNamespace) -> Generator[None]:
    """Simulate only platform, sysctl and backend boundaries, never readiness gates."""

    def sysctl(command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="Apple M3\n" if "brand_string" in command[-1] else "17179869184\n"
        )

    with (
        patch.object(runtime.platform, "system", return_value="Darwin"),
        patch.object(runtime.platform, "machine", return_value="arm64"),
        patch.object(runtime.platform, "mac_ver", return_value=("15.0", ("", "", ""), "")),
        patch("subprocess.run", side_effect=sysctl),
        patch.object(runtime, "_load_mlx_core", return_value=backend),
        patch.object(runtime, "_reduced_float32_matmul", return_value=False),
    ):
        yield


def test_simulated_darwin_arm64_selects_metal_and_refuses_cpu_fallback() -> None:
    selected: list[str] = []
    backend = SimpleNamespace(
        __version__="0.32.4", metal=SimpleNamespace(is_available=lambda: True), gpu="gpu",
        default_device=lambda: "Device(gpu, 0)", set_default_device=selected.append,
    )
    with _darwin_arm64(backend):
        report = runtime.require_mlx_device("  MLX ")
        assert report.available and report.reason is None
        assert (report.apple_chip, report.memory_bytes) == ("Apple M3", 17179869184)
        assert runtime.import_mlx() is backend
        assert selected == ["gpu"]
        with pytest.raises(errors.MissingDependencyError):
            runtime.require_mlx_device("cpu")
        backend.metal.is_available = lambda: False
        assert not runtime.probe_mlx_runtime().available
        with pytest.raises(errors.MissingDependencyError):
            runtime.require_mlx_runtime()


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "x86_64"), ("Linux", "x86_64")])
def test_unsupported_hosts_are_rejected_before_loading_mlx(system: str, machine: str) -> None:
    with (
        patch.object(runtime.platform, "system", return_value=system),
        patch.object(runtime.platform, "machine", return_value=machine),
        patch.object(runtime, "_load_mlx_core") as load_backend,
    ):
        assert not runtime.probe_mlx_runtime().available
        with pytest.raises(errors.MissingDependencyError):
            runtime.require_mlx_runtime()
        load_backend.assert_not_called()


def test_missing_mlx_reports_unavailable() -> None:
    with patch.object(runtime, "_load_mlx_core", side_effect=ImportError("not installed")):
        assert not runtime.probe_mlx_runtime().available
        with pytest.raises(errors.MissingDependencyError):
            runtime.import_mlx()


def test_report_serializes_without_backend_objects() -> None:
    metal = SimpleNamespace(is_available=lambda: True)
    backend = SimpleNamespace(__version__="0.32.4", metal=metal)
    with patch.object(runtime, "_load_mlx_core", return_value=backend):
        payload = runtime.probe_mlx_runtime().to_payload()
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize("version", ["0.32.3", "0.33.0", "unknown"])
def test_unsupported_mlx_versions_fail_before_backend_execution(version: str) -> None:
    with (
        patch.object(runtime.platform, "system", return_value="Darwin"),
        patch.object(runtime.platform, "machine", return_value="arm64"),
        patch.object(runtime, "_load_mlx_core", return_value=SimpleNamespace(__version__=version)),
        patch.object(runtime, "_metal_available") as metal,
    ):
        report = runtime.probe_mlx_runtime()
        assert not report.available and report.mlx_version == version
        with pytest.raises(errors.MissingDependencyError):
            runtime.import_mlx()
        metal.assert_not_called()


@pytest.mark.parametrize(("value", "expected"), [(None, "0"), ("1", "1")])
def test_full_float32_default_keeps_an_explicit_choice(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str
) -> None:
    if value is None:
        monkeypatch.delenv(runtime.TF32_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(runtime.TF32_VARIABLE, value)
    runtime.default_full_float32()
    assert os.environ[runtime.TF32_VARIABLE] == expected


@pytest.mark.parametrize(("value", "refused"), [("0", True), ("1", False)])
def test_reduced_float32_is_refused_without_opt_in(
    monkeypatch: pytest.MonkeyPatch, value: str, refused: bool
) -> None:
    monkeypatch.setenv(runtime.TF32_VARIABLE, value)
    with patch.object(runtime, "_reduced_float32_matmul", return_value=True):
        if refused:
            with pytest.raises(errors.MissingDependencyError):
                runtime.require_full_float32()
        else:
            runtime.require_full_float32()


def test_import_selects_full_float32_before_mlx_reads_it() -> None:
    # MLX fixes the setting at its first matmul, so each case needs a fresh process.
    code = """
import os
import medmlx_core
assert os.environ["MLX_ENABLE_TF32"] == "0"
mx = medmlx_core.import_mlx()
value = mx.full((64, 64), 1.0 + 2.0**-15, dtype=mx.float32)
product = mx.matmul(value, mx.ones((64, 64), dtype=mx.float32))
assert product[0, 0].item() == 64.0 + 2.0**-9
"""
    environment = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)
