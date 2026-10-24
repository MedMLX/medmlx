"""Report the MLX runtime from a child process, so a native failure stays contained."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import cast

from medmlx.payloads import MlxProbe, RuntimePayload

_TIMEOUT_SECONDS = 60


def runtime_info() -> RuntimePayload:
    """Return platform facts and what MLX reports, probed in a child process.

    ``packages`` gives the installed versions of ``mlx``, ``medmlx-core`` and
    ``mcp`` (``None`` when absent). ``mlx.usable`` is true only on macOS arm64
    with Metal available; otherwise ``mlx.reason`` says why. A probe that
    crashes, fails or exceeds 60 seconds is reported as unusable rather than
    raised, because a native MLX failure would otherwise end this process.
    """

    payload: RuntimePayload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in ("mlx", "medmlx-core", "mcp")},
        "mlx": {"usable": False},
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "medmlx.probe"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        payload["mlx"] = {"usable": False, "reason": f"probe timed out after {_TIMEOUT_SECONDS}s"}
        return payload
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or [f"exit {completed.returncode}"]
        payload["mlx"] = {"usable": False, "reason": f"probe failed: {detail[0]}"}
        return payload
    payload["mlx"] = cast(MlxProbe, json.loads(completed.stdout))
    return payload


def _probe() -> MlxProbe:
    system, machine = platform.system(), platform.machine()
    if (system, machine) != ("Darwin", "arm64"):
        return {
            "usable": False,
            "reason": f"MedMLX requires macOS arm64 (Apple Silicon); found {system} {machine}",
        }
    try:
        import mlx.core as mx

        metal_available = bool(mx.metal.is_available())
        report: MlxProbe = {
            "usable": metal_available,
            "version": cast(str | None, getattr(mx, "__version__", None)),
            "metal_available": metal_available,
        }
        if not metal_available:
            return {**report, "reason": "MLX Metal GPU is not available"}
        report["default_device"] = str(mx.default_device())
        report["device_info"] = {
            str(key): _plain(value) for key, value in mx.metal.device_info().items()
        }
        return report
    except Exception as error:
        return {"usable": False, "reason": f"{type(error).__name__}: {error}"}


def _plain(value: object) -> str | int | float | bool | None:
    return value if isinstance(value, str | int | float | bool) or value is None else str(value)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    print(json.dumps(_probe()))
