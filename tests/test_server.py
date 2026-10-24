"""MCP contracts: stdio wiring, entry point discovery, and supervised runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, cast, overload

import anyio
import pytest
from mcp import Client, StdioServerParameters

from medmlx.payloads import (
    DiscoveryError,
    JobListPayload,
    JobPayload,
    ModelDescription,
    ModelListPayload,
    ModelPayload,
    PackageListPayload,
    ReadinessPayload,
    RuntimeToolPayload,
)
from medmlx.server import build_server
from tests.conftest import BROKEN_MLX, NO_METAL_MLX, add_distribution, set_mlx

_TOOLS = {
    "cancel_job",
    "describe_model",
    "get_job",
    "list_jobs",
    "list_models",
    "list_packages",
    "model_readiness",
    "run_model",
    "runtime_info",
}
_SERVED = ["crashing", "failing", "sleeper", "threshold"]
# Real MedMLX packages may be installed alongside the fakes; assertions look only at the fakes.
_FAKE_PACKAGES = {"fakemlx", "otherpkg", "loose-mlx"}
_FAKE_ENTRY_POINTS = ("fakemlx:", "fakemlx_absent:")


def _fake_ids(models: Sequence[ModelPayload]) -> list[str]:
    return sorted(model["id"] for model in models if model["package"] in _FAKE_PACKAGES)


def _fake_errors(errors: list[DiscoveryError]) -> list[DiscoveryError]:
    return [
        item
        for item in errors
        if item["entry_point"].split(" = ")[1].startswith(_FAKE_ENTRY_POINTS)
    ]


def _with_client(root: Path, body: Callable[[Client], Awaitable[None]]) -> None:
    async def run() -> None:
        async with Client(build_server(root)) as client:
            await body(client)

    anyio.run(run)


type ToolPayload = (
    ModelListPayload | ModelDescription | ReadinessPayload | RuntimeToolPayload
    | PackageListPayload | JobListPayload | JobPayload
)


@overload
async def _call(
    client: Client, tool: Literal["list_models"], arguments: dict[str, object]
) -> ModelListPayload: ...


@overload
async def _call(
    client: Client, tool: Literal["describe_model"], arguments: dict[str, object]
) -> ModelDescription: ...


@overload
async def _call(
    client: Client, tool: Literal["model_readiness"], arguments: dict[str, object]
) -> ReadinessPayload: ...


@overload
async def _call(
    client: Client, tool: Literal["runtime_info"], arguments: dict[str, object]
) -> RuntimeToolPayload: ...


@overload
async def _call(
    client: Client, tool: Literal["list_packages"], arguments: dict[str, object]
) -> PackageListPayload: ...


@overload
async def _call(
    client: Client, tool: Literal["list_jobs"], arguments: dict[str, object]
) -> JobListPayload: ...


@overload
async def _call(
    client: Client,
    tool: Literal["run_model", "get_job", "cancel_job"],
    arguments: dict[str, object],
) -> JobPayload: ...


async def _call(client: Client, tool: str, arguments: dict[str, object]) -> ToolPayload:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return cast(ToolPayload, result.structured_content)


async def _finish(client: Client, model_id: str, arguments: dict[str, object]) -> JobPayload:
    started = await _call(client, "run_model", {"model_id": model_id, "arguments": arguments})
    assert started["state"] == "running"
    return await _call(client, "get_job", {"job_id": started["job_id"], "wait_seconds": 60})


def test_stdio_server_discovers_models_and_reports_the_runtime(
    model_site: Path, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medmlx", "serve", "--root", str(root)],
        env=dict(os.environ),
        cwd=tmp_path,
    )

    async def run() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == _TOOLS
            listed = await _call(client, "list_models", {})
            assert _fake_ids(listed["models"]) == _SERVED
            runtime = await _call(client, "runtime_info", {})
            assert runtime["root"] == root.resolve().as_posix()
            assert runtime["python"] == platform.python_version()
            assert runtime["mlx"] == {
                "usable": True,
                "version": "0.0-fake",
                "metal_available": True,
                "default_device": "Device(gpu, 0)",
                "device_info": {"device_name": "Apple test GPU"},
            }

    anyio.run(run)


def test_run_model_executes_in_the_root_and_returns_the_result(
    model_site: Path, tmp_path: Path
) -> None:
    (tmp_path / "values.json").write_text("[0.2, 0.7, 0.9]")

    async def body(client: Client) -> None:
        job = await _finish(
            client, "threshold", {"image_path": "values.json", "output_path": "mask.json"}
        )
        assert job["state"] == "completed", job["log_tail"]
        assert job["exit_code"] == 0
        assert job.get("result") == {"output_path": "mask.json", "foreground": 2}

    _with_client(tmp_path, body)
    assert json.loads((tmp_path / "mask.json").read_text()) == [False, True, True]


def test_run_failure_reports_the_exception(model_site: Path, tmp_path: Path) -> None:
    async def body(client: Client) -> None:
        job = await _finish(client, "failing", {"message": "bad input"})
        assert job["state"] == "failed"
        assert job.get("error") == {"type": "ValueError", "message": "bad input"}
        assert "ValueError: bad input" in job["log_tail"]

    _with_client(tmp_path, body)


@pytest.mark.parametrize(
    ("result", "error_type", "message"),
    [
        ("[1, 2]", "TypeError", "Model run must return a JSON object"),
        *[
            (result, "TypeError", "Model run JSON object keys must be strings")
            for result in (
                '{1: "first", "1": "second"}',
                '{"details": {1.5: "value"}}',
                '{"details": [{False: "value"}]}',
                '{"details": ({None: "value"},)}',
            )
        ],
        (
            '{"details": {"value": object()}}',
            "TypeError",
            "Object of type object is not JSON serializable",
        ),
        *[
            (
                f'{{"details": {{"value": float({value!r})}}}}',
                "ValueError",
                "Out of range float values are not JSON compliant",
            )
            for value in ("nan", "inf", "-inf")
        ],
    ],
)
def test_run_result_contract_reports_invalid_output(
    model_site: Path, tmp_path: Path, result: str, error_type: str, message: str
) -> None:
    (model_site / "invalid_result.py").write_text(
        'MODEL = {"id": "invalid-result", "label": "Invalid result", "task": "testing", '
        '"run": "invalid_result:run"}\n'
        f"def run():\n    return {result}\n"
    )
    add_distribution(
        model_site, "invalid-result", "[medmlx.models]\ninvalid-result = invalid_result:MODEL\n"
    )

    async def body(client: Client) -> None:
        job = await _finish(client, "invalid-result", {})
        assert job["state"] == "failed"
        assert job["exit_code"] == 1
        error = job.get("error")
        assert error is not None
        assert error["type"] == error_type
        assert message in error["message"]

    _with_client(tmp_path, body)


def test_a_crashing_worker_fails_the_job_and_the_server_keeps_serving(
    model_site: Path, tmp_path: Path
) -> None:
    async def body(client: Client) -> None:
        job = await _finish(client, "crashing", {})
        assert job["state"] == "failed"
        assert job["exit_code"] == 3
        error = job.get("error")
        assert error is not None
        assert error["type"] == "WorkerExited"
        listed = await _call(client, "list_models", {})
        assert _fake_ids(listed["models"]) == _SERVED

    _with_client(tmp_path, body)


@pytest.mark.parametrize(
    ("model_id", "arguments", "message"),
    [
        (
            "threshold",
            {"image_path": "a", "output_path": "b", "bogus": 1},
            "unexpected keyword argument 'bogus'",
        ),
        ("threshold", {"image_path": "a"}, "missing a required argument: 'output_path'"),
        ("nope", {}, "Unknown MedMLX model 'nope'"),
    ],
)
def test_run_model_rejects_bad_requests_before_starting_a_worker(
    model_site: Path, tmp_path: Path, model_id: str, arguments: dict[str, object], message: str
) -> None:
    async def body(client: Client) -> None:
        result = await client.call_tool(
            "run_model", {"model_id": model_id, "arguments": arguments}
        )
        assert result.is_error
        assert message in str(result.content)
        assert (await _call(client, "list_jobs", {}))["jobs"] == []

    _with_client(tmp_path, body)


def test_cancel_job_stops_a_running_worker(model_site: Path, tmp_path: Path) -> None:
    async def body(client: Client) -> None:
        started = await _call(
            client, "run_model", {"model_id": "sleeper", "arguments": {"seconds": 30}}
        )
        job = await _call(client, "cancel_job", {"job_id": started["job_id"]})
        assert job["state"] == "cancelled"
        assert job["exit_code"] != 0

    _with_client(tmp_path, body)


def test_discovery_reports_bad_entry_points_and_drops_duplicate_ids(
    model_site: Path, tmp_path: Path
) -> None:
    add_distribution(model_site, "otherpkg", "[medmlx.models]\nsleeper2 = fakemlx:SLEEPER\n")

    async def body(client: Client) -> None:
        listed = await _call(client, "list_models", {})
        assert _fake_ids(listed["models"]) == ["crashing", "failing", "threshold"]
        errors = {
            item["entry_point"]: item["error"] for item in _fake_errors(listed["discovery_errors"])
        }
        assert set(errors) == {
            "invalid = fakemlx:INVALID",
            "missing = fakemlx_absent:MODEL",
            "sleeper = fakemlx:SLEEPER",
            "sleeper2 = fakemlx:SLEEPER",
        }
        assert "field 'run' must be non-empty text" in errors["invalid = fakemlx:INVALID"]
        assert "ModuleNotFoundError" in errors["missing = fakemlx_absent:MODEL"]
        assert "declared more than once" in errors["sleeper2 = fakemlx:SLEEPER"]

    _with_client(tmp_path, body)


def test_readiness_and_description_come_from_the_package(
    model_site: Path, tmp_path: Path
) -> None:
    async def body(client: Client) -> None:
        readiness = await _call(client, "model_readiness", {"model_ids": ["threshold", "sleeper"]})
        assert readiness["mlx_usable"] is True
        assert readiness["models"] == [
            {
                "id": "threshold",
                "ready": False,
                "reason": "weights not staged",
                "setup_hint": "stage the weights",
            },
            {"id": "sleeper", "ready": True, "reason": None, "setup_hint": None},
        ]
        described = await _call(client, "describe_model", {"model_id": "threshold"})
        assert (described["package"], described["version"]) == ("fakemlx", "1.2.3")
        assert described["family"] == "segment"
        assert [
            (row["name"], row["required"], row["default"]) for row in described["parameters"]
        ] == [("image_path", True, None), ("output_path", True, None), ("level", False, 0.5)]

    _with_client(tmp_path, body)


def test_models_command_prints_discovery_and_fails_on_bad_entry_points(
    model_site: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "medmlx", "models"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert _fake_ids(payload["models"]) == _SERVED
    assert len(_fake_errors(payload["discovery_errors"])) == 2


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (BROKEN_MLX, "ImportError: simulated broken MLX installation"),
        (NO_METAL_MLX, "MLX Metal GPU is not available"),
    ],
)
def test_readiness_fails_every_model_when_mlx_is_unusable(
    model_site: Path, tmp_path: Path, source: str, reason: str
) -> None:
    set_mlx(model_site, source)

    async def body(client: Client) -> None:
        runtime = await _call(client, "runtime_info", {})
        assert runtime["mlx"]["usable"] is False
        assert runtime["mlx"].get("reason") == reason
        readiness = await _call(client, "model_readiness", {"model_ids": ["threshold", "sleeper"]})
        assert readiness["mlx_usable"] is False
        threshold, sleeper = readiness["models"]
        assert threshold["ready"] is False
        assert threshold["reason"] == f"weights not staged; MLX is not usable: {reason}"
        assert threshold["setup_hint"] == "stage the weights"
        assert sleeper == {
            "id": "sleeper",
            "ready": False,
            "reason": f"MLX is not usable: {reason}",
            "setup_hint": "Use macOS arm64 (Apple Silicon) with MLX installed and Metal available.",
        }
        listed = await _call(client, "list_models", {})
        assert _fake_ids(listed["models"]) == _SERVED
        result = await client.call_tool(
            "run_model", {"model_id": "sleeper", "arguments": {"seconds": 0}}
        )
        assert result.is_error
        assert f"MLX is not usable: {reason}" in str(result.content)
        assert (await _call(client, "list_jobs", {}))["jobs"] == []

    _with_client(tmp_path, body)


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "x86_64"), ("Windows", "arm64")])
def test_runtime_probe_rejects_unsupported_hosts_before_importing_mlx(
    model_site: Path, system: str, machine: str
) -> None:
    set_mlx(model_site, 'raise AssertionError("MLX must not import on unsupported hosts")\n')
    code = (
        "import platform, runpy\n"
        f"platform.system = lambda: {system!r}\n"
        f"platform.machine = lambda: {machine!r}\n"
        "runpy.run_module('medmlx.probe', run_name='__main__')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {
        "usable": False,
        "reason": f"MedMLX requires macOS arm64 (Apple Silicon); found {system} {machine}",
    }


def test_package_import_loads_no_mlx_or_reference_stack() -> None:
    code = (
        "import sys\n"
        "import medmlx.server\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "assert not loaded & {'mlx', 'torch', 'monai'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_list_packages_shows_families_sub_packages_and_their_models(
    model_site: Path, tmp_path: Path
) -> None:
    add_distribution(
        model_site, "loose-mlx", "[medmlx.models]\nloose = fakemlx:LOOSE\n"
    )
    (model_site / "fakemlx" / "__init__.py").write_text(
        (model_site / "fakemlx" / "__init__.py").read_text()
        + '\nLOOSE = {"id": "loose", "label": "Loose", "task": "testing", "run": "fakemlx:sleep"}\n'
    )

    async def body(client: Client) -> None:
        tree = await _call(client, "list_packages", {})
        assert tree["families"] == [
            {
                "name": "segment",
                "version": "1.2.3",
                "packages": [
                    {
                        "name": "fakemlx",
                        "version": "1.2.3",
                        "source": "",
                        "models": ["crashing", "failing", "sleeper", "threshold"],
                    }
                ],
            },
            {
                "name": "generate",
                "version": "1.2.3",
                "packages": [
                    {
                        "name": "absent-mlx",
                        "version": None,
                        "source": "git+https://example.invalid/absent-mlx.git@v9.0.0",
                        "models": [],
                    }
                ],
            },
            {"name": "vision", "version": "1.2.3", "packages": []},
        ]
        fake_other = [model_id for model_id in tree["other_models"] if model_id in {"loose"}]
        assert fake_other == ["loose"]
        listed = await _call(client, "list_models", {"family": "segment"})
        assert sorted(model["id"] for model in listed["models"]) == _SERVED
        assert {model["family"] for model in listed["models"]} == {"segment"}

    _with_client(tmp_path, body)
