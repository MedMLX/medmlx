"""MCP contracts: stdio wiring, entry point discovery, and supervised runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import Client, StdioServerParameters

from medmlx import api
from medmlx.payloads import (
    CheckpointListPayload,
    DiscoveryError,
    JobListPayload,
    JobPayload,
    ModelDescription,
    ModelListPayload,
    ModelPayload,
    ModelReadiness,
    PackageListPayload,
    ReadinessPayload,
    RuntimeToolPayload,
)
from medmlx.server import build_server
from tests.conftest import BROKEN_MLX, NO_METAL_MLX, add_distribution, set_mlx

_TOOLS = {
    "cancel_job", "describe_model", "get_job", "list_jobs", "list_models",
    "list_checkpoints", "list_packages", "model_readiness", "prepare_model", "run_model",
    "runtime_info", "list_dicom_series", "convert_dicom", "summarize_output", "preview_output",
    "memory_status",
}
_SERVED = ["crashing", "failing", "sleeper", "threshold"]
# Real MedMLX packages may be installed alongside the fakes; assertions look only at the fakes.
_FAKE_PACKAGES = {"fakemlx", "otherpkg", "loose-mlx"}
_FAKE_ENTRY_POINTS = ("fakemlx:", "fakemlx_absent:")


def _fake_ids(models: Sequence[ModelPayload]) -> list[str]:
    return sorted(model["id"] for model in models if model["package"] in _FAKE_PACKAGES)


def _fake_errors(errors: list[DiscoveryError]) -> list[DiscoveryError]:
    return [item for item in errors
            if item["entry_point"].split(" = ")[1].startswith(_FAKE_ENTRY_POINTS)]


def _with_client(root: Path, body: Callable[[Client], Awaitable[None]]) -> None:
    async def run() -> None:
        async with Client(build_server(root)) as client:
            await body(client)

    anyio.run(run)


async def _call[T](client: Client, tool: str, arguments: dict[str, object], kind: type[T]) -> T:
    """Call a tool that must succeed and return its structured payload as ``kind``."""
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return cast(T, result.structured_content)


async def _jobs(client: Client) -> list[JobPayload]:
    return (await _call(client, "list_jobs", {}, JobListPayload))["jobs"]


async def _start(client: Client, model_id: str, arguments: dict[str, object]) -> JobPayload:
    request: dict[str, object] = {"model_id": model_id, "arguments": arguments}
    return await _call(client, "run_model", request, JobPayload)


async def _finish(client: Client, model_id: str, arguments: dict[str, object]) -> JobPayload:
    started = await _start(client, model_id, arguments)
    assert started["state"] == "running"
    request: dict[str, object] = {"job_id": started["job_id"], "wait_seconds": 60}
    return await _call(client, "get_job", request, JobPayload)


def test_stdio_server_discovers_models_and_reports_the_runtime(
    model_site: Path, tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    server = StdioServerParameters(command=sys.executable,
                                   args=["-m", "medmlx", "serve", "--root", str(root)],
                                   env=dict(os.environ), cwd=tmp_path)

    async def run() -> None:
        async with Client(server) as client:
            assert {tool.name for tool in (await client.list_tools()).tools} == _TOOLS
            listed = await _call(client, "list_models", {}, ModelListPayload)
            assert _fake_ids(listed["models"]) == _SERVED
            runtime = await _call(client, "runtime_info", {}, RuntimeToolPayload)
            assert runtime["root"] == root.resolve().as_posix()
            assert runtime["mlx"] == {
                "usable": True, "version": "0.0-fake", "metal_available": True,
                "default_device": "Device(gpu, 0)",
                "device_info": {"device_name": "Apple test GPU"},
            }

    anyio.run(run)


def test_run_model_executes_in_the_root_and_returns_the_result(
    model_site: Path, tmp_path: Path,
) -> None:
    (tmp_path / "values.json").write_text("[0.2, 0.7, 0.9]")

    async def body(client: Client) -> None:
        job = await _finish(client, "threshold",
                            {"image_path": "values.json", "output_path": "mask.json"})
        assert job["state"] == "completed", job["log_tail"]
        assert job.get("result") == {"output_path": "mask.json", "foreground": 2}

    _with_client(tmp_path, body)
    assert json.loads((tmp_path / "mask.json").read_text()) == [False, True, True]


def test_worker_failures_are_reported_and_the_server_keeps_serving(
    model_site: Path, tmp_path: Path,
) -> None:
    async def body(client: Client) -> None:
        failed = await _finish(client, "failing", {"message": "bad input"})
        assert failed["state"] == "failed"
        assert failed.get("error") == {
            "type": "AssetNotReadyError", "message": "bad input",
            "category": "AssetNotReadyError", "hint": "Run medmlx setup failing",
        }
        crashed = await _finish(client, "crashing", {})
        assert crashed["state"] == "failed" and crashed["exit_code"] == 3
        error = crashed.get("error")
        assert error is not None and error["type"] == "WorkerExited"
        listed = await _call(client, "list_models", {}, ModelListPayload)
        assert _fake_ids(listed["models"]) == _SERVED

    _with_client(tmp_path, body)


@pytest.mark.parametrize(
    ("result", "error_type"),
    [
        ("[1, 2]", "TypeError"),
        ('{"details": [{False: "value"}]}', "TypeError"),
        ('{"details": {"value": object()}}', "TypeError"),
        ('{"details": {"value": float("nan")}}', "ValueError"),
    ],
    ids=["not-an-object", "non-string-key", "not-serializable", "non-finite"],
)
def test_run_result_contract_reports_invalid_output(
    model_site: Path, tmp_path: Path, result: str, error_type: str,
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
        assert job["state"] == "failed" and job["exit_code"] == 1
        error = job.get("error")
        assert error is not None and error["type"] == error_type

    _with_client(tmp_path, body)


@pytest.mark.parametrize(
    ("model_id", "arguments"),
    [("threshold", {"image_path": "a", "output_path": "b", "bogus": 1}), ("nope", {})],
    ids=["bad-arguments", "unknown-model"],
)
def test_run_model_rejects_bad_requests_before_starting_a_worker(
    model_site: Path, tmp_path: Path, model_id: str, arguments: dict[str, object],
) -> None:
    async def body(client: Client) -> None:
        result = await client.call_tool("run_model", {"model_id": model_id, "arguments": arguments})
        assert result.is_error
        assert await _jobs(client) == []

    _with_client(tmp_path, body)


def test_cancel_job_stops_a_running_worker(model_site: Path, tmp_path: Path) -> None:
    async def body(client: Client) -> None:
        started = await _start(client, "sleeper", {"seconds": 30})
        job = await _call(client, "cancel_job", {"job_id": started["job_id"]}, JobPayload)
        assert job["state"] == "cancelled" and job["exit_code"] != 0

    _with_client(tmp_path, body)


def test_run_model_limits_a_session_to_one_active_worker(
    model_site: Path, tmp_path: Path,
) -> None:
    async def body(client: Client) -> None:
        first = await _start(client, "sleeper", {"seconds": 30})
        rejected = await client.call_tool(
            "run_model", {"model_id": "sleeper", "arguments": {"seconds": 30}})
        assert rejected.is_error
        assert [job["job_id"] for job in await _jobs(client)] == [first["job_id"]]
        await _call(client, "cancel_job", {"job_id": first["job_id"]}, JobPayload)
        assert (await _finish(client, "sleeper", {"seconds": 0.1}))["state"] == "completed"

    _with_client(tmp_path, body)


def test_discovery_reports_bad_entry_points_and_drops_duplicate_ids(
    model_site: Path, tmp_path: Path,
) -> None:
    add_distribution(model_site, "otherpkg", "[medmlx.models]\nsleeper2 = fakemlx:SLEEPER\n")

    async def body(client: Client) -> None:
        listed = await _call(client, "list_models", {}, ModelListPayload)
        assert _fake_ids(listed["models"]) == ["crashing", "failing", "threshold"]
        assert {item["entry_point"] for item in _fake_errors(listed["discovery_errors"])} == {
            "invalid = fakemlx:INVALID",
            "missing = fakemlx_absent:MODEL",
            "sleeper = fakemlx:SLEEPER",
            "sleeper2 = fakemlx:SLEEPER",
        }

    _with_client(tmp_path, body)


def test_discovery_keeps_a_model_with_unknown_fields_and_warns(model_site: Path) -> None:
    add_distribution(model_site, "otherpkg", "[medmlx.models]\nfuture = fakemlx:FUTURE\n")
    listed = api.list_models()
    assert "future" in _fake_ids(listed["models"])
    assert [item for item in listed["discovery_warnings"]
            if item["entry_point"] == "future = fakemlx:FUTURE"] == [
        {"entry_point": "future = fakemlx:FUTURE",
         "warning": "Ignored unknown field(s): future_field"}]
    assert "future = fakemlx:FUTURE" not in {
        item["entry_point"] for item in listed["discovery_errors"]}


def test_models_command_prints_discovery_and_fails_on_bad_entry_points(
    model_site: Path,
) -> None:
    completed = subprocess.run([sys.executable, "-m", "medmlx", "models"],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert _fake_ids(payload["models"]) == _SERVED
    assert len(_fake_errors(payload["discovery_errors"])) == 2


def test_readiness_and_description_come_from_the_package(
    model_site: Path, tmp_path: Path,
) -> None:
    async def body(client: Client) -> None:
        readiness = await _call(client, "model_readiness",
                                {"model_ids": ["threshold", "sleeper"]}, ReadinessPayload)
        assert readiness["mlx_usable"] is True
        threshold, sleeper = readiness["models"]
        assert (threshold["ready"], threshold["run_ready"]) == (False, False)
        assert threshold["setup_hint"] == "stage the weights"
        assert (sleeper["ready"], sleeper["run_ready"]) == (True, None)
        described = await _call(client, "describe_model", {"model_id": "threshold"},
                                ModelDescription)
        assert (described["package"], described["family"]) == ("fakemlx", "segment")
        assert [
            (row["name"], row["required"], row["default"]) for row in described["parameters"]
        ] == [("image_path", True, None), ("output_path", True, None), ("level", False, 0.5)]
        assert described["parameters"][2]["description"] == (
            "Unitless cutoff; values strictly greater become foreground."
        )

    _with_client(tmp_path, body)


def test_readiness_checks_supplied_paths_without_claiming_checkpoint_validation(
    model_site: Path, tmp_path: Path,
) -> None:
    (tmp_path / "image.json").write_text("[0.2, 0.8]")
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + '\nTHRESHOLD.pop("readiness")\n')

    async def body(client: Client) -> None:
        async def readiness(model_id: str, arguments: dict[str, object]) -> ModelReadiness:
            request: dict[str, object] = {"model_ids": [model_id], "arguments": arguments}
            return (await _call(client, "model_readiness", request, ReadinessPayload))["models"][0]

        paths = {"output_path": "mask.json"}
        missing = await readiness("threshold", {"image_path": "absent.json", **paths})
        assert (missing["ready"], missing["run_ready"], missing["arguments_valid"]) == (
            True, False, True)
        present = await readiness("threshold", {"image_path": "image.json", **paths})
        assert present["run_ready"] is None
        assert (present["weights_validated"], present["inputs_validated"]) == (False, False)
        invalid = await readiness("sleeper", {"misspelled": 0})
        assert (invalid["run_ready"], invalid["arguments_valid"]) == (False, False)
        assert not (tmp_path / "mask.json").exists()
        assert await _jobs(client) == []

    _with_client(tmp_path, body)


@pytest.mark.parametrize("source", [BROKEN_MLX, NO_METAL_MLX], ids=["broken-mlx", "no-metal"])
def test_readiness_fails_every_model_when_mlx_is_unusable(
    model_site: Path, tmp_path: Path, source: str,
) -> None:
    set_mlx(model_site, source)

    async def body(client: Client) -> None:
        runtime = await _call(client, "runtime_info", {}, RuntimeToolPayload)
        assert runtime["mlx"]["usable"] is False and runtime["mlx"].get("reason")
        readiness = await _call(client, "model_readiness",
                                {"model_ids": ["threshold", "sleeper"]}, ReadinessPayload)
        assert readiness["mlx_usable"] is False
        assert [(row["ready"], row["run_ready"]) for row in readiness["models"]] == [
            (False, False), (False, False)]
        assert readiness["models"][0]["setup_hint"] == "stage the weights"
        listed = await _call(client, "list_models", {}, ModelListPayload)
        assert _fake_ids(listed["models"]) == _SERVED
        result = await client.call_tool(
            "run_model", {"model_id": "sleeper", "arguments": {"seconds": 0}})
        assert result.is_error
        assert await _jobs(client) == []

    _with_client(tmp_path, body)


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "x86_64"), ("Windows", "arm64")])
def test_runtime_probe_rejects_unsupported_hosts_before_importing_mlx(
    model_site: Path, system: str, machine: str,
) -> None:
    set_mlx(model_site, 'raise AssertionError("MLX must not import on unsupported hosts")\n')
    code = (
        "import platform, runpy\n"
        f"platform.system = lambda: {system!r}\n"
        f"platform.machine = lambda: {machine!r}\n"
        "runpy.run_module('medmlx.probe', run_name='__main__')\n"
    )
    completed = subprocess.run([sys.executable, "-c", code],
                               check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["usable"] is False
    assert f"{system} {machine}" in report["reason"]  # Not the import of the poisoned MLX.


def test_package_import_loads_no_mlx_or_reference_stack() -> None:
    code = (
        "import sys\n"
        "import medmlx.server\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "assert not loaded & {'mlx', 'torch', 'monai'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_list_packages_shows_families_sub_packages_and_their_models(
    model_site: Path, tmp_path: Path,
) -> None:
    add_distribution(model_site, "loose-mlx", "[medmlx.models]\nloose = fakemlx:LOOSE\n")
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + '\nLOOSE = {"id": "loose", "label": "Loose", '
                      '"task": "testing", "run": "fakemlx:sleep"}\n')

    async def body(client: Client) -> None:
        tree = await _call(client, "list_packages", {}, PackageListPayload)
        families = {family["name"]: family["packages"] for family in tree["families"]}
        assert list(families) == ["segment", "generate", "vision"]
        assert [(row["name"], row["models"]) for row in families["segment"]] == [
            ("fakemlx", _SERVED)]
        assert [(row["name"], row["version"]) for row in families["generate"]] == [
            ("absent-mlx", None)]
        assert "loose" in tree["other_models"]
        listed = await _call(client, "list_models", {"family": "segment"}, ModelListPayload)
        assert sorted(model["id"] for model in listed["models"]) == _SERVED

    _with_client(tmp_path, body)


def test_checkpoint_catalog_does_not_register_or_qualify_research_sources(tmp_path: Path) -> None:
    async def body(client: Client) -> None:
        listed = await _call(client, "list_checkpoints", {"model_id": "nnunet-autopet"},
                             CheckpointListPayload)
        assert listed["checkpoints"]
        assert all(row["model_id"] == "nnunet-autopet" for row in listed["checkpoints"])
        assert all(row["qualification"] == "not_assessed_by_catalog"
                   for row in listed["checkpoints"])
        models = {row["id"] for row in (await _call(client, "list_models", {},
                                                    ModelListPayload))["models"]}
        assert not models & {row["id"] for row in listed["checkpoints"]}
        assert await _jobs(client) == []
        assert list(tmp_path.iterdir()) == []

    _with_client(tmp_path, body)
