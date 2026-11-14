"""Agent-facing contracts: readiness that can say ready, weight downloads as jobs,
coded errors, an optional root restriction, and background runs from Python."""

from __future__ import annotations

import gzip
import struct
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import Client

import medmlx
from medmlx.payloads import JobPayload, JsonValue, ModelReadiness, ReadinessPayload
from medmlx.server import build_server
from tests.conftest import add_distribution

# A weight-free model with a NIfTI input, a model with automatic weights, and one
# whose weights can only be supplied by hand.
_AGENT_MODELS = '''
SHAPE = {"id": "shape", "label": "Shape", "task": "testing", "run": "fakemlx:shape",
         "requires_weights": False}
WEIGHTED = {"id": "weighted", "label": "Weighted", "task": "testing", "run": "fakemlx:weighted"}
MANUAL = {"id": "manual", "label": "Manual", "task": "testing", "run": "fakemlx:weighted",
          "setup_supported": False}


def shape(image_path: str) -> dict[str, object]:
    return {"image_path": image_path}


def weighted(image_path: str, weights_path: str | None = None) -> dict[str, object]:
    return {"weights_path": weights_path}
'''
_AGENT_ENTRY_POINTS = (
    "[medmlx.models]\nshape = fakemlx:SHAPE\nweighted = fakemlx:WEIGHTED\nmanual = fakemlx:MANUAL\n"
)
REVISION = "a" * 40


@pytest.fixture
def agent_site(model_site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The fake model site plus the agent models, with an empty managed weights cache."""
    add_distribution(model_site, "agent-mlx", _AGENT_ENTRY_POINTS)
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + _AGENT_MODELS)
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "cache"))
    return model_site


def write_nifti(path: Path, shape: tuple[int, ...], *, version: int = 1, order: str = "<") -> Path:
    """Write a header-only NIfTI file; readiness never reads voxel data."""
    dims = (len(shape), *shape, *([1] * (7 - len(shape))))
    if version == 1:
        header = bytearray(348)
        struct.pack_into(f"{order}i", header, 0, 348)
        struct.pack_into(f"{order}8h", header, 40, *dims)
        header[344:348] = b"n+1\0"
    else:
        header = bytearray(540)
        struct.pack_into(f"{order}i", header, 0, 540)
        header[4:12] = b"n+2\0\r\n\x1a\n"
        struct.pack_into(f"{order}8q", header, 16, *dims)
    data = bytes(header) + bytes(4)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as stream:
            stream.write(data)
    else:
        path.write_bytes(data)
    return path


def _row(model_id: str, arguments: dict[str, object], root: Path) -> ModelReadiness:
    values = cast(dict[str, JsonValue], arguments)
    report = medmlx.model_readiness([model_id], arguments=values, root=root)
    return report["models"][0]


def _with_client(root: Path, body: Callable[[Client], Awaitable[None]], **options: bool) -> None:
    async def run() -> None:
        async with Client(build_server(root, **options)) as client:
            await body(client)

    anyio.run(run)


async def _error(client: Client, tool: str, arguments: dict[str, object]) -> str:
    """Call a tool that must fail and return its message after the SDK's prefix."""
    result = await client.call_tool(tool, arguments)
    assert result.is_error
    text = "".join(getattr(item, "text", "") for item in result.content)
    prefix = f"Error executing tool {tool}: "
    return text.removeprefix(prefix)


def test_readiness_is_true_only_when_every_check_passes(agent_site: Path, tmp_path: Path) -> None:
    ct = write_nifti(tmp_path / "ct.nii.gz", (32, 32, 16))
    assert medmlx.nifti_shape(write_nifti(tmp_path / "b.nii", (8, 9, 2), order=">")) == [8, 9, 2]
    assert medmlx.nifti_shape(write_nifti(tmp_path / "c.nii", (5, 6, 7), version=2)) == [5, 6, 7]
    ready = _row("shape", {"image_path": "ct.nii.gz"}, tmp_path)
    assert ready["run_ready"] is True and ready["reason"] is None
    assert ready["weights"] == {"required": False, "source": None, "path": None}
    assert ready["image_checks"][0]["shape"] == [32, 32, 16]

    (tmp_path / "broken.nii.gz").write_bytes(b"not gzip")
    broken = _row("shape", {"image_path": "broken.nii.gz"}, tmp_path)
    assert broken["run_ready"] is False and "Unreadable NIfTI" in str(broken["reason"])

    (tmp_path / "image.json").write_text("[1]")
    unchecked = _row("shape", {"image_path": "image.json"}, tmp_path)
    assert unchecked["run_ready"] is None and "not NIfTI" in str(unchecked["reason"])

    downloads = _row("weighted", {"image_path": str(ct)}, tmp_path)
    assert (downloads["run_ready"], downloads["weights"]["source"]) == (None, "download")

    cached = tmp_path / "cache" / "weighted"
    cached.mkdir(parents=True)
    (cached / ".medmlx-revision").write_text(REVISION)
    from_cache = _row("weighted", {"image_path": str(ct)}, tmp_path)
    assert (from_cache["run_ready"], from_cache["weights"]) == (
        True, {"required": True, "source": "cache", "path": cached.as_posix()})

    (tmp_path / "weights").mkdir()
    supplied = _row("weighted", {"image_path": str(ct), "weights_path": "weights"}, tmp_path)
    assert (supplied["run_ready"], supplied["weights"]["source"]) == (True, "argument")

    manual = _row("manual", {"image_path": str(ct)}, tmp_path)
    assert manual["run_ready"] is False and "needs a local weights_path" in str(manual["reason"])


def test_prepare_model_downloads_as_a_job_and_rejects_manual_assets(
    agent_site: Path, tmp_path: Path,
) -> None:
    async def body(client: Client) -> None:
        started = await client.call_tool("prepare_model", {"model_id": "shape"})
        assert not started.is_error, started.content
        job = cast(JobPayload, started.structured_content)
        finished = await client.call_tool("get_job", {"job_id": job["job_id"], "wait_seconds": 60})
        payload = cast(JobPayload, finished.structured_content)
        assert payload["state"] == "completed", payload["log_tail"]
        assert payload.get("result") == {
            "model_id": "shape", "arguments": {}, "source_url": None, "revision": None}
        manual = await _error(client, "prepare_model", {"model_id": "manual"})
        assert manual.startswith("manual_setup: ") and "\nNext: " in manual
        assert (await _error(client, "prepare_model", {"model_id": "absent"})).startswith(
            "unknown_model: ")
        assert (await _error(client, "get_job", {"job_id": "nope"})).startswith("unknown_job: ")

    _with_client(tmp_path, body)


def test_restricted_root_rejects_paths_outside_it(agent_site: Path, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write_nifti(root / "ct.nii.gz", (4, 4, 4))
    outside = write_nifti(tmp_path / "outside.nii.gz", (4, 4, 4))

    async def body(client: Client) -> None:
        inside = await client.call_tool(
            "model_readiness", {"model_ids": ["shape"], "arguments": {"image_path": "ct.nii.gz"}})
        report = cast(ReadinessPayload, inside.structured_content)
        assert report["models"][0]["run_ready"] is True
        requests: tuple[tuple[str, dict[str, object]], ...] = (
            ("model_readiness",
             {"model_ids": ["shape"], "arguments": {"image_path": str(outside)}}),
            ("run_model", {"model_id": "shape", "arguments": {"image_path": "../outside.nii.gz"}}),
        )
        for tool, arguments in requests:
            assert (await _error(client, tool, arguments)).startswith("path_outside_root: ")

    _with_client(root, body, restrict_paths=True)


def test_python_start_returns_at_once_and_can_wait_or_cancel(
    model_site: Path, tmp_path: Path,
) -> None:
    (tmp_path / "image.json").write_text("[0.1, 0.9]")
    arguments: dict[str, JsonValue] = {"image_path": "image.json", "output_path": "mask.json"}
    finished = medmlx.start("threshold", arguments=arguments, root=tmp_path).wait()
    assert finished.job["state"] == "completed"
    with medmlx.start("sleeper", arguments={"seconds": 30}, root=tmp_path) as run:
        assert run.status()["state"] == "running"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.2)
        assert run.cancel()["state"] == "cancelled"
