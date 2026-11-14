"""One model holds unified memory at a time across every MedMLX session on a Mac."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import medmlx
from medmlx import memory
from tests.conftest import add_distribution

_HUGE = '''
HUGE = {"id": "huge", "label": "Huge", "task": "testing", "run": "fakemlx:sleep",
        "requires_weights": False, "peak_memory_gb": 1000000}
'''

_MEASURED = '''
MEASURED = {"id": "measured", "label": "Measured", "task": "testing", "run": "fakemlx:measure",
            "requires_weights": False, "peak_memory_gb": 8}
RESIDENT = {"id": "resident", "label": "Resident", "task": "testing", "run": "fakemlx:resident",
            "requires_weights": False}


def measure(peak_bytes: int, *, fail: bool = False, seconds: float = 0) -> dict[str, object]:
    import mlx.core as mx
    mx.peak_bytes = peak_bytes
    mx.get_active_memory = lambda: peak_bytes // 2
    mx.get_cache_memory = lambda: peak_bytes // 4
    time.sleep(seconds)
    if fail:
        raise ValueError("measurement run failed")
    return {"done": True}


def resident(*, owner: str, allocation_bytes: int) -> dict[str, object]:
    import subprocess
    import sys
    import mlx.core as mx
    mx.peak_bytes = 0
    if owner == "worker":
        data = bytearray(allocation_bytes)
        for offset in range(0, len(data), 4096):
            data[offset] = 1
        time.sleep(0.3)
        del data
    else:
        # A bridge starts another process, like an adapter delegating inference.
        # Both descendants exit before the runner returns, so final RSS misses it.
        leaf = (
            f"data = bytearray({allocation_bytes}); "
            "data[::4096] = b'x' * len(data[::4096]); "
            "print('ready', flush=True); input()"
        )
        bridge = (
            "import subprocess, sys; "
            f"child = subprocess.Popen([sys.executable, '-c', {leaf!r}], "
            "stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True); "
            "print(child.stdout.readline().strip(), flush=True); input(); "
            "child.communicate('done' + chr(10), timeout=5)"
        )
        with subprocess.Popen([sys.executable, "-c", bridge], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as child:
            try:
                if child.stdout.readline().strip() != "ready":
                    raise RuntimeError("bridge did not become ready")
                time.sleep(0.3)
                child.communicate("done\\n", timeout=5)
                if child.returncode:
                    raise RuntimeError("bridge failed")
            finally:
                if child.poll() is None:
                    child.kill()
    return {"done": True}
'''


@pytest.fixture
def measured_model(model_site: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    add_distribution(model_site, "measured-mlx", "[medmlx.models]\n"
                     "measured = fakemlx:MEASURED\nresident = fakemlx:RESIDENT\n")
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + _MEASURED)
    monkeypatch.setattr(memory, "available_bytes", lambda: 10_000_000_000)
    return model_site / "measured_mlx-1.2.3.dist-info" / "METADATA"


def test_successful_workers_record_the_highest_mlx_peak(
    measured_model: Path, tmp_path: Path, peak_store: Path,
) -> None:
    for peak_bytes, measured, highest in [
        (2_500_000_000, 2.5, 2.5), (1_000_000_000, 1.0, 2.5), (3_000_000_000, 3.0, 3.0),
    ]:
        job = medmlx.run("measured", arguments={"peak_bytes": peak_bytes}, root=tmp_path).job
        assert job.get("measured_peak_memory_gb") == measured
        metrics = job.get("memory_metrics")
        assert metrics is not None
        assert metrics.get("mlx_peak_bytes") == peak_bytes
        assert metrics.get("mlx_active_bytes") == peak_bytes // 2
        assert metrics.get("mlx_cache_bytes") == peak_bytes // 4
        assert job.get("result") == {"done": True}
        assert medmlx.describe_model("measured")["measured_peak_memory_gb"] == highest
    with pytest.raises(RuntimeError, match="measurement run failed"):
        medmlx.run("measured", arguments={"peak_bytes": 9_000_000_000, "fail": True}, root=tmp_path)
    assert json.loads(peak_store.read_text()) == {"measured": {"1.2.3": 3.0}}


@pytest.mark.parametrize("owner", ["worker", "bridge"])
def test_worker_reports_host_and_exited_bridge_allocations_separately_from_mlx(
    measured_model: Path, tmp_path: Path, peak_store: Path, owner: str,
) -> None:
    allocation_bytes = 64 * 1024**2
    job = medmlx.run("resident", arguments={"owner": owner, "allocation_bytes": allocation_bytes},
                     root=tmp_path).job
    metrics = job.get("memory_metrics")
    assert metrics is not None
    assert job.get("result") == {"done": True}
    assert job.get("measured_peak_memory_gb") == 0
    assert metrics.get("mlx_peak_bytes") == 0
    assert metrics["scope"] == "runner_import_through_return"
    assert metrics["sample_count"] >= 3
    # This small host workload also detects Darwin-bytes/Linux-KiB unit errors.
    assert 0 < metrics["worker_peak_rss_bytes"] < 2 * 1024**3
    if owner == "worker":
        assert metrics["worker_peak_rss_bytes"] >= allocation_bytes
        worker_peak = metrics["sampled_worker_rss_peak_bytes"]
        assert worker_peak is not None and worker_peak >= allocation_bytes
    else:
        descendant_peak = metrics["sampled_descendants_rss_sum_peak_bytes"]
        assert descendant_peak is not None and descendant_peak >= allocation_bytes
        assert metrics["sampled_process_count_peak"] >= 3
    # Host residency must not change the historical MLX-only admission cache.
    assert json.loads(peak_store.read_text()) == {"resident": {"1.2.3": 0.0}}


def test_peak_updates_preserve_maxima_across_processes(peak_store: Path) -> None:
    script = """
import sys
from medmlx.memory import record_peak
print("ready", flush=True)
sys.stdin.readline()
for value in range(20):
    record_peak("shared", "1.0", float(value))
    record_peak(sys.argv[1], "2.0", float(value))
"""
    writers = [subprocess.Popen(
        [sys.executable, "-c", script, model_id], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for model_id in ("first", "second", "third")]
    try:
        for writer in writers:
            assert writer.stdout is not None and writer.stdout.readline() == "ready\n"
        for writer in writers:
            assert writer.stdin is not None
            writer.stdin.write("go\n")
            writer.stdin.flush()
        for writer in writers:
            _, error = writer.communicate(timeout=10)
            assert writer.returncode == 0, error
    finally:
        for writer in writers:
            if writer.poll() is None:
                writer.kill()
            writer.communicate(timeout=5)
    assert json.loads(peak_store.read_text()) == {
        "shared": {"1.0": 19.0}, "first": {"2.0": 19.0},
        "second": {"2.0": 19.0}, "third": {"2.0": 19.0},
    }


def test_guard_uses_measurements_only_for_the_current_package_version(
    measured_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory.record_peak("measured", "1.2.3", 3.0)
    expected = {
        "peak_memory_gb": 8.0, "measured_peak_memory_gb": 3.0,
        "effective_peak_memory_gb": 8.0, "peak_memory_source": "declared",
    }
    description = medmlx.describe_model("measured")
    assert {key: value for key, value in description.items() if key in expected} == expected
    row = medmlx.model_readiness(["measured"], arguments={"peak_bytes": 0})["models"][0]
    assert row["memory"] == {**expected, "available_gb": 10.0, "fits": True, "other_runs": []}
    assert row["run_ready"] is True
    with medmlx.start(
        "measured", arguments={"peak_bytes": 0, "seconds": 30}, root=tmp_path,
    ):
        active = medmlx.memory_status()["active_runs"][0]
        assert {key: value for key, value in active.items() if key in expected} == expected

    memory.record_peak("measured", "1.2.3", 9.0)
    monkeypatch.setattr(memory, "available_bytes", lambda: 8_500_000_000)
    row = medmlx.model_readiness(["measured"], arguments={"peak_bytes": 0})["models"][0]
    assert row["memory"]["fits"] is False
    assert "Needs about 9.0 GB at peak" in str(row["reason"])
    # Memory is reported, not enforced: the run is neither marked blocked nor refused.
    assert row["run_ready"] is True
    assert medmlx.run("measured", arguments={"peak_bytes": 0}, root=tmp_path).job["state"] == (
        "completed")

    measured_model.write_text(
        measured_model.read_text().replace("Version: 1.2.3", "Version: 1.2.4"))
    description = medmlx.describe_model("measured")
    assert description["version"] == "1.2.4"
    assert description["measured_peak_memory_gb"] is None
    assert description["effective_peak_memory_gb"] == 8.0
    assert description["peak_memory_source"] == "declared"
    job = medmlx.run("measured", arguments={"peak_bytes": 0}, root=tmp_path).job
    assert job["state"] == "completed"
    assert medmlx.describe_model("measured")["measured_peak_memory_gb"] == 0.0


@pytest.mark.parametrize("contents", [b"{broken", b'{"measured": []}', b"\xff"])
def test_corrupt_peak_store_falls_back_and_can_be_replaced(
    measured_model: Path, peak_store: Path, contents: bytes,
) -> None:
    peak_store.write_bytes(contents)
    description = medmlx.describe_model("measured")
    assert description["measured_peak_memory_gb"] is None
    assert description["effective_peak_memory_gb"] == 8.0
    assert description["peak_memory_source"] == "declared"
    memory.record_peak("measured", "1.2.3", 3.0)
    assert json.loads(peak_store.read_text()) == {"measured": {"1.2.3": 3.0}}


def test_one_model_holds_memory_per_machine(
    model_site: Path, tmp_path: Path, run_registry: Path,
) -> None:
    add_distribution(model_site, "huge-mlx", "[medmlx.models]\nhuge = fakemlx:HUGE\n")
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + _HUGE)
    first = medmlx.start("sleeper", arguments={"seconds": 30}, root=tmp_path)
    assert [run["model_id"] for run in medmlx.memory_status()["active_runs"]] == ["sleeper"]
    # A second session sees the first one's run and is refused before loading anything.
    with pytest.raises(RuntimeError, match="already running on this Mac"):
        medmlx.start("sleeper", arguments={"seconds": 0}, root=tmp_path)
    first.cancel()
    assert medmlx.memory_status()["active_runs"] == []
    # A record whose worker died without cleanup no longer holds memory.
    (run_registry / "dead.lock").touch()
    (run_registry / "dead.json").write_text('{"model_id": "dead"}')
    assert medmlx.memory_status()["active_runs"] == []
    assert not (run_registry / "dead.json").exists()
    assert medmlx.start("sleeper", arguments={"seconds": 0}, root=tmp_path).wait().job["state"] == (
        "completed")

    row = medmlx.model_readiness(["huge"], arguments={"seconds": 0}, root=tmp_path)["models"][0]
    assert row["memory"]["fits"] is False and "GB at peak" in str(row["reason"])
    assert row["run_ready"] is not False
    assert medmlx.start("huge", arguments={"seconds": 0}, root=tmp_path).wait().job["state"] == (
        "completed")


def test_active_runs_never_miss_a_run_while_its_pid_is_recorded(run_registry: Path) -> None:
    script = """
import sys
from medmlx.memory import record_pid
print("ready", flush=True)
sys.stdin.readline()
for pid in range(2000):
    record_pid("busy", pid)
"""
    with memory.admit("busy", "sleeper", None):
        writer = subprocess.Popen(
            [sys.executable, "-c", script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            assert writer.stdout is not None and writer.stdout.readline() == "ready\n"
            assert writer.stdin is not None
            writer.stdin.write("go\n")
            writer.stdin.flush()
            reads = 0
            while writer.poll() is None:
                assert [run["job_id"] for run in memory.active_runs()] == ["busy"]
                reads += 1
            _, error = writer.communicate(timeout=10)
            assert writer.returncode == 0, error
            assert reads > 0
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.communicate(timeout=5)
        assert memory.active_runs()[0]["pid"] == 1999
