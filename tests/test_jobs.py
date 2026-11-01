"""Worker subprocesses remain owned and receive only their runtime environment."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from medmlx.jobs import Jobs
from tests.conftest import BROKEN_MLX, set_mlx

_RUNNER = """
import os
import subprocess
import sys
import time
from pathlib import Path

def run(*, pid_path, outcome):
    ready = Path(pid_path).with_suffix(".ready")
    child = subprocess.Popen([sys.executable, "-c", (
        "import signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )])
    deadline = time.monotonic() + 5
    while not ready.exists():
        if time.monotonic() >= deadline:
            child.kill()
            raise RuntimeError("child did not start")
        time.sleep(0.01)
    Path(pid_path).write_text(str(child.pid))
    if outcome == "crash":
        os._exit(3)
    if outcome == "running":
        time.sleep(60)
    return {}
"""

_ENVIRONMENT_RUNNER = """
import os

def run(*, allowed, denied):
    return {
        "allowed": os.environ.get(allowed),
        "denied_present": denied in os.environ,
        "control_configured": bool(os.environ.get("MEDMLX_WORKER_CONTROL_DIR")),
    }
"""

_ZOMBIE_SUPERVISOR = """
import os
import subprocess
import sys
import time
from pathlib import Path

os.setpgid(0, 0)
Path(sys.argv[2]).write_text(str(os.getpid()))
group = int(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "pass"],
                         preexec_fn=lambda: os.setpgid(0, group))
deadline = time.monotonic() + 5
while True:
    state = subprocess.run(["ps", "-p", str(child.pid), "-o", "stat="],
                           capture_output=True, text=True, check=False).stdout.strip()
    if state.startswith("Z"):
        break
    if time.monotonic() >= deadline:
        raise RuntimeError("child did not become a zombie")
    time.sleep(0.01)
Path(sys.argv[3]).write_text(str(child.pid))
time.sleep(60)
"""

_ZOMBIE_RUNNER = """
import os
import subprocess
import sys
import time
from pathlib import Path

def run(*, supervisor_script, supervisor_pid, zombie_pid):
    subprocess.Popen([sys.executable, supervisor_script, str(os.getpgrp()),
                      supervisor_pid, zombie_pid])
    deadline = time.monotonic() + 5
    while not Path(zombie_pid).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("zombie child did not start")
        time.sleep(0.01)
    return {}
"""


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")


def _wait(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def _state(pid: int) -> str:
    return subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                          capture_output=True, text=True, check=False, timeout=5).stdout.strip()


def _stopped(pid: int) -> bool:
    state = _state(pid)
    # An orphaned zombie has terminated and awaits the OS reaper.
    return not state or state.startswith("Z")


def _kill(path: Path) -> None:
    if path.exists():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(path.read_text()), signal.SIGKILL)


def test_worker_receives_runtime_configuration_without_unrelated_secrets(
    model_site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_mlx(model_site, BROKEN_MLX)
    _install(tmp_path, monkeypatch, "environment_runner", _ENVIRONMENT_RUNNER)
    monkeypatch.setenv("MEDMLX_TEST_SETTING", "enabled")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-worker")
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start("environment-test", "environment_runner:run",
                         {"allowed": "MEDMLX_TEST_SETTING", "denied": "OPENAI_API_KEY"})
        job.process.wait(timeout=5)
        payload = jobs.get(job.job_id).to_payload()
        assert payload["state"] == "completed", payload["log_tail"]
        assert payload.get("result") == {
            "allowed": "enabled", "denied_present": False, "control_configured": True,
        }
        assert "measured_peak_memory_gb" not in payload
    finally:
        jobs.close()


def test_invalid_outputs_are_reported_without_changing_a_successful_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(tmp_path, monkeypatch, "outputs_runner", '''
def run():
    return {"outputs": {"mask": "mask.nii.gz"}, "details": {"voxels": 42}}
''')
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start("outputs-test", "outputs_runner:run", {}, holds_memory=False)
        job.process.wait(timeout=5)
        payload = jobs.get(job.job_id).to_payload()
        assert payload["state"] == "completed", payload["log_tail"]
        assert payload.get("outputs_error") == "outputs must be a list"
        assert payload.get("result") == {
            "outputs": {"mask": "mask.nii.gz"}, "details": {"voxels": 42},
        }
    finally:
        jobs.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group ownership")
@pytest.mark.parametrize(
    ("outcome", "close_session"), [("completed", False), ("crash", False), ("running", True)],
)
def test_worker_exit_or_session_close_stops_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, close_session: bool,
) -> None:
    _install(tmp_path, monkeypatch, "child_runner", _RUNNER)
    pid_path = tmp_path / "child.pid"
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start("child-test", "child_runner:run",
                         {"pid_path": str(pid_path), "outcome": outcome})
        assert _wait(pid_path.exists), job.to_payload()
        child_pid = int(pid_path.read_text())
        if close_session:
            jobs.close()
        else:
            job.process.wait(timeout=5)
        assert _wait(lambda: _stopped(child_pid)), "worker child outlived its job or session"
        if not close_session:
            assert _wait(lambda: job.state != "running")
            assert job.state == ("completed" if outcome == "completed" else "failed")
    finally:
        jobs.close()
        _kill(pid_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-only group semantics")
def test_zombie_only_group_finalizes_without_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(tmp_path, monkeypatch, "zombie_runner", _ZOMBIE_RUNNER)
    supervisor_script = tmp_path / "zombie_supervisor.py"
    supervisor_script.write_text(_ZOMBIE_SUPERVISOR)
    supervisor_pid, zombie_pid = tmp_path / "supervisor.pid", tmp_path / "zombie.pid"
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start("zombie-test", "zombie_runner:run", {
            "supervisor_script": str(supervisor_script),
            "supervisor_pid": str(supervisor_pid), "zombie_pid": str(zombie_pid),
        })
        assert _wait(zombie_pid.exists), job.to_payload()
        assert _state(int(zombie_pid.read_text())).startswith("Z")
        assert _wait(lambda: job.state != "running")
        assert job.state == "completed", job.to_payload()
        assert not _stopped(int(supervisor_pid.read_text()))
    finally:
        try:
            jobs.close()
        finally:
            _kill(supervisor_pid)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin group inspection")
def test_group_signal_denial_for_a_live_group_remains_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(tmp_path, monkeypatch, "child_runner", _RUNNER)
    pid_path = tmp_path / "child.pid"
    jobs = Jobs(tmp_path)

    def denied(group: int, sig: int) -> None:
        raise PermissionError("original signal denial")

    try:
        job = jobs.start("child-test", "child_runner:run",
                         {"pid_path": str(pid_path), "outcome": "running"})
        assert _wait(pid_path.exists), job.to_payload()
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", denied)
            with pytest.raises(PermissionError):
                jobs.cancel(job.job_id)
        assert job.process.poll() is None
    finally:
        jobs.close()
        _kill(pid_path)
