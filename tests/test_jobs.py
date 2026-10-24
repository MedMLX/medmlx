"""Subprocesses cannot keep running after their owning session ends."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from medmlx.jobs import Jobs

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


def _stopped(pid: int) -> bool:
    state = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    ).stdout.strip()
    # An orphaned zombie has terminated and awaits the OS reaper.
    return not state or state.startswith("Z")


def _kill_child(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group ownership")
@pytest.mark.parametrize("outcome", ["completed", "crash", "running"])
def test_session_close_stops_children_even_after_worker_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    (tmp_path / "child_runner.py").write_text(_RUNNER)
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    pid_path = tmp_path / "child.pid"
    jobs = Jobs(tmp_path)
    try:
        job = jobs.start(
            "child-test", "child_runner:run", {"pid_path": str(pid_path), "outcome": outcome}
        )
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_path.exists(), job.to_payload()
        child_pid = int(pid_path.read_text())
        if outcome != "running":
            job.process.wait(timeout=5)
        jobs.close()
        deadline = time.monotonic() + 5
        while not _stopped(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _stopped(child_pid), "worker child outlived its MCP session"
    finally:
        jobs.close()
        if pid_path.exists():
            _kill_child(int(pid_path.read_text()))
