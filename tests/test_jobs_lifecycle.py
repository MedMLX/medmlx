"""Session shutdown owns workers that are still being registered."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medmlx import jobs as jobs_module
from medmlx.jobs import Job, Jobs

_RUNNER = """
import time
from pathlib import Path

def run(*, ready_path):
    Path(ready_path).touch()
    time.sleep(60)
    return {}
"""


def test_closed_session_rejects_new_workers(tmp_path: Path) -> None:
    jobs = Jobs(tmp_path)
    jobs.close()
    with pytest.raises(RuntimeError):
        jobs.start("sleeper", "session_runner:run", {})


def test_close_waits_for_starting_worker_and_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause registration after a real worker starts, then close the session."""
    (tmp_path / "session_runner.py").write_text(_RUNNER, encoding="utf-8")
    ready = tmp_path / "ready"
    registered, release, closing, closed = (threading.Event() for _ in range(4))
    observed: list[Job] = []
    jobs = Jobs(tmp_path)

    def paused_job(*, job_id: str, model_id: str, control_dir: Path,
                   process: subprocess.Popen[bytes], started_at: str,
                   version: str | None = None) -> Job:
        job = Job(job_id, model_id, control_dir, process, started_at, version=version)
        observed.append(job)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), job.to_payload()
        registered.set()
        assert release.wait(timeout=5), "registration was not released"
        return job

    def close_session() -> None:
        closing.set()
        jobs.close()
        closed.set()

    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        (str(tmp_path), str(source), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setattr(jobs_module, "Job", paused_job)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            started = executor.submit(
                jobs.start, "sleeper", "session_runner:run", {"ready_path": str(ready)})
            try:
                assert registered.wait(timeout=5), "worker did not start"
                finished = executor.submit(close_session)
                assert closing.wait(timeout=5), "shutdown did not start"
                assert not closed.wait(timeout=0.1), "shutdown lost a starting worker"
            finally:
                release.set()
            job = started.result(timeout=5)
            finished.result(timeout=10)
            assert job.process.poll() is not None
            assert job.state == "cancelled"
            assert not job.control_dir.exists()
    finally:
        release.set()
        for job in observed:
            if job.process.poll() is None:
                job.process.kill()
            job.process.wait(timeout=5)
        jobs.close()
