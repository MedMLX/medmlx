"""Session shutdown owns workers that are still being registered."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medmlx.jobs import Job, Jobs

_RUNNER = """
import time
from pathlib import Path

def run(*, ready_path):
    Path(ready_path).touch()
    time.sleep(60)
    return {}
"""


class JobLifecycleTests(unittest.TestCase):
    def test_closed_session_rejects_new_workers(self) -> None:
        with TemporaryDirectory() as directory:
            jobs = Jobs(Path(directory))
            jobs.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                jobs.start("sleeper", "session_runner:run", {})
            jobs.close()

    def test_close_waits_for_starting_worker_and_stops_it(self) -> None:
        """Pause registration after a real worker starts, then close the session."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session_runner.py").write_text(_RUNNER, encoding="utf-8")
            ready = root / "ready"
            registered = threading.Event()
            release = threading.Event()
            closing = threading.Event()
            closed = threading.Event()
            observed: list[Job] = []
            jobs = Jobs(root)

            def paused_job(
                *,
                job_id: str,
                model_id: str,
                control_dir: Path,
                process: subprocess.Popen[bytes],
                started_at: str,
            ) -> Job:
                job = Job(job_id, model_id, control_dir, process, started_at)
                observed.append(job)
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), job.to_payload())
                registered.set()
                self.assertTrue(release.wait(timeout=5), "registration was not released")
                return job

            def close_session() -> None:
                closing.set()
                jobs.close()
                closed.set()

            source = Path(__file__).resolve().parents[1] / "src"
            pythonpath = os.pathsep.join((str(root), str(source), os.environ.get("PYTHONPATH", "")))
            try:
                with (
                    patch.dict(os.environ, {"PYTHONPATH": pythonpath}),
                    patch("medmlx.jobs.Job", side_effect=paused_job),
                    ThreadPoolExecutor(max_workers=2) as executor,
                ):
                    started = executor.submit(
                        jobs.start, "sleeper", "session_runner:run", {"ready_path": str(ready)}
                    )
                    try:
                        self.assertTrue(registered.wait(timeout=5), "worker did not start")
                        finished = executor.submit(close_session)
                        self.assertTrue(closing.wait(timeout=5), "shutdown did not start")
                        self.assertFalse(
                            closed.wait(timeout=0.1), "shutdown lost a starting worker"
                        )
                    finally:
                        release.set()
                    job = started.result(timeout=5)
                    finished.result(timeout=10)
                    self.assertIsNotNone(job.process.poll())
                    self.assertEqual(job.state, "cancelled")
                    self.assertFalse(job.control_dir.exists())
            finally:
                release.set()
                for job in observed:
                    if job.process.poll() is None:
                        job.process.kill()
                    job.process.wait(timeout=5)
                jobs.close()


if __name__ == "__main__":
    unittest.main()
