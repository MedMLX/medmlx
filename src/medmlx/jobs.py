"""Model runs in supervised worker processes, owned by one action or MCP session."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from uuid import uuid4

from medmlx.payloads import JobPayload, JobState, JsonValue, WorkerResponse

_TERMINATE_GRACE_SECONDS = 5.0
_LOG_TAIL_BYTES = 4000


@dataclass
class Job:
    """One worker process and what is known about its outcome."""

    job_id: str
    model_id: str
    control_dir: Path
    process: subprocess.Popen[bytes]
    started_at: str
    state: JobState = "running"
    exit_code: int | None = None
    finished_at: str | None = None
    response: WorkerResponse | None = None
    cancel_requested: bool = False

    def to_payload(self) -> JobPayload:
        """Return the JSON-serializable job record sent to API and MCP clients.

        ``log_tail`` is the end of the worker's combined stdout and stderr (at
        most 4000 bytes). ``result`` appears only after a successful run and
        ``error`` only after a failed one; both are absent while running and
        after a cancellation that left no response.
        """
        payload: JobPayload = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "state": self.state,
            "pid": self.process.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_tail": _tail(self.control_dir / "worker.log"),
        }
        if self.response is not None:
            if self.response.get("ok"):
                payload["result"] = self.response.get("result")
            else:
                payload["error"] = self.response.get("error")
        return payload


class Jobs:
    """Start, observe, and stop worker processes for one action or server session.

    Each run executes in ``python -m medmlx.worker`` with the server root
    as its working directory. Model execution has separate process state;
    relative paths resolve against the root. Jobs belong to this session;
    ``close`` cancels active workers and removes their control files.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._control = TemporaryDirectory(prefix="medmlx-")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(self, model_id: str, run: str, arguments: dict[str, JsonValue]) -> Job:
        """Start a worker for one validated run call and return without waiting.

        The returned job is initially running. Call ``get`` or ``all`` to
        collect the worker's completion status and response after it exits.
        A closed session rejects new runs before creating files or processes.
        """

        # Shutdown must see every launched worker before removing its files.
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start a worker in a closed session")
            return self._start(model_id, run, arguments)

    def _start(self, model_id: str, run: str, arguments: dict[str, JsonValue]) -> Job:
        """Launch and register a worker while the session lock is held."""

        job_id = uuid4().hex[:12]
        control_dir = Path(self._control.name) / job_id
        control_dir.mkdir()
        request_path = control_dir / "request.json"
        request_path.write_text(
            json.dumps({"run": run, "arguments": arguments}, allow_nan=False), encoding="utf-8"
        )
        with (control_dir / "worker.log").open("wb") as log:
            # The worker must never touch the server's stdio: it carries the MCP wire.
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "medmlx.worker",
                    str(request_path),
                    str(control_dir / "response.json"),
                ],
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # A native crash in MLX or a model then leaves a traceback in worker.log.
                env={**os.environ, "PYTHONFAULTHANDLER": "1"},
                start_new_session=os.name != "nt",
            )
        job = Job(
            job_id=job_id,
            model_id=model_id,
            control_dir=control_dir,
            process=process,
            started_at=_now(),
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        """Return one job, finalizing it if its worker has exited."""

        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise LookupError(f"Unknown job_id {job_id!r}; jobs last only for this MCP session")
        code = job.process.poll()
        if code is not None:
            self._finalize(job, code)
        return job

    def all(self) -> list[Job]:
        """Return every job started in this session, oldest first."""

        with self._lock:
            job_ids = list(self._jobs)
        return [self.get(job_id) for job_id in job_ids]

    def cancel(self, job_id: str) -> Job:
        """Terminate a running worker, escalating to a kill if it does not stop."""

        job = self.get(job_id)
        if job.state != "running":
            return job
        job.cancel_requested = True
        _stop(job.process, kill=False)
        try:
            code = job.process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _stop(job.process, kill=True)
            code = job.process.wait()
        self._finalize(job, code)
        return job

    def close(self) -> None:
        """Cancel running workers and remove session control files."""

        with self._lock:
            self._closed = True
        for job in self.all():
            self.cancel(job.job_id)
        self._control.cleanup()

    def _finalize(self, job: Job, code: int) -> None:
        with self._lock:
            if job.state != "running":
                return
            # The leader can exit while its descendants still own resources or
            # output files. No subprocess may outlive this session's job, even
            # when it ignored SIGTERM or the worker crashed before reporting.
            _stop(job.process, kill=True)
            response_path = job.control_dir / "response.json"
            if response_path.is_file():
                # The worker owns this packet's serialization and schema.
                job.response = cast(
                    WorkerResponse, json.loads(response_path.read_text(encoding="utf-8"))
                )
            job.exit_code = code
            job.finished_at = _now()
            if job.cancel_requested:
                job.state = "cancelled"
            elif code == 0 and job.response is not None and job.response.get("ok"):
                job.state = "completed"
            else:
                job.state = "failed"
                if job.response is None:
                    job.response = {
                        "ok": False,
                        "error": {
                            "type": "WorkerExited",
                            "message": f"Worker exited with code {code} before reporting",
                        },
                    }


def _stop(process: subprocess.Popen[bytes], *, kill: bool) -> None:
    """Signal the worker's process group so subprocesses it started stop with it."""

    try:
        if os.name == "nt":
            process.kill() if kill else process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
    except ProcessLookupError:
        return


def _tail(path: Path) -> str:
    """Return the last ``_LOG_TAIL_BYTES`` of a log (invalid UTF-8 replaced); empty if absent."""
    try:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - _LOG_TAIL_BYTES))
            return log.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
