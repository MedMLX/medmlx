"""Model runs in supervised worker processes, owned by one action or MCP session."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from uuid import uuid4

from medmlx_core.outputs import validate_outputs

from medmlx import memory
from medmlx.payloads import JobPayload, JobState, JsonValue, WorkerResponse

_TERMINATE_GRACE_SECONDS = 5.0
_LOG_TAIL_BYTES = 4000

# Workers execute third-party model packages. Preserve only the process, cache,
# network, and runtime settings those packages need instead of exposing every
# ambient credential from the MCP server. This narrows credential exposure; it
# is not a filesystem or network sandbox.
_WORKER_ENVIRONMENT_NAMES = frozenset({
    "ALL_PROXY",
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LOGNAME",
    "LOG_LEVEL",
    "NO_COLOR",
    "NO_PROXY",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONUTF8",
    "PYTHONWARNINGS",
    "RANK",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "TOKENIZERS_PARALLELISM",
    "TZ",
    "USER",
    "VECLIB_MAXIMUM_THREADS",
    "VIRTUAL_ENV",
    "WORLD_SIZE",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
})
_WORKER_ENVIRONMENT_PREFIXES = (
    "HF_",
    "HUGGING_FACE_",
    "LC_",
    "LESIONDIFFUSION_",
    "MEDMLX_",
    "MLX_",
    "TRANSFORMERS_",
    "XDG_",
)


def _worker_environment(*, control_dir: Path, cwd: Path) -> dict[str, str]:
    """Build the supported environment boundary for a model subprocess."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _WORKER_ENVIRONMENT_NAMES
        or name.startswith(_WORKER_ENVIRONMENT_PREFIXES)
    }
    environment.update({
        "MEDMLX_WORKER_CONTROL_DIR": str(control_dir),
        "PWD": str(cwd),
        "PYTHONFAULTHANDLER": "1",
    })
    return environment


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
    version: str | None = None
    outputs_error: str | None = None

    def to_payload(self) -> JobPayload:
        """Return the JSON-serializable job record sent to API and MCP clients.

        ``log_tail`` is the end of the worker's combined stdout and stderr (at
        most 4000 bytes). Once a worker has stopped, its response supplies either
        ``result`` or ``error``. Both are absent while running and after a
        cancellation that left no response; a cancelled job can still carry a
        response written before it stopped.
        Successful responses include ``measured_peak_memory_gb`` when MLX was
        loaded in the worker; this measures MLX allocations only. Separate
        ``memory_metrics`` report worker/descendant RSS and named MLX counters.
        ``outputs_error`` reports invalid output metadata without changing the
        result or failing the run.
        """
        # Finalization publishes terminal state last; use one state snapshot so
        # a concurrent completion cannot expose a response on a running job.
        state = self.state
        payload: JobPayload = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "state": state,
            "pid": self.process.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at if state != "running" else None,
            "exit_code": self.exit_code if state != "running" else None,
            "log_tail": _tail(self.control_dir / "worker.log"),
        }
        if state != "running" and self.response is not None:
            if self.response.get("ok"):
                payload["result"] = self.response.get("result")
                if "measured_peak_memory_gb" in self.response:
                    payload["measured_peak_memory_gb"] = self.response["measured_peak_memory_gb"]
                if "memory_metrics" in self.response:
                    payload["memory_metrics"] = self.response["memory_metrics"]
                if self.outputs_error is not None:
                    payload["outputs_error"] = self.outputs_error
            else:
                payload["error"] = self.response.get("error")
        return payload


class Jobs:
    """Start, observe, and stop worker processes for one action or server session.

    Each run executes in ``python -m medmlx.worker`` with the server root
    as its working directory. Model execution has separate process state;
    relative paths resolve against the root. At most one worker runs at a time
    so concurrent model loads cannot exhaust Apple unified memory. Jobs belong
    to this session; ``close`` cancels active workers and removes their control
    files.
    """

    def __init__(self, root: Path) -> None:
        """Create an empty session rooted at the server directory."""

        self.root = root
        self._control = TemporaryDirectory(prefix="medmlx-")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._closed = False

    def start(
        self, model_id: str, run: str, arguments: dict[str, JsonValue], *,
        holds_memory: bool = True, peak_memory_gb: float | None = None,
        version: str | None = None,
    ) -> Job:
        """Start a worker for one run call and return without waiting.

        The caller owns argument and runtime validation. Worker exit is finalized
        automatically, even if the client never polls.
        The job may finish before the caller observes it. A closed session or
        an existing active run rejects new work before creating files or processes.
        MEDMLX_WORKER_CONTROL_DIR gives adapters a job-owned location for helper
        control files, which remains removable when a worker is killed.
        A run that ``holds_memory`` (every model run) must also be admitted
        machine-wide by ``medmlx.memory``, which refuses it while another
        session's model holds memory.
        ``version`` identifies the declaring package version for peak measurements.
        """

        # Shutdown must see every launched worker before removing its files.
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start a worker in a closed session")
            active = next((job for job in self._jobs.values() if job.state == "running"), None)
            if active is not None:
                raise RuntimeError(
                    f"MedMLX already has an active model run ({active.job_id}); "
                    "wait for it to finish or cancel it before starting another."
                )
            return self._start(model_id, run, arguments, holds_memory, peak_memory_gb, version)

    def _start(
        self, model_id: str, run: str, arguments: dict[str, JsonValue], holds_memory: bool,
        peak_memory_gb: float | None, version: str | None,
    ) -> Job:
        """Launch and register a worker while the session lock is held."""

        job_id = uuid4().hex[:12]
        if not holds_memory:
            return self._launch(job_id, model_id, run, arguments, None, None)
        with memory.admit(job_id, model_id, peak_memory_gb, version=version) as lock:
            job = self._launch(job_id, model_id, run, arguments, lock, version)
        memory.record_pid(job_id, job.process.pid)
        return job

    def _launch(
        self, job_id: str, model_id: str, run: str, arguments: dict[str, JsonValue],
        lock: int | None, version: str | None,
    ) -> Job:
        """Start the worker; it inherits ``lock`` so the run's memory claim ends with it."""

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
                env=_worker_environment(control_dir=control_dir, cwd=self.root),
                start_new_session=os.name != "nt",
                pass_fds=() if lock is None else (lock,),
            )
        job = Job(
            job_id=job_id,
            model_id=model_id,
            control_dir=control_dir,
            process=process,
            started_at=_now(),
            version=version,
        )
        self._jobs[job_id] = job
        threading.Thread(target=self._wait, args=(job,), daemon=True).start()
        return job

    def _wait(self, job: Job) -> None:
        """Finalize worker exit even when the client never polls the job."""
        self._finalize(job, job.process.wait())

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
        # Finalization and cancellation must agree on the terminal state and
        # signal only a group that still belongs to an unfinished job.
        with self._lock:
            if job.state != "running":
                return job
            job.cancel_requested = True
            _stop(job.process, kill=False)
        try:
            code = job.process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with self._lock:
                if job.state == "running":
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
            memory.release(job.job_id)
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
                result = job.response.get("result")
                if isinstance(result, dict) and "outputs" in result:
                    try:
                        validate_outputs(result["outputs"])
                    except ValueError as error:
                        job.outputs_error = str(error)
                measured = job.response.get("measured_peak_memory_gb")
                if measured is not None:
                    # An unwritable measurement cache must not strand a successful job.
                    with suppress(OSError):
                        memory.record_peak(job.model_id, job.version, measured)
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
            try:
                os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            except PermissionError:
                # Darwin's killpg returns EPERM when a group contains only zombies:
                # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_sig.c
                if sys.platform != "darwin" or not _group_stopped(process.pid):
                    raise
    except ProcessLookupError:
        return


def _group_stopped(group: int) -> bool:
    """Confirm a Darwin process group is absent or contains only zombies."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-g", str(group), "-o", "stat="],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.stderr.strip():
        return False
    states = result.stdout.split()
    if not states:
        return result.returncode == 1
    return result.returncode == 0 and all(state.startswith("Z") for state in states)


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
