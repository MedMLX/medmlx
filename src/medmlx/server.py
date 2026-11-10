"""MCP server for discovered MedMLX models: listing, readiness, and supervised runs."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

import anyio
from anyio import to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from medmlx import __version__, api, outputs
from medmlx.jobs import Jobs
from medmlx.payloads import (
    CheckpointListPayload,
    ConvertedSeries,
    DicomSeries,
    JobListPayload,
    JobPayload,
    JsonValue,
    MemoryStatus,
    ModelDescription,
    ModelListPayload,
    OutputSummary,
    PackageListPayload,
    ReadinessPayload,
    RuntimeToolPayload,
)
from medmlx.probe import runtime_info as probe_runtime

_MAX_WAIT_SECONDS = 60.0
_POLL_SECONDS = 0.25

_INSTRUCTIONS = """\
MedMLX runs medical imaging models natively on this Mac with MLX and Metal. It is
research software: outputs require human review and are not for clinical use.
Workflow: list_models -> describe_model (parameters, example_arguments, labels and
the provider's instructions) -> convert_dicom when the input is a DICOM folder
(list_dicom_series first if it holds several series) -> prepare_model when weights
are not cached -> model_readiness with the intended arguments -> run_model -> get_job
until the state is terminal -> summarize_output and preview_output on the outputs.
summarize_output reports label volumes in mL. Pass job_id for this run's label names,
or model_id for declared names. preview_output returns a PNG and the same summary.
Both report names_from. Neither interprets findings.
model_readiness: run_ready true means every local check passed (runtime and Metal,
argument binding, input and weight paths, supplied or cached weights, NIfTI headers);
null means something could not be checked; false names the blocker in reason. No
check runs the model or proves that weights and images are compatible.
Weights: when weights_path is omitted, runs download the model's pinned weights on
first use; prepare_model downloads them ahead of time as a job. Some models need a
local weights_path; describe_model says which.
Memory: one model runs at a time on this Mac across every MedMLX session, so models
never pile up in unified memory; memory_status shows what is running. Readiness
compares the larger of a model's declared peak and its highest measured MLX peak
for the installed version with the available memory and reports it as a reason; it
never blocks a run. Measurements exclude NumPy and preprocessing memory.
Jobs: get_job waits up to 60 seconds per call; disconnecting cancels running jobs.
Paths: relative paths resolve against {root}. {paths}
Error messages carry a code, such as invalid_arguments, and a next step.
"""
_PATHS_OPEN = "This root is not a filesystem sandbox."
_PATHS_RESTRICTED = "Path arguments outside this root are rejected."

# Each error code's suggested next step for the client.
_HINTS = {
    "unknown_model": "Call list_models for the installed model IDs.",
    "unknown_job": "Call list_jobs for this session's jobs.",
    "invalid_arguments": "Call describe_model for the runner's parameter names.",
    "manual_setup": "Pass a local weights_path folder; describe_model lists the assets.",
    "busy": "Wait for the running model to finish (memory_status shows it), or cancel it.",
    "metal_unavailable": "Use an Apple Silicon Mac with MLX installed and Metal available.",
    "missing_dependency": "Install the model's family, for example medmlx[all,mcp].",
    "path_outside_root": "Use paths inside the server root.",
}

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_RUNS = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)
_CANCELS = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)
_WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
_PREPARES = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True,
)

Tool = Callable[..., object]


def _failure(code: str, message: str) -> ToolError:
    """A tool error whose text starts with a stable code and ends with a next step."""
    hint = _HINTS.get(code)
    return ToolError(f"{code}: {message}" + (f"\nNext: {hint}" if hint else ""))


@contextmanager
def _tool_errors(
    *, missing: str = "unknown_model", invalid: str = "invalid_arguments",
) -> Generator[None]:
    """Report anticipated failures to the client with a code, message and next step.

    ``missing`` names the code for a lookup failure and ``invalid`` the code for
    a rejected value in this tool.
    """

    try:
        yield
    except (LookupError, OSError, ValueError, ImportError, RuntimeError) as error:
        message = str(error) or type(error).__name__
        if isinstance(error, LookupError):
            code = missing
        elif isinstance(error, ImportError):
            code = "missing_dependency"
        elif isinstance(error, ValueError):
            code = invalid
        elif isinstance(error, OSError):
            code = "filesystem_error"
        elif "already has an active" in message or "already running on this Mac" in message:
            code = "busy"
        elif "MLX is not usable" in message:
            code = "metal_unavailable"
        else:
            code = "runtime_error"
        raise _failure(code, message) from error


def build_server(root: Path, *, restrict_paths: bool = False) -> MCPServer[Jobs]:
    """Build the stdio MCP server whose runs use ``root`` as working directory.

    With ``restrict_paths``, readiness checks and runs reject path arguments,
    outputs included, that resolve outside ``root``. Runners can still read
    their own managed caches.
    """

    root = root.resolve()

    def check_paths(arguments: dict[str, JsonValue] | None) -> None:
        if restrict_paths and arguments and (outside := api.paths_outside(arguments, root)):
            raise _failure(
                "path_outside_root", "Paths outside the server root: " + ", ".join(outside),
            )

    @asynccontextmanager
    async def lifespan(_server: MCPServer[Jobs]) -> AsyncGenerator[Jobs]:
        """Own one job supervisor per server session; on exit, cancel its running workers."""
        jobs = Jobs(root)
        try:
            yield jobs
        finally:
            await to_thread.run_sync(jobs.close)

    server: MCPServer[Jobs] = MCPServer(
        "medmlx",
        version=__version__,
        instructions=_INSTRUCTIONS.format(
            root=root, paths=_PATHS_RESTRICTED if restrict_paths else _PATHS_OPEN,
        ),
        lifespan=lifespan,
    )

    def list_packages() -> PackageListPayload:
        """Show optional runtime families, installed versions and declared models."""
        return api.list_packages()

    def list_models(task: str | None = None, family: str | None = None) -> ModelListPayload:
        """List installed models, filtered by task or segment/generate/vision family."""
        return api.list_models(task, family)

    def list_checkpoints(model_id: str | None = None) -> CheckpointListPayload:
        """List checkpoint source references and channel order; no downloads or qualification."""
        return api.list_checkpoints(model_id)

    def describe_model(model_id: str) -> ModelDescription:
        """Return a model's declaration and provider-authored guidance, not validation."""
        with _tool_errors():
            return api.describe_model(model_id)

    def model_readiness(
        model_ids: list[str] | None = None,
        arguments: dict[str, JsonValue] | None = None,
    ) -> ReadinessPayload:
        """Check runtime and optional input/weight paths for one model without running it.

        ready and runtime_ready report the same boolean imports/provider checks
        plus Metal result. run_ready=null means input/checkpoint contents remain
        unverified; false identifies a blocker. Optional arguments require exactly
        one model id and resolve paths against the server root.
        """
        check_paths(arguments)
        with _tool_errors():
            return api.model_readiness(model_ids, arguments=arguments, root=root)

    def runtime_info() -> RuntimeToolPayload:
        """Report the platform, package versions, and MLX Metal availability and device."""

        return {"root": root.as_posix(), **probe_runtime()}

    def run_model(
        ctx: Context[Jobs, object], model_id: str, arguments: dict[str, JsonValue],
    ) -> JobPayload:
        """Check argument binding and Metal, start a worker, and return its job; poll get_job."""
        check_paths(arguments)
        with _tool_errors():
            return api.start_model(_jobs(ctx), model_id, arguments)

    def prepare_model(ctx: Context[Jobs, object], model_id: str) -> JobPayload:
        """Download a model's pinned weights as a job and return it; poll get_job.

        Cached weights finish at once. The log tail shows status lines and JSON
        byte events; the result gives the weights_path folder. Models that need
        manual assets are rejected.
        """
        with _tool_errors(invalid="manual_setup"):
            return api.start_setup(_jobs(ctx), model_id)

    def memory_status() -> MemoryStatus:
        """Report this Mac's memory and every MedMLX model run holding it, from any session."""
        return api.memory_status()

    def summarize_output(
        ctx: Context[Jobs, object], path: str, model_id: str | None = None,
        job_id: str | None = None,
    ) -> OutputSummary:
        """Report a NIfTI output's geometry and, for a mask, each label's voxels and mL.

        job_id supplies names from the matching outputs entry. Otherwise model_id
        supplies declared names. names_from reports outputs, declaration or null.
        Images report their value range. Nothing is interpreted.
        """
        check_paths({"output_path": path})
        jobs = _jobs(ctx)
        if job_id is not None:
            with _tool_errors(missing="unknown_job"):
                jobs.get(job_id)
        with _tool_errors():
            return api.summarize_output(
                path, job_id=job_id, model_id=model_id, jobs=jobs, root=root,
            )

    def preview_output(
        ctx: Context[Jobs, object], path: str, image_path: str | None = None,
        model_id: str | None = None, job_id: str | None = None,
    ) -> Annotated[CallToolResult, OutputSummary]:
        """Return a PNG and summary of a NIfTI output, over its scan when given.

        Masks show the slice with the most labeled voxels, colored by label,
        in radiological orientation (patient right on the image left).
        job_id supplies output names before model_id supplies declared names.
        names_from reports outputs, declaration or null.
        """
        check_paths({"output_path": path, "image_path": image_path})
        jobs = _jobs(ctx)
        if job_id is not None:
            with _tool_errors(missing="unknown_job"):
                jobs.get(job_id)
        with _tool_errors():
            summary = api.summarize_output(
                path, job_id=job_id, model_id=model_id, jobs=jobs, root=root,
            )
            png = api.preview_output(
                path, image_path=image_path, job_id=job_id, model_id=model_id, jobs=jobs, root=root,
            )
        return CallToolResult(
            content=[Image(data=png, format="png").to_image_content(),
                     TextContent(type="text", text=json.dumps(summary))],
            structured_content=summary,
        )

    async def list_dicom_series(directory: str) -> dict[str, list[DicomSeries]]:
        """List the DICOM series in a folder with their modality and description."""
        check_paths({"input_dir": directory})
        with _tool_errors():
            series = await to_thread.run_sync(
                lambda: outputs.list_dicom_series(directory, root=root))
        return {"series": series}

    async def convert_dicom(
        directory: str, output_path: str, series_uid: str | None = None,
    ) -> ConvertedSeries:
        """Write one DICOM series as NIfTI (.nii.gz) for model input; never replaces a file.

        series_uid may be omitted when the folder holds one series.
        """
        check_paths({"input_dir": directory, "output_path": output_path})
        with _tool_errors():
            return await to_thread.run_sync(lambda: outputs.convert_dicom(
                directory, output_path, series_uid=series_uid, root=root))

    async def get_job(
        ctx: Context[Jobs, object],
        job_id: str,
        wait_seconds: Annotated[
            float,
            Field(ge=0, le=_MAX_WAIT_SECONDS, description="Wait up to this long for the job."),
        ] = 0,
    ) -> JobPayload:
        """Return a job's state, log tail, and its result or error once finished."""

        with _tool_errors(missing="unknown_job"):
            deadline = anyio.current_time() + wait_seconds
            job = _jobs(ctx).get(job_id)
            while job.state == "running" and anyio.current_time() < deadline:
                await anyio.sleep(_POLL_SECONDS)
                job = _jobs(ctx).get(job_id)
            return job.to_payload()

    def list_jobs(ctx: Context[Jobs, object]) -> JobListPayload:
        """List the runs started in this MCP session."""

        return {"jobs": [job.to_payload() for job in _jobs(ctx).all()]}

    def cancel_job(ctx: Context[Jobs, object], job_id: str) -> JobPayload:
        """Stop a running job and its child processes."""

        with _tool_errors(missing="unknown_job"):
            return _jobs(ctx).cancel(job_id).to_payload()

    tools: tuple[tuple[Tool, ToolAnnotations], ...] = (
        (list_packages, _READ_ONLY),
        (list_models, _READ_ONLY),
        (list_checkpoints, _READ_ONLY),
        (describe_model, _READ_ONLY),
        (model_readiness, _READ_ONLY),
        (runtime_info, _READ_ONLY),
        (memory_status, _READ_ONLY),
        (list_dicom_series, _READ_ONLY),
        (convert_dicom, _WRITES),
        (prepare_model, _PREPARES),
        (run_model, _RUNS),
        (get_job, _READ_ONLY),
        (list_jobs, _READ_ONLY),
        (cancel_job, _CANCELS),
        (summarize_output, _READ_ONLY),
        (preview_output, _READ_ONLY),
    )
    for tool, hints in tools:
        server.add_tool(tool, annotations=hints)
    return server


def _jobs(ctx: Context[Jobs, object]) -> Jobs:
    return ctx.request_context.lifespan_context



__all__ = ["build_server"]
