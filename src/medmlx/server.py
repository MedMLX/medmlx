"""MCP server for discovered MedMLX models: listing, readiness, and supervised runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

import anyio
from anyio import to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from medmlx import __version__, api
from medmlx.jobs import Jobs
from medmlx.payloads import (
    CheckpointListPayload,
    JobListPayload,
    JobPayload,
    JsonValue,
    ModelDescription,
    ModelListPayload,
    PackageListPayload,
    ReadinessPayload,
    RuntimeToolPayload,
)
from medmlx.probe import runtime_info as probe_runtime

_MAX_WAIT_SECONDS = 60.0
_POLL_SECONDS = 0.25

_INSTRUCTIONS = """\
MedMLX runs native MLX medical imaging models on macOS Apple Silicon with Metal.
Model families are optional segment, generate and vision installs.
list_packages reports their independently maintained runtime packages.
list_checkpoints exposes source references, including catalog-only checkpoints;
it does not download weights or assert that checkpoints have been qualified.
Typical flow: list_models -> describe_model -> model_readiness -> run_model ->
get_job (with wait_seconds). runtime_info reports MLX Metal availability and device.
Readiness preserves ready as a boolean alias of runtime_ready (provider/import
checks plus Metal). run_ready is null for unverified input/checkpoint contents, or
false for a known blocker; this check cannot establish a validated run.
Supply arguments with one model id to check argument binding and local input/weight
paths without inference or downloads. describe_model reproduces provider instructions
and argument descriptions; it does not independently validate or qualify their claims.
Some providers fetch default weights when assets are omitted (for example SAM3
checkpoint_path). Read those instructions and supply local assets for offline runs.
run_model takes the model's run-function keyword arguments, as listed by
describe_model. Runs execute with the server root as working directory, so
relative paths resolve against: {root}.
MedMLX is research software, not a medical device. Numerical parity with an
upstream model does not establish clinical equivalence, and model output
requires human review.
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_RUNS = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True)
_CANCELS = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

Tool = Callable[..., object]


@contextmanager
def _tool_errors() -> Generator[None]:
    """Report anticipated failures to the client with their message."""

    try:
        yield
    except (LookupError, OSError, ValueError, ImportError, RuntimeError) as error:
        raise ToolError(str(error) or type(error).__name__) from error


def build_server(root: Path) -> MCPServer[Jobs]:
    """Build the stdio MCP server whose runs use ``root`` as working directory."""

    root = root.resolve()

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
        instructions=_INSTRUCTIONS.format(root=root),
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
        with _tool_errors():
            return api.model_readiness(model_ids, arguments=arguments, root=root)

    def runtime_info() -> RuntimeToolPayload:
        """Report the platform, package versions, and MLX Metal availability and device."""

        return {"root": root.as_posix(), **probe_runtime()}

    def run_model(
        ctx: Context[Jobs, object], model_id: str, arguments: dict[str, JsonValue],
    ) -> JobPayload:
        """Start a validated run in a worker and return its job; poll get_job."""
        with _tool_errors():
            return api.start_model(_jobs(ctx), model_id, arguments)

    async def get_job(
        ctx: Context[Jobs, object],
        job_id: str,
        wait_seconds: Annotated[
            float,
            Field(ge=0, le=_MAX_WAIT_SECONDS, description="Wait up to this long for the job."),
        ] = 0,
    ) -> JobPayload:
        """Return a job's state, log tail, and its result or error once finished."""

        with _tool_errors():
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

        with _tool_errors():
            return _jobs(ctx).cancel(job_id).to_payload()

    tools: tuple[tuple[Tool, ToolAnnotations], ...] = (
        (list_packages, _READ_ONLY),
        (list_models, _READ_ONLY),
        (list_checkpoints, _READ_ONLY),
        (describe_model, _READ_ONLY),
        (model_readiness, _READ_ONLY),
        (runtime_info, _READ_ONLY),
        (run_model, _RUNS),
        (get_job, _READ_ONLY),
        (list_jobs, _READ_ONLY),
        (cancel_job, _CANCELS),
    )
    for tool, hints in tools:
        server.add_tool(tool, annotations=hints)
    return server


def _jobs(ctx: Context[Jobs, object]) -> Jobs:
    return ctx.request_context.lifespan_context



__all__ = ["build_server"]
