"""MedGemma bridge with automatic weights and a managed isolated MedVLM runtime.

MedVLM's dependency graph stays separate from the unified MLX environment. First
use installs its pinned runtime under the MedMLX cache; readiness only inspects
existing installations. An explicit MEDMLX_MEDVLM_PYTHON override is still
supported. Child processes inherit the supervised worker's process group.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.weights import weights_dir, weights_revision

MEDGEMMA = {
    "id": "medgemma", "label": "MedGemma (provisional bridge)", "task": "generation",
    "run": "medmlx.adapters.medvlm:medgemma", "readiness": "medmlx.adapters.medvlm:readiness",
    "summary": "Provisional local MedGemma/Gemma3 image-to-text bridge; checkpoint inference "
               "and numerical accuracy are not yet qualified by MedMLX.",
}
_CONFIGURATION = "MEDMLX_MEDVLM_PYTHON"
_RUNTIME_VERSION = "0.3.3"


def _managed_prefix() -> Path:
    cache = Path(os.environ.get("MEDMLX_ASSET_CACHE", Path.home() / ".cache/medmlx")).expanduser()
    return cache / ".runtimes" / f"med-vlm-mlx-{_RUNTIME_VERSION}"


def _python(*, provision: bool = False) -> Path:
    value = os.environ.get(_CONFIGURATION)
    if value is not None:
        path = Path(value).expanduser()
        if (not value or not path.is_absolute() or not path.is_file()
                or not os.access(path, os.X_OK)):
            raise MissingDependencyError(
                f"{_CONFIGURATION} must name an executable absolute Python",
            )
        # Resolving a venv's Python symlink would select its base environment.
        return path
    prefix = _managed_prefix()
    executable = prefix / "bin/python"
    if executable.is_file() and (prefix / ".ready").is_file():
        return executable
    if not provision:
        raise MissingDependencyError(
            "MedGemma's isolated runtime will be installed automatically on its first run. "
            f"An existing environment can optionally be selected with {_CONFIGURATION}."
        )
    return _install_runtime(prefix)


def _install_runtime(prefix: Path) -> Path:
    """Install the independent dependency graph once; interrupted installs retry.

    uv's Python and package caches remain shared, but its target environment is
    private to MedGemma. Caller project configuration is ignored; no dependency
    overrides or parent-environment changes occur. Commands inherit the worker's
    process group so cancellation stops them.
    """
    uv = shutil.which("uv")
    if uv is not None:
        command = [uv, "--no-config"]
    elif importlib.util.find_spec("uv") is not None:
        command = [sys.executable, "-m", "uv", "--no-config"]
    else:
        raise MissingDependencyError("Install or upgrade medmlx to include its uv installer.")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock_path = prefix.parent / ".medgemma.lock"
    executable = prefix / "bin/python"
    ready = prefix / ".ready"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if executable.is_file() and ready.is_file():
            return executable
        print("Preparing the isolated MedGemma runtime…", file=sys.stderr, flush=True)
        operations: list[list[str]] = []
        if not executable.is_file():
            operations.append([
                *command, "venv", "--allow-existing", "--python", "3.13", str(prefix),
            ])
        operations.append([
            *command, "pip", "install", "--python", str(executable),
            f"med-vlm-mlx=={_RUNTIME_VERSION}", "packaging>=24",
        ])
        for operation in operations:
            completed = subprocess.run(
                operation, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, check=False,
            )
            if completed.returncode:
                raise MissingDependencyError(
                    "MedGemma runtime installation failed. Retry to resume it: "
                    + completed.stderr.strip(),
                )
        with tempfile.NamedTemporaryFile(mode="w", dir=prefix, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(f"{_RUNTIME_VERSION}\n")
        try:
            temporary.replace(ready)
        finally:
            temporary.unlink(missing_ok=True)
    return executable


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise ModelExecutionError("Malformed MedVLM bridge response")
    return cast(dict[str, object], value)


def _invoke(request: dict[str, object], timeout: float) -> dict[str, object]:
    """Isolate helper output from MCP stdio and keep control files owned by the job."""
    executable = _python(provision=request.get("action") == "run")
    with tempfile.TemporaryDirectory(
        prefix="medmlx-medvlm-", dir=os.environ.get("MEDMLX_WORKER_CONTROL_DIR"),
    ) as directory:
        root = Path(directory)
        request_path, response_path = root / "request.json", root / "response.json"
        request_path.write_text(json.dumps({"protocol": 1, **request}, allow_nan=False))
        helper = Path(__file__).with_name("_medvlm_worker.py")
        with (root / "helper.log").open("wb") as log, subprocess.Popen(
            [str(executable), "-I", "-B", str(helper), str(request_path), str(response_path)],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                process.kill()
                process.wait()
                raise
        if not response_path.is_file():
            with (root / "helper.log").open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 4000))
                diagnostic = log.read().decode("utf-8", errors="replace").strip()
            raise ModelExecutionError(
                f"MedVLM helper exited with code {code} without a response: {diagnostic}"
            )
        packet = _object(json.loads(response_path.read_text()))
        if packet.get("protocol") != 1 or type(packet.get("ok")) is not bool:
            raise ModelExecutionError("Malformed MedVLM bridge response")
        if not packet["ok"]:
            error = _object(packet.get("error"))
            message = f"MedVLM {error.get('type')}: {error.get('message')}"
            if packet.get("stage") == "assets":
                if error.get("type") == "FileNotFoundError":
                    raise AssetNotReadyError(message)
                raise InvalidInputError(message)
            if packet.get("stage") == "runtime":
                raise MissingDependencyError(message)
            raise ModelExecutionError(message)
        if code != 0:
            raise ModelExecutionError(f"MedVLM helper exited with code {code}")
        return _object(packet.get("result"))


def readiness() -> dict[str, object]:
    """Check isolated dependency metadata only; do not import MLX or load checkpoints."""
    try:
        _invoke({"action": "probe"}, 15)
    except (
        OSError, ValueError, RuntimeError, MissingDependencyError, subprocess.TimeoutExpired,
    ) as error:
        return {"ready": False, "reason": str(error),
                "setup_hint": "Run MedGemma to prepare its isolated runtime automatically, "
                              "or repair the explicitly selected environment."}
    return {"ready": True, "reason": None,
            "setup_hint": "Isolated dependency metadata passed. Adapter remains provisional; "
                          "isolated MLX/Metal, checkpoint inputs and end-to-end inference "
                          "are unverified."}


def _timeout(value: object) -> float:
    """Validate the bridge's wall-clock limit in seconds, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not (
        math.isfinite(value) and 1 <= value <= 3600
    ):
        raise InvalidInputError("timeout_seconds must be finite seconds in [1, 3600]")
    return float(value)


def _run(
    model_id: str, weights_path: str | None, image_path: str, prompt: str, output_dir: str,
    max_new_tokens: int, timeout_seconds: float, overwrite: bool,
) -> dict[str, object]:
    """Publish one JSON report only after successful isolated generation."""
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be boolean")
    timeout = _timeout(timeout_seconds)
    if (isinstance(max_new_tokens, bool)
            or not isinstance(cast(object, max_new_tokens), int)
            or not 1 <= max_new_tokens <= 2048):
        raise InvalidInputError("max_new_tokens must be an integer in [1, 2048]")
    if not isinstance(cast(object, prompt), str) or not prompt.strip() or len(prompt) > 16384:
        raise InvalidInputError("prompt must contain nonblank text of at most 16384 characters")
    image = Path(image_path).expanduser().absolute()
    if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise InvalidInputError("image_path must be an existing local PNG or JPEG file")
    destination = Path(output_dir).expanduser().resolve() / "report.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}; set overwrite=True to replace it")
    model = (Path(weights_path).expanduser().absolute()
             if weights_path is not None else weights_dir(model_id))
    if not model.is_dir():
        raise AssetNotReadyError(f"weights_path must be an existing local folder: {model}")
    # Resolve only for output alias protection; pass the folder's original name
    # into the loader so checkpoint-file symlinks keep their suffixes.
    real_model = model.resolve()
    if (destination == image.resolve() or destination == real_model
            or real_model in destination.parents):
        raise InvalidInputError("Output must differ from inputs and stay outside the checkpoint")
    try:
        result = _invoke({"action": "run", "model_id": model_id, "model_dir": str(model),
                          "image_path": str(image), "prompt": prompt,
                          "max_new_tokens": max_new_tokens}, timeout)
    except subprocess.TimeoutExpired as error:
        raise ModelExecutionError(f"MedVLM exceeded timeout_seconds={timeout_seconds}") from error
    report = {**result, "model_id": model_id, "model_dir": str(model), "image_path": str(image),
              "adapter_status": "provisional", "checkpoint_identity": "caller_supplied_unverified",
              "qualification": "not_assessed_by_adapter",
              "weights_revision": weights_revision(model)}
    text = json.dumps(report, allow_nan=False, indent=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"outputs": {"report": str(destination)},
            "details": {"adapter_status": "provisional", "qualification": "not_assessed_by_adapter",
                        "model_id": model_id, "versions": result.get("versions"),
                        "weights_revision": weights_revision(model)}}


def medgemma(
    *, image_path: str, prompt: str, output_dir: str, weights_path: str | None = None,
    max_new_tokens: int = 2000, timeout_seconds: float = 300, overwrite: bool = False,
) -> dict[str, object]:
    """Generate text from a local MedGemma folder in a managed isolated runtime.

    Provisional bridge: no real-checkpoint inference or numerical qualification
    is established by MedMLX. The isolated compatible runtime installs on first use. Greedy
    decoding uses temperature 0 and a 64-token prefill chunk. Only one image is
    supported; no clinical interpretation or checkpoint provenance is validated.

    Args:
        weights_path: Folder with model_type=gemma3, config.json,
            tokenizer.json, tokenizer_config.json, preprocessor_config.json and
            nonempty safetensors weights. Architecture and paths are checked;
            strict tensor matching occurs during loading. The default is None,
            but automatic assets are not configured, so supply a local folder.
        image_path: Existing local PNG/JPEG image; decoded by the isolated provider.
            Volumetric NIfTI/DICOM input is outside this adapter's contract.
        prompt: Nonblank text, at most 16384 Unicode characters (bridge limit).
        output_dir: Directory for report.json containing text, token counts and
            provisional status. Must be outside the checkpoint directory.
        max_new_tokens: Integer output-token limit in [1, 2048] (bridge limit). The
            default, 2000, is the value in Google's model card examples.
        timeout_seconds: Wall-clock seconds for helper startup, load and generation,
            finite in [1, 3600]. Supervised cancellation stops its process group.
        overwrite: Replace report.json atomically; false refuses existing output.
    """
    return _run("medgemma", weights_path, image_path, prompt, output_dir,
                max_new_tokens, timeout_seconds, overwrite)
