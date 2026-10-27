"""Provisional MedGemma/Lingshu bridges to an explicitly configured MedVLM environment.

Set MEDMLX_MEDVLM_PYTHON to that environment's absolute Python executable. No
package installation, interpreter search, downloads or persistent configuration
changes occur. Helpers inherit the supervised worker's process group so session
cancellation also stops isolated model execution. Checkpoint identity, numerical
parity and medical accuracy are not established by these adapters.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)

MEDGEMMA = {
    "id": "medgemma", "label": "MedGemma (provisional bridge)", "task": "generation",
    "run": "medmlx.adapters.medvlm:medgemma", "readiness": "medmlx.adapters.medvlm:readiness",
    "summary": "Provisional local MedGemma/Gemma3 image-to-text bridge; checkpoint inference "
               "and numerical accuracy are not yet qualified by MedMLX.",
}
LINGSHU = {
    "id": "lingshu", "label": "Lingshu (provisional bridge)", "task": "generation",
    "run": "medmlx.adapters.medvlm:lingshu", "readiness": "medmlx.adapters.medvlm:readiness",
    "summary": "Provisional local Lingshu/Qwen2.5-VL image-to-text bridge; checkpoint inference "
               "and numerical accuracy are not yet qualified by MedMLX.",
}
_CONFIGURATION = "MEDMLX_MEDVLM_PYTHON"


def _python() -> Path:
    value = os.environ.get(_CONFIGURATION, "")
    path = Path(value).expanduser()
    if not value or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise MissingDependencyError(
            f"Set {_CONFIGURATION} to the absolute Python executable of an existing "
            "med-vlm-mlx==0.2.0 environment. Install missing dependencies there explicitly."
        )
    # Resolving a venv's Python symlink would select its base environment instead.
    return path


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise ModelExecutionError("Malformed MedVLM bridge response")
    return cast(dict[str, object], value)


def _invoke(request: dict[str, object], timeout: float) -> dict[str, object]:
    """Isolate helper output from MCP stdio and keep control files owned by the job."""
    executable = _python()
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
                "setup_hint": f"Configure {_CONFIGURATION} with a compatible existing MedVLM "
                              "environment; no dependencies are installed automatically."}
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
    model_id: str, model_dir: str, image_path: str, prompt: str, output_dir: str,
    max_new_tokens: int, timeout_seconds: float, overwrite: bool,
) -> dict[str, object]:
    """Publish one JSON report only after successful isolated generation."""
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be boolean")
    timeout = _timeout(timeout_seconds)
    model, image = Path(model_dir).expanduser().resolve(), Path(image_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() / "report.json"
    if destination in (image, model) or model in destination.parents:
        raise InvalidInputError("Output must differ from inputs and stay outside the checkpoint")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}; set overwrite=True to replace it")
    try:
        result = _invoke({"action": "run", "model_id": model_id, "model_dir": str(model),
                          "image_path": str(image), "prompt": prompt,
                          "max_new_tokens": max_new_tokens}, timeout)
    except subprocess.TimeoutExpired as error:
        raise ModelExecutionError(f"MedVLM exceeded timeout_seconds={timeout_seconds}") from error
    report = {**result, "model_id": model_id, "model_dir": str(model), "image_path": str(image),
              "adapter_status": "provisional", "checkpoint_identity": "caller_supplied_unverified",
              "qualification": "not_assessed_by_adapter"}
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
                        "model_id": model_id, "versions": result.get("versions")}}


def medgemma(
    *, model_dir: str, image_path: str, prompt: str, output_dir: str,
    max_new_tokens: int = 256, timeout_seconds: float = 300, overwrite: bool = False,
) -> dict[str, object]:
    """Generate text using a caller-supplied local MedGemma/Gemma3 checkpoint.

    Provisional bridge: no real-checkpoint inference or numerical qualification
    is established by MedMLX. Configure MEDMLX_MEDVLM_PYTHON explicitly. Greedy
    decoding uses temperature 0 and a 64-token prefill chunk. Only one image is
    supported; no clinical interpretation or checkpoint provenance is validated.

    Args:
        model_dir: Existing local bundle with model_type=gemma3, config.json,
            tokenizer.json, tokenizer_config.json, preprocessor_config.json and
            nonempty safetensors weights. Architecture and paths are checked;
            strict tensor matching occurs during loading. Downloads are disabled.
        image_path: Existing local PNG/JPEG image; decoded by the isolated provider.
            Volumetric NIfTI/DICOM input is outside this adapter's contract.
        prompt: Nonblank text, at most 16384 Unicode characters (bridge limit).
        output_dir: Directory for report.json containing text, token counts and
            provisional status. Must be outside the checkpoint directory.
        max_new_tokens: Integer output-token limit in [1, 2048] (bridge limit).
        timeout_seconds: Wall-clock seconds for helper startup, load and generation,
            finite in [1, 3600]. Supervised cancellation stops its process group.
        overwrite: Replace report.json atomically; false refuses existing output.
    """
    return _run("medgemma", model_dir, image_path, prompt, output_dir,
                max_new_tokens, timeout_seconds, overwrite)


def lingshu(
    *, model_dir: str, image_path: str, prompt: str, output_dir: str,
    max_new_tokens: int = 256, timeout_seconds: float = 300, overwrite: bool = False,
) -> dict[str, object]:
    """Generate text using a caller-supplied local Lingshu/Qwen2.5-VL checkpoint.

    Provisional bridge with the same isolation, limits and output contract as
    MedGemma. Configure MEDMLX_MEDVLM_PYTHON explicitly; no assets are downloaded.
    Greedy generation uses temperature 0 and a 64-token prefill chunk.

    Args:
        model_dir: Existing local bundle with model_type=qwen2_5_vl, config.json,
            tokenizer.json, tokenizer_config.json, preprocessor_config.json and
            nonempty safetensors weights. Identity and medical accuracy are unverified.
        image_path: Existing local PNG/JPEG image; decoded by the isolated provider.
        prompt: Nonblank text, at most 16384 Unicode characters (bridge limit).
        output_dir: Directory for report.json; must be outside the checkpoint.
        max_new_tokens: Integer output-token limit in [1, 2048] (bridge limit).
        timeout_seconds: Finite helper startup/load/generation seconds in [1, 3600].
        overwrite: Replace report.json atomically; false refuses existing output.
    """
    return _run("lingshu", model_dir, image_path, prompt, output_dir,
                max_new_tokens, timeout_seconds, overwrite)
