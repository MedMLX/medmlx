"""Isolated MedVLM JSON bridge, executed by absolute filename with Python -I.

This file must not import MedMLX or medmlx_core: the selected interpreter owns
MedVLM's conflicting dependency set. Probe mode reads distribution metadata only.
Run mode accepts local assets and disables Hub access before importing the model.
The caller owns timeout, process-group cancellation and final output publication.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Protocol, cast


class Model(Protocol):
    config: object


class Generation(Protocol):
    text: object
    generation_tokens: object
    prompt_tokens: object


class Load(Protocol):
    def __call__(
        self, path: str, *, strict: bool, local_files_only: bool,
    ) -> tuple[Model, object]: ...


class Template(Protocol):
    def __call__(
        self, processor: object, config: object, prompt: str, *, num_images: int,
    ) -> object: ...


class Generate(Protocol):
    def __call__(
        self, model: object, processor: object, prompt: str, *, image: list[str],
        max_tokens: int, temperature: float, prefill_step_size: int, verbose: bool,
    ) -> Generation: ...


class Utilities(Protocol):
    load: Load


class Prompts(Protocol):
    apply_chat_template: Template


class Generator(Protocol):
    generate: Generate


class Metal(Protocol):
    def is_available(self) -> bool: ...


class Mlx(Protocol):
    metal: Metal
    gpu: object

    def set_default_device(self, device: object) -> None: ...


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in cast(dict[object, object], value)
    ):
        raise ValueError("Expected a JSON object with string keys")
    return cast(dict[str, object], value)


def _runtime() -> dict[str, str]:
    """Check the scoped provider version and its installed runtime dependency graph."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MedVLM requires macOS arm64")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("The isolated bridge currently requires Python 3.13")
    if version("med-vlm-mlx") != "0.3.1":
        raise RuntimeError("This provisional bridge targets med-vlm-mlx==0.3.1")
    try:
        version("mlx-vlm")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Remove the conflicting mlx-vlm distribution from this environment")
    pending = ["med-vlm-mlx"]
    found: dict[str, str] = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in found:
            continue
        package = distribution(name)
        found[name] = package.version
        for text in package.requires or []:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed = version(requirement.name)
            if not requirement.specifier.contains(installed):
                raise RuntimeError(f"{name} requires {requirement}; installed {installed}")
            pending.append(requirement.name)
    return {name: found[name] for name in (
        "med-vlm-mlx", "mlx", "transformers", "huggingface-hub", "tokenizers",
    )}


def _local_assets(request: dict[str, object]) -> tuple[Path, Path, str, int]:
    """Check paths and declared architecture without qualifying checkpoint tensors."""
    model_id = request.get("model_id")
    if model_id != "medgemma":
        raise ValueError("model_id must be medgemma")
    model_value, image_value = request.get("model_dir"), request.get("image_path")
    if not isinstance(model_value, str) or not isinstance(image_value, str):
        raise ValueError("model_dir and image_path must be local paths")
    model, image = Path(model_value), Path(image_value)
    if not model.is_absolute() or not model.is_dir():
        raise FileNotFoundError("model_dir must be an existing absolute local directory")
    if not image.is_absolute() or not image.is_file() or image.suffix.lower() not in {
        ".png", ".jpg", ".jpeg",
    }:
        raise FileNotFoundError("image_path must be an existing local PNG or JPEG file")
    for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
    ):
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Local checkpoint requires {name}; downloads are disabled")
    config = _object(json.loads((model / "config.json").read_text()))
    expected = "gemma3"
    if config.get("model_type") != expected:
        raise ValueError(f"{model_id} requires model_type={expected}")
    weights = list(model.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise FileNotFoundError("Local checkpoint requires nonempty safetensors weights")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        mapping = _object(_object(json.loads(index.read_text())).get("weight_map"))
        if not mapping:
            raise ValueError("Checkpoint weight_map must not be empty")
        for shard in mapping.values():
            if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(
                ".safetensors"
            ):
                raise ValueError("Checkpoint shard names must be local safetensors filenames")
            if not (model / shard).is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard}")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("prompt must contain 1..16384 characters, including non-whitespace")
    tokens = request.get("max_new_tokens")
    if type(tokens) is not int or not 1 <= tokens <= 2048:
        raise ValueError("max_new_tokens must be an integer in [1, 2048]")
    return model, image, prompt, tokens


def _generate(
    assets: tuple[Path, Path, str, int], versions: dict[str, str],
) -> dict[str, object]:
    """Use the provider's public strict loader, template and greedy generation APIs."""
    model_path, image_path, prompt, tokens = assets
    mx = cast(Mlx, importlib.import_module("mlx.core"))
    if not mx.metal.is_available():
        raise RuntimeError("The isolated MedVLM interpreter has no MLX Metal device")
    mx.set_default_device(mx.gpu)
    load = cast(Utilities, importlib.import_module("mlx_vlm.utils")).load
    template = cast(Prompts, importlib.import_module("mlx_vlm.prompt_utils")).apply_chat_template
    generate = cast(Generator, importlib.import_module("mlx_vlm.generate")).generate
    model, processor = load(str(model_path), strict=True, local_files_only=True)
    rendered = template(processor, model.config, prompt, num_images=1)
    if not isinstance(rendered, str):
        raise TypeError("The medical chat template must return text")
    result = generate(model, processor, rendered, image=[str(image_path)], max_tokens=tokens,
                      temperature=0.0, prefill_step_size=64, verbose=False)
    text: object = result.text
    generated: object = result.generation_tokens
    prefilling: object = result.prompt_tokens
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("MedVLM generated no text")
    if type(generated) is not int or not 1 <= generated <= tokens:
        raise RuntimeError("MedVLM returned an invalid generation token count")
    if type(prefilling) is not int or prefilling < 1:
        raise RuntimeError("MedVLM returned an invalid prompt token count")
    return {"text": text, "generation_tokens": generated, "prompt_tokens": prefilling,
            "versions": versions, "python": sys.executable}


def main() -> int:
    """Read one request and atomically return a versioned JSON success/error packet."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: _medvlm_worker.py REQUEST_JSON RESPONSE_JSON")
    request_path, response_path = map(Path, sys.argv[1:])
    # These are process-local execution settings, not persistent access changes.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    stage = "request"
    try:
        request = _object(json.loads(request_path.read_text()))
        if request.get("protocol") != 1:
            raise ValueError("Unsupported MedVLM bridge protocol")
        if request.get("action") == "probe":
            stage = "runtime"
            result: dict[str, object] = {"versions": _runtime(), "python": sys.executable}
        elif request.get("action") == "run":
            stage = "assets"
            assets = _local_assets(request)
            stage = "runtime"
            versions = _runtime()
            stage = "execution"
            result = _generate(assets, versions)
        else:
            raise ValueError("Unknown MedVLM bridge action")
        packet: dict[str, object] = {"protocol": 1, "ok": True, "result": result}
        code = 0
    except Exception as error:
        packet = {"protocol": 1, "ok": False, "stage": stage,
                  "error": {"type": type(error).__name__, "message": str(error)}}
        code = 1
    text = json.dumps(packet, allow_nan=False)
    temporary = response_path.with_suffix(".tmp")
    temporary.write_text(text)
    os.replace(temporary, response_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
