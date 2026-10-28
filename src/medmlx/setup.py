"""Install optional runtimes and obtain published native bundles for client applications.

Downloads use the Hub cache and resume there. A completed setup establishes runtime
availability and required files, not numerical qualification or clinical validity.
No patient input is read and no inference is launched.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict, cast

from medmlx import families
from medmlx.payloads import JsonValue


class SetupPlan(TypedDict):
    model_id: str
    supported: bool
    package: str | None
    runtime_installed: bool
    source_url: str | None
    message: str


class SetupResult(TypedDict):
    model_id: str
    arguments: dict[str, JsonValue]
    source_url: str | None
    files_checked: bool


class _HubApi(Protocol):
    def list_repo_files(self, repo_id: str, *, revision: str) -> list[str]: ...


class _Hub(Protocol):
    def HfApi(self) -> _HubApi: ...

    def snapshot_download(self, repo_id: str, *, revision: str, local_files_only: bool) -> str: ...

    def hf_hub_download(self, repo_id: str, filename: str, *, revision: str) -> str: ...


@dataclass(frozen=True)
class _Bundle:
    package: str
    repo: str
    revision: str
    argument: str
    files: tuple[str, ...]


# These are published native bundles, never source-framework checkpoints.
_BUNDLES = {
    "nv-segment-ct": _Bundle(
        "mlx-segment-ct",
        "MedMLX/vista3d-ct-mlx",
        "b7551a0fbaccd9fa5eccfde44b52fb40d2a9b65f",
        "weights_path",
        ("automatic.safetensors", "correction.safetensors"),
    ),
    "mlx-reason-ct": _Bundle(
        "mlx-reason-ct",
        "MedMLX/MLX-Reason-CT",
        "f60c688952468934acc0630090ef08832ca88d31",
        "weights_path",
        (
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "generation_config.json",
            "image_processor_3d/preprocessor_config.json",
            "mlx/manifest.json",
            "mlx/embedding.safetensors",
            "mlx/norm.safetensors",
            "mlx/vision.safetensors",
            *(f"mlx/layer_{index:02d}.safetensors" for index in range(32)),
        ),
    ),
    "flexray": _Bundle(
        "flexray-mlx",
        "MedMLX/flexray-mlx",
        "074a0132bccd579a990317f72ce6fea26b051c7c",
        "weights_path",
        (
            "ensemble.json",
            "members/flux0375/config.yml",
            "members/flux0375/label_schema.json",
            "members/flux0375/model.safetensors",
            "members/flux0375/preprocessing.json",
        ),
    ),
    "nnunet-brats2021": _Bundle(
        "nnunet-mlx",
        "MedMLX/nnunet-brats-mlx",
        "22087e0b2ca02c6ef8d8e8c55be301466c6c7d8a",
        "model_dir",
        (
            "plans.json",
            "dataset.json",
            "conversion.json",
            "fold_0/checkpoint_final/state.json",
            "fold_0/checkpoint_final/weights.safetensors",
        ),
    ),
}
_RUNTIME_ONLY = {"mlxradiomics": "mlxradiomics"}
_NOTICES = ("LICENSE*", "NOTICE*", "*NOTICES*", "APACHE*", "README.md")


def list_setup_plans() -> list[SetupPlan]:
    """List supported setup operations, including runtimes not yet installed."""
    return [setup_plan(model_id) for model_id in sorted({*_BUNDLES, *_RUNTIME_ONLY})]


def setup_plan(model_id: str) -> SetupPlan:
    """Describe automatic setup without network access, installation or downloads.

    Unsupported models retain their native setup path; this API never substitutes
    another checkpoint. Runtime versions come from this distribution's family pins.
    """
    bundle = _BUNDLES.get(model_id)
    package = bundle.package if bundle else _RUNTIME_ONLY.get(model_id)
    installed = False
    if package is not None:
        try:
            version(package)
            installed = True
        except PackageNotFoundError:
            pass
    return {
        "model_id": model_id,
        "supported": package is not None,
        "package": package,
        "runtime_installed": installed,
        "source_url": None if bundle is None else f"https://huggingface.co/{bundle.repo}",
        "message": (
            "Use cached model files, or download the published bundle automatically."
            if bundle
            else "Install the analysis runtime; no model download is needed."
            if package
            else "Automatic setup is not available for this model's checkpoint."
        ),
    }


def setup_model(
    model_id: str,
    *,
    existing_paths: Sequence[str | Path] = (),
    progress: Callable[[str], None] | None = None,
) -> SetupResult:
    """Install the pinned optional runtime and prepare one native model bundle.

    Run this on a worker process. ``existing_paths`` are explicitly chosen model
    directories, checked before the provider's Hub cache. Downloads resume in that
    cache; unrelated directories and model outputs are not searched. The returned
    keyword arguments can be saved by the calling application. Progress messages
    describe stages and completed files; callbacks must not touch GUI widgets.

    Raises:
        ValueError: This model has no automatic setup recipe.
        RuntimeError: uv is unavailable, installation fails, or required files are absent.
        OSError: The cache or model files cannot be accessed.
        ImportError: A required runtime or download library could not be imported.
        huggingface_hub.errors.HfHubHTTPError: Download or model access fails.
    """
    notify: Callable[[str], None] = progress or (lambda _message: None)
    plan = setup_plan(model_id)
    package = plan["package"]
    if package is None:
        raise ValueError(plan["message"])
    if not plan["runtime_installed"]:
        notify("Installing model software…")
        _install_runtime(package)
        importlib.invalidate_caches()
    bundle = _BUNDLES.get(model_id)
    if bundle is None:
        notify("Model software is installed.")
        return {"model_id": model_id, "arguments": {}, "source_url": None, "files_checked": True}
    notify("Checking existing model files…")
    for path in (*existing_paths, *_cached_directories()):
        candidate = Path(path).expanduser().resolve()
        if _files_available(bundle, candidate):
            return _result(model_id, bundle, candidate, notify)
    from huggingface_hub.errors import LocalEntryNotFoundError

    hub = cast(_Hub, importlib.import_module("huggingface_hub"))

    try:
        cached = Path(
            hub.snapshot_download(bundle.repo, revision=bundle.revision, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        cached = None
    if cached is not None and _files_available(bundle, cached):
        return _result(model_id, bundle, cached, notify)
    notify("Finding the published model download…")
    filenames = hub.HfApi().list_repo_files(bundle.repo, revision=bundle.revision)
    notices = tuple(
        name for name in filenames if any(fnmatch.fnmatch(name, pattern) for pattern in _NOTICES)
    )
    files = (*bundle.files, *notices)
    directory: Path | None = None
    for index, filename in enumerate(files):
        notify(f"Downloading model files · {index + 1} of {len(files)} · {filename}")
        downloaded = Path(hub.hf_hub_download(bundle.repo, filename, revision=bundle.revision))
        directory = downloaded
        for _part in Path(filename).parts:
            directory = directory.parent
    if directory is None or not _files_available(bundle, directory):
        raise RuntimeError("The download is incomplete. Try setup again to resume it.")
    return _result(model_id, bundle, directory, notify)


def _files_available(bundle: _Bundle, directory: Path) -> bool:
    return directory.is_dir() and all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in bundle.files
    )


def _cached_directories() -> list[Path]:
    """Inspect only the provider asset cache, at most two directory levels deep."""
    root = Path(os.environ.get("MEDMLX_ASSET_CACHE", Path.home() / ".cache" / "medmlx"))
    if not root.is_dir():
        return []
    children = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    nested = [
        child
        for path in children
        for child in path.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ]
    return [root, *children, *sorted(nested)]


def _result(
    model_id: str,
    bundle: _Bundle,
    directory: Path,
    notify: Callable[[str], None],
) -> SetupResult:
    """Check provider metadata where public admission APIs exist before returning paths."""
    if model_id == "mlx-reason-ct":
        from mlx_reason_ct.mlx_weights import read_manifest

        read_manifest(directory)
    elif model_id == "flexray":
        from flexray_mlx.medmlx import admit_bundle

        admit_bundle(directory, ensemble=False)
    notify("Model files are available.")
    return {
        "model_id": model_id,
        "arguments": {bundle.argument: str(directory)},
        "source_url": f"https://huggingface.co/{bundle.repo}",
        "files_checked": True,
    }


def _install_runtime(package: str) -> None:
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            "This application build does not include the model software. "
            "Install an updated application build."
        )
    uv = shutil.which("uv")
    if uv is not None:
        command = [uv]
    elif importlib.util.find_spec("uv") is not None:
        command = [sys.executable, "-m", "uv"]
    else:
        raise RuntimeError(
            "The model installer (uv) is missing. Repair the application installation."
        )
    packages = [item for family in families.families() for item in family.packages]
    requirement = next((item for item in packages if item.name == package), None)
    if requirement is None:
        raise RuntimeError(f"No pinned installation source is declared for {package}.")
    source = requirement.source
    dependency = f"{package} @ {source}" if "://" in source else f"{package}{source}"
    completed = subprocess.run(
        [*command, "pip", "install", "--python", sys.executable, dependency],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Model software installation failed: {completed.stderr.strip()}")
