"""Automatic model folders shared by independent MedMLX runtimes.

Registered model IDs select their published Hugging Face repositories. A completed
download records its Hub commit so subsequent runs can reuse it offline and report
the exact revision.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict, cast

from httpx import HTTPError

from medmlx_core.errors import AssetNotReadyError, MissingDependencyError

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_REVISION_FILE = ".medmlx-revision"


class DownloadProgress(TypedDict):
    """Completed file bytes, including resumed bytes; totals may grow or be unknown."""

    message: str
    model_id: str
    bytes: int
    total_bytes: int | None


class _ModelInfo(Protocol):
    sha: str | None


class _HubApi(Protocol):
    def model_info(self, repo_id: str) -> _ModelInfo: ...


class _Hub(Protocol):
    """The Hub operations used by the downloader's optional import boundary."""

    def HfApi(self) -> _HubApi: ...
    def snapshot_download(
        self, repo_id: str, *, revision: str, local_dir: Path, force_download: bool = False,
        allow_patterns: list[str] | None = None, tqdm_class: type[object],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _HubSource:
    """A published bundle and, when needed, the model's folder or file in it."""

    repo: str
    subfolder: str = ""
    filename: str = ""
    revision: str | None = None
    companion_subfolder: str = ""


_SOURCES = {
    "musclemap": _HubSource(
        "MedMLX/musclemap-mlx",
        revision="a798b3073e3b2c449987c13aa193626aaacc3092",
    ),
    "biomedparse-v2": _HubSource("MedMLX/biomedparse-v2-mlx"),
    "flexible-unet": _HubSource("MedMLX/flexible-unet-effnetb2-mlx"),
    "flexray": _HubSource("MedMLX/flexray-mlx"),
    "lesiondiffusion": _HubSource("MedMLX/lesiondiffusion-mlx"),
    "lung-nodule": _HubSource("MedMLX/retinanet-lung-nodule-mlx"),
    "medsam2": _HubSource("MedMLX/medsam2-ct-lesion-mlx"),
    "mlx-generate-ctmr-ct": _HubSource(
        "MedMLX/mlx-generate-ctmr", "ct", "model.safetensors", companion_subfolder="mask",
    ),
    "mlx-generate-ctmr-mr": _HubSource("MedMLX/mlx-generate-ctmr", "mr", "model.safetensors"),
    "mlx-generate-ctmr-mr-brain": _HubSource(
        "MedMLX/mlx-generate-ctmr", "mr-brain", "model.safetensors",
    ),
    "mlx-reason-ct": _HubSource("MedMLX/MLX-Reason-CT"),
    "mlx-segment-ct": _HubSource("MedMLX/vista3d-ct-mlx"),
    "mlx-segment-ctmr": _HubSource("MedMLX/vista3d-ctmr-mlx"),
    "nnunet-autopet": _HubSource("MedMLX/nnunet-autopet-mlx"),
    "nnunet-brats2021": _HubSource("MedMLX/nnunet-brats-mlx"),
    "nnunet-liver": _HubSource("MedMLX/nnunet-liver-mlx"),
    "nnunet-pancreas": _HubSource("MedMLX/nnunet-pancreas-mlx"),
    "radimagenet-densenet121": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-DenseNet121_notop.safetensors",
    ),
    "radimagenet-inceptionresnetv2": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-IRV2_notop.safetensors",
    ),
    "radimagenet-inceptionv3": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-InceptionV3_notop.safetensors",
    ),
    "radimagenet-resnet50": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-ResNet50_notop.safetensors",
    ),
    # The SAM3 runtime explicitly pins this native FP32 checkpoint.
    "sam3": _HubSource(
        "mlx-community/sam3-image", revision="b72a14d8127e17e6f2a3d2e075bbbf4307ba146e",
    ),
    "segresnet-brats": _HubSource("MedMLX/segresnet-brats-mlx"),
    "segresnet-renal-cect": _HubSource("MedMLX/segresnet-renal-cect-mlx"),
    "swin-unetr": _HubSource("MedMLX/swin-unetr-mlx"),
    "synthseg": _HubSource("MedMLX/synthseg-mlx"),
    "totalsegmentator": _HubSource("MedMLX/totalsegmentator-mlx"),
    "unest": _HubSource("MedMLX/unest-renal-ct-mlx"),
}
_NO_AUTOMATIC_WEIGHTS = frozenset({"medgemma", "mrsegmentator"})


def _source(model_id: str) -> _HubSource | None:
    if not isinstance(cast(object, model_id), str) or not _MODEL_ID.fullmatch(model_id):
        raise ValueError("Model IDs must use lowercase letters, digits, '.', '_' or '-'")
    if model_id in _NO_AUTOMATIC_WEIGHTS:
        return None
    return _SOURCES.get(model_id, _HubSource(f"MedMLX/{model_id}"))


def weights_repo(model_id: str) -> str | None:
    """Return the Hub repository, or None when automatic weights are unavailable.

    Unlisted model IDs use the ``MedMLX/<model_id>`` naming convention.
    """
    source = _source(model_id)
    return None if source is None else source.repo


def weights_dir(
    model_id: str, *, progress: Callable[[str], None] | None = None, refresh: bool = False,
    download_progress: Callable[[DownloadProgress], None] | None = None,
) -> Path:
    """Return a model folder, downloading it on first use and resuming partial work.

    Download the published repository or its model-specific subset as ordinary files
    under ``~/.cache/medmlx/<model_id>`` (or ``MEDMLX_ASSET_CACHE``). Resolve
    ``main`` to a commit once, or use a runtime's pinned revision, then atomically
    record it after downloading. Completed folders need no network access.
    ``refresh=True`` downloads the selected revision again to repair damaged local
    files. Concurrent processes share a lock and the Hub's resumable downloads;
    interruption never marks a partial download done.

    ``progress`` receives status messages. ``download_progress`` receives byte
    events at most twice a second, plus a final event when the download stops.
    Hub totals can grow as files are discovered; unknown totals are ``None``.

    Call this only when ``weights_path`` is omitted. Explicit user paths belong
    to the runner and must not trigger downloads or have their links resolved.
    This helper does not load tensors or certify a model's numerical behavior.
    """
    source = _source(model_id)
    if source is None:
        raise AssetNotReadyError(
            f"No automatic weights are configured for {model_id}.",
            hint="Pass a compatible local weights_path folder.",
        )
    try:
        import fcntl
    except ImportError as error:
        raise MissingDependencyError(
            "Automatic model downloads require a macOS or Unix host.",
        ) from error
    notify: Callable[[str], None] = progress or (lambda _message: None)
    cache, target, folder = _locations(model_id, source)
    lock_dir = cache / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    # Keep lock files: unlinking one can let concurrent processes lock different
    # inodes. A cancelled worker automatically releases its advisory lock.
    with (lock_dir / f"{model_id}.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not refresh and (cached := cached_weights(model_id)) is not None:
            notify(f"Using cached weights for {model_id}.")
            return cached
        try:
            from medmlx_core._download_progress import download_progress_bar

            hub = cast(_Hub, import_module("huggingface_hub"))
        except ImportError as error:
            raise MissingDependencyError(
                "Automatic model downloads require huggingface-hub.",
                hint="Upgrade medmlx-core to install its download dependency.",
            ) from error
        repo = source.repo
        notify(f"Downloading {repo}…")
        try:
            revision = source.revision or hub.HfApi().model_info(repo).sha
            if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
                raise AssetNotReadyError(
                    f"The Hub did not return a commit for {repo}",
                    hint="Retry the download, or pass your own weights_path folder.",
                )
            # Invalidate completion immediately before files can change. An
            # interrupted refresh resumes on the next ordinary call.
            (folder / _REVISION_FILE).unlink(missing_ok=True)
            with download_progress_bar(
                model_id, f"Downloading {repo}…", download_progress,
            ) as tqdm_class:
                if source.subfolder or source.filename:
                    selected = f"{source.subfolder}/**" if source.subfolder else source.filename
                    patterns = [selected]
                    if source.companion_subfolder:
                        # CT's learned mask model is a sibling of its image weights.
                        patterns.append(f"{source.companion_subfolder}/**")
                    hub.snapshot_download(
                        repo, revision=revision, local_dir=target, force_download=refresh,
                        allow_patterns=[
                            *patterns, "README.md", "LICENSE*", "NOTICE", "SOURCE_LICENSE",
                            "UPSTREAM_NOTICES.md", ".gitattributes",
                        ],
                        tqdm_class=tqdm_class,
                    )
                else:
                    hub.snapshot_download(
                        repo, revision=revision, local_dir=target, force_download=refresh,
                        tqdm_class=tqdm_class,
                    )
            if source.filename and not (folder / source.filename).is_file():
                raise FileNotFoundError(f"Missing {source.filename} in the downloaded model folder")
            if source.companion_subfolder and not (
                target / source.companion_subfolder / source.filename
            ).is_file():
                raise FileNotFoundError(
                    f"Missing {source.companion_subfolder}/{source.filename} "
                    "in the downloaded model bundle"
                )
        except (HTTPError, OSError) as error:
            raise AssetNotReadyError(
                f"Could not download {repo}: {error}",
                hint="Retry to resume the download, or pass your own weights_path folder.",
            ) from error
        # The marker contains one commit, not a model manifest. Write it only
        # after snapshot_download has completed every file at that revision.
        folder.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".revision-", dir=folder)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(revision + "\n")
            os.replace(temporary, folder / _REVISION_FILE)
        finally:
            Path(temporary).unlink(missing_ok=True)
        notify(f"Weights are available for {model_id}.")
        return folder


def cached_weights(model_id: str) -> Path | None:
    """Return a model's completed managed folder without downloading, or ``None``.

    The folder must carry a recorded revision that matches any pinned revision,
    with its companion files present: the same test ``weights_dir`` uses before
    reusing a cache. This is a local read; it takes no lock, contacts no Hub,
    creates no directories and writes no files. It does not inspect tensors.
    Models without automatic weights return ``None``.
    """
    source = _source(model_id)
    if source is None:
        return None
    _cache, target, folder = _locations(model_id, source)
    return folder if _complete(source, target, folder) else None


def _locations(model_id: str, source: _HubSource) -> tuple[Path, Path, Path]:
    """Return the cache root, the model's managed target and its weights folder."""
    cache = Path(os.environ.get("MEDMLX_ASSET_CACHE", Path.home() / ".cache/medmlx")).expanduser()
    target = cache / model_id
    # Old managed folders used the model ID as the repository name. Keep a
    # different source separate so its revision marker cannot certify this one.
    if source.repo.lower() != f"medmlx/{model_id}":
        target /= source.repo.replace("/", "--")
    return cache, target, target / source.subfolder


def _complete(source: _HubSource, target: Path, folder: Path) -> bool:
    """Whether a managed folder finished downloading the selected revision."""
    cached_revision = weights_revision(folder)
    companion_ready = (
        not source.companion_subfolder
        or (target / source.companion_subfolder / source.filename).is_file()
    )
    return (cached_revision is not None
            and (source.revision is None or cached_revision == source.revision)
            and companion_ready)


def weights_revision(directory: str | Path) -> str | None:
    """Return a managed folder's downloaded commit, or ``None`` for local weights.

    This is a local read only. It does not contact the Hub, resolve user paths,
    inspect tensors, or assert checkpoint compatibility.
    """
    try:
        revision = (Path(directory) / _REVISION_FILE).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return revision if _REVISION.fullmatch(revision) else None


__all__ = ["DownloadProgress", "cached_weights", "weights_dir", "weights_repo", "weights_revision"]
