"""Automatic model folders shared by independent MedMLX runtimes.

Registered model IDs select their published Hugging Face repositories and pinned
commits. A completed download records its Hub commit so subsequent runs can reuse
it offline and report the exact revision.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from hashlib import file_digest, sha1, sha256
from importlib import import_module
from pathlib import Path
from shutil import copyfile
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
    siblings: list[_RepoFile] | None


class _LfsInfo(Protocol):
    sha256: str


class _RepoFile(Protocol):
    rfilename: str
    size: int | None
    blob_id: str | None
    lfs: _LfsInfo | None


class _HubApi(Protocol):
    def model_info(
        self, repo_id: str, *, revision: str | None = None, files_metadata: bool = False,
    ) -> _ModelInfo: ...


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
    # Model card linked by biomedparse-mlx/README.md@v0.1.5.
    "biomedparse-v2": _HubSource(
        "MedMLX/biomedparse-v2-mlx",
        revision="1c251c6561aa143cd79ff0ccdfa53a1a6e60c7f0",
    ),
    # Model card linked by flexible-unet-mlx/README.md@v0.2.6.
    "flexible-unet": _HubSource(
        "MedMLX/flexible-unet-effnetb2-mlx",
        revision="ccbbc596f2a9da8477e25cba0d7548cbb2191feb",
    ),
    # flexray-mlx/README.md@v0.2.6.
    "flexray": _HubSource(
        "MedMLX/flexray-mlx",
        revision="722d8341570e8b9e1196cf50e57892da84974fcc",
    ),
    # Model card linked by lesiondiffusion-mlx/README.md@v0.2.6.post2.
    "lesiondiffusion": _HubSource(
        "MedMLX/lesiondiffusion-mlx",
        revision="d00bbb8ad1c7170062f60b4386a38c3a4178f7ce",
    ),
    # lung-nodule-mlx/src/lung_nodule_mlx/medmlx.py@v0.2.6.
    "lung-nodule": _HubSource(
        "MedMLX/retinanet-lung-nodule-mlx",
        revision="d2d635fbac34254f14192ac348a05d275a949317",
    ),
    # Model card linked by medsam2-mlx/README.md@v0.2.8.
    "medsam2": _HubSource(
        "MedMLX/medsam2-ct-lesion-mlx",
        revision="22f5e6dc4850a71abf92ef1a8fea0db14c34d460",
    ),
    # mlx-generate-ctmr/README.md@v0.5.2.post2.
    "mlx-generate-ctmr-ct": _HubSource(
        "MedMLX/mlx-generate-ctmr", "ct", "model.safetensors",
        revision="63018cdac46394b3c12619bc9fd1cbe096c23b20", companion_subfolder="mask",
    ),
    # mlx-generate-ctmr/README.md@v0.5.2.post2.
    "mlx-generate-ctmr-mr": _HubSource(
        "MedMLX/mlx-generate-ctmr", "mr", "model.safetensors",
        revision="63018cdac46394b3c12619bc9fd1cbe096c23b20",
    ),
    # mlx-generate-ctmr/README.md@v0.5.2.post2.
    "mlx-generate-ctmr-mr-brain": _HubSource(
        "MedMLX/mlx-generate-ctmr", "mr-brain", "model.safetensors",
        revision="63018cdac46394b3c12619bc9fd1cbe096c23b20",
    ),
    # mlx-reason-ct/README.md@v0.4.0.
    "mlx-reason-ct": _HubSource(
        "MedMLX/MLX-Reason-CT",
        revision="5e5b5202fd04ac9f63ede203ebc9327aa676ae94",
    ),
    # mlx-segment-ct/src/mlx_segment_ct/medmlx.py@v0.3.5.
    "mlx-segment-ct": _HubSource(
        "MedMLX/vista3d-ct-mlx",
        revision="e6f45856b4a59212ca10b44a9ade2e2d672f412b",
    ),
    # Model card linked by mlx-segment-ct/README.md@v0.3.5.
    "mlx-segment-ctmr": _HubSource(
        "MedMLX/vista3d-ctmr-mlx",
        revision="5965596ca734e3c62c30a93cde8d43d2edc8fb56",
    ),
    # Model card linked by nnunet-mlx/README.md@v0.7.0.
    "nnunet-autopet": _HubSource(
        "MedMLX/nnunet-autopet-mlx",
        revision="459d23802ae289be0f6539d40ca70c7facb1a9de",
    ),
    # Model card linked by nnunet-mlx/README.md@v0.7.0.
    "nnunet-brats2021": _HubSource(
        "MedMLX/nnunet-brats-mlx",
        revision="7632a1f3c74d350eb6f2daeb35b43b75ef7601bd",
    ),
    # nnunet-mlx/README.md@v0.7.0.
    "nnunet-liver": _HubSource(
        "MedMLX/nnunet-liver-mlx",
        revision="574a9c601b9b1a9e906ce985fbd46b106c5b7caa",
    ),
    # Model card linked by nnunet-mlx/README.md@v0.7.0.
    "nnunet-pancreas": _HubSource(
        "MedMLX/nnunet-pancreas-mlx",
        revision="b2d5f14bb31f9793e3d9ab2c24096427eea983ae",
    ),
    # radimagenet-mlx/README.md@v0.3.3.
    "radimagenet-densenet121": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-DenseNet121_notop.safetensors",
        revision="fd1c2663266360c899211c9a23ffc24f805d282a",
    ),
    # radimagenet-mlx/README.md@v0.3.3.
    "radimagenet-inceptionresnetv2": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-IRV2_notop.safetensors",
        revision="fd1c2663266360c899211c9a23ffc24f805d282a",
    ),
    # radimagenet-mlx/README.md@v0.3.3.
    "radimagenet-inceptionv3": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-InceptionV3_notop.safetensors",
        revision="fd1c2663266360c899211c9a23ffc24f805d282a",
    ),
    # radimagenet-mlx/README.md@v0.3.3.
    "radimagenet-resnet50": _HubSource(
        "MedMLX/radimagenet-mlx", filename="RadImageNet-ResNet50_notop.safetensors",
        revision="fd1c2663266360c899211c9a23ffc24f805d282a",
    ),
    # sam3-mlx/README.md@v0.2.4.
    "sam3": _HubSource(
        "mlx-community/sam3-image",
        revision="b72a14d8127e17e6f2a3d2e075bbbf4307ba146e",
    ),
    # Model card linked by segresnet-mlx/README.md@v0.3.3.
    "segresnet-renal-cect": _HubSource(
        "MedMLX/segresnet-renal-cect-mlx",
        revision="fa67ec6dca88f1dbbf6f52d6061b34ddefe45057",
    ),
    # Model card linked by swin-unetr-mlx/README.md@v0.2.6.
    "swin-unetr": _HubSource(
        "MedMLX/swin-unetr-mlx",
        revision="0052abb4bfb1c56bd7d52d5353267393399b8743",
    ),
    # synthseg-mlx/README.md@v0.3.4.
    "synthseg": _HubSource(
        "MedMLX/synthseg-mlx",
        revision="b4fa99020571cd5921f2205834dd5028fa804046",
    ),
    # totalsegmentator-mlx/README.md@v0.1.6.
    "totalsegmentator": _HubSource(
        "MedMLX/totalsegmentator-mlx",
        revision="6c03d6f90ccf53f224c3985cbdecf1eaa96555af",
    ),
    # Model card linked by unest-mlx/README.md@v0.2.7.
    "unest": _HubSource(
        "MedMLX/unest-renal-ct-mlx",
        revision="dea2e468a364f5f1675af372147077c5b6d74c85",
    ),
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
    under ``~/.cache/medmlx/<model_id>`` (or ``MEDMLX_ASSET_CACHE``). Use
    the registered model's pinned commit, or resolve ``main`` once for unlisted
    model IDs, then atomically record it after downloading. Completed folders
    need no network access.
    ``refresh=True`` downloads the selected revision again to repair damaged local
    files. Concurrent processes share a lock and the Hub's resumable downloads;
    interruption never marks a partial download done.

    Without a matching completion marker, reuse existing files only after checking
    every selected file against the installed commit's Hub hashes. Verified files
    from the older top-level layout are copied into the managed target, even when
    other files are missing, so the download only fetches the rest; the old files
    are retained and never linked to files a refresh could overwrite.

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
        try:
            revision = source.revision or hub.HfApi().model_info(repo).sha
            if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
                raise AssetNotReadyError(
                    f"The Hub did not return a commit for {repo}",
                    hint="Retry the download, or pass your own weights_path folder.",
                )
            patterns = _allow_patterns(source)
            reused = False
            legacy = cache / model_id
            if not refresh and (target.is_dir() or legacy.is_dir()):
                notify(f"Checking cached files for {model_id}…")
                info = hub.HfApi().model_info(repo, revision=revision, files_metadata=True)
                reused = _reuse_verified_files(info, revision, patterns, target, legacy)
            if reused:
                notify(f"Using verified cached weights for {model_id}.")
            else:
                notify(f"Downloading {repo}…")
                # Invalidate completion immediately before files can change. An
                # interrupted refresh resumes on the next ordinary call.
                (folder / _REVISION_FILE).unlink(missing_ok=True)
                with download_progress_bar(
                    model_id, f"Downloading {repo}…", download_progress,
                ) as tqdm_class:
                    hub.snapshot_download(
                        repo, revision=revision, local_dir=target, force_download=refresh,
                        allow_patterns=patterns, tqdm_class=tqdm_class,
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
        # after downloading or verifying every selected file at that revision.
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


def _allow_patterns(source: _HubSource) -> list[str] | None:
    """Use the same repository subset for verification and downloading."""
    if not source.subfolder and not source.filename:
        return None
    selected = f"{source.subfolder}/**" if source.subfolder else source.filename
    patterns = [selected]
    if source.companion_subfolder:
        # CT's learned mask model is a sibling of its image weights.
        patterns.append(f"{source.companion_subfolder}/**")
    return [*patterns, "README.md", "LICENSE*", "NOTICE", "SOURCE_LICENSE",
            "UPSTREAM_NOTICES.md", ".gitattributes"]


def _reuse_verified_files(
    info: _ModelInfo, revision: str, patterns: list[str] | None, target: Path, legacy: Path,
) -> bool:
    """Adopt hash-verified files and report whether the whole selection is present.

    Verified files from the older top-level layout are copied into the target even
    when the selection is incomplete, so a following download only fetches what is
    missing or changed (the Hub keeps local files whose hashes match). Old user
    files are never moved, linked or deleted.
    """
    if info.sha != revision or not info.siblings:
        return False
    selected = [entry for entry in info.siblings
                if patterns is None or any(fnmatch(entry.rfilename, p) for p in patterns)]
    if not selected:
        return False
    complete = True
    for entry in selected:
        relative = Path(entry.rfilename)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        destination = target / relative
        if _matches_hub_file(destination, entry):
            continue
        source = legacy / relative
        if legacy == target or not _matches_hub_file(source, entry):
            complete = False
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
        # The original may have changed between verification and copying.
        if not _matches_hub_file(destination, entry):
            destination.unlink(missing_ok=True)
            complete = False
    return complete


def _matches_hub_file(path: Path, entry: _RepoFile) -> bool:
    """Stream LFS content through SHA-256, or small files through Git's blob hash."""
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if entry.size is not None and size != entry.size:
                return False
            if entry.lfs is not None:
                expected = entry.lfs.sha256
                if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                    return False
                digest = sha256()
            else:
                expected = entry.blob_id
                if expected is None or not _REVISION.fullmatch(expected):
                    return False
                digest = sha1(f"blob {size}\0".encode("ascii"))
            return file_digest(stream, lambda: digest).hexdigest() == expected
    except OSError:
        # Missing, unreadable or partial files belong on the normal download path.
        return False


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
    primary_ready = not source.filename or (folder / source.filename).is_file()
    companion_ready = (
        not source.companion_subfolder
        or (target / source.companion_subfolder / source.filename).is_file()
    )
    return (cached_revision is not None
            and (source.revision is None or cached_revision == source.revision)
            and primary_ready
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
