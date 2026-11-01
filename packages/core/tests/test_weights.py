"""Automatic downloads resume safely and completed folders work offline."""

from __future__ import annotations

import fcntl
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import huggingface_hub
import pytest
from httpx import TimeoutException
from huggingface_hub.hf_api import RepoFile
from tqdm.auto import tqdm

from medmlx_core import _download_progress
from medmlx_core.errors import AssetNotReadyError
from medmlx_core.weights import DownloadProgress, cached_weights, weights_dir, weights_revision

REVISION = "a" * 40
type Download = Callable[..., str]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path))
    return tmp_path


def hub(
    monkeypatch: pytest.MonkeyPatch, download: Download, repo: str = "MedMLX/test-model",
) -> None:
    """Serve ``REVISION`` for ``repo`` and route snapshot downloads to ``download``."""

    class Hub:
        def model_info(self, requested: str) -> SimpleNamespace:
            assert requested == repo
            return SimpleNamespace(sha=REVISION)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)


def files(*names: str, calls: list[bool] | None = None) -> Download:
    """Write each named file that the requested allow patterns select."""

    def download(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
                 allow_patterns: list[str] | None = None,
                 tqdm_class: type[object] | None = None) -> str:
        assert revision == REVISION
        if calls is not None:
            calls.append(force_download)
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            if allow_patterns is None or any(fnmatch(name, pattern) for pattern in allow_patterns):
                (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (local_dir / name).write_text(name)
        return str(local_dir)

    return download


def test_download_uses_the_published_repository_and_keeps_other_revisions(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_folder = cache / "flexible-unet"
    old_folder.mkdir()
    (old_folder / ".medmlx-revision").write_text("b" * 40)
    hub(monkeypatch, files("flexibleunet.safetensors"), "MedMLX/flexible-unet-effnetb2-mlx")
    folder = weights_dir("flexible-unet")
    assert folder != old_folder and (folder / "flexibleunet.safetensors").is_file()
    assert weights_revision(folder) == REVISION
    assert weights_revision(old_folder) == "b" * 40


def test_nested_generation_bundle_returns_its_variant_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub(monkeypatch, files("mr/model.safetensors"), "MedMLX/mlx-generate-ctmr")
    folder = weights_dir("mlx-generate-ctmr-mr")
    assert folder.name == "mr" and (folder / "model.safetensors").is_file()
    assert weights_revision(folder) == REVISION


def test_ct_download_includes_separate_mask_bundle_for_offline_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ("ct/model.safetensors", "mask/model.safetensors", "mr/model.safetensors")
    hub(monkeypatch, files(*names), "MedMLX/mlx-generate-ctmr")
    ct = weights_dir("mlx-generate-ctmr-ct")
    assert (ct / "model.safetensors").is_file()
    assert (ct.parent / "mask" / "model.safetensors").is_file()
    assert not (ct.parent / "mr" / "model.safetensors").exists()
    # Older caches may contain a completion marker for CT alone.
    (ct.parent / "mask" / "model.safetensors").unlink()
    assert weights_dir("mlx-generate-ctmr-ct") == ct
    assert (ct.parent / "mask" / "model.safetensors").is_file()


def test_missing_selected_variant_is_not_marked_complete(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub(monkeypatch, files(), "MedMLX/mlx-generate-ctmr")
    with pytest.raises(AssetNotReadyError):
        weights_dir("mlx-generate-ctmr-mr")
    assert not list(cache.rglob(".medmlx-revision"))


def test_missing_automatic_source_requests_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "unused"))
    with pytest.raises(AssetNotReadyError):
        weights_dir("mrsegmentator")
    assert not (tmp_path / "unused").exists()


def test_sam3_uses_the_runtime_pinned_native_checkpoint(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
    old_folder = cache / "sam3" / "mlx-community--sam3-image"
    old_folder.mkdir(parents=True)
    (old_folder / ".medmlx-revision").write_text("b" * 40)

    def download(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
                 tqdm_class: type[object] | None = None) -> str:
        assert (repo, revision) == ("mlx-community/sam3-image", pinned)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    folder = weights_dir("sam3")
    assert (folder / "model.safetensors").is_file()
    assert weights_revision(folder) == pinned


def test_complete_folder_downloaded_once_then_usable_offline(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    hub(monkeypatch, files("model.safetensors", "CLIP_LICENSE", calls=calls))
    folder = weights_dir("test-model")
    assert folder == cache / "test-model"
    assert (folder / "CLIP_LICENSE").read_text() == "CLIP_LICENSE"
    assert not (folder / "model.safetensors").is_symlink()
    assert weights_revision(folder) == REVISION

    def offline() -> None:
        pytest.fail("A completed cache must not need network access")

    monkeypatch.setattr(huggingface_hub, "HfApi", offline)
    assert weights_dir("test-model") == folder
    assert len(calls) == 1


def test_interruption_resumes_partial_download_without_marking_it_complete(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def download(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
                 tqdm_class: type[object] | None = None) -> str:
        calls.append(repo)
        local_dir.mkdir(exist_ok=True)
        (local_dir / "first.safetensors").write_bytes(b"downloaded")
        if len(calls) == 1:
            raise KeyboardInterrupt
        assert (local_dir / "first.safetensors").read_bytes() == b"downloaded"
        (local_dir / "second.safetensors").write_bytes(b"resumed")
        return str(local_dir)

    hub(monkeypatch, download)
    with pytest.raises(KeyboardInterrupt):
        weights_dir("test-model")
    assert weights_revision(cache / "test-model") is None
    folder = weights_dir("test-model")
    assert (folder / "second.safetensors").read_bytes() == b"resumed"
    assert weights_revision(folder) == REVISION
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("error", "reported"),
    [(OSError("connection interrupted"), AssetNotReadyError),
     (TimeoutException("timeout"), AssetNotReadyError),
     (ValueError("invalid download parameters"), ValueError)],
    ids=["connection", "timeout", "unexpected"],
)
def test_failed_download_leaves_no_completed_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, reported: type[Exception],
) -> None:
    """Connectivity failures become retryable asset errors; other errors are not disguised."""

    def failed(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
               tqdm_class: type[object] | None = None) -> str:
        raise error

    hub(monkeypatch, failed)
    with pytest.raises(reported):
        weights_dir("test-model")
    assert weights_revision(cache / "test-model") is None


def test_concurrent_first_use_shares_one_download(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    hub(monkeypatch, files("model.safetensors", calls=calls))
    with ThreadPoolExecutor(max_workers=2) as pool:
        folders = list(pool.map(weights_dir, ["test-model", "test-model"]))
    assert folders == [cache / "test-model"] * 2
    assert len(calls) == 1


@pytest.mark.parametrize("model_id", ["../outside", "owner/model", "UPPER"])
def test_invalid_model_ids_never_create_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str,
) -> None:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "unused"))
    with pytest.raises(ValueError):
        weights_dir(model_id)
    assert not (tmp_path / "unused").exists()


def test_refresh_repairs_cache_and_failed_refresh_is_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def download(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
                 tqdm_class: type[object] | None = None) -> str:
        calls.append(force_download)
        local_dir.mkdir(exist_ok=True)
        if len(calls) == 3:
            raise OSError("interrupted refresh")
        (local_dir / "model.safetensors").write_bytes(b"repaired")
        return str(local_dir)

    hub(monkeypatch, download)
    folder = weights_dir("test-model")
    (folder / "model.safetensors").unlink()
    assert weights_dir("test-model", refresh=True) == folder
    assert (folder / "model.safetensors").read_bytes() == b"repaired"
    with pytest.raises(AssetNotReadyError):
        weights_dir("test-model", refresh=True)
    assert weights_revision(folder) is None
    weights_dir("test-model")
    assert weights_revision(folder) == REVISION
    assert calls == [False, True, True, False]


def test_cached_weights_never_downloads_locks_or_creates_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness and setup plans may ask about the cache; they must never download."""
    root = cache / "unused"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(root))

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("cached_weights must not contact the Hub or take a lock")

    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    monkeypatch.setattr(fcntl, "flock", unexpected)
    assert cached_weights("test-model") is None
    assert cached_weights("medgemma") is None
    assert cached_weights("mrsegmentator") is None
    assert not root.exists()

    folder = root / "test-model"
    folder.mkdir(parents=True)
    marker = folder / ".medmlx-revision"
    marker.write_text(REVISION)
    assert cached_weights("test-model") == folder
    assert marker.read_text() == REVISION
    assert sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) == [
        "test-model", "test-model/.medmlx-revision",
    ]


@pytest.mark.parametrize("marker", [None, "not a revision"], ids=["partial", "invalid-marker"])
def test_cached_weights_requires_a_valid_completion_marker(cache: Path, marker: str | None) -> None:
    folder = cache / "test-model"
    folder.mkdir()
    (folder / "README.md").write_text("Partial download")
    if marker is not None:
        (folder / ".medmlx-revision").write_text(marker)
    assert cached_weights("test-model") is None


@pytest.mark.parametrize(
    ("model_id", "relative_folder"),
    [
        ("test-model", "test-model"),
        ("flexible-unet", "flexible-unet/MedMLX--flexible-unet-effnetb2-mlx"),
        ("mlx-generate-ctmr-mr", "mlx-generate-ctmr-mr/MedMLX--mlx-generate-ctmr/mr"),
    ],
    ids=["default-repo", "published-repo", "source-subfolder"],
)
def test_cached_weights_reads_the_model_folder_in_the_custom_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, relative_folder: str,
) -> None:
    root = cache / "custom-assets"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(root))
    folder = root / relative_folder
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(REVISION + "\n")
    assert cached_weights(model_id) == folder


@pytest.mark.parametrize("matches_pin", [False, True], ids=["mismatch", "matching"])
def test_cached_weights_requires_the_runtime_pinned_revision(
    cache: Path, matches_pin: bool,
) -> None:
    pinned = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
    folder = cache / "sam3" / "mlx-community--sam3-image"
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(pinned if matches_pin else REVISION)
    assert cached_weights("sam3") == (folder if matches_pin else None)


def test_cached_weights_requires_the_companion_file(cache: Path) -> None:
    target = cache / "mlx-generate-ctmr-ct" / "MedMLX--mlx-generate-ctmr"
    folder = target / "ct"
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(REVISION)
    assert cached_weights("mlx-generate-ctmr-ct") is None

    companion = target / "mask" / "model.safetensors"
    companion.parent.mkdir()
    companion.touch()
    assert cached_weights("mlx-generate-ctmr-ct") == folder


@pytest.mark.parametrize("byte_reporting", [True, False], ids=["bytes", "legacy-strings"])
def test_progress_throttles_byte_updates_and_always_flushes_the_final_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], byte_reporting: bool,
) -> None:
    now = 0.0
    monkeypatch.setattr(_download_progress, "monotonic", lambda: now)
    messages: list[str] = []
    events: list[tuple[float, DownloadProgress]] = []

    def record(event: DownloadProgress) -> None:
        events.append((now, event))

    def download(
        repo: str, *, local_dir: Path, tqdm_class: type[tqdm[object]], **_kwargs: object,
    ) -> str:
        nonlocal now
        # Hub 1.23 creates both byte streams and a file-count bar. Only completed
        # file bytes describe progress toward the known file sizes.
        transfer = tqdm_class((), total=0, unit="B", desc="Downloading bytes")
        reconstructed = tqdm_class(
            (), total=0, unit="B", desc="Reconstructing (incomplete total...)",
        )
        assert list(tqdm_class(range(3), desc="Fetching 3 files")) == [0, 1, 2]
        now = 0.49
        reconstructed.total = 100
        reconstructed.refresh()
        reconstructed.update(10)
        now = 0.5
        transfer.update(999)
        reconstructed.update(20)
        now = 0.7
        reconstructed.total = 200
        reconstructed.refresh()
        reconstructed.update(50)
        now = 1.0
        reconstructed.update(40)
        now = 1.1
        reconstructed.update(80)
        reconstructed.close()
        transfer.close()
        return str(local_dir)

    hub(monkeypatch, download)
    if byte_reporting:
        folder = weights_dir("test-model", progress=messages.append, download_progress=record)
    else:
        folder = weights_dir("test-model", progress=messages.append)
    assert messages == ["Downloading MedMLX/test-model…", "Weights are available for test-model."]
    if byte_reporting:
        assert events == [
            (0.0, {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
                   "bytes": 0, "total_bytes": None}),
            (0.5, {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
                   "bytes": 30, "total_bytes": 100}),
            (1.0, {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
                   "bytes": 120, "total_bytes": 200}),
            (1.1, {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
                   "bytes": 200, "total_bytes": 200}),
        ]
    else:
        assert events == []
    assert weights_dir("test-model", progress=messages.append, download_progress=record) == folder
    assert messages[-1] == "Using cached weights for test-model."
    assert len(events) == (4 if byte_reporting else 0)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("interrupted", [False, True], ids=["completed", "interrupted"])
def test_progress_keeps_unknown_totals_and_resumed_bytes_even_on_interruption(
    cache: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool,
) -> None:
    monkeypatch.setattr(_download_progress, "monotonic", lambda: 0.0)
    events: list[DownloadProgress] = []

    def download(
        repo: str, *, local_dir: Path, tqdm_class: type[tqdm[object]], **_kwargs: object,
    ) -> str:
        bar = tqdm_class((), total=None, initial=7, unit="B")
        bar.update(20)
        if interrupted:
            raise KeyboardInterrupt("cancelled")
        return str(local_dir)

    hub(monkeypatch, download)
    if interrupted:
        with pytest.raises(KeyboardInterrupt, match="cancelled"):
            weights_dir("test-model", download_progress=events.append)
        assert weights_revision(cache / "test-model") is None
    else:
        folder = weights_dir("test-model", download_progress=events.append)
        assert weights_revision(folder) == REVISION
    assert events == [
        {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
         "bytes": 7, "total_bytes": None},
        {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
         "bytes": 27, "total_bytes": None},
    ]


def test_real_hub_snapshot_hook_aggregates_concurrent_files_without_rendering_bars(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the installed Hub orchestration, replacing only metadata and file IO."""
    monkeypatch.setattr(_download_progress, "monotonic", lambda: 0.0)

    def info(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha=REVISION)

    def tree(*_args: object, **_kwargs: object) -> list[RepoFile]:
        return [RepoFile(path="first", size=40, oid="b" * 40),
                RepoFile(path="second", size=60, oid="c" * 40)]

    class FileProgress(Protocol):
        def __init__(self, **kwargs: object) -> None: ...
        def update(self, n: int) -> None: ...
        def update_transfer(self, n: int) -> None: ...

    def download(
        repo: str, filename: str, *, local_dir: str | Path, tqdm_class: type[FileProgress],
        **_kwargs: object,
    ) -> str:
        size = 40 if filename == "first" else 60
        bar = tqdm_class(total=size, initial=10)
        bar.update(size - 10)
        bar.update_transfer(5)  # Xet's compressed transfer count is a separate stream.
        directory = Path(local_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_bytes(b"x" * size)
        return str(path)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", info)
    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", info)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", tree)
    monkeypatch.setattr(import_module("huggingface_hub._snapshot_download"),
                        "hf_hub_download", download)
    events: list[DownloadProgress] = []
    folder = weights_dir("test-model", download_progress=events.append)
    assert (folder / "first").stat().st_size == 40
    assert (folder / "second").stat().st_size == 60
    assert events == [
        {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
         "bytes": 0, "total_bytes": None},
        {"message": "Downloading MedMLX/test-model…", "model_id": "test-model",
         "bytes": 100, "total_bytes": 100},
    ]
    assert capsys.readouterr() == ("", "")

