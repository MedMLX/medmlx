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
from huggingface_hub.hf_api import BlobLfsInfo, RepoFile, RepoSibling
from tqdm.auto import tqdm

from medmlx_core import _download_progress
from medmlx_core.errors import AssetNotReadyError
from medmlx_core.weights import (
    DownloadProgress,
    cached_weights,
    weights_dir,
    weights_repo,
    weights_revision,
)

REVISION = "a" * 40
CTMR_REVISION = "63018cdac46394b3c12619bc9fd1cbe096c23b20"
FLEXIBLE_UNET_REVISION = "ccbbc596f2a9da8477e25cba0d7548cbb2191feb"

# Expected commits from the runtime release documents and their linked model cards.
PINNED_BUNDLES = [
    ("biomedparse-v2", "MedMLX/biomedparse-v2-mlx",
     "1c251c6561aa143cd79ff0ccdfa53a1a6e60c7f0"),
    ("flexible-unet", "MedMLX/flexible-unet-effnetb2-mlx",
     FLEXIBLE_UNET_REVISION),
    ("flexray", "MedMLX/flexray-mlx",
     "722d8341570e8b9e1196cf50e57892da84974fcc"),
    ("lesiondiffusion", "MedMLX/lesiondiffusion-mlx",
     "d00bbb8ad1c7170062f60b4386a38c3a4178f7ce"),
    ("lung-nodule", "MedMLX/retinanet-lung-nodule-mlx",
     "d2d635fbac34254f14192ac348a05d275a949317"),
    ("medsam2", "MedMLX/medsam2-ct-lesion-mlx",
     "22f5e6dc4850a71abf92ef1a8fea0db14c34d460"),
    ("mlx-generate-ctmr-ct", "MedMLX/mlx-generate-ctmr",
     CTMR_REVISION),
    ("mlx-generate-ctmr-mr", "MedMLX/mlx-generate-ctmr",
     CTMR_REVISION),
    ("mlx-generate-ctmr-mr-brain", "MedMLX/mlx-generate-ctmr",
     CTMR_REVISION),
    ("mlx-reason-ct", "MedMLX/MLX-Reason-CT",
     "5e5b5202fd04ac9f63ede203ebc9327aa676ae94"),
    ("mlx-segment-ct", "MedMLX/vista3d-ct-mlx",
     "e6f45856b4a59212ca10b44a9ade2e2d672f412b"),
    ("mlx-segment-ctmr", "MedMLX/vista3d-ctmr-mlx",
     "5965596ca734e3c62c30a93cde8d43d2edc8fb56"),
    ("nnunet-autopet", "MedMLX/nnunet-autopet-mlx",
     "459d23802ae289be0f6539d40ca70c7facb1a9de"),
    ("nnunet-brats2021", "MedMLX/nnunet-brats-mlx",
     "7632a1f3c74d350eb6f2daeb35b43b75ef7601bd"),
    ("nnunet-liver", "MedMLX/nnunet-liver-mlx",
     "574a9c601b9b1a9e906ce985fbd46b106c5b7caa"),
    ("nnunet-pancreas", "MedMLX/nnunet-pancreas-mlx",
     "b2d5f14bb31f9793e3d9ab2c24096427eea983ae"),
    ("radimagenet-densenet121", "MedMLX/radimagenet-mlx",
     "fd1c2663266360c899211c9a23ffc24f805d282a"),
    ("radimagenet-inceptionresnetv2", "MedMLX/radimagenet-mlx",
     "fd1c2663266360c899211c9a23ffc24f805d282a"),
    ("radimagenet-inceptionv3", "MedMLX/radimagenet-mlx",
     "fd1c2663266360c899211c9a23ffc24f805d282a"),
    ("radimagenet-resnet50", "MedMLX/radimagenet-mlx",
     "fd1c2663266360c899211c9a23ffc24f805d282a"),
    ("sam3", "mlx-community/sam3-image",
     "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"),
    ("segresnet-brats", "MedMLX/segresnet-brats-mlx",
     "14f9cda3649965729663c7365c49fd839c81a01b"),
    ("segresnet-renal-cect", "MedMLX/segresnet-renal-cect-mlx",
     "fa67ec6dca88f1dbbf6f52d6061b34ddefe45057"),
    ("swin-unetr", "MedMLX/swin-unetr-mlx",
     "0052abb4bfb1c56bd7d52d5353267393399b8743"),
    ("synthseg", "MedMLX/synthseg-mlx",
     "b4fa99020571cd5921f2205834dd5028fa804046"),
    ("totalsegmentator", "MedMLX/totalsegmentator-mlx",
     "6c03d6f90ccf53f224c3985cbdecf1eaa96555af"),
    ("unest", "MedMLX/unest-renal-ct-mlx",
     "dea2e468a364f5f1675af372147077c5b6d74c85"),
]
BUNDLE_LAYOUTS = {
    "mlx-generate-ctmr-ct": ("ct", "model.safetensors"),
    "mlx-generate-ctmr-mr": ("mr", "model.safetensors"),
    "mlx-generate-ctmr-mr-brain": ("mr-brain", "model.safetensors"),
    "radimagenet-densenet121": ("", "RadImageNet-DenseNet121_notop.safetensors"),
    "radimagenet-inceptionresnetv2": ("", "RadImageNet-IRV2_notop.safetensors"),
    "radimagenet-inceptionv3": ("", "RadImageNet-InceptionV3_notop.safetensors"),
    "radimagenet-resnet50": ("", "RadImageNet-ResNet50_notop.safetensors"),
}
type Download = Callable[..., str]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path))
    return tmp_path


def hub(
    monkeypatch: pytest.MonkeyPatch, download: Download, repo: str = "MedMLX/test-model",
    *, revision: str = REVISION, siblings: list[RepoSibling] | None = None,
    metadata_requests: list[tuple[str | None, bool]] | None = None,
) -> None:
    """Serve a revision and file hashes, routing snapshot downloads to ``download``."""
    resolved = revision

    class Hub:
        def model_info(
            self, requested: str, *, revision: str | None = None, files_metadata: bool = False,
        ) -> SimpleNamespace:
            assert requested == repo
            assert revision in (None, resolved)
            if metadata_requests is not None:
                metadata_requests.append((revision, files_metadata))
            return SimpleNamespace(sha=resolved, siblings=siblings)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)


def cache_files(weight_name: str = "network.safetensors") -> list[RepoSibling]:
    """Hub hashes for literal ``b'weights'`` and ``b'hello\\n'`` file contents."""
    return [
        RepoSibling(weight_name, size=7, blob_id="b" * 40, lfs=BlobLfsInfo(
            size=7, pointer_size=126,
            sha256="9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        )),
        RepoSibling("README.md", size=6, blob_id="ce013625030ba8dba906f756967f9e9ca394464a"),
    ]


@pytest.mark.parametrize("marker", [None, "interrupted"], ids=["missing", "invalid"])
def test_unmarked_cache_is_reused_only_after_all_hub_hashes_match(
    cache: Path, monkeypatch: pytest.MonkeyPatch, marker: str | None,
) -> None:
    folder = cache / "test-model"
    folder.mkdir()
    (folder / "network.safetensors").write_bytes(b"weights")
    (folder / "README.md").write_bytes(b"hello\n")
    if marker is not None:
        (folder / ".medmlx-revision").write_text(marker)
    requests: list[tuple[str | None, bool]] = []

    def unexpected(*_args: object, **_kwargs: object) -> str:
        pytest.fail("Hash-matching files must be reused without a snapshot download")

    hub(monkeypatch, unexpected, siblings=cache_files(), metadata_requests=requests)
    assert cached_weights("test-model") is None
    assert weights_dir("test-model") == folder
    assert weights_revision(folder) == REVISION
    assert requests == [(None, False), (REVISION, True)]
    assert cached_weights("test-model") == folder
    assert (folder / "network.safetensors").read_bytes() == b"weights"
    assert (folder / "README.md").read_bytes() == b"hello\n"

    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected)
    assert weights_dir("test-model") == folder


@pytest.mark.parametrize(
    ("weight_bytes", "readme_bytes"),
    [(b"damaged", b"hello\n"), (b"weights", b"hallo\n"), (b"wei", None)],
    ids=["same-size-lfs-mismatch", "same-size-git-mismatch", "partial-files"],
)
def test_unmarked_mismatched_or_partial_files_resume_download(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
    weight_bytes: bytes, readme_bytes: bytes | None,
) -> None:
    folder = cache / "test-model"
    folder.mkdir()
    (folder / "network.safetensors").write_bytes(weight_bytes)
    if readme_bytes is not None:
        (folder / "README.md").write_bytes(readme_bytes)
    calls: list[str] = []

    def download(repo: str, *, revision: str, local_dir: Path, **_kwargs: object) -> str:
        assert revision == REVISION
        assert weights_revision(local_dir) is None
        calls.append(repo)
        (local_dir / "network.safetensors").write_bytes(b"weights")
        (local_dir / "README.md").write_bytes(b"hello\n")
        return str(local_dir)

    hub(monkeypatch, download, siblings=cache_files())
    assert weights_dir("test-model") == folder
    assert calls == ["MedMLX/test-model"]
    assert weights_revision(folder) == REVISION
    assert (folder / "network.safetensors").read_bytes() == b"weights"
    assert (folder / "README.md").read_bytes() == b"hello\n"


@pytest.mark.parametrize("matches_pin", [True, False], ids=["reuse", "download"])
def test_legacy_files_are_checked_at_the_pin_and_copied_without_changing_originals(
    cache: Path, monkeypatch: pytest.MonkeyPatch, matches_pin: bool,
) -> None:
    pinned = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
    legacy = cache / "sam3"
    legacy.mkdir()
    original = b"weights" if matches_pin else b"damaged"
    (legacy / "model.safetensors").write_bytes(original)
    (legacy / "README.md").write_bytes(b"hello\n")
    (legacy / ".medmlx-revision").write_text(REVISION)
    requests: list[tuple[str | None, bool]] = []
    calls: list[str] = []

    def download(repo: str, *, revision: str, local_dir: Path, **_kwargs: object) -> str:
        assert revision == pinned
        calls.append(repo)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "model.safetensors").write_bytes(b"weights")
        (local_dir / "README.md").write_bytes(b"hello\n")
        return str(local_dir)

    hub(monkeypatch, download, "mlx-community/sam3-image", revision=pinned,
        siblings=cache_files("model.safetensors"), metadata_requests=requests)
    folder = weights_dir("sam3")
    assert folder == legacy / "mlx-community--sam3-image"
    assert calls == ([] if matches_pin else ["mlx-community/sam3-image"])
    assert requests == [(pinned, True)]
    assert weights_revision(folder) == pinned
    assert weights_revision(legacy) == REVISION
    assert (folder / "model.safetensors").read_bytes() == b"weights"
    # Copies stay independent when a later refresh or local edit changes the target.
    (folder / "model.safetensors").write_bytes(b"changed")
    assert (legacy / "model.safetensors").read_bytes() == original
    assert (legacy / "README.md").read_bytes() == b"hello\n"


def test_verified_legacy_files_are_kept_when_the_hub_added_files(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large verified weights are not downloaded again for a few new small files."""
    pinned = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
    legacy = cache / "sam3"
    legacy.mkdir()
    (legacy / "model.safetensors").write_bytes(b"weights")
    # The repository has since gained or changed README.md.
    seen: list[bytes] = []

    def download(repo: str, *, revision: str, local_dir: Path, **_kwargs: object) -> str:
        assert revision == pinned
        seen.append((local_dir / "model.safetensors").read_bytes())
        (local_dir / "README.md").write_bytes(b"hello\n")
        return str(local_dir)

    hub(monkeypatch, download, "mlx-community/sam3-image", revision=pinned,
        siblings=cache_files("model.safetensors"))
    folder = weights_dir("sam3")
    assert seen == [b"weights"]
    assert weights_revision(folder) == pinned
    assert (legacy / "model.safetensors").read_bytes() == b"weights"
    assert not (legacy / "README.md").exists()


@pytest.mark.parametrize("with_companion", [True, False], ids=["complete", "missing-mask"])
def test_cache_verification_selects_the_same_files_and_companions_as_downloads(
    cache: Path, monkeypatch: pytest.MonkeyPatch, with_companion: bool,
) -> None:
    target = cache / "mlx-generate-ctmr-ct" / "MedMLX--mlx-generate-ctmr"
    (target / "ct").mkdir(parents=True)
    (target / "ct/model.safetensors").write_bytes(b"weights")
    (target / "README.md").write_bytes(b"hello\n")
    if with_companion:
        (target / "mask").mkdir()
        (target / "mask/model.safetensors").write_bytes(b"weights")
    siblings = [*cache_files("ct/model.safetensors"),
                cache_files("mask/model.safetensors")[0],
                cache_files("mr/model.safetensors")[0]]
    calls: list[bool] = []
    hub(monkeypatch, files("ct/model.safetensors", "mask/model.safetensors", calls=calls,
                           expected_revision=CTMR_REVISION),
        "MedMLX/mlx-generate-ctmr", siblings=siblings, revision=CTMR_REVISION)
    assert weights_dir("mlx-generate-ctmr-ct") == target / "ct"
    assert calls == ([] if with_companion else [False])
    assert weights_revision(target / "ct") == CTMR_REVISION
    assert not (target / "mr").exists()


def test_missing_hub_hashes_cannot_certify_an_existing_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = cache / "test-model"
    folder.mkdir()
    (folder / "network.safetensors").write_bytes(b"weights")
    calls: list[bool] = []
    hub(monkeypatch, files("network.safetensors", calls=calls),
        siblings=[RepoSibling("network.safetensors", size=7)])
    assert weights_dir("test-model") == folder
    assert calls == [False]
    assert weights_revision(folder) == REVISION


def test_refresh_downloads_even_when_existing_files_match_hub_hashes(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = cache / "test-model"
    folder.mkdir()
    (folder / "network.safetensors").write_bytes(b"weights")
    (folder / "README.md").write_bytes(b"hello\n")
    (folder / ".medmlx-revision").write_text(REVISION)
    requests: list[tuple[str | None, bool]] = []
    calls: list[bool] = []
    hub(monkeypatch, files("network.safetensors", "README.md", calls=calls),
        siblings=cache_files(), metadata_requests=requests)
    assert weights_dir("test-model", refresh=True) == folder
    assert calls == [True]
    assert requests == [(None, False)]
    assert weights_revision(folder) == REVISION


def files(
    *names: str, calls: list[bool] | None = None, expected_revision: str = REVISION,
) -> Download:
    """Write each named file that the requested allow patterns select."""

    def download(repo: str, *, revision: str, local_dir: Path, force_download: bool = False,
                 allow_patterns: list[str] | None = None,
                 tqdm_class: type[object] | None = None) -> str:
        assert revision == expected_revision
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
    hub(monkeypatch, files("flexibleunet.safetensors", expected_revision=FLEXIBLE_UNET_REVISION),
        "MedMLX/flexible-unet-effnetb2-mlx", revision=FLEXIBLE_UNET_REVISION)
    folder = weights_dir("flexible-unet")
    assert folder != old_folder and (folder / "flexibleunet.safetensors").is_file()
    assert weights_revision(folder) == FLEXIBLE_UNET_REVISION
    assert weights_revision(old_folder) == "b" * 40


def test_nested_generation_bundle_returns_its_variant_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub(monkeypatch, files("mr/model.safetensors", expected_revision=CTMR_REVISION),
        "MedMLX/mlx-generate-ctmr", revision=CTMR_REVISION)
    folder = weights_dir("mlx-generate-ctmr-mr")
    assert folder.name == "mr" and (folder / "model.safetensors").is_file()
    assert weights_revision(folder) == CTMR_REVISION


def test_ct_download_includes_separate_mask_bundle_for_offline_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ("ct/model.safetensors", "mask/model.safetensors", "mr/model.safetensors")
    hub(monkeypatch, files(*names, expected_revision=CTMR_REVISION),
        "MedMLX/mlx-generate-ctmr", revision=CTMR_REVISION)
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
    hub(monkeypatch, files(expected_revision=CTMR_REVISION),
        "MedMLX/mlx-generate-ctmr", revision=CTMR_REVISION)
    with pytest.raises(AssetNotReadyError, match=r"Missing model\.safetensors"):
        weights_dir("mlx-generate-ctmr-mr")
    assert not list(cache.rglob(".medmlx-revision"))


def test_missing_automatic_source_requests_local_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "unused"))
    with pytest.raises(AssetNotReadyError):
        weights_dir("mrsegmentator")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(("model_id", "repo", "pinned"), PINNED_BUNDLES)
def test_registered_bundles_use_release_pins_and_reuse_completed_caches_offline(
    cache: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, repo: str, pinned: str,
) -> None:
    assert weights_repo(model_id) == repo
    target = cache / model_id
    if repo.lower() != f"medmlx/{model_id}":
        target /= repo.replace("/", "--")
    subfolder, filename = BUNDLE_LAYOUTS.get(model_id, ("", "network.safetensors"))
    folder = target / subfolder
    folder.mkdir(parents=True)
    marker = folder / ".medmlx-revision"
    marker.write_text(REVISION)
    assert cached_weights(model_id) is None
    names = [(Path(subfolder) / filename).as_posix()]
    if model_id == "mlx-generate-ctmr-ct":
        names.append("mask/model.safetensors")
    calls: list[bool] = []
    requests: list[tuple[str | None, bool]] = []
    hub(monkeypatch, files(*names, calls=calls, expected_revision=pinned), repo,
        revision=pinned, metadata_requests=requests)
    assert weights_dir(model_id) == folder
    assert (folder / filename).is_file()
    assert marker.read_text() == pinned + "\n"
    assert calls == [False]
    assert requests == [(pinned, True)]

    def unexpected(*_args: object, **_kwargs: object) -> str:
        pytest.fail("A completed cache at the pinned commit must remain usable offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected)
    assert cached_weights(model_id) == folder
    assert weights_dir(model_id) == folder
    assert marker.read_text() == pinned + "\n"


@pytest.mark.parametrize(
    ("model_id", "repo", "pinned"),
    [bundle for bundle in PINNED_BUNDLES if bundle[0].startswith("nnunet-")],
)
def test_nnunet_cache_from_another_commit_is_verified_at_the_label_map_pin(
    cache: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, repo: str, pinned: str,
) -> None:
    folder = cache / model_id / repo.replace("/", "--")
    folder.mkdir(parents=True)
    (folder / "network.safetensors").write_bytes(b"weights")
    (folder / "README.md").write_bytes(b"hello\n")
    (folder / ".medmlx-revision").write_text(REVISION)
    requests: list[tuple[str | None, bool]] = []

    def unexpected(*_args: object, **_kwargs: object) -> str:
        pytest.fail("Weights matching the pinned hashes must be reused across commits")

    hub(monkeypatch, unexpected, repo, revision=pinned, siblings=cache_files(),
        metadata_requests=requests)
    assert cached_weights(model_id) is None
    assert weights_dir(model_id) == folder
    assert weights_revision(folder) == pinned
    assert requests == [(pinned, True)]
    assert cached_weights(model_id) == folder


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
                 allow_patterns: list[str] | None = None,
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
               allow_patterns: list[str] | None = None,
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
                 allow_patterns: list[str] | None = None,
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
    ("model_id", "relative_folder", "revision"),
    [
        ("test-model", "test-model", REVISION),
        ("flexible-unet", "flexible-unet/MedMLX--flexible-unet-effnetb2-mlx",
         FLEXIBLE_UNET_REVISION),
        ("mlx-generate-ctmr-mr", "mlx-generate-ctmr-mr/MedMLX--mlx-generate-ctmr/mr",
         CTMR_REVISION),
    ],
    ids=["default-repo", "published-repo", "source-subfolder"],
)
def test_cached_weights_reads_the_model_folder_in_the_custom_cache(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
    model_id: str, relative_folder: str, revision: str,
) -> None:
    root = cache / "custom-assets"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(root))
    folder = root / relative_folder
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(revision + "\n")
    if model_id in BUNDLE_LAYOUTS:
        (folder / BUNDLE_LAYOUTS[model_id][1]).touch()
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
    (folder / ".medmlx-revision").write_text(CTMR_REVISION)
    (folder / "model.safetensors").touch()
    assert cached_weights("mlx-generate-ctmr-ct") is None

    companion = target / "mask" / "model.safetensors"
    companion.parent.mkdir()
    companion.touch()
    assert cached_weights("mlx-generate-ctmr-ct") == folder


def test_deleted_primary_weights_are_downloaded_again(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = cache / "mlx-generate-ctmr-mr" / "MedMLX--mlx-generate-ctmr" / "mr"
    folder.mkdir(parents=True)
    (folder / ".medmlx-revision").write_text(CTMR_REVISION)
    assert cached_weights("mlx-generate-ctmr-mr") is None
    downloads: list[str] = []

    def download(repo: str, *, local_dir: Path, **_kwargs: object) -> str:
        downloads.append(repo)
        (local_dir / "mr/model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    hub(monkeypatch, download, "MedMLX/mlx-generate-ctmr", revision=CTMR_REVISION)
    assert weights_dir("mlx-generate-ctmr-mr") == folder
    assert downloads == ["MedMLX/mlx-generate-ctmr"]
    assert cached_weights("mlx-generate-ctmr-mr") == folder


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

