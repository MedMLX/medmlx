"""Pinned upstream source fetched from GitHub, for packages that aren't installable as-is."""

import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import torch


def checkpoint_archive(url: str, cache_key: str) -> Path:
    """Download an archive once and stage only nnU-Net inference assets.

    Keep the caller's release cache key stable so existing archives are reused.
    """
    root = Path(torch.hub.get_dir()) / "nnunet-archives" / cache_key
    marker = root / ".complete"
    if marker.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root.with_suffix(".zip")
    if not archive.is_file():
        torch.hub.download_url_to_file(url, str(archive))
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():
            if Path(member).name not in {"plans.json", "dataset.json", "checkpoint_final.pth"}:
                continue
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise ValueError(f"Unsafe checkpoint archive path: {member}")
            zipped.extract(member, root)
    marker.touch()
    return root


def github_source(repo: str, commit: str, rename: Callable[[str], str | None]) -> Path:
    """Extract files from ``repo`` at ``commit`` into the torch hub cache and return the root.

    ``rename`` maps each archive path (without the top directory) to its path under the root,
    or None to skip it, so prebuilt binaries and unrelated files are never extracted.
    """
    root = Path(torch.hub.get_dir()) / "github" / repo.replace("/", "__") / commit
    marker = root / ".complete"
    if not marker.exists():
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.tar.gz"
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive) as tar:
                members: list[tarfile.TarInfo] = []
                for member in tar.getmembers():
                    target = rename(member.name.split("/", 1)[-1])
                    if member.isfile() and target is not None:
                        member.name = target
                        members.append(member)
                tar.extractall(root, members=members, filter="data")
        marker.touch()
    return root
