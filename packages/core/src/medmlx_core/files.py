"""Check runner inputs and output targets, and publish a run's files together.

Every MedMLX runner takes keyword paths, refuses to replace existing outputs
unless ``overwrite`` is true, and must never leave a mix of old and new files
behind. These helpers hold that contract so runtimes don't each copy it:

- ``input_file`` and ``input_channels`` check caller paths before weights load.
- ``output_targets`` turns ``output_dir`` and role file names into checked paths.
- ``publish`` writes each file into a private staging folder next to its
  target, then replaces the targets together and restores the earlier files if
  any step fails.

Paths are expanded (``~``) but explicit symlinks are not resolved, except to
detect an output that is the same file as an input. No file contents are read.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

from medmlx_core.errors import InvalidInputError

type PathLike = str | os.PathLike[str]


def input_file(name: str, value: PathLike) -> Path:
    """Return the existing input file named by argument ``name``.

    Raises:
        InvalidInputError: ``value`` is not a path, or names no regular file.
    """
    path = _path(name, value)
    if not path.is_file():
        raise InvalidInputError(f"{name} not found: {path}")
    return path


def input_channels(
    image_paths: Mapping[str, PathLike], channels: Sequence[str],
) -> dict[str, Path]:
    """Return a case's channel files in the model's channel order.

    ``image_paths`` maps each of the model's ``channels`` to one existing file.
    Keys must match exactly, so a swapped or missing channel fails instead of
    silently changing the network input.

    Raises:
        InvalidInputError: ``image_paths`` is not a mapping, its keys differ from
            ``channels``, or a channel file does not exist.
    """
    if not isinstance(cast(object, image_paths), Mapping):
        raise InvalidInputError(
            f"image_paths must map each channel to a file: {', '.join(channels)}"
        )
    given = set(image_paths)
    expected = set(channels)
    if given != expected:
        missing = [name for name in channels if name not in given]
        extra = sorted(str(name) for name in given - expected)
        problems = [f"missing {', '.join(missing)}"] if missing else []
        problems += [f"unexpected {', '.join(extra)}"] if extra else []
        raise InvalidInputError(
            f"image_paths needs exactly {', '.join(channels)}; {'; '.join(problems)}"
        )
    return {name: input_file(f"image_paths[{name!r}]", image_paths[name]) for name in channels}


def output_targets(
    output_dir: PathLike, files: Mapping[str, str], *, overwrite: bool,
    inputs: Iterable[PathLike] = (),
) -> dict[str, Path]:
    """Return absolute output paths by role, after checking them against the run.

    ``files`` maps each output role to its file name inside ``output_dir``.
    Nothing is created; ``publish`` creates ``output_dir`` when it writes.

    Raises:
        InvalidInputError: ``overwrite`` is not a bool; ``output_dir`` or one of its
            parents is not a directory; a target is a directory, or exists (a
            dangling symlink counts) without ``overwrite``; or a target is the same
            path or file as an input or another target.
    """
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be true or false")
    root = _path("output_dir", output_dir).absolute()
    if any(path.exists() and not path.is_dir() for path in (root, *root.parents)):
        raise InvalidInputError(f"output_dir is not a directory: {root}")
    targets = {role: root / name for role, name in files.items()}
    for path in targets.values():
        if path.is_dir():
            raise InvalidInputError(f"Output path is a directory: {path}")
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise InvalidInputError(f"Output exists: {path}; pass overwrite=True to replace it")
    sources = [Path(source).expanduser() for source in inputs]
    for path in targets.values():
        for source in sources:
            if _same(path, source):
                raise InvalidInputError(f"Output {path} would replace input {source}")
    for first, second in combinations(targets.values(), 2):
        if _same(first, second):
            raise InvalidInputError(f"Outputs {first} and {second} are the same file")
    return targets


def publish(
    writes: Mapping[Path, Callable[[Path], object]], *, remove: Iterable[Path] = (),
) -> None:
    """Write a run's files and replace their targets together.

    Each writer receives a fresh path in a private staging folder created next
    to its target (in the same filesystem, so the final rename is atomic) and
    must write the complete file there. Only after every writer succeeds are
    existing targets hard-linked as backups, without following symlinks, and the
    staged files renamed over them; ``remove`` then deletes stale outputs. If a
    writer, rename or removal fails, every target is restored to its earlier
    file, or removed if it had none, and the error propagates. Staging folders
    are deleted afterwards, unless restoring failed, in which case the backups
    are kept and an ``OSError`` names their folder. Target parents are created.

    Concurrent writers to the same targets need their own lock; this function
    makes one run's files consistent, not exclusive.
    """
    if not writes:
        raise ValueError("publish needs at least one output")
    staging: dict[Path, Path] = {}
    cleanup = True
    try:
        partial: dict[Path, Path] = {}
        for index, (target, write) in enumerate(writes.items()):
            folder = target.parent
            if folder not in staging:
                folder.mkdir(parents=True, exist_ok=True)
                staging[folder] = Path(mkdtemp(prefix=".medmlx-staging-", dir=folder))
            partial[target] = staging[folder] / f"{index}-{target.name}"
            write(partial[target])
        stale = [path for path in remove if path not in writes]
        backups: dict[Path, Path] = {}
        for index, path in enumerate((*writes, *stale)):
            if path.exists() or path.is_symlink():
                if path.parent not in staging:
                    staging[path.parent] = Path(mkdtemp(prefix=".medmlx-staging-", dir=path.parent))
                backups[path] = staging[path.parent] / f"backup-{index}"
                os.link(path, backups[path], follow_symlinks=False)
        changed: list[Path] = []
        try:
            for target, staged in partial.items():
                os.replace(staged, target)
                changed.append(target)
            for path in stale:
                if path.exists() or path.is_symlink():
                    path.unlink()
                    changed.append(path)
        except BaseException:
            try:
                for path in reversed(changed):
                    backup = backups.get(path)
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
            except OSError as error:
                cleanup = False
                kept = ", ".join(str(folder) for folder in staging.values())
                raise OSError(f"Cannot restore outputs; backups retained at {kept}") from error
            raise
    finally:
        if cleanup:
            for folder in staging.values():
                shutil.rmtree(folder)


def _path(name: str, value: object) -> Path:
    if not isinstance(value, str | os.PathLike):
        raise InvalidInputError(f"{name} must be a path")
    text = os.fspath(cast(PathLike, value))
    if not text.strip():
        raise InvalidInputError(f"{name} must be a path")
    return Path(text).expanduser()


def _same(first: Path, second: Path) -> bool:
    """Whether two paths name one file, by resolved path or by existing inode."""
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


__all__ = ["PathLike", "input_channels", "input_file", "output_targets", "publish"]
