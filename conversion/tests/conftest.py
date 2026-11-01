"""Isolate mutable native assets for each test session and remove them afterward."""

from collections.abc import Iterator
from contextlib import chdir
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_artifact_workspace() -> Iterator[None]:
    """Run tests and recipe subprocesses in a disposable system-temp workspace.

    Native save operations replace directories, so concurrent invocations need
    separate output paths. Source and fixture paths resolve from their owning
    modules; only generated test output moves and is removed on session exit.
    Collection alone does not allocate an artifact workspace.
    """
    with TemporaryDirectory(prefix="medmlx-convert-qa-") as directory, chdir(directory):
        yield
