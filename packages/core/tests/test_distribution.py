"""Build artifacts retain runtime inputs while excluding local agent material."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def test_distributions_contain_runtime_and_not_local_agent_files(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(source / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(source / name, project / name)
    (project / ".gitignore").write_text("__pycache__/\n")
    runtime_files = {
        "medmlx_core/" + path.relative_to(source / "src/medmlx_core").as_posix()
        for path in (source / "src/medmlx_core").rglob("*.py")
        if not any(part.startswith(".") for part in path.relative_to(source / "src").parts)
    } | {"medmlx_core/py.typed"}
    for relative in (
        "aGeNtS.Md",
        "src/medmlx_core/graph/AgEnTs.md",
        "src/medmlx_core/.agents/state.py",
    ):
        private = project / relative
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text("synthetic local agent material\n")
    output = tmp_path / "dist"
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--offline",
            "--no-python-downloads",
            "--python",
            sys.executable,
            "--sdist",
            "--wheel",
            "--out-dir",
            str(output),
            str(project),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (wheel,) = output.glob("*.whl")
    (sdist,) = output.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        wheel_paths = {PurePosixPath(name) for name in archive.namelist()}
    with tarfile.open(sdist) as archive:
        source_paths = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if member.isfile()
        }
    for paths in (wheel_paths, source_paths):
        assert all(path.name.casefold() != "agents.md" for path in paths)
        assert all(
            not any(part.startswith(".") and part != ".gitignore" for part in path.parts)
            for path in paths
        )
        assert {"LICENSE", "THIRD_PARTY_NOTICES.md"} <= {path.name for path in paths}
    assert runtime_files <= {str(path) for path in wheel_paths}
    assert {"src/" + path for path in runtime_files} <= {str(path) for path in source_paths}
    assert {PurePosixPath(name) for name in ("pyproject.toml", "README.md")} <= source_paths
