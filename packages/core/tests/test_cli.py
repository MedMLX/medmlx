"""The generated command parses a runner's signature, runs it and reports its result."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from medmlx_core.cli import main
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

if TYPE_CHECKING:
    from medmlx_core.outputs import RunResult

PROG = "medmlx-core"  # an installed distribution, which --version reads


def run(
    *,
    image_path: str | os.PathLike[str],
    output_dir: Path,
    weights_path: str | os.PathLike[str] | None = None,
    steps: int = 3,
    threshold: float = 0.5,
    overwrite: bool = False,
    smooth: bool = True,
    mode: Literal["standard", "fast"] = "standard",
    center: Sequence[float] | None = None,
    channels: Mapping[str, str | os.PathLike[str]] | None = None,
    options: Mapping[str, object] | None = None,
) -> RunResult[dict[str, object]]:
    """Segment one image.

    Args:
        image_path: Input image.
        output_dir: Folder for results.
        threshold: Probability cutoff, 100% sure.
    """
    return json.loads(json.dumps(locals(), default=str))


def test_arguments_become_keywords_and_result_is_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(
        run,
        prog=PROG,
        argv=[
            "--image-path", "a.nii.gz", "--output-dir", "out", "--steps", "7",
            "--threshold", "0.25", "--overwrite", "--no-smooth", "--mode", "fast",
            "--center", "-1", "2.5", "3", "--channels", "t1=a.nii", "--channels", "t2=b.nii",
            "--options", '{"k": [1, 2]}',
        ],
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out) == {
        "image_path": "a.nii.gz", "output_dir": "out", "weights_path": None, "steps": 7,
        "threshold": 0.25, "overwrite": True, "smooth": False, "mode": "fast",
        "center": [-1.0, 2.5, 3.0], "channels": {"t1": "a.nii", "t2": "b.nii"},
        "options": {"k": [1, 2]},
    }


def test_omitted_options_keep_the_runner_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(run, prog=PROG, argv=["--image-path", "a", "--output-dir", "out"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["steps"], result["threshold"], result["smooth"], result["mode"]) == (
        3, 0.5, True, "standard",
    )


def test_json_option_reads_a_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    options = tmp_path / "options.json"
    options.write_text('{"a": 1}', encoding="utf-8")
    argv = ["--image-path", "a", "--output-dir", "out", "--options", f"@{options}"]
    assert main(run, prog=PROG, argv=argv) == 0
    assert json.loads(capsys.readouterr().out)["options"] == {"a": 1}


@pytest.mark.parametrize(
    "argv",
    [
        ["--image-path", "a"],
        ["--image-path", "a", "--output-dir", "out", "--mode", "slow"],
        ["--image-path", "a", "--output-dir", "out", "--steps", "x"],
        ["--image-path", "a", "--output-dir", "out", "--channels", "t1"],
    ],
)
def test_invalid_usage_exits_with_status_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(run, prog=PROG, argv=argv)
    assert exit_info.value.code == 2


def test_runner_errors_become_one_stderr_line_and_status_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing(*, image_path: str) -> dict[str, object]:
        raise InvalidInputError(f"{image_path} is not a NIfTI file")

    assert main(failing, prog=PROG, argv=["--image-path", "x.txt"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{PROG}: InvalidInputError: x.txt is not a NIfTI file\n"


def test_error_hint_follows_the_error_line(capsys: pytest.CaptureFixture[str]) -> None:
    def failing() -> dict[str, object]:
        raise AssetNotReadyError("weights missing", hint="Run the download command.")

    assert main(failing, prog=PROG, argv=[]) == 1
    assert capsys.readouterr().err == (
        f"{PROG}: AssetNotReadyError: weights missing\n{PROG}: hint: Run the download command.\n"
    )


def test_bound_parameter_is_fixed_and_not_an_option(capsys: pytest.CaptureFixture[str]) -> None:
    fast = functools.partial(run, mode="fast")
    argv = ["--image-path", "a.nii.gz", "--output-dir", "out"]

    assert main(fast, prog="fast-segment", distribution=PROG, argv=argv) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "fast"
    with pytest.raises(SystemExit) as usage:
        main(fast, prog="fast-segment", distribution=PROG, argv=[*argv, "--mode", "standard"])
    assert usage.value.code == 2
