"""Public Python/CLI calls preserve worker failures and finite JSON arguments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

import medmlx
from medmlx.cli import read_arguments
from medmlx.payloads import JsonValue

_VALUES = cast(
    st.SearchStrategy[JsonValue],
    st.recursive(
        st.one_of(st.none(), st.booleans(), st.integers(),
                  st.floats(allow_nan=False, allow_infinity=False), st.text()),
        lambda children: st.one_of(st.lists(children), st.dictionaries(st.text(), children)),
        max_leaves=20,
    ),
)


def test_python_and_cli_execute_the_same_runner_and_report_failures(
    model_site: Path, tmp_path: Path,
) -> None:
    """Exercise real worker processes, relative paths and the CLI's JSON boundary."""
    (tmp_path / "image.json").write_text("[0.1, 0.7, 0.8]")
    arguments: dict[str, JsonValue] = {"image_path": "image.json", "output_path": "mask.json"}
    result = medmlx.run("threshold", arguments=arguments, root=tmp_path)
    assert result["state"] == "completed"
    assert json.loads((tmp_path / "mask.json").read_text()) == [False, True, True]
    (tmp_path / "arguments.json").write_text(json.dumps({**arguments, "level": 0.75}))
    command = [sys.executable, "-m", "medmlx", "run", "threshold",
               "--arguments", "arguments.json", "--root", str(tmp_path)]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["result"]["foreground"] == 1
    with pytest.raises(RuntimeError, match="failing: provider failure"):
        medmlx.run("failing", arguments={"message": "provider failure"}, root=tmp_path)
    (tmp_path / "arguments.json").write_text('{"level": NaN}')
    failed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    assert "finite JSON" in failed.stderr


@given(st.dictionaries(st.text(), _VALUES, max_size=8))
def test_argument_objects_round_trip(value: dict[str, JsonValue]) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "arguments.json"
        path.write_text(json.dumps(value, allow_nan=False))
        assert read_arguments(path) == value


@given(st.text(max_size=3000))
def test_argument_text_returns_an_object_or_a_parse_error(text: str) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "arguments.json"
        path.write_text(text)
        try:
            value = read_arguments(path)
        except ValueError:
            return
        assert isinstance(value, dict)
        json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("text", ['{"x": [1e400]}', '{"x": NaN}', "[]", "null"])
def test_arguments_reject_nonfinite_values_and_nonobjects(tmp_path: Path, text: str) -> None:
    path = tmp_path / "arguments.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_arguments(path)
