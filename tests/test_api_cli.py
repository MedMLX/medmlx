"""Public Python/CLI calls preserve worker failures and finite JSON arguments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from medmlx_core import _download_progress
from tqdm.auto import tqdm

import medmlx
from medmlx import cli, registry
from medmlx.cli import read_arguments
from medmlx.payloads import JsonValue


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
    inline = [sys.executable, "-m", "medmlx", "run", "threshold",
              "--arg", "image_path=image.json", "output_path=mask.json", "level=0.75"]
    completed = subprocess.run(inline, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["result"]["foreground"] == 1
    with pytest.raises(RuntimeError):
        medmlx.run("failing", arguments={"message": "provider failure"}, root=tmp_path)
    (tmp_path / "arguments.json").write_text('{"level": NaN}')
    failed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert failed.returncode == 1


@pytest.mark.parametrize("text", ['{"x": [1e400]}', '{"x": NaN}', "[]", "{"])
def test_arguments_reject_nonfinite_values_nonobjects_and_bad_json(
    tmp_path: Path, text: str,
) -> None:
    path = tmp_path / "arguments.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_arguments(path)


@pytest.mark.parametrize(
    "pair", ["level=high", "level=nan", "levle=0.75", "level:=[1", "level", "image_path=a"],
)
def test_inline_arguments_reject_unknown_ill_typed_and_repeated_values(
    model_site: Path, pair: str,
) -> None:
    with pytest.raises(ValueError):
        cli.inline_arguments("threshold", [pair, "image_path=image.json"])


@pytest.mark.parametrize(
    ("options", "total"),
    [(["threshold", "--progress"], 100), (["threshold", "--progress"], None),
     (["--all", "--progress"], 100), (["threshold"], 100)],
    ids=["known-total", "unknown-total", "all", "quiet"],
)
def test_setup_cli_emits_only_progress_json_on_stderr(
    model_site: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], options: list[str], total: int | None,
) -> None:
    """Use the real API and core downloader; only Hub IO is replaced."""
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(_download_progress, "monotonic", lambda: 0.0)
    model = registry.get_model("threshold")
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery({model.id: model}, []))

    class Hub:
        def model_info(self, repo: str) -> SimpleNamespace:
            assert repo == "MedMLX/threshold"
            return SimpleNamespace(sha="a" * 40)

    def download(
        repo: str, *, local_dir: Path, tqdm_class: type[tqdm[object]], **_kwargs: object,
    ) -> str:
        bar = tqdm_class((), total=total, unit="B", desc="Reconstructing (incomplete total...)")
        bar.update(100)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert cli.main(["setup", *options]) == 0
    captured = capsys.readouterr()
    arguments = {"weights_path": str(tmp_path / "cache" / "threshold")}
    if "--all" in options:
        assert json.loads(captured.out) == {
            "complete": True, "models": [{
                "model_id": "threshold", "arguments": arguments,
                "source_url": "https://huggingface.co/MedMLX/threshold", "revision": "a" * 40,
            }], "errors": [], "discovery_errors": [],
        }
    else:
        assert json.loads(captured.out) == arguments
    if "--progress" in options:
        lines = [json.loads(line) for line in captured.err.splitlines()]
        preparing = [{"message": "Preparing threshold · 1 of 1"}] if "--all" in options else []
        assert lines == [
            *preparing,
            {"message": "Downloading MedMLX/threshold…"},
            {"message": "Downloading MedMLX/threshold…", "model_id": "threshold",
             "bytes": 0, "total_bytes": total},
            {"message": "Downloading MedMLX/threshold…", "model_id": "threshold",
             "bytes": 100, "total_bytes": total},
            {"message": "Weights are available for threshold."},
        ]
    else:
        assert captured.err == ""
