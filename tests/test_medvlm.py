"""Real isolated processes with synthetic provider stubs; no model/GPU qualification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import TypedDict, cast

import packaging
import pytest
from medmlx_core.errors import MissingDependencyError

from medmlx import api
from medmlx.adapters import medvlm
from medmlx.jobs import Jobs
from medmlx.payloads import JsonValue


class Inputs(TypedDict):
    weights_path: str
    image_path: str
    prompt: str
    output_dir: str


_CORE = '''
from pathlib import Path
Path(__file__).with_name("imported").touch()
gpu = object()
class metal:
    @staticmethod
    def is_available():
        return True

def set_default_device(device):
    assert device is gpu
'''
_UTILS = '''
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

def load(path, *, strict, local_files_only):
    assert strict and local_files_only
    assert not any(name == "medmlx" or name.startswith("medmlx.") for name in sys.modules)
    assert sys.flags.isolated == 1 and sys.dont_write_bytecode
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    return SimpleNamespace(config=json.loads((Path(path) / "config.json").read_text())), object()
'''
_GENERATE = '''
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

def generate(model, processor, prompt, *, image, max_tokens, temperature,
             prefill_step_size, verbose):
    assert temperature == 0.0 and prefill_step_size == 64 and not verbose
    assert len(image) == 1 and Path(image[0]).is_absolute()
    if prompt == "rendered:fail":
        raise ValueError("synthetic provider failure")
    if prompt == "rendered:crash":
        os._exit(7)
    if prompt == "rendered:hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(image[0]).with_suffix(".pids").write_text(f"{os.getpid()} {child.pid}")
        time.sleep(60)
    return SimpleNamespace(text=f"{model.config['model_type']}:{prompt}",
                           generation_tokens=min(2, max_tokens), prompt_tokens=12)
'''
_PROMPT = '''
def apply_chat_template(processor, config, prompt, *, num_images):
    assert num_images == 1
    return "rendered:" + prompt
'''


def _distribution(site: Path, name: str, version: str, requires: list[str]) -> None:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines.extend(f"Requires-Dist: {value}" for value in requires)
    (info / "METADATA").write_text("\n".join(lines) + "\n")


def _site(python: Path) -> Path:
    return next(python.parent.parent.glob("lib/python*/site-packages"))


def _mlx_imported(python: Path) -> bool:
    return (_site(python) / "mlx" / "imported").exists()


@pytest.fixture
def isolated_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_site: Path) -> Path:
    """Supply a real venv without medmlx and with a synthetic five-distribution runtime."""
    root = tmp_path / "foreign"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    site = next(root.glob("lib/python*/site-packages"))
    shutil.copytree(Path(packaging.__file__).parent, site / "packaging",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _distribution(site, "med-vlm-mlx", "0.3.3", ["mlx<0.33,>=0.32.4", "transformers==5.19.0"])
    _distribution(site, "mlx", "0.32.4", [])
    _distribution(site, "transformers", "5.19.0", ["huggingface-hub>=1.31", "tokenizers>=0.23.1"])
    _distribution(site, "huggingface-hub", "1.33.0", [])
    _distribution(site, "tokenizers", "0.23.2", [])
    for name, files in {
        "mlx": {"__init__.py": "", "core.py": _CORE},
        "mlx_vlm": {"__init__.py": "", "utils.py": _UTILS,
                    "generate.py": _GENERATE, "prompt_utils.py": _PROMPT},
    }.items():
        package = site / name
        package.mkdir()
        for filename, text in files.items():
            (package / filename).write_text(text)
    # Without -I this package shadows the isolated environment through PYTHONPATH.
    poison = model_site / "mlx_vlm"
    poison.mkdir()
    (poison / "__init__.py").write_text('raise AssertionError("host import leaked")\n')
    executable = root / "bin" / "python"
    monkeypatch.setenv("MEDMLX_MEDVLM_PYTHON", str(executable))
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "assets"))
    return executable


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (model / name).write_text("{}")
    (model / "config.json").write_text('{"model_type": "gemma3"}')
    (model / "model.safetensors").write_bytes(b"synthetic weights; not model tensors")
    image = tmp_path / "image.png"
    image.write_bytes(b"synthetic transport fixture; not decoded by the provider stub")
    return {"weights_path": str(model), "image_path": str(image), "prompt": "describe",
            "output_dir": str(tmp_path / "output")}


def _arguments(inputs: Inputs, **changes: JsonValue) -> dict[str, JsonValue]:
    return {"weights_path": inputs["weights_path"], "image_path": inputs["image_path"],
            "prompt": inputs["prompt"], "output_dir": inputs["output_dir"], **changes}


def test_isolated_worker_uses_local_runtime_and_publishes_report(
    isolated_python: Path, inputs: Inputs,
) -> None:
    ready = api.model_readiness(["medgemma"], arguments=_arguments(inputs))
    assert ready["models"][0]["ready"] is True
    assert not _mlx_imported(isolated_python)
    assert api.run("medgemma", arguments=_arguments(inputs))["state"] == "completed"
    destination = Path(inputs["output_dir"]) / "report.json"
    report = json.loads(destination.read_text())
    assert report["text"] == "gemma3:rendered:describe"
    assert report["python"] == str(isolated_python)
    assert report["versions"]["med-vlm-mlx"] == "0.3.3"
    assert report["versions"]["mlx"] == "0.32.4"
    original = destination.read_bytes()
    with pytest.raises(RuntimeError):
        api.run("medgemma", arguments=_arguments(inputs))
    assert destination.read_bytes() == original
    api.run("medgemma", arguments=_arguments(inputs, prompt="replacement", overwrite=True))
    assert json.loads(destination.read_text())["text"] == "gemma3:rendered:replacement"
    with pytest.raises(RuntimeError):
        api.run("medgemma", arguments=_arguments(inputs, output_dir=inputs["weights_path"]))
    assert not (Path(inputs["weights_path"]) / "report.json").exists()


def test_readiness_checks_existing_runtime_without_installing(
    isolated_python: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str],
) -> None:
    site = _site(isolated_python)
    (site / "sitecustomize.py").write_text('print("isolated startup noise")\n')
    assert medvlm.readiness()["ready"] is True
    assert "isolated startup noise" not in capfd.readouterr().out
    monkeypatch.delenv("MEDMLX_MEDVLM_PYTHON")
    assert medvlm.readiness()["ready"] is False
    monkeypatch.setenv("MEDMLX_MEDVLM_PYTHON", str(isolated_python))
    metadata = next(site.glob("transformers-*.dist-info/METADATA"))
    original = metadata.read_text()
    metadata.write_text(original.replace("tokenizers>=0.23.1", "tokenizers==0.22.2"))
    assert medvlm.readiness()["ready"] is False
    metadata.write_text(original)
    _distribution(site, "mlx-vlm", "0.0.1", [])
    assert medvlm.readiness()["ready"] is False
    assert not _mlx_imported(isolated_python)


def test_older_selected_runtime_is_rejected_before_importing_mlx_or_writing(
    isolated_python: Path, inputs: Inputs,
) -> None:
    metadata = next(_site(isolated_python).glob("med_vlm_mlx-*.dist-info/METADATA"))
    metadata.write_text(metadata.read_text().replace("Version: 0.3.3", "Version: 0.2.3"))
    assert medvlm.readiness()["ready"] is False
    with pytest.raises(MissingDependencyError, match=r"targets med-vlm-mlx==0\.3\.3"):
        medvlm.medgemma(**inputs)
    assert not _mlx_imported(isolated_python)
    assert not Path(inputs["output_dir"]).exists()


def test_stale_index_beside_single_weights_file_runs(
    isolated_python: Path, inputs: Inputs,
) -> None:
    # The pinned mlx-community 4-bit folders index two shards that do not exist beside
    # one model.safetensors; med-vlm-mlx then loads that file, so the bridge must too.
    (Path(inputs["weights_path"]) / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}')
    assert api.run("medgemma", arguments=_arguments(inputs))["state"] == "completed"


@pytest.mark.parametrize("case", ["missing", "architecture", "shard", "tokens"])
def test_invalid_assets_and_options_fail_before_importing_mlx_or_writing(
    isolated_python: Path, inputs: Inputs, case: str,
) -> None:
    arguments = _arguments(inputs)
    model = Path(inputs["weights_path"])
    if case == "missing":
        arguments["weights_path"] = str(model / "absent")
    elif case == "architecture":
        (model / "config.json").write_text('{"model_type": "unsupported"}')
    elif case == "shard":
        (model / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model.safetensors", "b": "missing.safetensors"}}')
    else:
        arguments["max_new_tokens"] = True
    with pytest.raises(RuntimeError):
        api.run("medgemma", arguments=arguments)
    assert not _mlx_imported(isolated_python)
    assert not Path(inputs["output_dir"]).exists()


@pytest.mark.parametrize("prompt", ["fail", "crash"])
def test_isolated_provider_errors_leave_no_output(
    isolated_python: Path, inputs: Inputs, prompt: str,
) -> None:
    with pytest.raises(RuntimeError):
        api.run("medgemma", arguments=_arguments(inputs, prompt=prompt))
    assert not Path(inputs["output_dir"]).exists()


def _assert_stopped(pids: list[int]) -> None:
    deadline = time.monotonic() + 5
    while True:
        states = [subprocess.run(["ps", "-p", str(pid), "-o", "stat="], check=False,
                                 capture_output=True, text=True).stdout.strip() for pid in pids]
        if all(not state or state.startswith("Z") for state in states):
            return
        assert time.monotonic() < deadline, states
        time.sleep(0.02)


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_supervision_stops_the_isolated_helper_and_its_descendant(
    isolated_python: Path, inputs: Inputs, stop: str,
) -> None:
    arguments = _arguments(inputs, prompt="hang", timeout_seconds=1 if stop == "timeout" else 300)
    pidfile = Path(inputs["image_path"]).with_suffix(".pids")
    jobs = Jobs(Path(inputs["weights_path"]).parent)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
    try:
        started = api.start_model(jobs, "medgemma", arguments)
        deadline = time.monotonic() + 5
        while not pidfile.exists():
            assert time.monotonic() < deadline, "Isolated helper did not start"
            time.sleep(0.02)
        pids = [int(value) for value in pidfile.read_text().split()]
        if stop == "cancel":
            control_dir = jobs.get(started["job_id"]).control_dir
            helper_dirs = [path for path in control_dir.iterdir() if path.is_dir()]
            assert len(helper_dirs) == 1, "Helper control files must belong to the supervised job"
            assert jobs.cancel(started["job_id"]).to_payload()["state"] == "cancelled"

        # Timeout cleanup must happen before the client polls or closes the session.
        _assert_stopped(pids)
        assert unrelated.poll() is None, "Cleanup stopped a process outside the job"
        finished = jobs.get(started["job_id"])
        control_dir = finished.control_dir
        if stop == "timeout":
            assert finished.to_payload()["state"] == "failed"
    finally:
        jobs.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)
    assert not control_dir.exists()
    assert not Path(inputs["output_dir"]).exists()


def test_managed_runtime_installs_privately_reuses_and_repairs_interrupted_setup(
    tmp_path: Path, isolated_python: Path, inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-created venv must not prevent the next run from finishing installation."""
    monkeypatch.delenv("MEDMLX_MEDVLM_PYTHON", raising=False)
    cache = tmp_path / "cache"
    prefix = cache / ".runtimes" / "med-vlm-mlx-0.3.3"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(cache))

    def find_uv(_name: str) -> str:
        return "/test/uv"

    monkeypatch.setattr(medvlm.shutil, "which", find_uv)
    commands: list[list[str]] = []
    fail = True

    def command(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        assert arguments[:2] == ["/test/uv", "--no-config"]
        assert options["stdin"] == subprocess.DEVNULL
        if "venv" in arguments:
            shutil.copytree(isolated_python.parent.parent, Path(arguments[-1]), symlinks=True)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        assert arguments[-2:] == ["med-vlm-mlx==0.3.3", "packaging>=24"]
        return subprocess.CompletedProcess(arguments, 1 if fail else 0, "", "network unavailable")

    monkeypatch.setattr(medvlm.subprocess, "run", command)
    assert medvlm.readiness()["ready"] is False
    assert commands == [] and not cache.exists()
    with pytest.raises(MissingDependencyError):
        medvlm.medgemma(**inputs)
    assert len(commands) == 2
    assert not (prefix / ".ready").exists()
    fail = False
    medvlm.medgemma(**inputs)
    assert len(commands) == 3
    assert (prefix / ".ready").read_text() == "0.3.3\n"
    assert medvlm.readiness()["ready"] is True
    medvlm.medgemma(**inputs, overwrite=True)
    assert len(commands) == 3


def test_medgemma_omitted_weights_download_and_record_the_revision(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = Path(inputs["weights_path"])
    (directory / ".medmlx-revision").write_text("a" * 40)
    downloads: list[str] = []
    requests: list[dict[str, object]] = []

    def cached(model_id: str) -> Path:
        downloads.append(model_id)
        return directory

    def invoke(request: dict[str, object], timeout: float) -> dict[str, object]:
        requests.append(request)
        return {"text": "synthetic transport output", "versions": {}}

    monkeypatch.setattr(medvlm, "weights_dir", cached)
    monkeypatch.setattr(medvlm, "_invoke", invoke)
    result = medvlm.medgemma(
        image_path=inputs["image_path"], prompt=inputs["prompt"], output_dir=inputs["output_dir"],
    )
    assert downloads == ["medgemma"]
    assert requests[0]["model_dir"] == str(directory)
    assert cast(dict[str, object], result["details"])["weights_revision"] == "a" * 40
    report = json.loads((Path(inputs["output_dir"]) / "report.json").read_text())
    assert report["weights_revision"] == "a" * 40
