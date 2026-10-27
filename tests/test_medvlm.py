"""Real isolated processes with synthetic provider stubs; no model/GPU qualification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import TypedDict

import packaging
import pytest

from medmlx import api
from medmlx.adapters import medvlm
from medmlx.jobs import Jobs
from medmlx.payloads import JsonValue


class Inputs(TypedDict):
    model_dir: str
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


@pytest.fixture
def isolated_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_site: Path) -> Path:
    """Supply a real venv with no frontend and a synthetic five-distribution runtime."""
    root = tmp_path / "foreign"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    site = next(root.glob("lib/python*/site-packages"))
    shutil.copytree(Path(packaging.__file__).parent, site / "packaging",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _distribution(site, "med-vlm-mlx", "0.2.0", ["mlx==0.32.3", "transformers==5.19.0"])
    _distribution(site, "mlx", "0.32.3", [])
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
    return {"model_dir": str(model), "image_path": str(image), "prompt": "describe",
            "output_dir": str(tmp_path / "output")}


def _arguments(inputs: Inputs, **changes: JsonValue) -> dict[str, JsonValue]:
    return {"model_dir": inputs["model_dir"], "image_path": inputs["image_path"],
            "prompt": inputs["prompt"], "output_dir": inputs["output_dir"], **changes}


@pytest.mark.parametrize(("model_id", "architecture"),
                         [("medgemma", "gemma3"), ("lingshu", "qwen2_5_vl")])
def test_isolated_worker_uses_local_runtime_and_publishes_report(
    isolated_python: Path, inputs: Inputs, model_id: str, architecture: str,
) -> None:
    (Path(inputs["model_dir"]) / "config.json").write_text(json.dumps({"model_type": architecture}))
    ready = api.model_readiness([model_id], arguments=_arguments(inputs))
    assert ready["models"][0]["ready"] is True
    assert ready["models"][0]["run_ready"] is None
    assert not list(isolated_python.parent.parent.glob("lib/python*/site-packages/mlx/imported"))
    job = api.run(model_id, arguments=_arguments(inputs))
    assert job["state"] == "completed"
    destination = Path(inputs["output_dir"]) / "report.json"
    report = json.loads(destination.read_text())
    assert report["text"] == f"{architecture}:rendered:describe"
    assert report["python"] == str(isolated_python)
    assert report["versions"]["mlx"] == "0.32.3"
    assert report["generation_tokens"] == 2
    assert report["qualification"] == "not_assessed_by_adapter"
    original = destination.read_bytes()
    with pytest.raises(RuntimeError, match="Output exists"):
        api.run(model_id, arguments=_arguments(inputs))
    assert destination.read_bytes() == original
    api.run(model_id, arguments=_arguments(inputs, prompt="replacement", overwrite=True))
    assert json.loads(destination.read_text())["text"] == f"{architecture}:rendered:replacement"
    with pytest.raises(RuntimeError, match="outside the checkpoint"):
        api.run(model_id, arguments=_arguments(inputs, output_dir=inputs["model_dir"]))
    assert not (Path(inputs["model_dir"]) / "report.json").exists()


def test_readiness_requires_explicit_configuration_and_consistent_dependencies(
    isolated_python: Path, monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    site = next(isolated_python.parent.parent.glob("lib/python*/site-packages"))
    (site / "sitecustomize.py").write_text('print("isolated startup noise")\n')
    assert medvlm.readiness()["ready"] is True
    assert "isolated startup noise" not in capfd.readouterr().out
    monkeypatch.delenv("MEDMLX_MEDVLM_PYTHON")
    missing = medvlm.readiness()
    assert missing["ready"] is False
    assert "MEDMLX_MEDVLM_PYTHON" in str(missing["reason"])
    monkeypatch.setenv("MEDMLX_MEDVLM_PYTHON", str(isolated_python))
    metadata = next(isolated_python.parent.parent.glob(
        "lib/python*/site-packages/transformers-*.dist-info/METADATA"
    ))
    original = metadata.read_text()
    metadata.write_text(original.replace("tokenizers>=0.23.1", "tokenizers==0.22.2"))
    inconsistent = medvlm.readiness()
    assert inconsistent["ready"] is False
    assert "requires tokenizers==0.22.2" in str(inconsistent["reason"])
    metadata.write_text(original)
    _distribution(site, "mlx-vlm", "0.0.1", [])
    conflicting = medvlm.readiness()
    assert conflicting["ready"] is False
    assert "conflicting mlx-vlm" in str(conflicting["reason"])
    assert not list(isolated_python.parent.parent.glob("lib/python*/site-packages/mlx/imported"))


@pytest.mark.parametrize(("case", "message"), [
    ("missing", "existing absolute local directory"),
    ("architecture", "requires model_type=gemma3"),
    ("shard", "Missing checkpoint shard"),
    ("tokens", "max_new_tokens"),
    ("prompt", "prompt must contain"), ("timeout", "timeout_seconds"),
])
def test_invalid_assets_and_options_fail_before_importing_mlx_or_writing(
    isolated_python: Path, inputs: Inputs, case: str, message: str,
) -> None:
    arguments = _arguments(inputs)
    model = Path(inputs["model_dir"])
    if case == "missing":
        arguments["model_dir"] = str(model / "absent")
    elif case == "architecture":
        (model / "config.json").write_text('{"model_type": "unsupported"}')
    elif case == "shard":
        (model / "model.safetensors.index.json").write_text(
            '{"weight_map": {"layer": "missing.safetensors"}}'
        )
    elif case == "tokens":
        arguments["max_new_tokens"] = True
    elif case == "prompt":
        arguments["prompt"] = " "
    else:
        arguments["timeout_seconds"] = 0
    with pytest.raises(RuntimeError, match=message):
        api.run("medgemma", arguments=arguments)
    assert not list(isolated_python.parent.parent.glob("lib/python*/site-packages/mlx/imported"))
    assert not Path(inputs["output_dir"]).exists()


@pytest.mark.parametrize(("prompt", "message"), [
    ("fail", "synthetic provider failure"), ("crash", "code 7 without a response"),
])
def test_isolated_provider_errors_leave_no_output(
    isolated_python: Path, inputs: Inputs, prompt: str, message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
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
    jobs = Jobs(Path(inputs["model_dir"]).parent)
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
            cancelled = jobs.cancel(started["job_id"]).to_payload()
            assert cancelled["state"] == "cancelled"

        # Timeout cleanup must happen before the client polls or closes the session.
        _assert_stopped(pids)
        assert unrelated.poll() is None, "Cleanup stopped a process outside the job"
        finished = jobs.get(started["job_id"])
        control_dir = finished.control_dir
        if stop == "timeout":
            payload = finished.to_payload()
            assert payload["state"] == "failed"
            assert "exceeded timeout_seconds" in str(payload.get("error"))
    finally:
        jobs.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)
    assert not control_dir.exists()
    assert not Path(inputs["output_dir"]).exists()
