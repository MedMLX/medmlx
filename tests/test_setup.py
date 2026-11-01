"""Optional pre-downloads share automatic model folders and report partial failures."""

import json
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType

import pytest

import medmlx
from medmlx import cli, registry, setup
from medmlx.payloads import DiscoveryError


def spec(
    model_id: str, *, requires_weights: bool = True, setup_supported: bool = True,
) -> registry.ModelSpec:
    return registry.ModelSpec(model_id, model_id, "testing", "fake:run", None, None,
                              "fake", "1.0", "fake = fake:MODEL", requires_weights, setup_supported)


@pytest.mark.parametrize(
    ("marker", "cached"),
    [(None, False), ("d2d635fbac34254f14192ac348a05d275a949317", True), ("a" * 40, False)],
    ids=["absent", "cached", "stale-revision"],
)
def test_setup_plans_report_the_local_cache_in_the_api_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], marker: str | None, cached: bool,
) -> None:
    root = tmp_path / "assets"
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(root))
    folder = root / "lung-nodule" / "MedMLX--retinanet-lung-nodule-mlx"
    if marker is not None:
        folder.mkdir(parents=True)
        (folder / ".medmlx-revision").write_text(marker)
    model = spec("lung-nodule")

    def get_model(_model_id: str) -> registry.ModelSpec:
        return model

    monkeypatch.setattr(registry, "get_model", get_model)
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery({model.id: model}, []))

    def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("Setup plans must not download weights")

    monkeypatch.setattr(setup, "weights_dir", unexpected)
    expected: setup.SetupPlan = {
        "model_id": "lung-nodule",
        "supported": True,
        "package": "fake",
        "runtime_installed": False,
        "source_url": "https://huggingface.co/MedMLX/retinanet-lung-nodule-mlx",
        "cached": cached,
        "weights_path": str(folder) if cached else None,
        "revision": marker if cached else None,
        "weights_from": [],
        "message": "Pre-download model files for offline use; normal runs download on first use.",
    }
    assert medmlx.setup_plan("lung-nodule") == expected
    assert cli.main(["setup", "lung-nodule", "--plan"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert cli.main(["setup"]) == 0
    assert json.loads(capsys.readouterr().out) == [expected]
    assert not (root / ".locks").exists()
    if marker is None:
        assert not root.exists()


@pytest.mark.parametrize(
    ("model_id", "requires_weights", "setup_supported"),
    [
        ("features", False, True),
        ("medgemma", True, True),
        ("mrsegmentator", True, True),
        ("reason-ct-tools", True, False),
    ],
    ids=["weight-free", "manual-medgemma", "manual-mrsegmentator", "manual-workflow"],
)
def test_setup_plan_without_automatic_weights_has_no_cache_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    model_id: str, requires_weights: bool, setup_supported: bool,
) -> None:
    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path))
    folder = tmp_path / model_id
    folder.mkdir()
    (folder / ".medmlx-revision").write_text("a" * 40)
    model = spec(model_id, requires_weights=requires_weights, setup_supported=setup_supported)

    def get_model(_model_id: str) -> registry.ModelSpec:
        return model

    monkeypatch.setattr(registry, "get_model", get_model)
    plan = medmlx.setup_plan(model_id)
    assert plan["cached"] is None
    assert plan["revision"] is None
    assert plan["weights_path"] is None
    assert plan["source_url"] is None


def test_setup_and_cli_return_the_core_folder_and_checkpoint_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The reported source must match the core helper's model-to-Hub mapping."""
    def get_model(_model_id: str) -> registry.ModelSpec:
        return spec("lung-nodule")

    requested: list[str] = []

    def download(model_id: str, **_kwargs: object) -> Path:
        requested.append(model_id)
        return tmp_path

    monkeypatch.setattr(registry, "get_model", get_model)
    monkeypatch.setattr(setup, "weights_dir", download)
    expected = "https://huggingface.co/MedMLX/retinanet-lung-nodule-mlx"
    assert medmlx.setup_plan("lung-nodule")["source_url"] == expected
    result = medmlx.setup_model("lung-nodule")
    assert result["arguments"] == {"weights_path": str(tmp_path)}
    assert result["source_url"] == expected
    assert cli.main(["setup", "lung-nodule"]) == 0
    assert '"weights_path"' in capsys.readouterr().out
    assert requested == ["lung-nodule", "lung-nodule"]


def test_weight_free_model_never_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_model(_model_id: str) -> registry.ModelSpec:
        return spec("features", requires_weights=False)

    def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("A weight-free model must not download a repository")

    monkeypatch.setattr(registry, "get_model", get_model)
    monkeypatch.setattr(setup, "weights_dir", unexpected)
    assert medmlx.setup_model("features")["arguments"] == {}


def test_workflow_prepares_each_model_it_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each weight argument gets the folder its own model's runner would use."""
    entry = EntryPoint(
        name="reason-ct-tools", value="medmlx.workflows.reason_ct:MODEL", group="medmlx.models",
    )

    def entry_points(**_kwargs: object) -> list[EntryPoint]:
        return [entry]

    requested: list[str] = []

    def download(model_id: str, **_kwargs: object) -> Path:
        requested.append(model_id)
        return tmp_path / model_id

    cached: set[str] = {"mlx-reason-ct", "lung-nodule"}

    def cached_weights(model_id: str) -> Path | None:
        return tmp_path / model_id if model_id in cached else None

    monkeypatch.setattr(registry, "entry_points", entry_points)
    monkeypatch.setattr(setup, "weights_dir", download)
    monkeypatch.setattr(setup, "cached_weights", cached_weights)
    models = ["mlx-reason-ct", "lung-nodule", "medsam2"]
    plan = medmlx.setup_plan("reason-ct-tools")
    assert plan["supported"] and plan["source_url"] is None
    assert plan["weights_from"] == models
    # Cached only when every model it uses is; there is no single folder.
    assert plan["cached"] is False
    assert plan["weights_path"] is None and plan["revision"] is None
    cached.add("medsam2")
    assert medmlx.setup_plan("reason-ct-tools")["cached"] is True
    assert medmlx.setup_model("reason-ct-tools")["arguments"] == {
        "reason_weights_path": str(tmp_path / "mlx-reason-ct"),
        "detector_weights_path": str(tmp_path / "lung-nodule"),
        "medsam_weights_path": str(tmp_path / "medsam2"),
    }
    assert medmlx.setup_all()["complete"]
    assert requested == models * 2


@pytest.mark.parametrize(
    "fields",
    [
        {"weights_from": {}},
        {"weights_from": ["lung-nodule"]},
        {"weights_from": {"not an argument": "lung-nodule"}},
        {"weights_from": {"weights_path": "Lung Nodule"}},
        {"weights_from": {"weights_path": "lung-nodule"}, "requires_weights": False},
    ],
)
def test_invalid_weights_from_is_a_discovery_error(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object],
) -> None:
    module = ModuleType("fake_workflow")
    declaration = {"id": "fake", "label": "Fake", "task": "testing", "run": "fake:run"}
    module.__dict__["MODEL"] = declaration | fields
    monkeypatch.setitem(sys.modules, "fake_workflow", module)
    entry = EntryPoint(name="fake", value="fake_workflow:MODEL", group="medmlx.models")

    def entry_points(**_kwargs: object) -> list[EntryPoint]:
        return [entry]

    monkeypatch.setattr(registry, "entry_points", entry_points)
    discovery = registry.discover()
    assert not discovery.models
    assert "weights_from" in discovery.errors[0]["error"]


def test_all_continues_after_failures_and_never_reports_partial_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = {name: spec(name) for name in ("first", "second", "third")}
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery(models, []))
    requested: list[str] = []

    def download(model_id: str, **_kwargs: object) -> Path:
        requested.append(model_id)
        if model_id == "second":
            raise OSError("download interrupted")
        return tmp_path / model_id

    monkeypatch.setattr(setup, "weights_dir", download)
    result = medmlx.setup_all()
    assert not result["complete"]
    assert [model["model_id"] for model in result["models"]] == ["first", "third"]
    assert [error["model_id"] for error in result["errors"]] == ["second"]
    assert requested == ["first", "second", "third"]
    errors: list[DiscoveryError] = [{"entry_point": "bad = broken:MODEL", "error": "Cannot import"}]
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery({}, errors))
    result = medmlx.setup_all()
    assert not result["complete"] and result["discovery_errors"] == errors


def test_cli_plan_succeeds_and_incomplete_setup_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery({}, []))
    assert cli.main(["setup", "--all", "--plan"]) == 0
    assert cli.main(["setup", "--all"]) == 1  # An empty install is not a completed setup.
    assert cli.main(["setup", "a-model", "--all"]) == 1


@pytest.mark.parametrize("batch", [False, True], ids=["model", "all"])
def test_public_setup_forwards_both_callbacks_and_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: bool,
) -> None:
    model = spec("lung-nodule")

    def get_model(_model_id: str) -> registry.ModelSpec:
        return model

    monkeypatch.setattr(registry, "get_model", get_model)
    monkeypatch.setattr(registry, "discover", lambda: registry.Discovery({model.id: model}, []))
    event: medmlx.DownloadProgress = {
        "message": "Downloading MedMLX/retinanet-lung-nodule-mlx…",
        "model_id": "lung-nodule", "bytes": 30, "total_bytes": 100,
    }

    def download(
        model_id: str, *, progress: Callable[[str], None] | None, refresh: bool,
        download_progress: Callable[[medmlx.DownloadProgress], None] | None,
    ) -> Path:
        assert model_id == "lung-nodule" and refresh is True
        if progress is not None:
            progress("Downloading MedMLX/retinanet-lung-nodule-mlx…")
        if download_progress is not None:
            download_progress(event)
        return tmp_path

    monkeypatch.setattr(setup, "weights_dir", download)
    messages: list[str] = []
    events: list[medmlx.DownloadProgress] = []
    if batch:
        result = medmlx.setup_all(
            progress=messages.append, download_progress=events.append, refresh=True,
        )
        assert result["complete"] is True
        assert result["models"][0]["arguments"] == {"weights_path": str(tmp_path)}
    else:
        assert medmlx.setup_model(
            "lung-nodule", progress=messages.append, download_progress=events.append, refresh=True,
        )["arguments"] == {"weights_path": str(tmp_path)}
    assert events == [{
        "message": "Downloading MedMLX/retinanet-lung-nodule-mlx…",
        "model_id": "lung-nodule", "bytes": 30, "total_bytes": 100,
    }]
    assert messages == (["Preparing lung-nodule · 1 of 1"] if batch else []) + [
        "Downloading MedMLX/retinanet-lung-nodule-mlx…",
    ]


def test_mcp_setup_worker_logs_byte_events_alongside_status_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def get_model(_model_id: str) -> registry.ModelSpec:
        return spec("lung-nodule")

    monkeypatch.setattr(registry, "get_model", get_model)

    def download(
        model_id: str, *, progress: Callable[[str], None] | None,
        download_progress: Callable[[medmlx.DownloadProgress], None] | None, **_kwargs: object,
    ) -> Path:
        if progress is not None:
            progress("Downloading MedMLX/retinanet-lung-nodule-mlx…")
        if download_progress is not None:
            download_progress({
                "message": "Downloading MedMLX/retinanet-lung-nodule-mlx…",
                "model_id": model_id, "bytes": 100, "total_bytes": None,
            })
        return tmp_path

    monkeypatch.setattr(setup, "weights_dir", download)
    assert setup.prepare("lung-nodule")["arguments"] == {"weights_path": str(tmp_path)}
    captured = capsys.readouterr()
    status, byte_line = captured.out.splitlines()
    assert status == "Downloading MedMLX/retinanet-lung-nodule-mlx…"
    assert json.loads(byte_line) == {
        "message": "Downloading MedMLX/retinanet-lung-nodule-mlx…",
        "model_id": "lung-nodule", "bytes": 100, "total_bytes": None,
    }
    assert captured.err == ""
