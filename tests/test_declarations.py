"""Declarations name their task's operation, and shared runners receive their model ID."""

from __future__ import annotations

from pathlib import Path

import pytest

import medmlx
from medmlx import api
from tests.conftest import add_distribution

_MODELS = '''
import types

def _segment(*, model, image_path, overwrite=False):
    return {"model": model, "image_path": image_path}

shared = types.SimpleNamespace(run=_segment)
FIRST = {"id": "shared-first", "label": "First", "task": "segmentation",
         "run": "fakemlx:shared.run", "requires_weights": False}
SECOND = dict(FIRST, id="shared-second", label="Second")
MISNAMED = {"id": "misnamed", "label": "Misnamed", "task": "segmentation",
            "run": "fakemlx:sleep", "requires_weights": False}
DEFAULTED = {"id": "defaulted", "label": "Defaulted", "task": "segmentation",
             "requires_weights": False}
'''


@pytest.fixture
def shared_site(model_site: Path) -> Path:
    module = model_site / "fakemlx" / "__init__.py"
    module.write_text(module.read_text() + _MODELS)
    add_distribution(model_site, "shared-mlx", (
        "[medmlx.models]\nshared-first = fakemlx:FIRST\nshared-second = fakemlx:SECOND\n"
        "misnamed = fakemlx:MISNAMED\ndefaulted = fakemlx:DEFAULTED\n"
    ))
    return model_site


def test_shared_runner_gets_each_declarations_id_and_hides_model(
    shared_site: Path, tmp_path: Path,
) -> None:
    names = [row["name"] for row in api.describe_model("shared-second")["parameters"]]
    assert names == ["image_path", "overwrite"]

    result = medmlx.run("shared-second", image_path="ct.nii.gz", root=tmp_path)
    assert result.result == {"model": "shared-second", "image_path": "ct.nii.gz"}
    with pytest.raises(ValueError, match="sets model itself"):
        medmlx.run("shared-first", image_path="ct.nii.gz", model="shared-second",
                   root=tmp_path)


def test_a_runner_not_named_for_its_task_is_a_discovery_error(shared_site: Path) -> None:
    listed = api.list_models()
    assert "misnamed" not in {model["id"] for model in listed["models"]}
    assert any("must name the package's run function, not sleep" in error["error"]
               for error in listed["discovery_errors"])


def test_an_omitted_run_is_the_entry_point_modules_run(shared_site: Path) -> None:
    described = api.describe_model("defaulted")
    assert described["run"] == "fakemlx:run"
    assert [row["name"] for row in described["parameters"]] == [
        "image_path", "output_path", "level"]
