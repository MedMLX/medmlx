"""Prepared cohorts preserve isolated results and supervised model ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import medmlx
from medmlx import memory
from medmlx.payloads import JsonValue

_PROVIDER = '''
import weakref

THRESHOLD["cohort"] = "fakemlx:prepare"

class Prepared:
    def __init__(self, level, close_path, weights_path):
        self.level = level
        self.close_path = Path(close_path)
        self.weights = json.loads(Path(weights_path).read_text())
        self.closed = False
        self.failed_case = None

    def run(self, *, image_path, output_path, seconds=0):
        if self.failed_case is not None and self.failed_case() is not None:
            raise RuntimeError("Failed case is still retained")
        if image_path == "invalid-result":
            class TensorOwner:
                pass
            owner = TensorOwner()
            self.failed_case = weakref.ref(owner)
            return {"tensor": owner}
        time.sleep(seconds)
        if Path(image_path).resolve() == Path(output_path).resolve():
            raise ValueError("Input/output alias")
        return threshold(image_path, output_path, level=self.level + self.weights)

    def close(self):
        self.closed = True
        self.close_path.touch()

def prepare(*, level=0.5, close_path, weights_path):
    return Prepared(level, close_path, weights_path)
'''


def _provider(site: Path) -> None:
    path = site / "fakemlx" / "__init__.py"
    path.write_text(path.read_text() + _PROVIDER)


def test_cohort_reuses_preparation_reports_case_errors_and_closes(
    model_site: Path, tmp_path: Path,
) -> None:
    """The public worker API matches isolated cases and does not leak failed-case state."""
    _provider(model_site)
    source, weights, closed = (tmp_path / name for name in ("image.json", "weights.json", "closed"))
    source.write_text("[0.1, 0.7, 0.8]")
    weights.write_text("0.1")
    reference = medmlx.run("threshold", root=tmp_path, arguments={
        "image_path": "image.json", "output_path": "reference.json", "level": 0.6,
    })
    cases: list[dict[str, JsonValue]] = [
        {"image_path": "image.json", "output_path": "a.json"},
        {"image_path": "image.json", "output_path": "image.json"},
        {"image_path": "invalid-result", "output_path": "invalid.json"},
        {"image_path": "image.json", "output_path": "b.json"},
    ]
    result = medmlx.start_cohort("threshold", preparation={
        "close_path": str(closed), "weights_path": str(weights),
    }, cases=cases, root=tmp_path).wait(timeout=10)
    assert result.get("result") == {"cases": [
        {"index": 0, "ok": True, "result": {"output_path": "a.json", "foreground": 2}},
        {"index": 1, "ok": False, "error": {"type": "ValueError", "message": "Input/output alias"}},
        {"index": 2, "ok": False, "error": {
            "type": "TypeError", "message": "Object of type TensorOwner is not JSON serializable"}},
        {"index": 3, "ok": True, "result": {"output_path": "b.json", "foreground": 2}},
    ]}
    assert reference["state"] == "completed"
    assert json.loads((tmp_path / "a.json").read_text()) == json.loads(
        (tmp_path / "reference.json").read_text()) == json.loads((tmp_path / "b.json").read_text())
    assert json.loads(source.read_text()) == [0.1, 0.7, 0.8]
    assert closed.exists()
    assert memory.active_runs() == []


def test_cohort_admission_is_held_until_cancel_and_new_weights_are_prepared(
    model_site: Path, tmp_path: Path,
) -> None:
    """A resident cohort excludes other model jobs; new cohorts observe changed preparation."""
    _provider(model_site)
    weights = tmp_path / "weights.json"
    weights.write_text("0")
    (tmp_path / "image.json").write_text("[0.1,0.7,0.8]")
    preparation: dict[str, JsonValue] = {
        "close_path": str(tmp_path / "closed"), "weights_path": str(weights),
    }
    with medmlx.start_cohort("threshold", preparation=preparation, root=tmp_path, cases=[{
        "image_path": "image.json", "output_path": "pending.json", "seconds": 60,
    }]) as running:
        with pytest.raises(RuntimeError):
            medmlx.start("threshold", root=tmp_path, arguments={
                "image_path": "image.json", "output_path": "denied.json",
            })
        assert running.cancel()["state"] == "cancelled"
    assert memory.active_runs() == []
    weights.write_text("0.4")
    result = medmlx.start_cohort("threshold", preparation=preparation, cases=[{
        "image_path": "image.json", "output_path": "changed.json",
    }], root=tmp_path).wait(timeout=10)
    assert result["state"] == "completed"
    assert json.loads((tmp_path / "changed.json").read_text()) == [False, False, False]


def test_cohort_rejects_unbounded_and_case_configuration_before_output(
    model_site: Path, tmp_path: Path,
) -> None:
    """Cases cannot override preparation and malformed batches leave no partial output."""
    _provider(model_site)
    with pytest.raises(ValueError):
        empty: dict[str, JsonValue] = {}
        medmlx.start_cohort("threshold", preparation={}, cases=[empty] * 257, root=tmp_path)
    weights = tmp_path / "weights.json"
    weights.write_text("0")
    closed = tmp_path / "closed"
    with pytest.raises(RuntimeError, match="level"):
        medmlx.start_cohort("threshold", preparation={
            "close_path": str(closed), "weights_path": str(weights),
        }, cases=[{"image_path": "unused", "output_path": "partial.json"},
                  {"image_path": "unused", "output_path": "other.json", "level": 1}],
            root=tmp_path).wait(timeout=10)
    assert not (tmp_path / "partial.json").exists()
    assert closed.exists()
    assert memory.active_runs() == []
