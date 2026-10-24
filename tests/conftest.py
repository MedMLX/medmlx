"""A fake installed model package that declares MedMLX entry points."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_MODULE = '''
import json
import os
import time
from pathlib import Path

THRESHOLD = {
    "id": "threshold",
    "label": "Threshold",
    "task": "segmentation",
    "run": "fakemlx:threshold",
    "readiness": "fakemlx:weights_missing",
    "summary": "Marks values above a level.",
}
SLEEPER = {"id": "sleeper", "label": "Sleeper", "task": "testing", "run": "fakemlx:sleep"}
FAILING = {"id": "failing", "label": "Failing", "task": "testing", "run": "fakemlx:fail"}
CRASHING = {"id": "crashing", "label": "Crashing", "task": "testing", "run": "fakemlx:crash"}
INVALID = {"id": "invalid", "label": "Invalid", "task": "testing"}


def threshold(image_path: str, output_path: str, *, level: float = 0.5) -> dict[str, object]:
    values = json.loads(Path(image_path).read_text())
    mask = [value > level for value in values]
    Path(output_path).write_text(json.dumps(mask))
    return {"output_path": output_path, "foreground": sum(mask)}


def weights_missing() -> dict[str, object]:
    return {"ready": False, "reason": "weights not staged", "setup_hint": "stage the weights"}


def sleep(seconds: float) -> dict[str, object]:
    time.sleep(seconds)
    return {}


def fail(message: str) -> dict[str, object]:
    raise ValueError(message)


def crash() -> None:
    os._exit(3)
'''

_ENTRY_POINTS = """\
[medmlx.models]
threshold = fakemlx:THRESHOLD
sleeper = fakemlx:SLEEPER
failing = fakemlx:FAILING
crashing = fakemlx:CRASHING
invalid = fakemlx:INVALID
missing = fakemlx_absent:MODEL
"""


_FAMILIES = {
    "segment": ["fakemlx", "pytest>=8; extra == 'dev'"],
    "generate": ["absent-mlx @ git+https://example.invalid/absent-mlx.git@v9.0.0"],
    "vision": [],
}

USABLE_MLX = """
__version__ = "0.0-fake"


def default_device():
    return "Device(gpu, 0)"


class metal:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_info():
        return {"device_name": "Apple test GPU"}
"""
BROKEN_MLX = 'raise ImportError("simulated broken MLX installation")\n'
NO_METAL_MLX = USABLE_MLX.replace("return True", "return False")


def add_distribution(
    site: Path, name: str, entry_points: str = "", requires: list[str] | None = None
) -> None:
    info = site / f"{name.replace('-', '_')}-1.2.3.dist-info"
    info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", "Version: 1.2.3"]
    lines += [f"Requires-Dist: {requirement}" for requirement in requires or []]
    (info / "METADATA").write_text("\n".join(lines) + "\n")
    if entry_points:
        (info / "entry_points.txt").write_text(entry_points)


def set_mlx(site: Path, source: str) -> None:
    """Shadow MLX with a fake whose import succeeds or fails, for the runtime probe."""

    (site / "mlx").mkdir(exist_ok=True)
    (site / "mlx" / "__init__.py").write_text("")
    (site / "mlx" / "core.py").write_text(source)


@pytest.fixture
def model_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Install ``fakemlx``, fake families, and a usable fake MLX for this process and workers."""

    site = tmp_path / "site"
    (site / "fakemlx").mkdir(parents=True)
    (site / "fakemlx" / "__init__.py").write_text(_MODULE)
    add_distribution(site, "fakemlx", _ENTRY_POINTS)
    requirements = [
        f"{item} ; extra == '{family}'"
        for family, members in _FAMILIES.items() for item in members
        if ";" not in item
    ]
    add_distribution(site, "medmlx", requires=requirements)
    set_mlx(site, USABLE_MLX)
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    importlib.invalidate_caches()
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(site) if not existing else f"{site}{os.pathsep}{existing}")
    # Each test writes its own copy of the fake package; never reuse another test's import.
    sys.modules.pop("fakemlx", None)
    yield site
    sys.modules.pop("fakemlx", None)
