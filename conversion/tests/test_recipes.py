"""Every recipe must export both formats and match PyTorch within its own tolerance."""

import subprocess
import sys
from pathlib import Path

import pytest

RECIPES = sorted(Path(__file__).resolve().parents[1].glob("recipes/*/export.py"))


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda path: path.parent.name)
def test_recipe_exports_and_matches_pytorch(recipe: Path) -> None:
    out_dir = Path("build") / recipe.parent.name
    for pattern in ("*.verification.json", "*.runtime.npz"):
        for report in out_dir.glob(pattern):
            report.unlink()
    result = subprocess.run(
        [sys.executable, str(recipe), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "export.log").write_text(result.stdout + result.stderr)
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr
    reports = {artifact.with_suffix(".verification.json")
               for pattern in ("*.aimodel", "*.mlpackage") for artifact in out_dir.glob(pattern)}
    assert reports, f"{recipe}: no native artifacts produced"
    missing = sorted(str(report) for report in reports if not report.is_file())
    assert not missing, f"Native artifacts lack current numerical comparisons: {missing}"
