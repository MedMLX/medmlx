"""Recipe provenance belongs to its checkout, independently of converter installation."""

import subprocess
from pathlib import Path

from medmlx_convert.provenance import BuildSource


def test_capture_records_the_recipe_checkout_and_local_changes(tmp_path: Path) -> None:
    project = tmp_path / "source project"
    recipe = project / "recipes" / "export.py"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("print('source recipe')\n")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments], check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("add", "recipes/export.py")
    git("-c", "user.name=Recipe Test", "-c", "user.email=recipe@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Source recipe")
    revision = git("rev-parse", "HEAD")

    source = BuildSource.capture(recipe, synthetic=True)
    assert source.recipe == "recipes/export.py"
    assert source.revision.repository == project.name
    assert source.revision.revision == revision
    assert source.revision.dirty is False

    recipe.write_text("print('modified source recipe')\n")
    changed = BuildSource.capture(recipe, synthetic=True)
    assert changed.revision.revision == revision
    assert changed.revision.dirty is True
