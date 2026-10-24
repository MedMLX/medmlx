"""The relocated verifier must preserve failed checks and refuse missing tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parents[1] / "verify.sh"


class VerifyTests(unittest.TestCase):
    def verify(
        self,
        *,
        tests_exit: int = 0,
        lint_exit: int = 0,
        missing_python: bool = False,
        repo_name: str = "model-mlx",
    ) -> subprocess.CompletedProcess[str]:
        with TemporaryDirectory(prefix="verify workspace ") as directory:
            root = Path(directory)
            scripts = root / "medmlx" / "tools" / "extraction"
            scripts.mkdir(parents=True)
            shutil.copy2(SCRIPT, scripts / "verify.sh")
            (root / "medmlx" / "packages" / "core" / "src").mkdir(parents=True)
            source = root / "model-mlx" / "src"
            source.mkdir(parents=True)
            (source / "pytest.py").write_text(
                f"import sys\nprint('pytest executed')\nsys.exit({tests_exit})\n"
            )
            tools = root / "tools"
            tools.mkdir()
            (tools / "git").write_text("#!/bin/sh\nprintf ' M current-work.py\\n'\n")
            (tools / "ruff").write_text(
                f"#!/bin/sh\nprintf 'lint executed\\n'\nexit {lint_exit}\n"
            )
            for tool in tools.iterdir():
                tool.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                "MEDMLX_PYTHON": str(root / "missing")
                if missing_python
                else sys.executable,
                "MEDMLX_RUFF": str(tools / "ruff"),
            }
            # Exercise default relative workspace/core lookup, independent of cwd.
            environment.pop("MEDMLX_REPO_ROOT", None)
            environment.pop("MEDMLX_CORE_SRC", None)
            return subprocess.run(
                ["bash", str(scripts / "verify.sh"), repo_name],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )

    def test_relocated_workspace_and_paths_with_spaces(self) -> None:
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pytest executed", result.stdout)
        self.assertIn("lint executed", result.stdout)

    def test_failed_tests_propagate_and_stop_before_lint(self) -> None:
        result = self.verify(tests_exit=17)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertIn("pytest executed", result.stdout)
        self.assertNotIn("lint executed", result.stdout)

    def test_failed_lint_propagates(self) -> None:
        result = self.verify(lint_exit=19)
        self.assertEqual(result.returncode, 19, result.stderr)
        self.assertIn("lint executed", result.stdout)

    def test_missing_python_is_actionable(self) -> None:
        result = self.verify(missing_python=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("set MEDMLX_PYTHON", result.stderr)

    def test_parent_traversal_is_rejected(self) -> None:
        result = self.verify(repo_name="../model-mlx")
        self.assertEqual(result.returncode, 2)
        self.assertIn("single directory name", result.stderr)


if __name__ == "__main__":
    unittest.main()
