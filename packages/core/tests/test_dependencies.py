import os
import subprocess
import sys
from pathlib import Path


def test_runtime_works_with_torch_and_monai_blocked():
    code = """
import importlib.abc
import sys

class BlockedDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'monai'}:
            raise ImportError(f'blocked: {fullname}')

sys.meta_path.insert(0, BlockedDependencies())
import medmlx_core as core
assert not {'torch', 'monai'} & sys.modules.keys()
report = core.require_mlx_device('mlx')
mx = core.import_mlx()
x = mx.ones((1, 1, 1, 1, 1))
w = mx.ones((1, 1, 2, 2, 2))
result = core.deconv2x_ncdhw(x, w, None, mx=mx)
mx.eval(result)
assert result.shape == (1, 1, 2, 2, 2)
assert mx.all(result == 1).item()
try:
    core.load_torch_checkpoint('unused.pt')
except core.MissingDependencyError as exc:
    assert exc.extra == 'conversion'
    assert str(exc) == 'Checkpoint conversion requires torch to unpickle the source files'
else:
    raise AssertionError('conversion must require optional torch')
"""
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
