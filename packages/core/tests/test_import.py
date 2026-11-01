"""Importing needs no MLX or source framework; inference and loading need no Torch or MONAI."""

import subprocess
import sys

_BLOCK = """
import importlib.abc
import sys

class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ImportError(f'blocked: {fullname}')

sys.meta_path.insert(0, Blocked())
"""


def _run_blocked(blocked: set[str], code: str) -> None:
    subprocess.run([sys.executable, "-c", f"BLOCKED = {sorted(blocked)!r}\n{_BLOCK}{code}"],
                   check=True)


def test_package_and_metadata_helpers_import_without_mlx_or_source_frameworks() -> None:
    _run_blocked({"mlx", "torch", "monai"}, """
import medmlx_core
from medmlx_core.maisi import validate_maisi_geometry
from medmlx_core.prompts import medsam2_prompt_payload
validate_maisi_geometry((256, 256, 128), (1.5, 1.5, 2.0))
assert not {'mlx', 'torch', 'monai'} & sys.modules.keys()
""")


def test_runtime_works_with_torch_and_monai_blocked() -> None:
    _run_blocked({"torch", "monai"}, """
import medmlx_core as core
core.require_mlx_device('mlx')
mx = core.import_mlx()
result = core.deconv2x_ncdhw(mx.ones((1, 1, 1, 1, 1)), mx.ones((1, 1, 2, 2, 2)), None, mx=mx)
mx.eval(result)
assert result.shape == (1, 1, 2, 2, 2) and mx.all(result == 1).item()
try:
    core.load_torch_checkpoint('unused.pt')
except core.MissingDependencyError as exc:
    assert exc.extra == 'conversion'
else:
    raise AssertionError('conversion must require optional torch')
assert not {'torch', 'monai'} & sys.modules.keys()
""")
