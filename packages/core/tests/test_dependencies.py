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


def test_model_metadata_works_without_accelerator_or_source_frameworks() -> None:
    code = """
import importlib.abc
import sys
from dataclasses import dataclass

class BlockedRuntimes(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
            raise ImportError(f'blocked: {fullname}')

sys.meta_path.insert(0, BlockedRuntimes())
from medmlx_core.maisi import NvGenerateAnatomySize, maisi_label, validate_maisi_geometry
from medmlx_core.prompts import medsam2_prompt_payload

assert maisi_label('HCC').value == 26
assert maisi_label('hepatic vessels').name == 'hepatic vessel'
validate_maisi_geometry((256, 256, 128), (1.5, 1.5, 2.0))
control = NvGenerateAnatomySize('HCC', 0.25)
assert (control.anatomy, control.scale) == ('hepatic tumor', 0.25)

@dataclass(frozen=True)
class Prompt:
    frame_idx: int = 3
    obj_id: int = 7
    mode: str = 'projected_box_plus_points'
    box_xyxy: tuple[float, float, float, float] = (1.0, 2.0, 8.0, 9.0)
    points_xy: tuple[tuple[float, float], ...] = ((4.0, 5.0),)
    point_labels: tuple[int, ...] = (1,)
    mask: object = object()
    clear_old_points: bool = False
    normalize_coords: bool = True

assert medsam2_prompt_payload(Prompt()) == {
    'frame_idx': 3,
    'obj_id': 7,
    'prompt_mode': 'projected_box_plus_points',
    'box_xyxy': [1.0, 2.0, 8.0, 9.0],
    'points_xy': [[4.0, 5.0]],
    'point_labels': [1],
    'mask_supplied': True,
    'clear_old_points': False,
    'normalize_coords': True,
}
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
