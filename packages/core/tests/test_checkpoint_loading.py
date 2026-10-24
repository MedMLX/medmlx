"""Checkpoint admission rejects executable pickle objects before conversion."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from medmlx_core.conversion import load_torch_checkpoint


class _ExecutableCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


def test_checkpoint_loader_rejects_pickle_execution(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    path = tmp_path / "weights.pt"
    marker = tmp_path / "executed"
    torch.save({"payload": _ExecutableCheckpoint(marker)}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_torch_checkpoint(path)
    assert not marker.exists()
    torch.save({"model": {"weight": torch.tensor([2.0, 3.0])}}, path)
    model = load_torch_checkpoint(path)["model"]
    assert isinstance(model, dict)
    weight = cast(Mapping[object, object], model)["weight"]
    assert isinstance(weight, torch.Tensor)
    assert torch.equal(weight, torch.tensor([2.0, 3.0]))


def test_missing_conversion_dependency_reports_install_extra() -> None:
    import subprocess
    import sys

    code = """
import importlib.abc
import sys

class BlockedTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError(f'blocked: {fullname}')

sys.meta_path.insert(0, BlockedTorch())
from medmlx_core import MissingDependencyError, load_torch_checkpoint
try:
    load_torch_checkpoint('unused.pt')
except MissingDependencyError as exc:
    assert exc.extra == 'conversion'
    assert exc.hint == 'Install medmlx-core[conversion] and rerun the converter.'
else:
    raise AssertionError('missing Torch must fail before checkpoint loading')
assert 'mlx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
