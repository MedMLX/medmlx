"""nnU-Net source loading rejects executable checkpoint objects."""

import pickle
from pathlib import Path

import pytest
import torch

from medmlx_convert.nnunet import load_nnunet_network


class _ExecutableCheckpoint:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).touch()",)


def test_nnunet_loader_rejects_executable_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "fold_0").mkdir()
    (tmp_path / "plans.json").write_text("{}")
    (tmp_path / "dataset.json").write_text("{}")
    marker = tmp_path / "executed"
    torch.save(
        {"payload": _ExecutableCheckpoint(marker)},
        tmp_path / "fold_0" / "checkpoint_final.pth",
    )

    try:
        with pytest.raises(pickle.UnpicklingError):
            load_nnunet_network(tmp_path)
    finally:
        assert not marker.exists()
