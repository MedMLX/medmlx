"""The SAM 3 image export must use a completely loaded video checkpoint."""
from pathlib import Path
from types import ModuleType

import pytest
import torch

from medmlx_convert import sam3
from medmlx_convert.boundary import Module


class Video(Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector_model = torch.nn.Linear(1, 1)


def install_loader(
    monkeypatch: pytest.MonkeyPatch, video: Video, report: dict[str, set[str] | list[str]],
    snapshot: Path,
) -> None:
    class VideoFactory:
        @staticmethod
        def from_pretrained(
            path: str, *, local_files_only: bool, output_loading_info: bool,
        ) -> tuple[Video, object]:
            assert path == str(snapshot) and local_files_only and output_loading_info
            return video, report

    transformers = ModuleType("transformers")
    monkeypatch.setattr(transformers, "Sam3VideoModel", VideoFactory, raising=False)

    def fake_library(name: str) -> ModuleType:
        assert name == "transformers"
        return transformers

    monkeypatch.setattr(sam3, "library", fake_library)


def test_returns_detector_from_fully_loaded_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video = Video()
    report: dict[str, set[str] | list[str]] = {
        "missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": [],
    }
    install_loader(monkeypatch, video, report, tmp_path)
    assert sam3.load_detector(tmp_path) is video.detector_model


@pytest.mark.parametrize(
    "key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"],
)
def test_rejects_incomplete_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str,
) -> None:
    report: dict[str, set[str] | list[str]] = {
        "missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": [],
    }
    report[key] = ["detector.weight"]
    install_loader(monkeypatch, Video(), report, tmp_path)
    with pytest.raises(RuntimeError, match=key):
        sam3.load_detector(tmp_path)
