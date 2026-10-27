"""Load the image detector from the pinned SAM 3 video checkpoint."""
from pathlib import Path

import torch

from medmlx_convert.boundary import as_module, as_sequence, class_, function, item, library, module


def load_detector(snapshot: Path) -> torch.nn.Module:
    """Require a fully loaded video checkpoint and return its image detector.

    The official snapshot contains a video model with detector and tracker
    submodels. Loading it directly as `Sam3Model` leaves detector parameters
    uninitialized, so reject every incomplete loading report before export.
    """
    transformers = library("transformers")
    loaded = as_sequence(
        function(class_(transformers, "Sam3VideoModel"), "from_pretrained")(
            str(snapshot), local_files_only=True, output_loading_info=True,
        ),
        "SAM 3 video model and loading report",
    )
    if len(loaded) != 2:
        raise ValueError("Expected a SAM 3 video model and loading report")
    video, info = as_module(loaded[0], "Sam3VideoModel"), loaded[1]
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        entries = item(info, key)
        if not isinstance(entries, (list, set, tuple)) or entries:
            raise RuntimeError(f"SAM 3 checkpoint loading {key}: {entries}")
    return module(video, "detector_model")
