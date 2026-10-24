#!/usr/bin/env python3
"""Record pinned MONAI window logits, coordinates, maps, and argmax labels."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from monai.data.utils import compute_importance_map
from monai.inferers import utils as inferer_utils
from monai.transforms.post.array import AsDiscrete
from reference_support import HostArray, output_directory, save_fixture

sliding_window_inference = cast(
    Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]],
    vars(inferer_utils)["sliding_window_inference"],
)

from_numpy = cast(Callable[[HostArray], torch.Tensor], vars(torch)["from_numpy"])


def main() -> None:
    root = output_directory(__doc__)
    rng = np.random.default_rng(20261007)
    inputs = rng.integers(-64, 65, (1, 1, 11, 9, 7)).astype(np.float32) / 64
    weights = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 8
    bias = rng.integers(-4, 5, (1, 3, 1, 1, 1)).astype(np.float32) / 16
    roi = (8, 6, 4)
    for mode in ("constant", "gaussian"):
        coordinates: list[list[int]] = []

        def predict(
            patch: torch.Tensor,
            coords: list[list[slice]],
            coordinates: list[list[int]] = coordinates,
        ) -> torch.Tensor:
            coordinates.extend([[int(s.start) for s in coord[2:]] for coord in coords])
            return patch * from_numpy(weights) + from_numpy(bias)

        scores = sliding_window_inference(
            from_numpy(inputs),
            roi,
            1,
            predict,
            mode=mode,
            padding_mode="replicate",
            with_coord=True,
        )
        assert isinstance(scores, torch.Tensor), "The fixture predictor returns one tensor"
        discrete = AsDiscrete(argmax=True)(scores[0])
        assert isinstance(discrete, torch.Tensor), "Tensor discretization preserves the tensor type"
        labels = discrete.numpy().astype(np.uint8)[None]
        save_fixture(
            root,
            f"sliding_window_{mode}",
            20261007,
            {
                "inputs": inputs,
                "weights": weights,
                "bias": bias,
                "roi_size": np.array(roi),
                "scores": scores.numpy(),
                "labels": labels,
                "coordinates": np.array(coordinates),
                "importance": compute_importance_map(roi, mode=mode).numpy(),
            },
            mode=mode,
        )


if __name__ == "__main__":
    main()
