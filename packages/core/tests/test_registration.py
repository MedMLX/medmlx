"""Compare native geometry directly to pinned official TensorFlow computation."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from medmlx_core.registration import dense_warp3d, integrate_stationary_velocity

FIXTURE = Path(__file__).with_name("oracles") / "synthmorph-geometry-tensorflow.npz"


def test_synthmorph_geometry_against_original_tensorflow() -> None:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        assert fixture["schema_version"].item() == 1
        image, shift = mx.array(fixture["image"]), mx.array(fixture["shift"])
        cases = ((dense_warp3d(image, shift), "border"),
                 (dense_warp3d(image, shift, fill_value=-7), "filled"),
                 (integrate_stationary_velocity(mx.array(fixture["velocity"])), "integrated"))
        for actual, name in cases:
            np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), fixture[name],
                                       atol=2e-6, rtol=2e-6)
