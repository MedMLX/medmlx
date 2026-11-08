# TensorFlow geometry oracle

`synthmorph-geometry-tensorflow.npz` is a schema-1 synthetic array fixture,
independent of the schema-2 graph format. It records pinned official VoxelMorph
and Neurite functions running on TensorFlow 2.20.0 CPU: dense trilinear pull-field
sampling with border extension and constant fill, plus seven-step stationary
velocity integration. It contains no model weights or patient data.

The fixture records source revisions, package versions and seed. It uses a
separate format from the MONAI fixtures.

From the MedMLX repository root, regenerate using the converter's oracle recorder
in an independent TensorFlow environment:

```sh
uv run --no-project --python python3.12 --with tensorflow==2.20.0 \
  python conversion/scripts/record_synthmorph_geometry.py \
  --voxelmorph ../voxelmorph --neurite PATH_TO_NEURITE_GIT \
  --out packages/core/tests/oracles/synthmorph-geometry-tensorflow.npz
```

`packages/core/tests/test_registration.py` compares core Metal outputs directly to
this oracle. The learned SynthMorph model, affine fitting and complete image pipelines remain
outside this fixture. TensorFlow is not an ordinary test or runtime dependency.
