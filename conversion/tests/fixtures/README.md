# Independent Keras reference outputs

The conversion recipes compare against CPU FP32 outputs from TensorFlow 2.20.0,
Keras 3.15.0, TF-Keras 2.20.1, NumPy 2.4.6 and h5py 3.15.1. Image-bearing
references are stored locally in `build/real-reference`, outside Git. Set
`MEDMLX_CONVERSION_REFERENCES` to use another reference directory.

- RadImageNet uses the released ResNet50 H5 and a chest X-ray resized to 224 by
  224 pixels. The reference contains pooled and spatial features.
- SynthSeg uses the released 1.0 weights and pinned upstream UNet definitions.
  The reference contains logits for a 16-cube HCC-TACE CT crop; the export
  comparison uses a 64-cube crop. These inputs test conversion numerics, not
  brain segmentation performance.
- `synthseg-v1-graph.json` contains the weight-free upstream architecture;
  `synthseg-v1-labels.json` records its label order. Logits precede the upstream
  postprocessing steps.

The default source files are `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` and
`~/.cache/medmlx/test-data/chest_xray.jpg`. Set `MEDMLX_TEST_CT` for the export
recipe or pass `--ct` and `--xray` to the recorder for other local copies.
Regenerate the references with the listed packages and original checkpoints:

```sh
python scripts/record_keras_references.py radimagenet --checkpoint PATH_TO_ORIGINAL_H5 \
  --output build/real-reference
python scripts/record_keras_references.py synthseg --checkpoint PATH_TO_ORIGINAL_H5 \
  --source PATH_TO_PINNED_SYNTHSEG/ext/neuron/models.py --output build/real-reference
```

Then run the affected conversion recipes to compare TensorFlow, Torch, Core AI
and Core ML outputs. The independent SynthMorph geometry oracle uses HCC-TACE
pixels and controlled displacement fields. Its recorder is
`scripts/record_synthmorph_geometry.py`; write its `--out` to
`../packages/core/tests/oracles/real-images/synthmorph-geometry-tensorflow.npz`
for the core test. This image-bearing archive is ignored by Git.
