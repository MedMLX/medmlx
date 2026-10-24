# Independent Keras reference outputs

Recorded on macOS arm64, CPU FP32, with TensorFlow 2.20.0, Keras 3.15.0,
TF-Keras 2.20.1, NumPy 2.4.6 and h5py 3.15.1. Every NPZ has `schema_version=3`;
the numerical arrays retain the original recording.
These files contain synthetic inputs and outputs, not weights or patient data.

- `radimagenet-tensorflow.npz`: original released ResNet50 H5 loaded by TensorFlow
  Keras, one 224x224 RGB input (Torch seed 0), pooled and spatial features.
- `synthseg-v1-tensorflow.npz`: original SynthSeg 1.0 weights and upstream
  `unet`/`conv_enc`/`conv_dec` definitions, one 16-cube uniform input (NumPy seed 0),
  32-channel logits and original label order. All learned widths are unchanged.
- `synthseg-v1-graph.json`: weight-free Keras graph serialized from that upstream
  graph through the maintained Keras H5 loader. Its metadata identifies the
  declared source/checkpoint and generating package versions.
  The final activation is linear to expose logits. The export recipe uses a
  fixed 64-cube input; the 16-cube fixture checks the TensorFlow/Torch bridge.

The recorded logits precede smoothing, flipping, topology cleanup, resampling,
parcellation and QC. Full-volume inference is outside these fixtures' scope.
Keras backend agreement does not establish clinical accuracy.

Each NPZ embeds its generation context and checkpoint/source declarations.
New recordings use the recorder path, Git revision, dirty state and actual oracle
package versions; existing fixtures retain their historical source metadata.
The recorder records its supplied paths and source declarations directly. It
rebuilds graph metadata from that invocation instead of inheriting claims from
an earlier graph. Metadata cleanup leaves all historical numerical arrays unchanged.

Regenerate with the project and the TensorFlow versions above importable:

```sh
python scripts/record_keras_references.py radimagenet --checkpoint PATH_TO_ORIGINAL_H5
python scripts/record_keras_references.py synthseg --checkpoint PATH_TO_ORIGINAL_H5 \
  --source PATH_TO_PINNED_SYNTHSEG/ext/neuron/models.py
```

The recorder reuses the persisted synthetic inputs and SynthSeg labels. After
regeneration, select the affected model with `make qualify PYTEST_ARGS='-k model_name'`
to exercise its Keras backends and Apple runtimes. Ordinary `make qa` is static.

The independent SynthMorph geometry oracle and its numerical test now live in
`packages/core/tests/oracles/` and `packages/core/tests/test_registration.py` in
the MedMLX repository. The TensorFlow recorder remains in
`scripts/record_synthmorph_geometry.py` here.
