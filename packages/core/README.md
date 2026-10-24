# medmlx-core

Shared MLX building blocks for [MedMLX](https://github.com/MedMLX/medmlx) medical
imaging models on Apple Silicon. Source is owned by
[`packages/core`](https://github.com/MedMLX/medmlx/tree/main/packages/core) in the
main MedMLX repository; model runtimes retain the `medmlx_core` import and distribution.
Static checks run with `make qa-core` at the repository root. Select affected
tests with `make test-core CORE_TESTS=packages/core/tests/path.py`; `make test-core`
runs the full core suite.

- Metal runtime checks
- 3D convolutions, resampling and sliding-window inference
- Tensor layouts, precision helpers and checkpoint conversion
- Portable saved-graph inference and dense registration geometry

[PyPI](https://pypi.org/project/medmlx-core/) ·
[Releases](https://github.com/MedMLX/medmlx/releases) ·
[Technical reference](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md)

[Install](#install) · [Python](#python) · [Technical details](#technical-details) · [License](#license)

## Requirements

Apple Silicon Mac, macOS, Python 3.13 or 3.14, and Metal GPU access.
NumPy and MLX install automatically. Inference raises an error when Metal is
unavailable.

## Install

```bash
uv pip install medmlx-core==0.2.3
```

For optional PyTorch checkpoint conversion:

```bash
uv pip install "medmlx-core[conversion]==0.2.3"
```

## Python

Run a small synthetic volume through overlapping patches:

```python
import numpy as np
from medmlx_core import sliding_window_inference

volume = np.ones((1, 1, 32, 32, 32), dtype=np.float32)
output = sliding_window_inference(
    volume,
    roi_size=(16, 16, 16),
    sw_batch_size=1,
    predictor=lambda patch: patch * 2,
)
print(output.shape)  # (1, 1, 32, 32, 32)
```

Arrays use batch, channel, depth, height and width axes. Replace the example
predictor with your model; the result is a NumPy array.

MedMLX model packages provide the architectures, weights and image workflows.
This library supplies their shared runtime operations.

Converted graphs run directly through core:

```python
from pathlib import Path
import mlx.core as mx
from medmlx_core.graph.runtime import Session

session = Session.load(Path("build/my-native-brats"))
logits, = session(mx.zeros((1, 4, 64, 64, 64), dtype=mx.float32))
```

Graph schema 2, its codec and Metal executor are owned here. Source capture,
ATen translation and source comparisons live in [`conversion`](https://github.com/MedMLX/medmlx/tree/main/conversion).
Inference does not require that converter, Torch, TensorFlow, Core ML or Core AI. Existing
schema-1 pilot bundles must be regenerated; no legacy graph loader is retained.
See the [graph and geometry contract](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/graph-runtime.html) for the bounded
operation coverage and numerical qualification scope.

`medmlx_core.compensated` provides explicit accuracy profiles using FP32 Metal
storage. `CompensatedFloat32Operators` keeps product and summation residuals
inside convolution, GroupNorm, frozen BatchNorm, ELU, softmax and Gaussian
filtering, with FP32 outputs. `CompensatedFloat48Operators` carries both words
between convolution, GroupNorm, transpose-convolution, addition and ReLU layers.
The latter packs high and low words along the batch axis (all high words, then
all low words). The package tests compare both profiles with float64 reference
operators on small synthetic inputs.

For paired network outputs, `sliding_window_inference(...,
predictor_input="numpy")` accepts a host NumPy callback and float64 windows and
stitching. This explicit mode lets the callback run a paired Metal network and
export both words for host blending. The default callback still uses MLX arrays
and FP16/FP32 inputs.

## Technical details

API contracts, numerical references and development commands are in the
[technical reference](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md).
See the [decoder note](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/transpose-convolution.md)
for the Metal transpose-convolution fix.

## License

[Apache-2.0](https://github.com/MedMLX/medmlx/blob/main/packages/core/LICENSE).
Includes [third-party notices](https://github.com/MedMLX/medmlx/blob/main/packages/core/THIRD_PARTY_NOTICES.md)
covering MLX-derived portions under MIT and the upstream MONAI Apache-2.0 license.

## Implementation mapping

[Central implementation map](https://github.com/MedMLX/Med-MLX/blob/main/implementation-maps/medmlx.json) records upstream-to-native changes, source references and qualification scope.
