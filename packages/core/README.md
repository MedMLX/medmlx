# medmlx-core

Shared MLX building blocks for [MedMLX](https://github.com/MedMLX) medical imaging models on Apple Silicon.

[Docs](https://github.com/MedMLX/medmlx/blob/main/packages/core/docs/technical.md) · [MedMLX](https://github.com/MedMLX)

- 3D convolutions, resampling and sliding-window inference
- Tensor layouts, precision helpers and checkpoint conversion
- Saved-graph inference and registration geometry

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install medmlx-core
```

Run a volume through overlapping patches:

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

Arrays use batch, channel, depth, height and width axes. Replace the predictor with your model.

Run a converted graph:

```python
from pathlib import Path
import mlx.core as mx
from medmlx_core.graph.runtime import Session

session = Session.load(Path("my-model"))
logits, = session(mx.zeros((1, 4, 64, 64, 64), dtype=mx.float32))
```

Graphs are produced by [medmlx-convert](https://github.com/MedMLX/medmlx/tree/main/conversion); inference needs no PyTorch. Add the `conversion` extra to convert PyTorch checkpoints.

## License

[Apache-2.0](https://github.com/MedMLX/medmlx/blob/main/packages/core/LICENSE). See [third-party notices](https://github.com/MedMLX/medmlx/blob/main/packages/core/THIRD_PARTY_NOTICES.md).

For research and education. Outputs require human review.
