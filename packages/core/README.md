# medmlx-core

Shared MLX building blocks for MedMLX medical imaging models on Apple Silicon.

[Models and weights](https://huggingface.co/MedMLX)

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

Graph inference needs no PyTorch. The `conversion` extra supplies PyTorch for checkpoint conversion.

## Automatic model folders

Registered runtimes call `weights_dir(model_id)` when their optional
`weights_path` is omitted:

```python
from medmlx_core.weights import weights_dir, weights_revision

folder = weights_dir("mlx-segment-ct")
revision = weights_revision(folder)
```

The published model repository, or the relevant files within it, downloads on first
use and is reused offline afterwards. Partial downloads resume. The returned folder
contains plain files; the runner selects its own files and records the revision in
run details. Explicit user folders bypass downloads and retain their supplied paths.

**Unreleased:** Registered bundles use the weight revisions documented by their
runtime releases and linked model cards. Hub `main` updates do not change those
downloads or their label maps. SAM3 retains its existing pin. Unlisted model IDs
still resolve `MedMLX/<model_id>` at `main` once on first use.

**Unreleased:** A folder without a matching revision marker can be reused after
`weights_dir` checks every selected file against the installed Hub commit's LFS
SHA-256 or Git blob hash. Missing or mismatched files use the resumable download
path. Verified files from the older top-level cache layout are copied into the
current layout, preserving the originals; copies keep later refreshes independent.
`refresh=True` forces a download. This verification needs network access once;
the completed folder is reused offline afterwards.

Check the cache without downloading:

```python
from medmlx_core.weights import cached_weights, weights_revision

folder = cached_weights("mlx-segment-ct")  # Path or None
revision = None if folder is None else weights_revision(folder)
```

`cached_weights` uses the same cache paths and completion checks as `weights_dir`,
including pinned revisions and companion files. It respects `MEDMLX_ASSET_CACHE`
and takes no locks, creates no directories and writes no files. Models without
automatic weights return `None`.

Pass `progress=callback` for status strings, or `download_progress=callback` for
`DownloadProgress` dictionaries with `message`, `model_id`, `bytes` and `total_bytes`.
Byte counts include resumed file bytes. Totals can grow as the Hub discovers files;
unknown totals are `None`. Updates arrive at most twice a second per model, plus a
final event when the download stops, including on interruption. Hub progress bars
stay silent.

## License

Apache-2.0; license and third-party notices are included in the distribution.
