# Third-party notices

MedMLX-authored API, CLI, MCP and tooling code is licensed under
[Apache-2.0](LICENSE). Upstream portions retain their own attribution and terms.

## MONAI-derived shared runtime

The shared runtime in `packages/core` contains sliding-window algorithms derived
from MONAI 1.6.0 and channel-last layer arithmetic translated from MONAI 1.4.0.
Copyright (c) MONAI Consortium. Licensed under Apache-2.0; see the
[core attribution and full MONAI license](packages/core/THIRD_PARTY_NOTICES.md).
The native NumPy/MLX implementations and their restricted APIs are MedMLX
modifications. The core is also distributed separately as `medmlx-core`.

## Dependencies and model runtimes

The frontend, core and conversion environments declare their dependencies in
[pyproject.toml](pyproject.toml),
[packages/core/pyproject.toml](packages/core/pyproject.toml) and
[conversion/pyproject.toml](conversion/pyproject.toml). Installed dependencies
retain their respective licenses.

Model runtime repositories register independently through `medmlx.models`.
Their code, checkpoints, tokenizer assets and datasets retain their original
terms; installing a frontend extra does not relicense those materials.
