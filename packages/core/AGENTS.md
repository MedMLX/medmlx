# medmlx-core

## Source documentation

For new or changed owned source code, document actual behavior and non-obvious
rationale in docstrings and inline comments, including scientific/data contracts;
verify claims against the implementation, and preserve native documentation
conventions. Avoid boilerplate and unrelated, upstream, or generated
documentation rewrites.

## Graph/runtime ownership

- Core owns `graph/` schema 2, its only JSON codec and Metal executor, plus
  `registration.py` shared geometry. Source capture and source-framework
  operation translation belong to `medmlx-convert`.
- Base inference dependencies remain MLX and NumPy. Do not import Torch,
  TensorFlow, Core AI, Core ML or a converter from inference code.
- Schema 1 is retired; require regeneration rather than an old-name shim or
  silent compatibility path. Keep source operation names out of graph data.
- Keep the independent TensorFlow oracle under `tests/oracles/`; the existing
  MONAI fixtures retain their separate schema and source metadata.

## Node runtime for development

- Use a patched host Node for Pyright. The `pyright[nodejs]` extra was rejected
  because its wheel currently installs an older Node and can undo runtime
  security updates during environment synchronization.
