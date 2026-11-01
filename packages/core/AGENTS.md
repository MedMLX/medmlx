# medmlx-core

Shared MLX runtime used by the `medmlx` package and the independent model
runtimes.

## Rules

- Core owns the graph schema 2 JSON codec and Metal executor, plus shared
  geometry in `registration.py`. Source capture and operator translation
  belong to `medmlx-convert`.
- Inference depends only on MLX and NumPy; never import Torch, TensorFlow,
  Core AI, Core ML or a converter.
- Schema 1 is retired: require regeneration rather than adding a compatibility
  path.
- Keep the independent TensorFlow oracle under `tests/oracles/`.

## Checks

- `make qa-core` and `make test-core` from the repository root.
