"""Source-model capture and translation to the medmlx-core graph contract.

Import ``capture.convert`` in the Torch conversion environment and
``medmlx_core.graph.runtime.Session`` in the MLX inference environment. Core
owns the graph format and execution; no converter is required for inference.
"""
