"""Regenerate the README peak-memory table from installed model declarations.

The table lists each discovered model's declared ``peak_memory_gb``, so it
describes the model packages installed in the running environment. Run it in
the locked environment after a pin bump (``make readme-memory``) so the README
matches the versions ``uv.lock`` resolves; ``--check`` exits 1 when the README
differs instead of rewriting it.
"""

import argparse
import sys
from pathlib import Path

from medmlx import registry

README = Path(__file__).resolve().parents[1] / "README.md"
START = "<!-- memory-table:start -->"
END = "<!-- memory-table:end -->"


def render(models: list[registry.ModelSpec]) -> str:
    """Return the table and undeclared-model sentence for the given declarations."""
    by_peak: dict[float, list[str]] = {}
    undeclared: list[str] = []
    for spec in models:
        if spec.peak_memory_gb is None:
            undeclared.append(spec.id)
        else:
            # Declarations carry measurement precision; one decimal is what users act on.
            by_peak.setdefault(round(spec.peak_memory_gb, 1), []).append(spec.id)
    lines = ["| Model | Peak MLX memory |", "| :--- | ---: |"]
    for peak in sorted(by_peak, reverse=True):
        ids = ", ".join(f"`{model_id}`" for model_id in sorted(by_peak[peak]))
        lines.append(f"| {ids} | {peak:.1f} GB |")
    if undeclared:
        names = ", ".join(f"`{model_id}`" for model_id in sorted(undeclared))
        lines += ["", f"Models without a declared peak ({names}) record their measured peak "
                  "after each run, and `medmlx describe` reports it."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the README peak-memory table.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the README is stale")
    args = parser.parse_args()
    discovery = registry.discover()
    if discovery.errors:
        print(f"Model discovery failed: {discovery.errors}", file=sys.stderr)
        return 1
    text = README.read_text()
    head, start, rest = text.partition(START)
    _, end, tail = rest.partition(END)
    if not start or not end:
        print(f"README is missing the {START} / {END} markers", file=sys.stderr)
        return 1
    updated = f"{head}{START}\n{render(list(discovery.models.values()))}\n{END}{tail}"
    if updated == text:
        return 0
    if args.check:
        print("README memory table is stale: run make readme-memory", file=sys.stderr)
        return 1
    README.write_text(updated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
