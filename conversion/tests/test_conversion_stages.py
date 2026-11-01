"""A bad intermediate graph must fail at the stage that introduced the error."""

import json
from pathlib import Path

import pytest
import torch

from medmlx_convert import export_and_check
from medmlx_convert.boundary import Module, function
from medmlx_convert.provenance import BuildSource


class Difference(Module):
    def forward(self, image: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return image - offset


def test_bad_decomposition_is_rejected_before_graph_edits_or_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject an upstream decomposition error through the real public export path."""
    original = function(torch.export.ExportedProgram, "run_decompositions")

    def broken_decomposition(program: torch.export.ExportedProgram,
                             table: object) -> torch.export.ExportedProgram:
        result = original(program, table)
        assert isinstance(result, torch.export.ExportedProgram)
        for node in result.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.sub.Tensor:
                node.target = torch.ops.aten.add.Tensor
        result.graph_module.recompile()
        return result

    monkeypatch.setattr(torch.export.ExportedProgram, "run_decompositions", broken_decomposition)
    stem = Path("build/stage_gate/bad_decomposition")
    stem.parent.mkdir(parents=True, exist_ok=True)
    inputs = {"image": torch.ones(2, 3), "offset": torch.full((2, 3), 0.25)}
    with pytest.raises(ValueError, match="coreai_decomposition"):
        export_and_check(Difference(), inputs, ["features"], stem, tolerance=1e-5,
                         source=BuildSource.capture(Path(__file__), synthetic=True))
    captured = json.loads(stem.with_name(f"{stem.name}_coreai_capture")
                          .with_suffix(".verification.json").read_text())
    failed = stem.with_name(f"{stem.name}_coreai_decomposition")
    decomposed = json.loads(failed.with_suffix(".verification.json").read_text())
    assert captured["comparisons"][0]["verdict"] == "pass"
    assert decomposed["comparisons"][0]["verdict"] == "fail"
    assert decomposed["comparisons"][0]["max_abs_diff"] == pytest.approx(0.5)
    assert failed.with_suffix(".graph.txt").is_file()
    assert not stem.with_name(f"{stem.name}_coreai_graph").with_suffix(".graph.txt").exists()
