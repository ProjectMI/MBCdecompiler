import json
from pathlib import Path

from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.manual_semantics import ManualSemanticAnalyzer


def _write_knowledge(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), "utf-8")
    return path


def test_literal_tag_inferred_for_opcode00(tmp_path) -> None:
    kb_path = _write_knowledge(
        tmp_path / "kb.json",
        {
            "schema": 1,
            "opcode_modes": {
                "00:67": {
                    "count": 1,
                    "stack_deltas": {"1": 1},
                    "operand_types": {},
                    "preceding": {},
                    "following": {},
                }
            },
            "annotations": {
                "00:67": {
                    "stack_delta": 1,
                    "operand_hint": "medium",
                }
            },
        },
    )
    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)
    semantics = analyzer.describe_key("00:67")

    assert "literal" in semantics.tags
    assert semantics.stack_effect.outputs >= 1
    assert semantics.stack_effect.inputs == 0


def test_binary_keyword_adjusts_inputs(tmp_path) -> None:
    kb_path = _write_knowledge(
        tmp_path / "kb.json",
        {
            "schema": 1,
            "opcode_modes": {
                "04:00": {
                    "count": 1,
                    "stack_deltas": {"-1": 1},
                    "operand_types": {},
                    "preceding": {},
                    "following": {},
                }
            },
            "annotations": {
                "04:00": {
                    "name": "reduce_pair",
                    "summary": "Primary binary reducer that consumes two operands",
                    "stack_delta": -2,
                }
            },
        },
    )
    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)
    semantics = analyzer.describe_key("04:00")

    assert "binary" in semantics.tags
    assert semantics.stack_effect.inputs >= 2

