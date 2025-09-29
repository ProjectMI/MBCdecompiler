import json
from pathlib import Path

from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.manual_semantics import ManualSemanticAnalyzer
from mbcdisasm.stack_annotations import summarise_stack_behaviour
from mbcdisasm.vm_analysis import estimate_stack_io


def _instruction(
    analyzer: ManualSemanticAnalyzer,
    offset: int,
    key: str,
    operand: int,
    control_flow: str | None,
) -> IRInstruction:
    semantics = analyzer.describe_key(key)
    inputs, outputs = estimate_stack_io(semantics)
    return IRInstruction(
        offset=offset,
        key=key,
        mnemonic=semantics.mnemonic,
        operand=operand,
        stack_delta=semantics.stack_delta,
        control_flow=control_flow or semantics.control_flow,
        semantics=semantics,
        stack_inputs=inputs,
        stack_outputs=outputs,
    )


def test_stack_summary_tracks_underflow(tmp_path: Path) -> None:
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_small",
                    "summary": "push literal",
                    "stack_delta": 1,
                    "tags": ["literal"],
                },
                "02:00": {
                    "name": "drop_value",
                    "summary": "consume value",
                    "stack_delta": -1,
                },
            }
        ),
        "utf-8",
    )
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    analyzer = ManualSemanticAnalyzer(knowledge)

    block = IRBlock(
        start=0,
        instructions=[
            _instruction(analyzer, 0, "01:00", 1, None),
            _instruction(analyzer, 4, "02:00", 0, None),
            _instruction(analyzer, 8, "02:00", 0, None),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=1, blocks={block.start: block})

    summary = summarise_stack_behaviour(program)
    lines = summary.summary_lines()

    assert summary.instruction_count == 3
    assert summary.total_underflows == 1
    assert summary.placeholder_values >= 1
    assert any("block 0x000000" in line for line in lines)
    assert summary.anomalies
    assert any("drop_value" in line for line in summary.anomaly_lines())
    assert summary.operations
