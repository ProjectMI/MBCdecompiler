import json
from pathlib import Path

from mbcdisasm.highlevel import HighLevelReconstructor
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.manual_semantics import ManualSemanticAnalyzer
from mbcdisasm.vm_analysis import estimate_stack_io


def _make_instruction(
    analyzer: ManualSemanticAnalyzer,
    offset: int,
    key: str,
    operand: int,
    control_flow,
):
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


def test_highlevel_reconstruction_generates_control_flow(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_small",
                    "summary": "Load an enum state",
                    "stack_delta": 1,
                    "tags": ["literal"],
                    "enum_values": {"0": "IDLE", "1": "RUN"},
                    "enum_namespace": "State",
                },
                "02:00": {
                    "name": "compare_equal",
                    "summary": "Check equality",
                    "stack_delta": -1,
                    "tags": ["comparison"],
                },
                "03:00": {
                    "name": "branch_if_true",
                    "summary": "Branch when true",
                    "stack_delta": -1,
                    "control_flow": "branch",
                },
                "04:00": {
                    "name": "return_value",
                    "summary": "Return top value",
                    "stack_delta": -1,
                    "control_flow": "return",
                },
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)

    block0 = IRBlock(
        start=0x0000,
        instructions=[
            _make_instruction(analyzer, 0x0000, "01:00", 1, None),
            _make_instruction(analyzer, 0x0004, "01:00", 0, None),
            _make_instruction(analyzer, 0x0008, "02:00", 0, None),
            _make_instruction(analyzer, 0x000C, "03:00", 0, "branch"),
        ],
        successors=[0x0010, 0x0020],
    )
    block1 = IRBlock(
        start=0x0010,
        instructions=[
            _make_instruction(analyzer, 0x0010, "04:00", 0, "return"),
        ],
        successors=[],
    )
    block2 = IRBlock(
        start=0x0020,
        instructions=[
            _make_instruction(analyzer, 0x0020, "01:00", 0, None),
            _make_instruction(analyzer, 0x0024, "04:00", 0, "return"),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=0, blocks={
        block0.start: block0,
        block1.start: block1,
        block2.start: block2,
    })

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "local State =" in rendered
    assert "if cmp_" in rendered
    assert "while" not in rendered  # forward branch only
    assert (
        "return literal_" in rendered
        or "return text_" in rendered
        or "return enum_" in rendered
        or "return State" in rendered
    )

    stats = reconstructor.literal_report.statistics()
    assert stats.total == 3
    assert stats.by_category.get("ENUM", 0) == 3
