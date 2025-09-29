import json

from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.stack_diagnostics import analyze_stack, render_stack_diagnostics
from mbcdisasm.vm_analysis import estimate_stack_io
from mbcdisasm.manual_semantics import ManualSemanticAnalyzer
from mbcdisasm.knowledge import KnowledgeBase


def _make_analyzer(tmp_path) -> ManualSemanticAnalyzer:
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_small",
                    "summary": "Push literal chunk",
                    "stack_delta": 1,
                    "tags": ["literal"],
                },
                "02:00": {
                    "name": "drop_stack_top",
                    "summary": "Consume one value",
                    "stack_delta": -1,
                },
            }
        ),
        "utf-8",
    )
    knowledge = KnowledgeBase.load(kb_path)
    return ManualSemanticAnalyzer(knowledge)


def _build_instruction(analyzer: ManualSemanticAnalyzer, offset: int, key: str) -> IRInstruction:
    semantics = analyzer.describe_key(key)
    inputs, outputs = estimate_stack_io(semantics)
    return IRInstruction(
        offset=offset,
        key=key,
        mnemonic=semantics.mnemonic,
        operand=0,
        stack_delta=semantics.stack_delta,
        control_flow=semantics.control_flow,
        semantics=semantics,
        stack_inputs=inputs,
        stack_outputs=outputs,
    )


def test_stack_diagnostics_highlight_underflows(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    literal = _build_instruction(analyzer, 0x0000, "01:00")
    consumer = _build_instruction(analyzer, 0x0004, "02:00")
    second_consumer = _build_instruction(analyzer, 0x0008, "02:00")
    block = IRBlock(start=0x0000, instructions=[literal, consumer, second_consumer], successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    diagnostics = analyze_stack(program)
    assert diagnostics.max_depth >= 1
    assert diagnostics.total_underflows == 1

    lines = render_stack_diagnostics(diagnostics)
    assert any("underflow" in line for line in lines)


def test_stack_diagnostics_merges_predecessors(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    push = _build_instruction(analyzer, 0x0000, "01:00")
    drop = _build_instruction(analyzer, 0x0004, "02:00")

    head = IRBlock(start=0x0000, instructions=[push], successors=[0x0010, 0x0020])
    left = IRBlock(start=0x0010, instructions=[drop], successors=[0x0030])
    right = IRBlock(start=0x0020, instructions=[], successors=[0x0030])
    merge = IRBlock(start=0x0030, instructions=[drop], successors=[])
    program = IRProgram(
        segment_index=0,
        blocks={
            head.start: head,
            left.start: left,
            right.start: right,
            merge.start: merge,
        },
    )

    diagnostics = analyze_stack(program)
    merge_state = diagnostics.block_states[0x0030]
    assert merge_state.exit_depth == 0
    assert merge_state.entry_tokens and merge_state.entry_tokens[0].startswith("value_")
    assert diagnostics.total_underflows >= merge_state.underflow_count()


def test_render_stack_diagnostics_groups_sections(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    literal = _build_instruction(analyzer, 0x0000, "01:00")
    drop = _build_instruction(analyzer, 0x0004, "02:00")
    block = IRBlock(start=0x0000, instructions=[literal, drop], successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    diagnostics = analyze_stack(program)
    rendered = render_stack_diagnostics(diagnostics)
    assert rendered[0] == "stack analysis:"
    assert any(line.startswith("block overview:") for line in rendered)
    assert any(line.startswith("instruction traces:") for line in rendered)


def test_stack_diagnostics_serialisation(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    literal = _build_instruction(analyzer, 0x0000, "01:00")
    drop = _build_instruction(analyzer, 0x0004, "02:00")
    block = IRBlock(start=0x0000, instructions=[literal, drop], successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    diagnostics = analyze_stack(program)
    payload = diagnostics.to_dict()
    assert "0x000000" in payload["blocks"]
    block_payload = payload["blocks"]["0x000000"]
    assert block_payload["entry_depth"] == 0
    assert block_payload["exit_depth"] == 0
    assert block_payload["instructions"][0]["mnemonic"] == literal.mnemonic
