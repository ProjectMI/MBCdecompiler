import json
from pathlib import Path

from mbcdisasm.highlevel import HighLevelReconstructor
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.lua_formatter import LuaRenderOptions
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


def _string_chunks(text: str) -> list[int]:
    data = text.encode("ascii")
    if len(data) % 2 == 1:
        data += b"\x00"
    values: list[int] = []
    for index in range(0, len(data), 2):
        low = data[index]
        high = data[index + 1]
        values.append((high << 8) | low)
    return values


def _extend_with_string(
    analyzer: ManualSemanticAnalyzer,
    instructions: list[IRInstruction],
    offset: int,
    text: str,
) -> int:
    for chunk in _string_chunks(text):
        instructions.append(_make_instruction(analyzer, offset, "01:00", chunk, None))
        offset += 4
    return offset


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
        or "return enum_" in rendered
        or "return State" in rendered
    )


def test_string_literal_sequences_annotated(tmp_path: Path) -> None:
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
                    "name": "return_top",
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

    instructions: list[IRInstruction] = []
    offset = 0x0000
    offset = _extend_with_string(analyzer, instructions, offset, "Hello")
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))
    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert 'string literal sequence: "Hello"' in rendered
    assert "-- literal runs:" in rendered
    assert '-- - 0x000000 kind=string count=3: "Hello"' in rendered
    assert "-- literal run report:" in rendered
    assert "-- block 0x000000:" in rendered
    assert "-- - literal tokens:" in rendered
    assert 'literal statistics' in rendered
    assert "-- stack summary:" in rendered
    assert "-- stack operations:" in rendered


def test_stack_anomalies_injected_into_render(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "drop_value",
                    "summary": "Consume without producing",
                    "stack_delta": -1,
                }
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)

    block = IRBlock(
        start=0,
        instructions=[_make_instruction(analyzer, 0, "01:00", 0, None)],
        successors=[],
    )
    program = IRProgram(segment_index=3, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "-- stack anomalies:" in rendered
    assert "placeholder" in rendered


def test_vm_trace_alignment_avoids_underflow_warning(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "10:00": {
                    "name": "consume_value",
                    "summary": "Drop top of stack",
                    "stack_delta": -1,
                }
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)

    block = IRBlock(
        start=0x0000,
        instructions=[
            _make_instruction(analyzer, 0x0000, "10:00", 0, None),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=0, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)

    assert not any(
        "underflow generated placeholder" in warning
        for warning in function.metadata.warnings
    )
    assert any("stack placeholder" in warning for warning in function.metadata.warnings)


def test_string_sequences_drive_function_naming(tmp_path: Path) -> None:
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
                    "name": "return_top",
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

    instructions: list[IRInstruction] = []
    offset = 0x0000
    offset = _extend_with_string(
        analyzer,
        instructions,
        offset,
        "Player name setting function 1.0",
    )
    offset = _extend_with_string(analyzer, instructions, offset, "set_name")
    offset = _extend_with_string(analyzer, instructions, offset, "Usage:")
    offset = _extend_with_string(
        analyzer,
        instructions,
        offset,
        "    set_name <value>,",
    )
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))

    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=4, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert function.name == "set_name"
    assert "function set_name()" in rendered
    assert "-- literal runs:" in rendered
    assert 'literal statistics' in rendered


def test_literal_report_toggle(tmp_path: Path) -> None:
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
                    "name": "return_top",
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

    instructions: list[IRInstruction] = []
    offset = 0x0000
    offset = _extend_with_string(analyzer, instructions, offset, "World")
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))
    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    options = LuaRenderOptions(emit_literal_report=False)
    reconstructor = HighLevelReconstructor(knowledge, options=options)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "-- literal runs:" in rendered
    assert "-- literal run report:" not in rendered
