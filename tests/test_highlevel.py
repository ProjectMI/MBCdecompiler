import json
from pathlib import Path

from mbcdisasm.highlevel import HighLevelReconstructor
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.manual_semantics import ManualSemanticAnalyzer
from mbcdisasm.lua_formatter import LuaRenderOptions
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


def _build_program(
    analyzer: ManualSemanticAnalyzer,
    texts: list[str],
    *,
    segment: int,
) -> IRProgram:
    instructions: list[IRInstruction] = []
    offset = 0
    for text in texts:
        offset = _extend_with_string(analyzer, instructions, offset, text)
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))
    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    return IRProgram(segment_index=segment, blocks={block.start: block})


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

    assert 'string literal sequence: "Hello" (len=5) (name=hello; locals=hello_0, hello_1, hello_2; type=identifier; tokens=[hello]; entropy=' in rendered
    assert 'case=title' in rendered
    assert 'density=0.200' in rendered
    assert "-- string literal sequences:" in rendered
    assert '-- - 0x000000 len=5 chunks=3: "Hello" (name=hello, symbols=[hello_0, hello_1, hello_2], type=identifier, tokens=[hello], entropy=' in rendered
    assert 'case=title' in rendered
    assert 'density=0.200' in rendered
    assert "-- string classification summary:" in rendered
    assert "-- - identifier: 1" in rendered
    assert "local hello_0 =" in rendered
    assert "local hello_1 =" in rendered
    assert "local hello_2 =" in rendered
    assert "return hello_0, hello_1, hello_2" in rendered
    assert "string categories: identifier=1" in rendered
    assert "-- string catalog:" in rendered
    assert "-- - hello: len=5 chunks=3 type=identifier" in rendered


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
    assert "-- string literal sequences:" in rendered


def test_string_sequences_rename_call_results(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    base_text = "SetNameAction"
    chunk_count = len(_string_chunks(base_text))
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
                "03:00": {
                    "name": "combine_name_chunks",
                    "summary": "Combine previously pushed name chunks",
                    "stack_delta": 0,
                    "tags": ["call"],
                    "stack_inputs": chunk_count,
                    "stack_outputs": 1,
                },
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)

    instructions: list[IRInstruction] = []
    offset = 0x0000
    offset = _extend_with_string(analyzer, instructions, offset, base_text)
    call_offset = offset
    instructions.append(_make_instruction(analyzer, call_offset, "03:00", 0, None))
    return_offset = call_offset + 4
    instructions.append(_make_instruction(analyzer, return_offset, "02:00", 0, "return"))

    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=5, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "local set_name_action_value =" in rendered
    consumer_hint = f"combine_name_chunks@0x{call_offset:06X}"
    assert consumer_hint in rendered
    assert f"used_by=[{consumer_hint}]" in rendered


def test_string_literals_generate_snake_case_locals(tmp_path: Path) -> None:
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
    offset = _extend_with_string(analyzer, instructions, offset, "SetPlayerName")
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))
    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "local set_player_name_0 =" in rendered
    assert 'string literal sequence: "SetPlayerName" (len=13) (name=set_player_name; locals=set_player_name_0, set_player_name_1, set_player_name_2, set_player_name_3, set_player_name_4, set_player_name_5, set_player_name_6; type=identifier; tokens=[set, player, name]; entropy=' in rendered
    assert 'case=mixed' in rendered
    assert "-- - set_player_name: len=13 chunks=7 type=identifier" in rendered
    assert "return set_player_name_0, set_player_name_1, set_player_name_2" in rendered


def test_disable_string_catalog_option(tmp_path: Path) -> None:
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
    offset = _extend_with_string(analyzer, instructions, offset, "CatalogTest")
    instructions.append(_make_instruction(analyzer, offset, "02:00", 0, "return"))
    block = IRBlock(start=0x0000, instructions=instructions, successors=[])
    program = IRProgram(segment_index=0, blocks={block.start: block})

    options = LuaRenderOptions(emit_string_catalog=False)
    reconstructor = HighLevelReconstructor(knowledge, options=options)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "string literal sequences:" in rendered
    assert "-- string catalog:" not in rendered


def test_string_catalog_groups_sequences(tmp_path: Path) -> None:
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

    program_a = _build_program(analyzer, ["AlphaBeta"], segment=1)
    program_b = _build_program(analyzer, ["system/path/config"], segment=2)

    reconstructor = HighLevelReconstructor(knowledge)
    function_a = reconstructor.from_ir(program_a)
    function_b = reconstructor.from_ir(program_b)
    rendered = reconstructor.render([function_a, function_b])

    assert "-- string catalog:" in rendered
    assert "-- - alpha_beta" in rendered
    assert "type=identifier" in rendered
    assert "-- - system:" in rendered
    assert "type=path" in rendered
    assert "tokens=[system, path, config]" in rendered


def test_string_insight_report_json_serialisation(tmp_path: Path) -> None:
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
    program = _build_program(analyzer, ["Alpha", "Beta"], segment=3)
    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    report = reconstructor.string_insight_report([function])

    output = tmp_path / "insights.json"
    output.write_text(json.dumps(report, indent=2), "utf-8")
    loaded = json.loads(output.read_text("utf-8"))
    assert loaded["entry_count"] == 1
    assert loaded["summary"]["total_sequences"] == 1


def test_string_insight_report_structure(tmp_path: Path) -> None:
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
    program = _build_program(analyzer, ["AlphaBeta"], segment=0)
    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)

    report = reconstructor.string_insight_report([function])
    assert report["entry_count"] == 1
    entry = report["entries"][0]
    assert report["classifications"][0][0] == "identifier"
    assert report["summary"]["total_sequences"] == 1
    assert report["summary"]["average_entropy"] > 0
    assert report["summary"]["top_tokens"][0][0] == "alpha"
    assert entry["function"] == function.name
    assert entry["classification"] == "identifier"
    assert entry["case_style"] == "mixed"
    assert entry["token_density"] > 0
    assert entry["text"].startswith("Alpha")
    assert entry["consumers"] == []
