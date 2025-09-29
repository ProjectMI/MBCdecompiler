import json
import json
from pathlib import Path

from mbcdisasm.highlevel import (
    FunctionMetadata,
    HighLevelFunction,
    HighLevelReconstructor,
    HighLevelStack,
    StackValue,
)
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.lua_ast import LiteralExpr, NameExpr, wrap_block
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
    assert "return literal_" in rendered or "return State" in rendered


def test_inline_string_collection(tmp_path: Path) -> None:
    kb_path = tmp_path / "inline_kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "72:65": {
                    "name": "inline_ascii_chunk_7265",
                    "stack_delta": 0,
                    "summary": "Inline chunk",
                },
                "00:01": {
                    "name": "no_op",
                    "stack_delta": 0,
                },
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)

    block = IRBlock(
        start=0,
        instructions=[
            _make_instruction(analyzer, 0x10, "72:65", 0x6E74, None),
            _make_instruction(analyzer, 0x14, "00:01", 0, None),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=5, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "inline_segment_005" in rendered
    assert "rent" in rendered
    assert "- inline string data: 1 entries (4 bytes)" in rendered
    assert "- inline segments: 1" in rendered
    assert "- average inline chunk: 4.0 bytes" in rendered
    assert "largest inline chunk: 4 bytes at segment 005 offset 0x000010" in rendered
    assert "- inline sample:" in rendered
    assert "-- inline chunk" in rendered
    summary = function.render()
    assert "- inline chunks: 1" in summary
    assert "- inline bytes: 4" in summary
    report = reconstructor.build_report([function])
    assert report["inline"]["entries"] == 1
    assert report["inline_samples"]
    assert report["functions"][0]["metadata"]["inline_chunks"] == 1

    # Disabling inline comments should remove the preview while keeping metadata.
    silent = HighLevelReconstructor(
        knowledge, options=LuaRenderOptions(emit_inline_comments=False)
    )
    quiet_function = silent.from_ir(program)
    quiet_rendered = silent.render([quiet_function])
    assert "-- inline chunk" not in quiet_rendered


def test_highlevel_stack_tracks_py_values() -> None:
    stack = HighLevelStack()
    literal = LiteralExpr("1", py_value=1)
    statements, value = stack.push_literal(literal)
    assert statements and isinstance(value, StackValue)
    assert value.py_value == 1
    alias_statements, alias = stack.push_expression(NameExpr(value.name))
    assert alias_statements == []
    assert alias.py_value == 1
    popped = stack.pop_single()
    assert popped.py_value == 1
    stack.push_literal(LiteralExpr("2", py_value=2))
    description = stack.describe()
    assert description and description[0].startswith("[0]")
    assert "2" in description[0]
    snapshot = stack.snapshot()
    assert snapshot[-1]["py_value"] == 2


def test_highlevel_function_to_dict() -> None:
    metadata = FunctionMetadata(block_count=1, instruction_count=2)
    function = HighLevelFunction(name="demo", body=wrap_block([]), metadata=metadata)
    payload = function.to_dict()
    assert payload["name"] == "demo"
    assert payload["metadata"]["blocks"] == 1


def test_build_report_without_inline(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb_report.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_small",
                    "summary": "literal",
                    "stack_delta": 1,
                    "tags": ["literal"],
                },
                "04:00": {
                    "name": "return_value",
                    "summary": "return",
                    "stack_delta": -1,
                    "control_flow": "return",
                },
            }
        ),
        "utf-8",
    )
    knowledge = KnowledgeBase.load(kb_path)
    analyzer = ManualSemanticAnalyzer(knowledge)
    block = IRBlock(
        start=0,
        instructions=[
            _make_instruction(analyzer, 0, "01:00", 1, None),
            _make_instruction(analyzer, 4, "04:00", 0, "return"),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=7, blocks={block.start: block})
    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    report = reconstructor.build_report([function])
    assert report["inline"]["entries"] == 0
    assert report["inline_samples"] == []
    assert report["functions"][0]["metadata"]["inline_bytes"] == 0
