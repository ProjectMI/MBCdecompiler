import json
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
    assert "if comparison" in rendered
    assert "while" not in rendered  # forward branch only
    assert "return state_value_2" in rendered or "return State" in rendered


def test_highlevel_reconstruction_infers_parameters(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb_params.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "10:00": {
                    "name": "consume_pair",
                    "summary": "Consumes two stack values and produces no output.",
                    "stack_delta": -2,
                },
                "11:00": {
                    "name": "return_void",
                    "summary": "Return top of stack",
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
        start=0x0000,
        instructions=[
            _make_instruction(analyzer, 0x0000, "10:00", 0, None),
            _make_instruction(analyzer, 0x0004, "11:00", 0, "return"),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=1, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert "function segment_001(arg_0, arg_1)" in rendered
    assert function.metadata.parameter_count == 2
    assert any("inferred function argument" in warning for warning in function.metadata.warnings)


def test_variable_renamer_prefers_string_hints(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb_strings.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_small",
                    "summary": "Push literal string",
                    "stack_delta": 1,
                    "tags": ["literal"],
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

    block = IRBlock(
        start=0,
        instructions=[
            _make_instruction(analyzer, 0x0000, "01:00", 0x0041, None),
            _make_instruction(analyzer, 0x0004, "04:00", 0, "return"),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=2, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    assert 'local text_a = "A"' in rendered


def test_usage_analyzer_populates_metadata(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb_usage.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "01:00": {
                    "name": "push_literal_char",
                    "summary": "Pushes a character literal",
                    "stack_delta": 1,
                    "tags": ["literal"],
                },
                "02:00": {
                    "name": "helper_call",
                    "summary": "Consumes a parameter value",
                    "stack_delta": -2,
                    "tags": ["call"],
                },
                "03:00": {
                    "name": "return_void",
                    "summary": "Return from function",
                    "stack_delta": 0,
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
            _make_instruction(analyzer, 0, "01:00", 0x0042, None),
            _make_instruction(analyzer, 4, "02:00", 0, None),
            _make_instruction(analyzer, 8, "03:00", 0, "return"),
        ],
        successors=[],
    )
    program = IRProgram(segment_index=3, blocks={block.start: block})

    reconstructor = HighLevelReconstructor(knowledge)
    function = reconstructor.from_ir(program)
    rendered = reconstructor.render([function])

    metadata = function.metadata
    assert metadata.helper_calls == 1
    assert metadata.variable_count >= 1
    assert metadata.parameter_count == 1
    assert metadata.parameter_usage.get(function.parameters[0], 0) == 1
    assert metadata.helper_breakdown.get("helper_call") == 1
    assert any(value == '"B"' for value in metadata.string_constants.values())

    lines = rendered.splitlines()
    assert "-- string literals:" in rendered
    assert any(line.startswith("-- -") and '"B"' in line for line in lines)
    assert "-- parameter usage:" in rendered
    expected_param_line = f"-- - {function.parameters[0]}: 1"
    assert expected_param_line in lines
    assert "-- helper usage:" in rendered
    assert any("-- - helper_call: 1" in line for line in lines)
