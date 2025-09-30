"""Unit tests covering Lua rendering utilities."""

from __future__ import annotations

from pathlib import Path

from mbcdisasm.highlevel import (
    FunctionMetadata,
    HighLevelFunction,
    HighLevelReconstructor,
)
from mbcdisasm.literal_sequences import (
    LiteralDescriptor,
    LiteralRun,
    compute_literal_statistics,
)
from mbcdisasm.lua_ast import Assignment, LiteralExpr, NameExpr, ReturnStatement, wrap_block
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.lua_formatter import (
    CommentFormatter,
    EnumRegistry,
    HelperRegistry,
    HelperSignature,
    LuaRenderOptions,
    LuaWriter,
    MethodSignature,
)
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect
from mbcdisasm.knowledge import KnowledgeBase


def _make_semantics(
    mnemonic: str,
    *,
    summary: str = "",
    tags: tuple[str, ...] = (),
    control_flow: str | None = None,
    inputs: int = 0,
    outputs: int = 0,
    delta: int = 0,
) -> InstructionSemantics:
    """Helper used by tests to craft :class:`InstructionSemantics` objects."""

    return InstructionSemantics(
        key="00:00",
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=summary,
        control_flow=control_flow,
        stack_delta=0,
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=delta, source="test"),
        tags=tuple(tag.lower() for tag in tags),
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=0,
        stack_outputs=0,
        uses_operand=False,
        operand_hint=None,
        vm_method="",
        vm_call_style="",
    )


def _render_lines(statements):
    writer = LuaWriter()
    for stmt in statements:
        stmt.emit(writer)
    return writer.render().splitlines()


def test_lua_writer_label_spacing() -> None:
    writer = LuaWriter()
    writer.write_line("function demo()")
    with writer.indented():
        writer.write_line("local value = 1")
        writer.write_label("block_000001")
        writer.write_line("return value")
    writer.write_line("end")

    rendered = writer.render().splitlines()
    assert rendered[0] == "function demo()"
    assert rendered[1] == "  local value = 1"
    # A blank line should separate the label from previous statements.
    assert rendered[2] == ""
    assert rendered[3] == "::block_000001::"
    assert rendered[4] == "  return value"


def test_helper_registry_metadata_toggle() -> None:
    registry = HelperRegistry()
    registry.register_function(
        HelperSignature(
            name="reduce_pair",
            summary="Test helper",
            inputs=2,
            outputs=0,
            uses_operand=False,
        )
    )
    registry.register_method(
        MethodSignature(
            name="structured_field_store",
            summary="Store value",
            inputs=2,
            outputs=0,
            uses_operand=True,
            struct="struct",
            method="structuredFieldStore",
        )
    )

    writer = LuaWriter()
    registry.render(writer, CommentFormatter())
    text = writer.render()
    assert "inputs=2; outputs=0" in text

    writer = LuaWriter()
    registry.render(
        writer,
        CommentFormatter(),
        options=LuaRenderOptions(emit_stub_metadata=False),
    )
    text_no_meta = writer.render()
    assert "inputs=2; outputs=0" not in text_no_meta


def test_helper_and_enum_counts() -> None:
    registry = HelperRegistry()
    registry.register_function(
        HelperSignature(
            name="reduce_pair",
            summary="",
            inputs=2,
            outputs=0,
            uses_operand=False,
        )
    )
    registry.register_method(
        MethodSignature(
            name="structured_field_store",
            summary="",
            inputs=2,
            outputs=0,
            uses_operand=False,
            struct="struct",
            method="structuredFieldStore",
        )
    )
    assert registry.function_count() == 1
    assert registry.method_count() == 1
    assert registry.struct_count() == 1

    enums = EnumRegistry()
    enums.register("demo", 1, "ONE")
    enums.register("demo", 2, "TWO")
    enums.register("other", 3, "THREE")
    assert enums.namespace_count() == 2
    assert enums.total_values() == 3


def test_enum_registry_metadata_toggle() -> None:
    registry = EnumRegistry()
    registry.register("test_enum", 1, "ONE")
    registry.register("test_enum", 2, "TWO")

    writer = LuaWriter()
    registry.render(writer)
    text = writer.render()
    assert "enum test_enum: 2 entries" in text

    writer = LuaWriter()
    registry.render(writer, options=LuaRenderOptions(emit_enum_metadata=False))
    text_no_meta = writer.render()
    assert "enum test_enum" not in text_no_meta


def test_highlevel_function_summary_and_warnings() -> None:
    metadata = FunctionMetadata(block_count=3, instruction_count=9, warnings=["stack underflow"])
    function = HighLevelFunction(
        name="segment_001",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata,
        segment_index=1,
    )
    rendered = function.render().splitlines()
    assert rendered[0] == "-- function summary:"
    assert "-- - blocks: 3" in rendered
    assert "-- - instructions: 9" in rendered
    assert "-- - literal instructions: 0" in rendered
    assert "-- - helper invocations: 0" in rendered
    assert "-- - branches: 0" in rendered
    assert "-- stack reconstruction warnings:" in rendered
    assert "-- - stack underflow" in rendered
    assert "function segment_001()" in rendered


def test_highlevel_function_instruction_profile_rendered() -> None:
    metadata = FunctionMetadata(
        block_count=1,
        instruction_count=5,
        literal_count=2,
        helper_calls=1,
        branch_count=1,
        mnemonic_counts={"load": 3, "store": 1, "branch": 1},
        tag_counts={"literal": 2, "control": 1},
    )
    function = HighLevelFunction(
        name="segment_profile",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata,
        segment_index=8,
    )
    rendered = function.render().splitlines()
    assert any("instruction profile (top mnemonics)" in line for line in rendered)
    assert any("load: 3" in line for line in rendered)
    assert any("instruction tags (top categories)" in line for line in rendered)
    assert any("literal: 2" in line for line in rendered)
    assert any("instruction density:" in line for line in rendered)


def test_highlevel_function_string_metadata_block() -> None:
    descriptor_a = LiteralDescriptor(
        kind="string",
        text="demo ",
        expression=LiteralExpr("\"demo \""),
    )
    descriptor_b = LiteralDescriptor(
        kind="string",
        text="string",
        expression=LiteralExpr("\"string\""),
    )
    run = LiteralRun(
        kind="string",
        descriptors=(descriptor_a, descriptor_b),
        offsets=(0x1234, 0x1238),
        block_start=0x1200,
    )
    stats = compute_literal_statistics([run])
    metadata = FunctionMetadata(
        block_count=1,
        instruction_count=2,
        literal_runs=[run],
        literal_stats=stats,
    )
    function = HighLevelFunction(
        name="segment_demo",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata,
        segment_index=2,
    )
    rendered = function.render().splitlines()
    assert "-- function summary:" in rendered
    assert "-- - string literal sequences: 1" in rendered
    assert "-- literal runs:" in rendered
    assert '-- - 0x001234 kind=string count=2: "demo string"' in rendered
    assert any("literal statistics" in line for line in rendered)


def test_placeholder_locals_rendered() -> None:
    metadata = FunctionMetadata(
        block_count=1,
        instruction_count=1,
        placeholders=("stack_0", "stack_1"),
    )
    function = HighLevelFunction(
        name="segment_placeholder",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata,
        segment_index=6,
    )
    rendered = function.render().splitlines()
    assert any("placeholder locals inserted" in line for line in rendered)
    assert any(line.strip() == "local stack_0" for line in rendered)
    assert any(line.strip() == "local stack_1" for line in rendered)


def test_module_summary_toggle(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb_summary.json")
    options = LuaRenderOptions()
    reconstructor = HighLevelReconstructor(knowledge, options=options)
    metadata = FunctionMetadata(block_count=1, instruction_count=2)
    function = HighLevelFunction(
        name="segment_010",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata,
        segment_index=3,
    )
    reconstructor._helper_registry.register_function(
        HelperSignature(
            name="reduce_pair",
            summary="Test helper",
            inputs=2,
            outputs=0,
            uses_operand=False,
        )
    )
    reconstructor._enum_registry.register("demo_enum", 1, "ONE")
    output = reconstructor.render([function])
    assert "module summary:" in output
    assert "- functions: 1" in output
    assert "- helper functions: 1" in output
    assert "- struct helpers: 0" in output
    assert "- literal instructions: 0" in output
    assert "- branch instructions: 0" in output
    assert "- enum namespaces: 1 (1 values)" in output
    assert "- stack warnings: 0" in output
    assert "module metadata table" in output
    assert "local __module_metadata = {" in output
    assert "module_metadata()" in output
    assert "local __metadata_index = {" in output
    assert "local __metadata_warnings = {" in output
    assert "local __metadata_placeholders = {" in output
    assert "local __metadata_literal_runs = {" in output
    assert "local __metadata_mnemonics = {" in output
    assert "local __metadata_tags = {" in output
    assert "local __module_mnemonics = {" in output
    assert "local __module_tags = {" in output
    assert "local __metadata_density = {" in output
    assert "local function function_metadata(" in output
    assert "local function function_warnings(" in output
    assert "local function function_placeholders(" in output
    assert "local function function_literal_runs(" in output
    assert "local function function_mnemonics(" in output
    assert "local function function_tags(" in output
    assert "local function module_mnemonics(" in output
    assert "local function module_tags(" in output
    assert "local function function_density(" in output
    assert "local function module_density(" in output
    assert "local __helper_functions = {" in output
    assert "local __struct_helpers = {" in output
    assert "local function helper_metadata(" in output
    assert "local function struct_helper_metadata(" in output

    no_summary = HighLevelReconstructor(
        knowledge,
        options=LuaRenderOptions(emit_module_summary=False),
    )
    no_summary._helper_registry.register_function(
        HelperSignature(
            name="reduce_pair",
            summary="Test helper",
            inputs=2,
            outputs=0,
            uses_operand=False,
        )
    )
    result = no_summary.render([function])
    assert "module summary" not in result
    assert "module metadata table" in result
    assert "local __metadata_index" in result


def test_module_summary_with_multiple_functions(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb_multi.json")
    reconstructor = HighLevelReconstructor(knowledge)

    metadata_a = FunctionMetadata(
        block_count=1,
        instruction_count=2,
        literal_count=1,
        helper_calls=0,
        branch_count=1,
        mnemonic_counts={"load": 1, "branch": 1},
        tag_counts={"literal": 1, "control": 1},
    )
    metadata_b = FunctionMetadata(
        block_count=2,
        instruction_count=3,
        literal_count=2,
        helper_calls=1,
        branch_count=0,
        warnings=["placeholder"],
        placeholders=("stack_0",),
        mnemonic_counts={"store": 2, "load": 1},
        tag_counts={"literal": 2, "call": 1},
    )
    fn_a = HighLevelFunction(
        name="segment_a",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata_a,
        segment_index=4,
    )
    fn_b = HighLevelFunction(
        name="segment_b",
        body=wrap_block([ReturnStatement()]),
        metadata=metadata_b,
        segment_index=5,
    )

    reconstructor._helper_registry.register_function(
        HelperSignature(
            name="reduce_pair",
            summary="",
            inputs=2,
            outputs=0,
            uses_operand=False,
        )
    )
    reconstructor._helper_registry.register_method(
        MethodSignature(
            name="structured_field_store",
            summary="",
            inputs=2,
            outputs=0,
            uses_operand=False,
            struct="struct",
            method="structuredFieldStore",
        )
    )
    reconstructor._enum_registry.register("demo", 1, "ONE")

    rendered = reconstructor.render([fn_a, fn_b])
    assert "module summary:" in rendered
    assert "- functions: 2" in rendered
    assert "- helper functions: 1" in rendered
    assert "- struct helpers: 1" in rendered
    assert "- literal instructions: 3" in rendered
    assert "- branch instructions: 1" in rendered
    assert "- stack warnings: 1" in rendered
    assert "- placeholder values: 1" in rendered
    assert "- instruction mnemonics:" in rendered
    assert "- top mnemonics:" in rendered
    assert "- instruction tags:" in rendered
    assert "- top tags:" in rendered
    assert "- instruction density:" in rendered
    assert "module metadata table" in rendered
    assert "density_map = {" in rendered
    assert "placeholders = {" in rendered
    assert '"stack_0"' in rendered
    assert "__metadata_warnings" in rendered
    assert '"placeholder"' in rendered
    assert "__helper_functions" in rendered
    assert "reduce_pair" in rendered


def test_comment_deduplication(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb.json")
    reconstructor = HighLevelReconstructor(knowledge)

    literal_semantics = _make_semantics(
        "literal_push",
        summary="Pushes a zero-extended literal constant to seed the evaluation stack.",
        tags=("literal",),
    )

    stmt_zero = Assignment([NameExpr("literal_0")], LiteralExpr("0"), is_local=True)
    stmt_one = Assignment([NameExpr("literal_1")], LiteralExpr("1"), is_local=True)

    first = reconstructor._decorate_with_comment([stmt_zero], literal_semantics)
    second = reconstructor._decorate_with_comment([stmt_one], literal_semantics)

    first_lines = _render_lines(first)
    second_lines = _render_lines(second)

    assert any(line.startswith("--") for line in first_lines), "first literal should carry a comment"
    assert all("--" not in line for line in second_lines), "duplicate comment should be suppressed"

    # Introducing a different comment resets the deduplication window.
    other_semantics = _make_semantics(
        "reduce_pair",
        summary="Primary reducer",
        tags=("generic",),
    )
    reducer_call = Assignment([NameExpr("result")], LiteralExpr("reduce_pair(x, y)"), is_local=True)
    reconstructor._decorate_with_comment([reducer_call], other_semantics)
    stmt_two = Assignment([NameExpr("literal_2")], LiteralExpr("2"), is_local=True)
    again = reconstructor._decorate_with_comment([stmt_two], literal_semantics)
    again_lines = _render_lines(again)
    assert any(line.startswith("--") for line in again_lines)


def test_function_metadata_counts(tmp_path: Path) -> None:
    knowledge = KnowledgeBase.load(tmp_path / "kb_counts.json")
    reconstructor = HighLevelReconstructor(knowledge)

    literal_semantics = _make_semantics(
        "literal_push",
        summary="literal",
        tags=("literal",),
        outputs=1,
        delta=1,
    )
    helper_semantics = _make_semantics(
        "helper_call",
        summary="helper",
        tags=("call",),
        inputs=1,
        delta=-1,
    )
    branch_semantics = _make_semantics(
        "branch",
        summary="branch",
        control_flow="branch",
        inputs=1,
        delta=-1,
    )

    literal_instr = IRInstruction(
        offset=0,
        key="00:00",
        mnemonic="literal_push",
        operand=0,
        stack_delta=1,
        control_flow=None,
        semantics=literal_semantics,
        stack_inputs=0,
        stack_outputs=1,
    )
    helper_instr = IRInstruction(
        offset=4,
        key="00:01",
        mnemonic="helper_call",
        operand=0,
        stack_delta=-1,
        control_flow=None,
        semantics=helper_semantics,
        stack_inputs=1,
        stack_outputs=0,
    )
    branch_instr = IRInstruction(
        offset=8,
        key="00:02",
        mnemonic="branch",
        operand=0,
        stack_delta=-1,
        control_flow="branch",
        semantics=branch_semantics,
        stack_inputs=1,
        stack_outputs=0,
    )

    block = IRBlock(start=0, instructions=[literal_instr, helper_instr, branch_instr], successors=[0x10])
    program = IRProgram(segment_index=0, blocks={0: block})

    function = reconstructor.from_ir(program)
    metadata = function.metadata
    assert metadata.literal_count == 1
    assert metadata.helper_calls == 1
    assert metadata.branch_count == 1

    rendered = function.render()
    assert "- literal instructions: 1" in rendered
    assert "- helper invocations: 1" in rendered
    assert "- branches: 1" in rendered
