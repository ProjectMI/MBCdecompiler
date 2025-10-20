from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.constants import RET_MASK
from mbcdisasm.ast import (
    ASTAssign,
    ASTConditional,
    ASTBuilder,
    ASTCallABI,
    ASTCallArgumentSlot,
    ASTCallExpr,
    ASTCallStatement,
    ASTFrameChannelEffect,
    ASTFrameDropEffect,
    ASTFrameMaskEffect,
    ASTFrameProtocolEffect,
    ASTFrameTeardownEffect,
    ASTIOEffect,
    ASTIntegerLiteral,
    ASTImmediateOperand,
    ASTMemoryRead,
    ASTReturn,
    ASTSwitch,
    ASTTailCall,
    ASTSymbolTypeFamily,
    ASTPredicateKind,
)
from mbcdisasm.ir.model import (
    IRBlock,
    IRAbiEffect,
    IRBankedLoad,
    IRCall,
    IRCallReturn,
    IRDispatchCase,
    IRDispatchIndex,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRStackEffect,
    IRSlot,
    MemRef,
    MemSpace,
    NormalizerMetrics,
)

from tests.test_ir_normalizer import build_container


def test_ast_builder_reconstructs_cfg(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    builder = ASTBuilder()
    ast_program = builder.build(program)

    assert ast_program.metrics.procedure_count >= 1
    assert ast_program.metrics.block_count >= 1
    assert ast_program.metrics.call_sites >= 1
    assert ast_program.metrics.call_args >= ast_program.metrics.resolved_call_args

    segment = ast_program.segments[0]
    assert segment.procedures
    procedure = segment.procedures[0]
    assert procedure.entry.reasons
    assert all(exit.reasons for exit in procedure.exits)
    assert procedure.blocks

    block = procedure.blocks[0]
    assert isinstance(block.successors, tuple)
    rendered = [statement.render() for statement in block.statements]
    assert any("call" in line for line in rendered)
    assert any(line.startswith("return") for line in rendered)
    assert procedure.name == f"proc_{procedure.entry_offset:04X}"

    summary = ast_program.metrics.describe()
    assert "procedures=" in summary
    assert "calls=" in summary


def test_ast_builder_splits_after_return_sequences() -> None:
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x20,
        blocks=(
            IRBlock(
                label="block_entry",
                start_offset=0x0100,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
            IRBlock(
                label="block_follow",
                start_offset=0x0110,
                nodes=(
                    IRLoad(slot=IRSlot(MemSpace.FRAME, 0), target="word0"),
                    IRReturn(values=("ret0",), varargs=False),
                ),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    ast_segment = ast_program.segments[0]
    entries = [procedure.entry_offset for procedure in ast_segment.procedures]
    assert entries == [0x0100, 0x0110]
    assert all(
        procedure.name == f"proc_{procedure.entry_offset:04X}"
        for procedure in ast_segment.procedures
    )
    first = ast_segment.procedures[0]
    assert tuple(exit.offset for exit in first.exits) == (0x0100,)
    assert first.exits[0].reasons and first.exits[0].reasons[0].kind == "return"
    assert len(first.blocks) == 1


def test_ast_builder_uses_postdominators_for_exits() -> None:
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x40,
        blocks=(
            IRBlock(
                label="block_cond",
                start_offset=0x0200,
                nodes=(IRIf(condition="bool0", then_target=0x0210, else_target=0x0220),),
            ),
            IRBlock(
                label="block_then",
                start_offset=0x0210,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
            IRBlock(
                label="block_else",
                start_offset=0x0220,
                nodes=(IRReturn(values=("ret1",), varargs=False),),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    ast_segment = ast_program.segments[0]
    assert len(ast_segment.procedures) == 1
    procedure = ast_segment.procedures[0]
    assert tuple(sorted(exit.offset for exit in procedure.exits)) == (0x0210, 0x0220)
    assert {block.start_offset for block in procedure.blocks} == {0x0200, 0x0210, 0x0220}


def test_ast_builder_converts_dispatch_with_trailing_table() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x01, target=0x2222, symbol="proc_2222"),),
        helper=0x1111,
        helper_symbol="helper_1111",
        default=0x3333,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0100,
        nodes=(dispatch, IRCall(target=0x1111, args=(), symbol="helper_1111"), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    switch_stmt = statements[0]
    assert isinstance(switch_stmt, ASTSwitch)
    assert switch_stmt.helper is not None
    assert switch_stmt.helper.address == 0x1111
    assert switch_stmt.helper.symbol == "helper_1111"
    assert switch_stmt.cases[0].key == 0x01
    assert switch_stmt.cases[0].target == 0x2222
    assert switch_stmt.enum_name is None
    assert switch_stmt.cases[0].key_alias is None
    segment = ast_program.segments[0]
    assert not segment.enums
    assert not ast_program.enums
    assert isinstance(statements[1], ASTReturn)


def test_ast_builder_strips_literal_marker_tokens(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    builder = ASTBuilder()
    ast_program = builder.build(program)

    for segment in ast_program.segments:
        for procedure in segment.procedures:
            for block in procedure.blocks:
                for statement in block.statements:
                    assert "literal_marker" not in statement.render()


def test_ast_builder_converts_dispatch_with_leading_call() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x02, target=0x4444, symbol=None),),
        helper=0x5555,
        helper_symbol=None,
        default=None,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0200,
        nodes=(IRCall(target=0x5555, args=()), dispatch, IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    call_stmt = statements[0]
    assert isinstance(call_stmt, ASTCallStatement)
    assert call_stmt.call.target == 0x5555
    assert call_stmt.call.symbol is None
    enums = ast_program.segments[0].enums
    assert not enums
    assert not ast_program.enums


def test_ast_builder_simplifies_single_case_dispatch_to_call() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x07, target=0xAAAA, symbol=None),),
        helper=0x6060,
        helper_symbol="helper_6060",
        default=None,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0220,
        nodes=(dispatch, IRCall(target=0x6060, args=(), symbol="helper_6060")),
    )
    segment = IRSegment(
        index=0,
        start=0x0220,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert all("dispatch.data" not in stmt.render() for stmt in statements)
    assert isinstance(statements[0], ASTCallStatement)
    assert statements[0].call.target == 0x6060
    assert statements[0].call.symbol == "helper_6060"


def test_ast_switch_marks_io_dispatch() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x10, target=0x2000, symbol=None),),
        helper=0x00F0,
        helper_symbol="io.flush_tail",
        default=0x1234,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0250,
        nodes=(dispatch, IRCall(target=0x00F0, args=(), symbol="io.flush_tail"), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0250,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = statements[0]
    assert isinstance(switch_stmt, ASTSwitch)
    assert switch_stmt.kind == "io"
    assert switch_stmt.enum_name is None
    assert switch_stmt.cases[0].key_alias is None


def test_ast_builder_drops_redundant_tailcall_after_switch() -> None:
    dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x05, target=0x3000, symbol=None),),
        helper=0x7777,
        helper_symbol=None,
    )
    helper_call = IRCall(target=0x7777, args=())
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0300,
        nodes=(
            dispatch,
            helper_call,
            IRTailCall(call=IRCall(target=0x7777, args=()), returns=()),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    assert any(isinstance(stmt, ASTCallStatement) for stmt in statements)
    assert not any(isinstance(stmt, ASTTailCall) for stmt in statements)


def test_ast_builder_prunes_redundant_branch_blocks() -> None:
    segment = IRSegment(
        index=0,
        start=0x0400,
        length=0x20,
        blocks=(
            IRBlock(
                label="block_entry",
                start_offset=0x0400,
                nodes=(IRIf(condition="stack_top", then_target=0x0410, else_target=0x0410),),
            ),
            IRBlock(
                label="block_return",
                start_offset=0x0410,
                nodes=(IRReturn(values=("ret0",), varargs=False),),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    segment_ast = ast_program.segments[0]
    assert len(segment_ast.procedures) == 1
    procedure = segment_ast.procedures[0]
    assert {block.start_offset for block in procedure.blocks} == {0x0410}
    assert all(
        not isinstance(statement, ASTConditional)
        for block in procedure.blocks
        for statement in block.statements
    )


def test_ast_builder_drops_empty_procedures() -> None:
    segment = IRSegment(
        index=0,
        start=0x0500,
        length=0x10,
        blocks=(
            IRBlock(
                label="block_empty",
                start_offset=0x0500,
                nodes=tuple(),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    assert not ast_program.segments[0].procedures
    assert ast_program.metrics.procedure_count == 0


def test_ast_switch_carries_index_metadata() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x03, target=0x8888, symbol=None),
            IRDispatchCase(key=0x04, target=0x9999, symbol=None),
        ),
        helper=0x7777,
        helper_symbol=None,
        default=None,
        index=IRDispatchIndex(source="word0", mask=0x0007, base=0x0001),
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0300,
        nodes=(dispatch, IRCall(target=0x7777, args=()), IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    switch_stmt = next(statement for statement in statements if isinstance(statement, ASTSwitch))
    assert switch_stmt.index.expression is not None
    assert switch_stmt.index.expression.render() == "word0"
    assert switch_stmt.index.mask == 0x0007
    assert switch_stmt.index.base == 0x0001


def test_ast_builder_resolves_slot_reference() -> None:
    builder = ASTBuilder()
    expr = builder._resolve_expr("slot(0x0004)", {})
    assert isinstance(expr, ASTMemoryRead)
    rendered = expr.render()
    assert rendered.startswith("addr{")
    assert "type=frame" in rendered
    assert "offset=0x0004" in rendered


def test_ast_builder_reconstructs_dispatch_call_index() -> None:
    call = IRCall(target=0x4444, args=("word0",), symbol="helper_4444")
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x01, target=0x9999, symbol=None),
            IRDispatchCase(key=0x02, target=0xAAAA, symbol=None),
        ),
        helper=0x4444,
        helper_symbol="helper_4444",
        default=None,
        index=IRDispatchIndex(source=call.describe(), mask=0x000F, base=None),
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0100,
        nodes=(call, dispatch, IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements
    switch_stmt = next(statement for statement in statements if isinstance(statement, ASTSwitch))

    assert isinstance(switch_stmt.call, ASTCallExpr)
    assert switch_stmt.index.expression is switch_stmt.call
    assert switch_stmt.index.mask == 0x000F


def test_ast_builder_reuses_call_frame_argument() -> None:
    builder = ASTBuilder()
    literal = ASTIntegerLiteral(0x1234)
    call = IRCall(target=0x1000, args=())
    operand = ASTImmediateOperand(token="lit0", literal=literal)
    abi = builder._build_call_abi(call, (operand,))
    assert abi is not None
    assert isinstance(abi, ASTCallABI)
    resolved = builder._resolve_expr("slot_0", {})
    assert resolved == literal


def test_ast_builder_merges_enum_members_across_switches() -> None:
    first_dispatch = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x03, target=0x1000, symbol=None),),
        helper=0x660A,
        helper_symbol=None,
        default=None,
    )
    second_dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x03, target=0x1000, symbol=None),
            IRDispatchCase(key=0x04, target=0x2000, symbol=None),
        ),
        helper=0x660A,
        helper_symbol=None,
        default=None,
    )
    block = IRBlock(
        label="block_dispatch",
        start_offset=0x0300,
        nodes=(first_dispatch, second_dispatch, IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    first_switch, second_switch = statements[0], statements[1]
    assert isinstance(first_switch, ASTSwitch)
    assert isinstance(second_switch, ASTSwitch)
    assert first_switch.enum_name == "Helper660A"
    assert second_switch.enum_name == "Helper660A"
    assert first_switch.cases[0].key_alias == "Helper660A.K_0003"
    assert second_switch.cases[1].key_alias == "Helper660A.K_0004"

    segment_enums = ast_program.segments[0].enums
    assert segment_enums and segment_enums[0].name == "Helper660A"
    assert {member.value for member in segment_enums[0].members} == {0x03, 0x04}


def test_ast_builder_deduplicates_enums_across_segments() -> None:
    dispatch_a = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x01, target=0x4000, symbol=None),
            IRDispatchCase(key=0x02, target=0x5000, symbol=None),
        ),
        helper=0x7000,
        helper_symbol="helper_7000",
        default=None,
    )
    dispatch_b = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x01, target=0x4000, symbol=None),
            IRDispatchCase(key=0x02, target=0x5000, symbol=None),
        ),
        helper=0x7000,
        helper_symbol="helper_7000",
        default=None,
    )
    first_segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x20,
        blocks=(
            IRBlock(
                label="block0",
                start_offset=0x0100,
                nodes=(dispatch_a, IRReturn(values=(), varargs=False)),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    second_segment = IRSegment(
        index=1,
        start=0x0200,
        length=0x20,
        blocks=(
            IRBlock(
                label="block1",
                start_offset=0x0200,
                nodes=(dispatch_b, IRReturn(values=(), varargs=False)),
            ),
        ),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(first_segment, second_segment), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    first_switch = ast_program.segments[0].procedures[0].blocks[0].statements[0]
    second_switch = ast_program.segments[1].procedures[0].blocks[0].statements[0]
    assert isinstance(first_switch, ASTSwitch)
    assert isinstance(second_switch, ASTSwitch)
    assert first_switch.enum_name == "Helper7000"
    assert second_switch.enum_name == "Helper7000"
    assert ast_program.segments[0].enums and ast_program.segments[0].enums[0] is ast_program.enums[0]
    assert not ast_program.segments[1].enums
    assert len(ast_program.enums) == 1


def test_ast_builder_uses_call_symbol_for_enum_naming() -> None:
    dispatch = IRSwitchDispatch(
        cases=(
            IRDispatchCase(key=0x00, target=0x5000, symbol=None),
            IRDispatchCase(key=0x01, target=0x5002, symbol=None),
        ),
        helper=0x0569,
        helper_symbol=None,
        default=None,
    )
    block = IRBlock(
        label="block_mask",
        start_offset=0x0400,
        nodes=(
            IRCall(target=0x0569, args=(), symbol="scheduler.mask_low"),
            dispatch,
            IRReturn(values=(), varargs=False),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0400,
        length=0x20,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    switch_stmt = ast_program.segments[0].procedures[0].blocks[0].statements[0]
    assert isinstance(switch_stmt, ASTSwitch)
    assert switch_stmt.enum_name == "SchedulerMask"
    assert switch_stmt.cases[0].key_alias == "SchedulerMask.MaskClear"
    assert switch_stmt.cases[1].key_alias == "SchedulerMask.MaskBit00"
    segment_enum = ast_program.segments[0].enums[0]
    assert segment_enum.name == "SchedulerMask"


def test_ast_builder_emits_call_frame_and_finally(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    program = IRNormalizer(knowledge).normalise_container(container)
    ast_program = ASTBuilder().build(program)

    segment = ast_program.segments[0]
    procedure = segment.procedures[0]
    block = procedure.blocks[0]

    call_stmt = next(statement for statement in block.statements if isinstance(statement, ASTCallStatement))
    assert call_stmt.abi is not None
    assert isinstance(call_stmt.abi, ASTCallABI)
    assert call_stmt.abi.live_mask is not None
    assert call_stmt.abi.live_mask.value == RET_MASK
    assert len(call_stmt.abi.slots) == 2
    assert all(isinstance(slot, ASTCallArgumentSlot) for slot in call_stmt.abi.slots)

    return_stmt = next(statement for statement in block.statements if isinstance(statement, ASTReturn))
    protocol_effect = next(
        effect for effect in return_stmt.effects if isinstance(effect, ASTFrameProtocolEffect)
    )
    assert protocol_effect.teardown == 1
    mask_values = {mask.value for mask in protocol_effect.masks}
    assert RET_MASK in mask_values
    assert 0x0001 in mask_values

    frame_masks = [
        effect for effect in return_stmt.effects if isinstance(effect, ASTFrameMaskEffect)
    ]
    frame_channels = [
        effect
        for effect in return_stmt.effects
        if isinstance(effect, ASTFrameChannelEffect)
    ]
    frame_teardowns = [
        effect for effect in return_stmt.effects if isinstance(effect, ASTFrameTeardownEffect)
    ]
    io_effects = [effect for effect in return_stmt.effects if isinstance(effect, ASTIOEffect)]

    assert any(effect.channel == "page_select" for effect in frame_channels)
    assert any(effect.pops == protocol_effect.teardown for effect in frame_teardowns)
    assert any(effect.mask.value == RET_MASK for effect in frame_masks)
    assert any(effect.mask.value == 0x0001 for effect in frame_masks)
    assert any(effect.operation.value == "bridge" for effect in io_effects)


def test_ast_tailcall_emits_protocol_and_finally() -> None:
    tail_call = IRTailCall(
        call=IRCall(target=0x1234, args=(), symbol="tail_helper"),
        returns=tuple(),
        cleanup=(
            IRStackEffect(mnemonic="stack_teardown", pops=2),
            IRStackEffect(mnemonic="op_01_0C", pops=1),
        ),
        abi_effects=(IRAbiEffect(kind="return_mask", operand=RET_MASK),),
    )
    block = IRBlock(label="tail_block", start_offset=0x1000, nodes=(tail_call,))
    segment = IRSegment(
        index=0,
        start=0x1000,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    ast_program = ASTBuilder().build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    tail_stmt = next(statement for statement in statements if isinstance(statement, ASTTailCall))
    assert tail_stmt.abi is not None
    assert isinstance(tail_stmt.abi, ASTCallABI)
    protocol_effect = next(
        effect for effect in tail_stmt.effects if isinstance(effect, ASTFrameProtocolEffect)
    )
    assert protocol_effect.teardown == 2
    assert protocol_effect.drops == 1
    mask_values = {mask.value for mask in protocol_effect.masks}
    assert RET_MASK in mask_values

    frame_masks = [
        effect for effect in tail_stmt.effects if isinstance(effect, ASTFrameMaskEffect)
    ]
    frame_teardowns = [
        effect for effect in tail_stmt.effects if isinstance(effect, ASTFrameTeardownEffect)
    ]
    frame_drops = [
        effect for effect in tail_stmt.effects if isinstance(effect, ASTFrameDropEffect)
    ]
    assert any(effect.mask.value == RET_MASK for effect in frame_masks)
    assert any(effect.pops == protocol_effect.teardown for effect in frame_teardowns)
    assert any(effect.pops == protocol_effect.drops for effect in frame_drops)

def test_ast_finally_summary_matches_frame_protocol() -> None:
    return_node = IRReturn(
        values=(),
        varargs=False,
        cleanup=(
            IRStackEffect(mnemonic="stack_teardown", pops=1),
            IRStackEffect(mnemonic="stack_teardown", pops=2),
            IRStackEffect(mnemonic="op_01_0C", pops=1),
            IRStackEffect(mnemonic="op_01_30", pops=3),
            IRStackEffect(mnemonic="op_29_10", operand=RET_MASK),
            IRStackEffect(mnemonic="op_29_10", operand=RET_MASK),
        ),
        abi_effects=(IRAbiEffect(kind="return_mask", operand=0x0001),),
    )
    block = IRBlock(label="summary_block", start_offset=0x2000, nodes=(return_node,))
    segment = IRSegment(
        index=0,
        start=0x2000,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    ast_program = ASTBuilder().build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    return_stmt = next(statement for statement in statements if isinstance(statement, ASTReturn))
    protocol_effect = next(
        effect for effect in return_stmt.effects if isinstance(effect, ASTFrameProtocolEffect)
    )
    mask_pairs = {
        (effect.mask.value, effect.mask.alias)
        for effect in return_stmt.effects
        if isinstance(effect, ASTFrameMaskEffect)
    }
    expected = {(mask.value, mask.alias) for mask in protocol_effect.masks}
    assert mask_pairs == expected

    teardown_effects = [
        effect
        for effect in return_stmt.effects
        if isinstance(effect, ASTFrameTeardownEffect)
    ]
    assert len(teardown_effects) == 1
    assert teardown_effects[0].pops == protocol_effect.teardown

    drop_effects = [
        effect
        for effect in return_stmt.effects
        if isinstance(effect, ASTFrameDropEffect)
    ]
    assert len(drop_effects) == 1
    assert drop_effects[0].pops == protocol_effect.drops


def test_symbol_table_synthesises_call_signatures() -> None:
    block = IRBlock(
        label="block_entry",
        start_offset=0x0100,
        nodes=(
            IRCallReturn(
                target=0x6601,
                args=("ptr0",),
                symbol=None,
                tail=False,
                returns=("ret0",),
                varargs=False,
            ),
            IRReturn(values=("ret0",), varargs=False),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0100,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    symbols = {entry.address: entry for entry in ast_program.symbols}
    assert 0x6601 in symbols
    signature = symbols[0x6601]
    assert signature.name == "helper_6601"
    assert tuple(value.type.family for value in signature.arguments) == (
        ASTSymbolTypeFamily.ADDRESS,
    )
    assert signature.arguments[0].type.space == "mem"
    assert tuple(value.type.family for value in signature.returns) == (
        ASTSymbolTypeFamily.OPAQUE,
    )
    assert signature.returns[0].name == "opaque0"
    assert signature.calling_conventions == ("call",)
    assert signature.attributes == tuple()
    assert signature.effects == tuple()


def test_symbol_table_records_call_attributes() -> None:
    call = IRCallReturn(
        target=0x3D30,
        args=("ptr0",),
        tail=True,
        returns=(),
        varargs=True,
        abi_effects=(IRAbiEffect(kind="io.write", operand=0x2910, alias="ChatOut"),),
        symbol="io.write",
    )
    block = IRBlock(
        label="block_call",
        start_offset=0x0200,
        nodes=(call, IRReturn(values=(), varargs=False)),
    )
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)

    symbols = {entry.address: entry for entry in ast_program.symbols}
    assert 0x3D30 in symbols
    signature = symbols[0x3D30]
    assert signature.name == "io.write"
    assert signature.calling_conventions == ("tailcall",)
    assert set(signature.attributes) == {"tail", "varargs"}
    assert signature.effects and signature.effects[0].startswith("io.write(")
    assert not signature.returns
    assert signature.arguments[0].name == "addr0"
    assert signature.arguments[0].type.family is ASTSymbolTypeFamily.ADDRESS
    assert signature.arguments[0].type.space == "mem"

def test_epilogue_effects_are_deduplicated() -> None:
    cleanup = (
        IRStackEffect(mnemonic="op_01_0C", operand=0x0000),
        IRStackEffect(mnemonic="op_01_0C", operand=0x0000),
    )
    block = IRBlock(
        label="block_return",
        start_offset=0x0200,
        nodes=(IRReturn(values=(), varargs=False, cleanup=cleanup),),
    )
    segment = IRSegment(
        index=0,
        start=0x0200,
        length=0x04,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    procedure = ast_program.segments[0].procedures[0]
    terminator = procedure.blocks[0].statements[-1]
    assert isinstance(terminator, ASTReturn)
    assert len(terminator.effects) == 1


def test_testset_branch_desugars_into_assignment() -> None:
    block = IRBlock(
        label="block_branch",
        start_offset=0x0300,
        nodes=(
            IRTestSetBranch(
                var="bool0",
                expr="cond0",
                then_target=0x0310,
                else_target=0x0320,
            ),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0300,
        length=0x04,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    procedure = ast_program.segments[0].procedures[0]
    statements = procedure.blocks[0].statements
    assert isinstance(statements[0], ASTAssign)
    assert isinstance(statements[1], ASTConditional)
    assert statements[1].predicate.kind is ASTPredicateKind.VALUE


def test_banked_memory_locations_are_canonical() -> None:
    memref = MemRef(region="bank_1230", bank=0x1230, base=0x0040, page=0x01, offset=0x10)
    block = IRBlock(
        label="block_load",
        start_offset=0x0400,
        nodes=(
            IRBankedLoad(
                ref=memref,
                target="word0",
                register=0x5000,
                pointer="ptr0",
            ),
        ),
    )
    segment = IRSegment(
        index=0,
        start=0x0400,
        length=0x04,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    builder = ASTBuilder()
    ast_program = builder.build(program)
    procedure = ast_program.segments[0].procedures[0]
    assign = procedure.blocks[0].statements[0]
    assert isinstance(assign, ASTAssign)
    rendered = assign.value.render()
    assert rendered.startswith("addr{")
    assert "type=banked" in rendered
    assert "region=bank_1230" in rendered
    assert "bank=0x1230" in rendered
    assert "page=0x01" in rendered
    assert "page_reg=0x5000" in rendered
    assert "base=0x0040" in rendered
    assert "offset=0x0010" in rendered
    assert "alias=region(bank_1230)" in rendered
