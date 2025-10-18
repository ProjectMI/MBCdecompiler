from pathlib import Path

from mbcdisasm import IRNormalizer
from mbcdisasm.constants import RET_MASK
from mbcdisasm.ast import (
    ASTBranch,
    ASTBuilder,
    ASTCallExpr,
    ASTCallFrame,
    ASTCallStatement,
    ASTFrameProtocol,
    ASTLiteral,
    ASTReturn,
    ASTSlotRef,
    ASTSwitch,
    ASTTailCall,
)
from mbcdisasm.ir.model import (
    IRBlock,
    IRAbiEffect,
    IRCall,
    IRDispatchCase,
    IRDispatchIndex,
    IRIf,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTailCall,
    IRStackEffect,
    IRSlot,
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
    assert procedure.entry_reasons
    assert procedure.blocks

    block = procedure.blocks[0]
    assert block.successors is not None
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
    assert first.exit_offsets == (0x0100,)
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
    assert tuple(sorted(procedure.exit_offsets)) == (0x0210, 0x0220)
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
    assert switch_stmt.helper == 0x1111
    assert switch_stmt.cases[0].key == 0x01
    assert switch_stmt.cases[0].target == 0x2222
    assert switch_stmt.enum_name == "Helper1111"
    assert switch_stmt.cases[0].key_alias == "Helper1111.K_0001"
    segment = ast_program.segments[0]
    assert segment.enums and segment.enums[0].name == "Helper1111"
    assert segment.enums[0].members[0].name == "K_0001"
    assert ast_program.enums and ast_program.enums[0].name == "Helper1111"
    assert isinstance(statements[1], ASTReturn)


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
    assert switch_stmt.enum_name == "IoFlushTail"
    assert switch_stmt.cases[0].key_alias == "IoFlushTail.K_0010"


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
        not isinstance(statement, ASTBranch)
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
    assert switch_stmt.index_expr is not None
    assert switch_stmt.index_expr.render() == "word0"
    assert switch_stmt.index_mask == 0x0007
    assert switch_stmt.index_base == 0x0001


def test_ast_builder_resolves_slot_reference() -> None:
    builder = ASTBuilder()
    expr = builder._resolve_expr("slot(0x0004)", {})
    assert isinstance(expr, ASTSlotRef)
    assert expr.slot.space is MemSpace.FRAME
    assert expr.slot.index == 0x0004


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
    assert switch_stmt.index_expr is switch_stmt.call
    assert switch_stmt.index_mask == 0x000F


def test_ast_builder_reuses_call_frame_argument() -> None:
    builder = ASTBuilder()
    literal = ASTLiteral(0x1234)
    call = IRCall(target=0x1000, args=())
    frame = builder._build_call_frame(call, (literal,))
    assert frame is not None
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
    assert first_switch.enum_name == "Dispatch_0x660A"
    assert second_switch.enum_name == "Dispatch_0x660A"
    assert first_switch.cases[0].key_alias == "Dispatch_0x660A.K_0003"
    assert second_switch.cases[1].key_alias == "Dispatch_0x660A.K_0004"

    segment_enums = ast_program.segments[0].enums
    assert segment_enums and segment_enums[0].name == "Dispatch_0x660A"
    assert {member.value for member in segment_enums[0].members} == {0x03, 0x04}


def test_ast_builder_deduplicates_enums_across_segments() -> None:
    dispatch_a = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x01, target=0x4000, symbol=None),),
        helper=0x7000,
        helper_symbol="helper_7000",
        default=None,
    )
    dispatch_b = IRSwitchDispatch(
        cases=(IRDispatchCase(key=0x01, target=0x4000, symbol=None),),
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

    frame = next(statement for statement in block.statements if isinstance(statement, ASTCallFrame))
    assert frame.live_mask == RET_MASK
    assert len(frame.slots) == 2

    protocol = next(statement for statement in block.statements if isinstance(statement, ASTFrameProtocol))
    assert protocol.teardown == 1
    mask_values = [value for value, _ in protocol.masks]
    assert RET_MASK in mask_values

    return_stmt = next(statement for statement in block.statements if isinstance(statement, ASTReturn))
    assert return_stmt.finally_branch is not None
    kinds = [step.kind for step in return_stmt.finally_branch.steps]
    assert "frame.page_select" in kinds
    assert "io.bridge" in kinds
    assert "frame.teardown" in kinds
    mask_steps = [step for step in return_stmt.finally_branch.steps if step.kind == "frame.return_mask"]
    assert any(step.operand == RET_MASK for step in mask_steps)
    assert any(step.operand == 0x0001 for step in mask_steps)


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

    protocol = next(statement for statement in statements if isinstance(statement, ASTFrameProtocol))
    assert protocol.teardown == 2
    assert protocol.drops == 1
    mask_values = {value for value, _ in protocol.masks}
    assert RET_MASK in mask_values

    tail_stmt = next(statement for statement in statements if isinstance(statement, ASTTailCall))
    assert tail_stmt.finally_branch is not None
    final_steps = tail_stmt.finally_branch.steps
    final_kinds = [step.kind for step in final_steps]
    assert "frame.return_mask" in final_kinds
    assert "frame.teardown" in final_kinds
    assert "frame.drop" in final_kinds
    assert any(step.operand == RET_MASK for step in final_steps if step.kind == "frame.return_mask")
    assert any(step.pops == protocol.teardown for step in final_steps if step.kind == "frame.teardown")


def test_ast_finally_reflects_frame_protocol_effects() -> None:
    return_node = IRReturn(
        values=tuple(),
        varargs=False,
        cleanup=(
            IRStackEffect(mnemonic="op_29_10", operand=RET_MASK),
            IRStackEffect(mnemonic="stack_teardown", pops=3),
            IRStackEffect(mnemonic="op_01_0C", pops=0),
        ),
        abi_effects=(IRAbiEffect(kind="return_mask", operand=0x0001, alias="MaskAlias"),),
    )
    block = IRBlock(label="exit_block", start_offset=0x2000, nodes=(return_node,))
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

    protocol = next(stmt for stmt in statements if isinstance(stmt, ASTFrameProtocol))
    return_stmt = next(stmt for stmt in statements if isinstance(stmt, ASTReturn))
    assert return_stmt.finally_branch is not None

    final_map = {
        (step.kind, step.operand, step.alias): step
        for step in return_stmt.finally_branch.steps
    }

    for value, alias in protocol.masks:
        key = ("frame.return_mask", value, alias)
        assert key in final_map
    if protocol.teardown:
        key = ("frame.teardown", None, None)
        assert key in final_map
        assert final_map[key].pops == protocol.teardown
    if protocol.drops:
        key = ("frame.drop", None, None)
        assert key in final_map
        assert final_map[key].pops == protocol.drops


def test_ast_finally_includes_zero_return_mask_operand() -> None:
    return_node = IRReturn(
        values=tuple(),
        varargs=False,
        cleanup=(IRStackEffect(mnemonic="op_29_10", operand=0),),
    )
    block = IRBlock(label="mask_block", start_offset=0x3000, nodes=(return_node,))
    segment = IRSegment(
        index=0,
        start=0x3000,
        length=0x10,
        blocks=(block,),
        metrics=NormalizerMetrics(),
    )
    program = IRProgram(segments=(segment,), metrics=NormalizerMetrics())

    ast_program = ASTBuilder().build(program)
    statements = ast_program.segments[0].procedures[0].blocks[0].statements

    protocol = next(stmt for stmt in statements if isinstance(stmt, ASTFrameProtocol))
    return_stmt = next(stmt for stmt in statements if isinstance(stmt, ASTReturn))

    assert protocol.masks == ((0, None),)
    assert return_stmt.finally_branch is not None
    final_masks = [step.operand for step in return_stmt.finally_branch.steps if step.kind == "frame.return_mask"]
    assert 0 in final_masks
