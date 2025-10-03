import pytest

from mbcdisasm.ir import (
    IRBuildMap,
    IRCall,
    IRLiteral,
    IRLoad,
    IRReturn,
    IRStore,
    IRTestSetBranch,
    MemSpace,
    build_segment_ir,
)
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo


def _make_word(opcode: int, mode: int, operand: int, offset: int) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | operand
    return InstructionWord(offset, raw)


def _knowledge() -> KnowledgeBase:
    def mk(
        mnemonic: str,
        *,
        control_flow: str = "fallthrough",
        stack_delta: int = 0,
        category: str | None = None,
    ) -> OpcodeInfo:
        return OpcodeInfo(
            mnemonic=mnemonic,
            summary=None,
            control_flow=control_flow,
            category=category,
            stack_delta=stack_delta,
        )

    annotations = {
        "00:00": mk("push_literal", stack_delta=1, category="literal"),
        "04:00": mk("reduce_pair", stack_delta=-1, category="reduce"),
        "29:00": mk("tailcall_dispatch", control_flow="call", stack_delta=-1, category="tailcall"),
        "30:00": mk("return_values", control_flow="return", stack_delta=0, category="return"),
        "26:00": mk("test_branch", control_flow="branch", stack_delta=-1, category="branch"),
        "27:00": mk("testset_branch", control_flow="branch", stack_delta=-1, category="branch"),
        "70:00": mk("indirect_access", stack_delta=1, category="indirect_load"),
        "70:01": mk("indirect_access", stack_delta=-1, category="indirect_store"),
    }
    return KnowledgeBase(annotations)


def test_tailcall_sequence_is_collapsed() -> None:
    knowledge = _knowledge()
    words = [
        _make_word(0x00, 0x00, 0x0001, 0),
        _make_word(0x29, 0x00, 0x1234, 4),
        _make_word(0x30, 0x00, 0x0000, 8),
    ]
    program = build_segment_ir(words, knowledge)
    block = program.blocks[0]
    assert len(block.operations) == 2
    call, ret = block.operations
    assert isinstance(call, IRCall)
    assert call.tail is True
    assert call.args == [IRLiteral(1)]
    assert isinstance(ret, IRReturn)
    assert ret.values == []


def test_literal_reduce_window_collapses_into_map() -> None:
    knowledge = _knowledge()
    words = [
        _make_word(0x00, 0x00, 0x0001, 0),
        _make_word(0x00, 0x00, 0x0002, 4),
        _make_word(0x00, 0x00, 0x0003, 8),
        _make_word(0x00, 0x00, 0x0004, 12),
        _make_word(0x04, 0x00, 0x0000, 16),
        _make_word(0x04, 0x00, 0x0000, 20),
    ]
    program = build_segment_ir(words, knowledge)
    block = program.blocks[0]
    assert len(block.operations) == 1
    op = block.operations[0]
    assert isinstance(op, IRBuildMap)
    assert len(op.items) == 2
    assert op.items[0] == (IRLiteral(1), IRLiteral(2))
    assert op.items[1] == (IRLiteral(3), IRLiteral(4))


def test_testset_branch_produces_assigning_predicate() -> None:
    knowledge = _knowledge()
    words = [
        _make_word(0x00, 0x00, 0x0001, 0),
        _make_word(0x27, 0x00, 0x0000, 4),
    ]
    program = build_segment_ir(words, knowledge)
    block = program.blocks[0]
    assert any(isinstance(op, IRTestSetBranch) for op in block.operations)


def test_indirect_access_classifies_spaces() -> None:
    knowledge = _knowledge()
    words = [
        _make_word(0x70, 0x00, 0x0002, 0),
        _make_word(0x00, 0x00, 0x0005, 4),
        _make_word(0x70, 0x01, 0x9000, 8),
    ]
    program = build_segment_ir(words, knowledge)
    block = program.blocks[0]
    loads = [op for op in block.operations if isinstance(op, IRLoad)]
    stores = [op for op in block.operations if isinstance(op, IRStore)]
    assert loads and stores
    assert loads[0].slot.space is MemSpace.FRAME
    assert stores[0].slot.space is MemSpace.CONST


def test_metrics_report_counts() -> None:
    knowledge = _knowledge()
    words = [
        _make_word(0x00, 0x00, 0x0001, 0),
        _make_word(0x29, 0x00, 0x1234, 4),
        _make_word(0x30, 0x00, 0x0000, 8),
        _make_word(0x70, 0x00, 0x0004, 12),
    ]
    program = build_segment_ir(words, knowledge)
    metrics = program.metrics
    assert metrics.calls == 1
    assert metrics.tail_calls == 1
    assert metrics.returns == 1
    assert metrics.loads == 1
    assert metrics.raw_remaining == 0
