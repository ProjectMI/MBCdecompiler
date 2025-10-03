from mbcdisasm import (
    InstructionWord,
    KnowledgeBase,
    Normalizer,
    OpcodeInfo,
)
from mbcdisasm.ir import (
    IRBuildMap,
    IRCall,
    IRLiteral,
    IRLoad,
    IRReturn,
    IRStore,
    IRTestSetBranch,
    MemSpace,
)


def make_word(offset: int, opcode: int, mode: int = 0, operand: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def build_knowledge(entries):
    return KnowledgeBase(entries)


def test_tail_call_is_collapsed_with_return():
    knowledge = build_knowledge(
        {
            "10:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "11:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "20:00": OpcodeInfo(
                mnemonic="tailcall_dispatch",
                control_flow="call",
                category="call",
                attributes={"arg_count": 2},
            ),
            "30:00": OpcodeInfo(
                mnemonic="return_values",
                control_flow="return",
                category="return",
                attributes={"return_arity": 1},
            ),
        }
    )
    normalizer = Normalizer(knowledge)
    words = [
        make_word(0, 0x10),
        make_word(4, 0x11),
        make_word(8, 0x20, operand=3),
        make_word(12, 0x30),
    ]
    result = normalizer.normalise(words)
    nodes = result.blocks[0].nodes
    assert isinstance(nodes[0], IRCall)
    assert nodes[0].tail is True
    assert len(nodes[0].args) == 2
    assert isinstance(nodes[1], IRReturn)
    assert nodes[1].arity == 1
    assert result.metrics.calls == 1
    assert result.metrics.tail_calls == 1
    assert result.metrics.returns == 1


def test_literal_reduce_chain_collapses_to_map():
    knowledge = build_knowledge(
        {
            "10:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "11:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "12:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "13:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "20:00": OpcodeInfo(mnemonic="reduce_pair", category="reduce", stack_delta=-1),
        }
    )
    normalizer = Normalizer(knowledge)
    words = [
        make_word(0, 0x10, operand=1),
        make_word(4, 0x11, operand=2),
        make_word(8, 0x12, operand=3),
        make_word(12, 0x13, operand=4),
        make_word(16, 0x20),
        make_word(20, 0x20),
    ]
    result = normalizer.normalise(words)
    nodes = result.blocks[0].nodes
    assert len(nodes) == 1
    aggregate = nodes[0]
    assert isinstance(aggregate, IRBuildMap)
    assert len(aggregate.pairs) == 2
    assert all(isinstance(pair[0], IRLiteral) for pair in aggregate.pairs)
    assert result.metrics.aggregates == 1
    assert result.metrics.reduce_replaced == 2


def test_testset_branch_is_lifted():
    knowledge = build_knowledge(
        {
            "10:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "20:00": OpcodeInfo(
                mnemonic="testset_branch",
                category="testset",
                control_flow="branch",
                attributes={"assign": "flag", "then_target": 0x100, "else_target": 0x200},
            ),
        }
    )
    normalizer = Normalizer(knowledge)
    words = [
        make_word(0, 0x10, operand=1),
        make_word(4, 0x20, operand=0x300),
    ]
    result = normalizer.normalise(words)
    nodes = result.blocks[0].nodes
    assert isinstance(nodes[0], IRTestSetBranch)
    assert nodes[0].var == "flag"
    assert isinstance(nodes[0].expr, IRLiteral)
    assert nodes[0].then_target == 0x100
    assert nodes[0].else_target == 0x200
    assert result.metrics.testset_branches == 1


def test_indirect_accesses_are_mapped():
    knowledge = build_knowledge(
        {
            "10:00": OpcodeInfo(mnemonic="literal", category="literal", stack_delta=1),
            "20:00": OpcodeInfo(mnemonic="indirect_store", category="indirect_store"),
            "21:00": OpcodeInfo(mnemonic="indirect_load", category="indirect_load"),
        }
    )
    normalizer = Normalizer(knowledge)
    words = [
        make_word(0, 0x10, operand=7),
        make_word(4, 0x20, operand=0x10),
        make_word(8, 0x21, operand=0x4000),
    ]
    result = normalizer.normalise(words)
    nodes = result.blocks[0].nodes
    assert isinstance(nodes[0], IRStore)
    assert nodes[0].slot.space is MemSpace.FRAME
    assert isinstance(nodes[0].value, IRLiteral)
    assert isinstance(nodes[1], IRLoad)
    assert nodes[1].slot.space is MemSpace.GLOBAL
    assert result.metrics.stores == 1
    assert result.metrics.loads == 1


def test_unknown_instructions_are_counted():
    knowledge = build_knowledge({})
    normalizer = Normalizer(knowledge)
    words = [
        make_word(0, 0xFF),
        make_word(4, 0x10, operand=1),
    ]
    result = normalizer.normalise(words)
    assert result.metrics.raw_remaining >= 1
    assert result.metrics.calls == 0


def test_plain_return_is_exposed():
    knowledge = build_knowledge(
        {
            "30:00": OpcodeInfo(
                mnemonic="return_values",
                control_flow="return",
                category="return",
                attributes={"return_arity": 2},
            ),
        }
    )
    normalizer = Normalizer(knowledge)
    words = [make_word(0, 0x30)]
    result = normalizer.normalise(words)
    nodes = result.blocks[0].nodes
    assert len(nodes) == 1
    assert isinstance(nodes[0], IRReturn)
    assert nodes[0].arity == 2
    assert result.metrics.returns == 1
