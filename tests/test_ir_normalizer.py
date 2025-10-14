import json
from pathlib import Path

import pytest

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.ir import IRTextRenderer
from dataclasses import replace

from mbcdisasm.ir.model import (
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRAsciiFinalize,
    IRAsciiHeader,
    IRLiteralChunk,
    IRPageRegister,
    IRTailCall,
    IRTailcallReturn,
    IRIf,
    IRReturn,
    IRFunctionPrologue,
    IRStackEffect,
    IRTestSetBranch,
    IRBankedLoad,
    IRIndirectLoad,
    IRIndirectStore,
    IRRaw,
    IRConditionMask,
    IRIOWrite,
    IRBuildTuple,
    IRBuildArray,
    IRSwitchDispatch,
    IRTablePatch,
    IRTableBuilderBegin,
    IRTableBuilderEmit,
    IRTableBuilderCommit,
    NormalizerMetrics,
    IRIORead,
    IRAbiEffect,
)
from mbcdisasm.ir.normalizer import RawBlock, RawInstruction, _ItemList
from mbcdisasm.constants import (
    IO_SLOT,
    IO_SLOT_ALIASES,
    IO_PORT_NAME,
    PAGE_REGISTER,
    RET_MASK,
    CALL_SHUFFLE_STANDARD,
)
from mbcdisasm.mbc import Segment
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo
from mbcdisasm.analyzer.instruction_profile import (
    InstructionKind,
    InstructionProfile,
    StackEffectHint,
)
from mbcdisasm.analyzer.stack import StackEvent, StackTracker


def build_word(offset: int, opcode: int, mode: int, operand: int) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset=offset, raw=raw)


def build_ascii_word(offset: int, text: str) -> InstructionWord:
    data = text.encode("ascii", "replace")[:4].ljust(4, b" ")
    return InstructionWord(offset=offset, raw=int.from_bytes(data, "big"))


def encode_instructions(words: list[InstructionWord]) -> bytes:
    return b"".join(word.raw.to_bytes(4, "big") for word in words)


def make_stack_neutral_instruction(
    offset: int,
    mnemonic: str,
    *,
    operand: int = 0,
    kind: InstructionKind = InstructionKind.UNKNOWN,
    annotations: tuple[str, ...] = tuple(),
) -> RawInstruction:
    opcode = 0
    mode = 0
    if mnemonic.startswith("op_") and len(mnemonic) >= 8:
        try:
            opcode = int(mnemonic[3:5], 16)
            mode = int(mnemonic[6:8], 16)
        except ValueError:
            pass
    raw_value = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    word = InstructionWord(offset=offset, raw=raw_value)
    profile = InstructionProfile(
        word=word,
        info=None,
        mnemonic=mnemonic,
        summary=None,
        category=None,
        control_flow=None,
        stack_hint=StackEffectHint(nominal=0, minimum=0, maximum=0, confidence=1.0),
        kind=kind,
        traits={},
    )
    event = StackEvent(
        profile=profile,
        delta=0,
        minimum=0,
        maximum=0,
        confidence=1.0,
        depth_before=0,
        depth_after=0,
        kind=kind,
    )
    return RawInstruction(
        profile=profile,
        event=event,
        annotations=annotations,
        ssa_values=tuple(),
        ssa_kinds=tuple(),
    )


def test_epilogue_compaction_merges_raw_markers() -> None:
    knowledge = KnowledgeBase({})
    normalizer = IRNormalizer(knowledge)

    marker = make_stack_neutral_instruction(0, "op_02_00")
    marker = replace(marker, event=replace(marker.event, delta=1, depth_after=1))
    trailer = make_stack_neutral_instruction(4, "op_15_4A")

    return_node = IRReturn(values=tuple(), varargs=False)
    items = _ItemList([marker, trailer, return_node])

    normalizer._pass_epilogue_prologue_compaction(items)

    assert all(not isinstance(node, RawInstruction) for node in items)
    assert len(items) == 1
    updated_return = items[0]
    assert isinstance(updated_return, IRReturn)
    assert [step.mnemonic for step in updated_return.cleanup] == ["op_02_00", "op_15_4A"]


def write_manual(path: Path) -> KnowledgeBase:
    manual = {
        "push_literal": {
            "opcodes": ["00:00"],
            "name": "push_literal",
            "category": "literal",
            "stack_push": 1,
        },
        "literal_marker": {
            "opcodes": ["00:38", "0x00:0x72"],
            "name": "literal_marker",
            "category": "literal_marker",
        },
        "reduce_pair": {
            "opcodes": ["04:00"],
            "name": "reduce_pair",
            "category": "reduce",
            "stack_delta": -1,
        },
        "tailcall_dispatch": {
            "opcodes": ["0x2B:0x00"],
            "name": "tailcall_dispatch",
            "category": "call_dispatch",
            "stack_delta": -1,
        },
        "return_values": {
            "opcodes": ["0x30:0x00"],
            "name": "return_values",
            "category": "return_values",
        },
        "call_dispatch": {
            "opcodes": ["0x28:0x00"],
            "name": "call_dispatch",
            "category": "call_dispatch",
            "stack_delta": -1,
        },
        "op_59_FE": {
            "opcodes": ["0x59:0xFE"],
            "name": "op_59_FE",
            "category": "call_wrapper",
            "stack_delta": 0,
        },
        "op_72_23": {
            "opcodes": ["0x72:0x23"],
            "name": "op_72_23",
            "category": "call_wrapper",
            "stack_delta": 0,
        },
        "op_4B_91": {
            "opcodes": ["0x4B:0x91"],
            "name": "op_4B_91",
            "category": "call_wrapper",
            "stack_delta": 0,
        },
        "op_E4_01": {
            "opcodes": ["0xE4:0x01"],
            "name": "op_E4_01",
            "category": "call_wrapper",
            "stack_delta": 0,
        },
        "op_95_FE": {
            "opcodes": ["0x95:0xFE"],
            "name": "op_95_FE",
            "category": "call_wrapper",
            "stack_delta": 0,
        },
        "call_helpers": {
            "opcodes": ["0x10:0xE8", "0x10:0x00", "0x10:0xAC"],
            "name": "call_helpers",
            "category": "call_helpers",
        },
        "fanout": {
            "opcodes": ["0x10:0x08"],
            "name": "fanout",
            "control_flow": "fallthrough",
            "operand_role": "flags",
            "operand_aliases": {"0x2C02": "FANOUT_FLAGS"},
        },
        "branch_eq": {
            "opcodes": ["0x23:0x00"],
            "name": "branch_eq",
            "category": "branch_eq",
        },
        "testset_branch": {
            "opcodes": ["0x27:0x00", "0x27:0x33"],
            "name": "testset_branch",
            "category": "testset_branch",
        },
        "indirect_access": {
            "opcodes": ["0x69:0x01"],
            "name": "indirect_access",
            "category": "indirect_access",
        },
        "stack_teardown_1": {
            "opcodes": ["0x01:0x00"],
            "name": "stack_teardown_1",
            "category": "stack_teardown",
        },
        "stack_teardown_4": {
            "opcodes": ["0x01:0xF0"],
            "name": "stack_teardown_4",
            "category": "stack_teardown",
            "stack_delta": -4,
        },
        "stack_shuffle": {
            "opcodes": ["0x66:0x15"],
            "name": "stack_shuffle",
            "category": "stack_shuffle",
        },
        "op_29_10": {
            "opcodes": ["0x29:0x10"],
            "name": "op_29_10",
            "category": "call_mask",
        },
        "op_70_29": {
            "opcodes": ["0x70:0x29"],
            "name": "op_70_29",
            "category": "call_cleanup",
        },
    }
    manual_path = path / "manual_annotations.json"
    manual_path.write_text(json.dumps(manual, indent=2), "utf-8")
    address_table = {
        "0x1234": "test_helper_1234",
        "0x0020": "call_helper_20",
        "0x0072": "tail_helper_72",
        "0x00F0": "tail_helper_f0",
        "0x003E": "tail_helper_3e",
        "0x00ED": "tail_helper_ed",
        "0x013D": "tail_helper_13d",
        "0x01EC": "tail_helper_1ec",
        "0x01F1": "tail_helper_1f1",
        "0x032C": "tail_helper_32c",
        "0x0BF0": "tail_helper_bf0",
        "0x0FF0": "tail_helper_ff0",
        "0x16F0": "tail_helper_16f0",
        "0x6623": "dispatch_helper_6623",
        "0x6624": "dispatch_helper_6624",
    }
    (path / "address_table.json").write_text(json.dumps(address_table, indent=2), "utf-8")
    call_signatures = {
        "0x0072": {
            "arity": 2,
            "cleanup_mask": "0x2910",
            "cleanup": [
                {"mnemonic": "stack_teardown", "pops": 1},
            ],
            "shuffle": "0x4B08",
            "shuffle_options": ["0x4B08", "0x3032"],
            "prelude": [
                {
                    "kind": "raw",
                    "mnemonic": "op_F0_4B",
                    "operand": "0x4B08",
                    "effect": {"mnemonic": "op_F0_4B", "operand": "0x4B08"},
                },
                {
                    "kind": "raw",
                    "mnemonic": "op_5E_29",
                    "operand": "0x2910",
                    "effect": {"mnemonic": "op_5E_29", "operand": "0x2910"},
                },
                {
                    "kind": "raw",
                    "mnemonic": "op_6C_01",
                    "operand": "0x6C01",
                    "effect": {"mnemonic": "page_register", "operand": "0x6C01"},
                },
            ],
            "postlude": [
                {
                    "kind": "raw",
                    "mnemonic": "op_70_29",
                    "effect": {"mnemonic": "op_70_29", "inherit_operand": True},
                    "optional": True,
                },
            ],
        },
        "0x3D30": {
            "tail": True,
            "prelude": [
                {"kind": "raw", "mnemonic": "op_D0_04", "optional": True},
                {"kind": "raw", "mnemonic": "op_D8_04", "optional": True},
                {"kind": "raw", "mnemonic": "op_C4_06", "optional": True},
            ],
            "postlude": [
                {"kind": "raw", "mnemonic": "op_D0_06", "optional": True},
            ],
        },
    }
    (path / "call_signatures.json").write_text(json.dumps(call_signatures, indent=2), "utf-8")
    return KnowledgeBase.load(manual_path)


def build_container(tmp_path: Path) -> tuple[MbcContainer, KnowledgeBase]:
    knowledge = write_manual(tmp_path)

    seg0_words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_word(4, 0x00, 0x00, 0x0002),
        build_word(8, 0x6C, 0x01, 0x6C01),
        build_word(12, 0x5E, 0x29, 0x2910),
        build_word(16, 0xF0, 0x4B, 0x4B08),
        build_word(20, 0x2B, 0x00, 0x0072),
        build_word(24, 0x29, 0x10, 0x2910),
        build_word(28, 0x70, 0x29, 0x0001),
        build_word(32, 0x27, 0x00, 0x0008),
        build_word(36, 0x30, 0x00, 0x0001),
    ]
    seg1_words = [
        build_word(0, 0x00, 0x00, 0x0003),
        build_word(4, 0x00, 0x00, 0x0004),
        build_word(8, 0x04, 0x00, 0x0000),
        build_word(12, 0x23, 0x00, 0x0010),
        build_word(16, 0x00, 0x00, 0x0005),
        build_word(20, 0x27, 0x00, 0x0008),
        build_word(24, 0x11, 0xBE, 0x0000),
        build_word(28, 0x03, 0x66, 0x0005),
        build_word(32, 0x69, 0x01, 0xAE05),
        build_word(36, 0x11, 0xEE, 0x0000),
        build_word(40, 0x03, 0x66, 0x9000),
        build_word(44, 0x69, 0x01, 0xAF05),
        build_word(48, 0x01, 0x00, 0x0000),
    ]

    seg0_bytes = encode_instructions(seg0_words)
    seg1_bytes = encode_instructions(seg1_words)

    segments = [
        Segment(SegmentDescriptor(0, 0, len(seg0_bytes)), seg0_bytes),
        Segment(
            SegmentDescriptor(1, len(seg0_bytes), len(seg0_bytes) + len(seg1_bytes)),
            seg1_bytes,
        ),
    ]

    container = MbcContainer(Path("dummy"), segments)
    return container, knowledge


def build_template_container(tmp_path: Path) -> tuple[MbcContainer, KnowledgeBase]:
    knowledge = write_manual(tmp_path)

    segments: list[Segment] = []
    offset = 0

    def add_segment(words: list[InstructionWord]) -> None:
        nonlocal offset
        data = encode_instructions(words)
        descriptor = SegmentDescriptor(len(segments), offset, offset + len(data))
        segments.append(Segment(descriptor, data))
        offset += len(data)

    add_segment(
        [
            build_word(0, 0x04, 0x00, 0x0000),
            build_word(4, 0x00, 0x00, 0x0067),
            build_word(8, 0x00, 0x00, 0x0400),
            build_word(12, 0x00, 0x00, 0x6704),
            build_word(16, 0x00, 0x00, 0x0067),
            build_word(20, 0x00, 0x00, 0x0400),
            build_word(24, 0x00, 0x00, 0x6704),
            build_word(28, 0x00, 0x00, 0x6910),
            build_word(32, 0x30, 0x00, 0x0000),
        ]
    )

    add_segment(
        [
            build_word(0, 0x00, 0x00, 0x0001),
            build_word(4, 0x2B, 0x00, 0x0010),
            build_ascii_word(8, "COND"),
            build_word(12, 0x23, 0x00, 0x0008),
        ]
    )

    add_segment(
        [
            build_ascii_word(0, "HEAD"),
            build_ascii_word(4, "ER00"),
            build_word(8, 0x30, 0x00, 0x0000),
        ]
    )

    add_segment(
        [
            build_word(0, 0x28, 0x00, 0x0020),
            build_word(4, 0x30, 0x00, 0x0001),
        ]
    )

    add_segment(
        [
            build_word(0, 0x27, 0x00, 0x0004),
        ]
    )

    add_segment(
        [
            build_word(0, 0x28, 0x00, 0x0030),
            build_ascii_word(4, "TEXT"),
            build_word(8, 0x30, 0x00, 0x0000),
        ]
    )

    container = MbcContainer(Path("dummy"), segments)
    return container, knowledge


def test_normalizer_builds_ir(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    assert program.metrics.calls == 1
    assert program.metrics.tail_calls == 1
    assert program.metrics.returns >= 1
    assert program.metrics.literals >= 5
    assert program.metrics.literal_chunks == 0
    assert program.metrics.aggregates == 1
    assert program.metrics.testset_branches >= 1
    assert program.metrics.if_branches == 1
    assert program.metrics.loads >= 1
    assert program.metrics.stores >= 1
    assert program.metrics.reduce_replaced == 1

    segment = program.segments[1]
    descriptions = [
        getattr(node, "describe", lambda: "")()
        for block in segment.blocks
        for node in block.nodes
    ]
    assert any("map" in text for text in descriptions)
    assert any(text.startswith("if cond") for text in descriptions)
    assert any(
        text.startswith("testset") or text.startswith("function_prologue")
        for text in descriptions
    )
    assert any(text.startswith("load ") and "[" in text for text in descriptions)
    assert any(text.startswith("store ") and "[" in text for text in descriptions)
    assert any(":0x" in text for text in descriptions)

    renderer = IRTextRenderer()
    text = renderer.render(program)
    assert "normalizer metrics" in text
    assert f"segment {segment.index}" in text

    call_nodes = [
        node
        for seg in program.segments
        for block in seg.blocks
        for node in block.nodes
        if isinstance(
            node,
            (
                IRCall,
                IRCallReturn,
                IRTailcallReturn,
            ),
        )
    ]
    contract_call = None
    for node in call_nodes:
        if getattr(node, "target", None) == 0x0072:
            contract_call = node
            break
    assert contract_call is not None
    assert isinstance(contract_call, IRCall)
    assert contract_call.tail
    assert contract_call.cleanup_mask == 0x2910
    assert contract_call.convention is not None
    assert contract_call.convention.operand == CALL_SHUFFLE_STANDARD
    assert contract_call.predicate is not None
    assert contract_call.predicate.kind == "testset"
    call_block = next(
        block
        for seg in program.segments
        for block in seg.blocks
        if contract_call in block.nodes
    )
    cleanup_nodes = [
        node for node in call_block.nodes if isinstance(node, IRCallCleanup)
    ]
    assert any(
        [step.mnemonic for step in node.steps]
        == ["page_register", "op_5E_29", "op_F0_4B"]
        for node in cleanup_nodes
    )
    cleanup_mnemonics = [step.mnemonic for step in contract_call.cleanup]
    assert cleanup_mnemonics and cleanup_mnemonics[0] == "stack_teardown"
    assert "op_29_10" in cleanup_mnemonics
    assert contract_call.cleanup[0].pops == 1

    if_nodes = [
        node
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRIf)
    ]
    assert if_nodes and all(node.condition.startswith("bool") for node in if_nodes)

    testset_nodes = [
        node
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRTestSetBranch)
    ]
    if testset_nodes:
        assert all(node.expr.startswith("bool") for node in testset_nodes)
    else:
        prologue_nodes = [
            node
            for block in segment.blocks
            for node in block.nodes
            if isinstance(node, IRFunctionPrologue)
        ]
        assert prologue_nodes


def test_tail_helper_wrappers_collapse(tmp_path: Path) -> None:
    knowledge_path = Path(__file__).resolve().parent.parent / "knowledge" / "manual_annotations.json"
    knowledge = KnowledgeBase.load(knowledge_path)

    helper_3d_words = [
        InstructionWord(0, int("3D306910", 16)),
        InstructionWord(4, int("4C000000", 16)),
        InstructionWord(8, int("2910003D", 16)),
        InstructionWord(12, int("30691050", 16)),
        InstructionWord(16, int("00000029", 16)),
        InstructionWord(20, int("10013D30", 16)),
    ]

    helper_72_words = [
        InstructionWord(0, int("306C0104", 16)),
        InstructionWord(4, int("0B00005F", 16)),
        InstructionWord(8, int("0000002C", 16)),
        InstructionWord(12, int("0163D3FE", 16)),
        InstructionWord(16, int("FFFF3032", 16)),
        InstructionWord(20, int("29100072", 16)),
        InstructionWord(24, int("302810FC", 16)),
        InstructionWord(28, int("012C0066", 16)),
        InstructionWord(32, int("27291000", 16)),
        InstructionWord(36, int("6C01040B", 16)),
        InstructionWord(40, int("00005F00", 16)),
        InstructionWord(44, int("00006C01", 16)),
    ]

    segments = []
    offset = 0
    for words in (helper_3d_words, helper_72_words):
        data = encode_instructions(words)
        descriptor = SegmentDescriptor(len(segments), offset, offset + len(data))
        segments.append(Segment(descriptor, data))
        offset += len(data)

    container = MbcContainer(Path("dummy"), segments)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    flattened = [
        node
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
    ]

    assert not any(isinstance(node, IRTailcallReturn) for node in flattened)

    return_nodes = [node for node in flattened if isinstance(node, IRReturn)]
    assert return_nodes

    helper_return = return_nodes[0]
    assert helper_return.cleanup == ()
    assert helper_return.mask is None

    tail_nodes = [node for node in flattened if isinstance(node, IRTailCall)]
    assert tail_nodes
    tail_targets = {node.call.target for node in tail_nodes}
    assert tail_targets == {0x3D30, 0x3032}
    assert any(node.cleanup and node.cleanup[0].mnemonic == "op_4C_00" for node in tail_nodes)

    assert not any(getattr(node, "target", 0) in {0x003D, 0x0072} for node in flattened if hasattr(node, "target"))

    ascii_finalize = [node for node in flattened if isinstance(node, IRAsciiFinalize)]
    assert ascii_finalize and all(node.helper in {0x3D30, 0x7223, 0xF172} for node in ascii_finalize)


def test_normalizer_handles_direct_io_sequences(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x2C03),
        build_word(4, 0x10, 0x64, 0x0800),
        build_word(8, 0x00, 0x00, 0x1000),
        build_ascii_word(12, "mask"),
        build_word(16, 0x74, 0x08, 0x0000),
        build_word(20, 0x39, 0x20, 0x0000),
        build_word(24, 0x00, 0x00, 0x2810),
        build_word(28, 0xC0, 0x00, 0x2E2F),
        build_word(32, 0x3D, 0x30, IO_SLOT),
        build_word(36, 0x30, 0x00, 0x0000),
        build_word(40, 0x5C, 0x08, 0x0000),
        build_word(44, 0x00, 0x00, 0x003F),
        build_word(48, 0x3D, 0x30, IO_SLOT),
        build_word(52, 0x5C, 0x08, 0x0000),
        build_word(56, 0x30, 0x00, 0x0000),
        build_word(60, 0x00, 0x00, 0x2669),
        build_word(64, 0x10, 0xF4, 0x0600),
        build_word(68, 0x30, 0x00, 0x0000),
        build_word(72, 0x00, 0x00, 0x006C),
        build_word(76, 0x11, 0x28, 0x0800),
        build_word(80, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    nodes = [
        node
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
    ]

    io_writes = [node for node in nodes if isinstance(node, IRIOWrite)]
    io_reads = [node for node in nodes if isinstance(node, IRIORead)]

    assert [node.port for node in io_writes] == [IO_PORT_NAME] * len(io_writes)
    assert [node.port for node in io_reads] == [IO_PORT_NAME] * len(io_reads)

    assert {node.mask for node in io_writes} == {0x2C03, 0x2669}
    assert len(io_reads) == 1

    raw_mnemonics = {
        node.mnemonic for node in nodes if isinstance(node, IRRaw)
    }
    assert {"op_10_64", "op_10_F4", "op_3D_30", "op_11_28"}.isdisjoint(raw_mnemonics)


def test_normalizer_collapses_io_facade_helpers(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x01, 0x00, 0x0029),  # stack_teardown
        build_word(4, 0x10, 0x00, 0x3E4B),  # call_helpers
        build_word(8, 0x13, 0x00, 0x3069),  # io write façade
        build_word(12, 0x10, 0xAC, 0x0100),  # call_helpers
        build_word(16, 0x00, 0x00, 0x109C),  # literal
        build_word(20, 0x10, 0xE4, 0x0100),  # io write façade (variant)
        build_word(24, 0x00, 0x29, 0x1003),  # literal mask
        build_word(28, 0xF0, 0x4B, 0x0500),  # helper scaffold
        build_word(32, 0x4A, 0x10, 0x0030),  # helper scaffold
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    cleanup = next(node for node in block.nodes if isinstance(node, IRCallCleanup))
    mnemonics = [step.mnemonic for step in cleanup.steps]
    assert "stack_teardown" in mnemonics
    assert "call_helpers" not in mnemonics

    helper_effects = [node for node in block.nodes if isinstance(node, IRAbiEffect)]
    helper_operands = [effect.operand for effect in helper_effects]
    assert 0x3E4B in helper_operands
    effect = next(effect for effect in helper_effects if effect.operand == 0x3E4B)
    assert effect.kind == "helper.io_service"

    io_writes = [node for node in block.nodes if isinstance(node, IRIOWrite)]
    assert len(io_writes) == 1

    io_write = io_writes[0]
    assert io_write.mask == 0x109C
    assert [step.mnemonic for step in io_write.pre_helpers] == ["call_helpers"]
    assert [step.operand for step in io_write.pre_helpers] == [0x0100]
    assert [step.mnemonic for step in io_write.post_helpers] == ["op_F0_4B", "op_4A_10"]

    raw_mnemonics = {node.mnemonic for node in block.nodes if isinstance(node, IRRaw)}
    assert "op_13_00" not in raw_mnemonics
    assert "op_10_E4" not in raw_mnemonics


def test_normalizer_converts_call_helper_variants(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0xFD, 0x4A, 0x9D01),  # helper scaffold before return
        build_word(4, 0x30, 0x00, 0x0000),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRReturn)
    assert node.cleanup == ()
    helper_effects = [effect for effect in node.abi_effects if effect.kind.startswith("helper.")]
    assert [effect.operand for effect in helper_effects] == [0x9D01]


def test_normalizer_attaches_f0_helper_cleanup(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x05, 0xF0, 0x4B76),  # helper scaffold before IO literal
        build_word(4, 0x00, 0x00, 0x6901),  # literal
        build_word(8, 0x6C, 0x01, 0xEC01),  # page register cleanup
        build_word(12, 0x30, 0x00, 0x0000),  # return
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    return_node = next(node for node in block.nodes if isinstance(node, IRReturn))
    cleanup_mnemonics = [step.mnemonic for step in return_node.cleanup]
    assert cleanup_mnemonics and cleanup_mnemonics[0] in {"page_register", "op_6C_01"}

    helper_effects = [
        node for node in block.nodes if isinstance(node, IRAbiEffect) and node.operand == 0x4B76
    ]
    assert helper_effects and helper_effects[0].kind == "helper.formatting"


def test_normalizer_labels_fanout_cleanup(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x10, 0x32, 0x2C03),
        build_word(4, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRReturn)
    assert [step.mnemonic for step in node.cleanup] == ["fanout"]
    assert [step.operand for step in node.cleanup] == [0x2C03]


def test_normalizer_structural_templates(tmp_path: Path) -> None:
    container, knowledge = build_template_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    descriptions = [
        getattr(node, "describe", lambda: "")()
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
    ]

    assert any("literal_block" in text and "via reduce_pair" in text for text in descriptions)
    assert any(text.startswith("call tail") for text in descriptions)
    assert any(text.startswith("ascii_header[") for text in descriptions)
    assert any(text.startswith("function_prologue") for text in descriptions)
    assert any(text.startswith("call_return") for text in descriptions)
    assert any("call_helper_20" in text for text in descriptions)


def test_normalizer_collapses_ascii_runs_and_literal_hints(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    words = [
        InstructionWord(0, int.from_bytes(b"A\x00B\x00", "big")),
        InstructionWord(4, int.from_bytes(b"\x00C\x00D", "big")),
        build_word(8, 0x00, 0x38, 0x6704),
        build_word(12, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    descriptions = [
        getattr(node, "describe", lambda: "")() for node in block.nodes
    ]

    header = next(
        (node for node in block.nodes if isinstance(node, IRAsciiHeader)), None
    )
    assert header is not None
    assert len(header.chunks) == 1

    pool = {const.name: const for const in program.string_pool}
    assert header.chunks[0] in pool
    constant = pool[header.chunks[0]]
    assert constant.data == b"A\x00B\x00\x00C\x00D"
    assert constant.segments == (constant.data,)
    assert "lit(0x6704)" in descriptions


def test_normalizer_glues_ascii_reduce_chains(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_ascii_word(0, "HEAD"),
        build_ascii_word(4, " ER"),
        build_word(8, 0x04, 0x00, 0x0000),
        build_ascii_word(12, " TEXT"),
        build_word(16, 0x04, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    header = next(node for node in block.nodes if isinstance(node, IRAsciiHeader))
    assert header.chunks and len(header.chunks) == 1
    symbol = header.chunks[0]

    descriptions = [getattr(node, "describe", lambda: "")() for node in block.nodes]
    assert not any("reduce_pair" in text for text in descriptions)

    pool = {const.name: const for const in program.string_pool}
    assert symbol in pool
    constant = pool[symbol]
    assert constant.data == b"HEAD ER  TEX"
    assert constant.segments == (constant.data,)


def test_normalizer_ignores_ascii_bridge_annotations() -> None:
    knowledge = KnowledgeBase({})
    normalizer = IRNormalizer(knowledge)

    words = [
        build_ascii_word(0, "TEXT"),
        build_word(4, 0xAA, 0x00, 0x0000),
        build_ascii_word(8, "MORE"),
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, metrics = normalizer._normalise_block(block)

    assert metrics.raw_remaining == 0
    assert ir_block.annotations and any("op_AA_00" in note for note in ir_block.annotations)
    assert any(
        isinstance(node, (IRAsciiHeader, IRLiteralChunk)) for node in ir_block.nodes
    )


def test_ascii_bridge_with_side_effect_operand_remains_raw() -> None:
    knowledge = KnowledgeBase({})
    normalizer = IRNormalizer(knowledge)

    words = [
        build_ascii_word(0, "TEXT"),
        build_word(4, 0xAA, 0x00, IO_SLOT),
        build_ascii_word(8, "TAIL"),
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, metrics = normalizer._normalise_block(block)

    assert metrics.raw_remaining == 0
    cleanup = next(node for node in ir_block.nodes if isinstance(node, IRCallCleanup))
    assert [step.mnemonic for step in cleanup.steps] == ["op_AA_00"]


def test_stack_neutral_bridge_rejects_edge_positions() -> None:
    normalizer = IRNormalizer(KnowledgeBase({}))
    ascii_node = IRLiteralChunk(data=b"TEXT", source="ascii")
    bridge = make_stack_neutral_instruction(4, "op_AA_00")
    trailing = make_stack_neutral_instruction(8, "op_AB_00")

    items_first = [bridge, ascii_node, trailing]
    assert not normalizer._is_stack_neutral_bridge(bridge, items_first, 0)

    items_last = [ascii_node, trailing, bridge]
    assert not normalizer._is_stack_neutral_bridge(bridge, items_last, 2)


def test_stack_neutral_bridge_respects_control_boundary() -> None:
    normalizer = IRNormalizer(KnowledgeBase({}))
    ascii_node = IRLiteralChunk(data=b"TEXT", source="ascii")
    bridge = make_stack_neutral_instruction(4, "op_AA_00")
    control = make_stack_neutral_instruction(
        8, "op_23_00", kind=InstructionKind.BRANCH
    )
    benign = make_stack_neutral_instruction(12, "op_AB_00")

    items = [ascii_node, bridge, control]
    assert not normalizer._is_stack_neutral_bridge(bridge, items, 1)

    safe_items = [ascii_node, bridge, benign]
    assert normalizer._is_stack_neutral_bridge(bridge, safe_items, 1)


def test_raw_instruction_renders_operand_alias(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [build_word(0, 0x10, 0x08, 0x2C02)]
    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRCallCleanup)
    assert len(node.steps) == 1
    step = node.steps[0]
    assert isinstance(step, IRStackEffect)
    assert step.mnemonic == "fanout"
    assert step.operand_role == "flags"
    assert step.operand_alias == "FANOUT_FLAGS"
    rendered = step.describe()
    assert rendered == "fanout(flags=FANOUT_FLAGS(0x2C02))"

    cleanup_rendered = node.describe()
    assert "fanout(flags=FANOUT_FLAGS(0x2C02))" in cleanup_rendered


def test_condition_mask_from_fanout(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [build_word(0, 0x10, 0x08, RET_MASK)]
    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRConditionMask)
    assert node.source == "fanout"
    assert node.mask == RET_MASK


def test_normalizer_coalesces_io_operations(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, RET_MASK),
        build_word(4, 0x01, 0x3D, 0x3069),
        build_word(8, 0x10, 0x24, IO_SLOT),
        build_word(12, 0x3D, 0x30, IO_SLOT),
        build_word(16, 0x3D, 0x30, IO_SLOT),
        build_word(20, 0x10, 0x38, IO_SLOT),
        build_word(24, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    descriptions = [getattr(node, "describe", lambda: "")() for node in block.nodes]

    assert "io.write(port=ChatOut, mask=0x2910)" in descriptions
    assert "io.read()" in descriptions
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_3D_30" for node in block.nodes
    )


def test_normalizer_handles_extended_io_variants(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x2910),
        build_word(4, 0x10, 0x84, 0x0000),
        build_word(8, 0x2B, 0x00, 0x0000),
        build_word(12, 0x3D, 0x30, IO_SLOT),
        build_word(16, 0x30, 0x00, 0x0000),
        build_word(20, 0x00, 0x00, 0x1234),
        build_word(24, 0x3D, 0x30, IO_SLOT),
        build_word(28, 0x0C, 0x09, 0x0000),
        build_word(32, 0x10, 0x50, 0x0000),
        build_word(36, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    tail_nodes = [
        node
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRTailCall)
    ]
    assert len(tail_nodes) == 1
    tail_node = tail_nodes[0]
    assert [step.mnemonic for step in tail_node.cleanup] == ["op_10_84", "op_3D_30"]

    cleanup_nodes = [
        node
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRCallCleanup)
    ]
    assert cleanup_nodes
    assert cleanup_nodes[0].steps and cleanup_nodes[0].steps[0].mnemonic == "op_3D_30"

    return_nodes = [
        node
        for segment in program.segments
        for block in segment.blocks
        for node in block.nodes
        if isinstance(node, IRReturn)
    ]
    assert return_nodes
    assert [step.mnemonic for step in return_nodes[0].cleanup] == ["op_0C_09", "op_10_50"]
def test_normalizer_handles_mirrored_io_bridges(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x3456),
        build_word(4, 0x64, 0x10, IO_SLOT),
        build_word(8, 0xF0, 0xE8, 0x0000),
        build_word(12, 0x3D, 0x30, IO_SLOT),
        build_word(16, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    cleanup = next(node for node in block.nodes if isinstance(node, IRCallCleanup))
    assert [step.mnemonic for step in cleanup.steps] == ["op_64_10"]

    ret = next(node for node in block.nodes if isinstance(node, IRReturn))
    assert [step.mnemonic for step in ret.cleanup] == ["op_F0_E8", "op_3D_30"]

    raw_mnemonics = {
        node.mnemonic for node in block.nodes if isinstance(node, IRRaw)
    }
    assert "op_64_10" not in raw_mnemonics
    assert "op_F0_E8" not in raw_mnemonics


def test_call_signature_consumes_io_write_helpers(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0xD0, 0x04, 0x0000),
        build_word(4, 0x10, 0xE8, 0x3D30),
        build_word(8, 0xD0, 0x06, 0x0000),
        build_word(12, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    ret = next(node for node in block.nodes if isinstance(node, IRReturn))
    cleanup_mnemonics = [step.mnemonic for step in ret.cleanup]
    assert cleanup_mnemonics == ["op_D0_04", "op_D0_06"]
    helper_effects = [effect for effect in ret.abi_effects if effect.kind.startswith("helper.")]
    assert [effect.operand for effect in helper_effects] == [0x3D30]
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic in {"op_D0_04", "op_D0_06"}
        for node in block.nodes
    )


def test_condition_mask_from_ret_mask_literal(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [build_word(0, 0x29, 0x10, RET_MASK)]
    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRCallCleanup)
    assert [step.mnemonic for step in node.steps] == ["op_29_10"]
    assert node.steps[0].operand == RET_MASK


def test_normalizer_groups_call_helper_cleanup(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x66, 0x15, 0x4B08),  # stack_shuffle
        build_word(4, 0x4A, 0x05, 0x0052),  # op_4A_05 helper
        build_word(8, 0x28, 0x00, 0x1234),  # call_dispatch
        build_word(12, 0x10, 0xE8, 0x0001),  # call_helpers
        build_word(16, 0x32, 0x29, 0x1000),  # op_32_29 helper
        build_word(20, 0x29, 0x10, 0x2910),  # ret mask literal
        build_word(24, 0x01, 0xF0, 0x0000),  # stack_teardown_4
        build_word(28, 0x30, 0x00, 0x0002),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    call_return = next(node for node in block.nodes if isinstance(node, IRCallReturn))
    assert [step.mnemonic for step in call_return.cleanup] == [
        "op_32_29",
        "op_29_10",
        "stack_teardown",
    ]
    assert [step.operand for step in call_return.cleanup[:2]] == [0x1000, 0x2910]
    assert call_return.cleanup[-1].pops == 4
    assert call_return.target == 0x1234
    assert call_return.symbol == "test_helper_1234"
    assert call_return.convention is not None
    assert call_return.convention.operand == 0x4B08
    assert call_return.cleanup_mask == 0x1000
    assert call_return.arity is None

    helper_effects = [effect for effect in call_return.abi_effects if effect.kind.startswith("helper.")]
    assert [effect.operand for effect in helper_effects] == [0x0001]

    assert not any(isinstance(node, IRCallPreparation) for node in block.nodes)
    assert not any(isinstance(node, IRCallCleanup) for node in block.nodes)
    assert not any(
        isinstance(node, IRRaw)
        and node.mnemonic in {"op_4A_05", "op_32_29", "op_29_10", "stack_teardown_4"}
        for node in block.nodes
    )


def test_normalizer_inlines_call_preparation_shuffle(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x66, 0x15, 0x4B08),  # stack_shuffle
        build_word(4, 0x66, 0x15, 0x4B10),  # stack_shuffle
        build_word(8, 0x28, 0x00, 0x0020),  # call_dispatch
        build_word(12, 0x30, 0x00, 0x0000),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    call_return = next(node for node in block.nodes if isinstance(node, IRCallReturn))
    assert call_return.convention is not None
    assert call_return.convention.operand == 0x4B10
    assert not any(isinstance(node, IRCallPreparation) for node in block.nodes)
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "stack_shuffle" for node in block.nodes
    )


def test_normalizer_absorbs_zero_stack_call_wrappers(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x59, 0xFE, 0x1111),
        build_word(4, 0x72, 0x23, 0x2222),
        build_word(8, 0x4B, 0x91, 0x3333),
        build_word(12, 0x28, 0x00, 0x0042),
        build_word(16, 0xE4, 0x01, 0x4444),
        build_word(20, 0x95, 0xFE, 0x5555),
        build_word(24, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    call_return = block.nodes[0]
    assert isinstance(call_return, IRCallReturn)

    cleanup_steps = list(call_return.cleanup)
    assert any(step.mnemonic == "op_E4_01" and step.operand == 0x4444 for step in cleanup_steps)
    assert any(step.mnemonic.startswith("op_95_FE") and step.operand == 0x5555 for step in cleanup_steps)
    assert not any(step.mnemonic in {"op_59_FE", "op_72_23", "op_4B_91"} for step in cleanup_steps)

    assert not any(
        isinstance(node, IRRaw)
        and node.mnemonic in {"op_59_FE", "op_72_23", "op_4B_91", "op_E4_01", "op_95_FE"}
        for node in block.nodes
    )


def test_normalizer_lifts_branch_predicate_from_call(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x28, 0x00, 0x1234),  # call_dispatch
        build_word(4, 0x27, 0x00, 0x0010),  # testset_branch
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    call = next(node for node in block.nodes if isinstance(node, IRCall))
    branch = next(node for node in block.nodes if isinstance(node, IRTestSetBranch))

    assert call.predicate is not None
    assert call.predicate.kind == "testset"
    assert call.predicate.var == branch.var
    assert call.predicate.expr == branch.expr
    assert call.predicate.then_target == branch.then_target
    assert call.predicate.else_target == branch.else_target


def test_normalizer_emits_if_for_testset_mode_33(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_word(4, 0x27, 0x33, 0x0010),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    try:
        predicate = next(node for node in block.nodes if isinstance(node, IRTestSetBranch))
    except StopIteration:
        predicate = next(node for node in block.nodes if isinstance(node, IRFunctionPrologue))

    branch = next(node for node in block.nodes if isinstance(node, IRIf))

    assert branch.then_target == predicate.then_target
    assert branch.else_target == predicate.else_target
    assert branch.condition == predicate.var


def test_normalizer_models_indirect_store_cleanup(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x1000),
        build_word(4, 0x00, 0x00, 0x2000),
        build_word(8, 0x69, 0x10, 0x0000),
        build_word(12, 0x01, 0xF0, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    store = next(node for node in block.nodes if isinstance(node, IRIndirectStore))
    cleanup = next(node for node in block.nodes if isinstance(node, IRCallCleanup))

    assert store.offset == 0
    assert store.value.startswith("word")
    assert store.base.startswith("ptr")
    assert cleanup.pops == 4


def test_normalizer_attaches_epilogue_to_return(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x01, 0xF0, 0x0000),  # stack_teardown_4
        build_word(4, 0x30, 0x00, 0x0001),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    ret = next(node for node in block.nodes if isinstance(node, IRReturn))
    assert ret.cleanup and ret.cleanup[0].mnemonic == "stack_teardown"
    assert ret.cleanup[0].pops == 4
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic.startswith("stack_teardown")
        for node in block.nodes
    )


def test_normalizer_collapses_tailcall_teardown(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x2B, 0x00, 0x0040),  # tailcall_dispatch
        build_word(4, 0x01, 0xF0, 0x0000),  # stack_teardown_4
        build_word(8, 0x30, 0x00, 0x0000),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    tail_call = next(node for node in block.nodes if isinstance(node, IRTailCall))
    assert tail_call.cleanup and tail_call.cleanup[-1].mnemonic == "stack_teardown"
    assert tail_call.cleanup[-1].pops == 4


def test_normalizer_coalesces_call_bridge(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x3C, 0x02, 0x0000),
        build_word(4, 0x28, 0x00, 0x1234),
        build_word(8, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_3C_02" for node in block.nodes
    )
    assert any(
        isinstance(node, (IRCall, IRCallReturn, IRTailcallReturn)) for node in block.nodes
    )


def test_normalizer_extracts_table_dispatch(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x2C, 0x01, 0x6623),
        build_word(4, 0x2C, 0x02, 0x6624),
        build_word(8, 0x28, 0x00, 0x6623),
        build_word(12, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    dispatch = next(node for node in block.nodes if isinstance(node, IRSwitchDispatch))
    keys = {case.key for case in dispatch.cases}
    targets = {case.target for case in dispatch.cases}

    assert dispatch.helper == 0x6623
    assert keys == {0x01, 0x02}
    assert targets == {0x6623, 0x6624}


def test_normalizer_cleans_dispatch_wrappers(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x10, 0x50, 0x1B00),
        build_word(4, 0x64, 0x20, 0x0800),
        build_word(8, 0x2C, 0x04, 0x6634),
        build_word(12, 0x10, 0x8C, 0x0900),
        build_word(16, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert not any(isinstance(node, IRRaw) for node in block.nodes)
    assert isinstance(block.nodes[0], IRCallCleanup)
    assert [step.mnemonic for step in block.nodes[0].steps] == [
        "op_10_50",
        "op_64_20",
    ]
    assert isinstance(block.nodes[1], IRSwitchDispatch)
    return_node = block.nodes[2]
    assert isinstance(return_node, IRReturn)
    assert any(step.mnemonic == "op_10_8C" for step in return_node.cleanup)


def test_normalizer_folds_nested_reduce_pair(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)
    normalizer = IRNormalizer(knowledge)

    reducer_words = [build_word(0, 0x04, 0x00, 0x0000)]
    reducer_data = encode_instructions(reducer_words)
    reducer_segment = Segment(SegmentDescriptor(0, 0, len(reducer_data)), reducer_data)
    raw_blocks = normalizer._parse_segment(reducer_segment)
    assert raw_blocks and raw_blocks[0].instructions
    reduce_instruction = raw_blocks[0].instructions[0]

    items = _ItemList(
        [
            IRBuildArray(elements=("lit(0x0001)", "lit(0x0002)")),
            IRBuildArray(elements=("lit(0x0003)", "lit(0x0004)")),
            reduce_instruction,
        ]
    )
    metrics = NormalizerMetrics()
    normalizer._pass_reduce_pair_constants(items, metrics)

    nodes = items.to_tuple()
    assert len(nodes) == 1
    assert isinstance(nodes[0], IRBuildTuple)
    assert metrics.reduce_replaced == 1

def test_normalizer_collapses_f0_tailcall(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x2B, 0x00, 0x00F0),  # tailcall_dispatch -> helper F0
        build_word(4, 0xF0, 0x4B, CALL_SHUFFLE_STANDARD),  # op_F0_4B shuffle
        build_word(8, 0x29, 0x10, RET_MASK),  # call mask fan-out
        build_word(12, 0x30, 0x00, 0x0000),  # return_values
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    tail_node = next(node for node in block.nodes if isinstance(node, IRTailCall))
    cleanup_mnemonics = [step.mnemonic for step in tail_node.cleanup]
    assert cleanup_mnemonics == ["op_F0_4B", "op_29_10"]
    assert any(step.operand == RET_MASK for step in tail_node.cleanup if step.mnemonic == "op_29_10")
    assert any(step.mnemonic == "op_F0_4B" for step in tail_node.cleanup)
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_F0_4B" for node in block.nodes
    )


@pytest.mark.parametrize(
    "target",
    [
        0x003E,
        0x00ED,
        0x013D,
        0x01EC,
        0x01F1,
        0x032C,
        0x0BF0,
        0x0FF0,
        0x16F0,
    ],
)
def test_normalizer_collapses_extended_tail_helpers(target: int) -> None:
    knowledge = KnowledgeBase.load(Path("knowledge"))

    words = [
        build_word(0, 0x2B, 0x00, target),
        build_word(4, 0x10, 0x0E, 0x0000),
        build_word(8, 0x5E, 0x29, RET_MASK),
        build_word(12, 0x29, 0x10, RET_MASK),
        build_word(16, 0x64, 0x20, 0x0000),
        build_word(20, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    call_node = next(
        node
        for node in block.nodes
        if isinstance(node, (IRTailCall, IRTailcallReturn, IRCallReturn))
    )

    if isinstance(call_node, IRTailCall):
        cleanup = call_node.cleanup
        cleanup_mask = call_node.cleanup_mask
    else:
        cleanup = call_node.cleanup
        cleanup_mask = call_node.cleanup_mask

    mnemonics = {step.mnemonic for step in cleanup}
    assert {"op_10_0E", "op_5E_29", "op_64_20"}.issubset(mnemonics)
    assert cleanup_mask in (None, RET_MASK)

    lingering = {
        node.mnemonic
        for node in block.nodes
        if isinstance(node, IRRaw)
        and node.mnemonic in {"op_10_0E", "op_5E_29", "op_64_20"}
    }
    assert not lingering


def test_normalizer_collapses_inline_tail_dispatch() -> None:
    knowledge = KnowledgeBase.load(Path("knowledge"))

    words = [
        build_word(0, 0x00, 0x52, 0x0000),
        build_word(4, 0x4A, 0x05, 0x0052),
        build_word(8, 0x03, 0x00, 0x1234),
        build_word(12, 0x29, 0x10, RET_MASK),
        build_word(16, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    tail_call = block.nodes[0]
    assert isinstance(tail_call, IRTailCall)
    assert tail_call.call.target == 0x1234
    assert tail_call.call.symbol == "test_helper_1234"
    assert tail_call.call.args == ()
    assert tail_call.returns == ("ret0",)


def test_normalizer_prunes_duplicate_testset_if(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x28, 0x00, 0x1234),  # call_dispatch
        build_word(4, 0x27, 0x00, 0x0010),  # testset_branch
        build_word(8, 0x23, 0x00, 0x0008),  # branch_eq on stored predicate
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert any(isinstance(node, IRTestSetBranch) for node in block.nodes)
    assert not any(isinstance(node, IRIf) for node in block.nodes)


def test_normalizer_handles_io_mask_write(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x3D, 0x30, IO_SLOT),  # io handshake
        build_word(4, 0x10, 0x24, 0x00FF),  # masked write
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    cleanup = next(node for node in block.nodes if isinstance(node, IRCallCleanup))
    assert [step.mnemonic for step in cleanup.steps] == ["op_3D_30"]

    io_write = next(node for node in block.nodes if isinstance(node, IRIOWrite))
    assert io_write.mask == 0x00FF
    assert io_write.port == "ChatOut"


@pytest.mark.parametrize("operand", sorted(IO_SLOT_ALIASES - {IO_SLOT}))
def test_normalizer_handles_io_slot_aliases(tmp_path: Path, operand: int) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x3D, 0x30, operand),  # io handshake via alias
        build_word(4, 0x10, 0x24, operand),  # direct write using the same cell
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    io_writes = [node for node in block.nodes if isinstance(node, IRIOWrite)]
    assert len(io_writes) == 1
    node = io_writes[0]
    assert node.port == IO_PORT_NAME
    raw_suffixes = {
        getattr(candidate, "mnemonic", "")
        for candidate in block.nodes
        if isinstance(candidate, IRRaw)
    }
    assert "op_3D_30" not in raw_suffixes
    assert "op_10_24" not in raw_suffixes


def test_normalizer_collapses_opcode_table_sequences() -> None:
    annotations = {
        f"{opcode:02X}:2A": OpcodeInfo(mnemonic=f"op_{opcode:02X}_2A", stack_delta=0)
        for opcode in range(0x10, 0x1C)
    }
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [build_word(index * 4, opcode, 0x2A, 0x0000) for index, opcode in enumerate(range(0x10, 0x1C))]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, metrics = normalizer._normalise_block(block)

    assert metrics.raw_remaining == 0
    assert len(ir_block.nodes) == 1
    node = ir_block.nodes[0]
    assert isinstance(node, IRTablePatch)
    assert len(node.operations) == len(raw_instructions)
    assert node.annotations and node.annotations[0] == "opcode_table"
    assert any(note == "mode=0x2A" for note in node.annotations)


def test_normalizer_collapses_zero_mode_opcode_tables() -> None:
    annotations = {
        "08:00": OpcodeInfo(mnemonic="op_08_00", stack_delta=0),
    }
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(index * 4, 0x08, 0x00, 0x0008)
        for index in range(6)
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, metrics = normalizer._normalise_block(block)

    assert metrics.raw_remaining == 0
    assert len(ir_block.nodes) == 1
    node = ir_block.nodes[0]
    assert isinstance(node, IRTablePatch)
    assert len(node.operations) == len(raw_instructions)
    assert node.annotations and node.annotations[0] == "opcode_table"
    assert any(note == "mode=0x00" for note in node.annotations)


def test_normalizer_absorbs_zero_mode_affixes_for_opcode_tables() -> None:
    annotations = {
        "08:00": OpcodeInfo(mnemonic="op_08_00", stack_delta=0),
    }
    annotations.update(
        {
            f"{opcode:02X}:2A": OpcodeInfo(
                mnemonic=f"op_{opcode:02X}_2A",
                stack_delta=0,
            )
            for opcode in range(0x10, 0x16)
        }
    )
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(0, 0x08, 0x00, 0x0008),
        *[
            build_word(4 + index * 4, opcode, 0x2A, 0x0000)
            for index, opcode in enumerate(range(0x10, 0x16))
        ],
        build_word(28, 0x08, 0x00, 0x0008),
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, _ = normalizer._normalise_block(block)

    assert len(ir_block.nodes) == 1
    node = ir_block.nodes[0]
    assert isinstance(node, IRTablePatch)
    assert node.operations[0][0] == "op_08_00"
    assert node.operations[-1][0] == "op_08_00"
    assert len(node.operations) == len(raw_instructions)


def test_normalizer_absorbs_separator_affixes_for_opcode_tables() -> None:
    annotations = {
        "04:00": OpcodeInfo(mnemonic="reduce_pair", stack_delta=0),
        "04:02": OpcodeInfo(mnemonic="op_04_02", stack_delta=0),
        "08:03": OpcodeInfo(mnemonic="op_08_03", stack_delta=0),
    }
    annotations.update(
        {
            f"{opcode:02X}:01": OpcodeInfo(
                mnemonic=f"op_{opcode:02X}_01",
                stack_delta=0,
            )
            for opcode in range(0x10, 0x14)
        }
    )
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(0, 0x04, 0x00, 0x0000),
        *[
            build_word(4 + index * 4, opcode, 0x01, 0x0000)
            for index, opcode in enumerate(range(0x10, 0x14))
        ],
        build_word(20, 0x04, 0x02, 0x0000),
        build_word(24, 0x08, 0x03, 0x0000),
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, _ = normalizer._normalise_block(block)

    assert len(ir_block.nodes) == 1
    node = ir_block.nodes[0]
    assert isinstance(node, IRTablePatch)
    assert [op for op, _ in node.operations] == [
        "reduce_pair",
        "op_10_01",
        "op_11_01",
        "op_12_01",
        "op_13_01",
        "op_04_02",
        "op_08_03",
    ]


def test_normalizer_extends_table_patch_with_affixes() -> None:
    annotations = {
        "2C:10": OpcodeInfo(mnemonic="op_2C_10", stack_delta=0),
        "2C:11": OpcodeInfo(mnemonic="op_2C_11", stack_delta=0),
        "04:00": OpcodeInfo(mnemonic="reduce_pair", stack_delta=0),
        "04:02": OpcodeInfo(mnemonic="op_04_02", stack_delta=0),
    }
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(0, 0x2C, 0x10, 0x6600),
        build_word(4, 0x2C, 0x11, 0x6608),
        build_word(8, 0x04, 0x00, 0x0000),
        build_word(12, 0x04, 0x02, 0x0000),
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    items = _ItemList(list(block.instructions))
    normalizer._pass_table_patches(items)

    assert len(items) == 1
    node = items[0]
    assert isinstance(node, IRTablePatch)
    assert [op for op, _ in node.operations] == [
        "op_2C_10",
        "op_2C_11",
        "reduce_pair",
        "op_04_02",
    ]


def test_normalizer_collapses_adaptive_unknown_tables() -> None:
    knowledge = KnowledgeBase({})
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(index * 4, 0x55 + index, 0x6E, 0x8000 + index)
        for index in range(10)
    ]
    profiles = [InstructionProfile.from_word(word, knowledge) for word in words]
    tracker = StackTracker()
    events = tracker.process_sequence(profiles)

    raw_instructions = [
        RawInstruction(
            profile=profile,
            event=event,
            annotations=tuple(),
            ssa_values=tuple(),
            ssa_kinds=tuple(),
        )
        for profile, event in zip(profiles, events)
    ]

    block = RawBlock(index=0, start_offset=0, instructions=tuple(raw_instructions))
    ir_block, _ = normalizer._normalise_block(block)

    assert len(ir_block.nodes) == 1
    node = ir_block.nodes[0]
    assert isinstance(node, IRTablePatch)
    assert len(node.operations) == len(words)
    assert node.annotations[0] == "adaptive_table"
    assert f"mode=0x{words[0].mode:02X}" in node.annotations
    assert any(note == "kind=unknown" for note in node.annotations)


def test_normalizer_builds_table_pipeline_nodes() -> None:
    knowledge = KnowledgeBase({})
    normalizer = IRNormalizer(knowledge)

    prologue = make_stack_neutral_instruction(0, "op_39_4D")
    literal = IRLiteralChunk(data=b"MODE", source="test", symbol="str_0000")
    table = IRTablePatch(
        operations=(("op_82_4D", 0x0000), ("op_88_4D", 0x0000)),
        annotations=("adaptive_table", "mode=0x4D"),
    )
    guard = IRTestSetBranch(
        var="slot0",
        expr="table_patch adaptive_table, mode=0x4D",
        then_target=0x0000,
        else_target=0x1000,
    )

    items = _ItemList([prologue, literal, table, guard])
    normalizer._pass_table_builders(items)

    assert len(items) == 3
    begin, emit, commit = items[0], items[1], items[2]
    assert isinstance(begin, IRTableBuilderBegin)
    assert begin.mode == 0x4D
    assert begin.prologue == (("op_39_4D", 0x0000),)

    assert isinstance(emit, IRTableBuilderEmit)
    assert emit.kind == "adaptive_table"
    assert emit.mode == 0x4D
    assert emit.parameters == ("str(str_0000)",)
    assert emit.operations[0][0] == "op_82_4D"

    assert isinstance(commit, IRTableBuilderCommit)
    assert commit.then_target == 0x0000
    assert commit.else_target == 0x1000

def test_normalizer_emits_page_register_for_single_write(tmp_path: Path) -> None:
    knowledge = KnowledgeBase({"31:30": OpcodeInfo(mnemonic="op_31_30")})
    normalizer = IRNormalizer(knowledge)

    words = [build_word(0, 0x31, 0x30, PAGE_REGISTER)]
    data = encode_instructions(words)
    segment = Segment(SegmentDescriptor(0, 0, len(data)), data)
    container = MbcContainer(tmp_path / "container", [segment])

    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRPageRegister)
    assert node.register == PAGE_REGISTER
    assert node.value is None
    assert node.literal is None


def test_normalizer_tracks_page_register_literal_for_memref(tmp_path: Path) -> None:
    annotations = {
        "00:00": OpcodeInfo(mnemonic="push_literal", category="literal", stack_push=1),
        "31:30": OpcodeInfo(mnemonic="op_31_30", stack_pop=1),
        "69:01": OpcodeInfo(mnemonic="op_69_01", category="indirect_load", stack_push=1),
    }
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(0, 0x00, 0x00, 0x4B10),
        build_word(4, 0x31, 0x30, PAGE_REGISTER),
        build_word(8, 0x69, 0x01, 0xDC05),
    ]
    data = encode_instructions(words)
    segment = Segment(SegmentDescriptor(0, 0, len(data)), data)
    container = MbcContainer(tmp_path / "container_memref", [segment])

    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    page_node = next(node for node in block.nodes if isinstance(node, IRPageRegister))
    assert page_node.register == PAGE_REGISTER
    assert page_node.value == "lit(0x4B10)"
    assert page_node.literal == 0x4B10

    load_node = next(node for node in block.nodes if isinstance(node, IRBankedLoad))
    assert load_node.ref is not None
    assert load_node.register_value == 0x4B10


def test_normalizer_coalesces_indirect_configuration(tmp_path: Path) -> None:
    annotations = {
        "00:00": OpcodeInfo(mnemonic="push_literal", category="literal", stack_push=1),
        "4B:0C": OpcodeInfo(mnemonic="op_4B_0C", stack_delta=0),
        "69:01": OpcodeInfo(mnemonic="op_69_01", category="indirect_load", stack_push=1),
        "D4:06": OpcodeInfo(mnemonic="op_D4_06", stack_delta=0),
        "3D:30": OpcodeInfo(mnemonic="op_3D_30", stack_delta=0),
        "C8:06": OpcodeInfo(mnemonic="op_C8_06", stack_delta=0),
    }
    knowledge = KnowledgeBase(annotations)
    normalizer = IRNormalizer(knowledge)

    words = [
        build_word(0, 0x00, 0x00, 0x4B0C),
        build_word(4, 0x4B, 0x0C, 0x0000),
        build_word(8, 0x69, 0x01, 0xC806),
        build_word(12, 0x00, 0x00, 0x6901),
        build_word(16, 0xD4, 0x06, 0x0000),
        build_word(20, 0x3D, 0x30, 0x6901),
        build_word(24, 0xC8, 0x06, 0x0000),
    ]

    data = encode_instructions(words)
    segment = Segment(SegmentDescriptor(0, 0, len(data)), data)
    container = MbcContainer(tmp_path / "container_indirect", [segment])

    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    load_node = next(node for node in block.nodes if isinstance(node, IRBankedLoad))
    page_nodes = [node for node in block.nodes if isinstance(node, IRPageRegister)]

    assert load_node.ref is not None
    assert load_node.register_value == 0x4B0C
    assert any(page.register == 0x06C8 for page in page_nodes)

    cleanup = next(node for node in block.nodes if isinstance(node, IRCallCleanup))
    assert [step.mnemonic for step in cleanup.steps] == ["op_D4_06", "op_3D_30"]
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic in {"op_D4_06", "op_C8_06"}
        for node in block.nodes
    )

