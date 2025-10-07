import json
from pathlib import Path

from mbcdisasm import IRNormalizer, KnowledgeBase, MbcContainer
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.ir import IRTextRenderer
from mbcdisasm.ir.model import (
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRAsciiFinalize,
    IRAsciiHeader,
    IRLiteralChunk,
    IRLiteralBlock,
    IRTailCall,
    IRTailcallReturn,
    IRIf,
    IRReturn,
    IRFunctionPrologue,
    IRStackEffect,
    IRTestSetBranch,
    IRIndirectLoad,
    IRIndirectStore,
    IRRaw,
    IRConditionMask,
    IRIOWrite,
    IRBuildTuple,
    IRBuildArray,
    IRSwitchDispatch,
    NormalizerMetrics,
)
from mbcdisasm.ir.normalizer import _ItemList
from mbcdisasm.constants import IO_SLOT, RET_MASK, CALL_SHUFFLE_STANDARD
from mbcdisasm.mbc import Segment
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase


def build_word(offset: int, opcode: int, mode: int, operand: int) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset=offset, raw=raw)


def build_ascii_word(offset: int, text: str) -> InstructionWord:
    data = text.encode("ascii", "replace")[:4].ljust(4, b" ")
    return InstructionWord(offset=offset, raw=int.from_bytes(data, "big"))


def encode_instructions(words: list[InstructionWord]) -> bytes:
    return b"".join(word.raw.to_bytes(4, "big") for word in words)


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
        "call_helpers": {
            "opcodes": ["0x10:0xE8"],
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
    assert program.metrics.literals == 5
    assert program.metrics.literal_chunks == 0
    assert program.metrics.aggregates == 1
    assert program.metrics.testset_branches >= 1
    assert program.metrics.if_branches == 1
    assert program.metrics.loads >= 2
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
    assert [step.mnemonic for step in contract_call.cleanup] == ["stack_teardown"]
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

    io_nodes = [node for node in flattened if isinstance(node, IRIOWrite)]
    assert io_nodes and all(node.port == "io.port_6910" for node in io_nodes)

    assert not any(getattr(node, "target", 0) in {0x003D, 0x0072} for node in flattened if hasattr(node, "target"))

    ascii_finalize = [node for node in flattened if isinstance(node, IRAsciiFinalize)]
    assert ascii_finalize and all(node.helper in {0x3D30, 0x7223, 0xF172} for node in ascii_finalize)


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


def test_normalizer_collapses_literal_reduce_chain_ex(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x67, 0x04, 0x0000),
        build_word(4, 0x00, 0x00, 0x0067),
        build_word(8, 0x00, 0x00, 0x0400),
        build_word(12, 0x04, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    literal_block = next(
        (node for node in block.nodes if isinstance(node, IRLiteralBlock)),
        None,
    )
    assert literal_block is not None
    assert literal_block.triplets == ((0x0067, 0x0400, 0x6704),)

    descriptions = [
        getattr(node, "describe", lambda: "")() for node in block.nodes
    ]
    assert not any("reduce_pair" in text for text in descriptions)


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

    assert "io.write(port=io.port_6910, mask=0x2910)" in descriptions
    assert "io.read()" in descriptions
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_3D_30" for node in block.nodes
    )


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
    assert cleanup_mnemonics == ["op_D0_04", "call_helpers", "op_D0_06"]
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
    assert isinstance(node, IRConditionMask)
    assert node.source == "op_29_10"
    assert node.mask == RET_MASK


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
        "call_helpers",
        "op_32_29",
        "op_29_10",
        "stack_teardown",
    ]
    assert [step.operand for step in call_return.cleanup[:3]] == [0x0001, 0x1000, 0x2910]
    assert call_return.cleanup[-1].pops == 4
    assert call_return.target == 0x1234
    assert call_return.symbol == "test_helper_1234"
    assert call_return.convention is not None
    assert call_return.convention.operand == 0x4B08
    assert call_return.cleanup_mask == 0x2910
    assert call_return.arity is None

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
    assert tail_node.cleanup_mask == RET_MASK
    assert any(step.mnemonic == "op_F0_4B" for step in tail_node.cleanup)
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_F0_4B" for node in block.nodes
    )


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

    assert len(block.nodes) == 1
    node = block.nodes[0]
    assert isinstance(node, IRIOWrite)
    assert node.mask == 0x00FF
    assert node.port == "io.port_6910"
