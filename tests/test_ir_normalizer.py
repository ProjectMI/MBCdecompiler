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
    IRTailcallReturn,
    IRAsciiWrapperCall,
    IRTailcallAscii,
    IRIf,
    IRReturn,
    IRFunctionPrologue,
    IRStackEffect,
    IRTestSetBranch,
    IRRaw,
    IRConditionMask,
    IRLiteralBlock,
)
from mbcdisasm.constants import IO_SLOT, RET_MASK, CALL_SHUFFLE_STANDARD
from mbcdisasm.mbc import Segment
from mbcdisasm.instruction import InstructionWord


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
            "opcodes": ["00:38"],
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
            "opcodes": ["0x27:0x00"],
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
                    "effect": {"mnemonic": "op_6C_01", "operand": "0x6C01"},
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
        }
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
    assert any("offset=" in text for text in descriptions)

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
                IRAsciiWrapperCall,
                IRTailcallAscii,
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
    assert contract_call.cleanup_mask == 0x2910
    assert contract_call.predicate is not None
    assert contract_call.predicate.kind == "testset"
    assert [step.mnemonic for step in contract_call.cleanup] == [
        "op_6C_01",
        "op_5E_29",
        "op_F0_4B",
        "stack_teardown",
    ]

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
    assert any(
        text.startswith("tailcall_ascii") or text.startswith("ascii_wrapper_call tail")
        for text in descriptions
    )
    assert any(text.startswith("ascii_header[") for text in descriptions)
    assert any(text.startswith("function_prologue") for text in descriptions)
    assert any(text.startswith("call_return") for text in descriptions)
    assert any("call_helper_20" in text for text in descriptions)


def test_literal_block_records_stack_delta(tmp_path: Path) -> None:
    container, knowledge = build_template_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    block = program.segments[0].blocks[0]
    literal_block = next(node for node in block.nodes if isinstance(node, IRLiteralBlock))

    triplets = len(literal_block.triplets)
    tail = len(literal_block.tail)
    expected_delta = triplets * 3 + tail - 1

    assert literal_block.stack_delta == expected_delta


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

    descriptions = [getattr(node, "describe", lambda: "")() for node in block.nodes]

    assert "ascii_header[ascii(A\\x00B\\x00), ascii(\\x00C\\x00D)]" in descriptions
    assert "lit(0x6704)" in descriptions

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

    assert "io.write(mask=0x2910)" in descriptions
    assert "io.read()" in descriptions
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "op_3D_30" for node in block.nodes
    )


def test_ascii_wrapper_tailcall_is_bundled(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x00, 0x00, 0x0001),
        build_word(4, 0x00, 0x00, 0x0002),
        build_word(8, 0x6C, 0x01, 0x6C01),
        build_word(12, 0x5E, 0x29, RET_MASK),
        build_word(16, 0xF0, 0x4B, CALL_SHUFFLE_STANDARD),
        build_word(20, 0x2B, 0x00, 0x0072),
        build_ascii_word(24, "TAIL"),
        build_word(28, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    tail_node = next(node for node in block.nodes if isinstance(node, IRTailcallReturn))

    assert tail_node.target == 0x0072
    assert tail_node.shuffle == CALL_SHUFFLE_STANDARD
    assert tail_node.cleanup_mask == RET_MASK
    assert tail_node.ascii_chunks == ("ascii(TAIL)",)


def test_indirect_access_memref_reports_offset(tmp_path: Path) -> None:
    container, knowledge = build_container(tmp_path)
    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)

    block = program.segments[1].blocks[1]
    load_node = next(node for node in block.nodes if node.__class__.__name__ == "IRIndirectLoad")
    store_node = next(node for node in block.nodes if node.__class__.__name__ == "IRIndirectStore")

    assert "offset=" in load_node.ref.describe()
    assert "offset=" in store_node.ref.describe()


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

    call_node = next(
        node for node in block.nodes if isinstance(node, (IRCallReturn, IRTailcallReturn, IRCall))
    )
    assert [step.mnemonic for step in call_node.cleanup] == [
        "call_helpers",
        "op_32_29",
    ]
    assert [step.operand for step in call_node.cleanup] == [0x0001, 0x1000]
    assert call_node.target == 0x1234
    assert call_node.symbol == "test_helper_1234"
    assert call_node.shuffle == 0x4B08
    assert call_node.cleanup_mask == 0x1000
    assert getattr(call_node, "arity", None) is None

    assert not any(isinstance(node, IRCallPreparation) for node in block.nodes)
    assert not any(isinstance(node, IRCallCleanup) for node in block.nodes)
    assert not any(
        isinstance(node, IRRaw)
        and node.mnemonic in {"op_4A_05", "op_32_29", "op_29_10", "stack_teardown_4"}
        for node in block.nodes
    )

    condition_mask = next(node for node in block.nodes if isinstance(node, IRConditionMask))
    assert condition_mask.mask == RET_MASK

    ret_node = next(node for node in block.nodes if isinstance(node, IRReturn))
    assert ret_node.cleanup and ret_node.cleanup[0].mnemonic == "stack_teardown"
    assert ret_node.cleanup[0].pops == 4


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

    call_node = next(
        node
        for node in block.nodes
        if isinstance(node, (IRCallReturn, IRTailcallReturn, IRCall))
    )
    assert call_node.shuffle == 0x4B10
    assert not any(isinstance(node, IRCallPreparation) for node in block.nodes)
    assert not any(
        isinstance(node, IRRaw) and node.mnemonic == "stack_shuffle" for node in block.nodes
    )


def test_call_epilogue_pair_collapses(tmp_path: Path) -> None:
    knowledge = write_manual(tmp_path)

    words = [
        build_word(0, 0x28, 0x00, 0x1234),
        build_word(4, 0x52, 0x05, RET_MASK),
        build_word(8, 0x32, 0x29, RET_MASK),
        build_word(12, 0x30, 0x00, 0x0000),
    ]

    data = encode_instructions(words)
    descriptor = SegmentDescriptor(0, 0, len(data))
    segment = Segment(descriptor, data)
    container = MbcContainer(Path("dummy"), [segment])

    normalizer = IRNormalizer(knowledge)
    program = normalizer.normalise_container(container)
    block = program.segments[0].blocks[0]

    call_node = next(
        node for node in block.nodes if isinstance(node, (IRCallReturn, IRTailcallReturn, IRCall))
    )
    cleanup = call_node.cleanup

    assert cleanup
    assert cleanup[0].mnemonic == "call_epilogue"
    assert cleanup[0].operand == RET_MASK


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

    tail_bundle = next(node for node in block.nodes if isinstance(node, IRTailcallReturn))
    assert tail_bundle.tail
    assert tail_bundle.cleanup and tail_bundle.cleanup[-1].mnemonic == "stack_teardown"
    assert tail_bundle.cleanup[-1].pops == 4
