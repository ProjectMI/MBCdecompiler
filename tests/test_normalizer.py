from mbcdisasm.analyzer.instruction_profile import InstructionProfile
from mbcdisasm.analyzer.normalizer import MacroNormalizer
from mbcdisasm.analyzer.report import build_block
from mbcdisasm.analyzer.stack import StackTracker
from mbcdisasm.analyzer.stats import StatisticsBuilder
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.knowledge import KnowledgeBase, OpcodeInfo


def make_word(opcode: int, mode: int = 0, operand: int = 0, offset: int = 0) -> InstructionWord:
    raw = (opcode << 24) | (mode << 16) | (operand & 0xFFFF)
    return InstructionWord(offset, raw)


def build_knowledge() -> KnowledgeBase:
    annotations = {
        "00:00": OpcodeInfo(
            mnemonic="literal",
            summary="literal",
            category="literal",
            stack_delta=1,
        ),
        "02:00": OpcodeInfo(
            mnemonic="push",
            summary="push",
            category="push",
            stack_delta=1,
        ),
        "04:00": OpcodeInfo(
            mnemonic="reduce",
            summary="reduce",
            category="reduce",
            stack_delta=-1,
        ),
        "26:00": OpcodeInfo(
            mnemonic="test_branch",
            summary="test",
            category="test",
            stack_delta=-1,
        ),
        "29:10": OpcodeInfo(
            mnemonic="tail_dispatch",
            summary="tailcall",
            control_flow="call",
            category="tailcall",
            stack_delta=0,
        ),
        "30:00": OpcodeInfo(
            mnemonic="return",
            summary="return",
            control_flow="return",
            category="return",
            stack_delta=-1,
        ),
        "69:10": OpcodeInfo(
            mnemonic="indirect_access",
            summary="indirect",
            category="indirect",
            stack_delta=0,
        ),
        "69:00": OpcodeInfo(
            mnemonic="indirect_access",
            summary="indirect",
            category="indirect",
            stack_delta=0,
        ),
    }
    return KnowledgeBase(annotations)


def load_profiles(words):
    knowledge = build_knowledge()
    return [InstructionProfile.from_word(word, knowledge) for word in words]


def make_summary(profiles):
    tracker = StackTracker()
    return tracker.process_block(profiles)


def test_tailcall_and_return_collapse_into_macros():
    words = [
        make_word(0x29, 0x10, 0, 0),
        make_word(0x30, 0x00, 0, 4),
    ]
    profiles = load_profiles(words)
    summary = make_summary(profiles)

    normalizer = MacroNormalizer()
    operations = normalizer.normalize(profiles, summary)
    names = {operation.name for operation in operations}

    assert "tail_dispatch" in names
    assert "frame_end" in names


def test_literal_reduce_forms_table_macro():
    words = [
        make_word(0x00, 0x00, 0x0001, 0),
        make_word(0x02, 0x00, 0x0000, 4),
        make_word(0x00, 0x00, 0x0002, 8),
        make_word(0x04, 0x00, 0x0000, 12),
    ]
    profiles = load_profiles(words)
    summary = make_summary(profiles)

    operations = MacroNormalizer().normalize(profiles, summary)
    tables = [operation for operation in operations if operation.name == "table_build"]
    assert tables
    table = tables[0]
    assert any(note == "literals=3" for note in table.notes)


def test_predicate_assignment_macro_detected():
    words = [
        make_word(0x00, 0x00, 0x0010, 0),
        make_word(0x26, 0x00, 0x0000, 4),
    ]
    profiles = load_profiles(words)
    summary = make_summary(profiles)

    operations = MacroNormalizer().normalize(profiles, summary)
    predicate = [operation for operation in operations if operation.name == "predicate_assign"]
    assert predicate
    notes = predicate[0].notes
    assert any(entry.startswith("source=") for entry in notes)


def test_indirect_access_grouped_by_zone():
    words = [
        make_word(0x69, 0x10, 0x0005, 0),
        make_word(0x69, 0x10, 0xED00, 4),
    ]
    profiles = load_profiles(words)
    summary = make_summary(profiles)

    operations = MacroNormalizer().normalize(profiles, summary)
    zones = [entry for op in operations for entry in op.notes if entry.startswith("zone=")]
    assert "zone=frame.locals" in zones
    assert "zone=global.state" in zones


def test_statistics_collects_macro_categories():
    words = [
        make_word(0x00, 0x00, 0x0001, 0),
        make_word(0x00, 0x00, 0x0002, 4),
        make_word(0x04, 0x00, 0x0000, 8),
        make_word(0x29, 0x10, 0x0000, 12),
        make_word(0x30, 0x00, 0x0000, 16),
    ]
    profiles = load_profiles(words)
    summary = make_summary(profiles)
    normalized = MacroNormalizer().normalize(profiles, summary)

    block = build_block(
        profiles,
        summary,
        pattern=None,
        category="literal",
        confidence=0.5,
        normalized=normalized,
    )
    stats = StatisticsBuilder().collect([block])
    assert stats.macro_categories.get("literal", 0) >= 1
    assert stats.macro_categories.get("call", 0) >= 1
    assert stats.macro_categories.get("return", 0) >= 1
