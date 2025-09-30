"""Synthetic opcode annotations for data heavy segments.

The :mod:`mbc` container backing the old Sphere server is packed with
hand-written scripts.  Large subsystems such as the character controller
(``_char``) mix executable bytecode with sizeable data blobs containing
resource descriptors, animation tables and diagnostic strings.  The legacy
disassembler treated the whole stream uniformly which meant that enormous
portions of the input ended up looking like opaque ``op_AB_CD`` placeholders.

While the pipeline analyser can operate without precise mnemonics, the lack of
categorisation is extremely harmful: the stack tracker cannot reason about
unknown instructions and pattern matchers lose the ability to confirm that a
block really is just a literal train feeding a reducer.  The character module is
particularly pathological – more than half of its 4k instructions are either
inline ASCII chunks or custom literal loaders that never made their way into the
manual annotation database.

This module implements a small but heavily documented *data signature* engine.
It inspects raw instruction words and manufactures :class:`~mbcdisasm.knowledge.
OpcodeInfo` records on the fly when the knowledge base does not provide a
matching entry.  The synthetic annotations are intentionally conservative: they
only kick in for instructions that clearly represent data embedded in the code
stream (ASCII chunks, zero padding, repeated half-words, ``0x00`` opcode literal
loaders).  Regular bytecode is left untouched so that existing metadata and
future manual annotations remain authoritative.

The heuristics were extracted manually from the ``_char`` disassembly.  Several
distinct forms repeat throughout the file:

``inline text``
    Consecutive words whose bytes spell out human readable strings with an
    occasional NUL terminator.  These make up debug logs (``"Created own
    char"``), configuration file paths and animation state names.

``literal paddings``
    The compiler emits a flurry of ``0x00`` opcodes with non-zero modes to push
    literal IDs or to toggle resource markers.  They are functionally identical
    to the documented ``push_literal`` family but the database only listed a
    handful of mode variants.  Treating the remaining ones as opaque instructions
    prevented the stack tracker from understanding otherwise regular literal
    trains.

``pattern sentinels``
    Repeated byte or half-word values such as ``0x7C7C7C7C`` or
    ``0xFFFF_FFFF`` delimit animation tables and other data sections.  They do
    not have an execution side effect yet appear in the instruction stream, so
    providing a friendly mnemonic helps operators map the structure quickly.

Each detector records a small set of attributes – the decoded ASCII text, the
literal value or the reason why a pattern matched.  The pipeline heuristics can
then exploit this information to recognise large literal/data blocks with high
confidence.  Synthetic annotations are marked explicitly via the ``synthetic``
trait so that tooling consuming the listing can decide whether to trust the
result or fall back to manual analysis.

The heuristics implemented below intentionally favour readability over raw
performance.  ``_char`` is a dense artefact and instrumenting the various data
shapes proved invaluable while reverse engineering the VM.  Keeping the logic in
pure Python with descriptive names means the generated listings double as field
notes: analysts can search for ``literal.palindrome`` markers or count how many
``literal_opcode00`` entries surround a configuration record and immediately
understand the structure of the encoded tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..instruction import InstructionWord
from ..knowledge import OpcodeInfo


# ---------------------------------------------------------------------------
# helper dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticSignature:
    """Description of a synthetic opcode annotation."""

    mnemonic: str
    summary: str
    category: str
    stack_push: Optional[int] = None
    stack_pop: Optional[int] = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def to_opcode_info(self) -> OpcodeInfo:
        """Convert the signature into an :class:`OpcodeInfo` instance."""

        return OpcodeInfo(
            mnemonic=self.mnemonic,
            summary=self.summary,
            control_flow="fallthrough",
            category=self.category,
            stack_push=self.stack_push,
            stack_pop=self.stack_pop,
            attributes=dict(self.attributes),
        )


# ---------------------------------------------------------------------------
# classification helpers
# ---------------------------------------------------------------------------


def _word_bytes(word: InstructionWord) -> Tuple[int, int, int, int]:
    """Return the four big-endian bytes composing ``word``."""

    raw = word.raw
    return (
        (raw >> 24) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 8) & 0xFF,
        raw & 0xFF,
    )


def _is_printable(byte: int) -> bool:
    """Return ``True`` if ``byte`` is a printable ASCII character."""

    return 0x20 <= byte <= 0x7E


def _decode_ascii(word: InstructionWord) -> Optional[str]:
    """Return an ASCII string if the bytes inside ``word`` look textual."""

    parts = _word_bytes(word)
    if sum(_is_printable(part) for part in parts) < 3:
        return None
    if not all(part == 0 or _is_printable(part) for part in parts):
        return None
    cleaned = bytes(part for part in parts if part).decode("latin-1", "replace")
    return cleaned


def _hex_word(word: InstructionWord) -> str:
    """Return a hexadecimal representation of ``word`` suitable for summaries."""

    return f"0x{word.raw:08X}"


def _attributes(**values: object) -> Mapping[str, object]:
    """Helper that marks all synthetic annotations with a common trait."""

    payload: Dict[str, object] = {"synthetic": True}
    payload.update(values)
    return payload


# ---------------------------------------------------------------------------
# detector implementations
# ---------------------------------------------------------------------------


Detector = Callable[[InstructionWord], Optional[SyntheticSignature]]


def _ascii_detector(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Detect inline ASCII chunks."""

    text = _decode_ascii(word)
    if text is None:
        return None
    mnemonic = f"ascii_chunk.{text}" if text else "ascii_chunk.empty"
    summary = f"Inline ASCII chunk '{text}'" if text else "Inline ASCII chunk"
    return SyntheticSignature(
        mnemonic=mnemonic,
        summary=summary,
        category="literal.ascii",
        attributes=_attributes(detector="ascii_chunk", ascii_text=text, opcode=word.raw),
    )


def _literal_opcode_zero(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Classify ``opcode == 0x00`` instructions as literal loaders."""

    if word.opcode != 0x00:
        return None
    if word.raw == 0:
        return None

    operand = word.raw & 0xFFFF
    mode = (word.raw >> 16) & 0xFF
    mnemonic = f"literal.op00_{mode:02X}"
    summary = (
        "Неаннотированная форма загрузки литерала (opcode 0x00)."
        " Автоматически распознано при анализе _char."
    )
    return SyntheticSignature(
        mnemonic=mnemonic,
        summary=summary,
        category="literal.synthetic",
        attributes=_attributes(
            detector="literal_opcode00",
            mode=mode,
            operand=operand,
            opcode=word.raw,
        ),
    )


def _repeated_halfword(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Detect ``AABB`` patterns (same half-word repeated twice)."""

    high = (word.raw >> 16) & 0xFFFF
    low = word.raw & 0xFFFF
    if high != low:
        return None
    if high == 0:
        return None

    mnemonic = f"literal.repeat16_{high:04X}"
    summary = f"Повторяющийся 16-битный литерал {high:04X}"
    return SyntheticSignature(
        mnemonic=mnemonic,
        summary=summary,
        category="literal.synthetic",
        attributes=_attributes(detector="repeat16", value=high, opcode=word.raw),
    )


def _repeated_byte(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Detect four-byte runs such as ``0x7C7C7C7C`` or ``0xFFFFFFFF``."""

    parts = _word_bytes(word)
    unique = {part for part in parts}
    if len(unique) != 1:
        return None

    value = next(iter(unique))
    if value == 0:
        return None
    mnemonic = f"literal.repeat8_{value:02X}"
    summary = f"Четырёхбайтовая последовательность {value:02X}"
    return SyntheticSignature(
        mnemonic=mnemonic,
        summary=summary,
        category="literal.synthetic",
        attributes=_attributes(detector="repeat8", value=value, opcode=word.raw),
    )


def _terminator_padding(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Mark well-known padding values that surround terminators."""

    if word.raw in {0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000}:
        mnemonic = f"literal.padding_{word.raw:08X}"
        summary = "Пэддинг вокруг блоков данных"
        return SyntheticSignature(
            mnemonic=mnemonic,
            summary=summary,
            category="literal.synthetic",
            attributes=_attributes(detector="padding", opcode=word.raw),
        )
    return None


def _zero_word(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Classify all-zero words explicitly."""

    if word.raw != 0:
        return None
    return SyntheticSignature(
        mnemonic="literal.zero",
        summary="Нулевое слово",
        category="literal.synthetic",
        attributes=_attributes(detector="zero_word", opcode=word.raw),
    )


def _palindrome_bytes(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Detect ``ABBA`` byte palindromes used for sentinel values."""

    b0, b1, b2, b3 = _word_bytes(word)
    if b0 == b3 and b1 == b2 and b0 != b1:
        mnemonic = f"literal.palindrome_{b0:02X}{b1:02X}"
        summary = "Байт-палиндром"
        return SyntheticSignature(
            mnemonic=mnemonic,
            summary=summary,
            category="literal.synthetic",
            attributes=_attributes(detector="palindrome", bytes=(b0, b1, b2, b3), opcode=word.raw),
        )
    return None


def _sentinel_pattern(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Detect hard-coded sentinel words observed in ``_char`` tables."""

    if word.raw in {0x00EDDEED, 0xDEED00ED}:
        mnemonic = f"literal.sentinel_{word.raw:08X}"
        summary = "Сервисный маркер DEED"
        return SyntheticSignature(
            mnemonic=mnemonic,
            summary=summary,
            category="literal.synthetic",
            attributes=_attributes(detector="sentinel", opcode=word.raw),
        )
    return None


def _numeric_marker(word: InstructionWord) -> Optional[SyntheticSignature]:
    """Recognise small numeric markers used in ``_char`` tables."""

    if word.raw in {0x00002910, 0xFFFF2910, 0x00291000, 0x29100000}:
        mnemonic = f"literal.marker_{word.raw:08X}"
        summary = "Числовой маркер структуры персонажа"
        return SyntheticSignature(
            mnemonic=mnemonic,
            summary=summary,
            category="literal.synthetic",
            attributes=_attributes(detector="char_marker", opcode=word.raw),
        )
    return None


# ---------------------------------------------------------------------------
# public classifier
# ---------------------------------------------------------------------------


class DataClassifier:
    """Classify raw instruction words lacking manual annotations."""

    def __init__(self) -> None:
        self.detectors: Tuple[Detector, ...] = (
            _ascii_detector,
            _repeated_byte,
            _repeated_halfword,
            _palindrome_bytes,
            _sentinel_pattern,
            _literal_opcode_zero,
            _zero_word,
            _terminator_padding,
            _numeric_marker,
        )

    def classify(self, word: InstructionWord) -> Optional[OpcodeInfo]:
        """Return synthetic metadata describing ``word`` or ``None``."""

        for detector in self.detectors:
            signature = detector(word)
            if signature is not None:
                return signature.to_opcode_info()
        return None


# Singleton used across the analyser.  The classifier is stateless and extremely
# cheap to invoke, therefore a module level instance keeps the call sites tidy
# while avoiding repeated allocations.
DATA_CLASSIFIER = DataClassifier()

