from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import struct
from typing import Any, Optional


def u24(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)


def s24(data: bytes, offset: int) -> int:
    value = u24(data, offset)
    if value & 0x800000:
        value -= 0x1000000
    return value


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def s32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


@dataclass(frozen=True)
class VMWord:
    """One decoded VM word.

    ``kind`` is a VM-level word kind, not a source-language operation.  Prefix
    bytes are preserved in ``prefixes`` and do not secretly alter call arity.
    """

    index: int
    offset: int
    size: int
    kind: str
    terminal_kind: str
    prefixes: list[int]
    operands: dict[str, Any]
    raw: bytes = field(repr=False, default=b"")
    confidence: float = 1.0
    decoder_rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_hex"] = self.raw.hex(" ")
        payload.pop("raw", None)
        payload["prefixes_hex"] = [f"0x{p:02X}" for p in self.prefixes]
        return payload


def signed_u16(value: int) -> int:
    """Interpret a 16-bit VM displacement as signed little-endian payload."""

    value = int(value) & 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def terminal_atom_offset(word: VMWord) -> int:
    """Return the byte offset of the terminal atom inside a decoded VM word.

    For unprefixed words this is ``word.offset``.  For prefixed words it is the
    first byte after the prefix chain.
    """

    return int(word.offset) + len(word.prefixes)


def branch_operand_base_offset(word: VMWord) -> int:
    """Return the coordinate base used by BR u16 displacements.

    A branch atom is encoded as ``4A/4B/4C/4D <lo> <hi>``.  The displacement is
    relative to ``<lo>``, i.e. the first operand byte, not to the start or end of
    the fused VM word.
    """

    if word.terminal_kind != "BR":
        raise ValueError("branch_operand_base_offset requires a BR word")
    return terminal_atom_offset(word) + 1


def branch_target_offset(word: VMWord) -> int:
    """Resolve a BR target in local function byte/sub-entry coordinates."""

    if word.terminal_kind != "BR":
        raise ValueError("branch_target_offset requires a BR word")
    return branch_operand_base_offset(word) + signed_u16(int(word.operands.get("off", 0) or 0))


def code_ref_target_offset(word: VMWord) -> int:
    """Resolve a CODE_REF target in local function byte/sub-entry coordinates."""

    if word.terminal_kind != "CODE_REF":
        raise ValueError("code_ref_target_offset requires a CODE_REF word")
    return terminal_atom_offset(word) + 1 + int(word.operands.get("rel", 0) or 0)


def call_script_target_offset(word: VMWord) -> int:
    """Resolve a CALL_SCRIPT target in local module/function coordinates."""

    if word.terminal_kind != "CALL_SCRIPT":
        raise ValueError("call_script_target_offset requires a CALL_SCRIPT word")
    return terminal_atom_offset(word) + 3 + int(word.operands.get("rel", 0) or 0)


@dataclass(frozen=True)
class DecodedAtom:
    kind: str
    size: int
    operands: dict[str, Any]
    rule: str


# These sets describe byte shapes, not source-level intent.  They are deliberately
# deliberately small and byte-shape based; no source-signature patterns live here.
SHORT_U16_OPS = {0x01, 0x02, 0x04, 0x0B, 0x20, 0x50, 0x52, 0x53, 0x55, 0x5B, 0x5D, 0x80, 0xCF, 0xD3, 0xD6, 0xD7}
SIGNED_IMM24_OPS = {0x6D}
UNSIGNED_IMM24_OPS = {0x18, 0x1C, 0x34, 0x67, 0x68, 0xA0, 0xD0, 0xE8}
GENERIC_ZERO_IMM24_OPS = {0x03, 0x06, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x28, 0x38, 0x40, 0x44, 0x50, 0x54, 0x5C, 0x98, 0xC8, 0xE8}
PREFIX_SENSITIVE_IMM24U_OPS = {0x3C, 0x3E, 0xED, 0xF0}
END_OPS = {0x23}
NOP_OPS = {0x7C, 0x48}
STRUCTURAL_MARK_OPS = {
    0x00, 0x21, 0x25, 0x26, 0x27, 0x28, 0x2A, 0x2B, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x3A, 0x3C, 0x3D, 0x3E, 0x5E, 0x63, 0x72, 0x74,
    0xE1, 0xE8, 0xEB, 0xEC, 0xED, 0xEF, 0xF0, 0xF1, 0xF3,
}

PREFIX_ALLOWED_ATOMIC: dict[int, set[str]] = {
    0x21: {"BR", "CALL_NATIVE", "CALL_SCRIPT"},
    0x25: {"IMM8", "IMM16", "CALL_NATIVE", "BR"},
    0x26: {"REF", "CALL_NATIVE", "IMM8", "IMM16", "IMM32", "CALL_SCRIPT"},
    0x2A: {"REF", "IMM8", "IMM16", "CALL_NATIVE", "U16", "F32"},
    0x2B: {"REF", "IMM8", "IMM16", "IMM32", "F32", "CALL_NATIVE", "CALL_SCRIPT", "REC61"},
    0x2D: {"REF", "IMM8", "IMM16", "CALL_NATIVE", "CALL_SCRIPT", "REC61", "F32", "U16"},
    0x2F: {"REF", "IMM8", "IMM16", "CALL_NATIVE", "F32", "U16"},
    0x30: {"REF", "REF16", "REC41", "REC61", "REC62", "CALL_NATIVE", "CALL_SCRIPT", "CODE_REF", "IMM8", "IMM16", "IMM24Z", "IMM24S", "IMM24U", "IMM32", "F32", "BR", "U16"},
    0x32: {"REF", "REC41", "CALL_NATIVE", "CALL_SCRIPT", "IMM8"},
    0x3D: {"REF", "REF16", "REC41", "REC61", "REC62", "CALL_NATIVE", "CALL_SCRIPT", "CODE_REF", "IMM8", "IMM16", "IMM24Z", "IMM24S", "IMM24U", "IMM32", "F32", "BR", "U16"},
    0x5E: {"REF", "IMM8", "IMM16", "BR", "CALL_NATIVE", "CALL_SCRIPT"},
    0x72: {"BR", "CODE_REF", "U16"},
    0xF1: {"REF", "CALL_NATIVE", "CALL_SCRIPT", "IMM8", "IMM16", "IMM32", "F32"},
    0xF3: {"REF", "BR", "IMM8"},
    0xF6: {"CALL_NATIVE", "BR", "REC61"},
    0xF7: {"BR", "CALL_NATIVE"},
    0xEC: {"BR"},
    0xE1: {"IMM8", "REF", "BR"},
    0xEB: {"IMM8", "REF", "BR"},
    0xE8: {"BR"},
    0xEF: {"IMM8", "IMM16", "REF", "BR", "REC61"},
    0xF0: {"BR", "IMM24U"},
    0xED: {"BR", "IMM24U"},
    0x3C: {"BR", "IMM24U"},
    0x3E: {"BR", "IMM24U"},
}

PREFIX_ALLOWED_NESTED: dict[int, set[int]] = {
    0x21: {0x3D, 0xE8},
    0x25: {0x3D},
    0x26: {0x72, 0x3D},
    0x2A: {0x2B, 0x2D, 0x2E, 0x2F, 0x3A, 0x3C, 0x3D, 0x3E, 0x60, 0x72, 0xE1, 0xF0},
    0x2B: {0x2A, 0x2B, 0x2E, 0x3A, 0x3C, 0x3D, 0x3E, 0x60, 0x72, 0xE1, 0xF0, 0xF1},
    0x2D: {0x2A, 0x2E, 0x3C, 0x3D, 0x3E, 0x60, 0x72, 0xF0, 0xF1},
    0x2E: {0x2A, 0x2B, 0x2D, 0x2F, 0x3C, 0x3D, 0x3E, 0xE1, 0xEC, 0xED, 0xF0},
    0x2F: {0x2A, 0x2B, 0x2D, 0x3A, 0x3C, 0x3D, 0x3E, 0x60, 0xE1, 0xEC},
    0x30: {0x32, 0xF1, 0x72},
    0x3A: {0x2A, 0x2B, 0x2D, 0x2F, 0x3C},
    0x3C: {0xE8, 0xEB},
    0x3D: {0x30, 0x32, 0xF1, 0xF0, 0xED, 0x3C, 0x3E, 0x72},
    0x3E: {0x72, 0xE8, 0xEB},
    0x5E: {0x21, 0x2A, 0x2B, 0x2D, 0x3D, 0x3E, 0x72, 0xEC, 0xED, 0xEF, 0xF0},
    0x60: {0x3D},
    0x72: {0x30, 0x32, 0x72, 0xF1},
    0xE1: {0xE8, 0xEB},
    0xE8: {0x3D, 0xEB, 0x21, 0xE8},
    0xEC: {0xE8, 0xEB},
    0xED: {0xE8, 0xEB},
    0xEB: {0x72, 0xE8},
    0xEF: {0x30, 0x3D, 0x72},
    0xF0: {0x21, 0x3D, 0x72, 0xE8, 0xEB},
    0xF1: {0x72, 0x30, 0x3D, 0x3E, 0xF0, 0xED, 0x3C, 0xF1},
    0xF3: {0x30, 0x72},
    0xF6: {0x30, 0x3D, 0x72},
    0xF7: {0x30, 0x3D, 0x72},
    0x3B: {0x2D},
}

# Prefix chains discovered by CFG sub-entry validation. These were previously
# mis-tokenized as bare/u24 payloads in the linear pass, while branch targets
# repeatedly entered the suffix prefixes and decoded them as normal VM words.
PREFIX_ALLOWED_NESTED.setdefault(0xEB, set()).update({0x21, 0x3D, 0xEB})
PREFIX_ALLOWED_NESTED.setdefault(0xF1, set()).update({0x2E})


def _aggregate_arity_candidates(raw_arity: int) -> list[int]:
    candidates: list[int] = []
    if 0 <= raw_arity <= 64:
        candidates.append(raw_arity)
    complement = (256 - raw_arity) & 0xFF
    if raw_arity >= 0xE0 and 0 < complement <= 64 and complement not in candidates:
        candidates.append(complement)
    return candidates


def _parse_children(data: bytes, start: int, arity: int) -> list[dict[str, int]]:
    children: list[dict[str, int]] = []
    j = start
    for _ in range(arity):
        children.append({"tag": data[j], "ref": u32(data, j + 1)})
        j += 5
    return children


def _match_aggregate(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    op = data[start]
    if start + 2 < limit and data[start + 1] == 0x4F and op in (0x23, 0x74):
        raw_arity = data[start + 2]
        for arity in _aggregate_arity_candidates(raw_arity):
            body = 3 + 5 * arity
            end = start + body
            if end <= limit:
                return DecodedAtom(
                    "AGG",
                    body,
                    {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 3, arity)},
                    "aggregate.prefixed.raw",
                )

    if op == 0x4F and start + 2 <= limit:
        raw_arity = data[start + 1]
        for arity in _aggregate_arity_candidates(raw_arity):
            body = 2 + 5 * arity
            end = start + body
            if end <= limit:
                payload = {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 2, arity)}
                return DecodedAtom("AGG0", body, payload, "aggregate0.raw")
    return None

def _match_pair_return(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    if start + 2 <= limit and data[start] == 0x72 and data[start + 1] == 0x23:
        return DecodedAtom("RETURN_PAIR", 2, {"bytes": "72 23", "optional_value": True}, "return.pair72_23")
    return None


def _match_atomic(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    if start >= limit:
        return None
    op = data[start]
    # Atomic literals and references.  No long source-signature patterns live here.
    if start + 4 <= limit and data[start:start + 4] == b"\x41\x00\x00\x00":
        return DecodedAtom("IMM24Z", 4, {"op": 0x41, "value": 0, "width": 24, "signed": False}, "literal.41_zero")
    if start + 4 <= limit and data[start:start + 4] == b"\x00\x10\x00\x00":
        return DecodedAtom("IMM24U", 4, {"op": 0x00, "value": 0x10, "width": 24, "signed": False}, "literal.00100000")
    if start + 6 <= limit and op == 0x39 and data[start + 1] == 0x20:
        bits = u32(data, start + 2)
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        return DecodedAtom("F32", 6, {"op": op, "mode": 0x20, "bits": bits, "value": value if math.isfinite(value) else None}, "literal.f32")
    if start + 6 <= limit and op == 0x39 and data[start + 1] == 0x10:
        return DecodedAtom("IMM32", 6, {"op": op, "mode": 0x10, "value": u32(data, start + 2), "width": 32, "signed": False}, "literal.imm32_39")
    if op in PREFIX_SENSITIVE_IMM24U_OPS and start + 4 <= limit:
        next_op = data[start + 1]
        if next_op not in (0x4A, 0x4B, 0x4C, 0x4D) and next_op not in PREFIX_ALLOWED_NESTED.get(op, set()):
            return DecodedAtom("IMM24U", 4, {"op": op, "value": u24(data, start + 1), "width": 24, "signed": False}, "literal.prefix_sensitive_u24")
    if op in SIGNED_IMM24_OPS and start + 4 <= limit:
        return DecodedAtom("IMM24S", 4, {"op": op, "value": s24(data, start + 1), "width": 24, "signed": True}, "literal.s24")
    if op in UNSIGNED_IMM24_OPS and start + 4 <= limit:
        return DecodedAtom("IMM24U", 4, {"op": op, "value": u24(data, start + 1), "width": 24, "signed": False}, "literal.u24")
    if op in GENERIC_ZERO_IMM24_OPS and start + 4 <= limit and data[start + 1] == 0 and data[start + 2] == 0 and data[start + 3] == 0:
        return DecodedAtom("IMM24Z", 4, {"op": op, "value": 0, "width": 24, "signed": False}, "literal.generic_zero24")
    if op in (0x69, 0x65, 0x6C) and start + 6 <= limit:
        return DecodedAtom("REF", 6, {"op": op, "mode": data[start + 1], "ref": u32(data, start + 2)}, "ref32")
    if op == 0x64 and start + 4 <= limit:
        return DecodedAtom("REF16", 4, {"op": op, "mode": data[start + 1], "ref": u16(data, start + 2)}, "ref16")
    if op == 0x41 and start + 7 <= limit:
        return DecodedAtom("REC41", 7, {"ref": u32(data, start + 1), "imm": u16(data, start + 5)}, "record.41")
    if op == 0x61 and start + 16 <= limit:
        return DecodedAtom("REC61", 16, {"mode": data[start + 1], "u16": u16(data, start + 2), "a": u32(data, start + 4), "b": u32(data, start + 8), "c": s32(data, start + 12)}, "record.61")
    if op == 0x62 and start + 8 <= limit:
        return DecodedAtom("REC62", 8, {"mode": data[start + 1], "u16": u16(data, start + 2), "c": s32(data, start + 4)}, "record.62")
    if op == 0x2C and start + 4 <= limit and data[start + 2] == 0x66:
        return DecodedAtom("CALL_NATIVE", 4, {"argc": data[start + 1], "opid": data[start + 3]}, "call.native")
    if op == 0x2C and start + 7 <= limit and data[start + 2] == 0x63:
        return DecodedAtom("CALL_SCRIPT", 7, {"argc": data[start + 1], "rel": s32(data, start + 3)}, "call.script_rel")
    if op == 0x63 and start + 5 <= limit:
        return DecodedAtom("CODE_REF", 5, {"rel": s32(data, start + 1)}, "code_ref")
    if op == 0x29 and start + 3 <= limit and data[start + 1] == 0x10:
        return DecodedAtom("IMM8", 3, {"value": data[start + 2], "width": 8, "signed": False}, "literal.imm8_29")
    if op == 0x28 and start + 4 <= limit and data[start + 1] == 0x10:
        return DecodedAtom("IMM16", 4, {"op": op, "value": u16(data, start + 2), "width": 16, "signed": False}, "literal.imm16_28")
    if op == 0x30 and start + 5 <= limit and data[start + 3] == 0 and data[start + 4] == 0:
        return DecodedAtom("IMM32", 5, {"op": op, "value": u32(data, start + 1), "width": 32, "signed": False}, "literal.imm32_30")
    if op in SHORT_U16_OPS and start + 3 <= limit:
        return DecodedAtom("U16", 3, {"op": op, "value": u16(data, start + 1), "width": 16, "signed": False}, "literal.op_u16")
    if op in (0x4A, 0x4B, 0x4C, 0x4D) and start + 3 <= limit:
        return DecodedAtom("BR", 3, {"op": op, "off": u16(data, start + 1)}, "branch.u16")
    return None


def _match_prefixed(data: bytes, start: int, limit: int, depth: int = 0) -> tuple[DecodedAtom, list[int]] | None:
    if start >= limit or depth > 8:
        return None
    pair = _match_pair_return(data, start, limit)
    if pair is not None:
        return pair, []
    op = data[start]
    if op == 0x3D and start + 3 <= limit and data[start + 1:start + 3] == b"\x72\x23":
        return None
    # Prefix chains are structural.  Prefer a valid nested prefix chain over an
    # immediate interpretation of the nested prefix byte; otherwise constructs
    # like F0 E8 3D 30 REF collapse into a fake u24 literal.
    if op in PREFIX_ALLOWED_NESTED and start + 1 < limit and data[start + 1] in PREFIX_ALLOWED_NESTED[op]:
        nested = _match_prefixed(data, start + 1, limit, depth + 1)
        if nested is not None:
            atom, prefixes = nested
            return DecodedAtom(atom.kind, atom.size + 1, dict(atom.operands), f"prefix.0x{op:02X}->{atom.rule}"), [op, *prefixes]
    if op in PREFIX_ALLOWED_ATOMIC:
        atom = _match_atomic(data, start + 1, limit)
        if atom is not None and atom.kind in PREFIX_ALLOWED_ATOMIC[op]:
            return DecodedAtom(atom.kind, atom.size + 1, dict(atom.operands), f"prefix.0x{op:02X}->{atom.rule}"), [op]
    return None


def _match_structural_single(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    op = data[start]
    if op in END_OPS:
        return DecodedAtom("END", 1, {"op": op, "role": "function_end", "optional_value": True}, "struct.end")
    if op in NOP_OPS:
        return DecodedAtom("NOP", 1, {"op": op, "role": "nop"}, "struct.nop")
    if op in STRUCTURAL_MARK_OPS:
        return DecodedAtom("MARK", 1, {"op": op, "role": "marker"}, "struct.marker")
    return None




BARE_U32_FOLLOWER_KINDS = {
    "REF", "REF16", "REC41", "REC61", "REC62",
    "CALL_NATIVE", "CALL_SCRIPT", "CODE_REF",
    "IMM8", "IMM16", "F32",
    # Some compiler tails encode a final bare literal immediately before the
    # structural 3D + RETURN_PAIR terminator. Keep this deliberately narrow:
    # do not treat arbitrary markers as bare-u32 followers.
    "MARK_RETURN_PAIR",
}


def _bare_u32_follower_kind(data: bytes, start: int, limit: int) -> str | None:
    if start >= limit:
        return None
    if start + 3 <= limit and data[start:start + 3] == b"\x3d\x72\x23":
        return "MARK_RETURN_PAIR"
    aggregate = _match_aggregate(data, start, limit)
    if aggregate is not None:
        return aggregate.kind
    prefixed = _match_prefixed(data, start, limit)
    if prefixed is not None:
        atom, _ = prefixed
        return atom.kind
    atom = _match_atomic(data, start, limit)
    if atom is not None:
        return atom.kind
    return None


def _is_structural_single_entry(data: bytes, offset: int, limit: int) -> bool:
    """Return True when ``offset`` is a valid one-byte structural VM entry."""

    atom = _match_structural_single(data, offset, limit)
    return atom is not None and atom.kind in {"MARK", "NOP", "END"}


def _decodes_as_explicit_transfer_entry(data: bytes, start: int, limit: int) -> bool:
    """Return True when ``start`` begins an explicit non-structural VM entry.

    Untagged scalar payloads are the weakest parse form in this decoder.  A
    tagged/prefixed call, branch, return, or code reference starts a real VM
    entry and therefore takes precedence over a preceding untagged u32 parse.
    """

    prefixed = _match_prefixed(data, start, limit)
    if prefixed is not None:
        atom, _prefixes = prefixed
        return atom.kind not in {"MARK", "NOP", "END", "UNKNOWN"}
    atom = _match_atomic(data, start, limit)
    if atom is not None:
        return atom.kind in {"BR", "CALL_NATIVE", "CALL_SCRIPT", "CODE_REF", "RETURN_PAIR"}
    return False


def _bare_u32_has_shorter_explicit_parse(data: bytes, start: int, limit: int) -> bool:
    """Return True when explicit VM entries outrank an untagged u32 parse.

    ``BARE_U32`` is an untagged scalar operand.  It is accepted only when the
    four bytes do not already form a shorter explicit entry sequence.  The rule
    keeps the precedence order inside the bytecode grammar: structural entries
    plus tagged/prefixed transfer entries are stronger than an untagged scalar
    that is recognized only by looking ahead to its follower.
    """

    if start + 4 > limit:
        return False

    for inner in range(start + 1, min(start + 4, limit)):
        prefix = range(start, inner)
        if all(_is_structural_single_entry(data, j, limit) for j in prefix):
            if _decodes_as_explicit_transfer_entry(data, inner, limit):
                return True

    # A dense non-zero run is not a scalar by default.  If every byte can begin
    # an explicit entry, the explicit bytecode parse wins over the untagged
    # follower-literal parse.  Zero-containing runs remain valid small scalar
    # payloads such as 00 01 00 00, 30 00 00 00, and 5e 5b 04 00.
    if all(data[j] != 0 for j in range(start, start + 4)):
        return all(
            _is_structural_single_entry(data, j, limit)
            or _decodes_as_explicit_transfer_entry(data, j, limit)
            for j in range(start, start + 4)
        )

    return False


def _match_bare_u32(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    """Match VM bare 32-bit literal words.

    Several bytecode forms encode a u32 value without a leading opcode and then
    immediately use it as the base literal for a following record/ref/call word.
    Because it has no tag byte, it is matched after explicit/prefixed entries and
    loses to any shorter explicit-entry parse inside the same four-byte window.
    """
    if start + 4 > limit:
        return None
    if _bare_u32_has_shorter_explicit_parse(data, start, limit):
        return None
    follower = _bare_u32_follower_kind(data, start + 4, limit)
    if follower not in BARE_U32_FOLLOWER_KINDS:
        return None
    value = u32(data, start)
    if follower in {"IMM8", "IMM16", "F32"} and value > 0x10000:
        return None
    return DecodedAtom("BARE_U32", 4, {"value": value, "width": 32, "signed": False, "follower_kind": follower}, "literal.bare_u32")

def _decode_atom_at(data: bytes, offset: int, limit: int) -> tuple[DecodedAtom, list[int]]:
    """Decode exactly one VM word candidate at ``offset``.

    This is the shared primitive for the linear decoder and for byte/sub-entry
    control-flow decoding.  The matching order is intentionally the same as the
    historical linear decoder; only the entry coordinate changes.
    """

    atom = _match_aggregate(data, offset, limit)
    prefixes: list[int] = []
    if atom is None:
        pair = _match_pair_return(data, offset, limit)
        if pair is not None:
            atom = pair
        else:
            pref = _match_prefixed(data, offset, limit)
            if pref is not None:
                atom, prefixes = pref
            else:
                atom = _match_atomic(data, offset, limit)
                if atom is None:
                    atom = _match_bare_u32(data, offset, limit)
                if atom is None:
                    atom = _match_structural_single(data, offset, limit)
    if atom is None:
        atom = DecodedAtom("UNKNOWN", 1, {"byte": data[offset]}, "unknown.byte")
    return atom, prefixes


def decode_word_at(data: bytes, offset: int, *, limit: Optional[int] = None, index: int = 0) -> VMWord:
    """Decode one VM word from an explicit byte/sub-entry offset.

    Branches in this VM may target prefix bytes or terminal atoms inside a
    top-level linear ``VMWord``.  This helper keeps that semantics explicit:
    callers choose the entry byte, and the VM decoder interprets bytes from
    there without snapping to the nearest linear word boundary.
    """

    body_limit = len(data) if limit is None else min(len(data), max(0, int(limit)))
    if offset < 0 or offset >= body_limit:
        raise ValueError(f"decode_word_at offset {offset} outside 0..{body_limit}")
    atom, prefixes = _decode_atom_at(data, offset, body_limit)
    raw = data[offset:offset + atom.size]
    kind = atom.kind if not prefixes else "PFX_" + "_".join(f"{p:02X}" for p in prefixes) + f"_{atom.kind}"
    return VMWord(
        index=index,
        offset=offset,
        size=atom.size,
        kind=kind,
        terminal_kind=atom.kind,
        prefixes=prefixes,
        operands=dict(atom.operands),
        raw=raw,
        confidence=0.0 if atom.kind == "UNKNOWN" else 1.0,
        decoder_rule=atom.rule,
    )


def decode_words(data: bytes, *, limit: Optional[int] = None) -> list[VMWord]:
    """Decode raw function bytes into VM words without source-level signatures.

    The decoder is intentionally literal.  It never emits source-signature words
    and it never treats prefix bytes as hidden call arguments.
    """

    body_limit = len(data) if limit is None else min(len(data), max(0, int(limit)))
    words: list[VMWord] = []
    i = 0
    while i < body_limit:
        word = decode_word_at(data, i, limit=body_limit, index=len(words))
        words.append(word)
        i += max(1, word.size)
    return words


VALUE_WORDS = {"BARE_U32", "IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32", "U16", "REF", "REF16", "REC41", "REC61", "REC62", "AGG", "AGG0", "CODE_REF"}
# Lower operand atoms are bytecode-level expression/argument payload atoms.  They
# are deliberately not persistent CFG stack producers.  AGG/AGG0 are excluded:
# they describe function ABI/prologue layout, not a local operand-frame value.
LOWER_OPERAND_ATOM_WORDS = {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "F32", "U16", "REF", "REF16", "REC41", "REC61", "REC62", "CODE_REF"}
RETURN_WORDS = {"RETURN_PAIR", "END"}
CALL_WORDS = {"CALL_NATIVE", "CALL_SCRIPT"}
BRANCH_WORDS = {"BR"}
CONTROL_BRANCH_OPS = {0x4A, 0x4B, 0x4C, 0x4D}
CONDITIONAL_CONTROL_BRANCH_OPS = {0x4A, 0x4B, 0x4C, 0x4D}


def is_control_branch_word(word: VMWord) -> bool:
    """Return True only for BR atoms that can change control flow.

    A BR-shaped conditional whose encoded target is exactly the next VM entry
    coordinate is not a control edge. It is kept as a decoded VM word for byte
    and stack/predicate accounting, but it must not enter CFG branch resolution
    or region/branch structuring.
    """

    if word.terminal_kind != "BR":
        return False
    op = int(word.operands.get("op", -1) or -1) & 0xFF
    if op not in CONTROL_BRANCH_OPS:
        return False
    try:
        if op in CONDITIONAL_CONTROL_BRANCH_OPS and branch_target_offset(word) == int(word.offset) + int(word.size):
            return False
    except Exception:
        return True
    return True


def word_role(word: VMWord) -> str:
    k = word.terminal_kind
    if k in CALL_WORDS:
        return "call"
    if k in BRANCH_WORDS:
        return "branch" if is_control_branch_word(word) else "predicate_no_transfer"
    if k in RETURN_WORDS:
        return "return"
    if k in {"NOP", "MARK"}:
        return "structural"
    if k == "UNKNOWN":
        return "unknown"
    if k in VALUE_WORDS:
        return "value"
    return "vm_word"


def prefix_chain_signature(word: VMWord) -> str:
    """Return a compact lower-byte prefix signature for provenance audits."""

    return ".".join(f"{int(p):02X}" for p in word.prefixes) if word.prefixes else "-"


def word_shape_signature(word: VMWord) -> str:
    """Return a stable low-level VM word shape signature.

    The signature includes prefix bytes and discriminating terminal operands.  It
    is used as provenance evidence only; it is not a source-level opcode name.
    """

    pref = prefix_chain_signature(word)
    k = word.terminal_kind
    if k == "CALL_NATIVE":
        return f"{pref}|CALL_NATIVE|opid={word.operands.get('opid')}|argc={word.operands.get('argc')}"
    if k == "CALL_SCRIPT":
        return f"{pref}|CALL_SCRIPT|argc={word.operands.get('argc')}"
    if k == "BR":
        return f"{pref}|BR|op={word.operands.get('op')}"
    if k in {"REF", "REF16"}:
        return f"{pref}|{k}|op={word.operands.get('op')}|mode={word.operands.get('mode')}"
    if k in {"REC41", "REC61", "REC62"}:
        return f"{pref}|{k}|mode={word.operands.get('mode')}|u16={word.operands.get('u16')}"
    if k.startswith("IMM") or k in {"U16", "F32", "CODE_REF"}:
        return f"{pref}|{k}|op={word.operands.get('op')}"
    return f"{pref}|{k}"


def is_lower_operand_atom(word: VMWord) -> bool:
    """Return True for bytecode atoms that feed a local operand/effect frame.

    This is intentionally lower than SSA and CFG.  The full corpus shows that
    refs/literals/records/code refs are not persistent stack values by default;
    they are local operands consumed by calls, branches, returns, or lower
    opcode/effect forms.  AGG/AGG0 stay out because they are ABI prologue shapes.
    """

    if word.terminal_kind == "BARE_U32":
        return False
    if word.terminal_kind in {"AGG", "AGG0"}:
        return False
    if word.terminal_kind in LOWER_OPERAND_ATOM_WORDS:
        return True
    return False


def stack_contract(word: VMWord) -> dict[str, Any]:
    """Lower VM stack/effect contract for one decoded word.

    A decoded VMWord is a byte-level word shape, not a proven persistent operand
    stack instruction.  Value-shaped terminal atoms are local operand-frame data
    until a consumer proves their use.  This removes the old false model where
    every REF/IMM/REC/CODE_REF was pushed through CFG joins as a stack fact.
    """

    k = word.terminal_kind
    role = word_role(word)
    prefixes_hex = [f"0x{int(p):02X}" for p in word.prefixes]
    base: dict[str, Any] = {
        "pop": 0,
        "push": 0,
        "role": role,
        "shape_signature": word_shape_signature(word),
    }
    if prefixes_hex:
        base["prefixes_hex"] = prefixes_hex
        base["prefix_count"] = len(prefixes_hex)
        base["prefix_effect_rule"] = "prefix_chain_is_lower_opcode_evidence_not_hidden_call_arity"

    if k == "CALL_NATIVE":
        argc = int(word.operands.get("argc", 0) or 0)
        base.update({
            "encoded_argc": argc,
            "opid": word.operands.get("opid"),
            "result": "deferred_native_return",
            "stack_effect_rule": "native_call_binds_lower_operand_frame_return_arity_deferred",
        })
        return base
    if k == "CALL_SCRIPT":
        argc = int(word.operands.get("argc", 0) or 0)
        base.update({
            "encoded_argc": argc,
            "encoded_rel": word.operands.get("rel"),
            "result": "deferred_script_return",
            "stack_effect_rule": "script_call_binds_lower_operand_frame_return_arity_deferred",
        })
        return base
    if k == "BR":
        op = int(word.operands.get("op", -1) or -1) & 0xFF
        base.update({
            "predicate": "prefix_defined" if word.prefixes else "stack_top_or_flag",
            "control_transfer": role == "branch",
            "stack_effect_rule": "conditional_predicate_transfer_deferred",
        })
        if op in CONDITIONAL_CONTROL_BRANCH_OPS:
            base["predicate_stack_effect"] = "lower_operand_frame_deferred"
            base["candidate_predicate_pop"] = [0, 1, 2]
        return base
    if k in RETURN_WORDS:
        base["stack_effect_rule"] = "terminal_return_is_lower_operand_frame_sink"
        if word.operands.get("optional_value"):
            base["return_payload"] = "optional_deferred"
        return base
    if k in {"NOP", "MARK", "UNKNOWN"}:
        base["stack_effect_rule"] = "structural_word_has_no_persistent_stack_effect"
        if k == "UNKNOWN":
            base["unknown_byte"] = word.operands.get("byte")
        return base
    if k in {"AGG", "AGG0"}:
        base.update({
            "role": "abi_prologue",
            "arity": word.operands.get("arity"),
            "children": word.operands.get("children", []),
            "stack_effect_rule": "aggregate_prologue_declares_abi_not_operand_frame_value",
        })
        return base
    if k == "BARE_U32":
        base.update({
            "role": "auxiliary_literal_payload",
            "auxiliary_literal": True,
            "value": word.operands.get("value"),
            "follower_kind": word.operands.get("follower_kind"),
            "stack_effect_rule": "bare_u32_is_follower_payload_not_stack_value",
        })
        return base
    if is_lower_operand_atom(word):
        atom_role = "literal_ref16_offset" if k == "REF16" and word.operands.get("mode") == 0x20 else "lower_operand_atom"
        base.update({
            "role": atom_role,
            "operand_frame_atom": True,
            "stack_effect_rule": "decoded_value_shape_is_local_operand_frame_atom_not_persistent_stack_value",
        })
        return base
    base["stack_effect_rule"] = "vm_word_has_no_persistent_stack_effect"
    return base
