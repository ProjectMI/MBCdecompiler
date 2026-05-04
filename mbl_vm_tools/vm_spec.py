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
    0x21: {"BR", "CALL_SCRIPT"},
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
    0x21: {0x3D},
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
    0xE8: {0x3D, 0xEB},
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
            if end + 2 <= limit and data[end] == 0x31 and data[end + 1] == 0x30:
                return DecodedAtom("AGG", body + 2, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 3, arity), "term2": 0x3130}, "aggregate.prefixed.term3130")
            if end + 2 <= limit and data[end] == 0x72 and data[end + 1] == 0x23:
                return DecodedAtom("AGG", body + 2, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 3, arity), "term2": 0x7223}, "aggregate.prefixed.term7223")
            if end + 1 <= limit and data[end] in (0x72, 0x30):
                return DecodedAtom("AGG", body + 1, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 3, arity), "term": data[end]}, "aggregate.prefixed.term")

    if op == 0x4F and start + 2 < limit:
        raw_arity = data[start + 1]
        for arity in _aggregate_arity_candidates(raw_arity):
            body = 2 + 5 * arity
            end = start + body
            if end > limit:
                continue
            payload = {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, start + 2, arity)}
            if end + 2 <= limit and data[end] == 0x31 and data[end + 1] == 0x30:
                return DecodedAtom("AGG0", body + 2, {**payload, "term2": 0x3130}, "aggregate0.term3130")
            if end + 2 <= limit and data[end] == 0x72 and data[end + 1] == 0x23:
                return DecodedAtom("AGG0", body + 2, {**payload, "term2": 0x7223}, "aggregate0.term7223")
            if end + 1 <= limit and data[end] in (0x31, 0x72, 0x30):
                return DecodedAtom("AGG0", body + 1, {**payload, "term": data[end]}, "aggregate0.term")
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


def _match_bare_u32(data: bytes, start: int, limit: int) -> DecodedAtom | None:
    """Match VM bare 32-bit literal words.

    Several bytecode forms encode a u32 value without a leading opcode and then
    immediately use it as the base literal for a following record/ref/call word.
    This is a generic VM word shape, not a per-function signature.
    """
    if start + 4 > limit:
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
RETURN_WORDS = {"RETURN_PAIR", "END"}
CALL_WORDS = {"CALL_NATIVE", "CALL_SCRIPT"}
BRANCH_WORDS = {"BR"}
CONTROL_BRANCH_OPS = {0x4A, 0x4B, 0x4C, 0x4D}
CONDITIONAL_CONTROL_BRANCH_OPS = {0x4B, 0x4C, 0x4D}


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


def stack_contract(word: VMWord) -> dict[str, Any]:
    """Conservative VM-stack contract for a single word.

    Important invariant: encoded call argc is the only call arity source.  Prefix
    chains are modifiers and are preserved, but never converted into hidden
    function arguments at IR level.
    """

    k = word.terminal_kind
    role = word_role(word)
    if k == "CALL_NATIVE":
        argc = int(word.operands.get("argc", 0) or 0)
        return {"pop": argc, "push": 1, "role": role, "encoded_argc": argc, "result": "unknown_native_return"}
    if k == "CALL_SCRIPT":
        argc = int(word.operands.get("argc", 0) or 0)
        return {"pop": argc, "push": 1, "role": role, "encoded_argc": argc, "result": "unknown_script_return"}
    if k == "BR":
        # Prefix-conditioned BR atoms usually compare/consume stack values, but
        # exact predicate shape belongs to a later analysis pass.  A
        # target==fallthrough conditional is not control flow; keep it as a VM
        # predicate atom rather than a CFG branch.
        return {
            "pop": None,
            "push": 0,
            "role": role,
            "predicate": "prefix_defined" if word.prefixes else "stack_top_or_flag",
            "control_transfer": role == "branch",
        }
    if k in RETURN_WORDS:
        return {"pop": None if word.operands.get("optional_value") else 1, "push": 0, "role": role}
    if k in {"NOP", "MARK", "UNKNOWN"}:
        return {"pop": 0, "push": 0, "role": role}
    if k in {"AGG", "AGG0"}:
        return {"pop": 0, "push": 0, "role": "abi_prologue", "arity": word.operands.get("arity"), "children": word.operands.get("children", [])}
    if k == "REF16" and word.operands.get("mode") == 0x20:
        return {"pop": 0, "push": 1, "role": "literal_ref16_offset"}
    if k in VALUE_WORDS:
        return {"pop": 0, "push": 1, "role": role}
    return {"pop": 0, "push": 0, "role": role}
