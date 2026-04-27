from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import struct
from typing import Any


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def s32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def u24(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16)


def s24(data: bytes, offset: int) -> int:
    value = u24(data, offset)
    if value & 0x800000:
        value -= 0x1000000
    return value


@dataclass
class Token:
    offset: int
    kind: str
    size: int
    payload: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "kind": self.kind,
            "size": self.size,
            "payload": self.payload,
        }


SINGLE_BYTE_OPS = {
    0x00, 0x21, 0x23, 0x27, 0x28, 0x2B, 0x2E, 0x30, 0x31, 0x32, 0x3A, 0x48, 0x63, 0x72, 0x7C,
    0x3D, 0x2A, 0x2D, 0x2F, 0x25,
    0xF0, 0xF1, 0xF3, 0xE1, 0xE8, 0xEC, 0xED, 0xEF, 0x5E, 0xEB, 0x3C, 0x3E, 0x26,
}
SHORT_U16_OPS = {0x01, 0x02, 0x04, 0x0B, 0x20, 0x50, 0x52, 0x53, 0x55, 0x5B, 0x5D, 0x80, 0xCF, 0xD3, 0xD6, 0xD7}
SIGNED_IMM24_OPS = {0x6D}
UNSIGNED_IMM24_OPS = {0x18, 0x1C, 0x34, 0x67, 0x68, 0xA0, 0xD0, 0xE8}
GENERIC_ZERO_IMM24_OPS = {0x03, 0x06, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x28, 0x38, 0x40, 0x44, 0x50, 0x54, 0x5C, 0x98, 0xC8, 0xE8}

# Single-byte bytecode words that are real structural VM bytes rather than
# unknown opcodes.  They are intentionally split by role so IR never has to
# treat them as opaque generic OP nodes.
END_OPS = {0x23}
NOP_OPS = {0x7C, 0x48}
MARKER_OPS = SINGLE_BYTE_OPS - END_OPS - NOP_OPS
PREFIX_SENSITIVE_IMM24U_OPS = {0x3C, 0x3E, 0xED, 0xF0}


def _single_byte_structural_token(op: int) -> tuple[str, dict[str, Any]]:
    if op in END_OPS:
        return "END", {"op": op, "role": "function_end"}
    if op in NOP_OPS:
        return "NOP", {"op": op, "role": "nop"}
    return "MARK", {"op": op, "role": "marker"}


def _is_printable_ascii(b: int) -> bool:
    # Conservative definition: common printable set + whitespace.
    return b in (9, 10, 13) or (0x20 <= b <= 0x7E)


def _ascii_run(data: bytes, start: int, limit: int) -> int:
    """Return end offset of a printable ASCII run starting at `start`."""
    i = start
    while i < limit and _is_printable_ascii(data[i]):
        i += 1
    return i


def _looks_like_text(run: bytes) -> bool:
    """Heuristic: decide whether a printable run is more likely text than bytecode."""
    if len(run) < 10:
        return False
    # Require at least some alnum content and some punctuation typical for format strings / tags.
    has_alnum = any((48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) for b in run)
    punct = sum(run.count(ch) for ch in (b',', b'"', b'%', b'<', b'>', b'=', b'(', b')', b'_', b'/', b':'))
    # Avoid classifying long runs of a single repeated character as "text".
    if len(set(run)) <= 2:
        return False
    return has_alnum and punct >= 2




def _looks_like_float32_word(word: int) -> bool:
    if word in (0, 0xFFFFFFFF, 0xDEEDDEED):
        return False
    value = struct.unpack('<f', struct.pack('<I', word))[0]
    if not math.isfinite(value):
        return False
    mag = abs(value)
    return 0.001 <= mag <= 100000.0


def _blob_word_score(word: int) -> int:
    raw = struct.pack('<I', word)
    zero_bytes = raw.count(0)
    if word == 0 or word == 0xDEEDDEED:
        return 3
    if zero_bytes >= 3:
        return 2
    if _looks_like_float32_word(word):
        return 2
    return 0


def _scan_dword_blob(data: bytes, start: int, limit: int) -> tuple[int, dict[str, Any] | None]:
    if start + 32 > limit:
        return 0, None

    j = start
    words: list[int] = []
    zeroish = 0
    floatish = 0
    sentinels = 0

    while j + 4 <= limit:
        word = u32(data, j)
        score = _blob_word_score(word)
        if score <= 0:
            break
        words.append(word)
        raw = struct.pack('<I', word)
        if raw.count(0) >= 3 or word == 0:
            zeroish += 1
        if word == 0xDEEDDEED:
            sentinels += 1
        if _looks_like_float32_word(word):
            floatish += 1
        j += 4

    count = len(words)
    if count < 8:
        return 0, None
    if zeroish < max(4, int(count * 0.5)) and (floatish + sentinels) < 2:
        return 0, None

    return j - start, {
        'word_count': count,
        'zeroish_words': zeroish,
        'floatish_words': floatish,
        'sentinel_words': sentinels,
    }

def _parse_children(data: bytes, start: int, arity: int) -> list[dict[str, int]]:
    children = []
    j = start
    for _ in range(arity):
        children.append({"tag": data[j], "ref": u32(data, j + 1)})
        j += 5
    return children


def _aggregate_arity_candidates(raw_arity: int) -> list[int]:
    candidates: list[int] = []
    if 0 <= raw_arity <= 64:
        candidates.append(raw_arity)
    complement = (256 - raw_arity) & 0xFF
    if raw_arity >= 0xE0 and 0 < complement <= 64 and complement not in candidates:
        candidates.append(complement)
    return candidates




def _match_atomic_semantic(data: bytes, start: int, limit: int) -> tuple[str, int, dict[str, Any]] | None:
    if start >= limit:
        return None
    op = data[start]

    if start + 4 <= limit and data[start:start + 4] == b"\x41\x00\x00\x00":
        return "IMM24Z", 4, {"op": 0x41, "imm": 0}
    if start + 4 <= limit and data[start:start + 4] == b"\x00\x10\x00\x00":
        return "IMM24U", 4, {"op": 0x00, "imm": 0x10}
    if start + 18 <= limit and data[start:start + 18] == b"\xff\x23\x4f\x00\x31\x30\x32\x6c\x01\x08\x00\x00\x00\x14\x00\x00\x00\x72":
        return "SIG_GETCASTLENUM_HEAD", 18, {}
    if op == 0x32 and start + 6 <= limit and data[start + 1] == 0x29 and data[start + 2] == 0x10:
        if data[start + 4:start + 6] == b"\x72\x23":
            return "SIG_RETURN_TAIL", 6, {"imm": data[start + 3], "has_f1_prefix": False, "tail_form": "pair72_23"}
        if start + 7 <= limit and data[start + 4] == 0xF1 and data[start + 5:start + 7] == b"\x72\x23":
            return "SIG_RETURN_TAIL", 7, {"imm": data[start + 3], "has_f1_prefix": True, "tail_form": "f1_pair72_23"}
    if start + 10 <= limit and data[start + 5] == 0x00 and data[start + 6:start + 10] == b"\x2c\x00\x66\x27":
        return "SIG_U32_U8_CALL66_TAIL", 10, {"value": u32(data, start), "arg": data[start + 4]}
    if start + 6 <= limit and op == 0x39 and data[start + 1] == 0x20:
        bits = u32(data, start + 2)
        value = struct.unpack('<f', struct.pack('<I', bits))[0]
        return "F32", 6, {"op": 0x39, "mode": 0x20, "bits": bits, "value": value if math.isfinite(value) else None}
    if start + 6 <= limit and op == 0x39 and data[start + 1] == 0x10:
        return "IMM32", 6, {"op": 0x39, "mode": 0x10, "value": u32(data, start + 2)}
    # Prefix-sensitive imm24 forms: these bytes can be prefixes before
    # branches, so direct tokenization only accepts the unambiguous immediate
    # shape here. Nested prefix matching can still use the same atomic form.
    if op in PREFIX_SENSITIVE_IMM24U_OPS and start + 4 <= limit:
        next_op = data[start + 1]
        # These opcodes double as prefix heads.  Do not greedily eat a nested
        # branch/prefix chain as a plain imm24.  Direct imm24 is only accepted
        # when the following byte cannot start a legal nested form.
        if next_op not in (0x4A, 0x4B, 0x4C, 0x4D) and next_op not in _PREFIX_ALLOWED_NESTED.get(op, set()):
            return "IMM24U", 4, {"op": op, "imm": u24(data, start + 1)}
    if op in SIGNED_IMM24_OPS and start + 4 <= limit:
        return "IMM24S", 4, {"op": op, "imm": s24(data, start + 1)}
    if op in UNSIGNED_IMM24_OPS and start + 4 <= limit:
        return "IMM24U", 4, {"op": op, "imm": u24(data, start + 1)}
    if op in GENERIC_ZERO_IMM24_OPS and start + 4 <= limit and data[start + 1] == 0 and data[start + 2] == 0 and data[start + 3] == 0:
        return "IMM24Z", 4, {"op": op, "imm": 0}
    if op == 0x08 and start + 2 <= limit and data[start + 1] == 0:
        return "MARK", 2, {"op": op, "arg": 0, "role": "marker"}
    if op in (0x69, 0x65, 0x6C) and start + 6 <= limit:
        return "REF", 6, {"op": op, "mode": data[start + 1], "ref": u32(data, start + 2)}
    if op == 0x64 and start + 4 <= limit:
        return "REF16", 4, {"op": op, "mode": data[start + 1], "ref": struct.unpack_from('<H', data, start + 2)[0]}
    if op == 0x41 and start + 7 <= limit:
        return "REC41", 7, {"ref": u32(data, start + 1), "imm": struct.unpack_from('<H', data, start + 5)[0]}
    if op == 0x61 and start + 16 <= limit:
        return "REC61", 16, {"mode": data[start + 1], "u16": struct.unpack_from("<H", data, start + 2)[0], "a": u32(data, start + 4), "b": u32(data, start + 8), "c": s32(data, start + 12)}
    if op == 0x62 and start + 8 <= limit:
        return "REC62", 8, {"mode": data[start + 1], "u16": struct.unpack_from("<H", data, start + 2)[0], "c": s32(data, start + 4)}
    if op == 0x2C and start + 4 <= limit and data[start + 2] == 0x66:
        return "CALL66", 4, {"argc": data[start + 1], "opid": data[start + 3]}
    if op == 0x2C and start + 7 <= limit and data[start + 2] == 0x63:
        return "CALL63A", 7, {"argc": data[start + 1], "rel": s32(data, start + 3)}
    if op == 0x63 and start + 5 <= limit:
        return "CALL63B", 5, {"rel": s32(data, start + 1)}
    if op == 0x29 and start + 3 <= limit and data[start + 1] == 0x10:
        return "IMM", 3, {"value": data[start + 2]}
    if op == 0x28 and start + 4 <= limit and data[start + 1] == 0x10:
        return "IMM16", 4, {"op": op, "value": struct.unpack_from('<H', data, start + 2)[0]}
    if op in SHORT_U16_OPS and start + 3 <= limit:
        return "OPU16", 3, {"op": op, "value": struct.unpack_from('<H', data, start + 1)[0]}
    if op in (0x4A, 0x4B, 0x4C, 0x4D) and start + 3 <= limit:
        return "BR", 3, {"op": op, "off": struct.unpack_from("<H", data, start + 1)[0]}
    if op == 0x30 and start + 5 <= limit and data[start + 3] == 0 and data[start + 4] == 0:
        return "IMM32", 5, {"op": op, "imm": u32(data, start + 1)}
    return None


_PREFIX_ALLOWED_ATOMIC = {
    0x21: {"BR", "CALL63A"},
    0x25: {"IMM", "IMM16", "CALL66", "BR"},
    0x2A: {"REF", "IMM", "IMM16", "CALL66", "OPU16", "F32"},
    0x2B: {"REF", "IMM", "IMM16", "IMM32", "F32", "CALL66", "CALL63A", "REC61"},
    0x2D: {"REF", "IMM", "IMM16", "CALL66", "CALL63A", "REC61", "F32", "OPU16"},
    0x2F: {"REF", "IMM", "IMM16", "CALL66", "F32", "OPU16"},
    0x30: {"REF", "REF16", "REC41", "REC61", "REC62", "CALL66", "CALL63A", "CALL63B", "IMM", "IMM16", "IMM24Z", "IMM24S", "IMM24U", "IMM32", "F32", "BR", "OPU16", "SIG_U32_U8_CALL66_TAIL", "SIG_GETCASTLENUM_HEAD"},
    0x32: {"REF", "REC41", "CALL66", "CALL63A", "IMM"},
    0xF1: {"REF", "CALL66", "CALL63A", "IMM", "IMM16", "IMM32", "F32"},
    0x3D: {"REF", "REF16", "REC41", "REC61", "REC62", "CALL66", "CALL63A", "CALL63B", "IMM", "IMM16", "IMM24Z", "IMM24S", "IMM24U", "IMM32", "F32", "BR", "OPU16"},
    0x5E: {"REF", "IMM", "IMM16", "BR", "CALL66", "CALL63A"},
    0x72: {"BR", "CALL63B", "OPU16"},
    0xF3: {"REF", "BR", "IMM"},
    0xF6: {"CALL66", "BR", "REC61"},
    0xF7: {"BR", "CALL66"},
    0x26: {"REF", "CALL66", "IMM", "IMM16", "IMM32", "CALL63A"},
    0xEC: {"BR"},
    0xE1: {"IMM", "REF", "BR"},
    0xEB: {"IMM", "REF", "BR"},
    0xE8: {"BR"},
    0xEF: {"IMM", "IMM16", "REF", "BR", "REC61"},
    0xF0: {"BR", "IMM24U"},
    0xED: {"BR", "IMM24U"},
    0x3C: {"BR", "IMM24U"},
    0x3E: {"BR", "IMM24U"},
}
_PREFIX_ALLOWED_NESTED = {
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


def _match_prefixed_semantic(data: bytes, start: int, limit: int, depth: int = 0) -> tuple[str, int, dict[str, Any]] | None:
    if start >= limit or depth > 8:
        return None
    op = data[start]

    if op == 0x72 and start + 2 <= limit and data[start + 1] == 0x23:
        return "PAIR72_23", 2, {"bytes": "72 23"}

    if op in _PREFIX_ALLOWED_ATOMIC:
        nested = _match_atomic_semantic(data, start + 1, limit)
        if nested is not None:
            nested_kind, nested_size, nested_payload = nested
            if nested_kind in _PREFIX_ALLOWED_ATOMIC[op]:
                return f"PFX_{op:02X}_{nested_kind}", 1 + nested_size, {"prefix_op": op, "nested_kind": nested_kind, "nested": nested_payload}

    if op in _PREFIX_ALLOWED_NESTED and start + 1 < limit and data[start + 1] in _PREFIX_ALLOWED_NESTED[op]:
        nested = _match_prefixed_semantic(data, start + 1, limit, depth + 1)
        if nested is not None:
            nested_kind, nested_size, nested_payload = nested
            flat_nested_kind = nested_kind[4:] if nested_kind.startswith("PFX_") else nested_kind
            return f"PFX_{op:02X}_{flat_nested_kind}", 1 + nested_size, {"prefix_op": op, "nested_kind": nested_kind, "nested": nested_payload}

    return None


_TAIL_REPAIR_DIRECT_START_OPS = {0x30, 0x63}
_TAIL_REPAIR_PREFIX_START_OPS = set(_PREFIX_ALLOWED_ATOMIC) | set(_PREFIX_ALLOWED_NESTED)
_TAIL_REPAIR_TERMINALS = {
    "REF", "REF16", "REC41", "REC61", "REC62",
    "CALL66", "CALL63A", "CALL63B",
    "IMM", "IMM16", "IMM24Z", "IMM24S", "IMM24U", "IMM32", "F32", "BR", "OPU16",
    "SIG_U32_U8_CALL66_TAIL", "SIG_GETCASTLENUM_HEAD",
}


def _terminal_kind_from_payload(kind: str, payload: dict[str, Any]) -> str:
    terminal_kind = kind
    terminal_payload: Any = payload
    while (
        isinstance(terminal_payload, dict)
        and "prefix_op" in terminal_payload
        and "nested_kind" in terminal_payload
        and "nested" in terminal_payload
    ):
        terminal_kind = str(terminal_payload["nested_kind"])
        terminal_payload = terminal_payload["nested"]
    return terminal_kind


def _tail_repair_match(data: bytes, start: int, body_limit: int, stream_limit: int) -> tuple[str, int, dict[str, Any]] | None:
    """
    Complete one token that starts inside the logical body but runs a few bytes
    past the table span. This is deliberately narrower than simply tokenizing
    against `len(data)`: only atomic/prefix bytecode forms are eligible. Larger
    wrappers/aggregates/ASCII/pad detection stay bounded by `body_limit`, so a
    trailing 0x23 cannot accidentally consume the beginning of the next body as
    an aggregate.
    """
    if stream_limit <= body_limit or start >= body_limit:
        return None

    op = data[start]
    candidates: list[tuple[str, int, dict[str, Any]]] = []

    if op in _TAIL_REPAIR_DIRECT_START_OPS:
        direct = _match_atomic_semantic(data, start, stream_limit)
        if direct is not None:
            candidates.append(direct)

    if op in _TAIL_REPAIR_PREFIX_START_OPS:
        prefixed = _match_prefixed_semantic(data, start, stream_limit)
        if prefixed is not None:
            candidates.append(prefixed)

    best: tuple[str, int, dict[str, Any]] | None = None
    for kind, size, payload in candidates:
        if start + size <= body_limit:
            continue
        if size <= 1 or start + size > stream_limit:
            continue
        terminal = _terminal_kind_from_payload(kind, payload)
        if terminal not in _TAIL_REPAIR_TERMINALS and not kind.startswith("PFX_"):
            continue
        bounded_atomic = _match_atomic_semantic(data, start, body_limit)
        bounded_prefixed = _match_prefixed_semantic(data, start, body_limit) if op in _TAIL_REPAIR_PREFIX_START_OPS else None
        if bounded_atomic is not None and bounded_atomic[1] >= size:
            continue
        if bounded_prefixed is not None and bounded_prefixed[1] >= size:
            continue
        if best is None or size > best[1]:
            best = (kind, size, payload)
    return best

def tokenize_stream(data: bytes, limit: int | None = None) -> list[Token]:
    # `limit` is the logical body boundary.  The byte buffer may contain a
    # small lookahead tail so an instruction that starts inside the body but
    # crosses the table span can still be completed.  No token is started at or
    # beyond `body_size`.
    body_size = len(data) if limit is None else min(len(data), limit)
    size = len(data)
    out: list[Token] = []
    i = 0

    while i < body_size:
        op = data[i]

        repaired = _tail_repair_match(data, i, body_size, size)
        if repaired is not None:
            kind, repair_size, payload = repaired
            payload = {**payload, "tail_repair": True, "logical_limit": body_size}
            out.append(Token(i, kind, repair_size, payload))
            i += repair_size
            continue

        if i + 32 <= body_size and data[i:i + 4] == b"\x4f\x00\x31\x30" and data[i + 4] in (0x69, 0x6C) and data[i + 10] == 0x6C and data[i + 11] == 0x01 and data[i + 16:i + 20] == b"\x04\x00\x00\x00" and data[i + 20:i + 23] == b"\x3d\x30\x32" and data[i + 23] in (0x69, 0x6C) and data[i + 29] == 0x5E and data[i + 30:i + 32] == b"\x72\x23":
            out.append(Token(i, "SIG_GETWEAR_WRAPPER", 32, {
                "ref_a_op": data[i + 4],
                "ref_a_mode": data[i + 5],
                "ref_a": u32(data, i + 6),
                "ref_b": u32(data, i + 12),
                "const": u32(data, i + 16),
                "ref_c_op": data[i + 23],
                "ref_c_mode": data[i + 24],
                "ref_c": u32(data, i + 25),
                "tail_op": 0x5E,
            }))
            i += 32
            continue

        if i + 19 <= body_size and data[i:i + 2] == b"\x4f\x01" and data[i + 7:i + 10] == b"\x31\x30\x32" and data[i + 10] in (0x69, 0x6C) and data[i + 16] == 0x26 and data[i + 17:i + 19] == b"\x72\x23":
            out.append(Token(i, "SIG_GETP_WRAPPER", 19, {
                "child_tag": data[i + 2],
                "child_ref": u32(data, i + 3),
                "ref_op": data[i + 10],
                "ref_mode": data[i + 11],
                "ref": u32(data, i + 12),
                "tail_op": 0x26,
            }))
            i += 19
            continue

        if i + 14 <= body_size and data[i:i + 4] == b"\x4f\x00\x31\x30" and data[i + 4] == 0x32 and data[i + 5] in (0x69, 0x6C) and data[i + 11] == 0x21 and data[i + 12:i + 14] == b"\x72\x23":
            out.append(Token(i, "SIG_INPUTDONE_SHORT", 14, {
                "ref_op": data[i + 5],
                "ref_mode": data[i + 6],
                "ref": u32(data, i + 7),
                "tail_op": 0x21,
            }))
            i += 14
            continue

        # aggregate families: 23 4f N, 74 4f N
        # children=(tag + ref32)
        # close is usually 31 30, but stable wrapper families also use the
        # paired trailer 72 23. Treating only 72 as the terminator leaves a
        # spurious trailing 23 and systematically under-scores compact wrappers.
        if i + 2 < body_size and data[i + 1] == 0x4F and op in (0x23, 0x74):
            raw_arity = data[i + 2]
            matched_agg = False
            for arity in _aggregate_arity_candidates(raw_arity):
                body = 3 + 5 * arity
                end = i + body
                # normal two-byte close
                if end + 2 <= body_size and data[end] == 0x31 and data[end + 1] == 0x30:
                    out.append(Token(i, "AGG", body + 2, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, i + 3, arity), "term2": 0x3130}))
                    i += body + 2
                    matched_agg = True
                    break

                # alternate paired close used by compact wrapper exports
                if end + 2 <= body_size and data[end] == 0x72 and data[end + 1] == 0x23:
                    out.append(Token(i, "AGG", body + 2, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, i + 3, arity), "term2": 0x7223}))
                    i += body + 2
                    matched_agg = True
                    break

                # fallback single-byte close (covers legacy / partial stubs)
                if end + 1 <= body_size and data[end] in (0x72, 0x30):
                    out.append(Token(i, "AGG", body + 1, {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, i + 3, arity), "term": data[end]}))
                    i += body + 1
                    matched_agg = True
                    break
            if matched_agg:
                continue

        # bare aggregate micro-pattern: 4f N <tag ref32>*
        # observed in many short wrapper exports where the leading selector op is absent
        # or has already been consumed by the previous export boundary.
        if op == 0x4F and i + 2 < body_size:
            raw_arity = data[i + 1]
            matched_agg0 = False
            for arity in _aggregate_arity_candidates(raw_arity):
                body = 2 + 5 * arity
                end = i + body
                if end > body_size:
                    continue
                payload = {"op": op, "raw_arity": raw_arity, "arity": arity, "children": _parse_children(data, i + 2, arity)}
                if end + 2 <= body_size and data[end] == 0x31 and data[end + 1] == 0x30:
                    out.append(Token(i, "AGG0", body + 2, {**payload, "term2": 0x3130}))
                    i += body + 2
                    matched_agg0 = True
                    break
                if end + 2 <= body_size and data[end] == 0x72 and data[end + 1] == 0x23:
                    out.append(Token(i, "AGG0", body + 2, {**payload, "term2": 0x7223}))
                    i += body + 2
                    matched_agg0 = True
                    break
                if end + 1 <= body_size and data[end] in (0x31, 0x72, 0x30):
                    out.append(Token(i, "AGG0", body + 1, {**payload, "term": data[end]}))
                    i += body + 1
                    matched_agg0 = True
                    break
                out.append(Token(i, "AGG0", body, payload))
                i += body
                matched_agg0 = True
                break
            if matched_agg0:
                continue

        if i + 10 <= body_size and data[i + 5] == 0x00 and data[i + 6:i + 10] == b"\x2c\x00\x66\x27":
            out.append(Token(i, "SIG_U32_U8_CALL66_TAIL", 10, {
                "value": u32(data, i),
                "arg": data[i + 4],
            }))
            i += 10
            continue

        if i + 10 <= body_size and data[i:i + 4] == b"\x23\x4f\x02\x10" and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8] == 0x10:
            out.append(Token(i, "SIG_AGG2_PARTIAL_HEAD", 10, {
                "a": data[i + 4],
                "b": data[i + 5],
                "tail": data[i + 9],
            }))
            i += 10
            continue

        if i + 9 <= body_size and data[i:i + 5] == b"\x00\x00\x23\x4f\x01" and data[i + 5] == 0x10 and data[i + 7] == 0x00 and data[i + 8] == 0x00:
            out.append(Token(i, "SIG_AGG1_PARTIAL_HEAD", 9, {
                "value": data[i + 6],
            }))
            i += 9
            continue

        if i + 10 <= body_size and data[i] == 0x10 and data[i + 3] == 0x00 and data[i + 4] == 0x00 and data[i + 5:i + 10] == b"\x29\x10\x01\x3d\x72":
            out.append(Token(i, "SIG_USECLIENT_ALT_HEAD", 10, {
                "lo": data[i + 1],
                "hi": data[i + 2],
            }))
            i += 10
            continue

        # exact recurring export heads recovered from the corpus. These are not learned
        # templates: they are deterministic byte signatures that recur across adb-stable
        # module families and deserve first-class micro-semantics.
        if i + 23 <= body_size and data[i:i + 4] == b"\x10\x01\xf1\x3d" and all(b == 0x7C for b in data[i + 4:i + 23]):
            out.append(Token(i, "SIG_PADDED_CHECKPUT", 23, {"pad_len": 19}))
            i += 23
            continue

        if i + 15 <= body_size and data[i:i + 14] == b"\x00\x00\x72\x30\x32\x29\x10\x01\xf1\x72\x23\x4f\x02\x10":
            out.append(Token(i, "SIG_USEOWNER_HEAD", 15, {"tail": data[i + 14]}))
            i += 15
            continue

        if i + 10 <= body_size and data[i + 1:i + 10] == b"\x29\x10\x01\xf1\x72\x23\x4f\x01\x10":
            out.append(Token(i, "SIG_USECLIENT_HEAD", 10, {"lead": data[i]}))
            i += 10
            continue

        if i + 13 <= body_size and data[i:i + 13] == b"\x10\x04\x00\x3d\x72\x23\x4f\x00\x31\x30\x32\x69\x10":
            out.append(Token(i, "SIG_UNIQUEGEN_HEAD", 13, {}))
            i += 13
            continue

        if i + 10 <= body_size and data[i + 1] == 0x00 and data[i + 2] == 0x00 and data[i + 3] == 0x10 and data[i + 5] == data[i] and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8] == 0x31 and data[i + 9] == 0x30:
            out.append(Token(i, "SIG_USEOFF_HEAD", 10, {"value": data[i], "selector": data[i + 4]}))
            i += 10
            continue

        if i + 14 <= body_size and data[i:i + 5] == b"\x00\x3d\x30\x69\x10" and data[i + 9:i + 11] == b"\x69\x10":
            out.append(Token(i, "SIG_INPUTDONE_HEAD", 14, {
                "first_ref": u32(data, i + 5),
                "tail_lo": data[i + 11],
                "tail_hi": data[i + 12],
                "tail_last": data[i + 13],
            }))
            i += 14
            continue

        # adb-backed recurring wrapper heads that appear verbatim across large module families.
        # These are intentionally exact / narrow to avoid overfitting random byte runs.
        if i + 13 <= body_size and data[i + 3:i + 10] == b"\x2c\x01\x66\x24\x5b\x01\x00" and data[i + 10] == 0x6C and data[i + 11] == 0x01:
            out.append(Token(i, "SIG_CALL66_REFPAIR_HEAD", 13, {
                "imm24": u24(data, i),
                "call_opid": data[i + 6],
                "tail": data[i + 12],
            }))
            i += 13
            continue

        if i + 10 <= body_size and data[i + 3:i + 10] == b"\x2c\x01\x66\x24\x5b\x01\x00":
            out.append(Token(i, "SIG_CALL66_SMALLIMM", 10, {
                "imm24": u24(data, i),
                "call_opid": data[i + 6],
            }))
            i += 10
            continue

        if i + 9 <= body_size and data[i] == 0x01 and data[i + 5] == 0x00 and data[i + 6] == 0x01 and data[i + 7] == 0x00 and data[i + 8] == 0x00:
            out.append(Token(i, "SIG_CONST_U32_TRAILER", 9, {
                "value": u32(data, i + 1),
            }))
            i += 9
            continue

        if i + 13 <= body_size and data[i + 4] == 0x64 and data[i + 5] == 0x30 and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8:i + 12] == b"\x0c\x00\x00\x00" and data[i + 12] in (0x26, 0x64):
            out.append(Token(i, "SIG_SLOT_CONST", 13, {
                "value": u32(data, i),
                "slot_mode": data[i + 5],
                "trailer": data[i + 12],
            }))
            i += 13
            continue

        if i + 29 <= body_size and data[i:i + 29] == b"\x84\xea\x07\x00\x29\x10\x07\x2a\x5b\x04\x00\x29\x10\x06\x5b\x04\x00\x69\x10\x74\xea\x07\x00\x26\x29\x10\x04\x2c\x03":
            out.append(Token(i, "SIG_SETOSST_HEAD", 29, {}))
            i += 29
            continue

        if i + 13 <= body_size and data[i:i + 13] == b"\x06\x00\x30\x00\x00\x00\xf4\xff\xff\xff\x29\x10\x00":
            out.append(Token(i, "SIG_GETPLAYERID_HEAD", 13, {}))
            i += 13
            continue

        if i + 17 <= body_size and data[i:i + 17] == (b"\x7c" * 17):
            out.append(Token(i, "SIG_PAD17", 17, {"byte": 0x7C}))
            i += 17
            continue

        if i + 13 <= body_size and data[i:i + 11] == (b"\x7c" * 11) and data[i + 11] == 0x4A:
            out.append(Token(i, "SIG_PAD11_BR", 13, {"op": 0x4A, "off": data[i + 12]}))
            i += 13
            continue

        matched_pad_sig = False
        for pad_len in (5, 8, 16):
            if i + pad_len + 3 <= body_size and data[i:i + pad_len] == (b"\x7c" * pad_len) and data[i + pad_len] in (0x4A, 0x4B, 0x4C, 0x4D):
                out.append(Token(i, "SIG_PADRUN_BR", pad_len + 3, {"pad_len": pad_len, "op": data[i + pad_len], "off": struct.unpack_from("<H", data, i + pad_len + 1)[0]}))
                i += pad_len + 3
                matched_pad_sig = True
                break
        if matched_pad_sig:
            continue

        # recurring wrapper head: short 0x7c pad run bridging directly into 30+REF.
        for pad_len in (5, 8, 9, 13):
            if i + pad_len + 7 <= body_size and data[i:i + pad_len] == (b"\x7c" * pad_len) and data[i + pad_len] == 0x30 and data[i + pad_len + 1] in (0x69, 0x65, 0x6C):
                out.append(Token(i, "SIG_PADRUN_OPREF", pad_len + 7, {"pad_len": pad_len, "ref_op": data[i + pad_len + 1], "mode": data[i + pad_len + 2], "ref": u32(data, i + pad_len + 3)}))
                i += pad_len + 7
                matched_pad_sig = True
                break
        if matched_pad_sig:
            continue

        if i + 11 <= body_size and data[i:i + 11] == b"\x00\x80\x3f\x28\x10\xc0\x00\x2e\x2f\x2d\x3d":
            out.append(Token(i, "SIG_USEOFF_CONST_CHAIN", 11, {}))
            i += 11
            continue

        if i + 17 <= body_size and data[i:i + 17] == b"\x00\x00\x29\x10\x1e\x3c\x4b\x15\x00" + (b"\x7c" * 8):
            out.append(Token(i, "SIG_GETMODIFIERS_PADTAIL", 17, {}))
            i += 17
            continue

        if i + 4 <= body_size and data[i:i + 4] == b"\x41\x00\x00\x00":
            out.append(Token(i, "IMM24Z", 4, {"op": 0x41, "imm": 0}))
            i += 4
            continue

        if i + 4 <= body_size and data[i:i + 4] == b"\x00\x10\x00\x00":
            out.append(Token(i, "IMM24U", 4, {"op": 0x00, "imm": 0x10}))
            i += 4
            continue

        if i + 18 <= body_size and data[i:i + 18] == b"\xff\x23\x4f\x00\x31\x30\x32\x6c\x01\x08\x00\x00\x00\x14\x00\x00\x00\x72":
            out.append(Token(i, "SIG_GETCASTLENUM_HEAD", 18, {}))
            i += 18
            continue

        if i + 32 <= body_size and data[i:i + 4] == b"\x4f\x00\x31\x30" and data[i + 4] in (0x69, 0x6C) and data[i + 10] == 0x6C and data[i + 11] == 0x01 and data[i + 16:i + 20] == b"\x04\x00\x00\x00" and data[i + 20:i + 23] == b"\x3d\x30\x32" and data[i + 23] in (0x69, 0x6C) and data[i + 29] == 0x5E and data[i + 30:i + 32] == b"r#":
            out.append(Token(i, "SIG_GETWEAR_WRAPPER", 32, {
                "ref_a_op": data[i + 4],
                "ref_a_mode": data[i + 5],
                "ref_a": u32(data, i + 6),
                "ref_b": u32(data, i + 12),
                "const": u32(data, i + 16),
                "ref_c_op": data[i + 23],
                "ref_c_mode": data[i + 24],
                "ref_c": u32(data, i + 25),
                "tail_op": 0x5E,
            }))
            i += 32
            continue

        if i + 19 <= body_size and data[i:i + 2] == b"O" and data[i + 7:i + 10] == b"102" and data[i + 10] in (0x69, 0x6C) and data[i + 16] == 0x26 and data[i + 17:i + 19] == b"r#":
            out.append(Token(i, "SIG_GETP_WRAPPER", 19, {
                "child_tag": data[i + 2],
                "child_ref": u32(data, i + 3),
                "ref_op": data[i + 10],
                "ref_mode": data[i + 11],
                "ref": u32(data, i + 12),
                "tail_op": 0x26,
            }))
            i += 19
            continue

        if i + 14 <= body_size and data[i:i + 4] == b"\x4f\x00\x31\x30" and data[i + 4] == 0x32 and data[i + 5] in (0x69, 0x6C) and data[i + 11] == 0x21 and data[i + 12:i + 14] == b"r#":
            out.append(Token(i, "SIG_INPUTDONE_SHORT", 14, {
                "ref_op": data[i + 5],
                "ref_mode": data[i + 6],
                "ref": u32(data, i + 7),
                "tail_op": 0x21,
            }))
            i += 14
            continue

        if i + 6 <= body_size and data[i] in (0x3C, 0x3E, 0xEC, 0xED) and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, f"PFX_{data[i]:02X}_E8_EB_BR", 6, {
                "prefix_op": data[i],
                "nested_kind": "PFX_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 4 <= body_size and data[i] == 0xEC and data[i + 1] == 0xE8 and data[i + 2:i + 4] == b"r#":
            out.append(Token(i, "PFX_EC_E8_PAIR72_23", 4, {
                "prefix_op": 0xEC,
                "nested_kind": "PFX_E8_PAIR72_23",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PAIR72_23", "nested": {"bytes": "72 23"}},
            }))
            i += 4
            continue
        if i + 5 <= body_size and data[i] == 0xEC and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3:i + 5] == b"r#":
            out.append(Token(i, "PFX_EC_E8_EB_PAIR72_23", 5, {
                "prefix_op": 0xEC,
                "nested_kind": "PFX_E8_EB_PAIR72_23",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_PAIR72_23", "nested": {"prefix_op": 0xEB, "nested_kind": "PAIR72_23", "nested": {"bytes": "72 23"}}},
            }))
            i += 5
            continue

        if i + 11 <= body_size and data[i + 4] == 0x3D and data[i + 5] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "SIG_CONST_U32_PFX_3D_REF", 11, {
                "value": u32(data, i),
                "ref_op": data[i + 5],
                "ref_mode": data[i + 6],
                "ref": u32(data, i + 7),
            }))
            i += 11
            continue

        if i + 8 <= body_size and data[i + 4] == 0x28 and data[i + 5] == 0x10:
            out.append(Token(i, "SIG_CONST_U32_IMM16", 8, {
                "value": u32(data, i),
                "imm16": struct.unpack_from('<H', data, i + 6)[0],
            }))
            i += 8
            continue

        if i + 11 <= body_size and data[i] == 0xEF and data[i + 5] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "PFX_EF_SIG_CONST_U32_REF", 11, {
                "prefix_op": 0xEF,
                "nested_kind": "SIG_CONST_U32_REF",
                "nested": {
                    "value": u32(data, i + 1),
                    "ref_op": data[i + 5],
                    "ref_mode": data[i + 6],
                    "ref": u32(data, i + 7),
                },
            }))
            i += 11
            continue

        if i + 6 <= body_size and data[i] == 0x2E and data[i + 1] == 0xEC and data[i + 2] == 0xE8 and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_2E_EC_E8_BR", 6, {
                "prefix_op": 0x2E,
                "nested_kind": "PFX_EC_E8_BR",
                "nested": {"prefix_op": 0xEC, "nested_kind": "PFX_E8_BR", "nested": {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 17 <= body_size and data[i:i + 17] == bytes.fromhex("5b0100412910000006006910e80d00002c") and data[i + 17:i + 20] == bytes.fromhex("036643"):
            out.append(Token(i, "SIG_TELEP_CREATEINFOPICTURE_TAIL", 20, {
                "signature": "5b0100412910000006006910e80d00002c036643",
            }))
            i += 20
            continue

        if i + 9 <= body_size and data[i:i + 9] == bytes.fromhex("012c89000014000000"):
            out.append(Token(i, "SIG_PLAYER_GETLEADER_TAIL", 9, {
                "signature": "012c89000014000000",
            }))
            i += 9
            continue

        if i + 22 <= body_size and data[i:i + 22] == bytes.fromhex("3c0000003b2d6c01040800003c0000002c04661a7223"):
            out.append(Token(i, "SIG_PLAYER_LOSTITEM2_TAIL", 22, {
                "signature": "3c0000003b2d6c01040800003c0000002c04661a7223",
            }))
            i += 22
            continue

        if i + 13 <= body_size and data[i:i + 13] == bytes.fromhex("200000003b2d291020e14b2200"):
            out.append(Token(i, "SIG_MAIN_PARSECOMMAND_TAIL", 13, {
                "signature": "200000003b2d291020e14b2200",
                "word_u32": u32(data, i),
                "cmp_value": data[i + 8],
                "branch_op": data[i + 10],
                "offset": struct.unpack_from('<H', data, i + 11)[0],
            }))
            i += 13
            continue


        # Rare high-prefix ref family observed in definition bodies:
        #   F0 E8 3D 30 <REF>
        # Without this explicit guard, F0+E8 is greedily misread as an
        # unsigned-imm24 prefix and leaves the REF payload as unknown bytes.
        if i + 10 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2:i + 4] == b"\x3d\x30" and data[i + 4] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "PFX_F0_E8_3D_30_REF", 10, {
                "prefix_op": 0xF0,
                "nested_kind": "PFX_E8_3D_30_REF",
                "nested": {
                    "prefix_op": 0xE8,
                    "nested_kind": "PFX_3D_30_REF",
                    "nested": {
                        "prefix_op": 0x3D,
                        "nested_kind": "PFX_30_REF",
                        "nested": {
                            "prefix_op": 0x30,
                            "nested_kind": "REF",
                            "nested": {
                                "op": data[i + 4],
                                "mode": data[i + 5],
                                "ref": u32(data, i + 6),
                            },
                        },
                    },
                },
            }))
            i += 10
            continue


        if i + 5 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_F0_E8_BR", 5, {
                "prefix_op": 0xF0,
                "nested_kind": "PFX_E8_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 2], "off": struct.unpack_from('<H', data, i + 3)[0]}},
            }))
            i += 5
            continue

        direct_atomic = _match_atomic_semantic(data, i, body_size)
        custom_e8_family = (
            (i + 5 <= body_size and data[i] == 0xE8 and data[i + 1] in (0x4A, 0x4B, 0x4C, 0x4D))
            or (i + 6 <= body_size and data[i] == 0xE8 and data[i + 1] in (0x3D, 0xEB) and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D))
            or (i + 6 <= body_size and data[i] in (0x3C, 0x3E, 0xEC, 0xED) and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D))
            or (i + 4 <= body_size and data[i] == 0xEC and data[i + 1] == 0xE8 and data[i + 2:i + 4] == b"r#")
            or (i + 5 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D))
            or (i + 6 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2] in (0x3D, 0xEB) and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D))
        )
        if direct_atomic is not None and direct_atomic[1] >= 4 and not custom_e8_family:
            kind, tok_size, payload = direct_atomic
            out.append(Token(i, kind, tok_size, payload))
            i += tok_size
            continue

        direct_prefixed = _match_prefixed_semantic(data, i, body_size)
        if direct_prefixed is not None and direct_prefixed[1] >= 4 and not custom_e8_family:
            kind, tok_size, payload = direct_prefixed
            out.append(Token(i, kind, tok_size, payload))
            i += tok_size
            continue

        if i + 12 <= body_size and data[i + 4] == 0x3D and data[i + 5] == 0x30 and data[i + 6] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "SIG_CONST_U32_PFX_3D_30_REF", 12, {
                "value": u32(data, i),
                "ref_op": data[i + 6],
                "ref_mode": data[i + 7],
                "ref": u32(data, i + 8),
            }))
            i += 12
            continue

        if i + 11 <= body_size and data[i + 4] == 0x5E and data[i + 5] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "SIG_CONST_U32_PFX_5E_REF", 11, {
                "value": u32(data, i),
                "ref_op": data[i + 5],
                "ref_mode": data[i + 6],
                "ref": u32(data, i + 7),
            }))
            i += 11
            continue

        if i + 7 <= body_size and data[i + 4] == 0x3D and data[i + 5:i + 7] == b"r#":
            out.append(Token(i, "SIG_CONST_U32_PFX_3D_PAIR72_23", 7, {
                "value": u32(data, i),
            }))
            i += 7
            continue

        if i + 8 <= body_size and data[i + 1] == 0 and data[i + 2] == 0 and data[i + 3] == 0 and data[i + 4] == 0x64 and data[i + 5] == 0x10:
            out.append(Token(i, "SIG_CONST_U32_REF16", 8, {
                "value": u32(data, i),
                "ref": struct.unpack_from('<H', data, i + 6)[0],
            }))
            i += 8
            continue

        if i + 7 <= body_size and data[i + 4] == 0x29 and data[i + 5] == 0x10:
            out.append(Token(i, "SIG_CONST_U32_IMM", 7, {
                "value": u32(data, i),
                "imm": data[i + 6],
            }))
            i += 7
            continue

        if i + 8 <= body_size and data[i + 4] == 0x2C and data[i + 6] == 0x66:
            out.append(Token(i, "SIG_CONST_U32_CALL66", 8, {
                "value": u32(data, i),
                "argc": data[i + 5],
                "opid": data[i + 7],
            }))
            i += 8
            continue

        if i + 11 <= body_size and data[i + 4] == 0x2C and data[i + 6] == 0x63:
            out.append(Token(i, "SIG_CONST_U32_CALL63A", 11, {
                "value": u32(data, i),
                "argc": data[i + 5],
                "rel": s32(data, i + 7),
            }))
            i += 11
            continue

        if i + 8 <= body_size and data[i + 4] == 0x5E and data[i + 5] == 0x29 and data[i + 6] == 0x10:
            out.append(Token(i, "SIG_CONST_U32_5E_IMM", 8, {
                "value": u32(data, i),
                "prefix_op": 0x5E,
                "imm": data[i + 7],
            }))
            i += 8
            continue

        if i + 9 <= body_size and data[i + 4] == 0x26 and data[i + 5] == 0x2C and data[i + 7] == 0x66:
            out.append(Token(i, "SIG_CONST_U32_26_CALL66", 9, {
                "value": u32(data, i),
                "argc": data[i + 6],
                "opid": data[i + 8],
            }))
            i += 9
            continue

        if i + 11 <= body_size and data[i + 4] == 0x26 and data[i + 5] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "SIG_CONST_U32_26_REF", 11, {
                "value": u32(data, i),
                "ref_op": data[i + 5],
                "ref_mode": data[i + 6],
                "ref": u32(data, i + 7),
            }))
            i += 11
            continue

        if i + 10 <= body_size and data[i + 4] in (0x69, 0x65, 0x6C):
            out.append(Token(i, "SIG_CONST_U32_REF", 10, {
                "value": u32(data, i),
                "ref_op": data[i + 4],
                "ref_mode": data[i + 5],
                "ref": u32(data, i + 6),
            }))
            i += 10
            continue

        # Guard against a false fusion seen in the createInfoPicture family:
        #   OPU16 + REC41 can look like <u32><0x41...> if we only check byte +4.
        # Prefer the cleaner structural split when the leading dword is actually
        # a short-u16 op followed immediately by REC41.
        if i + 11 <= body_size and data[i + 4] == 0x41 and not (data[i] in SHORT_U16_OPS and data[i + 3] == 0x41):
            out.append(Token(i, "SIG_CONST_U32_REC41", 11, {
                "value": u32(data, i),
                "ref": u32(data, i + 5),
                "imm": struct.unpack_from('<H', data, i + 9)[0],
            }))
            i += 11
            continue

        if i + 5 <= body_size and data[i + 4] == 0x72:
            nested = _match_prefixed_semantic(data, i + 4, body_size)
            if nested is not None and (nested[0].startswith("PFX_72_") or nested[0] == "PAIR72_23"):
                nested_kind, nested_size, nested_payload = nested
                out.append(Token(i, "SIG_CONST_U32_PFX72", 4 + nested_size, {
                    "value": u32(data, i),
                    "nested_kind": nested_kind,
                    "nested": nested_payload,
                }))
                i += 4 + nested_size
                continue

        if i + 5 <= body_size and data[i] == 0xE8 and data[i + 1] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_E8_BR", 4, {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 1], "off": struct.unpack_from('<H', data, i + 2)[0]}}))
            i += 4
            continue

        if i + 6 <= body_size and data[i] == 0xE8 and data[i + 1] == 0x3D and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_E8_3D_BR", 5, {
                "prefix_op": 0xE8,
                "nested_kind": "PFX_3D_BR",
                "nested": {"prefix_op": 0x3D, "nested_kind": "BR", "nested": {"op": data[i + 2], "off": struct.unpack_from('<H', data, i + 3)[0]}},
            }))
            i += 5
            continue

        if i + 6 <= body_size and data[i] == 0xE8 and data[i + 1] == 0xEB and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_E8_EB_BR", 5, {
                "prefix_op": 0xE8,
                "nested_kind": "PFX_EB_BR",
                "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 2], "off": struct.unpack_from('<H', data, i + 3)[0]}},
            }))
            i += 5
            continue

        if i + 6 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2] == 0x3D and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_F0_E8_3D_BR", 6, {
                "prefix_op": 0xF0,
                "nested_kind": "PFX_E8_3D_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_3D_BR", "nested": {"prefix_op": 0x3D, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 6 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_F0_E8_EB_BR", 6, {
                "prefix_op": 0xF0,
                "nested_kind": "PFX_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 6 <= body_size and data[i] in (0xEC, 0xE1, 0x21, 0xEB) and data[i + 1] == 0xE8 and data[i + 2] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, f"PFX_{data[i]:02X}_E8_BR", 5, {
                "prefix_op": data[i],
                "nested_kind": "PFX_E8_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 2], "off": struct.unpack_from('<H', data, i + 3)[0]}},
            }))
            i += 5
            continue

        if i + 7 <= body_size and data[i] in (0xEC, 0xE1) and data[i + 1] == 0xE8 and data[i + 2] == 0x3D and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, f"PFX_{data[i]:02X}_E8_3D_BR", 6, {
                "prefix_op": data[i],
                "nested_kind": "PFX_E8_3D_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_3D_BR", "nested": {"prefix_op": 0x3D, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 7 <= body_size and data[i] == 0xF0 and data[i + 1] == 0xEB and data[i + 2] == 0xE8 and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_F0_EB_E8_BR", 6, {
                "prefix_op": 0xF0,
                "nested_kind": "PFX_EB_E8_BR",
                "nested": {"prefix_op": 0xEB, "nested_kind": "PFX_E8_BR", "nested": {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 6 <= body_size and data[i] == 0xE8 and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_E8_E8_EB_BR", 6, {
                "prefix_op": 0xE8,
                "nested_kind": "PFX_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 7 <= body_size and data[i] in (0xEC, 0xE1) and data[i + 1] == 0xE8 and data[i + 2] == 0xE8 and data[i + 3] == 0xEB and data[i + 4] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, f"PFX_{data[i]:02X}_E8_E8_EB_BR", 7, {
                "prefix_op": data[i],
                "nested_kind": "PFX_E8_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_E8_EB_BR", "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 4], "off": struct.unpack_from('<H', data, i + 5)[0]}}}},
            }))
            i += 7
            continue

        if i + 6 <= body_size and data[i] in (0xEC, 0xE1) and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, f"PFX_{data[i]:02X}_E8_EB_BR", 6, {
                "prefix_op": data[i],
                "nested_kind": "PFX_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        # duplicated signature block removed; handled above



        if i + 6 <= body_size and data[i] == 0x3C and data[i + 1] == 0xE8 and data[i + 2] == 0xEB and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_3C_E8_EB_BR", 6, {
                "prefix_op": 0x3C,
                "nested_kind": "PFX_E8_EB_BR",
                "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        if i + 7 <= body_size and data[i] == 0x3E and data[i + 1] == 0xEB and data[i + 2] == 0xE8 and data[i + 3] == 0xEB and data[i + 4] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_3E_EB_E8_EB_BR", 7, {
                "prefix_op": 0x3E,
                "nested_kind": "PFX_EB_E8_EB_BR",
                "nested": {"prefix_op": 0xEB, "nested_kind": "PFX_E8_EB_BR", "nested": {"prefix_op": 0xE8, "nested_kind": "PFX_EB_BR", "nested": {"prefix_op": 0xEB, "nested_kind": "BR", "nested": {"op": data[i + 4], "off": struct.unpack_from('<H', data, i + 5)[0]}}}},
            }))
            i += 7
            continue

        if i + 6 <= body_size and data[i] == 0x3E and data[i + 1] == 0xEB and data[i + 2] == 0xE8 and data[i + 3] in (0x4A, 0x4B, 0x4C, 0x4D):
            out.append(Token(i, "PFX_3E_EB_E8_BR", 6, {
                "prefix_op": 0x3E,
                "nested_kind": "PFX_EB_E8_BR",
                "nested": {"prefix_op": 0xEB, "nested_kind": "PFX_E8_BR", "nested": {"prefix_op": 0xE8, "nested_kind": "BR", "nested": {"op": data[i + 3], "off": struct.unpack_from('<H', data, i + 4)[0]}}},
            }))
            i += 6
            continue

        prefixed = _match_prefixed_semantic(data, i, body_size)
        if prefixed is not None:
            kind, tok_size, payload = prefixed
            out.append(Token(i, kind, tok_size, payload))
            i += tok_size
            continue

        if i + 6 <= body_size and data[i] == 0x39 and data[i + 1] == 0x20:
            bits = u32(data, i + 2)
            value = struct.unpack('<f', struct.pack('<I', bits))[0]
            out.append(Token(i, "F32", 6, {"op": 0x39, "mode": 0x20, "bits": bits, "value": value if math.isfinite(value) else None}))
            i += 6
            continue

        if i + 4 <= body_size and data[i:i + 4] == b"\x00\x01\x00\x00":
            out.append(Token(i, "SIG_CONST_0100", 4, {"value": 256}))
            i += 4
            continue

        if i + 12 <= body_size and data[i + 4] == 0x62 and data[i + 5] == 0x10:
            out.append(Token(i, "SIG_CONST_U32_REC62", 12, {
                "value": u32(data, i),
                "mode": 0x10,
                "u16": struct.unpack_from('<H', data, i + 6)[0],
                "c": s32(data, i + 8),
            }))
            i += 12
            continue

        blob_size, blob_payload = _scan_dword_blob(data, i, size)
        if blob_size:
            out.append(Token(i, "DWBLOB", blob_size, blob_payload))
            i += blob_size
            continue

        # compact immediate families inferred from repeated low-coverage wrapper motifs
        if op in SIGNED_IMM24_OPS and i + 4 <= body_size:
            out.append(Token(i, "IMM24S", 4, {"op": op, "imm": s24(data, i + 1)}))
            i += 4
            continue

        if op in UNSIGNED_IMM24_OPS and i + 4 <= body_size:
            out.append(Token(i, "IMM24U", 4, {"op": op, "imm": u24(data, i + 1)}))
            i += 4
            continue

        # generic compact op + zero-imm24 form. This shows up repeatedly in still-moderate
        # long bodies such as InitObj/GetParam, usually as small structural arguments between
        # recognized refs/calls. Keep it conservative: only accept the exact zero-imm shape.
        if op in GENERIC_ZERO_IMM24_OPS and i + 4 <= body_size and data[i + 1] == 0 and data[i + 2] == 0 and data[i + 3] == 0:
            out.append(Token(i, "IMM24Z", 4, {"op": op, "imm": 0}))
            i += 4
            continue

        if op == 0x08 and i + 2 <= body_size and data[i + 1] == 0:
            out.append(Token(i, "MARK", 2, {"op": op, "arg": 0, "role": "marker"}))
            i += 2
            continue

        # long / short ref families
        if op in (0x69, 0x65, 0x6C) and i + 6 <= body_size:
            out.append(Token(i, "REF", 6, {"op": op, "mode": data[i + 1], "ref": u32(data, i + 2)}))
            i += 6
            continue

        if op == 0x64 and i + 4 <= body_size:
            out.append(Token(i, "REF16", 4, {"op": op, "mode": data[i + 1], "ref": struct.unpack_from('<H', data, i + 2)[0]}))
            i += 4
            continue

        # structured literals
        if op == 0x41 and i + 7 <= body_size:
            out.append(Token(i, "REC41", 7, {"ref": u32(data, i + 1), "imm": struct.unpack_from("<H", data, i + 5)[0]}))
            i += 7
            continue

        if op == 0x61 and i + 16 <= body_size:
            out.append(Token(i, "REC61", 16, {
                "mode": data[i + 1],
                "u16": struct.unpack_from("<H", data, i + 2)[0],
                "a": u32(data, i + 4),
                "b": u32(data, i + 8),
                "c": s32(data, i + 12),
            }))
            i += 16
            continue

        if op == 0x62 and i + 8 <= body_size:
            out.append(Token(i, "REC62", 8, {
                "mode": data[i + 1],
                "u16": struct.unpack_from("<H", data, i + 2)[0],
                "c": s32(data, i + 4),
            }))
            i += 8
            continue

        # call / transfer families
        if op == 0x2C and i + 4 <= body_size and data[i + 2] == 0x66:
            out.append(Token(i, "CALL66", 4, {"argc": data[i + 1], "opid": data[i + 3]}))
            i += 4
            continue

        if op == 0x2C and i + 7 <= body_size and data[i + 2] == 0x63:
            out.append(Token(i, "CALL63A", 7, {"argc": data[i + 1], "rel": s32(data, i + 3)}))
            i += 7
            continue

        if op == 0x63 and i + 5 <= body_size:
            out.append(Token(i, "CALL63B", 5, {"rel": s32(data, i + 1)}))
            i += 5
            continue

        # short imm / branch
        if op == 0x29 and i + 3 <= body_size and data[i + 1] == 0x10:
            out.append(Token(i, "IMM", 3, {"value": data[i + 2]}))
            i += 3
            continue

        # related compact u16 literal form observed in several short wrappers
        if op == 0x28 and i + 4 <= body_size and data[i + 1] == 0x10:
            out.append(Token(i, "IMM16", 4, {"op": op, "value": struct.unpack_from('<H', data, i + 2)[0]}))
            i += 4
            continue

        if op in SHORT_U16_OPS and i + 3 <= body_size:
            out.append(Token(i, "OPU16", 3, {"op": op, "value": struct.unpack_from('<H', data, i + 1)[0]}))
            i += 3
            continue

        if op in (0x4A, 0x4B, 0x4C, 0x4D) and i + 3 <= body_size:
            out.append(Token(i, "BR", 3, {"op": op, "off": struct.unpack_from("<H", data, i + 1)[0]}))
            i += 3
            continue

        # common compact imm32 family observed in many service-style exports:
        #   30 <imm32-le>
        # A fully-open 30+u32 rule causes regressions because plain 0x30 also occurs
        # as a real single-byte op. A safe expansion is to accept 16-bit immediates
        # encoded as 30 xx yy 00 00 in addition to the old 8-bit-only form.
        if op == 0x30 and i + 5 <= body_size and data[i + 3] == 0 and data[i + 4] == 0:
            out.append(Token(i, "IMM32", 5, {"op": op, "imm": u32(data, i + 1)}))
            i += 5
            continue

        # pad / filler blocks (often '||||' or '\xff\xff\xff\xff')
        if op in (0x7C, 0xFF):
            j = i
            while j < body_size and data[j] == op:
                j += 1
            run_len = j - i
            if run_len >= 4:
                out.append(Token(i, "PAD", run_len, {"byte": op, "len": run_len}))
                i = j
                continue

        # ASCII-ish data blocks (format strings, HTML-ish snippets, CSV-like rows, etc.)
        if _is_printable_ascii(op):
            j = _ascii_run(data, i, body_size)
            run = data[i:j]
            if _looks_like_text(run):
                # Keep payload small; we only need hints for debugging.
                preview = run[:120].decode("latin1", "replace")
                out.append(Token(i, "ASCII", len(run), {"preview": preview}))
                i = j
                continue

        if op in SINGLE_BYTE_OPS:
            kind, payload = _single_byte_structural_token(op)
            out.append(Token(i, kind, 1, payload))
            i += 1
            continue

        out.append(Token(i, "UNK", 1, {"op": op}))
        i += 1

    return out

