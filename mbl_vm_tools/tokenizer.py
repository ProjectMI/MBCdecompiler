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
    0x00, 0x23, 0x27, 0x28, 0x30, 0x31, 0x32, 0x48, 0x72, 0x7C,
    0x3D, 0x2A, 0x2D, 0x2F, 0x25,
    0xF0, 0xF1, 0xE1, 0xED, 0x5E, 0xEB, 0x3C, 0x3E,
}
SHORT_U16_OPS = {0x52, 0xCF}
SIGNED_IMM24_OPS = {0x6D}
UNSIGNED_IMM24_OPS = {0x67}


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


def tokenize_stream(data: bytes, limit: int | None = None) -> list[Token]:
    size = len(data) if limit is None else min(len(data), limit)
    out: list[Token] = []
    i = 0

    while i < size:
        op = data[i]

        # aggregate families: 23 4f N, 74 4f N
        # children=(tag + ref32)
        # close is usually 31 30, but some stubs use a single-byte terminator (observed: 0x72, 0x30)
        if i + 2 < size and data[i + 1] == 0x4F and op in (0x23, 0x74):
            arity = data[i + 2]
            body = 3 + 5 * arity
            end = i + body
            # normal two-byte close
            if end + 2 <= size and data[end] == 0x31 and data[end + 1] == 0x30:
                out.append(Token(i, "AGG", body + 2, {"op": op, "arity": arity, "children": _parse_children(data, i + 3, arity)}))
                i += body + 2
                continue

            # alternate single-byte close (covers common micro-dispatch stubs)
            if end + 1 <= size and data[end] in (0x72, 0x30):
                out.append(Token(i, "AGG", body + 1, {"op": op, "arity": arity, "children": _parse_children(data, i + 3, arity), "term": data[end]}))
                i += body + 1
                continue

        # bare aggregate micro-pattern: 4f N <tag ref32>*
        # observed in many short wrapper exports where the leading selector op is absent
        # or has already been consumed by the previous export boundary.
        if op == 0x4F and i + 2 < size:
            arity = data[i + 1]
            if 0 < arity <= 8:
                body = 2 + 5 * arity
                end = i + body
                if end <= size:
                    payload = {"op": op, "arity": arity, "children": _parse_children(data, i + 2, arity)}
                    if end + 2 <= size and data[end] == 0x31 and data[end + 1] == 0x30:
                        out.append(Token(i, "AGG0", body + 2, {**payload, "term2": 0x3130}))
                        i += body + 2
                        continue
                    if end + 1 <= size and data[end] in (0x72, 0x30):
                        out.append(Token(i, "AGG0", body + 1, {**payload, "term": data[end]}))
                        i += body + 1
                        continue
                    out.append(Token(i, "AGG0", body, payload))
                    i += body
                    continue

        if i + 10 <= size and data[i + 5] == 0x00 and data[i + 6:i + 10] == b"\x2c\x00\x66\x27":
            out.append(Token(i, "SIG_U32_U8_CALL66_TAIL", 10, {
                "value": u32(data, i),
                "arg": data[i + 4],
            }))
            i += 10
            continue

        if i + 10 <= size and data[i:i + 4] == b"\x23\x4f\x02\x10" and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8] == 0x10:
            out.append(Token(i, "SIG_AGG2_PARTIAL_HEAD", 10, {
                "a": data[i + 4],
                "b": data[i + 5],
                "tail": data[i + 9],
            }))
            i += 10
            continue

        if i + 9 <= size and data[i:i + 5] == b"\x00\x00\x23\x4f\x01" and data[i + 5] == 0x10 and data[i + 7] == 0x00 and data[i + 8] == 0x00:
            out.append(Token(i, "SIG_AGG1_PARTIAL_HEAD", 9, {
                "value": data[i + 6],
            }))
            i += 9
            continue

        if i + 10 <= size and data[i] == 0x10 and data[i + 3] == 0x00 and data[i + 4] == 0x00 and data[i + 5:i + 10] == b"\x29\x10\x01\x3d\x72":
            out.append(Token(i, "SIG_USECLIENT_ALT_HEAD", 10, {
                "lo": data[i + 1],
                "hi": data[i + 2],
            }))
            i += 10
            continue

        # exact recurring export heads recovered from the corpus. These are not learned
        # templates: they are deterministic byte signatures that recur across adb-stable
        # module families and deserve first-class micro-semantics.
        if i + 23 <= size and data[i:i + 4] == b"\x10\x01\xf1\x3d" and all(b == 0x7C for b in data[i + 4:i + 23]):
            out.append(Token(i, "SIG_PADDED_CHECKPUT", 23, {"pad_len": 19}))
            i += 23
            continue

        if i + 15 <= size and data[i:i + 14] == b"\x00\x00\x72\x30\x32\x29\x10\x01\xf1\x72\x23\x4f\x02\x10":
            out.append(Token(i, "SIG_USEOWNER_HEAD", 15, {"tail": data[i + 14]}))
            i += 15
            continue

        if i + 10 <= size and data[i + 1:i + 10] == b"\x29\x10\x01\xf1\x72\x23\x4f\x01\x10":
            out.append(Token(i, "SIG_USECLIENT_HEAD", 10, {"lead": data[i]}))
            i += 10
            continue

        if i + 13 <= size and data[i:i + 13] == b"\x10\x04\x00\x3d\x72\x23\x4f\x00\x31\x30\x32\x69\x10":
            out.append(Token(i, "SIG_UNIQUEGEN_HEAD", 13, {}))
            i += 13
            continue

        if i + 10 <= size and data[i + 1] == 0x00 and data[i + 2] == 0x00 and data[i + 3] == 0x10 and data[i + 5] == data[i] and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8] == 0x31 and data[i + 9] == 0x30:
            out.append(Token(i, "SIG_USEOFF_HEAD", 10, {"value": data[i], "selector": data[i + 4]}))
            i += 10
            continue

        if i + 14 <= size and data[i:i + 5] == b"\x00\x3d\x30\x69\x10" and data[i + 9:i + 11] == b"\x69\x10":
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
        if i + 13 <= size and data[i + 3:i + 10] == b"\x2c\x01\x66\x24\x5b\x01\x00" and data[i + 10] == 0x6C and data[i + 11] == 0x01:
            out.append(Token(i, "SIG_CALL66_REFPAIR_HEAD", 13, {
                "imm24": u24(data, i),
                "call_opid": data[i + 6],
                "tail": data[i + 12],
            }))
            i += 13
            continue

        if i + 10 <= size and data[i + 3:i + 10] == b"\x2c\x01\x66\x24\x5b\x01\x00":
            out.append(Token(i, "SIG_CALL66_SMALLIMM", 10, {
                "imm24": u24(data, i),
                "call_opid": data[i + 6],
            }))
            i += 10
            continue

        if i + 9 <= size and data[i] == 0x01 and data[i + 5] == 0x00 and data[i + 6] == 0x01 and data[i + 7] == 0x00 and data[i + 8] == 0x00:
            out.append(Token(i, "SIG_CONST_U32_TRAILER", 9, {
                "value": u32(data, i + 1),
            }))
            i += 9
            continue

        if i + 13 <= size and data[i + 4] == 0x64 and data[i + 5] == 0x30 and data[i + 6] == 0x00 and data[i + 7] == 0x00 and data[i + 8:i + 12] == b"\x0c\x00\x00\x00" and data[i + 12] in (0x26, 0x64):
            out.append(Token(i, "SIG_SLOT_CONST", 13, {
                "value": u32(data, i),
                "slot_mode": data[i + 5],
                "trailer": data[i + 12],
            }))
            i += 13
            continue

        if i + 29 <= size and data[i:i + 29] == b"\x84\xea\x07\x00\x29\x10\x07\x2a\x5b\x04\x00\x29\x10\x06\x5b\x04\x00\x69\x10\x74\xea\x07\x00\x26\x29\x10\x04\x2c\x03":
            out.append(Token(i, "SIG_SETOSST_HEAD", 29, {}))
            i += 29
            continue

        if i + 13 <= size and data[i:i + 13] == b"\x06\x00\x30\x00\x00\x00\xf4\xff\xff\xff\x29\x10\x00":
            out.append(Token(i, "SIG_GETPLAYERID_HEAD", 13, {}))
            i += 13
            continue

        if i + 17 <= size and data[i:i + 17] == (b"\x7c" * 17):
            out.append(Token(i, "SIG_PAD17", 17, {"byte": 0x7C}))
            i += 17
            continue

        if i + 13 <= size and data[i:i + 11] == (b"\x7c" * 11) and data[i + 11] == 0x4A:
            out.append(Token(i, "SIG_PAD11_BR", 13, {"off": data[i + 12]}))
            i += 13
            continue

        matched_pad_sig = False
        for pad_len in (5, 8, 16):
            if i + pad_len + 3 <= size and data[i:i + pad_len] == (b"\x7c" * pad_len) and data[i + pad_len] in (0x4A, 0x4B, 0x4C, 0x4D):
                out.append(Token(i, "SIG_PADRUN_BR", pad_len + 3, {"pad_len": pad_len, "op": data[i + pad_len], "off": struct.unpack_from("<H", data, i + pad_len + 1)[0]}))
                i += pad_len + 3
                matched_pad_sig = True
                break
        if matched_pad_sig:
            continue

        # recurring wrapper head: short 0x7c pad run bridging directly into 30+REF.
        for pad_len in (5, 8, 9, 13):
            if i + pad_len + 7 <= size and data[i:i + pad_len] == (b"\x7c" * pad_len) and data[i + pad_len] == 0x30 and data[i + pad_len + 1] in (0x69, 0x65, 0x6C):
                out.append(Token(i, "SIG_PADRUN_OPREF", pad_len + 7, {"pad_len": pad_len, "ref_op": data[i + pad_len + 1], "mode": data[i + pad_len + 2], "ref": u32(data, i + pad_len + 3)}))
                i += pad_len + 7
                matched_pad_sig = True
                break
        if matched_pad_sig:
            continue

        if i + 11 <= size and data[i:i + 11] == b"\x00\x80\x3f\x28\x10\xc0\x00\x2e\x2f\x2d\x3d":
            out.append(Token(i, "SIG_USEOFF_CONST_CHAIN", 11, {}))
            i += 11
            continue

        if i + 17 <= size and data[i:i + 17] == b"\x00\x00\x29\x10\x1e\x3c\x4b\x15\x00" + (b"\x7c" * 8):
            out.append(Token(i, "SIG_GETMODIFIERS_PADTAIL", 17, {}))
            i += 17
            continue

        if i + 18 <= size and data[i:i + 18] == b"\xff\x23\x4f\x00\x31\x30\x32\x6c\x01\x08\x00\x00\x00\x14\x00\x00\x00\x72":
            out.append(Token(i, "SIG_GETCASTLENUM_HEAD", 18, {}))
            i += 18
            continue

        if i + 6 <= size and data[i] == 0x39 and data[i + 1] == 0x20:
            bits = u32(data, i + 2)
            value = struct.unpack('<f', struct.pack('<I', bits))[0]
            out.append(Token(i, "F32", 6, {"op": 0x39, "mode": 0x20, "bits": bits, "value": value if math.isfinite(value) else None}))
            i += 6
            continue

        blob_size, blob_payload = _scan_dword_blob(data, i, size)
        if blob_size:
            out.append(Token(i, "DWBLOB", blob_size, blob_payload))
            i += blob_size
            continue

        # compact immediate families inferred from repeated low-coverage wrapper motifs
        if op in SIGNED_IMM24_OPS and i + 4 <= size:
            out.append(Token(i, "IMM24S", 4, {"op": op, "imm": s24(data, i + 1)}))
            i += 4
            continue

        if op in UNSIGNED_IMM24_OPS and i + 4 <= size:
            out.append(Token(i, "IMM24U", 4, {"op": op, "imm": u24(data, i + 1)}))
            i += 4
            continue

        # long / short ref families
        if op in (0x69, 0x65, 0x6C) and i + 6 <= size:
            out.append(Token(i, "REF", 6, {"op": op, "mode": data[i + 1], "ref": u32(data, i + 2)}))
            i += 6
            continue

        if op == 0x64 and i + 4 <= size:
            out.append(Token(i, "REF16", 4, {"op": op, "mode": data[i + 1], "ref": struct.unpack_from('<H', data, i + 2)[0]}))
            i += 4
            continue

        # structured literals
        if op == 0x41 and i + 7 <= size:
            out.append(Token(i, "REC41", 7, {"ref": u32(data, i + 1), "imm": struct.unpack_from("<H", data, i + 5)[0]}))
            i += 7
            continue

        if op == 0x61 and i + 16 <= size:
            out.append(Token(i, "REC61", 16, {
                "mode": data[i + 1],
                "u16": struct.unpack_from("<H", data, i + 2)[0],
                "a": u32(data, i + 4),
                "b": u32(data, i + 8),
                "c": s32(data, i + 12),
            }))
            i += 16
            continue

        if op == 0x62 and i + 8 <= size:
            out.append(Token(i, "REC62", 8, {
                "mode": data[i + 1],
                "u16": struct.unpack_from("<H", data, i + 2)[0],
                "c": s32(data, i + 4),
            }))
            i += 8
            continue

        # call / transfer families
        if op == 0x2C and i + 4 <= size and data[i + 2] == 0x66:
            out.append(Token(i, "CALL66", 4, {"argc": data[i + 1], "opid": data[i + 3]}))
            i += 4
            continue

        if op == 0x2C and i + 7 <= size and data[i + 2] == 0x63:
            out.append(Token(i, "CALL63A", 7, {"argc": data[i + 1], "rel": s32(data, i + 3)}))
            i += 7
            continue

        if op == 0x63 and i + 5 <= size:
            out.append(Token(i, "CALL63B", 5, {"rel": s32(data, i + 1)}))
            i += 5
            continue

        # short imm / branch
        if op == 0x29 and i + 3 <= size and data[i + 1] == 0x10:
            out.append(Token(i, "IMM", 3, {"value": data[i + 2]}))
            i += 3
            continue

        # related compact u16 literal form observed in several short wrappers
        if op == 0x28 and i + 4 <= size and data[i + 1] == 0x10:
            out.append(Token(i, "IMM16", 4, {"op": op, "value": struct.unpack_from('<H', data, i + 2)[0]}))
            i += 4
            continue

        if op in SHORT_U16_OPS and i + 3 <= size:
            out.append(Token(i, "OPU16", 3, {"op": op, "value": struct.unpack_from('<H', data, i + 1)[0]}))
            i += 3
            continue

        if op in (0x4A, 0x4B, 0x4C, 0x4D) and i + 3 <= size:
            out.append(Token(i, "BR", 3, {"op": op, "off": struct.unpack_from("<H", data, i + 1)[0]}))
            i += 3
            continue

        # common compact imm32 family observed in many service-style exports:
        #   30 <imm32-le>
        # A fully-open 30+u32 rule causes regressions because plain 0x30 also occurs
        # as a real single-byte op. A safe expansion is to accept 16-bit immediates
        # encoded as 30 xx yy 00 00 in addition to the old 8-bit-only form.
        if op == 0x30 and i + 5 <= size and data[i + 3] == 0 and data[i + 4] == 0:
            out.append(Token(i, "IMM32", 5, {"op": op, "imm": u32(data, i + 1)}))
            i += 5
            continue

        # pad / filler blocks (often '||||' or '\xff\xff\xff\xff')
        if op in (0x7C, 0xFF):
            j = i
            while j < size and data[j] == op:
                j += 1
            run_len = j - i
            if run_len >= 4:
                out.append(Token(i, "PAD", run_len, {"byte": op, "len": run_len}))
                i = j
                continue

        # ASCII-ish data blocks (format strings, HTML-ish snippets, CSV-like rows, etc.)
        if _is_printable_ascii(op):
            j = _ascii_run(data, i, size)
            run = data[i:j]
            if _looks_like_text(run):
                # Keep payload small; we only need hints for debugging.
                preview = run[:120].decode("latin1", "replace")
                out.append(Token(i, "ASCII", len(run), {"preview": preview}))
                i = j
                continue

        if op in SINGLE_BYTE_OPS:
            out.append(Token(i, "OP", 1, {"op": op}))
            i += 1
            continue

        out.append(Token(i, "UNK", 1, {"op": op}))
        i += 1

    return out


def coverage(tokens: list[Token], total_size: int) -> dict:
    covered = sum(tok.size for tok in tokens if tok.kind != "UNK")
    return {
        "covered_bytes": covered,
        "total_bytes": total_size,
        "coverage_ratio": (covered / total_size) if total_size else 0.0,
        "token_counts": dict(__import__("collections").Counter(tok.kind for tok in tokens)),
    }
