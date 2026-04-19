from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import struct
from collections import Counter
from typing import Optional


KNOWN_FLAGS = {0, 1, 255, 257, 511, 767, 0x101, 0x1FF, 0x201, 0x2FF}
MAGIC_HEADER = b"MBL script v4.0\x00"
FIXED_CODE_BASE = 0x20
MAX_INTER_RECORD_ZERO_RUN = 16


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def s32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def read_cstr(data: bytes, offset: int) -> tuple[Optional[str], int]:
    end = data.find(b"\x00", offset)
    if end == -1:
        return None, offset
    raw = data[offset:end]
    try:
        text = raw.decode("cp1251")
    except Exception:
        text = raw.decode("latin1", "replace")
    return text, end + 1


def _looks_like_filler_name(name: str) -> bool:
    if not name:
        return False
    if len(name) >= 64 and len(set(name)) <= 3:
        return True
    most_common = max(name.count(ch) for ch in set(name))
    if len(name) >= 24 and (most_common / len(name)) > 0.85:
        return True
    # Short low-entropy garbage like "нЮнЮ", "Ю", "Ќ" should never be treated as symbol names.
    if len(name) <= 8 and len(set(name)) <= 2 and all(ord(ch) > 127 for ch in name):
        return True
    return False


def _is_symbolish_name(name: str) -> bool:
    """
    Table names in this corpus are overwhelmingly ASCII-like identifiers.
    Reject short non-ASCII garbage that may arise from filler patterns.
    """
    if not name:
        return False
    if _looks_like_filler_name(name):
        return False
    if any(ord(ch) < 32 for ch in name):
        return False
    if not any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') or ch == '_' for ch in name):
        return False
    return True


@dataclass
class TableRecord:
    offset: int
    name: str
    a: int
    b: int
    c: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableCandidate:
    start: int
    end: int
    kind: str
    score: float
    records: list[TableRecord]

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "score": self.score,
            "records": [r.to_dict() for r in self.records],
        }


@dataclass
class ADBInfo:
    present: bool
    path: str | None
    size: int
    word_count: int
    first_words: list[int]
    last_word: int | None
    delta_signature: list[int]
    top_deltas: list[dict[str, int]]
    exact_signature: str | None
    shape_signature: str | None
    family_signature: str | None
    all_words_strictly_increasing: bool | None
    usable_for_family_shape: bool
    quality: str

    def to_dict(self) -> dict:
        return asdict(self)


EMPTY_ADB_INFO = ADBInfo(
    present=False,
    path=None,
    size=0,
    word_count=0,
    first_words=[],
    last_word=None,
    delta_signature=[],
    top_deltas=[],
    exact_signature=None,
    shape_signature=None,
    family_signature=None,
    all_words_strictly_increasing=None,
    usable_for_family_shape=False,
    quality="missing",
)


def parse_table_with_padding(data: bytes, start: int, globals_mode: bool = False, max_records: int = 5000, max_inter_record_zero_run: int = MAX_INTER_RECORD_ZERO_RUN) -> tuple[list[TableRecord], int]:
    records: list[TableRecord] = []
    pos = start
    size = len(data)
    seen_record = False

    while pos < size and len(records) < max_records:
        zero_start = pos
        while pos < size and data[pos] == 0:
            pos += 1
        if pos >= size:
            break

        zero_run = pos - zero_start
        if seen_record and zero_run > max_inter_record_zero_run:
            break

        if pos > 0 and data[pos - 1] != 0:
            break

        name, end = read_cstr(data, pos)
        if name is None or end + 12 > size:
            break

        a, b, c = struct.unpack_from("<III", data, end)

        if not _is_symbolish_name(name):
            break
        if a > size * 2:
            break

        if globals_mode and not (b == 0xFFFFFFFF and c == 0):
            break

        records.append(TableRecord(pos, name, a, b, c))
        seen_record = True
        pos = end + 12

    return records, pos


def _looks_like_definition(records: list[TableRecord]) -> bool:
    if len(records) < 1:
        return False
    good = 0
    for rec in records:
        if rec.a <= rec.b and rec.c in KNOWN_FLAGS:
            good += 1
    return good >= max(1, int(len(records) * 0.55))


def _looks_like_globals(records: list[TableRecord]) -> bool:
    if len(records) < 1:
        return False
    good = 0
    prev = None
    for r in records:
        if r.b == 0xFFFFFFFF and r.c == 0:
            good += 1
            if prev is not None and r.a - prev == 5:
                good += 1
            prev = r.a
    return good >= max(2, int(len(records) * 1.1))


def _looks_like_exports(records: list[TableRecord]) -> bool:
    if len(records) < 1:
        return False
    # Allow a mostly-normal export stream to contain embedded import-like mini-blocks.
    starts = [r.a for r in records]
    cvals = [r.c for r in records]
    normal = [r for r in records if r.b != 0xFFFFFFFF]
    if not normal:
        return False

    starts_increasing = sum(1 for i in range(len(starts) - 1) if starts[i] < starts[i + 1]) >= max(0, int((len(starts) - 1) * 0.55))
    small_ordinals = sum(1 for r in normal if r.b < 0x10000 and r.c == 0) >= max(1, int(len(normal) * 0.65))
    mostly_zero_c = sum(1 for c in cvals if c == 0) >= max(1, int(len(records) * 0.7))
    return starts_increasing and small_ordinals and mostly_zero_c


def _score_defs(records: list[TableRecord]) -> float:
    score = 0.0
    prev = None
    for r in records[:32]:
        if r.c in KNOWN_FLAGS:
            score += 4
        if r.a <= r.b:
            score += 2
        if prev is not None and r.a >= prev:
            score += 1
        prev = r.a
    return score


def _score_globals(records: list[TableRecord]) -> float:
    score = 0.0
    prev = None
    for r in records[:32]:
        if r.b == 0xFFFFFFFF and r.c == 0:
            score += 5
        if prev is not None and r.a - prev == 5:
            score += 2
        prev = r.a
    return score


def _score_exports(records: list[TableRecord]) -> float:
    score = 0.0
    prev_a = None
    prev_b = None
    for r in records[:32]:
        if r.b != 0xFFFFFFFF and r.c == 0:
            score += 4
        if r.b == 0xFFFFFFFF and r.c == 0:
            score += 1.5
        if prev_a is not None and r.a > prev_a:
            score += 1
        if r.b != 0xFFFFFFFF and prev_b is not None and prev_b != 0xFFFFFFFF and r.b == prev_b + 1:
            score += 2
        prev_a = r.a
        prev_b = r.b
    return score



def _is_strict_definition_record(rec: TableRecord, code_size: int) -> bool:
    return rec.b != 0xFFFFFFFF and rec.a <= rec.b < code_size and rec.c in KNOWN_FLAGS


def _iter_candidate_starts_in_window(data: bytes, start: int, end: int):
    start = max(0, start)
    end = min(len(data), end)
    if start >= end:
        return

    yielded = set()
    if data[start:start + 1] == b"\x00":
        yielded.add(start)
        yield start

    i = max(1, start)
    while i < end:
        if data[i] == 0 and data[i - 1] != 0:
            run_start = i
            j = i
            while j + 1 < end and data[j + 1] == 0:
                j += 1
            next_pos = j + 1
            if next_pos < end:
                b = data[next_pos]
                if ((65 <= b <= 90) or (97 <= b <= 122) or b == 95) and run_start not in yielded:
                    yielded.add(run_start)
                    yield run_start
            i = j + 1
            continue
        i += 1


def _score_header_guided_definition(start: int, records: list[TableRecord], predicted_start: int, code_size: int) -> float:
    valid = sum(1 for rec in records if _is_strict_definition_record(rec, code_size))
    import_like = sum(1 for rec in records if rec.b == 0xFFFFFFFF and rec.c == 0)
    invalid = len(records) - valid - import_like
    if valid == 0:
        return float('-inf')

    score = (valid * 12.0) + (len(records) * 0.5) - (invalid * 30.0) - (import_like * 20.0)
    if records and _is_strict_definition_record(records[0], code_size):
        score += 10.0
    score -= abs(start - predicted_start) * 0.05
    return score


def _find_header_guided_definition(data: bytes) -> TableCandidate | None:
    if len(data) < FIXED_CODE_BASE or not data.startswith(MAGIC_HEADER):
        return None

    header = [u32(data, 0x10 + i * 4) for i in range(4)]
    code_size = header[2]
    predicted_start = FIXED_CODE_BASE + header[2] + header[3] + 1
    search_start = max(0, predicted_start - 128)
    search_end = min(len(data), predicted_start + 512)

    best: TableCandidate | None = None
    best_score = float('-inf')
    for pos in _iter_candidate_starts_in_window(data, search_start, search_end):
        recs, end = parse_table_with_padding(data, pos, globals_mode=False, max_records=5000)
        if not recs:
            continue
        score = _score_header_guided_definition(pos, recs, predicted_start, code_size)
        if score > best_score:
            best_score = score
            best = TableCandidate(pos, end, "definitions", score, recs)

    return best if best_score != float('-inf') else None


def _find_first_table_after(data: bytes, start: int, kind: str) -> TableCandidate | None:
    scan_start = max(0, start - 64)
    scan_end = min(len(data), start + 65536)
    for pos in _iter_candidate_starts_in_window(data, scan_start, scan_end):
        if kind == "globals":
            recs, end = parse_table_with_padding(data, pos, globals_mode=True, max_records=5000)
            if recs and recs[0].offset >= start and _looks_like_globals(recs):
                return TableCandidate(pos, end, "globals", _score_globals(recs), recs)
            continue

        recs, end = parse_table_with_padding(data, pos, globals_mode=False, max_records=5000)
        if recs and recs[0].offset >= start and _looks_like_exports(recs):
            return TableCandidate(pos, end, "exports", _score_exports(recs), recs)
    return None


def _iter_candidate_starts(data: bytes):
    """
    Yield only plausible table starts:
    - file start
    - the first byte of each zero-padding run that is followed by an ASCII-ish identifier start

    The old logic tried nearly every byte offset and relied on parse_table_with_padding()
    to reject almost all of them. That is correct but far too slow on larger modules.
    """
    size = len(data)
    if size == 0:
        return

    yield 0
    i = 1
    while i < size:
        if data[i] == 0 and data[i - 1] != 0:
            run_start = i
            j = i
            while j + 1 < size and data[j + 1] == 0:
                j += 1
            next_pos = j + 1
            if next_pos < size:
                b = data[next_pos]
                if (65 <= b <= 90) or (97 <= b <= 122) or b == 95:
                    yield run_start
            i = j + 1
            continue
        i += 1


def find_table_candidates(data: bytes, min_records: int = 3) -> list[TableCandidate]:
    candidates: list[TableCandidate] = []

    for pos in _iter_candidate_starts(data):
        grecs, gend = parse_table_with_padding(data, pos, globals_mode=True, max_records=500)
        if len(grecs) >= min_records and _looks_like_globals(grecs):
            candidates.append(TableCandidate(pos, gend, "globals", _score_globals(grecs), grecs))

        recs, end = parse_table_with_padding(data, pos, globals_mode=False, max_records=500)
        if len(recs) >= min_records:
            if _looks_like_definition(recs):
                candidates.append(TableCandidate(pos, end, "definitions", _score_defs(recs), recs))
            if _looks_like_exports(recs):
                candidates.append(TableCandidate(pos, end, "exports", _score_exports(recs), recs))

    merged: list[TableCandidate] = []
    for cand in sorted(candidates, key=lambda c: (c.kind, c.start, -c.score)):
        if merged and cand.kind == merged[-1].kind and cand.start - merged[-1].start <= 12:
            if cand.score > merged[-1].score:
                merged[-1] = cand
        else:
            merged.append(cand)
    return merged


def detect_module_layout(data: bytes) -> dict:
    cands = find_table_candidates(data)
    if not cands:
        cands = find_table_candidates(data, min_records=1)

    definitions_best = _find_header_guided_definition(data)
    globals_best = None
    exports_best = None

    if definitions_best is not None:
        first_globals = _find_first_table_after(data, definitions_best.end, "globals")
        first_exports = _find_first_table_after(data, definitions_best.end, "exports")

        if first_globals is not None and first_exports is not None and first_globals.start == first_exports.start:
            later_exports = _find_first_table_after(data, first_globals.end, "exports")
            if later_exports is not None and later_exports.start > first_globals.start:
                globals_best = first_globals
                exports_best = later_exports
            else:
                exports_best = first_exports
        elif first_globals is not None and (first_exports is None or first_globals.start < first_exports.start):
            globals_best = first_globals
            exports_best = _find_first_table_after(data, globals_best.end, "exports")
        else:
            exports_best = first_exports

    if definitions_best is None:
        globals_cands = [c for c in cands if c.kind == "globals"]
        export_cands = [c for c in cands if c.kind == "exports"]
        def_cands = [c for c in cands if c.kind == "definitions"]

        globals_best = max(globals_cands, key=lambda c: (c.score, -c.start), default=None)
        exports_best = max(export_cands, key=lambda c: (c.score, -c.start), default=None)

        if globals_best is not None:
            defs_before = [c for c in def_cands if c.start < globals_best.start]
            if defs_before:
                definitions_best = max(defs_before, key=lambda c: (c.score, -c.start))
        if definitions_best is None and exports_best is not None:
            defs_before = [c for c in def_cands if c.start < exports_best.start]
            if defs_before:
                definitions_best = max(defs_before, key=lambda c: (c.score, -c.start))
        if definitions_best is None and def_cands:
            definitions_best = max(def_cands, key=lambda c: (c.score, -c.start))

    return {
        "definitions": definitions_best,
        "globals": globals_best,
        "exports": exports_best,
        "candidates": cands,
    }


def read_adb_info(mbc_path: str | Path) -> ADBInfo:
    mbc_path = Path(mbc_path)
    adb_path = mbc_path.with_suffix(".adb")
    if not adb_path.exists():
        return EMPTY_ADB_INFO

    data = adb_path.read_bytes()
    exact_signature = hashlib.sha1(data).hexdigest() if data else None
    if not data:
        return ADBInfo(
            present=True,
            path=str(adb_path),
            size=0,
            word_count=0,
            first_words=[],
            last_word=None,
            delta_signature=[],
            top_deltas=[],
            exact_signature=exact_signature,
            shape_signature=None,
            family_signature=None,
            all_words_strictly_increasing=None,
            usable_for_family_shape=False,
            quality="empty",
        )

    if len(data) % 4 != 0:
        return ADBInfo(
            present=True,
            path=str(adb_path),
            size=len(data),
            word_count=0,
            first_words=[],
            last_word=None,
            delta_signature=[],
            top_deltas=[],
            exact_signature=exact_signature,
            shape_signature=None,
            family_signature=None,
            all_words_strictly_increasing=None,
            usable_for_family_shape=False,
            quality="unaligned_bytes",
        )

    words = list(struct.unpack("<" + ("I" * (len(data) // 4)), data))
    increasing = all(words[i] < words[i + 1] for i in range(len(words) - 1)) if words else None
    deltas = [words[i + 1] - words[i] for i in range(len(words) - 1)] if len(words) >= 2 else []
    top_deltas = [
        {"delta": delta, "count": count}
        for delta, count in Counter(deltas).most_common(8)
    ]

    usable_for_family_shape = len(words) >= 2 and bool(deltas)
    quality = "usable" if usable_for_family_shape else "too_short"
    shape_signature = None
    family_signature = None
    if usable_for_family_shape:
        shape_signature = hashlib.sha1((str(len(words)) + '|' + ','.join(str(x) for x in deltas)).encode('ascii')).hexdigest()
        family_signature = hashlib.sha1((str(len(words)) + '|' + ','.join(str(x) for x in deltas[:16])).encode('ascii')).hexdigest()

    return ADBInfo(
        present=True,
        path=str(adb_path),
        size=len(data),
        word_count=len(words),
        first_words=words[:32],
        last_word=words[-1] if words else None,
        delta_signature=deltas[:32],
        top_deltas=top_deltas,
        exact_signature=exact_signature,
        shape_signature=shape_signature,
        family_signature=family_signature,
        all_words_strictly_increasing=increasing,
        usable_for_family_shape=usable_for_family_shape,
        quality=quality,
    )


class MBCModule:
    def __init__(self, path: str | Path, overrides: dict | None = None):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.has_magic_header = self.data.startswith(MAGIC_HEADER)
        self.header = [u32(self.data, 0x10 + i * 4) for i in range(4)] if len(self.data) >= FIXED_CODE_BASE else [0, 0, 0, 0]
        self.code_base = FIXED_CODE_BASE if self.has_magic_header and len(self.data) >= FIXED_CODE_BASE else 0
        self.code_size = self.header[2] if self.has_magic_header else 0
        self.data_blob_size = (self.header[3] + 1) if self.has_magic_header else 0
        self.adb_info = read_adb_info(self.path)

        layout = detect_module_layout(self.data)
        entry = overrides.get(self.path.name) if overrides else None
        if entry:
            if "definitions" in entry:
                dstart = int(entry["definitions"])
                recs, end = parse_table_with_padding(self.data, dstart, globals_mode=False)
                layout["definitions"] = TableCandidate(dstart, end, "definitions", 1e9, recs)
            if "globals" in entry:
                gstart = int(entry["globals"])
                recs, end = parse_table_with_padding(self.data, gstart, globals_mode=True)
                layout["globals"] = TableCandidate(gstart, end, "globals", 1e9, recs)
            if "exports" in entry:
                estart = int(entry["exports"])
                recs, end = parse_table_with_padding(self.data, estart, globals_mode=False)
                layout["exports"] = TableCandidate(estart, end, "exports", 1e9, recs)

        self.definition_table: Optional[TableCandidate] = layout["definitions"]
        self.globals_table: Optional[TableCandidate] = layout["globals"]
        self.exports_table: Optional[TableCandidate] = layout["exports"]
        self.candidates: list[TableCandidate] = layout["candidates"]

        raw_exports = self.exports_table.records if self.exports_table else []
        self.embedded_import_like_exports: list[TableRecord] = [
            r for r in raw_exports if r.b == 0xFFFFFFFF and r.c == 0
        ]
        self.normal_exports: list[TableRecord] = [
            r for r in raw_exports if not (r.b == 0xFFFFFFFF and r.c == 0)
        ]
        self._export_index_by_name = {r.name: idx for idx, r in enumerate(self.normal_exports)}
        self._definition_by_name = {r.name: r for r in self.definitions}

    @property
    def exports(self) -> list[TableRecord]:
        return self.normal_exports

    @property
    def globals(self) -> list[TableRecord]:
        return self.globals_table.records if self.globals_table else []

    @property
    def definitions(self) -> list[TableRecord]:
        return self.definition_table.records if self.definition_table else []

    def export_names(self) -> list[str]:
        return [r.name for r in self.exports]

    def _export_index(self, name: str) -> int:
        idx = self._export_index_by_name.get(name)
        if idx is None:
            raise KeyError(f"Export not found: {name}")
        return idx

    def get_export_record(self, name: str) -> TableRecord:
        return self.exports[self._export_index(name)]

    def get_definition_record(self, name: str) -> Optional[TableRecord]:
        return self._definition_by_name.get(name)

    def get_export_public_code_span(self, name: str) -> tuple[int, int]:
        idx = self._export_index(name)
        start = self.exports[idx].a
        if idx + 1 < len(self.exports):
            end = self.exports[idx + 1].a
        elif self.code_size:
            end = self.code_size
        else:
            end = max(start, len(self.data) - self.code_base)
        return start, end

    def get_export_exact_code_span(self, name: str) -> Optional[tuple[int, int]]:
        rec = self.get_definition_record(name)
        if rec is None or rec.b < rec.a:
            return None
        if self.code_size and rec.b + 1 > self.code_size:
            return None
        return rec.a, rec.b + 1

    def _slice_code_span(self, start: int, end: int) -> bytes:
        file_start = self.code_base + start
        file_end = self.code_base + end
        code_limit = self.code_base + (self.code_size or max(0, len(self.data) - self.code_base))
        if file_start < 0 or file_end > min(code_limit, len(self.data)) or file_start >= file_end:
            return b""
        return self.data[file_start:file_end]

    def get_export_body(self, name: str, exact: bool = True) -> bytes:
        span = self.get_export_exact_code_span(name) if exact else None
        if span is None:
            span = self.get_export_public_code_span(name)
        start, end = span
        return self._slice_code_span(start, end)

    def stitch_export_body(self, name: str, next_head_bytes: int = 16) -> bytes:
        idx = self._export_index(name)
        body = self.get_export_body(name, exact=True)
        if next_head_bytes > 0 and idx + 1 < len(self.exports):
            body += self.get_export_body(self.exports[idx + 1].name, exact=True)[:next_head_bytes]
        return body

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "header": self.header,
            "has_magic_header": self.has_magic_header,
            "definition_table_start": self.definition_table.start if self.definition_table else None,
            "globals_table_start": self.globals_table.start if self.globals_table else None,
            "exports_table_start": self.exports_table.start if self.exports_table else None,
            "definitions": [r.to_dict() for r in self.definitions],
            "globals": [r.to_dict() for r in self.globals],
            "exports": [r.to_dict() for r in self.exports],
            "embedded_import_like_exports": [r.to_dict() for r in self.embedded_import_like_exports],
            "adb_info": self.adb_info.to_dict(),
        }
