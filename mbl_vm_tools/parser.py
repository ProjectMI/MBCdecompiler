from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import struct
from collections import Counter
from typing import Optional


KNOWN_FLAGS = {0, 1, 255, 257, 511, 767, 0x101, 0x1FF, 0x201, 0x2FF}
MAGIC_HEADER = b"MBL script v4.0\x00"


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
)


def parse_table_with_padding(data: bytes, start: int, globals_mode: bool = False, max_records: int = 5000) -> tuple[list[TableRecord], int]:
    records: list[TableRecord] = []
    pos = start
    size = len(data)

    while pos < size and len(records) < max_records:
        while pos < size and data[pos] == 0:
            pos += 1
        if pos >= size:
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
    # Extremely small service modules can contain a legitimate one-record table.
    # Keep the default min_records=3 for normal operation, but fall back only when
    # the strict pass finds nothing at all.
    if not cands:
        cands = find_table_candidates(data, min_records=1)

    globals_cands = [c for c in cands if c.kind == "globals"]
    export_cands = [c for c in cands if c.kind == "exports"]
    def_cands = [c for c in cands if c.kind == "definitions"]

    globals_best = max(globals_cands, key=lambda c: (c.score, -c.start), default=None)
    exports_best = max(export_cands, key=lambda c: (c.score, -c.start), default=None)

    definitions_best = None
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
    words: list[int] = []
    if len(data) % 4 == 0 and data:
        words = list(struct.unpack("<" + ("I" * (len(data) // 4)), data))

    increasing = None
    if words:
        increasing = all(words[i] < words[i + 1] for i in range(len(words) - 1))

    deltas = [words[i + 1] - words[i] for i in range(len(words) - 1)] if len(words) >= 2 else []
    top_deltas = [
        {"delta": delta, "count": count}
        for delta, count in Counter(deltas).most_common(8)
    ]

    return ADBInfo(
        present=True,
        path=str(adb_path),
        size=len(data),
        word_count=len(words),
        first_words=words[:32],
        last_word=words[-1] if words else None,
        delta_signature=deltas[:32],
        top_deltas=top_deltas,
        exact_signature=hashlib.sha1(data).hexdigest(),
        shape_signature=hashlib.sha1((str(len(words)) + '|' + ','.join(str(x) for x in deltas)).encode('ascii')).hexdigest(),
        family_signature=hashlib.sha1((str(len(words)) + '|' + ','.join(str(x) for x in deltas[:16])).encode('ascii')).hexdigest(),
        all_words_strictly_increasing=increasing,
    )


class MBCModule:
    def __init__(self, path: str | Path, overrides: dict | None = None):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.header = [u32(self.data, 0x10 + i * 4) for i in range(4)]
        self.has_magic_header = self.data.startswith(MAGIC_HEADER)
        self.adb_info = read_adb_info(self.path)

        layout = {"definitions": None, "globals": None, "exports": None, "candidates": []}
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
        else:
            layout = detect_module_layout(self.data)

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

    def header_overlap_exports(self, header_size: int = len(MAGIC_HEADER)) -> list[TableRecord]:
        return [r for r in self.exports if r.a < header_size]

    def get_export_body(self, name: str) -> bytes:
        names = self.export_names()
        if name not in names:
            raise KeyError(f"Export not found: {name}")
        idx = names.index(name)
        start = self.exports[idx].a
        if idx + 1 < len(self.exports):
            end = self.exports[idx + 1].a
        elif self.definition_table is not None:
            end = self.definition_table.start
        elif self.globals_table is not None:
            end = self.globals_table.start
        else:
            end = len(self.data)
        return self.data[start:end]

    def stitch_export_body(self, name: str, next_head_bytes: int = 16) -> bytes:
        names = self.export_names()
        idx = names.index(name)
        body = self.get_export_body(name)
        if idx + 1 < len(self.exports):
            body += self.get_export_body(names[idx + 1])[:next_head_bytes]
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
