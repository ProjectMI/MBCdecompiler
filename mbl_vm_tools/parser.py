from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
import struct
from collections import Counter, defaultdict
from typing import Optional


KNOWN_FLAGS = {0, 1, 255, 257, 511, 767, 0x101, 0x1FF, 0x201, 0x2FF}
MAGIC_HEADER = b"MBL script v4.0\x00"
FIXED_CODE_BASE = 0x20
MAX_INTER_RECORD_ZERO_RUN = 16
SYMBOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{1,63}$")
PARSER_CONTRACT_VERSION = "mbc-parser-v2"
TABLE_ANCHOR_SYNC_WINDOW = 32



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
    if len(name) <= 8 and len(set(name)) <= 2 and all(ord(ch) > 127 for ch in name):
        return True
    return False


def _is_symbolish_name(name: str) -> bool:
    """
    Table names in this corpus are overwhelmingly ASCII identifiers.
    Be intentionally strict here: false negatives are cheaper than accepting
    header strings / magic text as pseudo-symbols and derailing layout
    detection.
    """
    if not name:
        return False
    if _looks_like_filler_name(name):
        return False
    if any(ord(ch) < 32 for ch in name):
        return False
    if name.startswith("MBL script v"):
        return False
    return bool(SYMBOL_NAME_RE.fullmatch(name))


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
class FunctionEntry:
    """
    Stable analysis entry point built from definitions and exports.

    Definitions carry exact code spans; exports carry public API order.  The
    VM IR pipeline consumes this de-duplicated view so default corpus runs can
    cover local definitions without doing the same exported definition twice.
    """

    name: str
    symbol: str
    source_kind: str
    definition_index: Optional[int]
    export_index: Optional[int]
    is_definition: bool
    is_exported: bool
    duplicate_definition_symbol: bool
    duplicate_export_symbol: bool
    definition_record: Optional[TableRecord]
    export_record: Optional[TableRecord]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "source_kind": self.source_kind,
            "definition_index": self.definition_index,
            "export_index": self.export_index,
            "is_definition": self.is_definition,
            "is_exported": self.is_exported,
            "duplicate_definition_symbol": self.duplicate_definition_symbol,
            "duplicate_export_symbol": self.duplicate_export_symbol,
            "definition_record": self.definition_record.to_dict() if self.definition_record else None,
            "export_record": self.export_record.to_dict() if self.export_record else None,
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



def _is_definition_record(rec: TableRecord, code_size: int) -> bool:
    """Authoritative definition-table record predicate.

    Definition records address a closed byte range inside the code segment and
    carry one of the observed VM definition flags.  This is a hard table-shape
    predicate, not a score.
    """

    return rec.b != 0xFFFFFFFF and rec.a <= rec.b < code_size and rec.c in KNOWN_FLAGS


def _is_global_record(rec: TableRecord) -> bool:
    """Authoritative globals/import record predicate."""

    return rec.b == 0xFFFFFFFF and rec.c == 0


def _is_export_record(rec: TableRecord) -> bool:
    """Authoritative export-table record predicate.

    Export tables contain normal public exports (``b`` is the public ordinal)
    and may also embed import-like extern records.  Both carry ``c == 0``.
    """

    if rec.c != 0:
        return False
    if rec.b == 0xFFFFFFFF:
        return True
    return 0 <= rec.b < 0x10000


def _is_normal_export_record(rec: TableRecord) -> bool:
    return rec.c == 0 and rec.b != 0xFFFFFFFF and 0 <= rec.b < 0x10000


def _record_at(data: bytes, offset: int) -> TableRecord | None:
    name, end = read_cstr(data, offset)
    if name is None or end + 12 > len(data):
        return None
    if not _is_symbolish_name(name):
        return None
    a, b, c = struct.unpack_from("<III", data, end)
    return TableRecord(offset, name, a, b, c)


def _find_record_start(data: bytes, start: int, *, limit: int | None = None) -> tuple[int, int] | None:
    """Find the next NUL-padded table record start in a bounded window.

    Returns ``(padding_start, record_name_offset)``.  The bounded sync is used
    only to cross deterministic table-boundary padding / one-byte sentinels near
    the header-predicted table anchor.  It does not rank candidates.
    """

    size = len(data)
    stop = min(size, start + TABLE_ANCHOR_SYNC_WINDOW if limit is None else limit)
    pos = max(0, start)
    while pos < stop:
        if data[pos] == 0:
            pad_start = pos
            while pos < size and data[pos] == 0:
                pos += 1
            if pos < size and pos < stop:
                rec = _record_at(data, pos)
                if rec is not None:
                    return pad_start, pos
            continue
        # A non-zero byte immediately before table padding is allowed at the
        # header/table seam, but only inside this bounded sync window.
        pos += 1
    return None


def _parse_strict_table(
    data: bytes,
    start: int,
    *,
    kind: str,
    predicate,
    max_records: int = 5000,
    max_inter_record_zero_run: int = MAX_INTER_RECORD_ZERO_RUN,
    sync_first_record: bool = True,
) -> TableCandidate | None:
    """Parse one table by a hard per-record predicate.

    The table starts at the NUL padding before the first record.  Records may be
    separated by short NUL padding runs; a longer run ends the table.  Encountering
    a syntactically valid record that does not satisfy the table predicate also
    ends the table without consuming that next table's record.
    """

    size = len(data)
    if start >= size:
        return None

    if sync_first_record:
        found = _find_record_start(data, start)
        if found is None:
            return None
        table_start, pos = found
    else:
        table_start = start
        pos = start

    records: list[TableRecord] = []
    while pos < size and len(records) < max_records:
        zero_start = pos
        while pos < size and data[pos] == 0:
            pos += 1
        zero_run = pos - zero_start
        if records and zero_run > max_inter_record_zero_run:
            return TableCandidate(table_start, zero_start, kind, 1.0, records)
        if pos >= size:
            break
        if pos > 0 and data[pos - 1] != 0:
            break
        rec = _record_at(data, pos)
        if rec is None:
            break
        if not predicate(rec):
            return TableCandidate(table_start, zero_start, kind, 1.0, records) if records else None
        records.append(rec)
        name_end = data.find(b"\x00", pos)
        pos = name_end + 1 + 12

    if not records:
        return None
    return TableCandidate(table_start, pos, kind, 1.0, records)


def _next_table_anchor(data: bytes, start: int) -> int | None:
    found = _find_record_start(data, start)
    return found[0] if found is not None else None


def _header_table_anchor(data: bytes) -> int | None:
    if len(data) < FIXED_CODE_BASE or not data.startswith(MAGIC_HEADER):
        return None
    header = [u32(data, 0x10 + i * 4) for i in range(4)]
    code_size = int(header[2])
    data_blob_size = int(header[3]) + 1
    raw_anchor = FIXED_CODE_BASE + code_size + data_blob_size
    if raw_anchor > len(data):
        return None
    synced = _find_record_start(data, raw_anchor)
    if synced is None:
        return None
    return synced[0]


def _detect_module_layout_deterministic(data: bytes) -> dict:
    """Detect MBC table layout from header and strict table predicates.

    Model:
    - The header fixes the end of code + data blob and therefore the table area.
    - The first table in that area is the definition table.
    - After definitions there may be a globals/import table.
    - The export table is the next table that contains at least one normal export
      record; import-like records inside that same stream remain embedded externs.

    No score, majority vote, whole-file sweep, or candidate ranking is used.
    """

    diagnostics: dict[str, object] = {
        "contract": PARSER_CONTRACT_VERSION,
        "policy": "header anchored deterministic layout; strict record predicates; no fallback sweep or scoring",
    }
    anchor = _header_table_anchor(data)
    diagnostics["header_table_anchor"] = anchor
    if anchor is None:
        return {"definitions": None, "globals": None, "exports": None, "candidates": [], "diagnostics": diagnostics}

    header = [u32(data, 0x10 + i * 4) for i in range(4)]
    code_size = int(header[2])
    raw_anchor = FIXED_CODE_BASE + code_size + int(header[3]) + 1
    diagnostics["raw_header_anchor"] = raw_anchor
    diagnostics["anchor_sync_delta"] = int(anchor) - int(raw_anchor)

    definitions = _parse_strict_table(
        data,
        anchor,
        kind="definitions",
        predicate=lambda rec: _is_definition_record(rec, code_size),
        sync_first_record=True,
    )
    if definitions is None:
        diagnostics["failure"] = "definition_table_missing"
        return {"definitions": None, "globals": None, "exports": None, "candidates": [], "diagnostics": diagnostics}

    diagnostics["definition_count"] = len(definitions.records)
    pos = definitions.end
    globals_table = None
    exports_table = None

    first_anchor = _next_table_anchor(data, pos)
    diagnostics["post_definition_anchor"] = first_anchor
    if first_anchor is not None:
        first_found = _find_record_start(data, first_anchor)
        first_record = _record_at(data, first_found[1]) if first_found is not None else None
        if first_record is not None and _is_global_record(first_record):
            globals_table = _parse_strict_table(
                data,
                first_anchor,
                kind="globals",
                predicate=_is_global_record,
                sync_first_record=True,
            )
            # In this format the zero-valued ``c`` field of the final
            # globals record doubles as the NUL padding before the first export
            # name.  Therefore the export-table anchor is four bytes before the
            # end of the globals stream, not after it.
            second_anchor = max(first_anchor, globals_table.end - 4) if globals_table else None
            diagnostics["post_globals_anchor"] = second_anchor
            if second_anchor is not None:
                exports_table = _parse_strict_table(
                    data,
                    second_anchor,
                    kind="exports",
                    predicate=_is_export_record,
                    sync_first_record=True,
                )
        else:
            exports_table = _parse_strict_table(
                data,
                first_anchor,
                kind="exports",
                predicate=_is_export_record,
                sync_first_record=True,
            )

        if exports_table is not None and not any(_is_normal_export_record(rec) for rec in exports_table.records):
            # A table containing only import-like records is not a public export
            # table.  Keep the role explicit instead of guessing.
            diagnostics["export_rejected"] = "no_normal_export_records"
            exports_table = None

    diagnostics["globals_count"] = len(globals_table.records) if globals_table else 0
    diagnostics["export_record_count"] = len(exports_table.records) if exports_table else 0
    diagnostics["normal_export_count"] = sum(1 for rec in exports_table.records if _is_normal_export_record(rec)) if exports_table else 0
    return {
        "definitions": definitions,
        "globals": globals_table,
        "exports": exports_table,
        "candidates": [],
        "diagnostics": diagnostics,
    }


def detect_module_layout(data: bytes, *, collect_candidates: bool = True, allow_fallback_scan: bool = True) -> dict:
    """Return deterministic MBC table layout.

    ``collect_candidates`` and ``allow_fallback_scan`` are accepted for API
    compatibility only.  Parser v2 intentionally performs no candidate sweep,
    score ranking, or fallback scan.
    """

    return _detect_module_layout_deterministic(data)


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
    def __init__(self, path: str | Path, collect_auxiliary: bool = True):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.has_magic_header = self.data.startswith(MAGIC_HEADER)
        self.header = [u32(self.data, 0x10 + i * 4) for i in range(4)] if len(self.data) >= FIXED_CODE_BASE else [0, 0, 0, 0]
        self.code_base = FIXED_CODE_BASE if self.has_magic_header and len(self.data) >= FIXED_CODE_BASE else 0
        self.code_size = self.header[2] if self.has_magic_header else 0
        self.data_blob_size = (self.header[3] + 1) if self.has_magic_header else 0
        self.adb_info = read_adb_info(self.path) if collect_auxiliary else EMPTY_ADB_INFO

        layout = detect_module_layout(
            self.data,
            collect_candidates=collect_auxiliary,
            allow_fallback_scan=False,
        )

        self.parser_contract = PARSER_CONTRACT_VERSION
        self.layout_diagnostics: dict = layout.get("diagnostics", {})
        self.definition_table: Optional[TableCandidate] = layout["definitions"]
        self.globals_table: Optional[TableCandidate] = layout["globals"]
        self.exports_table: Optional[TableCandidate] = layout["exports"]
        self.candidates: list[TableCandidate] = []

        raw_exports = self.exports_table.records if self.exports_table else []
        self.embedded_import_like_exports: list[TableRecord] = [
            r for r in raw_exports if r.b == 0xFFFFFFFF and r.c == 0
        ]
        self.normal_exports: list[TableRecord] = [
            r for r in raw_exports if not (r.b == 0xFFFFFFFF and r.c == 0)
        ]

        self._export_index_by_name: dict[str, int] = {}
        self._export_records_by_name: dict[str, list[tuple[int, TableRecord]]] = defaultdict(list)
        for idx, record in enumerate(self.normal_exports):
            self._export_index_by_name.setdefault(record.name, idx)
            self._export_records_by_name[record.name].append((idx, record))

        self._definition_records_by_name: dict[str, list[TableRecord]] = defaultdict(list)
        self._definition_indices_by_name: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(self.definitions):
            self._definition_records_by_name[record.name].append(record)
            self._definition_indices_by_name[record.name].append(idx)

        self.definition_name_collisions: dict[str, list[TableRecord]] = {
            name: records
            for name, records in self._definition_records_by_name.items()
            if len(records) > 1
        }
        self.export_name_collisions: dict[str, list[TableRecord]] = {
            name: [record for _, record in records]
            for name, records in self._export_records_by_name.items()
            if len(records) > 1
        }
        self._definition_by_name = {
            name: records[0]
            for name, records in self._definition_records_by_name.items()
            if len(records) == 1
        }

        self._definition_function_entries: list[FunctionEntry] = []
        self._export_function_entries: list[FunctionEntry] = []
        self._function_entries: list[FunctionEntry] = []
        self._function_entry_by_name: dict[str, FunctionEntry] = {}
        self._build_function_entries()

    def _build_function_entries(self) -> None:
        definition_occurrences: Counter[str] = Counter()
        export_occurrences: Counter[str] = Counter()

        for definition_index, record in enumerate(self.definitions):
            definition_occurrences[record.name] += 1
            duplicate_definition_symbol = len(self._definition_records_by_name.get(record.name, [])) > 1
            export_refs = self._export_records_by_name.get(record.name, [])
            duplicate_export_symbol = len(export_refs) > 1
            export_index = export_refs[0][0] if export_refs and not duplicate_export_symbol else None
            export_record = export_refs[0][1] if export_refs and not duplicate_export_symbol else None
            unique_name = (
                record.name
                if not duplicate_definition_symbol
                else f"{record.name}#def{definition_occurrences[record.name]}"
            )
            entry = FunctionEntry(
                name=unique_name,
                symbol=record.name,
                source_kind="definition",
                definition_index=definition_index,
                export_index=export_index,
                is_definition=True,
                is_exported=bool(export_refs),
                duplicate_definition_symbol=duplicate_definition_symbol,
                duplicate_export_symbol=duplicate_export_symbol,
                definition_record=record,
                export_record=export_record,
            )
            self._definition_function_entries.append(entry)
            self._function_entries.append(entry)

        definition_symbols = set(self._definition_records_by_name.keys())
        for export_index, record in enumerate(self.normal_exports):
            export_occurrences[record.name] += 1
            duplicate_export_symbol = len(self._export_records_by_name.get(record.name, [])) > 1
            duplicate_definition_symbol = len(self._definition_records_by_name.get(record.name, [])) > 1
            unique_name = (
                record.name
                if not duplicate_export_symbol
                else f"{record.name}#export{export_occurrences[record.name]}"
            )
            export_entry = FunctionEntry(
                name=unique_name,
                symbol=record.name,
                source_kind="export",
                definition_index=None,
                export_index=export_index,
                is_definition=False,
                is_exported=True,
                duplicate_definition_symbol=duplicate_definition_symbol,
                duplicate_export_symbol=duplicate_export_symbol,
                definition_record=None,
                export_record=record,
            )
            self._export_function_entries.append(export_entry)

            if record.name not in definition_symbols:
                default_name = unique_name
                if default_name in self._function_entry_by_name or any(e.name == default_name for e in self._function_entries):
                    default_name = f"{record.name}#export{export_index + 1}"
                    export_entry = FunctionEntry(
                        name=default_name,
                        symbol=record.name,
                        source_kind="export",
                        definition_index=None,
                        export_index=export_index,
                        is_definition=False,
                        is_exported=True,
                        duplicate_definition_symbol=duplicate_definition_symbol,
                        duplicate_export_symbol=True,
                        definition_record=None,
                        export_record=record,
                    )
                self._function_entries.append(export_entry)

        for entry in self._function_entries:
            self._function_entry_by_name.setdefault(entry.name, entry)
        for entry in self._export_function_entries:
            self._function_entry_by_name.setdefault(entry.name, entry)

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

    def function_entries(
        self,
        *,
        include_definitions: bool = True,
        include_exports: bool = True,
        dedupe: bool = True,
    ) -> list[FunctionEntry]:
        if include_definitions and include_exports and dedupe:
            return list(self._function_entries)
        entries: list[FunctionEntry] = []
        if include_definitions:
            entries.extend(self._definition_function_entries)
        if include_exports:
            if include_definitions and dedupe:
                definition_symbols = {entry.symbol for entry in self._definition_function_entries}
                entries.extend(entry for entry in self._export_function_entries if entry.symbol not in definition_symbols)
            else:
                entries.extend(self._export_function_entries)
        return list(entries)

    def function_names(
        self,
        *,
        include_definitions: bool = True,
        include_exports: bool = True,
        dedupe: bool = True,
    ) -> list[str]:
        return [
            entry.name
            for entry in self.function_entries(
                include_definitions=include_definitions,
                include_exports=include_exports,
                dedupe=dedupe,
            )
        ]

    def get_function_entry(self, name: str) -> FunctionEntry:
        entry = self._function_entry_by_name.get(name)
        if entry is not None:
            return entry

        records = self._definition_records_by_name.get(name)
        if records and len(records) == 1:
            for candidate in self._definition_function_entries:
                if candidate.symbol == name:
                    return candidate
        raise KeyError(f"Function entry not found: {name}")

    def _export_index(self, name: str) -> int:
        idx = self._export_index_by_name.get(name)
        if idx is None:
            raise KeyError(f"Export not found: {name}")
        return idx

    def get_export_record(self, name: str) -> TableRecord:
        return self.exports[self._export_index(name)]

    def get_definition_record(self, name: str) -> Optional[TableRecord]:
        return self._definition_by_name.get(name)

    def get_definition_records(self, name: str) -> list[TableRecord]:
        return list(self._definition_records_by_name.get(name, []))

    def get_real_code_size(self) -> int:
        real_limit = max(0, len(self.data) - self.code_base)
        if self.code_size:
            return min(self.code_size, real_limit)
        return real_limit

    def get_definition_record_code_span_with_reason(self, rec: TableRecord) -> tuple[Optional[tuple[int, int]], Optional[str]]:
        if rec.b < rec.a:
            return None, "definition_inverted"

        code_limit = self.get_real_code_size()
        start = max(0, rec.a)
        end = rec.b + 1
        if start >= code_limit:
            return None, "definition_start_oob"
        if end > code_limit:
            return None, "definition_end_oob"
        if end <= start:
            return None, "definition_empty"
        return (start, end), None

    def get_function_exact_code_span_with_reason(self, entry: FunctionEntry | str) -> tuple[Optional[tuple[int, int]], Optional[str]]:
        if isinstance(entry, str):
            entry = self.get_function_entry(entry)
        if entry.definition_record is None:
            if entry.symbol in self._definition_records_by_name:
                return self.get_export_exact_code_span_with_reason(entry.symbol)
            return None, "definition_missing"
        return self.get_definition_record_code_span_with_reason(entry.definition_record)

    def get_export_exact_code_span_with_reason(self, name: str) -> tuple[Optional[tuple[int, int]], Optional[str]]:
        definition_records = self.get_definition_records(name)
        if not definition_records:
            return None, "definition_missing"
        if len(definition_records) > 1:
            return None, "definition_ambiguous"

        return self.get_definition_record_code_span_with_reason(definition_records[0])

    def get_export_public_code_span(self, name: str) -> tuple[int, int]:
        idx = self._export_index(name)
        code_limit = self.get_real_code_size()
        start = max(0, min(self.exports[idx].a, code_limit))
        if idx + 1 < len(self.exports):
            end = self.exports[idx + 1].a
        else:
            end = code_limit
        end = max(start, min(end, code_limit))
        return start, end

    def get_export_exact_code_span(self, name: str) -> Optional[tuple[int, int]]:
        span, _ = self.get_export_exact_code_span_with_reason(name)
        return span

    def _slice_code_span(self, start: int, end: int) -> bytes:
        file_start = self.code_base + start
        file_end = self.code_base + end
        code_limit = self.code_base + self.get_real_code_size()
        if file_start < 0 or file_end > min(code_limit, len(self.data)) or file_start >= file_end:
            return b""
        return self.data[file_start:file_end]

    def get_export_body(self, name: str, exact: bool = True) -> bytes:
        span = self.get_export_exact_code_span(name) if exact else None
        if span is None:
            span = self.get_export_public_code_span(name)
        start, end = span
        return self._slice_code_span(start, end)

    def get_function_body(self, entry: FunctionEntry | str) -> bytes:
        if isinstance(entry, str):
            entry = self.get_function_entry(entry)
        if entry.source_kind == "export" and entry.definition_record is None:
            start, end = self.get_export_public_code_span(entry.symbol)
            return self._slice_code_span(start, end)
        span, _ = self.get_function_exact_code_span_with_reason(entry)
        if span is None:
            return b""
        return self._slice_code_span(*span)

    def stitch_export_body(self, name: str, next_head_bytes: int = 16) -> bytes:
        idx = self._export_index(name)
        body = self.get_export_body(name, exact=True)
        if next_head_bytes > 0 and idx + 1 < len(self.exports):
            body += self.get_export_body(self.exports[idx + 1].name, exact=True)[:next_head_bytes]
        return body

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "parser_contract": self.parser_contract,
            "layout_diagnostics": self.layout_diagnostics,
            "header": self.header,
            "has_magic_header": self.has_magic_header,
            "definition_table_start": self.definition_table.start if self.definition_table else None,
            "globals_table_start": self.globals_table.start if self.globals_table else None,
            "exports_table_start": self.exports_table.start if self.exports_table else None,
            "definitions": [r.to_dict() for r in self.definitions],
            "globals": [r.to_dict() for r in self.globals],
            "exports": [r.to_dict() for r in self.exports],
            "function_entries": [entry.to_dict() for entry in self._function_entries],
            "embedded_import_like_exports": [r.to_dict() for r in self.embedded_import_like_exports],
            "definition_name_collisions": {
                name: [record.to_dict() for record in records]
                for name, records in self.definition_name_collisions.items()
            },
            "export_name_collisions": {
                name: [record.to_dict() for record in records]
                for name, records in self.export_name_collisions.items()
            },
            "adb_info": self.adb_info.to_dict(),
        }

