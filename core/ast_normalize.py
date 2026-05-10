from __future__ import annotations

"""Normalization helpers for the experimental stack-AST layer.

The builder is intentionally conservative and emits low-level metadata that is
useful while debugging the VM model.  This module prepares the AST for the
pseudo-source view: it removes noisy metadata comments and adds visible labels
for branch targets so `goto` statements have concrete landing points.
"""

import re
from typing import Any, Iterable

from .ast_model import AstStatement
from .opcodes import CODE_FILE_OFFSET


_BRANCH_KINDS = {"goto", "if_goto"}
_ARGC_META_PREFIX = "// argc ="
_TYPE_COMMENT_RE = re.compile(
    r"\s*/\*\s*(?:char|string|int32|int32_ref_or_span|float|float_ref|span/string|slice_descriptor|type_\d+)\s*\*/"
)


def label_for_offset(offset: Any) -> str:
    """Return the canonical pseudo-source label for a code offset."""
    if isinstance(offset, int):
        return f"loc_{offset:08X}"
    return "loc_UNKNOWN"


def normalize_ast_statements(statements: Iterable[AstStatement]) -> list[AstStatement]:
    """Return a source-oriented AST statement list.

    Normalization currently does three small things:
    * drops transient `set_arg_count` comments (`// argc = N`);
    * removes inline comments that only repeat scalar MBC type names;
    * inserts `loc_XXXXXXXX:` labels before visible branch targets.

    Labels are inserted before the first emitted statement at the target offset.
    If the exact target instruction was suppressed as metadata/no-op, the label
    is placed before the next visible statement, which keeps the generated text
    navigable without reintroducing low-level VM noise.
    """
    cleaned = [_normalize_statement(stmt) for stmt in statements if not _is_argc_meta(stmt)]
    return _insert_branch_labels(cleaned)


def _is_argc_meta(stmt: AstStatement) -> bool:
    return stmt.kind == "meta" and stmt.text.strip().startswith(_ARGC_META_PREFIX)


def _normalize_statement(stmt: AstStatement) -> AstStatement:
    text = _TYPE_COMMENT_RE.sub("", stmt.text)
    if text == stmt.text:
        return stmt
    return AstStatement(
        offset=stmt.offset,
        file_offset=stmt.file_offset,
        kind=stmt.kind,
        text=text,
        opcode=stmt.opcode,
        mnemonic=stmt.mnemonic,
        operands=dict(stmt.operands or {}),
    )


def _insert_branch_labels(statements: list[AstStatement]) -> list[AstStatement]:
    targets = sorted(
        {
            stmt.operands.get("target")
            for stmt in statements
            if stmt.kind in _BRANCH_KINDS and isinstance(stmt.operands.get("target"), int)
        }
    )
    if not targets:
        return statements

    result: list[AstStatement] = []
    emitted: set[int] = set()

    for stmt in statements:
        for target in targets:
            if target in emitted:
                continue
            if target <= stmt.offset:
                result.append(_make_label_statement(target))
                emitted.add(target)
        result.append(stmt)

    for target in targets:
        if target not in emitted:
            result.append(_make_label_statement(target, unresolved=True))
            emitted.add(target)

    return result


def _make_label_statement(offset: int, *, unresolved: bool = False) -> AstStatement:
    suffix = " // target outside emitted statement stream" if unresolved else ""
    return AstStatement(
        offset=offset,
        file_offset=offset + CODE_FILE_OFFSET,
        kind="label",
        text=f"{label_for_offset(offset)}:{suffix}",
        opcode=-1,
        mnemonic="label",
        operands={"target": offset},
    )
