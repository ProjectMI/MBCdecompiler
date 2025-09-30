"""High level Lua reconstruction leveraging manual annotations and stack heuristics."""

from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ir import IRBlock, IRInstruction, IRProgram
from .knowledge import KnowledgeBase
from .lua_ast import (
    Assignment,
    BlankLine,
    BlockStatement,
    BinaryExpr,
    CallExpr,
    CallStatement,
    CommentStatement,
    IfClause,
    IfStatement,
    LuaExpression,
    LiteralExpr,
    MethodCallExpr,
    MultiAssignment,
    NameExpr,
    LuaStatement,
    ReturnStatement,
    SwitchCase,
    SwitchStatement,
    TableExpr,
    TableField,
    WhileStatement,
    wrap_block,
)
from .lua_formatter import (
    CommentFormatter,
    EnumRegistry,
    HelperRegistry,
    HelperSignature,
    LuaRenderOptions,
    LuaWriter,
    MethodSignature,
    render_lua_value,
    join_sections,
)
from .manual_semantics import InstructionSemantics
from .literal_sequences import (
    LiteralRun,
    LiteralRunTracker,
    LiteralRunReport,
    LiteralStatistics,
    build_literal_run_report,
    compute_literal_statistics,
)
from .lua_literals import LuaLiteral, LuaLiteralFormatter, escape_lua_string
from .vm_analysis import estimate_stack_io


class HighLevelStack:
    """Track symbolic stack values during reconstruction."""

    def __init__(self) -> None:
        self._values: List[LuaExpression] = []
        self._counter = 0
        self.warnings: List[str] = []
        self.placeholders: List[str] = []

    def new_symbol(self, prefix: str = "value") -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def push_literal(
        self, expression: LuaExpression
    ) -> Tuple[List[LuaStatement], LuaExpression]:
        self._values.append(expression)
        return [], expression

    def push_expression(
        self,
        expression: LuaExpression,
        *,
        prefix: str = "tmp",
        make_local: bool = False,
    ) -> Tuple[List[LuaStatement], LuaExpression]:
        if make_local:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            statement = Assignment([target], expression, is_local=True)
            self._values.append(target)
            return [statement], target
        if isinstance(expression, NameExpr):
            self._values.append(expression)
            return [], expression
        name = self.new_symbol(prefix)
        target = NameExpr(name)
        statement = Assignment([target], expression, is_local=True)
        self._values.append(target)
        return [statement], target

    def push_call_results(
        self, expression: CallExpr, outputs: int, prefix: str = "result"
    ) -> Tuple[List[LuaStatement], List[NameExpr]]:
        if outputs <= 0:
            return [], []
        if outputs == 1:
            name = self.new_symbol(prefix)
            target = NameExpr(name)
            stmt = Assignment([target], expression, is_local=True)
            self._values.append(target)
            return [stmt], [target]
        targets = [NameExpr(self.new_symbol(prefix)) for _ in range(outputs)]
        stmt = MultiAssignment(targets, [expression], is_local=True)
        for target in targets:
            self._values.append(target)
        return [stmt], targets

    def pop_single(self) -> LuaExpression:
        if self._values:
            return self._values.pop()
        placeholder = NameExpr(self.new_symbol("stack"))
        self.warnings.append(
            f"underflow generated placeholder {placeholder.name}"
        )
        self.placeholders.append(placeholder.name)
        return placeholder

    def pop_many(self, count: int) -> List[LuaExpression]:
        items = [self.pop_single() for _ in range(count)]
        items.reverse()
        return items

    def pop_pair(self) -> Tuple[LuaExpression, LuaExpression]:
        lhs, rhs = self.pop_many(2)
        return lhs, rhs

    def flush(self) -> List[LuaExpression]:
        values = list(self._values)
        self._values.clear()
        return values


class StringLiteralCollector:
    """Group consecutive string literal assignments into annotated sequences."""

    def __init__(self) -> None:
        self._pending: List[List[LuaStatement]] = []
        self._fragments: List[str] = []
        self._offsets: List[int] = []

    def reset(self) -> None:
        self._pending.clear()
        self._fragments.clear()
        self._offsets.clear()

    def enqueue(
        self,
        offset: int,
        statements: List[LuaStatement],
        expression: LuaExpression,
    ) -> List[LuaStatement]:
        literal_text = self._string_value(expression)
        if literal_text is None:
            flushed = self.flush()
            return flushed + statements
        self._pending.append(statements)
        self._fragments.append(literal_text)
        self._offsets.append(offset)
        return []

    def flush(self) -> List[LuaStatement]:
        if not self._pending:
            return []
        combined = "".join(self._fragments)
        comment = CommentStatement(
            f"string literal sequence: {escape_lua_string(combined)}"
            f" (len={len(combined)})"
        )
        result: List[LuaStatement] = [comment]
        for group in self._pending:
            result.extend(group)
        self.reset()
        return result

    def finalize(self) -> List[LuaStatement]:
        return self.flush()

    @staticmethod
    def _string_value(expression: LuaExpression) -> Optional[str]:
        if isinstance(expression, LiteralExpr):
            literal = expression.literal
            if literal is not None and literal.kind == "string":
                return str(literal.value)
        return None


@dataclass
class FunctionMetadata:
    """Descriptive statistics collected while reconstructing a function."""

    block_count: int
    instruction_count: int
    warnings: List[str] = field(default_factory=list)
    helper_calls: int = 0
    branch_count: int = 0
    literal_count: int = 0
    literal_runs: Sequence[LiteralRun] = field(default_factory=tuple)
    literal_stats: Optional[LiteralStatistics] = None
    literal_report: Optional[LiteralRunReport] = None
    placeholders: Sequence[str] = field(default_factory=tuple)
    mnemonic_counts: Dict[str, int] = field(default_factory=dict)
    tag_counts: Dict[str, int] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = ["function summary:"]
        lines.append(f"- blocks: {self.block_count}")
        lines.append(f"- instructions: {self.instruction_count}")
        lines.append(f"- literal instructions: {self.literal_count}")
        lines.append(f"- helper invocations: {self.helper_calls}")
        lines.append(f"- branches: {self.branch_count}")
        if self.placeholders:
            lines.append(f"- placeholder values: {len(self.placeholders)}")
        string_runs = [run for run in self.literal_runs if run.kind == "string"]
        if string_runs:
            lines.append(f"- string literal sequences: {len(string_runs)}")
        numeric_runs = [run for run in self.literal_runs if run.kind == "number"]
        if numeric_runs:
            lines.append(f"- numeric literal runs: {len(numeric_runs)}")
        if self.literal_stats and self.literal_stats.kind_counts:
            breakdown = ", ".join(
                f"{kind}={count}" for kind, count in sorted(self.literal_stats.kind_counts.items())
            )
            lines.append(f"- literal run kinds: {breakdown}")
        return lines

    def warning_lines(self) -> List[str]:
        return [f"- {warning}" for warning in self.warnings]

    def placeholder_names(self) -> List[str]:
        if not self.placeholders:
            return []
        return sorted(dict.fromkeys(self.placeholders))

    def literal_run_lines(
        self, *, limit: int = 8, preview: int = 72
    ) -> List[str]:
        if not self.literal_runs:
            return []
        lines: List[str] = []
        for run in list(self.literal_runs)[:limit]:
            summary = run.render_preview(preview)
            lines.append(
                "- 0x"
                f"{run.start_offset():06X}"
                f" kind={run.kind}"
                f" count={run.length()}: {summary}"
            )
        remaining = len(self.literal_runs) - limit
        if remaining > 0:
            lines.append(f"- ... ({remaining} additional runs)")
        return lines

    def literal_statistics_lines(self) -> List[str]:
        if not self.literal_stats:
            return []
        return self.literal_stats.summary_lines()

    def literal_report_lines(self, *, block_limit: int = 3) -> List[str]:
        if not self.literal_report or not self.literal_report.runs:
            return []
        summary = self.literal_report.summary_lines()
        blocks = self.literal_report.block_lines(limit=block_limit)
        return summary + blocks

    # ------------------------------------------------------------------
    def _top_items(
        self, data: Dict[str, int], *, limit: int
    ) -> List[Tuple[str, int, float]]:
        if not data:
            return []
        total = max(sum(data.values()), 1)
        items = sorted(data.items(), key=lambda item: (-item[1], item[0]))
        result: List[Tuple[str, int, float]] = []
        for name, count in items[:limit]:
            percentage = round((count / total) * 100, 1)
            result.append((name, count, percentage))
        return result

    def mnemonic_summary(self, limit: int = 6) -> List[Tuple[str, int, float]]:
        return self._top_items(self.mnemonic_counts, limit=limit)

    def tag_summary(self, limit: int = 6) -> List[Tuple[str, int, float]]:
        return self._top_items(self.tag_counts, limit=limit)

    def instruction_profile_lines(self, limit: int = 6) -> List[str]:
        top = self.mnemonic_summary(limit=limit)
        if not top:
            return []
        lines = ["instruction profile (top mnemonics):"]
        for name, count, percentage in top:
            lines.append(f"- {name}: {count} ({percentage:.1f}% of instructions)")
        return lines

    def tag_profile_lines(self, limit: int = 6) -> List[str]:
        top = self.tag_summary(limit=limit)
        if not top:
            return []
        lines = ["instruction tags (top categories):"]
        for name, count, percentage in top:
            lines.append(f"- {name}: {count} ({percentage:.1f}% of instructions)")
        return lines

    def density_stats(self) -> Dict[str, float]:
        if self.instruction_count <= 0:
            return {}
        base = max(self.instruction_count, 1)
        stats: Dict[str, float] = {}
        if self.literal_count:
            stats["literal"] = round((self.literal_count / base) * 100, 2)
        if self.helper_calls:
            stats["helper"] = round((self.helper_calls / base) * 100, 2)
        if self.branch_count:
            stats["branch"] = round((self.branch_count / base) * 100, 2)
        return stats

    def density_lines(self) -> List[str]:
        stats = self.density_stats()
        if not stats:
            return []
        order = sorted(stats.items(), key=lambda item: (-item[1], item[0]))
        lines = ["instruction density:"]
        for name, percentage in order:
            lines.append(f"- {name}: {percentage:.2f}% of instructions")
        return lines


@dataclass
class HighLevelFunction:
    """Container describing a reconstructed Lua function."""

    name: str
    body: BlockStatement
    metadata: FunctionMetadata
    segment_index: int

    def render(self) -> str:
        writer = LuaWriter()
        summary = self.metadata.summary_lines()
        if summary:
            writer.write_comment_block(summary)
            writer.write_line("")
        profile_lines = self.metadata.instruction_profile_lines()
        if profile_lines:
            writer.write_comment_block(profile_lines)
            writer.write_line("")
        tag_lines = self.metadata.tag_profile_lines()
        if tag_lines:
            writer.write_comment_block(tag_lines)
            writer.write_line("")
        density_lines = self.metadata.density_lines()
        if density_lines:
            writer.write_comment_block(density_lines)
            writer.write_line("")
        literal_lines = self.metadata.literal_run_lines()
        if literal_lines:
            writer.write_comment_block(["literal runs:"] + literal_lines)
            writer.write_line("")
        report_lines = self.metadata.literal_report_lines()
        if report_lines:
            writer.write_comment_block(report_lines)
            writer.write_line("")
        stats_lines = self.metadata.literal_statistics_lines()
        if stats_lines:
            writer.write_comment_block(stats_lines)
            writer.write_line("")
        if self.metadata.warnings:
            writer.write_comment_block(
                ["stack reconstruction warnings:"] + self.metadata.warning_lines()
            )
            writer.write_line("")
        writer.write_line(f"function {self.name}()")
        with writer.indented():
            placeholders = self.metadata.placeholder_names()
            if placeholders:
                writer.write_comment(
                    "placeholder locals inserted for stack underflow warnings"
                )
                for name in placeholders:
                    writer.write_line(f"local {name}")
                writer.ensure_blank_line()
            self.body.emit(writer)
        writer.write_line("end")
        return writer.render()


@dataclass
class FunctionMetadataEntry:
    """Snapshot of the metadata that feeds the module level summary."""

    name: str
    segment_index: int
    blocks: int
    instructions: int
    helper_calls: int
    branches: int
    literal_instructions: int
    literal_runs: Sequence[LiteralRun]
    literal_kind_counts: Dict[str, int]
    warnings: List[str]
    placeholders: List[str]
    literal_report_blocks: int = 0
    top_tokens: List[Tuple[str, int]] = field(default_factory=list)
    top_numbers: List[Tuple[int, int]] = field(default_factory=list)
    longest_runs: List[str] = field(default_factory=list)
    mnemonics: List[Dict[str, object]] = field(default_factory=list)
    tags: List[Dict[str, object]] = field(default_factory=list)
    density: Dict[str, float] = field(default_factory=dict)

    def literal_run_count(self) -> int:
        return len(self.literal_runs)

    def string_run_count(self) -> int:
        return self.literal_kind_counts.get("string", 0)

    def numeric_run_count(self) -> int:
        return self.literal_kind_counts.get("number", 0)

    def warning_count(self) -> int:
        return len(self.warnings)

    def placeholder_count(self) -> int:
        return len(self.placeholders)

    def to_table(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "name": self.name,
            "segment": self.segment_index,
            "blocks": self.blocks,
            "instructions": self.instructions,
            "helpers": self.helper_calls,
            "branches": self.branches,
            "literal_instructions": self.literal_instructions,
            "literal_runs": self.literal_run_count(),
        }
        if self.literal_kind_counts:
            data["literal_kinds"] = dict(sorted(self.literal_kind_counts.items()))
        if self.literal_report_blocks:
            data["literal_report_blocks"] = self.literal_report_blocks
        if self.top_tokens:
            data["top_tokens"] = [list(item) for item in self.top_tokens]
        if self.top_numbers:
            data["top_numbers"] = [list(item) for item in self.top_numbers]
        if self.longest_runs:
            data["longest_runs"] = list(self.longest_runs)
        if self.warnings:
            data["warnings"] = list(self.warnings)
        if self.placeholders:
            data["placeholders"] = list(self.placeholders)
        if self.mnemonics:
            data["mnemonics"] = [dict(item) for item in self.mnemonics]
        if self.tags:
            data["tags"] = [dict(item) for item in self.tags]
        if self.density:
            data["density"] = dict(self.density)
        return data


@dataclass
class ModuleMetadataSnapshot:
    """Aggregated view of module level metadata and helper information."""

    summary: Dict[str, Any]
    functions: List[FunctionMetadataEntry]
    helper_info: Dict[str, Any]
    enum_info: List[Dict[str, Any]]
    name_index: Dict[str, int] = field(default_factory=dict)
    warning_map: Dict[str, List[str]] = field(default_factory=dict)
    placeholder_map: Dict[str, List[str]] = field(default_factory=dict)
    literal_preview_map: Dict[str, List[str]] = field(default_factory=dict)
    helper_function_map: Dict[str, Any] = field(default_factory=dict)
    struct_method_map: Dict[str, Any] = field(default_factory=dict)
    mnemonic_map: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    tag_map: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    module_mnemonics: List[Dict[str, object]] = field(default_factory=list)
    module_tags: List[Dict[str, object]] = field(default_factory=list)
    density_map: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        lines = ["module summary:"]
        lines.append(f"- functions: {self.summary.get('functions', 0)}")
        lines.append(f"- helper functions: {self.summary.get('helper_functions', 0)}")
        lines.append(f"- struct helpers: {self.summary.get('struct_helpers', 0)}")
        lines.append(
            f"- literal instructions: {self.summary.get('literal_instructions', 0)}"
        )
        lines.append(
            f"- branch instructions: {self.summary.get('branch_instructions', 0)}"
        )
        enum_info = self.summary.get("enum_namespaces")
        if isinstance(enum_info, dict):
            lines.append(
                "- enum namespaces: "
                f"{enum_info.get('count', 0)} "
                f"({enum_info.get('values', 0)} values)"
            )
        lines.append(f"- stack warnings: {self.summary.get('stack_warnings', 0)}")
        placeholder_total = self.summary.get("placeholder_values")
        if placeholder_total:
            lines.append(f"- placeholder values: {placeholder_total}")
        string_total = self.summary.get("string_literal_sequences")
        if string_total:
            lines.append(f"- string literal sequences: {string_total}")
        numeric_total = self.summary.get("numeric_literal_runs")
        if numeric_total:
            lines.append(f"- numeric literal runs: {numeric_total}")
        literal_report = self.summary.get("literal_report")
        if isinstance(literal_report, dict):
            blocks = literal_report.get("blocks")
            if blocks:
                lines.append(f"- literal run blocks: {blocks}")
            tokens = literal_report.get("top_tokens") or []
            if tokens:
                token_line = ", ".join(f"{token}:{count}" for token, count in tokens)
                lines.append(f"- literal tokens: {token_line}")
            numbers = literal_report.get("top_numbers") or []
            if numbers:
                number_line = ", ".join(f"{value}:{count}" for value, count in numbers)
                lines.append(f"- literal numbers: {number_line}")
            previews = literal_report.get("longest_runs") or []
            if previews:
                lines.append("- notable literal runs: " + "; ".join(previews[:2]))
        mnemonic_info = self.summary.get("mnemonics")
        if isinstance(mnemonic_info, dict):
            unique = mnemonic_info.get("unique", 0)
            lines.append(f"- instruction mnemonics: {unique} unique")
            top = mnemonic_info.get("top") or []
            if top:
                preview = ", ".join(
                    f"{item['name']}:{item['count']}" for item in top[:4]
                )
                lines.append(f"- top mnemonics: {preview}")
        tag_info = self.summary.get("tags")
        if isinstance(tag_info, dict):
            unique_tags = tag_info.get("unique", 0)
            lines.append(f"- instruction tags: {unique_tags} categories")
            tag_preview = tag_info.get("top") or []
            if tag_preview:
                preview = ", ".join(
                    f"{item['name']}:{item['count']}" for item in tag_preview[:4]
                )
                lines.append(f"- top tags: {preview}")
        density_info = self.summary.get("density")
        if isinstance(density_info, dict) and density_info:
            density_line = ", ".join(
                f"{name}={value:.2f}%" for name, value in sorted(density_info.items())
            )
            lines.append(f"- instruction density: {density_line}")
        return lines

    def has_table(self) -> bool:
        return bool(
            self.functions
            or self.helper_info
            or self.enum_info
            or self.module_mnemonics
            or self.module_tags
            or self.mnemonic_map
            or self.tag_map
            or self.density_map
        )

    def to_table(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"summary": self.summary}
        if self.functions:
            payload["functions"] = [entry.to_table() for entry in self.functions]
        if self.helper_info:
            payload["helpers"] = self.helper_info
        if self.enum_info:
            payload["enums"] = self.enum_info
        if self.module_mnemonics:
            payload["mnemonics"] = [dict(item) for item in self.module_mnemonics]
        if self.module_tags:
            payload["tags"] = [dict(item) for item in self.module_tags]
        if self.density_map:
            payload["density_map"] = {
                name: dict(values) for name, values in sorted(self.density_map.items())
            }
        return payload

    def render_table(self, writer: LuaWriter) -> None:
        if not self.has_table():
            return
        writer.write_comment("module metadata tables (auto-generated)")
        table_lines = render_lua_value(self.to_table()).splitlines()
        if not table_lines:
            return
        writer.write_line(f"local __module_metadata = {table_lines[0]}")
        for line in table_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        index_lines = render_lua_value(self.name_index or {}).splitlines()
        writer.write_line(f"local __metadata_index = {index_lines[0]}")
        for line in index_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        warning_lines = render_lua_value(self.warning_map or {}).splitlines()
        writer.write_line(f"local __metadata_warnings = {warning_lines[0]}")
        for line in warning_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        placeholder_lines = render_lua_value(self.placeholder_map or {}).splitlines()
        writer.write_line(f"local __metadata_placeholders = {placeholder_lines[0]}")
        for line in placeholder_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        literal_lines = render_lua_value(self.literal_preview_map or {}).splitlines()
        writer.write_line(f"local __metadata_literal_runs = {literal_lines[0]}")
        for line in literal_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        mnemonic_lines = render_lua_value(self.mnemonic_map or {}).splitlines()
        writer.write_line(f"local __metadata_mnemonics = {mnemonic_lines[0]}")
        for line in mnemonic_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        tag_lines = render_lua_value(self.tag_map or {}).splitlines()
        writer.write_line(f"local __metadata_tags = {tag_lines[0]}")
        for line in tag_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        module_mnemonics = render_lua_value(self.module_mnemonics or {}).splitlines()
        writer.write_line(f"local __module_mnemonics = {module_mnemonics[0]}")
        for line in module_mnemonics[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        module_tags = render_lua_value(self.module_tags or {}).splitlines()
        writer.write_line(f"local __module_tags = {module_tags[0]}")
        for line in module_tags[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        density_lines = render_lua_value(self.density_map or {}).splitlines()
        writer.write_line(f"local __metadata_density = {density_lines[0]}")
        for line in density_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()

        writer.write_comment("metadata query helpers")
        writer.write_line("local function module_metadata()")
        with writer.indented():
            writer.write_line("return __module_metadata")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_metadata(name)")
        with writer.indented():
            writer.write_line("local index = __metadata_index[name]")
            writer.write_line("if not index then")
            with writer.indented():
                writer.write_line("return nil")
            writer.write_line("end")
            writer.write_line("return __module_metadata.functions[index]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_warnings(name)")
        with writer.indented():
            writer.write_line("return __metadata_warnings[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_placeholders(name)")
        with writer.indented():
            writer.write_line("return __metadata_placeholders[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_literal_runs(name)")
        with writer.indented():
            writer.write_line("return __metadata_literal_runs[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_mnemonics(name)")
        with writer.indented():
            writer.write_line("return __metadata_mnemonics[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_tags(name)")
        with writer.indented():
            writer.write_line("return __metadata_tags[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function module_mnemonics()")
        with writer.indented():
            writer.write_line("return __module_mnemonics")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function module_tags()")
        with writer.indented():
            writer.write_line("return __module_tags")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function function_density(name)")
        with writer.indented():
            writer.write_line("return __metadata_density[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function module_density()")
        with writer.indented():
            writer.write_line("local summary = __module_metadata.summary")
            writer.write_line("if not summary then")
            with writer.indented():
                writer.write_line("return nil")
            writer.write_line("end")
            writer.write_line("return summary.density")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_comment("helper signature tables (auto-generated)")
        helper_lines = render_lua_value(self.helper_function_map or {}).splitlines()
        writer.write_line(f"local __helper_functions = {helper_lines[0]}")
        for line in helper_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()
        struct_lines = render_lua_value(self.struct_method_map or {}).splitlines()
        writer.write_line(f"local __struct_helpers = {struct_lines[0]}")
        for line in struct_lines[1:]:
            writer.write_line(line)
        writer.ensure_blank_line()
        writer.write_line("local function helper_metadata(name)")
        with writer.indented():
            writer.write_line("return __helper_functions[name]")
        writer.write_line("end")
        writer.ensure_blank_line()
        writer.write_line("local function struct_helper_metadata(struct, method)")
        with writer.indented():
            writer.write_line("local bucket = __struct_helpers[struct]")
            writer.write_line("if not bucket then")
            with writer.indented():
                writer.write_line("return nil")
            writer.write_line("end")
            writer.write_line("return bucket[method]")
        writer.write_line("end")


class ModuleMetadataBuilder:
    """Construct structured metadata for the reconstructed module."""

    def __init__(
        self,
        functions: Sequence[HighLevelFunction],
        *,
        helper_registry: HelperRegistry,
        enum_registry: EnumRegistry,
        options: LuaRenderOptions,
    ) -> None:
        self._functions = functions
        self._helper_registry = helper_registry
        self._enum_registry = enum_registry
        self._options = options

    def build(self) -> ModuleMetadataSnapshot:
        function_entries = [self._build_function_entry(func) for func in self._functions]
        name_index = {entry.name: idx + 1 for idx, entry in enumerate(function_entries)}
        warning_map = {
            entry.name: list(entry.warnings)
            for entry in function_entries
            if entry.warnings
        }
        placeholder_map = {
            entry.name: list(entry.placeholders)
            for entry in function_entries
            if entry.placeholders
        }
        literal_preview_map = {
            entry.name: list(entry.longest_runs)
            for entry in function_entries
            if entry.longest_runs
        }
        mnemonic_map = {
            entry.name: [dict(item) for item in entry.mnemonics]
            for entry in function_entries
            if entry.mnemonics
        }
        tag_map = {
            entry.name: [dict(item) for item in entry.tags]
            for entry in function_entries
            if entry.tags
        }
        density_map = {
            entry.name: dict(entry.density)
            for entry in function_entries
            if entry.density
        }
        mnemonic_counter: Counter[str] = Counter()
        tag_counter: Counter[str] = Counter()
        for entry in function_entries:
            for item in entry.mnemonics:
                mnemonic_counter[item["name"]] += int(item["count"])
            for item in entry.tags:
                tag_counter[item["name"]] += int(item["count"])
        summary = self._build_summary(function_entries, mnemonic_counter, tag_counter)
        total_instructions = sum(entry.instructions for entry in function_entries)
        module_mnemonics = self._profile_from_counter(
            mnemonic_counter, total_instructions, limit=64
        )
        module_tags = self._profile_from_counter(
            tag_counter, total_instructions, limit=64
        )
        helper_info, helper_function_map, struct_method_map = self._build_helper_info()
        enum_info = self._build_enum_info()
        return ModuleMetadataSnapshot(
            summary=summary,
            functions=function_entries,
            helper_info=helper_info,
            enum_info=enum_info,
            name_index=name_index,
            warning_map=warning_map,
            placeholder_map=placeholder_map,
            literal_preview_map=literal_preview_map,
            helper_function_map=helper_function_map,
            struct_method_map=struct_method_map,
            mnemonic_map=mnemonic_map,
            tag_map=tag_map,
            module_mnemonics=module_mnemonics,
            module_tags=module_tags,
            density_map=density_map,
        )

    def _build_function_entry(self, function: HighLevelFunction) -> FunctionMetadataEntry:
        metadata = function.metadata
        kind_counts = (
            dict(metadata.literal_stats.kind_counts)
            if metadata.literal_stats
            else {}
        )
        report_blocks = 0
        top_tokens: List[Tuple[str, int]] = []
        top_numbers: List[Tuple[int, int]] = []
        longest_runs: List[str] = []
        if metadata.literal_report:
            report_blocks = len(metadata.literal_report.block_summaries)
            top_tokens = metadata.literal_report.top_tokens(limit=5)
            top_numbers = metadata.literal_report.top_numbers(limit=5)
            longest_runs = metadata.literal_report.longest_previews(limit=5)
        mnemonic_profile = self._profile_from_counts(
            metadata.mnemonic_counts,
            metadata.instruction_count,
        )
        tag_profile = self._profile_from_counts(
            metadata.tag_counts,
            metadata.instruction_count,
        )
        density = metadata.density_stats()
        return FunctionMetadataEntry(
            name=function.name,
            segment_index=function.segment_index,
            blocks=metadata.block_count,
            instructions=metadata.instruction_count,
            helper_calls=metadata.helper_calls,
            branches=metadata.branch_count,
            literal_instructions=metadata.literal_count,
            literal_runs=tuple(metadata.literal_runs),
            literal_kind_counts=kind_counts,
            warnings=list(metadata.warnings),
            placeholders=metadata.placeholder_names(),
            literal_report_blocks=report_blocks,
            top_tokens=top_tokens,
            top_numbers=top_numbers,
            longest_runs=longest_runs,
            mnemonics=mnemonic_profile,
            tags=tag_profile,
            density=density,
        )

    def _build_summary(
        self,
        entries: Sequence[FunctionMetadataEntry],
        mnemonic_counter: Counter[str],
        tag_counter: Counter[str],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "functions": len(entries),
            "helper_functions": self._helper_registry.function_count(),
            "struct_helpers": self._helper_registry.method_count(),
            "literal_instructions": sum(
                entry.literal_instructions for entry in entries
            ),
            "branch_instructions": sum(entry.branches for entry in entries),
            "stack_warnings": sum(entry.warning_count() for entry in entries),
        }
        placeholder_total = sum(entry.placeholder_count() for entry in entries)
        if placeholder_total:
            summary["placeholder_values"] = placeholder_total
        if not self._enum_registry.is_empty():
            summary["enum_namespaces"] = {
                "count": self._enum_registry.namespace_count(),
                "values": self._enum_registry.total_values(),
            }
        string_total = sum(entry.string_run_count() for entry in entries)
        if string_total:
            summary["string_literal_sequences"] = string_total
        numeric_total = sum(entry.numeric_run_count() for entry in entries)
        if numeric_total:
            summary["numeric_literal_runs"] = numeric_total
        if self._options.emit_literal_report:
            aggregated: List[LiteralRun] = []
            for entry in entries:
                aggregated.extend(entry.literal_runs)
            if aggregated:
                report = build_literal_run_report(aggregated)
                summary["literal_report"] = {
                    "blocks": len(report.block_summaries),
                    "top_tokens": report.top_tokens(limit=5),
                    "top_numbers": report.top_numbers(limit=5),
                    "longest_runs": report.longest_previews(limit=5),
                }
        total_instructions = sum(entry.instructions for entry in entries)
        if mnemonic_counter:
            summary["mnemonics"] = {
                "unique": len(mnemonic_counter),
                "top": self._profile_from_counter(
                    mnemonic_counter, total_instructions, limit=8
                ),
            }
        if tag_counter:
            summary["tags"] = {
                "unique": len(tag_counter),
                "top": self._profile_from_counter(
                    tag_counter, total_instructions, limit=8
                ),
            }
        density_summary = self._density_summary(entries)
        if density_summary:
            summary["density"] = density_summary
        return summary

    def _build_helper_info(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        counts = {
            "functions": self._helper_registry.function_count(),
            "methods": self._helper_registry.method_count(),
            "structs": self._helper_registry.struct_count(),
        }
        functions: List[Dict[str, Any]] = []
        function_map: Dict[str, Any] = {}
        for signature in self._helper_registry.iter_functions():
            functions.append(
                {
                    "name": signature.name,
                    "inputs": signature.inputs,
                    "outputs": signature.outputs,
                    "uses_operand": signature.uses_operand,
                    "summary": signature.summary,
                }
            )
            function_map[signature.name] = {
                "inputs": signature.inputs,
                "outputs": signature.outputs,
                "uses_operand": signature.uses_operand,
                "summary": signature.summary,
            }
        methods: Dict[str, List[Dict[str, Any]]] = {}
        struct_map: Dict[str, Any] = {}
        for struct_name, signatures in self._helper_registry.iter_methods():
            entries: List[Dict[str, Any]] = []
            detail: Dict[str, Any] = {}
            for signature in signatures:
                entries.append(
                    {
                        "name": signature.name,
                        "method": signature.method,
                        "inputs": signature.inputs,
                        "outputs": signature.outputs,
                        "uses_operand": signature.uses_operand,
                        "summary": signature.summary,
                    }
                )
                detail[signature.method] = {
                    "name": signature.name,
                    "inputs": signature.inputs,
                    "outputs": signature.outputs,
                    "uses_operand": signature.uses_operand,
                    "summary": signature.summary,
                }
            methods[struct_name] = entries
            struct_map[struct_name] = detail
        payload: Dict[str, Any] = {"counts": counts}
        if functions:
            payload["functions"] = functions
        if methods:
            payload["methods"] = methods
        return payload, function_map, struct_map

    def _build_enum_info(self) -> List[Dict[str, Any]]:
        namespaces: List[Dict[str, Any]] = []
        for namespace in self._enum_registry.iter_namespaces():
            entry: Dict[str, Any] = {"name": namespace.name}
            if namespace.description:
                entry["description"] = namespace.description
            values = [
                {"value": value, "label": label}
                for value, label in sorted(namespace.values.items())
            ]
            entry["values"] = values
            namespaces.append(entry)
        return namespaces

    @staticmethod
    def _profile_from_counter(
        counter: Counter[str],
        total: int,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        if not counter:
            return []
        safe_total = max(total, 1)
        items = counter.most_common()
        if limit is not None:
            items = items[:limit]
        profile: List[Dict[str, object]] = []
        for name, count in items:
            percentage = round((count / safe_total) * 100, 2)
            profile.append({
                "name": name,
                "count": count,
                "percentage": percentage,
            })
        return profile

    @staticmethod
    def _profile_from_counts(
        counts: Dict[str, int],
        total: int,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        if not counts:
            return []
        counter: Counter[str] = Counter(counts)
        return ModuleMetadataBuilder._profile_from_counter(
            counter, total, limit=limit
        )

    @staticmethod
    def _density_summary(
        entries: Sequence[FunctionMetadataEntry],
    ) -> Dict[str, float]:
        total_instructions = sum(entry.instructions for entry in entries)
        if total_instructions <= 0:
            return {}
        safe_total = float(total_instructions)
        literal = sum(entry.literal_instructions for entry in entries)
        helpers = sum(entry.helper_calls for entry in entries)
        branches = sum(entry.branches for entry in entries)
        result = {
            "literal": round((literal / safe_total) * 100, 2) if literal else 0.0,
            "helper": round((helpers / safe_total) * 100, 2) if helpers else 0.0,
            "branch": round((branches / safe_total) * 100, 2) if branches else 0.0,
        }
        return {name: value for name, value in result.items() if value > 0}


@dataclass
class Terminator:
    """Base terminator node summarising the end of a block."""


@dataclass
class ReturnTerminator(Terminator):
    values: List[LuaExpression]
    comment: Optional[str]


@dataclass
class JumpTerminator(Terminator):
    target: Optional[int]
    fallthrough: Optional[int]
    comment: Optional[str]


@dataclass
class BranchTerminator(Terminator):
    condition: LuaExpression
    true_target: Optional[int]
    false_target: Optional[int]
    fallthrough: Optional[int]
    comment: Optional[str]
    enum_namespace: Optional[str] = None
    enum_values: Dict[int, str] = field(default_factory=dict)


@dataclass
class BlockInfo:
    start: int
    statements: List[LuaStatement]
    terminator: Optional[Terminator]
    successors: List[int]


class BlockTranslator:
    """Translate IR blocks into :class:`BlockInfo` instances."""

    def __init__(self, reconstructor: "HighLevelReconstructor", program: IRProgram) -> None:
        self._reconstructor = reconstructor
        self._program = program
        self._stack = HighLevelStack()
        self._string_collector = StringLiteralCollector()
        self._literal_tracker = LiteralRunTracker()
        self.literal_count = 0
        self.helper_calls = 0
        self.branch_count = 0
        self.instruction_total = 0
        self.mnemonic_counts: Counter[str] = Counter()
        self.tag_counts: Counter[str] = Counter()

    def translate(self) -> Dict[int, BlockInfo]:
        blocks: Dict[int, BlockInfo] = {}
        order = sorted(self._program.blocks)
        for idx, start in enumerate(order):
            block = self._program.blocks[start]
            next_offset = order[idx + 1] if idx + 1 < len(order) else None
            self._literal_tracker.start_block(block.start)
            blocks[start] = self._translate_block(block, next_offset)
            self.instruction_total += len(block.instructions)
        self._literal_tracker.finalize()
        return blocks

    # ------------------------------------------------------------------
    def _translate_block(self, block: IRBlock, fallthrough: Optional[int]) -> BlockInfo:
        statements: List[LuaStatement] = []
        terminator: Optional[Terminator] = None
        self._string_collector.reset()
        for index, instruction in enumerate(block.instructions):
            semantics = instruction.semantics
            self._reconstructor._register_enums(semantics)
            is_last = index == len(block.instructions) - 1
            self.mnemonic_counts[semantics.mnemonic] += 1
            for tag in semantics.tags:
                self.tag_counts[tag] += 1
            if semantics.has_tag("literal"):
                operand_expr = self._reconstructor._operand_expression(
                    semantics, instruction.operand
                )
                literal_statements, operand_expr = self._reconstructor._translate_literal(
                    instruction,
                    semantics,
                    self,
                    operand_expr=operand_expr,
                )
                self._literal_tracker.observe(instruction.offset, operand_expr)
                queued = self._string_collector.enqueue(
                    instruction.offset, literal_statements, operand_expr
                )
                statements.extend(queued)
                continue
            else:
                self._literal_tracker.break_sequence()
            statements.extend(self._string_collector.flush())
            if semantics.has_tag("comparison"):
                statements.extend(
                    self._reconstructor._translate_comparison(instruction, semantics, self)
                )
                continue
            if semantics.control_flow == "branch" and is_last:
                terminator = self._reconstructor._translate_branch(
                    block.start,
                    instruction,
                    semantics,
                    self,
                    fallthrough,
                )
                continue
            if semantics.control_flow == "return" and is_last:
                terminator = self._reconstructor._translate_return(
                    instruction,
                    semantics,
                    self,
                )
                continue
            if semantics.has_tag("structure"):
                statements.extend(
                    self._reconstructor._translate_structure(instruction, semantics, self)
                )
                continue
            if semantics.has_tag("call"):
                statements.extend(
                    self._reconstructor._translate_call(instruction, semantics, self)
                )
                continue
            statements.extend(
                self._reconstructor._translate_generic(instruction, semantics, self)
            )
        statements.extend(self._string_collector.finalize())
        self._literal_tracker.break_sequence()
        if terminator is None:
            target = block.successors[0] if block.successors else fallthrough
            terminator = JumpTerminator(target=target, fallthrough=fallthrough, comment=None)
        return BlockInfo(
            start=block.start,
            statements=statements,
            terminator=terminator,
            successors=list(block.successors),
        )

    # ------------------------------------------------------------------
    @property
    def stack(self) -> HighLevelStack:
        return self._stack

    @property
    def program(self) -> IRProgram:
        return self._program

    @property
    def literal_runs(self) -> Sequence[LiteralRun]:
        return self._literal_tracker.runs()


class ControlFlowStructurer:
    """Best effort structuring of :class:`BlockInfo` sequences."""

    def __init__(self, blocks: Dict[int, BlockInfo]) -> None:
        self._blocks = blocks
        self._emitted: set[int] = set()
        self._predecessors: Dict[int, set[int]] = self._compute_predecessors(blocks)

    @staticmethod
    def _compute_predecessors(blocks: Dict[int, BlockInfo]) -> Dict[int, set[int]]:
        result: Dict[int, set[int]] = {start: set() for start in blocks}
        for start, info in blocks.items():
            for succ in info.successors:
                if succ in result:
                    result[succ].add(start)
        return result

    def structure(self, entry: int) -> BlockStatement:
        statements = self._structure_sequence(entry, stop=None)
        return wrap_block(statements)

    # ------------------------------------------------------------------
    def _structure_sequence(
        self, start: Optional[int], stop: Optional[int]
    ) -> List[LuaStatement]:
        current = start
        result: List[LuaStatement] = []
        while current is not None and current != stop and current in self._blocks:
            if current in self._emitted:
                # Create a comment to note repeated entry and avoid infinite loops.
                result.append(
                    CommentStatement(f"revisit block 0x{current:06X}")
                )
                break
            info = self._blocks[current]
            self._emitted.add(current)
            result.extend(info.statements)
            terminator = info.terminator
            if isinstance(terminator, ReturnTerminator):
                if terminator.comment:
                    result.append(CommentStatement(terminator.comment))
                result.append(ReturnStatement(terminator.values))
                current = None
                continue
            if isinstance(terminator, BranchTerminator):
                statement, next_block = self._handle_branch(current, terminator)
                result.append(statement)
                current = next_block
                continue
            if isinstance(terminator, JumpTerminator):
                if terminator.comment:
                    result.append(CommentStatement(terminator.comment))
                if terminator.target is None or terminator.target == terminator.fallthrough:
                    current = terminator.fallthrough
                else:
                    current = terminator.target
                continue
            break
        return result

    def _handle_branch(
        self, current: int, terminator: BranchTerminator
    ) -> Tuple[LuaStatement, Optional[int]]:
        condition = terminator.condition
        true_target = terminator.true_target
        false_target = terminator.false_target or terminator.fallthrough
        fallthrough = terminator.fallthrough

        # Loop heuristic: condition guarding a backward edge.
        if (
            true_target is not None
            and true_target in self._blocks
            and true_target <= current
            and false_target is not None
            and false_target != true_target
        ):
            body_statements = self._structure_sequence(false_target, stop=current)
            body = wrap_block(body_statements)
            if terminator.comment:
                body.statements.insert(0, CommentStatement(terminator.comment))
            return WhileStatement(condition, body), fallthrough

        if true_target is not None and false_target is None:
            then_body = wrap_block(self._structure_sequence(true_target, fallthrough))
            clauses = [IfClause(condition=condition, body=then_body)]
            if terminator.comment:
                then_body.statements.insert(0, CommentStatement(terminator.comment))
            return IfStatement(clauses), fallthrough

        if true_target is not None and false_target is not None:
            then_body = wrap_block(self._structure_sequence(true_target, fallthrough))
            else_body = wrap_block(self._structure_sequence(false_target, fallthrough))
            clauses = [IfClause(condition=condition, body=then_body), IfClause(None, else_body)]
            if terminator.comment:
                then_body.statements.insert(0, CommentStatement(terminator.comment))
            return IfStatement(clauses), fallthrough

        comment = terminator.comment or "unstructured branch"
        return CommentStatement(comment), fallthrough


class HighLevelReconstructor:
    """Best-effort reconstruction of high-level Lua from IR programs."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        *,
        options: Optional[LuaRenderOptions] = None,
    ) -> None:
        self.knowledge = knowledge
        self._literal_formatter = LuaLiteralFormatter()
        self._enum_registry = EnumRegistry()
        self._comment_formatter = CommentFormatter()
        self._helper_registry = HelperRegistry()
        self.options = options or LuaRenderOptions()
        self._last_summary: Optional[str] = None
        self._used_function_names: Set[str] = set()

    # ------------------------------------------------------------------
    def from_ir(self, program: IRProgram) -> HighLevelFunction:
        self._last_summary = None
        translator = BlockTranslator(self, program)
        blocks = translator.translate()
        entry = min(blocks)
        structurer = ControlFlowStructurer(blocks)
        body = structurer.structure(entry)
        literal_runs = sorted(
            translator.literal_runs, key=lambda run: (run.start_offset(), -run.length())
        )
        stats = compute_literal_statistics(literal_runs)
        report = (
            build_literal_run_report(literal_runs)
            if self.options.emit_literal_report
            else None
        )
        base_name = self._derive_function_name(program, literal_runs)
        function_name = self._unique_function_name(base_name)
        metadata = FunctionMetadata(
            block_count=len(blocks),
            instruction_count=translator.instruction_total,
            warnings=list(translator.stack.warnings),
            helper_calls=translator.helper_calls,
            branch_count=translator.branch_count,
            literal_count=translator.literal_count,
            literal_runs=tuple(literal_runs),
            literal_stats=stats,
            literal_report=report,
            placeholders=tuple(translator.stack.placeholders),
            mnemonic_counts=dict(translator.mnemonic_counts),
            tag_counts=dict(translator.tag_counts),
        )
        return HighLevelFunction(
            name=function_name,
            body=body,
            metadata=metadata,
            segment_index=program.segment_index,
        )

    # ------------------------------------------------------------------
    def render(self, functions: Sequence[HighLevelFunction]) -> str:
        sections: List[str] = []
        metadata_snapshot = ModuleMetadataBuilder(
            functions,
            helper_registry=self._helper_registry,
            enum_registry=self._enum_registry,
            options=self.options,
        ).build()
        if self.options.emit_module_summary:
            summary_writer = LuaWriter()
            summary_writer.write_comment_block(metadata_snapshot.summary_lines())
            sections.append(summary_writer.render())
        if metadata_snapshot.has_table():
            metadata_writer = LuaWriter()
            metadata_snapshot.render_table(metadata_writer)
            sections.append(metadata_writer.render())
        if not self._enum_registry.is_empty():
            writer = LuaWriter()
            self._enum_registry.render(writer, options=self.options)
            sections.append(writer.render())
        if not self._helper_registry.is_empty():
            writer = LuaWriter()
            self._helper_registry.render(
                writer,
                self._comment_formatter,
                options=self.options,
            )
            sections.append(writer.render())
        for function in functions:
            sections.append(function.render().rstrip())
        return join_sections(sections)

    # ------------------------------------------------------------------
    def _register_enums(self, semantics: InstructionSemantics) -> None:
        if semantics.enum_namespace and semantics.enum_values:
            for value, label in semantics.enum_values.items():
                self._enum_registry.register(semantics.enum_namespace, value, label)

    # Translation helpers -------------------------------------------------
    def _translate_literal(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        *,
        operand_expr: Optional[LuaExpression] = None,
    ) -> Tuple[List[LuaStatement], LuaExpression]:
        operand = operand_expr or self._operand_expression(semantics, instruction.operand)
        statements, _ = translator.stack.push_literal(operand)
        translator.literal_count += 1
        decorated = self._decorate_with_comment(statements, semantics)
        return decorated, operand

    def _translate_comparison(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        lhs, rhs = translator.stack.pop_pair()
        operator = semantics.comparison_operator or "=="
        expression = BinaryExpr(lhs, operator, rhs)
        statements, _ = translator.stack.push_expression(expression, prefix="cmp", make_local=True)
        return self._decorate_with_comment(statements, semantics)

    def _translate_branch(
        self,
        block_start: int,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> BranchTerminator:
        condition = translator.stack.pop_single()
        # Determine targets based on IR metadata.
        true_target, false_target = self._select_branch_targets(
            block_start, translator, fallthrough
        )
        translator.branch_count += 1
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        return BranchTerminator(
            condition=condition,
            true_target=true_target,
            false_target=false_target,
            fallthrough=fallthrough,
            comment=comment if comment else None,
            enum_namespace=semantics.enum_namespace,
            enum_values=dict(semantics.enum_values),
        )

    def _translate_return(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> ReturnTerminator:
        values = translator.stack.flush()
        comment = self._comment_formatter.format_inline(semantics.summary or "")
        return ReturnTerminator(values=values, comment=comment)

    def _translate_structure(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = translator.stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        method = _snake_to_camel(semantics.mnemonic)
        target = NameExpr(semantics.struct_context or "struct")
        call_expr = MethodCallExpr(target, method, args)
        signature = MethodSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
            struct=semantics.struct_context or "struct",
            method=method,
        )
        self._helper_registry.register_method(signature)
        translator.helper_calls += 1
        if outputs > 0:
            statements, _ = translator.stack.push_call_results(call_expr, outputs, prefix="struct")
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

    def _translate_call(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = translator.stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if outputs <= 0:
            statements = [CallStatement(call_expr)]
        else:
            statements, _ = translator.stack.push_call_results(call_expr, outputs)
        return self._decorate_with_comment(statements, semantics)

    def _translate_generic(
        self,
        instruction: IRInstruction,
        semantics: InstructionSemantics,
        translator: BlockTranslator,
    ) -> List[LuaStatement]:
        inputs, outputs = estimate_stack_io(semantics)
        args = translator.stack.pop_many(inputs)
        if semantics.uses_operand:
            args.append(self._operand_expression(semantics, instruction.operand))
        call_expr = CallExpr(NameExpr(semantics.mnemonic), args)
        signature = HelperSignature(
            name=semantics.mnemonic,
            summary=semantics.summary or "",
            inputs=inputs,
            outputs=outputs,
            uses_operand=semantics.uses_operand,
        )
        self._helper_registry.register_function(signature)
        translator.helper_calls += 1
        if outputs > 0:
            statements, _ = translator.stack.push_call_results(call_expr, outputs)
        else:
            statements = [CallStatement(call_expr)]
        return self._decorate_with_comment(statements, semantics)

    def _select_branch_targets(
        self,
        block_start: int,
        translator: BlockTranslator,
        fallthrough: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        block = translator.program.blocks[block_start]
        succ = list(block.successors)
        false_target = None
        true_target = None
        if fallthrough is not None and fallthrough in succ:
            false_target = fallthrough
            succ = [value for value in succ if value != fallthrough]
        if succ:
            true_target = succ[0]
        elif fallthrough is not None:
            true_target = fallthrough
        return true_target, false_target

    # ------------------------------------------------------------------
    def _operand_expression(
        self, semantics: InstructionSemantics, operand: int
    ) -> LuaExpression:
        if semantics.enum_values and operand in semantics.enum_values:
            label = semantics.enum_values[operand]
            if semantics.enum_namespace:
                return NameExpr(f"{semantics.enum_namespace}.{label}")
            return NameExpr(label)
        literal = self._literal_formatter.format_operand(operand)
        return LiteralExpr(literal)

    # ------------------------------------------------------------------
    def _decorate_with_comment(
        self, statements: List[LuaStatement], semantics: InstructionSemantics
    ) -> List[LuaStatement]:
        summary = semantics.summary or ""
        if not self._should_emit_comment(summary):
            return statements
        comment = self._comment_formatter.format_inline(summary)
        if not comment:
            wrapped = self._comment_formatter.wrap(summary)
            return [CommentStatement(line) for line in wrapped] + statements
        return [CommentStatement(comment)] + statements

    def _should_emit_comment(self, summary: str) -> bool:
        if not summary:
            self._last_summary = None
            return False
        if self.options.deduplicate_comments and summary == self._last_summary:
            return False
        self._last_summary = summary
        return True

    def _derive_function_name(
        self,
        program: IRProgram,
        sequences: Sequence[LiteralRun],
    ) -> str:
        candidate = self._select_string_name(program, sequences)
        if candidate:
            return candidate
        return f"segment_{program.segment_index:03d}"

    def _select_string_name(
        self,
        program: IRProgram,
        sequences: Sequence[LiteralRun],
    ) -> Optional[str]:
        if not sequences:
            return None
        entry_offset = min(program.blocks) if program.blocks else 0
        raw_candidates: List[Tuple[LiteralRun, str, int]] = []
        frequency: Counter[str] = Counter()
        for sequence in sequences:
            if sequence.kind != "string":
                continue
            text = (sequence.combined_string() or "").strip()
            if text and not any(ch.isspace() for ch in text):
                sanitized = _sanitize_identifier(text)
                if sanitized and sanitized.lower() not in _STRING_NAME_STOPWORDS:
                    key = sanitized.lower()
                    raw_candidates.append((sequence, sanitized, sequence.start_offset()))
                    frequency[key] += 1
            for token, token_offset in _identifier_tokens(sequence):
                sanitized = _sanitize_identifier(token)
                if not sanitized:
                    continue
                lowered = sanitized.lower()
                if lowered in _STRING_NAME_STOPWORDS:
                    continue
                raw_candidates.append((sequence, sanitized, token_offset))
                frequency[lowered] += 1
        if not raw_candidates:
            return None
        candidates: List[Tuple[Tuple[int, int, int, int, int, str], str]] = []
        seen: set[str] = set()
        for sequence, sanitized, offset in raw_candidates:
            lowered = sanitized.lower()
            if sanitized in seen:
                continue
            count = frequency[lowered]
            score = self._score_name_candidate(
                sequence,
                sanitized,
                entry_offset,
                count=count,
                candidate_offset=offset,
            )
            candidates.append((score, sanitized))
            seen.add(sanitized)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        selected = candidates[0][1]
        if selected.lower() in _LUA_KEYWORDS:
            selected = f"{selected}_fn"
        return selected

    def _score_name_candidate(
        self,
        sequence: LiteralRun,
        sanitized: str,
        entry_offset: int,
        *,
        count: int = 1,
        candidate_offset: Optional[int] = None,
    ) -> Tuple[int, int, int, int, int, str]:
        offset = candidate_offset if candidate_offset is not None else sequence.start_offset()
        distance = max(0, offset - entry_offset)
        underscore_penalty = 0 if "_" in sanitized else 1
        lower = sum(1 for ch in sanitized if ch.islower())
        upper = sum(1 for ch in sanitized if ch.isupper())
        if lower and upper:
            case_penalty = 0
        elif lower:
            case_penalty = 1
        elif upper:
            case_penalty = 2
        else:
            case_penalty = 3
        digit_penalty = sum(1 for ch in sanitized if ch.isdigit())
        length_penalty = len(sanitized)
        tie_breaker = sanitized.lower()
        return (
            -count,
            underscore_penalty,
            distance,
            case_penalty,
            digit_penalty,
            length_penalty,
            tie_breaker,
        )

    def _unique_function_name(self, base: str) -> str:
        candidate = base
        index = 1
        while candidate in self._used_function_names:
            candidate = f"{base}_{index}"
            index += 1
        self._used_function_names.add(candidate)
        return candidate

_LUA_KEYWORDS = {
    "and",
    "break",
    "do",
    "else",
    "elseif",
    "end",
    "false",
    "for",
    "function",
    "goto",
    "if",
    "in",
    "local",
    "nil",
    "not",
    "or",
    "repeat",
    "return",
    "then",
    "true",
    "until",
    "while",
}

_STRING_NAME_STOPWORDS = {"usage", "warning"}


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _identifier_tokens(
    sequence: LiteralRun,
) -> Iterable[Tuple[str, int]]:
    if sequence.kind != "string":
        return []
    text = sequence.combined_string() or ""
    if not text:
        return []
    offsets = sequence.offsets
    limit = len(offsets) - 1
    results: List[Tuple[str, int]] = []
    for match in _IDENTIFIER_PATTERN.finditer(text):
        start = match.start()
        # Align tokens to instruction boundaries to avoid partial fragments.
        if start % 2 != 0:
            continue
        chunk_index = start // 2
        if chunk_index > limit:
            chunk_index = limit
        token = match.group(0)
        underscore_index = token.find("_")
        include_full = True
        if underscore_index != -1 and any(
            ch.isupper() for ch in token[underscore_index + 1 :]
        ):
            include_full = False
        if include_full:
            results.append((token, offsets[chunk_index]))
        for rel_start, sub in _split_identifier_subtokens(token):
            if rel_start == 0 and include_full:
                continue
            absolute = start + rel_start
            if absolute % 2 != 0:
                continue
            sub_index = absolute // 2
            if sub_index > limit:
                sub_index = limit
            results.append((sub, offsets[sub_index]))
    return results


def _split_identifier_subtokens(token: str) -> List[Tuple[int, str]]:
    parts: List[Tuple[int, str]] = []
    start = 0
    for index in range(1, len(token)):
        if token[index].isupper() and token[index - 1].islower():
            if index - start >= 3:
                parts.append((start, token[start:index]))
            start = index
    if len(token) - start >= 3:
        parts.append((start, token[start:]))
    return parts


def _sanitize_identifier(text: str) -> Optional[str]:
    if not text:
        return None
    pieces: List[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            pieces.append(char)
        else:
            pieces.append("_")
    candidate = "".join(pieces)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    if not candidate:
        return None
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
