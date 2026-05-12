from __future__ import annotations

"""High-level pseudo-source generation for MBC scripts.

This is the orchestration layer: project loading, local-helper discovery,
per-program CFG traversal, stack-AST building, and final text assembly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from mbc_format.bytecode import MbcControlFlow, MbcDecoder
from .linker import MbcProjectLinker, MbcStaticLinker
from mbc_format.loader import MbcLoader, MbcProgram, MbcProject, MbcScript
from .vm_ast import SymbolicDataMemory, build_program_ast, pseudo_type_name

LOCAL_HELPER_PREFIX = "local"


def helper_name(offset: int) -> str:
    return f"{LOCAL_HELPER_PREFIX}_{offset:08X}"


@dataclass(frozen=True)
class LocalHelper:
    offset: int
    name: str
    owner_index: int
    program: MbcProgram


@dataclass
class LocalHelperIndex:
    helpers: dict[int, LocalHelper] = field(default_factory=dict)

    @property
    def names(self) -> dict[int, str]:
        return {offset: helper.name for offset, helper in self.helpers.items()}

    @property
    def stop_offsets(self) -> set[int]:
        return set(self.helpers)

    def by_owner(self) -> dict[int, list[LocalHelper]]:
        grouped: dict[int, list[LocalHelper]] = {}
        for helper in self.helpers.values():
            grouped.setdefault(helper.owner_index, []).append(helper)
        for items in grouped.values():
            items.sort(key=lambda item: item.offset)
        return grouped

    def annotate_call_names(self, instructions: Iterable[object]) -> None:
        names = self.names
        for ins in instructions:
            if getattr(ins, "mnemonic", None) != "call_rel32":
                continue
            operands = getattr(ins, "operands", None)
            if not isinstance(operands, dict):
                continue
            target = operands.get("target")
            if isinstance(target, int) and target in names:
                operands["target_name"] = names[target]


@dataclass
class DataUse:
    offset: int
    owners: set[int] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    type_ids: set[int] = field(default_factory=set)
    length: int | None = None

    def add(self, *, owner_index: int, role: str, type_id: int | None, length: int | None = None) -> None:
        self.owners.add(owner_index)
        self.roles.add(role)
        if isinstance(type_id, int):
            self.type_ids.add(type_id)
        if isinstance(length, int) and length > 0:
            self.length = length if self.length is None else max(self.length, length)

    @property
    def primary_role(self) -> str:
        if "array" in self.roles:
            return "array"
        if "span" in self.roles:
            return "span"
        return "data"

    @property
    def primary_type_id(self) -> int | None:
        if not self.type_ids:
            return None
        return sorted(self.type_ids)[0]


def _instruction_data_uses(ins: object) -> Iterable[tuple[int, str, int | None, int | None]]:
    mnemonic = getattr(ins, "mnemonic", "")
    operands = getattr(ins, "operands", None)
    if not isinstance(operands, dict):
        return

    if mnemonic == "push_data_ref":
        off = operands.get("data_offset")
        if isinstance(off, int):
            yield off, "data", operands.get("type") if isinstance(operands.get("type"), int) else None, None
        return

    if mnemonic in {"push_typed_span_ref", "push_inline_typed_span"}:
        off = operands.get("data_offset")
        length = operands.get("length")
        if isinstance(off, int):
            yield off, "span", operands.get("type") if isinstance(operands.get("type"), int) else None, length if isinstance(length, int) else None
        return

    if mnemonic == "array_index_abs":
        base = operands.get("base")
        count = operands.get("count")
        if isinstance(base, int):
            array_count = abs(count) if isinstance(count, int) and count != 0 else None
            yield base, "array", operands.get("type") if isinstance(operands.get("type"), int) else None, array_count
        return


def collect_data_scope_index(
    script: MbcScript,
    flow: MbcControlFlow,
    local_index: LocalHelperIndex,
    *,
    stop_offsets: set[int],
) -> tuple[dict[int, DataUse], dict[int, str]]:
    """Infer which data-section slots are program-local and which are shared.

    MBC bytecode stores both original locals and true module globals in the same
    data section.  Without a debug local table, the most reliable recoverable
    distinction is ownership: a slot referenced only by one program/helper owner
    is rendered as local, while a slot referenced by several owners is rendered
    as global.
    """
    usage: dict[int, DataUse] = {}
    helpers_by_owner = local_index.by_owner()

    def record(entry_program: MbcProgram, owner_index: int) -> None:
        instructions = flow.decode_program(entry_program, follow_local_calls=False, stop_offsets=stop_offsets)
        for ins in instructions:
            for offset, role, type_id, length in _instruction_data_uses(ins):
                usage.setdefault(offset, DataUse(offset)).add(
                    owner_index=owner_index,
                    role=role,
                    type_id=type_id,
                    length=length,
                )

    for program in script.programs:
        record(program, program.index)
        for helper in helpers_by_owner.get(program.index, []):
            record(helper.program, helper.owner_index)

    for helper in helpers_by_owner.get(-1, []):
        record(synthetic_helper_program(script, helper.offset, None), helper.owner_index)

    scope_map: dict[int, str] = {}
    for offset, use in usage.items():
        scope_map[offset] = "local" if len(use.owners) == 1 and next(iter(use.owners)) >= 0 else "global"
    return usage, scope_map


def render_global_data_declarations(script: MbcScript, usage: dict[int, DataUse], scope_map: dict[int, str]) -> list[str]:
    memory = SymbolicDataMemory(module_name=script.path.stem, data=script.data, scope_map=scope_map)
    for offset, use in sorted(usage.items()):
        if scope_map.get(offset) != "global":
            continue
        memory.location(
            offset=offset,
            type_id=use.primary_type_id,
            role=use.primary_role,
            length=use.length,
        )
    declarations = memory.declarations(scope="global")
    if not declarations:
        return []
    lines = ["globals", "{"]
    lines.extend(f"    {SymbolicDataMemory.declaration_text(loc)}" for loc in declarations)
    lines.extend(["}", ""])
    return lines


def synthetic_helper_program(script: MbcScript, entry: int, owner: MbcProgram | None = None) -> MbcProgram:
    """Create a synthetic program-table-like record for a local helper entry."""
    if owner is None:
        owner = script.program_for_offset(entry)
    if owner is not None:
        end = owner.end if owner.end >= entry else len(script.code) - 1
        return MbcProgram(owner.index, helper_name(entry), entry, end, owner.state_raw, owner.queue_id, owner.unknown_48)
    return MbcProgram(-1, helper_name(entry), entry, len(script.code) - 1, 0, 0, 0)


def discover_local_helpers(script: MbcScript, flow: MbcControlFlow, linker: MbcStaticLinker) -> LocalHelperIndex:
    """Find anonymous same-script call targets and expose them as local helpers.

    We intentionally do not infer semantic names here.  The only commitment is
    structural: a direct call target that is neither a named program entry nor an
    import/native stub should be rendered as a separate local routine instead of
    being inlined into the caller's linear stream.
    """
    named_starts = {program.start for program in script.programs}
    entries_by_offset: dict[int, MbcProgram] = {program.start: program for program in script.programs}
    helpers: dict[int, LocalHelper] = {}
    seen_entries: set[int] = set()
    queue: list[MbcProgram] = list(script.programs)

    while queue:
        entry_program = queue.pop(0)
        entry = entry_program.start
        if entry in seen_entries or not (0 <= entry < len(script.code)):
            continue
        seen_entries.add(entry)

        stop_offsets = set(entries_by_offset)
        instructions = flow.decode_program(entry_program, follow_local_calls=False, stop_offsets=stop_offsets)

        for ins in instructions:
            if ins.mnemonic != "call_rel32":
                continue
            target = ins.operands.get("target")
            if not isinstance(target, int) or not (0 <= target < len(script.code)):
                continue
            if target in named_starts or linker.import_stub_at(target) is not None:
                continue

            target_owner = script.program_for_offset(target)
            if target_owner is not None and entry_program.index >= 0 and target_owner.index != entry_program.index:
                # A call into another named program range is not a local helper of
                # the current body.  Leave it as an ordinary call edge.
                continue

            if target in entries_by_offset:
                continue

            owner = target_owner or (entry_program if entry_program.index >= 0 else None)
            helper_program = synthetic_helper_program(script, target, owner)
            entries_by_offset[target] = helper_program
            helper = LocalHelper(
                offset=target,
                name=helper_name(target),
                owner_index=owner.index if owner is not None else -1,
                program=helper_program,
            )
            helpers[target] = helper
            queue.append(helper_program)

    return LocalHelperIndex(helpers=helpers)

def load_project_for(mbc_path: Path) -> tuple[MbcProject, MbcScript, MbcProjectLinker]:
    project = MbcProject.load_for_script(mbc_path)
    script = project.by_module.get(mbc_path.stem)
    if script is None:
        # Fallback for unusual paths outside the default mbc/ tree.
        script = MbcLoader.load(mbc_path)
        project = MbcProject(root=mbc_path.parent, scripts=[script])
    project_linker = MbcProjectLinker.from_ffprc_plan(project.scripts)
    return project, script, project_linker


def _signature_text(symbol: object | None) -> str:
    signature = getattr(symbol, "signature", None)
    if signature is None or getattr(signature, "source", "unknown") == "unknown" and not getattr(signature, "args", ()):  # type: ignore[truthy-function]
        return "()"
    parts = [f"{pseudo_type_name(arg.type_id)} {arg.name}" for arg in signature.args]
    if signature.variadic:
        parts.append("...")
    ret = "" if signature.return_type == "unknown" else f" -> {signature.return_type}"
    return f"({', '.join(parts)}){ret}"


def _program_header(program: MbcProgram, linker: MbcStaticLinker, *, local_helper: LocalHelper | None = None) -> str:
    if local_helper is not None:
        return f"function {local_helper.name}()"

    symbol = linker.internal_at(program.start) or linker.symbol_at(program.start)
    name = program.name or (symbol.name if symbol else f"program_{program.index}")
    return f"function {name}{_signature_text(symbol)}"


@dataclass(frozen=True)
class DecompileDocument:
    """Rendered pseudo-source plus source-line → bytecode-offset anchors.

    The plain text output remains compatible with the old decompiler API, but
    the editor can now use ``line_offsets`` instead of guessing that every line
    inside a function belongs to the function prologue.  Multi-line structured
    statements keep the offset of the branch/statement that produced them; this
    is still conservative, but it is good enough for navigation and safe value
    patch candidates.
    """

    text: str
    line_offsets: dict[int, int]
    function_ranges: list[tuple[int, int, int, int, str]]


def decompile_to_text(script: MbcScript, *, project_linker: MbcProjectLinker | None = None) -> str:
    return decompile_to_document(script, project_linker=project_linker).text


def decompile_to_document(script: MbcScript, *, project_linker: MbcProjectLinker | None = None) -> DecompileDocument:
    linker = (project_linker.module(script.path.stem) if project_linker is not None else None) or MbcStaticLinker(script)
    decoder = MbcDecoder(script, linker=linker, cache_decodes=True)
    flow = MbcControlFlow(script, decoder=decoder)
    local_index = discover_local_helpers(script, flow, linker)
    helpers_by_owner = local_index.by_owner()
    stop_offsets = {program.start for program in script.programs} | local_index.stop_offsets
    data_usage, data_scope_map = collect_data_scope_index(script, flow, local_index, stop_offsets=stop_offsets)

    chunks: list[str] = []
    line_offsets: dict[int, int] = {}
    function_ranges: list[tuple[int, int, int, int, str]] = []

    def append(line: str = "", *, offset: int | None = None) -> int:
        chunks.append(line)
        line_no = len(chunks)
        if offset is not None:
            line_offsets[line_no] = int(offset)
        return line_no

    for line in [
        "// Experimental MBC pseudo-source",
        f"// source: {script.path.name}",
        f"// programs: {len(script.programs)}",
        "// data naming: argN = program_prologue binding; v_XXXX/buf_XXXX/rec_XXXX = local slots; g_XXXX/g_buf_XXXX/g_rec_XXXX = shared slots",
        "// synthetic call returns are now kept as expressions, not ret_* placeholder variables",
        "// local call targets are split into local_XXXXXXXX() helpers instead of being inlined into caller bodies",
        "// coroutine model: yield_program() suspends the current scheduler slice and the following loc_XXXXXXXX label is the saved-PC resume point",
        "// control-flow structuring: if/else/else-if/switch/while recovery with folded short-circuit boolean chains; unresolved or coroutine-sensitive branches remain as labels/gotos",
    ]:
        append(line)
    if local_index.helpers:
        append(f"// local helpers: {len(local_index.helpers)}")
    if project_linker is not None:
        for line in project_linker.summary_lines():
            append(line)
    for line in linker.summary_lines():
        append(line)
    append("")
    for line in render_global_data_declarations(script, data_usage, data_scope_map):
        append(line)

    def render_entry(entry_program: MbcProgram, *, local_helper: LocalHelper | None = None) -> None:
        header_line = append(_program_header(entry_program, linker, local_helper=local_helper), offset=entry_program.start)
        append("{", offset=entry_program.start)

        if not (0 <= entry_program.start < len(script.code)):
            append("    // warning: program start is outside code section", offset=entry_program.start)
        elif entry_program.end < entry_program.start:
            append("    // warning: program end is before start", offset=entry_program.start)
        else:
            instructions = flow.decode_program(entry_program, follow_local_calls=False, stop_offsets=stop_offsets)
            local_index.annotate_call_names(instructions)
            ast = build_program_ast(script, entry_program, instructions, linker=linker, scope_map=data_scope_map)
            statements = ast.get("statements")
            if isinstance(statements, list) and statements:
                for stmt in statements:
                    if not isinstance(stmt, dict):
                        continue
                    stmt_offset = stmt.get("offset")
                    if not isinstance(stmt_offset, int):
                        stmt_offset = entry_program.start
                    text = str(stmt.get("text", ""))
                    if not text:
                        continue
                    for line in text.splitlines():
                        append(f"    {line}" if line else "", offset=stmt_offset)
            else:
                body = str(ast.get("source", "")) or "// no decoded statements"
                for line in body.splitlines():
                    append(f"    {line}" if line else "", offset=entry_program.start)

        end_line = append("}", offset=entry_program.start)
        append("")
        function_ranges.append((header_line, end_line, entry_program.start, entry_program.end, entry_program.name))

    for program in script.programs:
        render_entry(program)
        for helper in helpers_by_owner.get(program.index, []):
            render_entry(helper.program, local_helper=helper)

    for helper in helpers_by_owner.get(-1, []):
        render_entry(synthetic_helper_program(script, helper.offset, None), local_helper=helper)

    return DecompileDocument(
        text="\n".join(chunks).rstrip() + "\n",
        line_offsets=line_offsets,
        function_ranges=function_ranges,
    )
