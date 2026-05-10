from __future__ import annotations

"""Runtime-style linker model for MBC scripts.

The real VM does not treat every .mbc as an isolated unit: import records in one
script are patched at runtime against function-table records in other loaded
scripts.  This module keeps that behaviour symbolic.  It builds per-module
function symbols, a project-wide provider table, signatures recovered from
``program_prologue``, and virtual patch records for 0x67 import stubs without
mutating bytecode.
"""

from dataclasses import dataclass, field
import re
from typing import Iterable, Optional, Sequence

from .bytecode import MbcControlFlow, MbcDecoder
from .loader import MbcFunction, MbcProgram, MbcScript
from .calls import NativeCallSpec, engine_native_import
from .opcodes import decode_opcode
from .common import CODE_FILE_OFFSET, type_name as _type_name


LINK_PATCH_OPCODE = 0x47  # 'G' / jmp_rel32


@dataclass(frozen=True)
class FunctionArg:
    index: int
    name: str
    type_id: int
    type_name: str
    data_offset: int

    @property
    def storage(self) -> str:
        return f"data[0x{self.data_offset:X}]"

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "data_offset": self.data_offset,
            "storage": self.storage,
        }


@dataclass(frozen=True)
class FunctionSignature:
    args: tuple[FunctionArg, ...] = ()
    return_type: str = "unknown"
    return_type_id: int | None = None
    variadic: bool = False
    source: str = "unknown"

    @property
    def arity(self) -> int:
        return len(self.args)

    @classmethod
    def unknown(cls) -> "FunctionSignature":
        return cls(source="unknown")

    @classmethod
    def from_prologue(cls, descriptors: list[dict[str, object]], *, signed_count: int) -> "FunctionSignature":
        args: list[FunctionArg] = []
        for idx, desc in enumerate(descriptors):
            type_id = int(desc.get("type", -1))
            args.append(
                FunctionArg(
                    index=idx,
                    name=f"arg{idx}",
                    type_id=type_id,
                    type_name=str(desc.get("type_name") or _type_name(type_id)),
                    data_offset=int(desc.get("data_offset", 0)),
                )
            )
        return cls(
            args=tuple(args),
            return_type="unknown",
            return_type_id=None,
            variadic=signed_count < 0,
            source="program_prologue",
        )

    def render(self, *, include_storage: bool = False) -> str:
        parts: list[str] = []
        for arg in self.args:
            text = f"{arg.type_name} {arg.name}"
            if include_storage:
                text += f" @ {arg.storage}"
            parts.append(text)
        if self.variadic:
            parts.append("...")
        ret = "" if self.return_type == "unknown" else f" -> {self.return_type}"
        return f"({', '.join(parts)}){ret}"

    def to_dict(self) -> dict[str, object]:
        return {
            "arity": self.arity,
            "variadic": self.variadic,
            "return_type": self.return_type,
            "return_type_id": self.return_type_id,
            "source": self.source,
            "args": [arg.to_dict() for arg in self.args],
        }


@dataclass(frozen=True)
class MbcFunctionSymbol:
    index: int
    name: str
    code_offset: int
    program_index: int
    flags_or_module: int
    program_name: Optional[str]
    import_stub_payload: Optional[int]
    module_name: str
    script_path: str
    signature: FunctionSignature = field(default_factory=FunctionSignature.unknown)

    @classmethod
    def from_function(cls, script: MbcScript, fn: MbcFunction) -> "MbcFunctionSymbol":
        program_index = fn.program_index
        program_name: Optional[str] = None
        if 0 <= program_index < len(script.programs):
            program_name = script.programs[program_index].name

        payload: Optional[int] = None
        if 0 <= fn.code_offset + 5 <= len(script.code) and script.code[fn.code_offset] == 0x67:
            payload = int.from_bytes(script.code[fn.code_offset + 1:fn.code_offset + 5], "little", signed=False)

        signature = _signature_from_function(script, fn) if program_index >= 0 else FunctionSignature.unknown()

        return cls(
            index=fn.index,
            name=fn.name,
            code_offset=fn.code_offset,
            program_index=program_index,
            flags_or_module=fn.flags_or_module,
            program_name=program_name,
            import_stub_payload=payload,
            module_name=script.path.stem,
            script_path=str(script.path),
            signature=signature,
        )

    @property
    def is_import(self) -> bool:
        return self.program_index < 0

    @property
    def is_internal(self) -> bool:
        return self.program_index >= 0

    @property
    def kind(self) -> str:
        return "external_import" if self.is_import else "internal"

    @property
    def qualified_name(self) -> str:
        return f"{self.module_name}.{self.name}"

    @property
    def file_offset(self) -> int:
        return self.code_offset + CODE_FILE_OFFSET

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind,
            "module": self.module_name,
            "script_path": self.script_path,
            "code_offset": self.code_offset,
            "file_offset": self.file_offset,
            "program_index": self.program_index,
            "program_name": self.program_name,
            "flags_or_module": self.flags_or_module,
            "import_stub_payload": self.import_stub_payload,
            "signature": self.signature.to_dict(),
        }


def _signature_from_function(script: MbcScript, fn: MbcFunction) -> FunctionSignature:
    if not (0 <= fn.code_offset < len(script.code)):
        return FunctionSignature.unknown()
    try:
        decoded = decode_opcode(script.code, fn.code_offset)
    except Exception:
        return FunctionSignature.unknown()
    if decoded.mnemonic != "program_prologue":
        return FunctionSignature.unknown()
    return FunctionSignature.from_prologue(
        decoded.operands.get("descriptors", []),
        signed_count=int(decoded.operands.get("signed_count", 0)),
    )


_NATIVE_TYPE_IDS = {
    "i8/char": 0x00,
    "char": 0x00,
    "span/string": 0x01,
    "string": 0x01,
    "int32": 0x10,
    "float32": 0x20,
    "slice": 0x30,
    "slice/span": 0x30,
    "slice_descriptor": 0x30,
}


def _signature_from_native_spec(spec: NativeCallSpec) -> FunctionSignature:
    arity = spec.arity if spec.arity is not None else len(spec.arg_types)
    args: list[FunctionArg] = []
    for idx in range(arity):
        type_name = spec.arg_types[idx] if idx < len(spec.arg_types) else "unknown"
        args.append(
            FunctionArg(
                index=idx,
                name=f"arg{idx}",
                type_id=_NATIVE_TYPE_IDS.get(type_name, -1),
                type_name=type_name,
                data_offset=0,
            )
        )
    return FunctionSignature(
        args=tuple(args),
        return_type=spec.return_type,
        return_type_id=spec.return_type_id,
        variadic=spec.arity is None,
        source=spec.layer,
    )


@dataclass(frozen=True)
class LinkPatch:
    source_module: str
    source_name: str
    source_function_index: int
    source_offset: int
    target_module: str
    target_name: str
    target_function_index: int
    target_program_index: int
    target_program_name: Optional[str]
    target_offset: int
    opcode: int = LINK_PATCH_OPCODE
    source_runtime_offset: int | None = None
    target_runtime_offset: int | None = None
    source_module_base: int = 0
    target_module_base: int = 0

    @property
    def source_patch_offset(self) -> int:
        return self.source_runtime_offset if self.source_runtime_offset is not None else self.source_offset

    @property
    def target_patch_offset(self) -> int:
        return self.target_runtime_offset if self.target_runtime_offset is not None else self.target_offset

    @property
    def rel32(self) -> int:
        # Engine rule from sub_4784C0: write target_runtime - source_runtime - 1
        # after opcode 0x47. Branch decoding also uses off + 1 as PC base.
        return self.target_patch_offset - self.source_patch_offset - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "source_module": self.source_module,
            "source_name": self.source_name,
            "source_function_index": self.source_function_index,
            "source_offset": self.source_offset,
            "source_runtime_offset": self.source_runtime_offset,
            "source_module_base": self.source_module_base,
            "target_module": self.target_module,
            "target_name": self.target_name,
            "target_function_index": self.target_function_index,
            "target_program_index": self.target_program_index,
            "target_program_name": self.target_program_name,
            "target_offset": self.target_offset,
            "target_runtime_offset": self.target_runtime_offset,
            "target_module_base": self.target_module_base,
            "opcode": self.opcode,
            "rel32": self.rel32,
        }


@dataclass(frozen=True)
class RuntimeResolvedLink:
    source: MbcFunctionSymbol
    target: MbcFunctionSymbol
    patch: LinkPatch
    ambiguity: int = 1
    alternatives: tuple[str, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguity > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "patch": self.patch.to_dict(),
            "ambiguity": self.ambiguity,
            "alternatives": list(self.alternatives),
            "signature": self.target.signature.to_dict(),
        }


@dataclass(frozen=True)
class RuntimeNativeLink:
    source: MbcFunctionSymbol
    spec: NativeCallSpec

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "native": self.spec.to_dict(),
            "signature": _signature_from_native_spec(self.spec).to_dict(),
        }


@dataclass(frozen=True)
class RuntimeModuleState:
    """One module after it has been merged into the VM process context.

    ``sub_4784C0`` does not resolve externs against a global directory.  It
    appends one MBC into the currently running process and compares only records
    already present in that process plus the newly appended records.  Offsets in
    patches are therefore process-code offsets, not the original per-file code
    offsets.
    """

    module_name: str
    linker: "MbcStaticLinker"
    load_index: int
    code_base: int
    program_base: int
    function_base: int

    @property
    def code_end(self) -> int:
        return self.code_base + self.linker.script.header.code_size

    @property
    def program_end(self) -> int:
        return self.program_base + len(self.linker.script.programs)

    @property
    def function_end(self) -> int:
        return self.function_base + len(self.linker.symbols)

    def runtime_offset(self, symbol: MbcFunctionSymbol) -> int:
        return self.code_base + symbol.code_offset

    def runtime_program_index(self, symbol: MbcFunctionSymbol) -> int:
        return symbol.program_index + self.program_base if symbol.is_internal else symbol.program_index

    def to_dict(self) -> dict[str, object]:
        return {
            "module": self.module_name,
            "load_index": self.load_index,
            "code_base": self.code_base,
            "code_end": self.code_end,
            "program_base": self.program_base,
            "program_end": self.program_end,
            "function_base": self.function_base,
            "function_end": self.function_end,
        }



FFPRC_LINK_MNEMONIC = "ffprc_link"
SPRINTF_MNEMONICS = {"sprintf", "snprintf"}

# ffprc_link targets that are used as process/module names by bytecode but
# are not present as .mbc files in the supplied corpus.  The quickfile lookup
# path does not expose a hard-coded alias for them; keep them as missing
# process dependencies instead of folding them into ordinary MBC module order
# or native function imports.
MISSING_PROCESS_MODULES: dict[str, str] = {
    "_cobj": "missing common-object process dependency; bytecode calls ffprc_link(\"_cobj\"), quickfile lookup has no hard-coded alias/normalization in the audited slice, and no _cobj.mbc/function provider exists in the supplied corpus",
}


@dataclass(frozen=True)
class _FfprcArgExpr:
    kind: str
    expr: str
    literal: str | None = None
    data_offset: int | None = None
    length: int | None = None
    type_id: int | None = None


@dataclass(frozen=True)
class FfprcLinkSite:
    source_module: str
    program_index: int | None
    program_name: str | None
    offset: int
    file_offset: int
    order_index: int
    argc: int | None
    target_expr: str
    literal_target: str | None = None
    format_string: str | None = None
    target_modules: tuple[str, ...] = ()
    confidence: str = "unknown"
    reachable: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source_module": self.source_module,
            "program_index": self.program_index,
            "program_name": self.program_name,
            "offset": self.offset,
            "file_offset": self.file_offset,
            "order_index": self.order_index,
            "argc": self.argc,
            "target_expr": self.target_expr,
            "literal_target": self.literal_target,
            "format_string": self.format_string,
            "target_modules": list(self.target_modules),
            "confidence": self.confidence,
            "reachable": self.reachable,
            "note": self.note,
        }


@dataclass(frozen=True)
class ProcessPlanDeviation:
    kind: str
    module_name: str
    offset: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "module_name": self.module_name,
            "offset": self.offset,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MbcProcessPlan:
    """Recovered runtime load order for a project.

    The plan is built from reachable ``ffprc_link`` calls and then consumed by
    :class:`MbcProjectLinker`; it lives in the linker layer because it decides
    inter-module patch order, not high-level source rendering.
    """

    root_modules: tuple[str, ...]
    ordered_modules: tuple[str, ...]
    fallback_modules: tuple[str, ...]
    link_sites: tuple[FfprcLinkSite, ...]
    selected_sites: tuple[FfprcLinkSite, ...]
    deviations: tuple[ProcessPlanDeviation, ...] = ()

    @property
    def planned_modules(self) -> tuple[str, ...]:
        return self.ordered_modules

    @property
    def has_fallback(self) -> bool:
        return bool(self.fallback_modules)

    @property
    def dynamic_sites(self) -> tuple[FfprcLinkSite, ...]:
        return tuple(site for site in self.link_sites if site.confidence == "dynamic_name")

    @property
    def unresolved_sites(self) -> tuple[FfprcLinkSite, ...]:
        return tuple(site for site in self.link_sites if site.confidence in {"literal_unresolved", "unknown"})

    def scripts_in_order(self, scripts: Iterable[MbcScript], *, include_fallback: bool = True) -> list[MbcScript]:
        by_module = {script.path.stem: script for script in scripts}
        names = list(self.ordered_modules)
        if include_fallback:
            names.extend(name for name in self.fallback_modules if name not in names)
        return [by_module[name] for name in names if name in by_module]

    def summary_lines(self) -> list[str]:
        literal = sum(1 for site in self.link_sites if site.confidence == "literal")
        dynamic = sum(1 for site in self.link_sites if site.confidence == "dynamic_name")
        missing_process = sum(1 for site in self.link_sites if site.confidence == "missing_process_module")
        unresolved = sum(1 for site in self.link_sites if site.confidence in {"literal_unresolved", "unknown"})
        unreachable = sum(1 for site in self.link_sites if not site.reachable)
        lines = [
            f"// ffprc process plan roots: {', '.join(self.root_modules) if self.root_modules else '<none>'}",
            f"// ffprc link sites: {len(self.link_sites)} total, {literal} literal, {dynamic} dynamic, {missing_process} missing-process, {unresolved} unresolved/unknown, {unreachable} unreachable",
            f"// ffprc planned modules: {len(self.ordered_modules)} ordered, {len(self.fallback_modules)} fallback",
        ]
        if self.deviations:
            lines.append(f"// ffprc plan deviations: {len(self.deviations)}")
            for dev in self.deviations[:16]:
                loc = "" if dev.offset is None else f" @ 0x{dev.offset:08X}"
                lines.append(f"// deviation {dev.kind}: {dev.module_name}{loc}: {dev.detail}")
            if len(self.deviations) > 16:
                lines.append(f"// ... {len(self.deviations) - 16} more deviations")
        return lines

    def to_dict(self) -> dict[str, object]:
        return {
            "root_modules": list(self.root_modules),
            "ordered_modules": list(self.ordered_modules),
            "fallback_modules": list(self.fallback_modules),
            "link_sites": [site.to_dict() for site in self.link_sites],
            "selected_sites": [site.to_dict() for site in self.selected_sites],
            "deviations": [dev.to_dict() for dev in self.deviations],
        }


class _FfprcLinkExtractor:
    """Minimal call-argument tracker for process-link calls.

    It reuses ``MbcControlFlow``/``MbcDecoder`` for reachability and opcode
    annotation.  The local stack kept here is deliberately narrow: enough to
    identify literal script names and sprintf-derived names passed into
    ``ffprc_link``.
    """

    def __init__(self, script: MbcScript, *, available_modules: set[str] | None = None, reachable_only: bool = True):
        self.script = script
        self.available_modules = available_modules or set()
        self.available_modules_sorted = tuple(sorted(self.available_modules))
        self.available_by_quickfile_key = _available_modules_by_quickfile_key(self.available_modules_sorted)
        self.reachable_only = reachable_only
        self.stack: list[_FfprcArgExpr] = []
        self.last_arg_count: int | None = None
        self.last_args: list[_FfprcArgExpr] = []
        self.formatted_buffers: dict[int, str] = {}
        self.sites: list[FfprcLinkSite] = []
        self.order_index = 0
        self._seen_site_offsets: set[int] = set()
        self._span_text_cache: dict[tuple[int, int | None], str | None] = {}
        self._literal_target_cache: dict[str, tuple[str, ...]] = {}
        self._printf_target_cache: dict[str, tuple[str, ...]] = {}

    def run(self) -> list[FfprcLinkSite]:
        # Process-plan extraction only needs decoded mnemonics/operands and a
        # cheap import-stub stop set.  Full linker annotations would resolve
        # symbols and serialize signatures for every visited instruction.
        import_stub_offsets = {fn.code_offset for fn in self.script.functions if fn.is_import}
        decoder = MbcDecoder(
            self.script,
            annotate_linkage=False,
            import_stub_offsets=import_stub_offsets,
            cache_decodes=True,
        )
        flow = MbcControlFlow(self.script, decoder=decoder)
        for program in sorted(self.script.programs, key=lambda p: (p.start, p.index)):
            self._scan_program(flow, program)
        return self.sites

    def _resolve_literal_targets(self, literal: str) -> tuple[str, ...]:
        if literal not in self._literal_target_cache:
            self._literal_target_cache[literal] = _resolve_literal_target_modules(
                literal,
                self.available_modules,
                available_by_key=self.available_by_quickfile_key,
            )
        return self._literal_target_cache[literal]

    def _resolve_printf_targets(self, fmt: str) -> tuple[str, ...]:
        if fmt not in self._printf_target_cache:
            self._printf_target_cache[fmt] = _resolve_printf_target_modules(
                fmt,
                self.available_modules_sorted,
            )
        return self._printf_target_cache[fmt]

    def _read_span_text_cached(self, data_offset: int, length: int | None) -> str | None:
        key = (data_offset, length)
        if key not in self._span_text_cache:
            self._span_text_cache[key] = _read_span_text(self.script, data_offset, length)
        return self._span_text_cache[key]

    def _scan_program(self, flow: MbcControlFlow, program: MbcProgram) -> None:
        self.stack.clear()
        self.last_arg_count = None
        self.last_args = []
        self.formatted_buffers.clear()

        if self.reachable_only:
            instructions = flow.decode_program(program, follow_local_calls=True)
        else:
            instructions = self._linear_program_instructions(flow.decoder, program)

        for ins in instructions:
            self._accept(program, ins)

    def _linear_program_instructions(self, decoder: MbcDecoder, program: MbcProgram) -> list[object]:
        instructions: list[object] = []
        off = max(0, program.start)
        end = min(len(self.script.code), max(program.end + 1, program.start + 1))
        seen = 0
        while off < end and seen < len(self.script.code):
            seen += 1
            ins = decoder.decode_at(off, program)
            instructions.append(ins)
            off += max(ins.length, 1)
        return instructions

    def _accept(self, program: MbcProgram, ins: object) -> None:
        mnemonic = getattr(ins, "mnemonic", "")
        operands = getattr(ins, "operands", {}) or {}

        if mnemonic in {"push_inline_span", "push_inline_typed_span", "push_typed_span_ref"}:
            self.stack.append(self._span_expr(operands, mnemonic))
            return

        if mnemonic in {"push_imm_i8", "push_imm_u16", "push_imm32"}:
            value = operands.get("value_i32", operands.get("value", operands.get("value_u32")))
            self.stack.append(_FfprcArgExpr("int", str(value)))
            return

        if mnemonic == "push_data_ref":
            data_offset = operands.get("data_offset")
            type_id = operands.get("type")
            if isinstance(data_offset, int):
                self.stack.append(
                    _FfprcArgExpr(
                        "data_ref",
                        f"data[0x{data_offset:X}]",
                        data_offset=data_offset,
                        type_id=type_id if isinstance(type_id, int) else None,
                    )
                )
            else:
                self.stack.append(_FfprcArgExpr("unknown", "<data_ref>"))
            return

        if mnemonic == "set_arg_count":
            value = operands.get("value")
            if isinstance(value, int):
                self.last_arg_count = value
                self.last_args = self.stack[-value:] if value > 0 and len(self.stack) >= value else ([] if value == 0 else list(self.stack))
            else:
                self.last_arg_count = None
                self.last_args = []
            return

        if mnemonic in SPRINTF_MNEMONICS:
            args = list(self.last_args)
            if len(args) >= 2:
                dst, fmt = args[0], args[1]
                if dst.data_offset is not None and fmt.literal is not None:
                    self.formatted_buffers[dst.data_offset] = fmt.literal
            self._consume_args(push_result=False)
            return

        if mnemonic == FFPRC_LINK_MNEMONIC:
            off = getattr(ins, "offset", -1)
            if isinstance(off, int) and off in self._seen_site_offsets:
                self._consume_args(push_result=True)
                return
            args = list(self.last_args)
            site = self._site_for_args(program, ins, args)
            self.sites.append(site)
            if isinstance(off, int):
                self._seen_site_offsets.add(off)
            self.order_index += 1
            self._consume_args(push_result=True)
            return

        # Unknown calls/branches make the local stack unreliable.  Keep the
        # exact CFG in bytecode.py, and reset only the argument snapshot here.
        if mnemonic in {"return", "return_local", "end_program", "halt_interpreter", "yield_program"}:
            self.stack.clear()
            self.last_arg_count = None
            self.last_args = []
        elif mnemonic.startswith("j") or mnemonic == "call_rel32":
            self.last_arg_count = None
            self.last_args = []

    def _consume_args(self, *, push_result: bool) -> None:
        argc = self.last_arg_count
        if isinstance(argc, int) and argc > 0 and len(self.stack) >= argc:
            del self.stack[-argc:]
        if push_result:
            self.stack.append(_FfprcArgExpr("int", "<ffprc_result>"))
        self.last_arg_count = None
        self.last_args = []

    def _span_expr(self, operands: dict[str, object], mnemonic: str) -> _FfprcArgExpr:
        data_offset = operands.get("data_offset")
        length = operands.get("length")
        type_id = operands.get("type")
        if not isinstance(data_offset, int):
            return _FfprcArgExpr("unknown", f"<{mnemonic}>")
        literal = self._read_span_text_cached(data_offset, length if isinstance(length, int) else None)
        if literal is not None and literal:
            return _FfprcArgExpr(
                "literal",
                repr(literal),
                literal=literal,
                data_offset=data_offset,
                length=length if isinstance(length, int) else None,
                type_id=type_id if isinstance(type_id, int) else None,
            )
        end = data_offset + length if isinstance(length, int) else data_offset
        return _FfprcArgExpr(
            "span_ref",
            f"data[0x{data_offset:X}:0x{end:X}]" if isinstance(length, int) else f"data[0x{data_offset:X}]",
            data_offset=data_offset,
            length=length if isinstance(length, int) else None,
            type_id=type_id if isinstance(type_id, int) else None,
        )

    def _site_for_args(self, program: MbcProgram, ins: object, args: list[_FfprcArgExpr]) -> FfprcLinkSite:
        argc = self.last_arg_count
        off = int(getattr(ins, "offset", -1))
        file_offset = int(getattr(ins, "file_offset", off + CODE_FILE_OFFSET))
        first = args[0] if args else None
        if first is None:
            return FfprcLinkSite(
                source_module=self.script.path.stem,
                program_index=program.index,
                program_name=program.name,
                offset=off,
                file_offset=file_offset,
                order_index=self.order_index,
                argc=argc,
                target_expr="<missing-arg>",
                confidence="unknown",
                reachable=True,
                note="ffprc_link was reached without a recoverable first argument",
            )

        if first.literal is not None:
            targets = self._resolve_literal_targets(first.literal)
            if targets:
                confidence = "literal"
                note = "literal ffprc_link target"
            elif first.literal in MISSING_PROCESS_MODULES:
                confidence = "missing_process_module"
                note = MISSING_PROCESS_MODULES[first.literal]
            else:
                confidence = "literal_unresolved"
                note = "literal target does not match a loaded module name exactly"
            return FfprcLinkSite(
                source_module=self.script.path.stem,
                program_index=program.index,
                program_name=program.name,
                offset=off,
                file_offset=file_offset,
                order_index=self.order_index,
                argc=argc,
                target_expr=first.expr,
                literal_target=first.literal,
                target_modules=targets,
                confidence=confidence,
                reachable=True,
                note=note,
            )

        if first.data_offset is not None and first.data_offset in self.formatted_buffers:
            fmt = self.formatted_buffers[first.data_offset]
            targets = self._resolve_printf_targets(fmt)
            return FfprcLinkSite(
                source_module=self.script.path.stem,
                program_index=program.index,
                program_name=program.name,
                offset=off,
                file_offset=file_offset,
                order_index=self.order_index,
                argc=argc,
                target_expr=f"sprintf({fmt!r}, ...)",
                format_string=fmt,
                target_modules=targets,
                confidence="dynamic_name",
                reachable=True,
                note="target comes from a buffer written by sprintf/snprintf; order is runtime-parametric",
            )

        return FfprcLinkSite(
            source_module=self.script.path.stem,
            program_index=program.index,
            program_name=program.name,
            offset=off,
            file_offset=file_offset,
            order_index=self.order_index,
            argc=argc,
            target_expr=first.expr,
            target_modules=(),
            confidence="unknown",
            reachable=True,
            note="first argument is not a literal and was not tied to a tracked sprintf buffer",
        )


def extract_ffprc_link_sites(
    script: MbcScript,
    *,
    available_modules: Iterable[str] | None = None,
    reachable_only: bool = True,
) -> list[FfprcLinkSite]:
    """Extract ``ffprc_link`` call-sites needed by the process-plan builder."""

    modules = set(available_modules or ())
    return _FfprcLinkExtractor(script, available_modules=modules, reachable_only=reachable_only).run()


def build_process_plan_from_ffprc_links(
    scripts: Iterable[MbcScript],
    *,
    root_modules: Iterable[str] | None = None,
    include_fallback: bool = True,
    reachable_only: bool = True,
) -> MbcProcessPlan:
    scripts_list = list(scripts)
    by_module = {script.path.stem: script for script in scripts_list}
    available = set(by_module)
    if root_modules is None:
        if "_main" in by_module:
            roots = ("_main",)
        elif scripts_list:
            roots = (scripts_list[0].path.stem,)
        else:
            roots = ()
    else:
        roots = tuple(root_modules)

    site_cache: dict[str, list[FfprcLinkSite]] = {}
    all_sites: list[FfprcLinkSite] = []

    def sites_for_module(module_name: str) -> list[FfprcLinkSite]:
        if module_name in site_cache:
            return site_cache[module_name]
        script = by_module.get(module_name)
        if script is None:
            site_cache[module_name] = []
            return []
        sites = extract_ffprc_link_sites(script, available_modules=available, reachable_only=reachable_only)
        site_cache[module_name] = sites
        all_sites.extend(sites)
        return sites

    ordered: list[str] = []
    selected_sites: list[FfprcLinkSite] = []
    deviations: list[ProcessPlanDeviation] = []
    queued: list[str] = list(roots)
    queued_set: set[str] = set(queued)
    seen: set[str] = set()

    while queued:
        module_name = queued.pop(0)
        queued_set.discard(module_name)
        if module_name in seen:
            continue
        if module_name not in by_module:
            deviations.append(ProcessPlanDeviation("missing_root_or_target", module_name, None, "module is not present in the loaded corpus"))
            continue
        seen.add(module_name)
        ordered.append(module_name)

        for site in sites_for_module(module_name):
            if site.confidence == "literal" and len(site.target_modules) == 1:
                target = site.target_modules[0]
                selected_sites.append(site)
                if target not in seen and target not in queued_set:
                    queued.append(target)
                    queued_set.add(target)
            elif site.confidence == "literal" and len(site.target_modules) > 1:
                selected_sites.append(site)
                deviations.append(ProcessPlanDeviation("ambiguous_literal_target", module_name, site.offset, f"{site.literal_target!r} -> {site.target_modules}"))
            elif site.confidence == "literal_unresolved":
                deviations.append(ProcessPlanDeviation("unresolved_literal_target", module_name, site.offset, f"{site.literal_target!r} did not match a module exactly"))
            elif site.confidence == "missing_process_module":
                deviations.append(ProcessPlanDeviation("missing_process_module", module_name, site.offset, f"{site.literal_target!r}: {site.note}"))
            elif site.confidence == "dynamic_name":
                deviations.append(ProcessPlanDeviation("dynamic_target", module_name, site.offset, f"{site.target_expr} -> {site.target_modules or '<unbounded>'}"))
            elif site.confidence == "unknown":
                deviations.append(ProcessPlanDeviation("unknown_target", module_name, site.offset, site.note))

    fallback = tuple(name for name in by_module if name not in seen) if include_fallback else ()
    if fallback:
        deviations.append(ProcessPlanDeviation("directory_order_fallback", "<project>", None, f"{len(fallback)} modules were not reached from ffprc_link literals"))
        ordered.extend(fallback)

    return MbcProcessPlan(
        root_modules=roots,
        ordered_modules=tuple(ordered),
        fallback_modules=fallback,
        link_sites=tuple(sorted(all_sites, key=lambda site: (site.source_module, site.order_index, site.offset))),
        selected_sites=tuple(selected_sites),
        deviations=tuple(deviations),
    )

def _read_span_text(script: MbcScript, data_offset: int, length: int | None) -> str | None:
    if data_offset < 0 or data_offset >= len(script.data):
        return None
    if length is None or length <= 0:
        end = script.data.find(b"\x00", data_offset)
        if end < 0:
            end = min(len(script.data), data_offset + 128)
    else:
        end = min(len(script.data), data_offset + length)
    raw = script.data[data_offset:end]
    raw = raw.split(b"\x00", 1)[0]
    if not raw:
        return ""
    if raw.count(0xFF) > max(1, len(raw) // 2):
        return None
    try:
        text = raw.decode("cp1251", errors="replace")
    except Exception:
        return None
    if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in text):
        return None
    return text


def _literal_to_module_candidates(literal: str) -> tuple[str, ...]:
    text = literal.rstrip("\x00")
    # Do not strip ordinary spaces. Some bad literals should remain bad so the
    # plan reports a deviation instead of silently fixing data.
    pathish = text.replace("\\", "/")
    name = pathish.rsplit("/", 1)[-1]
    stem = name[:-4] if name.lower().endswith(".mbc") else name
    candidates: list[str] = []
    for candidate in (text, name, stem):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _runtime_quickfile_key(name: str) -> str:
    """Runtime quickfile hash lowercases byte names when lookup flag=1.

    This is only for process/module filenames. Function-table linking remains
    exact and case-sensitive.
    """

    return name.lower()



def _available_modules_by_quickfile_key(available_modules: Iterable[str]) -> dict[str, list[str]]:
    by_key: dict[str, list[str]] = {}
    for module_name in available_modules:
        by_key.setdefault(_runtime_quickfile_key(module_name), []).append(module_name)
    return by_key

def _resolve_literal_target_modules(literal: str, available_modules: set[str], *, available_by_key: dict[str, list[str]] | None = None) -> tuple[str, ...]:
    candidates = _literal_to_module_candidates(literal)
    if not available_modules:
        return candidates[:1]

    exact = tuple(candidate for candidate in candidates if candidate in available_modules)
    if exact:
        return exact

    if available_by_key is None:
        available_by_key = _available_modules_by_quickfile_key(sorted(available_modules))

    resolved: list[str] = []
    for candidate in candidates:
        resolved.extend(available_by_key.get(_runtime_quickfile_key(candidate), ()))
    return tuple(dict.fromkeys(resolved))


def _resolve_printf_target_modules(fmt: str, available_modules: Iterable[str]) -> tuple[str, ...]:
    names = tuple(available_modules)
    if not names:
        return ()
    regex = _printf_format_to_regex(fmt)
    if regex is None:
        return ()
    compiled = re.compile(regex)
    return tuple(name for name in names if compiled.fullmatch(name))


def _printf_format_to_regex(fmt: str) -> str | None:
    parts: list[str] = []
    i = 0
    saw_format = False
    while i < len(fmt):
        ch = fmt[i]
        if ch != "%":
            parts.append(re.escape(ch))
            i += 1
            continue
        if i + 1 < len(fmt) and fmt[i + 1] == "%":
            parts.append("%")
            i += 2
            continue
        match = re.match(r"%([0 #+\-]*)(\d+)?(?:\.\d+)?[hlL]*([diuoxXs])", fmt[i:])
        if not match:
            return None
        width = match.group(2)
        typ = match.group(3)
        saw_format = True
        if typ in "diuoxX":
            if width and match.group(1) and "0" in match.group(1):
                parts.append(rf"\d{{{int(width)}}}")
            else:
                parts.append(r"\d+")
        else:
            parts.append(r".+")
        i += len(match.group(0))
    return "".join(parts) if saw_format else None


class MbcProjectLinker:
    """Runtime-style linker for a sequence of MBC modules.

    The order of ``scripts`` is significant.  It models the order in which the
    VM process receives scripts: first script is the base process, every next
    script is equivalent to one successful ``ffprc_link`` / ``sub_4784C0``.

    The real loop is intentionally narrow:
    * names are matched with C ``strcmp`` semantics (exact bytes after MBC
      decoding), not casefold/stricmp;
    * imports in the newly linked module are resolved only against providers
      already present in the process;
    * still-unresolved old imports are then resolved against providers from the
      newly linked module;
    * providers that appear later do not override an already patched import;
    * patch rel32 is computed from process-code offsets:
      ``target_runtime_offset - source_runtime_offset - 1``.
    """

    def __init__(
        self,
        scripts: Iterable[MbcScript],
        *,
        module_order: Sequence[str] | None = None,
        process_plan: MbcProcessPlan | None = None,
        include_unplanned: bool = True,
    ):
        original_scripts: list[MbcScript] = list(scripts)
        self.process_plan = process_plan
        if process_plan is not None:
            module_order = process_plan.ordered_modules

        if module_order is not None:
            by_module = {script.path.stem: script for script in original_scripts}
            ordered_names: list[str] = []
            for name in module_order:
                if name in by_module and name not in ordered_names:
                    ordered_names.append(name)
            if include_unplanned:
                ordered_names.extend(script.path.stem for script in original_scripts if script.path.stem not in ordered_names)
            self.scripts = [by_module[name] for name in ordered_names if name in by_module]
        else:
            self.scripts = original_scripts

        self._module_order: dict[str, int] = {script.path.stem: i for i, script in enumerate(self.scripts)}
        self.modules: dict[str, MbcStaticLinker] = {}
        for script in self.scripts:
            linker = MbcStaticLinker(script, project=None)
            self.modules[script.path.stem] = linker

        self._states: dict[str, RuntimeModuleState] = {}
        self._states_by_symbol: dict[tuple[str, int], RuntimeModuleState] = {}
        self._runtime_order: list[RuntimeModuleState] = []
        self._links: list[RuntimeResolvedLink] = []
        self._resolved_cache: dict[tuple[str, int], RuntimeResolvedLink | None] = {}
        self._native_cache: dict[tuple[str, int], RuntimeNativeLink | None] = {}
        self._unresolved_imports: dict[tuple[str, int], MbcFunctionSymbol] = {}
        self._providers_by_name: dict[str, list[MbcFunctionSymbol]] = {}

        for linker in self.modules.values():
            linker.project = self

        self._build_runtime_context()

    def _build_runtime_context(self) -> None:
        code_base = 0
        program_base = 0
        function_base = 0
        loaded_first_provider_by_name: dict[str, MbcFunctionSymbol] = {}

        for load_index, script in enumerate(self.scripts):
            linker = self.modules[script.path.stem]
            state = RuntimeModuleState(
                module_name=script.path.stem,
                linker=linker,
                load_index=load_index,
                code_base=code_base,
                program_base=program_base,
                function_base=function_base,
            )
            self._runtime_order.append(state)
            self._states[state.module_name] = state
            for symbol in linker.symbols:
                self._states_by_symbol[(symbol.module_name, symbol.index)] = state

            # First half of sub_4784C0: new imports resolve against old internal
            # records only.  Keep the first visible provider by exact name so
            # each lookup is O(1) instead of scanning all loaded functions.
            for symbol in linker.imports:
                target = loaded_first_provider_by_name.get(symbol.name)
                if target is None:
                    self._unresolved_imports[(symbol.module_name, symbol.index)] = symbol
                    continue
                self._record_link(symbol, target)

            new_providers = tuple(linker.internals)
            new_first_provider_by_name: dict[str, MbcFunctionSymbol] = {}
            for provider in new_providers:
                new_first_provider_by_name.setdefault(provider.name, provider)

            # Second half: old unresolved imports resolve against providers from
            # the module that has just been linked.  Already patched imports are
            # not reconsidered when later duplicate providers appear.
            if new_first_provider_by_name and self._unresolved_imports:
                for key, source in list(self._unresolved_imports.items()):
                    if source.module_name == state.module_name:
                        continue
                    target = new_first_provider_by_name.get(source.name)
                    if target is None:
                        continue
                    self._record_link(source, target)
                    self._unresolved_imports.pop(key, None)

            for provider in new_providers:
                loaded_first_provider_by_name.setdefault(provider.name, provider)
                self._providers_by_name.setdefault(provider.name, []).append(provider)

            code_base += script.header.code_size
            program_base += len(script.programs)
            function_base += len(linker.symbols)

    @staticmethod
    def _find_first_provider(name: str, providers: Iterable[MbcFunctionSymbol]) -> MbcFunctionSymbol | None:
        for provider in providers:
            if provider.is_internal and provider.name == name:
                return provider
        return None

    def _record_link(self, source: MbcFunctionSymbol, target: MbcFunctionSymbol) -> RuntimeResolvedLink:
        source_state = self.state_for_symbol(source)
        target_state = self.state_for_symbol(target)
        source_runtime_offset = source_state.runtime_offset(source) if source_state is not None else source.code_offset
        target_runtime_offset = target_state.runtime_offset(target) if target_state is not None else target.code_offset

        same_name_providers = self._providers_by_name.get(source.name, ())
        alternatives = tuple(sym.qualified_name for sym in same_name_providers if sym is not target)[:8]
        ambiguity = max(1, len(same_name_providers))

        patch = LinkPatch(
            source_module=source.module_name,
            source_name=source.name,
            source_function_index=source.index,
            source_offset=source.code_offset,
            target_module=target.module_name,
            target_name=target.name,
            target_function_index=target.index,
            target_program_index=target.program_index,
            target_program_name=target.program_name,
            target_offset=target.code_offset,
            source_runtime_offset=source_runtime_offset,
            target_runtime_offset=target_runtime_offset,
            source_module_base=source_state.code_base if source_state is not None else 0,
            target_module_base=target_state.code_base if target_state is not None else 0,
        )
        resolved = RuntimeResolvedLink(source=source, target=target, patch=patch, ambiguity=ambiguity, alternatives=alternatives)
        self._resolved_cache[(source.module_name, source.index)] = resolved
        self._links.append(resolved)
        return resolved

    @property
    def runtime_order(self) -> tuple[RuntimeModuleState, ...]:
        return tuple(self._runtime_order)

    def state_for_module(self, module_name: str) -> RuntimeModuleState | None:
        return self._states.get(module_name)

    def state_for_symbol(self, symbol: MbcFunctionSymbol) -> RuntimeModuleState | None:
        return self._states_by_symbol.get((symbol.module_name, symbol.index))

    def module(self, module_name: str) -> Optional["MbcStaticLinker"]:
        return self.modules.get(module_name)

    def providers_named(self, name: str) -> list[MbcFunctionSymbol]:
        return list(self._providers_by_name.get(name, ()))

    def runtime_offset_for(self, symbol: MbcFunctionSymbol) -> int:
        state = self.state_for_symbol(symbol)
        return symbol.code_offset if state is None else state.runtime_offset(symbol)

    def runtime_program_index_for(self, symbol: MbcFunctionSymbol) -> int:
        state = self.state_for_symbol(symbol)
        return symbol.program_index if state is None else state.runtime_program_index(symbol)

    def resolve_import(self, source: MbcFunctionSymbol) -> RuntimeResolvedLink | None:
        key = (source.module_name, source.index)
        if key in self._resolved_cache:
            return self._resolved_cache[key]
        self._resolved_cache[key] = None
        return None

    def resolve_native_import(self, source: MbcFunctionSymbol) -> RuntimeNativeLink | None:
        key = (source.module_name, source.index)
        if key in self._native_cache:
            return self._native_cache[key]
        if not source.is_import:
            self._native_cache[key] = None
            return None
        spec = engine_native_import(source.name)
        if spec is None:
            self._native_cache[key] = None
            return None
        native = RuntimeNativeLink(source=source, spec=spec)
        self._native_cache[key] = native
        return native

    def native_imports_for(self, module: "MbcStaticLinker") -> list[RuntimeNativeLink]:
        links: list[RuntimeNativeLink] = []
        for symbol in module.imports:
            if self.resolve_import(symbol) is None:
                native = self.resolve_native_import(symbol)
                if native is not None:
                    links.append(native)
        return links

    def resolved_imports_for(self, module: "MbcStaticLinker") -> list[RuntimeResolvedLink]:
        links: list[RuntimeResolvedLink] = []
        for symbol in module.imports:
            link = self.resolve_import(symbol)
            if link is not None:
                links.append(link)
        return links

    def unresolved_imports_for(self, module: "MbcStaticLinker") -> list[MbcFunctionSymbol]:
        return [
            symbol
            for symbol in module.imports
            if self.resolve_import(symbol) is None and self.resolve_native_import(symbol) is None
        ]

    def summary_lines(self) -> list[str]:
        total_imports = sum(len(module.imports) for module in self.modules.values())
        resolved = sum(len(self.resolved_imports_for(module)) for module in self.modules.values())
        native = sum(len(self.native_imports_for(module)) for module in self.modules.values())
        unresolved = total_imports - resolved - native
        lines = [
            f"// project modules: {len(self.modules)}",
            "// runtime linker: sequential sub_4784C0 model; exact strcmp names; process-relative rel32 patches",
            f"// runtime links: {resolved} resolved, {native} engine-native, {unresolved} unresolved",
        ]
        if self.process_plan is not None:
            lines.extend(self.process_plan.summary_lines())
        return lines

    @classmethod
    def from_ffprc_plan(
        cls,
        scripts: Iterable[MbcScript],
        *,
        root_modules: Iterable[str] | None = None,
        include_fallback: bool = True,
        reachable_only: bool = True,
    ) -> "MbcProjectLinker":
        scripts_list = list(scripts)
        plan = build_process_plan_from_ffprc_links(
            scripts_list,
            root_modules=root_modules,
            include_fallback=include_fallback,
            reachable_only=reachable_only,
        )
        return cls(scripts_list, process_plan=plan, include_unplanned=include_fallback)


class MbcStaticLinker:
    """Resolve one script's function-table entries, optionally inside a project."""

    def __init__(self, script: MbcScript, *, project: MbcProjectLinker | None = None):
        self.script = script
        self.project = project
        self.module_name = script.path.stem
        self.symbols: list[MbcFunctionSymbol] = [
            MbcFunctionSymbol.from_function(script, fn) for fn in script.functions
        ]
        self._by_offset: dict[int, list[MbcFunctionSymbol]] = {}
        self._by_name: dict[str, list[MbcFunctionSymbol]] = {}
        self._import_by_offset: dict[int, MbcFunctionSymbol] = {}
        self._internal_by_offset: dict[int, MbcFunctionSymbol] = {}
        self._program_entries: dict[int, MbcProgram] = {p.start: p for p in script.programs}
        imports: list[MbcFunctionSymbol] = []
        internals: list[MbcFunctionSymbol] = []

        for symbol in self.symbols:
            self._by_offset.setdefault(symbol.code_offset, []).append(symbol)
            self._by_name.setdefault(symbol.name, []).append(symbol)
            if symbol.is_import:
                imports.append(symbol)
                self._import_by_offset.setdefault(symbol.code_offset, symbol)
            else:
                internals.append(symbol)
                self._internal_by_offset.setdefault(symbol.code_offset, symbol)
        self._imports = tuple(imports)
        self._internals = tuple(internals)

    @property
    def imports(self) -> tuple[MbcFunctionSymbol, ...]:
        return self._imports

    @property
    def internals(self) -> tuple[MbcFunctionSymbol, ...]:
        return self._internals

    def symbols_at(self, code_offset: int) -> list[MbcFunctionSymbol]:
        return list(self._by_offset.get(code_offset, ()))

    def symbol_at(self, code_offset: int, *, prefer_internal: bool = False) -> Optional[MbcFunctionSymbol]:
        symbols = self._by_offset.get(code_offset)
        if not symbols:
            return None
        if prefer_internal:
            for symbol in symbols:
                if symbol.is_internal:
                    return symbol
        return symbols[0]

    def import_stub_at(self, code_offset: int) -> Optional[MbcFunctionSymbol]:
        return self._import_by_offset.get(code_offset)

    def internal_at(self, code_offset: int) -> Optional[MbcFunctionSymbol]:
        return self._internal_by_offset.get(code_offset)

    def symbols_named(self, name: str) -> list[MbcFunctionSymbol]:
        return list(self._by_name.get(name, ()))

    def internal_named(self, name: str) -> Optional[MbcFunctionSymbol]:
        for symbol in self.symbols_named(name):
            if symbol.is_internal:
                return symbol
        return None

    def program_entry_at(self, code_offset: int) -> Optional[MbcProgram]:
        return self._program_entries.get(code_offset)

    def runtime_link_for_import(self, symbol: MbcFunctionSymbol) -> RuntimeResolvedLink | None:
        if self.project is None or not symbol.is_import:
            return None
        return self.project.resolve_import(symbol)

    def native_link_for_import(self, symbol: MbcFunctionSymbol) -> RuntimeNativeLink | None:
        if self.project is None or not symbol.is_import:
            return None
        return self.project.resolve_native_import(symbol)

    def runtime_link_at_import_offset(self, code_offset: int) -> RuntimeResolvedLink | None:
        symbol = self.import_stub_at(code_offset)
        return None if symbol is None else self.runtime_link_for_import(symbol)

    def callable_name_for_offset(self, code_offset: int) -> Optional[str]:
        symbol = self.internal_at(code_offset) or self.symbol_at(code_offset)
        if symbol is not None:
            if symbol.is_import:
                link = self.runtime_link_for_import(symbol)
                if link is not None:
                    return link.target.qualified_name
                native = self.native_link_for_import(symbol)
                return native.spec.name if native is not None else symbol.name
            return symbol.name
        program = self.program_entry_at(code_offset)
        if program is not None:
            return program.name
        return None

    def signature_for_offset(self, code_offset: int) -> FunctionSignature:
        symbol = self.internal_at(code_offset) or self.symbol_at(code_offset)
        if symbol is None:
            return FunctionSignature.unknown()
        if symbol.is_import:
            link = self.runtime_link_for_import(symbol)
            if link is not None:
                return link.target.signature
            native = self.native_link_for_import(symbol)
            return _signature_from_native_spec(native.spec) if native is not None else FunctionSignature.unknown()
        return symbol.signature

    def unresolved_import_names(self) -> list[str]:
        if self.project is None:
            return [sym.name for sym in self.imports]
        return [sym.name for sym in self.project.unresolved_imports_for(self)]


    def patch_bytes(self, patch: LinkPatch) -> bytes:
        return bytes([LINK_PATCH_OPCODE]) + int(patch.rel32).to_bytes(4, "little", signed=True)

    def summary_lines(self, *, max_imports: int = 128, max_links: int = 128) -> list[str]:
        lines = [
            f"// function symbols: {len(self.symbols)} total, {len(self.internals)} internal, {len(self.imports)} external imports",
        ]

        if self.project is not None:
            links = self.project.resolved_imports_for(self)
            native_links = self.project.native_imports_for(self)
            unresolved = self.project.unresolved_imports_for(self)
            lines.append(
                f"// module runtime links: {len(links)} resolved, "
                f"{len(native_links)} engine-native, {len(unresolved)} unresolved"
            )
            if links:
                lines.append("// === virtual runtime links ===")
                for link in links[:max_links]:
                    sig = link.target.signature.render()
                    amb = f", ambiguous={link.ambiguity}" if link.is_ambiguous else ""
                    src_rt = link.patch.source_runtime_offset
                    dst_rt = link.patch.target_runtime_offset
                    rt = ""
                    if src_rt is not None and dst_rt is not None:
                        rt = f", runtime=0x{src_rt:08X}->0x{dst_rt:08X}, rel32={link.patch.rel32}"
                    lines.append(
                        f"// {link.source.name} @ 0x{link.source.code_offset:08X} -> "
                        f"{link.target.qualified_name}{sig} "
                        f"[program={link.target.program_index}, offset=0x{link.target.code_offset:08X}{rt}{amb}]"
                    )
                if len(links) > max_links:
                    lines.append(f"// ... {len(links) - max_links} more resolved links")
            if native_links:
                lines.append("// === engine-native imports ===")
                for native in native_links[:max_imports]:
                    symbol = native.source
                    payload = "" if symbol.import_stub_payload is None else f", payload={symbol.import_stub_payload}"
                    lines.append(
                        f"// native {symbol.name} @ 0x{symbol.code_offset:08X}{payload} -> "
                        f"{native.spec.name}{native.spec.render_signature()} [{native.spec.layer}; {native.spec.note}]"
                    )
                if len(native_links) > max_imports:
                    lines.append(f"// ... {len(native_links) - max_imports} more engine-native imports")
            if unresolved:
                lines.append("// === unresolved imports ===")
                for symbol in unresolved[:max_imports]:
                    payload = "" if symbol.import_stub_payload is None else f", payload={symbol.import_stub_payload}"
                    lines.append(f"// extern unresolved {symbol.name} @ 0x{symbol.code_offset:08X}{payload}")
                if len(unresolved) > max_imports:
                    lines.append(f"// ... {len(unresolved) - max_imports} more unresolved imports")
            return lines

        if self.imports:
            lines.append("// === imports ===")
            for symbol in self.imports[:max_imports]:
                payload = "" if symbol.import_stub_payload is None else f", payload={symbol.import_stub_payload}"
                native = engine_native_import(symbol.name)
                if native is not None:
                    lines.append(f"// native {symbol.name} @ 0x{symbol.code_offset:08X}{payload} -> {native.name}{native.render_signature()}")
                else:
                    lines.append(f"// extern {symbol.name} @ 0x{symbol.code_offset:08X}{payload}")
            if len(self.imports) > max_imports:
                lines.append(f"// ... {len(self.imports) - max_imports} more imports")
        return lines
