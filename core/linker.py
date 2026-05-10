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
from typing import Iterable, Optional

from .loader import MbcFunction, MbcProgram, MbcScript
from .native_api import NativeCallSpec, engine_native_import
from .opcodes import CODE_FILE_OFFSET, TYPE_NAMES, decode_opcode


LINK_PATCH_OPCODE = 0x47  # 'G' / jmp_rel32


def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _type_name(type_id: int | None) -> str:
    if type_id is None:
        return "unknown"
    return TYPE_NAMES.get(type_id, f"type_{type_id}")


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

    @property
    def rel32(self) -> int:
        # Engine rule from sub_4784C0: write target_offset - source_offset - 1
        # after opcode 0x47. Branch decoding also uses off + 1 as PC base.
        return self.target_offset - self.source_offset - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "source_module": self.source_module,
            "source_name": self.source_name,
            "source_function_index": self.source_function_index,
            "source_offset": self.source_offset,
            "target_module": self.target_module,
            "target_name": self.target_name,
            "target_function_index": self.target_function_index,
            "target_program_index": self.target_program_index,
            "target_program_name": self.target_program_name,
            "target_offset": self.target_offset,
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


class MbcProjectLinker:
    """Project-wide runtime linker for a loaded mbc/ directory."""

    def __init__(self, scripts: Iterable[MbcScript]):
        self.scripts: list[MbcScript] = list(scripts)
        self._module_order: dict[str, int] = {script.path.stem: i for i, script in enumerate(self.scripts)}
        self.modules: dict[str, MbcStaticLinker] = {}
        for script in self.scripts:
            linker = MbcStaticLinker(script, project=None)
            self.modules[script.path.stem] = linker

        self._providers_by_name: dict[str, list[MbcFunctionSymbol]] = {}
        for linker in self.modules.values():
            linker.project = self
            for symbol in linker.internals:
                self._providers_by_name.setdefault(symbol.name.casefold(), []).append(symbol)

        for providers in self._providers_by_name.values():
            providers.sort(key=lambda sym: (self._module_order.get(sym.module_name, 10**9), sym.index))

        self._resolved_cache: dict[tuple[str, int], RuntimeResolvedLink | None] = {}
        self._native_cache: dict[tuple[str, int], RuntimeNativeLink | None] = {}

    def module(self, module_name: str) -> Optional["MbcStaticLinker"]:
        return self.modules.get(module_name)

    def providers_named(self, name: str) -> list[MbcFunctionSymbol]:
        return list(self._providers_by_name.get(name.casefold(), ()))

    def resolve_import(self, source: MbcFunctionSymbol) -> RuntimeResolvedLink | None:
        key = (source.module_name, source.index)
        if key in self._resolved_cache:
            return self._resolved_cache[key]

        providers = [sym for sym in self.providers_named(source.name) if sym.is_internal]
        if not providers:
            self._resolved_cache[key] = None
            return None

        # Runtime load order is the only stable global ordering available in the
        # raw bytecode corpus.  When a name is duplicated across many scripts we
        # still select the first provider to model the VM table lookup, but keep
        # the ambiguity metadata in the IR so the result is auditable.
        target = providers[0]
        alternatives = tuple(sym.qualified_name for sym in providers[1:8])
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
        )
        resolved = RuntimeResolvedLink(source=source, target=target, patch=patch, ambiguity=len(providers), alternatives=alternatives)
        self._resolved_cache[key] = resolved
        return resolved

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
        return [
            f"// project modules: {len(self.modules)}",
            f"// runtime links: {resolved} resolved, {native} engine-native, {unresolved} unresolved",
        ]


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
        self._program_entries: dict[int, MbcProgram] = {p.start: p for p in script.programs}

        for symbol in self.symbols:
            self._by_offset.setdefault(symbol.code_offset, []).append(symbol)
            self._by_name.setdefault(symbol.name.casefold(), []).append(symbol)

    @property
    def imports(self) -> list[MbcFunctionSymbol]:
        return [sym for sym in self.symbols if sym.is_import]

    @property
    def internals(self) -> list[MbcFunctionSymbol]:
        return [sym for sym in self.symbols if sym.is_internal]

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
        for symbol in self._by_offset.get(code_offset, ()):  # function entries own the 0x67 stubs
            if symbol.is_import:
                return symbol
        return None

    def internal_at(self, code_offset: int) -> Optional[MbcFunctionSymbol]:
        for symbol in self._by_offset.get(code_offset, ()):  # a real bytecode function/program entry
            if symbol.is_internal:
                return symbol
        return None

    def symbols_named(self, name: str) -> list[MbcFunctionSymbol]:
        return list(self._by_name.get(name.casefold(), ()))

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

    def plan_patches_against(self, providers: Iterable["MbcStaticLinker"]) -> list[LinkPatch]:
        """Create patch plan against explicit providers without writing bytes."""
        patches: list[LinkPatch] = []
        for import_symbol in self.imports:
            for provider in providers:
                target = provider.internal_named(import_symbol.name)
                if target is None:
                    continue
                patches.append(
                    LinkPatch(
                        source_module=self.module_name,
                        source_name=import_symbol.name,
                        source_function_index=import_symbol.index,
                        source_offset=import_symbol.code_offset,
                        target_module=provider.module_name,
                        target_name=target.name,
                        target_function_index=target.index,
                        target_program_index=target.program_index,
                        target_program_name=target.program_name,
                        target_offset=target.code_offset,
                    )
                )
                break
        return patches

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
                    lines.append(
                        f"// {link.source.name} @ 0x{link.source.code_offset:08X} -> "
                        f"{link.target.qualified_name}{sig} "
                        f"[program={link.target.program_index}, offset=0x{link.target.code_offset:08X}{amb}]"
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
