from __future__ import annotations

"""Static function-link model for MBC scripts.

The engine linker uses the runtime function table as the source of truth.  Each
record has a fixed 0x34-byte layout after loading: name at +0, bytecode/stub
code offset at +0x24, and a program index / unresolved marker at +0x28.
Unresolved imports carry -1 at +0x28 and normally point at a 5-byte 0x67 stub;
when a matching resolved function is found, the engine patches that stub into a
0x47 ('G') relative jump whose target is computed from pc_after_opcode.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from .loader import MbcFunction, MbcProgram, MbcScript


LINK_PATCH_OPCODE = 0x47  # 'G' / jmp_rel32
def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


@dataclass(frozen=True)
class MbcFunctionSymbol:
    index: int
    name: str
    code_offset: int
    program_index: int
    flags_or_module: int
    program_name: Optional[str]
    import_stub_payload: Optional[int]

    @classmethod
    def from_function(cls, script: MbcScript, fn: MbcFunction) -> "MbcFunctionSymbol":
        program_index = fn.program_index
        program_name: Optional[str] = None
        if 0 <= program_index < len(script.programs):
            program_name = script.programs[program_index].name

        payload: Optional[int] = None
        if 0 <= fn.code_offset + 5 <= len(script.code) and script.code[fn.code_offset] == 0x67:
            payload = int.from_bytes(script.code[fn.code_offset + 1:fn.code_offset + 5], "little", signed=False)

        return cls(
            index=fn.index,
            name=fn.name,
            code_offset=fn.code_offset,
            program_index=program_index,
            flags_or_module=fn.flags_or_module,
            program_name=program_name,
            import_stub_payload=payload,
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
    def file_offset(self) -> int:
        return self.code_offset + 0x20

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "code_offset": self.code_offset,
            "file_offset": self.file_offset,
            "program_index": self.program_index,
            "program_name": self.program_name,
            "flags_or_module": self.flags_or_module,
            "import_stub_payload": self.import_stub_payload,
        }


@dataclass(frozen=True)
class LinkPatch:
    source_module: str
    source_name: str
    source_offset: int
    target_module: str
    target_name: str
    target_offset: int
    opcode: int = LINK_PATCH_OPCODE

    @property
    def rel32(self) -> int:
        # Engine rule from sub_4784C0: write target_offset - source_offset - 1
        # after opcode 0x47.  Branch decoding also uses off + 1 as PC base.
        return self.target_offset - self.source_offset - 1


class MbcStaticLinker:
    """Resolve function-table entries and import stubs without mutating bytecode."""

    def __init__(self, script: MbcScript):
        self.script = script
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
        for symbol in self._by_offset.get(code_offset, ()):
            if symbol.is_import:
                return symbol
        return None

    def internal_at(self, code_offset: int) -> Optional[MbcFunctionSymbol]:
        for symbol in self._by_offset.get(code_offset, ()):
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

    def callable_name_for_offset(self, code_offset: int) -> Optional[str]:
        symbol = self.internal_at(code_offset) or self.symbol_at(code_offset)
        if symbol is not None:
            return symbol.name
        program = self.program_entry_at(code_offset)
        if program is not None:
            return program.name
        return None

    def unresolved_import_names(self) -> list[str]:
        return [sym.name for sym in self.imports]

    def plan_patches_against(self, providers: Iterable["MbcStaticLinker"]) -> list[LinkPatch]:
        """Create the same patch plan as the engine linker, but do not write bytes.

        For every unresolved import in this script, find the first provider with
        an internal function of the same name and describe the exact `G rel32`
        patch that would be applied to the import stub.
        """
        patches: list[LinkPatch] = []
        module_name = self.script.path.stem
        for import_symbol in self.imports:
            for provider in providers:
                target = provider.internal_named(import_symbol.name)
                if target is None:
                    continue
                patches.append(
                    LinkPatch(
                        source_module=module_name,
                        source_name=import_symbol.name,
                        source_offset=import_symbol.code_offset,
                        target_module=provider.script.path.stem,
                        target_name=target.name,
                        target_offset=target.code_offset,
                    )
                )
                break
        return patches

    def patch_bytes(self, patch: LinkPatch) -> bytes:
        return bytes([LINK_PATCH_OPCODE]) + int(patch.rel32).to_bytes(4, "little", signed=True)

    def summary_lines(self, *, max_imports: int = 256) -> list[str]:
        lines = [
            f"// function symbols: {len(self.symbols)} total, {len(self.internals)} internal, {len(self.imports)} external imports",
        ]
        if self.imports:
            lines.append("// === imports ===")
            for symbol in self.imports[:max_imports]:
                payload = "" if symbol.import_stub_payload is None else f", payload={symbol.import_stub_payload}"
                lines.append(f"// extern {symbol.name} @ 0x{symbol.code_offset:08X}{payload}")
            if len(self.imports) > max_imports:
                lines.append(f"// ... {len(self.imports) - max_imports} more imports")
        return lines
