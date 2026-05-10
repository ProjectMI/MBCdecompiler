from __future__ import annotations

"""Instruction-level MBC bytecode decoder.

This module owns only bytecode decoding and direct edge extraction. It does not
walk program reachability, assemble basic blocks, render DOT, or produce dump
artifacts.
"""

from dataclasses import dataclass
from typing import List, Optional

from .loader import MbcProgram, MbcScript
from .linker import MbcStaticLinker
from .opcodes import CODE_FILE_OFFSET, decode_opcode


@dataclass
class Edge:
    kind: str
    src: int
    dst: Optional[int]
    dst_program: Optional[str] = None
    note: str = ""


@dataclass
class Instruction:
    offset: int
    file_offset: int
    opcode: int
    mnemonic: str
    length: int
    raw: str
    operands: dict
    edges: List[Edge]
    terminal: bool = False
    known: bool = True


class MbcDecoder:
    """Decode a single MBC instruction at a code-section offset.

    PC convention: all offsets are code-section offsets. File offset is
    ``code_offset + 0x20``. The original VM increments PC before invoking a
    handler, so relative branches are based at ``off + 1``.
    """

    def __init__(self, script: MbcScript, *, linker: MbcStaticLinker | None = None):
        self.script = script
        self.code = script.code
        self.linker = linker or MbcStaticLinker(script)

    def decode_at(self, off: int, program: Optional[MbcProgram] = None) -> Instruction:
        if not (0 <= off < len(self.code)):
            raise ValueError(f"code offset 0x{off:X} is outside code section")

        decoded = decode_opcode(self.code, off)
        length = max(decoded.length, 1)
        raw = self.code[off:min(len(self.code), off + length)].hex(" ")
        edges = [
            Edge(edge.kind, off, edge.dst, self._program_name_for(edge.dst) if edge.dst is not None else None, edge.note)
            for edge in decoded.edges
        ]

        # Add program references as metadata edges where the operand is a valid
        # program table index. They are useful navigation hints, not fallthrough.
        operands = dict(decoded.operands)
        program_index = operands.get("program_index") if operands else None
        if isinstance(program_index, int) and 0 <= program_index < len(self.script.programs):
            target_program = self.script.programs[program_index]
            operands["program_name"] = target_program.name
            operands["program_start"] = target_program.start
            operands["program_start_file"] = target_program.file_start
            edges.append(Edge("program_ref", off, target_program.start, target_program.name, decoded.mnemonic))

        self._annotate_linkage(off, operands, edges)

        return Instruction(
            offset=off,
            file_offset=off + CODE_FILE_OFFSET,
            opcode=self.code[off],
            mnemonic=decoded.mnemonic,
            length=length,
            raw=raw,
            operands=operands,
            edges=edges,
            terminal=decoded.terminal,
            known=decoded.known,
        )


    def _annotate_linkage(self, off: int, operands: dict, edges: List[Edge]) -> None:
        symbols_here = self.linker.symbols_at(off)
        if symbols_here:
            operands["function_symbols"] = [symbol.to_dict() for symbol in symbols_here]
            preferred = next((symbol for symbol in symbols_here if symbol.is_internal), symbols_here[0])
            operands["function_name"] = preferred.name
            operands["function_qualified_name"] = preferred.qualified_name
            operands["function_kind"] = preferred.kind
            operands["function_index"] = preferred.index
            operands["function_program_index"] = preferred.program_index
            operands["function_signature"] = preferred.signature.to_dict()

        import_symbol = self.linker.import_stub_at(off)
        if import_symbol is not None:
            operands["link_name"] = import_symbol.name
            operands["link_kind"] = import_symbol.kind
            operands["link_function_index"] = import_symbol.index
            operands["link_program_index"] = import_symbol.program_index
            link = self.linker.runtime_link_for_import(import_symbol)
            if link is not None:
                operands["resolved_import"] = link.to_dict()
                operands["link_target_name"] = link.target.qualified_name
                operands["link_target_module"] = link.target.module_name
                operands["link_target_program_index"] = link.target.program_index
                operands["link_target_program_name"] = link.target.program_name
                operands["link_target_offset"] = link.target.code_offset
                operands["link_target_signature"] = link.target.signature.to_dict()
                edges.append(Edge("runtime_link_symbol", off, None, link.target.program_name, link.target.qualified_name))
            else:
                native = self.linker.native_link_for_import(import_symbol)
                if native is not None:
                    operands["resolved_native_import"] = native.to_dict()
                    operands["link_target_name"] = native.spec.name
                    operands["link_target_module"] = native.spec.layer
                    operands["link_target_signature"] = native.to_dict().get("signature")
                    edges.append(Edge("engine_native_symbol", off, None, None, native.spec.name))
                else:
                    edges.append(Edge("import_symbol", off, None, None, import_symbol.name))

        target = operands.get("target")
        if isinstance(target, int):
            target_name = self.linker.callable_name_for_offset(target)
            if target_name is not None:
                operands["target_name"] = target_name
                symbol = self.linker.internal_at(target) or self.linker.symbol_at(target)
                if symbol is not None:
                    operands["target_function_index"] = symbol.index
                    operands["target_function_kind"] = symbol.kind
                    operands["target_signature"] = self.linker.signature_for_offset(target).to_dict()
                    if symbol.is_import:
                        link = self.linker.runtime_link_for_import(symbol)
                        if link is not None:
                            operands["resolved_call"] = link.to_dict()
                            operands["target_function_kind"] = "runtime_link"
                            operands["target_module"] = link.target.module_name
                            operands["target_script"] = link.target.script_path
                            operands["target_program_index"] = link.target.program_index
                            operands["target_program_name"] = link.target.program_name
                            operands["target_offset"] = link.target.code_offset
                            operands["target_signature"] = link.target.signature.to_dict()
                            edges.append(Edge("runtime_link_call", off, None, link.target.program_name, link.target.qualified_name))
                        else:
                            native = self.linker.native_link_for_import(symbol)
                            if native is not None:
                                operands["resolved_native_call"] = native.to_dict()
                                operands["target_function_kind"] = "engine_native"
                                operands["target_module"] = native.spec.layer
                                operands["target_script"] = "<interpreter-native>"
                                operands["target_signature"] = native.to_dict().get("signature")
                                edges.append(Edge("engine_native_call", off, None, None, native.spec.name))

    def _program_name_for(self, offset: Optional[int]) -> Optional[str]:
        if offset is None:
            return None
        p = self.script.program_for_offset(offset)
        return None if p is None else p.name
