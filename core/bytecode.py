from __future__ import annotations

"""Decoded bytecode stream and program-local reachability traversal.

The opcode table remains in :mod:`opcodes`; this module wraps decoded opcodes
into instruction objects and follows direct control-flow edges for one program.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .common import CODE_FILE_OFFSET
from .loader import MbcProgram, MbcScript
from .opcodes import decode_opcode

if TYPE_CHECKING:
    from .linker import MbcStaticLinker

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

    def __init__(
        self,
        script: MbcScript,
        *,
        linker: MbcStaticLinker | None = None,
        annotate_linkage: bool = True,
        import_stub_offsets: set[int] | None = None,
        cache_decodes: bool = False,
    ):
        self.script = script
        self.code = script.code
        self.annotate_linkage = annotate_linkage
        self._decode_cache_enabled = cache_decodes
        self._decode_cache: dict[int, Instruction] = {}
        self._import_stub_offsets = import_stub_offsets
        if linker is None and annotate_linkage:
            # Import lazily so linker.py can reuse the CFG decoder without a
            # module-level circular import.
            from .linker import MbcStaticLinker

            linker = MbcStaticLinker(script)
        self.linker = linker

    def is_import_stub_offset(self, off: int) -> bool:
        if self.linker is not None:
            return self.linker.import_stub_at(off) is not None
        return off in self._import_stub_offsets if self._import_stub_offsets is not None else False

    def decode_at(self, off: int, program: Optional[MbcProgram] = None) -> Instruction:
        if not (0 <= off < len(self.code)):
            raise ValueError(f"code offset 0x{off:X} is outside code section")
        if self._decode_cache_enabled and off in self._decode_cache:
            return self._decode_cache[off]

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

        if self.annotate_linkage and self.linker is not None:
            self._annotate_linkage(off, operands, edges)

        instruction = Instruction(
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
        if self._decode_cache_enabled:
            self._decode_cache[off] = instruction
        return instruction


    def _annotate_linkage(self, off: int, operands: dict, edges: List[Edge]) -> None:
        if self.linker is None:
            return
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

CONTROL_EDGE_KINDS = {
    "jmp",
    "jfalse",
    "jtrue",
    "jfalse_fallthrough",
    "jtrue_fallthrough",
    "call_rel32",
    "call_return",
    # Coroutine continuation: opcode '|' ends the current VM scheduler slice,
    # but the interpreter saves PC=off+1 and resumes from there later.
    "yield_resume",
}


class MbcControlFlow:
    def __init__(self, script: MbcScript, *, decoder: Optional[MbcDecoder] = None):
        self.script = script
        self.code = script.code
        self.decoder = decoder or MbcDecoder(script)
        self._program_owner_cache: dict[int, Optional[MbcProgram]] = {}

    def _program_for_offset_cached(self, code_offset: int) -> Optional[MbcProgram]:
        if code_offset not in self._program_owner_cache:
            self._program_owner_cache[code_offset] = self.script.program_for_offset(code_offset)
        return self._program_owner_cache[code_offset]

    def decode_program(
        self,
        program: MbcProgram,
        *,
        follow_local_calls: bool = True,
        stop_offsets: set[int] | None = None,
    ) -> List[Instruction]:
        """Decode the instruction stream reachable from a program entry.

        Recovered MBC files can place local helper routines after a nominal
        program-table body. Those helpers are reached by ``call_rel32`` and may
        live outside ``program.end``. When ``follow_local_calls`` is false, call
        targets are preserved as calls but not inlined into the caller stream;
        this is the mode used by the decompiler renderer to split local helpers
        into separate synthetic functions.
        """
        stop_offsets = stop_offsets or set()
        if not (0 <= program.start < len(self.code)):
            return []
        if program.end < program.start:
            return []

        decoded: dict[int, Instruction] = {}
        worklist: list[int] = [program.start]

        def enqueue(target: Optional[int]) -> None:
            if target is None:
                return
            if not (0 <= target < len(self.code)):
                return
            if target != program.start and target in stop_offsets:
                return
            owner = self._program_for_offset_cached(target)
            if owner is not None and owner.index != program.index:
                # Keep the edge in the instruction metadata, but do not merge a
                # different named program into this program's stream.
                return
            if self.decoder.is_import_stub_offset(target):
                # 0x67 function-table stubs are runtime link points, not real
                # bytecode bodies.  Calls keep their resolved-call metadata in
                # the instruction operands, but CFG traversal must not inline
                # the stub as an `extern` pseudo-statement.
                return
            if target in decoded or target in worklist:
                return
            worklist.append(target)

        while worklist:
            off = worklist.pop()
            if off in decoded or not (0 <= off < len(self.code)):
                continue

            ins = self.decoder.decode_at(off, program)
            if ins.length <= 0:
                ins.length = 1
                ins.known = False
            decoded[off] = ins

            for edge in ins.edges:
                if edge.kind == "call_rel32" and not follow_local_calls:
                    continue
                if edge.kind in CONTROL_EDGE_KINDS:
                    enqueue(edge.dst)

            if not ins.terminal:
                enqueue(ins.offset + ins.length)

        return [decoded[o] for o in sorted(decoded)]
