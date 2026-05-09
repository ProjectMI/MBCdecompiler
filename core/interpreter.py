from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .ast_builder import build_program_ast
from .loader import MbcProgram, MbcScript
from .opcodes import CODE_FILE_OFFSET, OPCODES, BUILTINS, decode_opcode, safe_chr, opcode_to_dict, builtin_to_dict


@dataclass
class Edge:
    kind: str
    src: int
    dst: Optional[int]
    dst_program: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "src": self.src,
            "src_file": self.src + CODE_FILE_OFFSET,
            "dst": self.dst,
            "dst_file": None if self.dst is None else self.dst + CODE_FILE_OFFSET,
            "dst_program": self.dst_program,
            "note": self.note,
        }


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

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "file_offset": self.file_offset,
            "opcode": self.opcode,
            "opcode_hex": f"0x{self.opcode:02X}",
            "opcode_chr": safe_chr(self.opcode),
            "mnemonic": self.mnemonic,
            "length": self.length,
            "raw": self.raw,
            "operands": self.operands,
            "terminal": self.terminal,
            "known": self.known,
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class BasicBlock:
    id: str
    start: int
    end: int
    instructions: List[Instruction]
    edges: List[Edge]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "start_file": self.start + CODE_FILE_OFFSET,
            "end": self.end,
            "end_file": self.end + CODE_FILE_OFFSET,
            "instructions": [i.to_dict() for i in self.instructions],
            "edges": [e.to_dict() for e in self.edges],
        }


class MbcInterpreter:
    """Static MBC bytecode decoder, CFG builder, and early pseudo-AST builder.

    This is intentionally not a full runtime VM.  It follows the recovered client
    dispatcher enough to recover instruction lengths, direct control-flow edges,
    program-management references, builtin subopcodes, and a symbolic stack AST.

    PC convention: all offsets are code-section offsets.  File offset is
    ``code_offset + 0x20``.  The original VM increments PC before invoking a
    handler, so relative branches are based at ``off + 1``.
    """

    def __init__(self, script: MbcScript, *, include_ast: bool = True):
        self.script = script
        self.code = script.code
        self.include_ast = include_ast

    def build_cfg(self) -> dict:
        programs = []
        for program in self.script.programs:
            if not (0 <= program.start < len(self.code)):
                programs.append(self._empty_program_cfg(program, "program start is outside code section"))
                continue
            if program.end < program.start:
                programs.append(self._empty_program_cfg(program, "program end is before start"))
                continue
            programs.append(self._build_program_cfg(program))

        return {
            "script": self.script.to_summary_dict(),
            "format_notes": {
                "offset_basis": "All instruction offsets are code-section offsets; file offset = offset + 0x20.",
                "pc_rule": "Relative targets are computed from pc_after_opcode (off + 1), matching sub_489410.",
                "opcode_model": "core/opcodes.py contains recovered top and builtin dispatch tables plus operand decoders.",
                "ast_precision": "The AST is an experimental symbolic stack AST seed; it is not yet a structured source-level decompilation.",
            },
            "opcode_tables": {
                "top_count": len(OPCODES),
                "builtin_count": len(BUILTINS),
            },
            "programs": programs,
        }

    def opcode_tables(self) -> dict:
        return {
            "top_opcodes": opcode_to_dict(),
            "builtins": builtin_to_dict(),
        }

    def _empty_program_cfg(self, program: MbcProgram, reason: str) -> dict:
        out = {
            "index": program.index,
            "name": program.name,
            "start": program.start,
            "end": program.end,
            "state": program.state,
            "queue_id": program.queue_id,
            "warning": reason,
            "blocks": [],
            "instructions": [],
        }
        if self.include_ast:
            out["ast"] = {
                "format": "experimental_stack_ast_v0",
                "warning": reason,
                "statement_count": 0,
                "statements": [],
                "source": "",
            }
        return out

    def _build_program_cfg(self, program: MbcProgram) -> dict:
        instructions = self.decode_program(program)

        leaders = {program.start}
        for ins in instructions:
            next_off = ins.offset + ins.length
            for edge in ins.edges:
                if edge.dst is not None and program.start <= edge.dst <= program.end:
                    leaders.add(edge.dst)
                if edge.kind in {
                    "jfalse", "jtrue", "jfalse_fallthrough", "jtrue_fallthrough",
                    "call_rel32", "call_return",
                } and program.start <= next_off <= program.end:
                    leaders.add(next_off)
            if ins.terminal and program.start <= next_off <= program.end:
                leaders.add(next_off)

        leader_set = set(sorted(leaders))
        blocks: List[BasicBlock] = []
        idx = 0

        while idx < len(instructions):
            start = instructions[idx].offset
            if start not in leader_set:
                # Resync safeguard for malformed bytecode or a still-imperfect length.
                leader_set.add(start)

            block_ins: List[Instruction] = []
            while idx < len(instructions):
                ins = instructions[idx]
                if block_ins and ins.offset in leader_set:
                    break
                block_ins.append(ins)
                idx += 1
                if ins.terminal:
                    break
                nxt = ins.offset + ins.length
                if nxt in leader_set:
                    break

            if not block_ins:
                idx += 1
                continue

            last = block_ins[-1]
            edges = list(last.edges)
            if not last.terminal:
                next_off = last.offset + last.length
                if program.start <= next_off <= program.end:
                    edges.append(Edge("fallthrough", last.offset, next_off, self._program_name_for(next_off)))

            blocks.append(
                BasicBlock(
                    id=f"{program.name or program.index}:0x{block_ins[0].offset:08X}",
                    start=block_ins[0].offset,
                    end=block_ins[-1].offset + block_ins[-1].length - 1,
                    instructions=block_ins,
                    edges=edges,
                )
            )

        out = {
            "index": program.index,
            "name": program.name,
            "start": program.start,
            "start_file": program.file_start,
            "end": program.end,
            "end_file": program.file_end,
            "state": program.state,
            "queue_id": program.queue_id,
            "blocks": [b.to_dict() for b in blocks],
            "instructions": [i.to_dict() for i in instructions],
        }
        if self.include_ast:
            out["ast"] = build_program_ast(self.script, program, instructions)
        return out

    def decode_program(self, program: MbcProgram) -> List[Instruction]:
        out: List[Instruction] = []
        off = program.start
        hard_end = min(program.end, len(self.code) - 1)

        while off <= hard_end:
            ins = self.decode_at(off)
            if ins.length <= 0:
                ins.length = 1
                ins.known = False
            if ins.offset + ins.length - 1 > hard_end:
                # Do not let a malformed/truncated operand swallow the next program.
                ins.length = 1
                ins.raw = self.code[off:off + 1].hex(" ")
                ins.mnemonic = f"{ins.mnemonic}_clamped"
                ins.known = False
                ins.edges = []
                ins.terminal = False
            out.append(ins)
            off += ins.length
        return out

    def decode_at(self, off: int, program: Optional[MbcProgram] = None) -> Instruction:
        decoded = decode_opcode(self.code, off)
        length = max(decoded.length, 1)
        raw = self.code[off:min(len(self.code), off + length)].hex(" ")
        edges = [
            Edge(edge.kind, off, edge.dst, self._program_name_for(edge.dst) if edge.dst is not None else None, edge.note)
            for edge in decoded.edges
        ]

        # Add non-CFG program references as explicit metadata edges where the
        # operand is a valid program table index.  They are not fallthrough
        # control-flow, but they are useful when navigating scripts.
        program_index = decoded.operands.get("program_index") if decoded.operands else None
        if isinstance(program_index, int) and 0 <= program_index < len(self.script.programs):
            target_program = self.script.programs[program_index]
            decoded.operands["program_name"] = target_program.name
            decoded.operands["program_start"] = target_program.start
            decoded.operands["program_start_file"] = target_program.file_start
            edges.append(Edge("program_ref", off, target_program.start, target_program.name, decoded.mnemonic))

        return Instruction(
            offset=off,
            file_offset=off + CODE_FILE_OFFSET,
            opcode=self.code[off],
            mnemonic=decoded.mnemonic,
            length=length,
            raw=raw,
            operands=dict(decoded.operands),
            edges=edges,
            terminal=decoded.terminal,
            known=decoded.known,
        )

    def _program_name_for(self, offset: int) -> Optional[str]:
        p = self.script.program_for_offset(offset)
        return None if p is None else p.name


def cfg_to_dot(cfg: dict) -> str:
    """Render a compact Graphviz DOT view of the CFG."""
    lines = ["digraph mbc_cfg {", "  graph [rankdir=LR];", "  node [shape=box, fontname=\"Consolas\"];"]
    for program in cfg.get("programs", []):
        pname = program.get("name") or f"program_{program.get('index')}"
        safe_pname = str(pname).replace('"', r'\"')
        lines.append(f'  subgraph "cluster_{program.get("index")}" {{')
        lines.append(f'    label="{safe_pname}";')
        for block in program.get("blocks", []):
            node_id = _dot_id(block["id"])
            label = f'{pname}\\n0x{block["start"]:X}-0x{block["end"]:X}'
            lines.append(f'    {node_id} [label="{label}"];')
        lines.append("  }")

    known_blocks = {
        block["start"]: block["id"]
        for program in cfg.get("programs", [])
        for block in program.get("blocks", [])
    }

    emitted_external = set()
    for program in cfg.get("programs", []):
        for block in program.get("blocks", []):
            src_id = _dot_id(block["id"])
            for edge in block.get("edges", []):
                dst = edge.get("dst")
                if dst is None:
                    continue
                dst_block_id = known_blocks.get(dst)
                if not dst_block_id:
                    dst_block_id = f"external_0x{dst:X}"
                    if dst_block_id not in emitted_external:
                        emitted_external.add(dst_block_id)
                        label = edge.get("dst_program") or f"0x{dst:X}"
                        lines.append(f'  {_dot_id(dst_block_id)} [label="{label}", style=dashed];')
                lines.append(
                    f'  {src_id} -> {_dot_id(dst_block_id)} '
                    f'[label="{edge.get("kind", "")}"];'
                )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_id(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "n_" + "".join(cleaned)
