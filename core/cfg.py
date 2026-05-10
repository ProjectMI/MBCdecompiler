from __future__ import annotations

"""Program-level control-flow traversal for MBC bytecode.

The module keeps CFG responsibility limited to reachability: it follows direct
control-flow edges from a program entry and returns the decoded instruction
stream used by later analysis layers. It intentionally has no CFG dump builder
and no DOT renderer.
"""

from typing import List, Optional

from .decoder import Instruction, MbcDecoder
from .loader import MbcProgram, MbcScript


CONTROL_EDGE_KINDS = {
    "jmp",
    "jfalse",
    "jtrue",
    "jfalse_fallthrough",
    "jtrue_fallthrough",
    "call_rel32",
    "call_return",
}


class MbcControlFlow:
    def __init__(self, script: MbcScript, *, decoder: Optional[MbcDecoder] = None):
        self.script = script
        self.code = script.code
        self.decoder = decoder or MbcDecoder(script)

    def decode_program(self, program: MbcProgram) -> List[Instruction]:
        """Decode the instruction stream reachable from a program entry.

        Recovered MBC files can place local helper routines after a nominal
        program-table body. Those helpers are reached by ``call_rel32`` and may
        live outside ``program.end``. The traversal follows direct edges and
        ordinary fallthrough, but it does not inline another named program.
        """
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
            owner = self.script.program_for_offset(target)
            if owner is not None and owner.index != program.index:
                # Keep the edge in the instruction metadata, but do not merge a
                # different named program into this program's stream.
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
                if edge.kind in CONTROL_EDGE_KINDS:
                    enqueue(edge.dst)

            if not ins.terminal:
                enqueue(ins.offset + ins.length)

        return [decoded[o] for o in sorted(decoded)]
