"""Normalize instruction blocks into macro-level operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .instruction_profile import InstructionKind, InstructionProfile
from .stack import StackEvent, StackSummary


@dataclass(frozen=True)
class NormalizedOperation:
    """High level operation recovered from a block of instructions."""

    name: str
    category: str
    start_offset: int
    end_offset: int
    length: int
    notes: Tuple[str, ...] = tuple()

    def describe(self) -> str:
        span = f"[{self.start_offset:08X}-{self.end_offset:08X}]"
        base = f"{self.name} {span} category={self.category} len={self.length}"
        if self.notes:
            return base + " " + ", ".join(self.notes)
        return base

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "length": self.length,
            "notes": list(self.notes),
        }


class MacroNormalizer:
    """Collapse low level instruction streams into macro operations."""

    def normalize(
        self,
        profiles: Sequence[InstructionProfile],
        stack: StackSummary,
    ) -> Tuple[NormalizedOperation, ...]:
        if not profiles:
            return tuple()
        events = stack.events or tuple()
        operations: List[NormalizedOperation] = []
        operations.extend(self._tail_and_return_macros(profiles))
        operations.extend(self._literal_reduction_macros(profiles))
        operations.extend(self._predicate_macros(profiles))
        operations.extend(self._indirect_macros(profiles, events))
        operations.sort(key=lambda op: op.start_offset)
        return tuple(operations)

    # ------------------------------------------------------------------
    # macro detectors
    # ------------------------------------------------------------------
    def _tail_and_return_macros(
        self, profiles: Sequence[InstructionProfile]
    ) -> List[NormalizedOperation]:
        operations: List[NormalizedOperation] = []
        call_indices = [
            idx
            for idx, profile in enumerate(profiles)
            if profile.kind in {InstructionKind.TAILCALL, InstructionKind.CALL}
        ]
        if call_indices:
            first = call_indices[0]
            last = call_indices[-1]
            start = profiles[first].word.offset
            end = profiles[last].word.offset
            names = {profile.mnemonic for profile in profiles[first : last + 1]}
            notes = tuple(sorted(names))
            has_tail = any(
                profile.kind is InstructionKind.TAILCALL
                for profile in profiles[first : last + 1]
            )
            name = "tail_dispatch" if has_tail else "call_dispatch"
            operations.append(
                NormalizedOperation(
                    name=name,
                    category="call",
                    start_offset=start,
                    end_offset=end,
                    length=last - first + 1,
                    notes=notes,
                )
            )

        return_indices = [
            idx
            for idx, profile in enumerate(profiles)
            if profile.kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}
        ]
        if return_indices:
            first = return_indices[0]
            last = return_indices[-1]
            start = profiles[first].word.offset
            end = profiles[last].word.offset
            mnemonics = {profile.mnemonic for profile in profiles[first : last + 1]}
            notes = tuple(sorted(mnemonics))
            operations.append(
                NormalizedOperation(
                    name="frame_end",
                    category="return",
                    start_offset=start,
                    end_offset=end,
                    length=last - first + 1,
                    notes=notes,
                )
            )
        return operations

    def _literal_reduction_macros(
        self, profiles: Sequence[InstructionProfile]
    ) -> List[NormalizedOperation]:
        operations: List[NormalizedOperation] = []
        literal_like = {
            InstructionKind.LITERAL,
            InstructionKind.ASCII_CHUNK,
            InstructionKind.PUSH,
            InstructionKind.TABLE_LOOKUP,
        }
        for idx, profile in enumerate(profiles):
            if profile.kind is not InstructionKind.REDUCE:
                continue
            left = idx - 1
            while left >= 0:
                previous = profiles[left]
                if previous.kind in literal_like or previous.kind is InstructionKind.META:
                    left -= 1
                    continue
                if previous.is_literal_marker():
                    left -= 1
                    continue
                break
            left += 1
            run = profiles[left:idx]
            if len(run) < 2:
                continue
            has_push = any(item.kind is InstructionKind.PUSH for item in run)
            has_table = any(item.kind is InstructionKind.TABLE_LOOKUP for item in run)
            has_ascii = any(item.kind is InstructionKind.ASCII_CHUNK for item in run)
            if has_push or has_table:
                name = "table_build"
            elif has_ascii:
                name = "tuple_build"
            else:
                name = "array_build"
            notes = [f"reduce={profile.label}"]
            notes.append(f"literals={len(run)}")
            operations.append(
                NormalizedOperation(
                    name=name,
                    category="literal",
                    start_offset=profiles[left].word.offset,
                    end_offset=profile.word.offset,
                    length=idx - left + 1,
                    notes=tuple(notes),
                )
            )
        return operations

    def _predicate_macros(
        self, profiles: Sequence[InstructionProfile]
    ) -> List[NormalizedOperation]:
        operations: List[NormalizedOperation] = []
        literal_like = {
            InstructionKind.LITERAL,
            InstructionKind.PUSH,
            InstructionKind.TABLE_LOOKUP,
        }
        for idx, profile in enumerate(profiles):
            if profile.kind is not InstructionKind.TEST:
                continue
            start_idx = idx
            notes: List[str] = [f"test={profile.label}"]
            if idx > 0 and (
                profiles[idx - 1].kind in literal_like
                or profiles[idx - 1].is_literal_marker()
            ):
                start_idx = idx - 1
                notes.append(f"source={profiles[idx - 1].label}")
            operations.append(
                NormalizedOperation(
                    name="predicate_assign",
                    category="predicate",
                    start_offset=profiles[start_idx].word.offset,
                    end_offset=profile.word.offset,
                    length=idx - start_idx + 1,
                    notes=tuple(notes),
                )
            )
        return operations

    def _indirect_macros(
        self,
        profiles: Sequence[InstructionProfile],
        events: Sequence[StackEvent],
    ) -> List[NormalizedOperation]:
        operations: List[NormalizedOperation] = []
        for idx, profile in enumerate(profiles):
            if not profile.label.startswith("69:"):
                continue
            operand = profile.operand
            zone = classify_address_zone(operand)
            action = "load"
            if idx < len(events):
                event = events[idx]
                if event.kind is InstructionKind.INDIRECT_STORE:
                    action = "store"
                elif event.kind is InstructionKind.INDIRECT_LOAD:
                    action = "load"
            notes = [f"zone={zone}", f"addr=0x{operand:04X}"]
            if profile.mode:
                notes.append(f"mode=0x{profile.mode:02X}")
            operations.append(
                NormalizedOperation(
                    name=f"indirect_{action}",
                    category="memory",
                    start_offset=profile.word.offset,
                    end_offset=profile.word.offset,
                    length=1,
                    notes=tuple(notes),
                )
            )
        return operations


def classify_address_zone(operand: int) -> str:
    """Return a coarse address zone for ``operand`` used by indirect ops."""

    if operand < 0:
        return "unknown"
    zone_selector = (operand >> 12) & 0xF
    zone_map = {
        0x0: "frame.locals",
        0x1: "frame.locals",
        0x2: "frame.environment",
        0x3: "frame.environment",
        0x4: "global.shared",
        0x5: "global.shared",
        0x6: "global.shared",
        0x7: "global.shared",
        0x8: "global.constants",
        0x9: "global.constants",
        0xA: "global.constants",
        0xB: "global.constants",
        0xC: "global.state",
        0xD: "global.state",
        0xE: "global.state",
        0xF: "global.state",
    }
    return zone_map.get(zone_selector, "global.state")
