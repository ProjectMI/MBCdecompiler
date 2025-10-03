"""Instruction macro normalisation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .instruction_profile import InstructionKind, InstructionProfile


LiteralLikeKinds = {
    InstructionKind.LITERAL,
    InstructionKind.ASCII_CHUNK,
    InstructionKind.PUSH,
}


@dataclass(frozen=True)
class MacroSignature:
    """Description of a macro match covering a slice of profiles."""

    name: str
    kind: InstructionKind
    category: str
    start: int
    end: int
    extra: Optional[Mapping[str, object]] = None

    @property
    def span(self) -> int:
        return max(1, self.end - self.start)


class MacroNormalizer:
    """Collapse frequently occurring instruction forms into macro operations."""

    def apply(self, profiles: Sequence[InstructionProfile]) -> None:
        idx = 0
        total = len(profiles)
        while idx < total:
            match = (
                self._match_tail_dispatch(profiles, idx)
                or self._match_frame_return(profiles, idx)
                or self._match_literal_reduce(profiles, idx)
                or self._match_predicate(profiles, idx)
                or self._match_indirect_access(profiles, idx)
            )
            if match is None:
                idx += 1
                continue

            self._apply_match(profiles, match)
            idx = match.end

    # ------------------------------------------------------------------
    # matchers
    # ------------------------------------------------------------------
    def _match_tail_dispatch(
        self, profiles: Sequence[InstructionProfile], idx: int
    ) -> Optional[MacroSignature]:
        start = idx
        total = len(profiles)
        helper_count = 0
        cursor = idx

        while cursor < total and profiles[cursor].kind in {InstructionKind.CALL, InstructionKind.META} and not self._is_tail_dispatch(profiles[cursor]):
            helper_count += 1
            cursor += 1

        if cursor >= total:
            return None

        if not self._is_tail_dispatch(profiles[cursor]):
            if start == cursor and self._is_tail_dispatch(profiles[cursor]):
                pass
            else:
                if self._is_tail_dispatch(profiles[idx]):
                    cursor = idx
                else:
                    return None

        if self._is_tail_dispatch(profiles[cursor]):
            end = cursor + 1
            extra = {"helper_count": helper_count, "tail_label": profiles[cursor].label}
            return MacroSignature(
                name="tail_dispatch",
                kind=InstructionKind.MACRO_CALL,
                category="call",
                start=start,
                end=end,
                extra=extra,
            )

        return None

    def _match_frame_return(
        self, profiles: Sequence[InstructionProfile], idx: int
    ) -> Optional[MacroSignature]:
        total = len(profiles)
        cursor = idx
        teardown = 0

        if self._is_tail_dispatch(profiles[cursor]):
            return None

        while cursor < total and profiles[cursor].kind is InstructionKind.STACK_TEARDOWN:
            teardown += 1
            cursor += 1

        if cursor >= total:
            return None

        if profiles[cursor].kind not in {InstructionKind.RETURN, InstructionKind.TERMINATOR}:
            return None

        end = cursor + 1
        while end < total and profiles[end].kind in {InstructionKind.RETURN, InstructionKind.TERMINATOR}:
            end += 1

        extra = {"teardown_count": teardown}
        return MacroSignature(
            name="frame_return",
            kind=InstructionKind.MACRO_FRAME_END,
            category="return",
            start=idx,
            end=end,
            extra=extra,
        )

    def _match_literal_reduce(
        self, profiles: Sequence[InstructionProfile], idx: int
    ) -> Optional[MacroSignature]:
        total = len(profiles)
        cursor = idx
        literals = 0

        while cursor < total and profiles[cursor].kind in LiteralLikeKinds:
            literals += 1
            cursor += 1

        if literals == 0 or cursor >= total:
            return None

        reduce_count = 0
        table_like = False

        while cursor < total and profiles[cursor].kind in {
            InstructionKind.REDUCE,
            InstructionKind.TABLE_LOOKUP,
        }:
            reduce_count += 1
            if profiles[cursor].kind is InstructionKind.TABLE_LOOKUP or (
                isinstance(profiles[cursor].category, str)
                and "table" in profiles[cursor].category.lower()
            ):
                table_like = True
            cursor += 1

        if reduce_count == 0:
            return None

        name = "literal_array_builder"
        category = "literal_array"
        kind = InstructionKind.MACRO_LITERAL_ARRAY

        if table_like:
            name = "literal_table_builder"
            category = "literal_table"
            kind = InstructionKind.MACRO_LITERAL_TABLE
        elif reduce_count > 1:
            name = "literal_tuple_builder"
            category = "literal_tuple"
            kind = InstructionKind.MACRO_LITERAL_TUPLE

        extra = {
            "literal_count": literals,
            "reduce_count": reduce_count,
        }
        return MacroSignature(
            name=name,
            kind=kind,
            category=category,
            start=idx,
            end=cursor,
            extra=extra,
        )

    def _match_predicate(
        self, profiles: Sequence[InstructionProfile], idx: int
    ) -> Optional[MacroSignature]:
        profile = profiles[idx]

        category = (profile.category or "").lower()
        mnemonic = profile.mnemonic.lower()
        summary = (profile.summary or "").lower()

        if not (
            "test" in category
            or "test" in mnemonic
            or "test" in summary
        ):
            return None

        if profile.kind not in {InstructionKind.BRANCH, InstructionKind.TEST}:
            return None

        return MacroSignature(
            name="predicate_assign",
            kind=InstructionKind.MACRO_PREDICATE,
            category="predicate",
            start=idx,
            end=idx + 1,
            extra={"source_label": profile.label},
        )

    def _match_indirect_access(
        self, profiles: Sequence[InstructionProfile], idx: int
    ) -> Optional[MacroSignature]:
        profile = profiles[idx]
        if profile.kind not in {
            InstructionKind.INDIRECT,
            InstructionKind.INDIRECT_LOAD,
            InstructionKind.INDIRECT_STORE,
            InstructionKind.TABLE_LOOKUP,
        }:
            return None

        operand = profile.operand
        if operand < 0x2000:
            name = "frame_slot_access"
            category = "frame_access"
            kind = InstructionKind.MACRO_FRAME_SLOT
            zone = "frame"
        else:
            name = "global_slot_access"
            category = "global_access"
            kind = InstructionKind.MACRO_GLOBAL_SLOT
            zone = "global"

        extra = {
            "zone": zone,
            "operand": operand,
        }
        return MacroSignature(
            name=name,
            kind=kind,
            category=category,
            start=idx,
            end=idx + 1,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_match(profiles: Sequence[InstructionProfile], match: MacroSignature) -> None:
        leader = profiles[match.start]
        leader.tag_macro(match.name, match.kind, category=match.category, span=match.span, extra=match.extra)
        for position in range(match.start + 1, match.end):
            profiles[position].mark_macro_member(match.name)

    @staticmethod
    def _is_tail_dispatch(profile: InstructionProfile) -> bool:
        if profile.kind is InstructionKind.TAILCALL:
            return True
        mnemonic = profile.mnemonic.lower()
        if "tail" in mnemonic:
            return True
        if profile.label.startswith("29:"):
            return True
        category = (profile.category or "").lower()
        if "tail" in category:
            return True
        return False
