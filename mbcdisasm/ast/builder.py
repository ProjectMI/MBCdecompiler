"""Reconstruct an AST and CFG from the normalised IR programme."""

from __future__ import annotations

import re

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..constants import FANOUT_FLAGS_A, FANOUT_FLAGS_B, RET_MASK
from ..ir.model import (
    IRBlock,
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRFunctionPrologue,
    IRIORead,
    IRIOWrite,
    IRIf,
    IRFlagCheck,
    IRDispatchCase,
    IRBankedLoad,
    IRBankedStore,
    IRIndirectLoad,
    IRIndirectStore,
    IRLoad,
    IRSlot,
    IRProgram,
    IRReturn,
    IRSegment,
    IRStore,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRAbiEffect,
    IRStackEffect,
    MemRef,
    MemSpace,
    SSAValueKind,
)


_DIRECT_EPILOGUE_KIND_MAP = {
    "call_helpers": "helpers.invoke",
    "fanout": "helpers.fanout",
    "page_register": "frame.page_select",
    "stack_teardown": "frame.teardown",
    "op_6C_01": "frame.page_select",
    "op_08_00": "helpers.dispatch",
    "op_72_23": "helpers.wrapper",
    "op_52_06": "io.handshake",
    "op_0B_00": "io.handshake",
    "op_0B_E1": "io.handshake",
    "op_76_41": "io.bridge",
    "op_04_EC": "io.bridge",
    "op_2D_01": "frame.cleanup",
    "reduce_passthrough": "helpers.reduce",
}

_MASK_STEP_MNEMONICS = {
    "epilogue",
    "op_29_10",
    "op_31_52",
    "op_32_29",
    "op_4B_0B",
    "op_4B_45",
    "op_4B_CC",
    "op_4F_01",
    "op_4F_02",
    "op_4F_03",
    "op_52_05",
    "op_5E_29",
    "op_70_29",
    "op_72_4A",
}

_MASK_OPERANDS = {RET_MASK, FANOUT_FLAGS_A, FANOUT_FLAGS_B}
_MASK_ALIASES = {"RET_MASK", "FANOUT_FLAGS"}

_FRAME_DROP_MNEMONICS = {
    "op_01_0C",
    "op_01_30",
    "op_01_78",
    "op_01_C0",
    "op_2D_01",
}

_IO_BRIDGE_MNEMONICS = {
    "op_3A_01",
    "op_3E_4B",
    "op_E1_4D",
    "op_E8_4B",
    "op_ED_4B",
    "op_F0_4B",
}

_IO_STEP_MNEMONICS = {
    "op_10_08",
    "op_10_0A",
    "op_10_0E",
    "op_10_18",
    "op_10_3C",
    "op_10_69",
    "op_10_C5",
    "op_10_CC",
    "op_10_CD",
    "op_10_E3",
    "op_10_E5",
    "op_15_4A",
    "op_18_2A",
    "op_20_8C",
    "op_21_94",
    "op_21_AC",
    "op_28_6C",
    "op_52_06",
    "op_78_00",
    "op_89_01",
    "op_A0_03",
    "op_C0_00",
    "op_C3_01",
}

_IO_OPCODE_FALLBACK = {
    0x10,
    0x11,
    0x14,
    0x15,
    0x18,
    0x1C,
    0x1D,
    0x20,
    0x21,
    0x28,
    0x52,
    0x78,
    0x89,
    0xA0,
    0xA2,
    0xC0,
    0xC3,
}
_MASK_OPCODE_FALLBACK = {0x29, 0x31, 0x32, 0x4B, 0x4F, 0x52, 0x5E, 0x70, 0x72}
_DROP_OPCODE_FALLBACK = {0x01, 0x2D}
_BRIDGE_OPCODE_FALLBACK = {0x3A, 0x3E, 0x04, 0x4A, 0x4B, 0x76, 0xE1, 0xE8, 0xED, 0xF0}

_FRAME_CLEAR_ALIASES = {"fmt.buffer_reset"}
_FORMAT_PREFIXES = ("fmt.", "text.")
_FRAME_OPERAND_KIND_OVERRIDES = {
    0x0000: "frame.reset",
    0x0020: "helpers.format",
    0x0029: "frame.scheduler",
    0x002C: "frame.scheduler",
    0x0041: "frame.scheduler",
    0x0069: "frame.scheduler",
    0x006C: "frame.page_select",
    0x10E1: "io.bridge",
    0x2C04: "helpers.format",
    0x2DF0: "io.step",
    0x2EF0: "io.step",
    0x3100: "helpers.format",
    0x3E4B: "helpers.format",
    0x5B01: "helpers.format",
    0xED4D: "frame.page_select",
    0xF0EB: "io.step",
}
from .model import (
    ASTAliasInfo,
    ASTAliasKind,
    ASTAddressDescriptor,
    ASTAddressKind,
    ASTAddressLayout,
    ASTAddressSpace,
    ASTAccessPathElement,
    ASTAssign,
    ASTBitField,
    ASTBlock,
    ASTBranch,
    ASTCallABI,
    ASTCallArgumentSlot,
    ASTCallReturnSlot,
    ASTCallExpr,
    ASTCallOperand,
    ASTCallOperandKind,
    ASTCallResult,
    ASTCallStatement,
    ASTSignatureValue,
    ASTSymbolSignature,
    ASTSymbolType,
    ASTSymbolTypeFamily,
    ASTComment,
    ASTEffect,
    ASTEnumDecl,
    ASTEnumMember,
    ASTExpression,
    ASTEntryPoint,
    ASTEntryReason,
    ASTExitPoint,
    ASTExitReason,
    ASTFieldElement,
    ASTFlagCheck,
    ASTFrameCleanupEffect,
    ASTFrameDropEffect,
    ASTFrameMaskEffect,
    ASTFramePageSelectEffect,
    ASTFrameProtocolEffect,
    ASTFrameResetEffect,
    ASTFrameTeardownEffect,
    ASTFrameWriteEffect,
    ASTHelperEffect,
    ASTHelperOperation,
    ASTIOEffect,
    ASTIOOperation,
    ASTIORead,
    ASTIOWrite,
    ASTIdentifier,
    ASTBooleanLiteral,
    ASTIndexElement,
    ASTIntegerLiteral,
    ASTMemoryLocation,
    ASTMemoryRead,
    ASTMemoryWrite,
    ASTMetrics,
    ASTNumericSign,
    ASTProcedure,
    ASTProgram,
    ASTReturn,
    ASTReturnPayload,
    ASTSegment,
    ASTStatement,
    ASTFunctionPrologue,
    ASTJump,
    ASTTerminator,
    ASTSwitch,
    ASTSwitchCase,
    ASTDispatchHelper,
    ASTDispatchIndex,
    ASTTailCall,
    ASTTestSet,
    ASTTupleExpr,
    ASTUnknown,
)


@dataclass
class _BlockAnalysis:
    """Cached information describing a block within a segment."""

    block: IRBlock
    successors: Tuple[int, ...]
    exit_reasons: Tuple[str, ...]
    fallthrough: Optional[int]


BranchStatement = ASTBranch | ASTTestSet | ASTFlagCheck | ASTFunctionPrologue


@dataclass
class _JumpLink:
    """Pending control-flow link for an unconditional jump."""

    statement: ASTJump
    target: int
    origin_offset: int


@dataclass
class _BranchLink:
    """Pending control-flow link for a branch-like statement."""

    statement: BranchStatement
    then_target: int
    else_target: int
    origin_offset: int


@dataclass
class _PendingBlock:
    """Block with unresolved successor references."""

    label: str
    start_offset: int
    statements: List[ASTStatement]
    successors: Tuple[int, ...]
    branch_links: List[_BranchLink]
    jump_links: List[_JumpLink]


@dataclass
class _FramePolicySummary:
    """Aggregated frame policy derived from cleanup sequences."""

    teardown: int = 0
    drops: int = 0
    masks: List[Tuple[int, Optional[str]]] = field(default_factory=list)

    def add_mask(self, value: Optional[int], alias: Optional[str]) -> None:
        if value is None:
            return
        entry = (value, alias)
        if entry not in self.masks:
            self.masks.append(entry)

    def merge(self, other: "_FramePolicySummary") -> None:
        self.teardown += other.teardown
        self.drops += other.drops
        for mask in other.masks:
            if mask not in self.masks:
                self.masks.append(mask)

    def has_effects(self) -> bool:
        return bool(self.teardown or self.drops or self.masks)


@dataclass
class _ProcedureAccumulator:
    """Partial reconstruction state for a single procedure."""

    entry_offset: int
    entry_reasons: Set[str] = field(default_factory=set)
    blocks: Dict[int, _PendingBlock] = field(default_factory=dict)


@dataclass
class _PendingDispatchCall:
    """Call statement awaiting an accompanying dispatch table."""

    helper: int
    index: int
    call: ASTCallExpr
    abi: Optional[ASTCallABI]


@dataclass
class _PendingDispatchTable:
    """Dispatch table awaiting a matching helper call."""

    dispatch: IRSwitchDispatch
    index: int


@dataclass
class _EnumInfo:
    """Aggregated information about a helper-backed enumeration."""

    decl: ASTEnumDecl
    member_names: Dict[int, str]
    owner_segment: int
    order: int
    switches: List["ASTSwitch"] = field(default_factory=list)


@dataclass
class _SignatureAccumulator:
    """Collect argument and return metadata for a single symbol."""

    address: int
    symbols: Set[str] = field(default_factory=set)
    argument_types: Dict[int, Set[ASTSymbolType]] = field(
        default_factory=lambda: defaultdict(set)
    )
    argument_names: Dict[int, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    return_types: Dict[int, Set[ASTSymbolType]] = field(
        default_factory=lambda: defaultdict(set)
    )
    return_names: Dict[int, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    calling_conventions: Set[str] = field(default_factory=set)
    attributes: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)

    def register_call(
        self,
        call: ASTCallExpr,
        returns: Sequence[ASTExpression],
        abi: Optional[ASTCallABI],
        payload_varargs: bool = False,
    ) -> None:
        if call.symbol:
            self.symbols.add(call.symbol)
        if call.varargs or payload_varargs:
            self.attributes.add("varargs")
        convention = "tailcall" if call.tail else "call"
        self.calling_conventions.add(convention)
        for index, operand in enumerate(call.operands):
            expr = operand.expression or ASTUnknown(operand.token or "")
            signature_type = self._classify_expr(expr)
            self.argument_types[index].add(signature_type)
            name = self._expr_name(expr)
            if name:
                self.argument_names[index].add(name)
        for index, expr in enumerate(returns):
            signature_type = self._classify_expr(expr)
            self.return_types[index].add(signature_type)
            name = self._expr_name(expr)
            if name:
                self.return_names[index].add(name)
        if abi is not None:
            if abi.tail:
                self.attributes.add("tail")
            for effect in abi.effects:
                self.effects.add(effect.render())

    @staticmethod
    def _expr_name(expr: ASTExpression) -> Optional[str]:
        if isinstance(expr, ASTIdentifier):
            return expr.name
        return None

    @staticmethod
    def _classify_expr(expr: ASTExpression) -> ASTSymbolType:
        if isinstance(expr, ASTIntegerLiteral):
            return ASTSymbolType(
                ASTSymbolTypeFamily.VALUE,
                width=expr.bits,
                sign=expr.sign,
            )
        if isinstance(expr, ASTBooleanLiteral):
            return ASTSymbolType(ASTSymbolTypeFamily.FLAG, width=1, sign=ASTNumericSign.UNSIGNED)
        if isinstance(expr, ASTMemoryRead) and expr.value_kind is not None:
            return _SignatureAccumulator._type_from_kind(expr.value_kind)
        if isinstance(expr, ASTIdentifier):
            return _SignatureAccumulator._type_from_kind(expr.kind())
        kind = expr.kind()
        return _SignatureAccumulator._type_from_kind(kind)

    @staticmethod
    def _type_from_kind(kind: SSAValueKind) -> ASTSymbolType:
        if kind is SSAValueKind.BOOLEAN:
            return ASTSymbolType(
                ASTSymbolTypeFamily.FLAG,
                width=1,
                sign=ASTNumericSign.UNSIGNED,
            )
        if kind is SSAValueKind.BYTE:
            return ASTSymbolType(
                ASTSymbolTypeFamily.VALUE,
                width=8,
                sign=ASTNumericSign.UNSIGNED,
            )
        if kind is SSAValueKind.WORD:
            return ASTSymbolType(
                ASTSymbolTypeFamily.VALUE,
                width=16,
                sign=ASTNumericSign.UNSIGNED,
            )
        if kind is SSAValueKind.POINTER:
            return ASTSymbolType(
                ASTSymbolTypeFamily.ADDRESS,
                width=16,
                space="mem",
            )
        if kind is SSAValueKind.IO:
            return ASTSymbolType(
                ASTSymbolTypeFamily.ADDRESS,
                width=16,
                space="io",
            )
        if kind is SSAValueKind.PAGE_REGISTER:
            return ASTSymbolType(
                ASTSymbolTypeFamily.ADDRESS,
                width=16,
                space="page",
            )
        if kind is SSAValueKind.IDENTIFIER:
            return ASTSymbolType(ASTSymbolTypeFamily.TOKEN)
        return ASTSymbolType(ASTSymbolTypeFamily.OPAQUE)

class ASTBuilder:
    """Construct a high level AST with CFG and reconstruction metrics."""

    def __init__(self) -> None:
        self._current_analyses: Mapping[int, _BlockAnalysis] = {}
        self._current_entry_reasons: Mapping[int, Tuple[str, ...]] = {}
        self._current_block_labels: Mapping[int, str] = {}
        self._current_exit_hints: Mapping[int, str] = {}
        self._current_redirects: Mapping[int, int] = {}
        self._pending_call_frame: List[IRStackEffect] = []
        self._pending_epilogue: List[IRStackEffect] = []
        self._enum_infos: Dict[str, _EnumInfo] = {}
        self._enum_order: List[str] = []
        self._enum_name_usage: Dict[str, str] = {}
        self._segment_declared_enum_keys: List[str] = []
        self._segment_declared_enum_set: Set[str] = set()
        self._current_segment_index: int = -1
        self._call_arg_values: Dict[str, ASTExpression] = {}
        self._expression_lookup: Dict[str, ASTExpression] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, program: IRProgram) -> ASTProgram:
        self._enum_infos = {}
        self._enum_order = []
        self._enum_name_usage = {}
        self._call_arg_values = {}
        self._expression_lookup = {}
        segments: List[ASTSegment] = []
        segment_enum_keys: List[Tuple[str, ...]] = []
        metrics = ASTMetrics()
        for segment in program.segments:
            segment_result, enum_keys = self._build_segment(segment, metrics)
            segments.append(segment_result)
            segment_enum_keys.append(enum_keys)
        metrics.procedure_count = sum(len(seg.procedures) for seg in segments)
        metrics.block_count = sum(len(proc.blocks) for seg in segments for proc in seg.procedures)
        metrics.edge_count = sum(len(block.successors) for seg in segments for proc in seg.procedures for block in proc.blocks)
        segments, program_enums = self._finalise_enums(segments, segment_enum_keys)
        symbol_table = self._synthesise_symbol_signatures(segments)
        return ASTProgram(
            segments=tuple(segments),
            metrics=metrics,
            enums=program_enums,
            symbols=symbol_table,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_segment(
        self, segment: IRSegment, metrics: ASTMetrics
    ) -> Tuple[ASTSegment, Tuple[str, ...]]:
        self._segment_declared_enum_keys = []
        self._segment_declared_enum_set = set()
        self._current_segment_index = segment.index
        block_map: Dict[int, IRBlock] = {block.start_offset: block for block in segment.blocks}
        analyses = self._build_cfg(segment, block_map)
        entry_reasons = self._detect_entries(segment, block_map, analyses)
        analyses, entry_reasons, redirects = self._compact_cfg(analyses, entry_reasons)
        self._current_analyses = analyses
        self._current_entry_reasons = entry_reasons
        self._current_block_labels = {offset: analysis.block.label for offset, analysis in analyses.items()}
        self._current_exit_hints = {
            offset: self._format_exit_hint(analysis.exit_reasons)
            for offset, analysis in analyses.items()
            if analysis.exit_reasons
        }
        self._current_redirects = redirects
        procedures = self._group_procedures(segment, analyses, entry_reasons, metrics)
        enum_keys = tuple(self._segment_declared_enum_keys)
        segment_kind = "code" if procedures else "data"
        result = ASTSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            procedures=tuple(procedures),
            enums=tuple(self._enum_infos[key].decl for key in enum_keys),
            kind=segment_kind,
        )
        self._clear_context()
        return result, enum_keys

    def _finalise_enums(
        self,
        segments: Sequence[ASTSegment],
        segment_enum_keys: Sequence[Tuple[str, ...]],
    ) -> Tuple[List[ASTSegment], Tuple[ASTEnumDecl, ...]]:
        active_keys: List[str] = []
        for key in self._enum_order:
            info = self._enum_infos[key]
            if not info.switches or len(info.member_names) <= 1:
                self._deactivate_enum(info)
                continue
            active_keys.append(key)
        active_lookup = set(active_keys)
        updated_segments = [
            replace(
                segment,
                enums=tuple(
                    self._enum_infos[key].decl for key in keys if key in active_lookup
                ),
            )
            for segment, keys in zip(segments, segment_enum_keys)
        ]
        program_enums = tuple(self._enum_infos[key].decl for key in active_keys)
        return updated_segments, program_enums

    def _synthesise_symbol_signatures(
        self, segments: Sequence[ASTSegment]
    ) -> Tuple[ASTSymbolSignature, ...]:
        accumulators: Dict[int, _SignatureAccumulator] = {}
        for segment in segments:
            for procedure in segment.procedures:
                for block in procedure.blocks:
                    for statement in block.statements:
                        if isinstance(statement, ASTCallStatement):
                            accumulator = accumulators.setdefault(
                                statement.call.target,
                                _SignatureAccumulator(statement.call.target),
                            )
                            accumulator.register_call(
                                statement.call,
                                statement.returns,
                                statement.abi,
                            )
                        elif isinstance(statement, ASTTailCall):
                            accumulator = accumulators.setdefault(
                                statement.call.target,
                                _SignatureAccumulator(statement.call.target),
                            )
                            accumulator.register_call(
                                statement.call,
                                statement.payload.values,
                                statement.abi,
                                payload_varargs=statement.payload.varargs,
                            )
        signatures: List[ASTSymbolSignature] = []
        for address, accumulator in accumulators.items():
            name = self._canonical_symbol_name(address, accumulator.symbols)
            arguments: List[ASTSignatureValue] = []
            for index, type_set in sorted(accumulator.argument_types.items()):
                signature_type = self._select_signature_type(type_set)
                arg_name = self._select_signature_name(
                    accumulator.argument_names.get(index, set()),
                    signature_type,
                    index,
                    is_return=False,
                )
                arguments.append(
                    ASTSignatureValue(index=index, type=signature_type, name=arg_name)
                )
            returns: List[ASTSignatureValue] = []
            for index, type_set in sorted(accumulator.return_types.items()):
                signature_type = self._select_signature_type(type_set)
                ret_name = self._select_signature_name(
                    accumulator.return_names.get(index, set()),
                    signature_type,
                    index,
                    is_return=True,
                )
                returns.append(
                    ASTSignatureValue(index=index, type=signature_type, name=ret_name)
                )
            calling_conventions = (
                tuple(sorted(accumulator.calling_conventions))
                if accumulator.calling_conventions
                else ("call",)
            )
            attributes = tuple(sorted(accumulator.attributes))
            effects = tuple(sorted(accumulator.effects))
            signatures.append(
                ASTSymbolSignature(
                    address=address,
                    name=name,
                    arguments=tuple(arguments),
                    returns=tuple(returns),
                    calling_conventions=calling_conventions,
                    attributes=attributes,
                    effects=effects,
                )
            )
        signatures.sort(key=lambda entry: (entry.name, entry.address))
        return tuple(signatures)

    @staticmethod
    def _select_signature_type(types: Set[ASTSymbolType]) -> ASTSymbolType:
        if not types:
            return ASTSymbolType(ASTSymbolTypeFamily.OPAQUE)

        def sort_key(entry: ASTSymbolType) -> Tuple[int, str, int]:
            priority = {
                ASTSymbolTypeFamily.VALUE: 0,
                ASTSymbolTypeFamily.FLAG: 1,
                ASTSymbolTypeFamily.ADDRESS: 2,
                ASTSymbolTypeFamily.TOKEN: 3,
                ASTSymbolTypeFamily.EFFECT: 4,
                ASTSymbolTypeFamily.OPAQUE: 5,
            }
            return (
                priority.get(entry.family, 99),
                entry.space or "",
                entry.width or 0,
            )

        ordered = sorted(types, key=sort_key)
        primary = ordered[0]
        same_family = [entry for entry in ordered if entry.family is primary.family]
        width_candidates = [entry.width for entry in same_family if entry.width is not None]
        width = max(width_candidates) if width_candidates else None
        sign_candidates = [entry.sign for entry in same_family if entry.sign is not None]
        sign: Optional[ASTNumericSign]
        if not sign_candidates:
            sign = primary.sign
        else:
            sign = sign_candidates[0]
            if any(candidate != sign for candidate in sign_candidates[1:]):
                sign = None
        space = next((entry.space for entry in same_family if entry.space), primary.space)
        return ASTSymbolType(primary.family, width=width, sign=sign, space=space)

    @staticmethod
    def _select_signature_name(
        names: Set[str],
        signature_type: ASTSymbolType,
        index: int,
        *,
        is_return: bool,
    ) -> Optional[str]:
        placeholder_pattern = re.compile(r"^(?:ret|id|word|byte|ptr|io|page|value)\d+$")
        filtered = sorted(
            name
            for name in names
            if name
            and all(ch not in name for ch in ":@")
            and not placeholder_pattern.match(name.lower())
        )
        if filtered:
            return filtered[0]
        base = ASTBuilder._placeholder_base(signature_type, is_return)
        return f"{base}{index}"

    @staticmethod
    def _placeholder_base(signature_type: ASTSymbolType, is_return: bool) -> str:
        mapping = {
            ASTSymbolTypeFamily.VALUE: "value" if not is_return else "result",
            ASTSymbolTypeFamily.ADDRESS: "addr",
            ASTSymbolTypeFamily.TOKEN: "token",
            ASTSymbolTypeFamily.FLAG: "flag",
            ASTSymbolTypeFamily.EFFECT: "effect",
            ASTSymbolTypeFamily.OPAQUE: "opaque",
        }
        return mapping.get(signature_type.family, "value")

    @staticmethod
    def _canonical_symbol_name(address: int, candidates: Set[str]) -> str:
        placeholder_pattern = re.compile(r"^(?:ret|id|word|byte|ptr|io|page|value|addr|opaque|result)\d+$")
        filtered = sorted(
            name
            for name in candidates
            if name and not placeholder_pattern.match(name.lower())
        )
        if filtered:
            return filtered[0]
        masked = address & 0xFFFF
        if 0x6600 <= masked <= 0x66FF:
            return f"helper_{masked:04X}"
        return f"proc_{masked:04X}"

    @staticmethod
    def _effect_identity(effect: ASTEffect) -> Tuple[Any, ...]:
        frame_kinds = (
            ASTFrameMaskEffect,
            ASTFrameWriteEffect,
            ASTFrameResetEffect,
            ASTFramePageSelectEffect,
            ASTFrameTeardownEffect,
            ASTFrameDropEffect,
            ASTFrameCleanupEffect,
        )
        if isinstance(effect, frame_kinds):
            if isinstance(effect, ASTFrameMaskEffect):
                mask = effect.mask
                return (
                    "frame_mask",
                    mask.width,
                    mask.value,
                    mask.alias,
                )
            if isinstance(effect, ASTFrameWriteEffect):
                mask_info = None
                if effect.mask is not None:
                    mask_info = (
                        effect.mask.width,
                        effect.mask.value,
                        effect.mask.alias,
                    )
                return ("frame_write", effect.channel, mask_info)
            if isinstance(effect, ASTFrameResetEffect):
                mask_info = None
                if effect.mask is not None:
                    mask_info = (
                        effect.mask.width,
                        effect.mask.value,
                        effect.mask.alias,
                    )
                return ("frame_reset", effect.channel, mask_info)
            if isinstance(effect, ASTFramePageSelectEffect):
                mask_info = None
                if effect.mask is not None:
                    mask_info = (
                        effect.mask.width,
                        effect.mask.value,
                        effect.mask.alias,
                    )
                return ("frame_page", effect.selector, mask_info)
            if isinstance(effect, ASTFrameTeardownEffect):
                return ("frame_teardown", effect.pops)
            if isinstance(effect, ASTFrameDropEffect):
                return ("frame_drop", effect.pops)
            if isinstance(effect, ASTFrameCleanupEffect):
                mask_info = None
                if effect.mask is not None:
                    mask_info = (
                        effect.mask.width,
                        effect.mask.value,
                        effect.mask.alias,
                    )
                return ("frame_effect", effect.label, mask_info)
        if isinstance(effect, ASTFrameProtocolEffect):
            masks = tuple(
                (mask.width, mask.value, mask.alias) for mask in effect.masks
            )
            return ("frame_protocol", masks, effect.teardown, effect.drops)
        if isinstance(effect, ASTIOEffect):
            mask = None
            if effect.mask is not None:
                mask = (
                    effect.mask.width,
                    effect.mask.value,
                    effect.mask.alias,
                )
            return ("io", effect.operation, effect.port, mask)
        if isinstance(effect, ASTHelperEffect):
            mask = None
            if effect.mask is not None:
                mask = (
                    effect.mask.width,
                    effect.mask.value,
                    effect.mask.alias,
                )
            return ("helper", effect.operation, effect.target, effect.symbol, mask)
        return ("effect", type(effect).__name__, effect.render())

    @staticmethod
    def _deactivate_enum(info: _EnumInfo) -> None:
        for switch in info.switches:
            switch.enum_name = None
            for case in switch.cases:
                case.key_alias = None

    def _build_cfg(
        self,
        segment: IRSegment,
        block_map: Mapping[int, IRBlock],
    ) -> Mapping[int, _BlockAnalysis]:
        analyses: Dict[int, _BlockAnalysis] = {}
        offsets = [block.start_offset for block in segment.blocks]
        for idx, block in enumerate(segment.blocks):
            successors: Set[int] = set()
            exit_reasons: List[str] = []
            fallthrough = offsets[idx + 1] if idx + 1 < len(offsets) else None
            for node in reversed(block.nodes):
                if isinstance(node, IRReturn):
                    exit_reasons.append("return")
                    successors.clear()
                    break
                if isinstance(node, (IRTailCall, IRTailcallReturn)):
                    exit_reasons.append("tail_call")
                    successors.clear()
                    break
                if isinstance(node, IRIf):
                    successors.update({node.then_target, node.else_target})
                    break
                if isinstance(node, IRTestSetBranch):
                    successors.update({node.then_target, node.else_target})
                    break
                if isinstance(node, IRFunctionPrologue):
                    successors.update({node.then_target, node.else_target})
                    break
            if not successors and fallthrough is not None and not exit_reasons:
                successors.add(fallthrough)
            analyses[block.start_offset] = _BlockAnalysis(
                block=block,
                successors=tuple(sorted(successors)),
                exit_reasons=tuple(exit_reasons),
                fallthrough=fallthrough,
            )
        return analyses

    def _detect_entries(
        self,
        segment: IRSegment,
        block_map: Mapping[int, IRBlock],
        analyses: Mapping[int, _BlockAnalysis],
    ) -> Mapping[int, Tuple[str, ...]]:
        entry_reasons: Dict[int, Set[str]] = defaultdict(set)
        entry_reasons[segment.start].add("segment_start")
        for offset, analysis in analyses.items():
            block = analysis.block
            for node in block.nodes:
                if isinstance(node, IRFunctionPrologue):
                    entry_reasons[offset].add("prologue")
                if isinstance(node, (IRCall, IRCallReturn, IRTailCall)):
                    target = node.target
                    if target in block_map:
                        reason = "tail_target" if getattr(node, "tail", False) else "call_target"
                        entry_reasons[target].add(reason)
        return {offset: tuple(sorted(reasons)) for offset, reasons in entry_reasons.items()}

    def _compact_cfg(
        self,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
    ) -> Tuple[
        Mapping[int, _BlockAnalysis],
        Mapping[int, Tuple[str, ...]],
        Mapping[int, int],
    ]:
        """Collapse trivial cleanup/terminator blocks into their successors."""

        trivial_targets: Dict[int, int] = {}
        for offset, analysis in analyses.items():
            if offset in entry_reasons:
                continue
            if analysis.exit_reasons:
                continue
            if len(analysis.successors) != 1:
                continue
            block = analysis.block
            if not block.nodes:
                trivial_targets[offset] = analysis.successors[0]
                continue
            if all(isinstance(node, (IRCallCleanup, IRTerminator)) for node in block.nodes):
                trivial_targets[offset] = analysis.successors[0]

        if not trivial_targets:
            return analyses, entry_reasons, {}

        def resolve(target: int) -> int:
            seen: Set[int] = set()
            while target in trivial_targets and target not in seen:
                seen.add(target)
                target = trivial_targets[target]
            return target

        redirects = {offset: resolve(target) for offset, target in trivial_targets.items()}

        compacted: Dict[int, _BlockAnalysis] = {}
        for offset, analysis in analyses.items():
            if offset in trivial_targets:
                continue
            successors = tuple(
                sorted({resolve(candidate) for candidate in analysis.successors})
            )
            fallthrough = analysis.fallthrough
            if fallthrough is not None:
                fallthrough = resolve(fallthrough)
                if fallthrough == offset:
                    fallthrough = None
            compacted[offset] = replace(
                analysis,
                successors=successors,
                fallthrough=fallthrough,
            )

        updated_entries = {
            offset: reasons
            for offset, reasons in entry_reasons.items()
            if offset in compacted
        }

        return compacted, updated_entries, redirects

    def _group_procedures(
        self,
        segment: IRSegment,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
        metrics: ASTMetrics,
    ) -> Sequence[ASTProcedure]:
        assigned: Dict[int, int] = {}
        accumulators: Dict[int, _ProcedureAccumulator] = {}
        order: List[int] = []

        for entry in sorted(entry_reasons):
            if entry not in analyses:
                continue
            reachable = self._collect_entry_blocks(entry, analyses, entry_reasons, assigned)
            if not reachable:
                continue
            accumulator = _ProcedureAccumulator(entry_offset=entry)
            accumulator.entry_reasons.update(entry_reasons[entry])
            order.append(entry)
            state: Dict[str, ASTExpression] = {}
            for offset in sorted(reachable):
                assigned[offset] = entry
                analysis = analyses[offset]
                accumulator.entry_reasons.update(entry_reasons.get(offset, ()))
                accumulator.blocks[offset] = self._convert_block(analysis, state, metrics)
            if not accumulator.entry_reasons:
                accumulator.entry_reasons.add("component")
            accumulators[entry] = accumulator

        for offset in sorted(analyses):
            if offset in assigned:
                continue
            reachable = self._collect_component(offset, analyses, assigned)
            if not reachable:
                continue
            accumulator = _ProcedureAccumulator(entry_offset=offset)
            accumulator.entry_reasons.add("component")
            order.append(offset)
            state: Dict[str, ASTExpression] = {}
            for node in sorted(reachable):
                assigned[node] = offset
                analysis = analyses[node]
                accumulator.entry_reasons.update(entry_reasons.get(node, ()))
                accumulator.blocks[node] = self._convert_block(analysis, state, metrics)
            accumulators[offset] = accumulator

        procedures: List[ASTProcedure] = []
        for index, entry in enumerate(order):
            accumulator = accumulators[entry]
            pending_blocks = [accumulator.blocks[offset] for offset in sorted(accumulator.blocks)]
            name = f"proc_{accumulator.entry_offset:04X}"
            procedure = self._finalise_procedure(
                name=name,
                entry_offset=accumulator.entry_offset,
                entry_reasons=self._normalise_entry_reasons(accumulator.entry_reasons),
                blocks=pending_blocks,
            )
            if procedure is None:
                continue
            procedures.append(procedure)
        return procedures

    @staticmethod
    def _normalise_entry_reasons(reasons: Set[str]) -> Tuple[str, ...]:
        if not reasons:
            return ("component",)
        return tuple(sorted(reasons))

    def _finalise_procedure(
        self,
        name: str,
        entry_offset: int,
        entry_reasons: Tuple[str, ...],
        blocks: Sequence[_PendingBlock],
    ) -> Optional[ASTProcedure]:
        realised_blocks = self._realise_blocks(blocks)
        simplified_blocks = self._simplify_blocks(realised_blocks, entry_offset)
        simplified_blocks = tuple(
            block for block in simplified_blocks if block.statements or block.successors
        )
        if not simplified_blocks:
            return None
        synthetic_only = True
        for block in simplified_blocks:
            analysis = self._current_analyses.get(block.start_offset)
            exit_reasons = analysis.exit_reasons if analysis else ()
            terminator = block.terminator
            if not (
                not block.body
                and isinstance(terminator, ASTReturn)
                and not terminator.effects
                and not terminator.payload.values
                and not terminator.payload.varargs
                and not exit_reasons
            ):
                synthetic_only = False
                break
        if synthetic_only:
            return None
        block_lookup = {block.start_offset: block for block in simplified_blocks}
        entry_block = block_lookup.get(entry_offset, simplified_blocks[0])
        entry_point = ASTEntryPoint(
            label=entry_block.label,
            offset=entry_block.start_offset,
            reasons=tuple(ASTEntryReason(kind=reason) for reason in entry_reasons),
        )
        exit_offsets = self._compute_exit_offsets_from_ast(simplified_blocks)
        exit_points: List[ASTExitPoint] = []
        for offset in exit_offsets:
            block = block_lookup.get(offset)
            if block is None:
                continue
            analysis = self._current_analyses.get(offset)
            reasons: Tuple[ASTExitReason, ...] = ()
            if analysis and analysis.exit_reasons:
                reasons = tuple(
                    ASTExitReason(kind=reason) for reason in analysis.exit_reasons
                )
            else:
                terminator = block.terminator
                reason: Optional[str] = None
                if isinstance(terminator, ASTReturn):
                    reason = "return"
                elif isinstance(terminator, ASTTailCall):
                    reason = "tail_call"
                elif isinstance(terminator, ASTJump):
                    reason = "jump"
                elif isinstance(terminator, ASTSwitch):
                    reason = "switch"
                elif isinstance(terminator, ASTBranch):
                    reason = "branch"
                elif isinstance(terminator, ASTTestSet):
                    reason = "testset"
                elif isinstance(terminator, ASTFlagCheck):
                    reason = "flag_check"
                elif isinstance(terminator, ASTFunctionPrologue):
                    reason = "prologue"
                if reason:
                    reasons = (ASTExitReason(kind=reason),)
            exit_points.append(
                ASTExitPoint(label=block.label, offset=offset, reasons=reasons)
            )
        successor_map: Dict[str, Tuple[str, ...]] = {
            block.label: tuple(sorted(successor.label for successor in block.successors))
            for block in simplified_blocks
        }
        predecessor_accum: Dict[str, List[str]] = {
            block.label: [] for block in simplified_blocks
        }
        for block in simplified_blocks:
            for successor in block.successors:
                if successor.label in predecessor_accum:
                    predecessor_accum[successor.label].append(block.label)
        predecessor_map = {
            label: tuple(sorted(values))
            for label, values in predecessor_accum.items()
        }
        return ASTProcedure(
            name=name,
            blocks=simplified_blocks,
            entry=entry_point,
            exits=tuple(exit_points),
            successor_map=successor_map,
            predecessor_map=predecessor_map,
        )

    def _collect_entry_blocks(
        self,
        entry: int,
        analyses: Mapping[int, _BlockAnalysis],
        entry_reasons: Mapping[int, Tuple[str, ...]],
        assigned: Mapping[int, int],
    ) -> Set[int]:
        reachable: Set[int] = set()
        stack: List[int] = [entry]
        while stack:
            offset = stack.pop()
            if offset in reachable:
                continue
            if offset in assigned and assigned[offset] != entry:
                continue
            analysis = analyses.get(offset)
            if analysis is None:
                continue
            reachable.add(offset)
            for successor in analysis.successors:
                if successor not in analyses:
                    continue
                if successor in entry_reasons and successor != entry:
                    continue
                stack.append(successor)
        return reachable

    def _collect_component(
        self,
        start: int,
        analyses: Mapping[int, _BlockAnalysis],
        assigned: Mapping[int, int],
    ) -> Set[int]:
        reachable: Set[int] = set()
        stack: List[int] = [start]
        while stack:
            offset = stack.pop()
            if offset in reachable or offset in assigned:
                continue
            analysis = analyses.get(offset)
            if analysis is None:
                continue
            reachable.add(offset)
            for successor in analysis.successors:
                if successor not in analyses:
                    continue
                stack.append(successor)
        return reachable

    def _compute_exit_offsets(self, blocks: Sequence[_PendingBlock]) -> Tuple[int, ...]:
        if not blocks:
            return tuple()

        exit_node = "__exit__"
        offsets = {block.start_offset for block in blocks}
        nodes = set(offsets)
        nodes.add(exit_node)

        postdom: Dict[int | str, Set[int | str]] = {
            offset: set(nodes) for offset in offsets
        }
        postdom[exit_node] = {exit_node}

        successors: Dict[int, Set[int | str]] = {}
        for block in blocks:
            analysis = self._current_analyses.get(block.start_offset)
            succ: Set[int | str] = set()
            if analysis is not None:
                if not analysis.successors:
                    succ.add(exit_node)
                for candidate in analysis.successors:
                    if candidate in offsets:
                        succ.add(candidate)
                    else:
                        succ.add(exit_node)
                if analysis.exit_reasons:
                    succ.add(exit_node)
            else:
                succ.add(exit_node)
            if not succ:
                succ.add(exit_node)
            successors[block.start_offset] = succ

        changed = True
        while changed:
            changed = False
            for offset in offsets:
                succ = successors.get(offset)
                if not succ:
                    succ = {exit_node}
                intersection = set(nodes)
                for candidate in succ:
                    intersection &= postdom[candidate]
                updated = {offset} | intersection
                if updated != postdom[offset]:
                    postdom[offset] = updated
                    changed = True

        exit_offsets: Set[int] = set()
        for offset in offsets:
            succ = successors.get(offset, {exit_node})
            if exit_node not in succ:
                continue
            candidates = postdom[offset] - {offset}
            if not candidates:
                continue
            immediate: Optional[int | str] = None
            for candidate in candidates:
                dominated = False
                for other in candidates:
                    if other == candidate:
                        continue
                    if candidate in postdom[other]:
                        dominated = True
                        break
                if not dominated:
                    immediate = candidate
                    break
            if immediate == exit_node or (immediate is None and exit_node in candidates):
                exit_offsets.add(offset)
        return tuple(sorted(exit_offsets))

    def _simplify_blocks(
        self, blocks: Tuple[ASTBlock, ...], entry_offset: int
    ) -> Tuple[ASTBlock, ...]:
        if not blocks:
            return tuple()

        block_order: List[ASTBlock] = list(blocks)
        entry_block = next(
            (block for block in block_order if block.start_offset == entry_offset),
            block_order[0],
        )

        id_map: Dict[int, ASTBlock] = {id(block): block for block in block_order}

        for block in list(block_order):
            self._simplify_branch_targets(block, block_order)
            self._simplify_stack_branches(block)

        for block in block_order:
            filtered = tuple(
                statement
                for statement in block.statements
                if not self._is_noise_statement(statement)
            )
            if filtered != block.statements:
                block.statements = filtered

        predecessors: Dict[int, Set[int]] = {block_id: set() for block_id in id_map}
        for block in block_order:
            deduped: List[ASTBlock] = []
            seen_ids: Set[int] = set()
            for successor in block.successors:
                succ_id = id(successor)
                if succ_id not in predecessors:
                    continue
                if succ_id not in seen_ids:
                    deduped.append(successor)
                    seen_ids.add(succ_id)
                predecessors[succ_id].add(id(block))
            block.successors = tuple(deduped)

        changed = True
        while changed:
            changed = False
            for block in list(block_order):
                if block.body:
                    continue
                if not isinstance(block.terminator, ASTJump):
                    continue
                if len(block.successors) != 1:
                    continue
                successor = block.successors[0]
                if successor is block:
                    continue
                succ_id = id(successor)
                block_id = id(block)
                if succ_id not in predecessors:
                    continue
                if block is entry_block:
                    entry_block = successor
                for pred_id in list(predecessors.get(block_id, set())):
                    pred = id_map.get(pred_id)
                    if pred is None:
                        continue
                    self._replace_successor(pred, block, successor)
                    predecessors[succ_id].add(pred_id)
                predecessors[succ_id].discard(block_id)
                block_order.remove(block)
                predecessors.pop(block_id, None)
                id_map.pop(block_id, None)
                changed = True
                break

        merged = True
        while merged:
            merged = False
            for block in list(block_order):
                if len(block.successors) != 1:
                    continue
                successor = block.successors[0]
                if successor is block:
                    continue
                block_id = id(block)
                succ_id = id(successor)
                if succ_id not in predecessors:
                    continue
                if len(predecessors[succ_id]) != 1:
                    continue
                if block.statements and isinstance(block.statements[-1], BranchStatement):
                    continue
                if successor in successor.successors:
                    continue
                block.statements = tuple(block.statements + successor.statements)
                block.successors = successor.successors
                for succ in successor.successors:
                    succ_key = id(succ)
                    pred_set = predecessors.setdefault(succ_key, set())
                    pred_set.discard(succ_id)
                    pred_set.add(block_id)
                block_order.remove(successor)
                predecessors.pop(succ_id, None)
                id_map.pop(succ_id, None)
                merged = True
                break

        removed = True
        while removed:
            removed = False
            for block in list(block_order):
                if block is entry_block:
                    continue
                block_id = id(block)
                if predecessors.get(block_id):
                    continue
                for succ in block.successors:
                    predecessors.setdefault(id(succ), set()).discard(block_id)
                block_order.remove(block)
                predecessors.pop(block_id, None)
                id_map.pop(block_id, None)
                removed = True
                break

        return tuple(block_order)

    def _simplify_stack_branches(self, block: ASTBlock) -> None:
        stack: List[Tuple[str, int | None]] = []
        for statement in list(block.statements):
            if isinstance(statement, ASTComment):
                effect = self._stack_effect_from_comment(statement.text)
                kind = effect[0]
                if kind == "literal":
                    stack.append(effect)
                elif kind == "marker":
                    stack.append(effect)
                else:
                    stack.clear()
                continue
            if isinstance(statement, ASTReturn):
                stack.clear()
                continue
            if isinstance(statement, ASTBranch):
                if self._is_stack_top_expr(statement.condition) and stack:
                    kind, value = stack[-1]
                    if kind == "literal" and value is not None:
                        target = statement.then_branch if value else statement.else_branch
                        if target is not None:
                            block.statements = tuple(
                                stmt for stmt in block.statements if stmt is not statement
                            )
                            block.successors = (target,)
                            return
                stack.clear()
                continue
            stack.clear()

    @staticmethod
    def _is_stack_top_expr(expr: ASTExpression) -> bool:
        return isinstance(expr, ASTIdentifier) and expr.name == "stack_top"

    def _simplify_branch_targets(
        self, block: ASTBlock, block_order: Sequence[ASTBlock]
    ) -> None:
        analysis = self._current_analyses.get(block.start_offset)
        fallthrough = analysis.fallthrough if analysis else None
        simplified: List[ASTStatement] = []
        for statement in block.statements:
            if isinstance(statement, BranchStatement):
                updated = self._simplify_branch_statement(
                    statement, block_order, fallthrough
                )
                if updated is None:
                    continue
                statement = updated
            simplified.append(statement)
        block.statements = tuple(simplified)

    def _simplify_branch_statement(
        self,
        statement: BranchStatement,
        block_order: Sequence[ASTBlock],
        fallthrough: Optional[int],
    ) -> Optional[ASTStatement]:
        then_target = self._ensure_branch_target(
            statement, "then", block_order, fallthrough
        )
        else_target = self._ensure_branch_target(
            statement, "else", block_order, fallthrough
        )
        if (
            then_target is not None
            and else_target is not None
            and then_target == else_target
        ):
            target_block = statement.then_branch or statement.else_branch
            target_offset = statement.then_offset or statement.else_offset
            if target_block is not None:
                target_offset = target_block.start_offset
            hint = statement.then_hint or statement.else_hint
            return ASTJump(target=target_block, target_offset=target_offset, hint=hint)
        return statement

    def _ensure_branch_target(
        self,
        statement: BranchStatement,
        prefix: str,
        block_order: Sequence[ASTBlock],
        fallthrough: Optional[int],
    ) -> Optional[int]:
        branch_attr = f"{prefix}_branch"
        hint_attr = f"{prefix}_hint"
        offset_attr = f"{prefix}_offset"
        branch = getattr(statement, branch_attr)
        offset = getattr(statement, offset_attr)
        if branch is not None:
            offset = branch.start_offset
        elif offset is None:
            hint = getattr(statement, hint_attr)
            if hint == "fallthrough":
                offset = fallthrough
        target_block = self._find_block_by_offset(block_order, offset)
        if target_block is not None:
            setattr(statement, branch_attr, target_block)
            setattr(statement, hint_attr, None)
            setattr(statement, offset_attr, target_block.start_offset)
            return target_block.start_offset
        setattr(statement, offset_attr, offset)
        return offset

    @staticmethod
    def _find_block_by_offset(
        blocks: Sequence[ASTBlock], offset: Optional[int]
    ) -> Optional[ASTBlock]:
        if offset is None:
            return None
        for candidate in blocks:
            if candidate.start_offset == offset:
                return candidate
        return None

    @staticmethod
    def _stack_effect_from_comment(text: str) -> Tuple[str, int | None]:
        body = text.strip()
        if body.startswith("lit(") and body.endswith(")"):
            literal = body[4:-1]
            try:
                value = int(literal, 16)
            except ValueError:
                return ("invalidate", None)
            return ("literal", value)
        if body.startswith("marker "):
            return ("marker", None)
        return ("invalidate", None)

    @staticmethod
    def _is_noise_statement(statement: ASTStatement) -> bool:
        if isinstance(statement, ASTComment):
            body = statement.text.strip()
            prefixes = ("lit(", "marker ", "literal_block", "ascii(")
            return body.startswith(prefixes)
        return False

    def _replace_successor(self, block: ASTBlock, old: ASTBlock, new: ASTBlock) -> None:
        block.successors = tuple(new if succ is old else succ for succ in block.successors)
        for statement in block.statements:
            if isinstance(statement, BranchStatement):
                if statement.then_branch is old:
                    statement.then_branch = new
                    statement.then_hint = None
                if statement.else_branch is old:
                    statement.else_branch = new
                    statement.else_hint = None
            elif isinstance(statement, ASTJump):
                if statement.target is old:
                    statement.target = new
                    statement.hint = None
                    statement.target_offset = new.start_offset

    def _compute_exit_offsets_from_ast(self, blocks: Sequence[ASTBlock]) -> Tuple[int, ...]:
        if not blocks:
            return tuple()

        exit_node = "__exit__"
        offsets = {block.start_offset for block in blocks}
        nodes: Set[int | str] = set(offsets)
        nodes.add(exit_node)

        postdom: Dict[int | str, Set[int | str]] = {
            offset: set(nodes) for offset in offsets
        }
        postdom[exit_node] = {exit_node}

        successors: Dict[int, Set[int | str]] = {}
        for block in blocks:
            succ: Set[int | str] = set()
            if not block.successors:
                succ.add(exit_node)
            for candidate in block.successors:
                if candidate.start_offset in offsets:
                    succ.add(candidate.start_offset)
            if self._block_has_exit(block):
                succ.add(exit_node)
            if not succ:
                succ.add(exit_node)
            successors[block.start_offset] = succ

        changed = True
        while changed:
            changed = False
            for offset in offsets:
                succ = successors.get(offset, {exit_node})
                if not succ:
                    succ = {exit_node}
                intersection = set(nodes)
                for candidate in succ:
                    intersection &= postdom.get(candidate, {exit_node})
                updated = {offset} | intersection
                if updated != postdom[offset]:
                    postdom[offset] = updated
                    changed = True

        exit_offsets: Set[int] = set()
        for offset in offsets:
            if exit_node not in successors.get(offset, {exit_node}):
                continue
            candidates = postdom[offset] - {offset}
            if not candidates:
                continue
            immediate: Optional[int | str] = None
            for candidate in candidates:
                dominated = False
                for other in candidates:
                    if other == candidate:
                        continue
                    if candidate in postdom.get(other, set()):
                        dominated = True
                        break
                if not dominated:
                    immediate = candidate
                    break
            if immediate == exit_node or (immediate is None and exit_node in candidates):
                exit_offsets.add(offset)
        return tuple(sorted(exit_offsets))

    @staticmethod
    def _block_has_exit(block: ASTBlock) -> bool:
        return isinstance(block.terminator, (ASTReturn, ASTTailCall))

    def _realise_blocks(self, blocks: Sequence[_PendingBlock]) -> Tuple[ASTBlock, ...]:
        block_map: Dict[int, ASTBlock] = {
            block.start_offset: ASTBlock(
                label=block.label,
                start_offset=block.start_offset,
                body=tuple(),
                terminator=ASTReturn(payload=ASTReturnPayload(), effects=tuple()),
                successors=tuple(),
            )
            for block in blocks
        }
        for pending in blocks:
            for link in pending.branch_links:
                then_block = block_map.get(link.then_target)
                if then_block is not None:
                    link.statement.then_branch = then_block
                else:
                    link.statement.then_hint = self._describe_branch_target(
                        link.origin_offset, link.then_target
                    )
                else_block = block_map.get(link.else_target)
                if else_block is not None:
                    link.statement.else_branch = else_block
                else:
                    link.statement.else_hint = self._describe_branch_target(
                        link.origin_offset, link.else_target
                    )
            for link in pending.jump_links:
                target_block = block_map.get(link.target)
                if target_block is not None:
                    link.statement.target = target_block
                    link.statement.target_offset = target_block.start_offset
                else:
                    link.statement.hint = self._describe_branch_target(
                        link.origin_offset, link.target
                    )
                    link.statement.target_offset = link.target
            realised = block_map[pending.start_offset]
            body, terminator = self._split_block_statements(pending.statements)
            realised.body = body
            realised.terminator = terminator
            realised.successors = tuple(
                block_map[target]
                for target in pending.successors
                if target in block_map
            )
        return tuple(block_map[block.start_offset] for block in blocks)

    @staticmethod
    def _split_block_statements(
        statements: Sequence[ASTStatement],
    ) -> Tuple[Tuple[ASTStatement, ...], ASTTerminator]:
        if not statements:
            raise ValueError("block without terminator")
        terminator = statements[-1]
        if not isinstance(terminator, ASTTerminator):
            raise ValueError(f"non-terminator {type(terminator).__name__} at block end")
        body = tuple(statements[:-1])
        return body, terminator

    def _ensure_block_terminator(
        self,
        analysis: _BlockAnalysis,
        statements: List[ASTStatement],
        jump_links: List[_JumpLink],
    ) -> Optional[ASTTerminator]:
        if statements:
            for index in range(len(statements) - 1, -1, -1):
                candidate = statements[index]
                if isinstance(candidate, ASTTerminator):
                    if index != len(statements) - 1:
                        del statements[index + 1 :]
                    return None
        if analysis.successors:
            targets = analysis.successors
            unique_targets = tuple(sorted({*targets}))
            if len(unique_targets) == 1:
                target = unique_targets[0]
                jump = ASTJump(target_offset=target)
                jump_links.append(
                    _JumpLink(
                        statement=jump,
                        target=target,
                        origin_offset=analysis.block.start_offset,
                    )
                )
                return jump
            raise ValueError(
                "block with multiple successors lacks terminator: "
                f"0x{analysis.block.start_offset:04X} -> "
                f"{', '.join(f'0x{target:04X}' for target in targets)}"
            )
        # No recorded successors: synthesise a canonical return terminator.
        payload = ASTReturnPayload()
        return ASTReturn(payload=payload, effects=tuple())

    def _convert_block(
        self,
        analysis: _BlockAnalysis,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> _PendingBlock:
        block = analysis.block
        statements: List[ASTStatement] = []
        branch_links: List[_BranchLink] = []
        jump_links: List[_JumpLink] = []
        pending_calls: List[_PendingDispatchCall] = []
        pending_tables: List[_PendingDispatchTable] = []
        self._pending_call_frame.clear()
        self._pending_epilogue.clear()
        for node in block.nodes:
            if isinstance(node, IRCall):
                self._handle_dispatch_call(
                    node,
                    value_state,
                    metrics,
                    statements,
                    pending_calls,
                    pending_tables,
                )
                continue
            if isinstance(node, IRSwitchDispatch):
                self._handle_dispatch_table(
                    node,
                    value_state,
                    statements,
                    pending_calls,
                    pending_tables,
                )
                continue
            node_statements, node_links = self._convert_node(
                node,
                block.start_offset,
                value_state,
                metrics,
            )
            statements.extend(node_statements)
            branch_links.extend(node_links)
        statements = self._collapse_dispatch_sequences(statements)
        terminator = self._ensure_block_terminator(analysis, statements, jump_links)
        if terminator is not None:
            statements.append(terminator)
        return _PendingBlock(
            label=block.label,
            start_offset=block.start_offset,
            statements=statements,
            successors=analysis.successors,
            branch_links=branch_links,
            jump_links=jump_links,
        )

    def _handle_dispatch_call(
        self,
        node: IRCall,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
        statements: List[ASTStatement],
        pending_calls: List[_PendingDispatchCall],
        pending_tables: List[_PendingDispatchTable],
    ) -> None:
        call_expr, arg_exprs = self._convert_call(
            node.target,
            node.args,
            node.symbol,
            node.tail,
            node.varargs if hasattr(node, "varargs") else False,
            value_state,
        )
        self._register_expression(node.describe(), call_expr)
        metrics.call_sites += 1
        metrics.observe_call_args(
            sum(1 for operand in call_expr.operands if operand.is_resolved()),
            len(call_expr.operands),
        )
        abi = self._build_call_abi(node, arg_exprs)
        if getattr(node, "cleanup", tuple()):
            self._pending_epilogue.extend(node.cleanup)
        pending_table = self._pop_dispatch_table(node.target, pending_tables)
        if pending_table is not None:
            target_index = pending_table.index
            statements[target_index] = self._build_dispatch_switch(
                call_expr, pending_table.dispatch, value_state, abi=abi
            )
            return
        index = len(statements)
        statements.append(ASTCallStatement(call=call_expr, abi=abi))
        pending_calls.append(
            _PendingDispatchCall(helper=node.target, index=index, call=call_expr, abi=abi)
        )

    def _handle_dispatch_table(
        self,
        dispatch: IRSwitchDispatch,
        value_state: MutableMapping[str, ASTExpression],
        statements: List[ASTStatement],
        pending_calls: List[_PendingDispatchCall],
        pending_tables: List[_PendingDispatchTable],
    ) -> None:
        call_info = self._pop_dispatch_call(dispatch, pending_calls)
        if call_info is not None:
            switch_statement = self._build_dispatch_switch(
                call_info.call, dispatch, value_state, abi=call_info.abi
            )
            statements[call_info.index] = switch_statement
            return
        index = len(statements)
        switch_statement = self._build_dispatch_switch(
            None, dispatch, value_state
        )
        statements.append(switch_statement)
        pending_tables.append(_PendingDispatchTable(dispatch=dispatch, index=index))

    def _dispatch_helper_matches(self, helper: int, dispatch: IRSwitchDispatch) -> bool:
        if dispatch.helper is not None:
            return dispatch.helper == helper
        return False

    def _pop_dispatch_table(
        self, helper: int, pending_tables: List[_PendingDispatchTable]
    ) -> Optional[_PendingDispatchTable]:
        for index in range(len(pending_tables) - 1, -1, -1):
            entry = pending_tables[index]
            if self._dispatch_helper_matches(helper, entry.dispatch):
                pending_tables.pop(index)
                return entry
        return None

    def _pop_dispatch_call(
        self, dispatch: IRSwitchDispatch, pending_calls: List[_PendingDispatchCall]
    ) -> Optional[_PendingDispatchCall]:
        helper = dispatch.helper
        if helper is None:
            return None
        for index in range(len(pending_calls) - 1, -1, -1):
            entry = pending_calls[index]
            if entry.helper == helper:
                return pending_calls.pop(index)
        return None

    def _collapse_dispatch_sequences(
        self, statements: List[ASTStatement]
    ) -> List[ASTStatement]:
        collapsed: List[ASTStatement] = []
        index = 0
        while index < len(statements):
            current = statements[index]
            replacements = self._simplify_dispatch_statement(current)
            for replacement in replacements:
                collapsed.append(replacement)
            if (
                replacements
                and isinstance(replacements[-1], (ASTSwitch, ASTCallStatement))
                and index + 1 < len(statements)
                and self._is_redundant_dispatch_followup(
                    replacements[-1], statements[index + 1]
                )
            ):
                index += 1
            index += 1
        return collapsed

    def _simplify_dispatch_statement(
        self, statement: ASTStatement
    ) -> List[ASTStatement]:
        if not isinstance(statement, ASTSwitch):
            return [statement]
        if len(statement.cases) != 1:
            return [statement]
        if statement.call is None:
            return [statement]
        if statement.default is not None:
            return [statement]
        if statement.call.tail:
            return [
                ASTTailCall(
                    call=statement.call,
                    payload=ASTReturnPayload(values=tuple()),
                    abi=statement.abi,
                )
            ]
        return [ASTCallStatement(call=statement.call, abi=statement.abi)]

    def _is_redundant_dispatch_followup(
        self, primary: ASTStatement, statement: ASTStatement
    ) -> bool:
        call_expr = self._extract_dispatch_call(primary)
        if call_expr is None:
            return False
        if isinstance(statement, ASTTailCall):
            payload = statement.payload
            if payload.values or payload.varargs:
                return False
            if statement.effects:
                return False
            return self._calls_match(statement.call, call_expr)
        if isinstance(statement, ASTCallStatement):
            if statement.returns:
                return False
            return self._calls_match(statement.call, call_expr)
        return False

    @staticmethod
    def _extract_dispatch_call(statement: ASTStatement) -> ASTCallExpr | None:
        if isinstance(statement, ASTSwitch):
            return statement.call
        if isinstance(statement, ASTCallStatement):
            return statement.call
        if isinstance(statement, ASTTailCall):
            return statement.call
        return None

    @staticmethod
    def _calls_match(lhs: ASTCallExpr, rhs: ASTCallExpr) -> bool:
        return (
            lhs.target == rhs.target
            and lhs.symbol == rhs.symbol
            and lhs.operands == rhs.operands
            and lhs.varargs == rhs.varargs
        )

    def _register_expression(self, token: Optional[str], expr: ASTExpression) -> None:
        if not token:
            return
        self._expression_lookup[token] = expr

    def _build_dispatch_switch(
        self,
        call_expr: ASTCallExpr | None,
        dispatch: IRSwitchDispatch,
        value_state: Mapping[str, ASTExpression],
        abi: Optional[ASTCallABI] = None,
    ) -> ASTSwitch:
        collapse_dispatch = self._should_collapse_dispatch(call_expr, dispatch)
        enum_info: _EnumInfo | None
        enum_name: str | None
        if collapse_dispatch:
            enum_info = None
            enum_name = None
        else:
            enum_info = self._ensure_dispatch_enum(
                dispatch, call_expr.symbol if call_expr else None
            )
            enum_name = enum_info.decl.name if enum_info else None
        cases = self._build_dispatch_cases(dispatch.cases)
        index_expr, index_mask, index_base = self._resolve_dispatch_index(
            dispatch, value_state
        )
        kind = self._classify_dispatch_kind(dispatch)
        index_info = ASTDispatchIndex(
            expression=index_expr, mask=index_mask, base=index_base
        )
        helper_info = (
            ASTDispatchHelper(address=dispatch.helper, symbol=dispatch.helper_symbol)
            if dispatch.helper is not None
            else None
        )
        switch = ASTSwitch(
            call=call_expr,
            cases=cases,
            index=index_info,
            helper=helper_info,
            default=dispatch.default,
            kind=kind,
            enum_name=enum_name,
            abi=abi,
        )
        if enum_info:
            self._attach_enum_to_switch(enum_info, switch)
        return switch

    @staticmethod
    def _should_collapse_dispatch(
        call_expr: ASTCallExpr | None, dispatch: IRSwitchDispatch
    ) -> bool:
        if call_expr is None:
            return False
        if len(dispatch.cases) != 1:
            return False
        if dispatch.default is not None:
            return False
        return True

    def _resolve_dispatch_index(
        self,
        dispatch: IRSwitchDispatch,
        value_state: Mapping[str, ASTExpression],
    ) -> Tuple[ASTExpression | None, int | None, int | None]:
        index_info = dispatch.index
        if index_info is None:
            return None, None, None
        index_expr: ASTExpression | None = None
        if index_info.source:
            index_expr = self._resolve_expr(index_info.source, value_state)
        return index_expr, index_info.mask, index_info.base

    def _classify_dispatch_kind(self, dispatch: IRSwitchDispatch) -> str | None:
        helper = dispatch.helper
        symbol = dispatch.helper_symbol or ""
        if helper is None and not symbol:
            return None
        io_helper_addresses = {0x00F0, 0x0029, 0x002C, 0x0041, 0x0069}
        if helper in io_helper_addresses:
            return "io"
        lowered = symbol.lower()
        if lowered.startswith("io.") or lowered.startswith("scheduler.mask_"):
            return "io"
        return None

    def _build_dispatch_cases(
        self, cases: Sequence[IRDispatchCase]
    ) -> Tuple[ASTSwitchCase, ...]:
        return tuple(
            ASTSwitchCase(
                key=case.key,
                target=case.target,
                symbol=case.symbol,
            )
            for case in cases
        )

    def _ensure_dispatch_enum(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> _EnumInfo | None:
        key = self._dispatch_helper_key(dispatch, call_symbol)
        if key is None:
            return None
        info = self._enum_infos.get(key)
        if info is None:
            name = self._allocate_enum_name(dispatch, key, call_symbol)
            enum_decl = ASTEnumDecl(name=name, members=tuple())
            info = _EnumInfo(
                decl=enum_decl,
                member_names={},
                owner_segment=self._current_segment_index,
                order=len(self._enum_order),
            )
            self._enum_infos[key] = info
            self._enum_order.append(key)
            self._register_segment_enum(key, info)
        else:
            if info.owner_segment == self._current_segment_index:
                self._register_segment_enum(key, info)
        self._update_enum_members(info, dispatch, call_symbol)
        return info

    def _attach_enum_to_switch(self, info: _EnumInfo, switch: ASTSwitch) -> None:
        info.switches.append(switch)
        enum_name = info.decl.name
        switch.enum_name = enum_name
        for case in switch.cases:
            member_name = info.member_names.get(case.key)
            case.key_alias = (
                f"{enum_name}.{member_name}" if member_name is not None else None
            )

    def _register_segment_enum(self, key: str, info: _EnumInfo) -> None:
        if info.owner_segment != self._current_segment_index:
            return
        if key in self._segment_declared_enum_set:
            return
        self._segment_declared_enum_set.add(key)
        self._segment_declared_enum_keys.append(key)

    def _update_enum_members(
        self, info: _EnumInfo, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> None:
        aliases = self._resolve_enum_aliases(dispatch, call_symbol)
        used_names = set(info.member_names.values())
        changed = False
        for case in sorted(dispatch.cases, key=lambda item: item.key):
            if case.key in info.member_names:
                continue
            name = aliases.get(case.key)
            if not name:
                name = self._format_enum_member_name(case.key)
            name = self._ensure_unique_member_name(name, used_names)
            used_names.add(name)
            info.member_names[case.key] = name
            changed = True
        if changed:
            members = tuple(
                ASTEnumMember(name=name, value=value)
                for value, name in sorted(info.member_names.items())
            )
            info.decl.members = members

    def _dispatch_helper_key(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> str | None:
        symbol = dispatch.helper_symbol or call_symbol
        if symbol:
            return f"symbol:{symbol}"
        helper = dispatch.helper
        if helper is not None:
            return f"helper:0x{helper:04X}"
        return None

    def _allocate_enum_name(
        self, dispatch: IRSwitchDispatch, key: str, call_symbol: str | None
    ) -> str:
        base = self._normalise_enum_name(
            dispatch.helper_symbol or call_symbol, dispatch.helper
        )
        name = base
        existing = self._enum_name_usage.get(name)
        if existing is not None and existing != key:
            helper = dispatch.helper
            if helper is not None:
                name = f"{base}_{helper:04X}"
            else:
                suffix = 2
                candidate = f"{base}_{suffix}"
                while candidate in self._enum_name_usage:
                    suffix += 1
                    candidate = f"{base}_{suffix}"
                name = candidate
        self._enum_name_usage[name] = key
        return name

    def _normalise_enum_name(
        self, symbol: str | None, helper: int | None
    ) -> str:
        if symbol:
            lowered = symbol.lower()
            if lowered.startswith("scheduler.mask_"):
                base = "SchedulerMask"
            elif lowered.startswith("io.") and lowered.endswith("write"):
                parts = [
                    piece.capitalize()
                    for part in symbol.split(".")
                    for piece in part.split("_")
                    if piece
                ]
                base = "".join(parts) or "Dispatch"
                if not base.endswith("Dispatch"):
                    base += "Dispatch"
            else:
                parts = [
                    piece.capitalize()
                    for part in symbol.replace("/", ".").split(".")
                    for piece in part.split("_")
                    if piece
                ]
                base = "".join(parts) or "Dispatch"
            return base
        if helper is not None:
            return f"Helper{helper:04X}"
        return "Helper"

    def _resolve_enum_aliases(
        self, dispatch: IRSwitchDispatch, call_symbol: str | None
    ) -> Mapping[int, str]:
        symbol = (dispatch.helper_symbol or call_symbol or "").lower()
        if "mask" in symbol:
            return self._build_mask_aliases(dispatch.cases)
        return {}

    def _build_mask_aliases(
        self, cases: Sequence[IRDispatchCase]
    ) -> Mapping[int, str]:
        aliases: Dict[int, str] = {}
        for case in cases:
            value = case.key
            if value == 0:
                aliases[value] = "MaskClear"
                continue
            if value & (value - 1) == 0:
                bit_index = value.bit_length() - 1
                aliases[value] = f"MaskBit{bit_index:02d}"
            else:
                aliases[value] = f"MaskValue_{value:04X}"
        return aliases

    @staticmethod
    def _format_enum_member_name(value: int) -> str:
        return f"K_{value:04X}"

    @staticmethod
    def _ensure_unique_member_name(name: str, used: Set[str]) -> str:
        if name not in used:
            return name
        index = 2
        while True:
            candidate = f"{name}_{index}"
            if candidate not in used:
                return candidate
            index += 1

    def _convert_node(
        self,
        node,
        origin_offset: int,
        value_state: MutableMapping[str, ASTExpression],
        metrics: ASTMetrics,
    ) -> Tuple[List[ASTStatement], List[_BranchLink]]:
        if isinstance(node, IRCallPreparation):
            for mnemonic, operand in node.steps:
                self._pending_call_frame.append(IRStackEffect(mnemonic=mnemonic, operand=operand))
            return [], []
        if isinstance(node, IRLoad):
            target = ASTIdentifier(node.target, self._infer_kind(node.target))
            location = self._build_slot_location(node.slot)
            expr = ASTMemoryRead(location=location, value_kind=SSAValueKind.POINTER)
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            metrics.observe_load(True)
            return [ASTAssign(target=target, value=expr)], []
        if isinstance(node, IRStore):
            location = self._build_slot_location(node.slot)
            value_expr = self._resolve_expr(node.value, value_state)
            metrics.observe_store(not isinstance(value_expr, ASTUnknown))
            return [ASTMemoryWrite(location=location, value=value_expr)], []
        if isinstance(node, IRIORead):
            return [ASTIORead(port=node.port)], []
        if isinstance(node, IRIOWrite):
            return [ASTIOWrite(port=node.port, mask=node.mask)], []
        if isinstance(node, IRBankedLoad):
            pointer_expr = (
                self._resolve_expr(node.pointer, value_state) if node.pointer else None
            )
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else None
            )
            location = self._build_banked_location(node, pointer_expr, offset_expr)
            expr = ASTMemoryRead(location=location, value_kind=SSAValueKind.WORD)
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            pointer_known = pointer_expr is None or not isinstance(pointer_expr, ASTUnknown)
            offset_known = offset_expr is None or not isinstance(offset_expr, ASTUnknown)
            metrics.observe_load(pointer_known and offset_known)
            return [
                ASTAssign(
                    target=ASTIdentifier(node.target, self._infer_kind(node.target)),
                    value=expr,
                )
            ], []
        if isinstance(node, IRBankedStore):
            pointer_expr = (
                self._resolve_expr(node.pointer, value_state) if node.pointer else None
            )
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else None
            )
            value_expr = self._resolve_expr(node.value, value_state)
            pointer_known = pointer_expr is None or not isinstance(pointer_expr, ASTUnknown)
            offset_known = offset_expr is None or not isinstance(offset_expr, ASTUnknown)
            value_known = not isinstance(value_expr, ASTUnknown)
            metrics.observe_store(pointer_known and offset_known and value_known)
            location = self._build_banked_location(node, pointer_expr, offset_expr)
            return [ASTMemoryWrite(location=location, value=value_expr)], []
        if isinstance(node, IRIndirectLoad):
            pointer = self._resolve_expr(node.pointer or node.base, value_state)
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else self._build_offset_literal(node.offset)
            )
            location = self._build_indirect_location(node, pointer, offset_expr)
            expr = ASTMemoryRead(
                location=location, value_kind=self._infer_indirect_kind(pointer)
            )
            value_state[node.target] = expr
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            metrics.observe_load(
                not isinstance(pointer, ASTUnknown) and not isinstance(offset_expr, ASTUnknown)
            )
            return [
                ASTAssign(
                    target=ASTIdentifier(node.target, self._infer_kind(node.target)),
                    value=expr,
                )
            ], []
        if isinstance(node, IRIndirectStore):
            pointer = self._resolve_expr(node.pointer or node.base, value_state)
            offset_expr = (
                self._resolve_expr(node.offset_source, value_state)
                if node.offset_source
                else self._build_offset_literal(node.offset)
            )
            value_expr = self._resolve_expr(node.value, value_state)
            metrics.observe_store(
                not any(isinstance(expr, ASTUnknown) for expr in (pointer, offset_expr, value_expr))
            )
            location = self._build_indirect_location(node, pointer, offset_expr)
            return [ASTMemoryWrite(location=location, value=value_expr)], []
        if isinstance(node, IRCallReturn):
            call_expr, arg_exprs = self._convert_call(
                node.target,
                node.args,
                node.symbol,
                node.tail,
                node.varargs,
                value_state,
            )
            self._register_expression(node.describe(), call_expr)
            statements: List[ASTStatement] = []
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for operand in call_expr.operands if operand.is_resolved()),
                len(call_expr.operands),
            )
            abi = self._build_call_abi(node, arg_exprs)
            if node.cleanup:
                self._pending_epilogue.extend(node.cleanup)
            return_identifiers = []
            for index, name in enumerate(node.returns):
                identifier = self._build_return_identifier(name, index)
                value_state[name] = ASTCallResult(call_expr, index)
                metrics.observe_values(int(not isinstance(value_state[name], ASTUnknown)))
                return_identifiers.append(identifier)
            statements.append(
                ASTCallStatement(
                    call=call_expr,
                    returns=tuple(return_identifiers),
                    abi=abi,
                )
            )
            return statements, []
        if isinstance(node, IRTailCall):
            call_expr, arg_exprs = self._convert_call(
                node.call.target,
                node.call.args,
                node.call.symbol,
                True,
                node.varargs,
                value_state,
            )
            self._register_expression(node.call.describe(), call_expr)
            self._register_expression(node.describe(), call_expr)
            statements: List[ASTStatement] = []
            metrics.call_sites += 1
            metrics.observe_call_args(
                sum(1 for operand in call_expr.operands if operand.is_resolved()),
                len(call_expr.operands),
            )
            abi = self._build_call_abi(node, arg_exprs)
            epilogue_effects = self._build_epilogue_effects(node.cleanup, node.abi_effects)
            resolved_returns = tuple(
                self._canonicalise_return_expr(
                    index, self._resolve_expr(name, value_state)
                )
                for index, name in enumerate(node.returns)
            )
            statements.append(
                ASTTailCall(
                    call=call_expr,
                    payload=ASTReturnPayload(values=resolved_returns, varargs=node.varargs),
                    abi=abi,
                    effects=epilogue_effects,
                )
            )
            return statements, []
        if isinstance(node, IRReturn):
            payload = self._build_return_payload(node, value_state)
            effects = self._build_epilogue_effects(node.cleanup, node.abi_effects)
            return [ASTReturn(payload=payload, effects=effects)], []
        if isinstance(node, IRCallCleanup):
            if any(step.mnemonic == "stack_shuffle" for step in node.steps):
                self._pending_call_frame.extend(node.steps)
            else:
                self._pending_epilogue.extend(node.steps)
            return [], []
        if isinstance(node, IRTerminator):
            return [], []
        if isinstance(node, IRIf):
            condition = self._resolve_expr(node.condition, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            branch = ASTBranch(
                condition=condition,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [branch], [
                _BranchLink(
                    statement=branch,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRTestSetBranch):
            expr = self._resolve_expr(node.expr, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            target = ASTIdentifier(node.var, self._infer_kind(node.var))
            value_state[node.var] = target
            metrics.observe_values(int(not isinstance(expr, ASTUnknown)))
            assignment = ASTAssign(target=target, value=expr)
            branch = ASTBranch(
                condition=target,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [assignment, branch], [
                _BranchLink(
                    statement=branch,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFunctionPrologue):
            var_expr = self._resolve_expr(node.var, value_state)
            expr = self._resolve_expr(node.expr, value_state)
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            statement = ASTFunctionPrologue(
                var=var_expr,
                expr=expr,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        if isinstance(node, IRFlagCheck):
            then_target = self._resolve_target(node.then_target)
            else_target = self._resolve_target(node.else_target)
            statement = ASTFlagCheck(
                flag=node.flag,
                then_offset=then_target,
                else_offset=else_target,
            )
            return [statement], [
                _BranchLink(
                    statement=statement,
                    then_target=then_target,
                    else_target=else_target,
                    origin_offset=origin_offset,
                )
            ]
        return [ASTComment(getattr(node, "describe", lambda: repr(node))())], []

    @staticmethod
    def _slot_descriptor(slot: IRSlot) -> ASTAddressDescriptor:
        if slot.space is MemSpace.FRAME:
            space = ASTAddressSpace.FRAME
        elif slot.space is MemSpace.GLOBAL:
            space = ASTAddressSpace.GLOBAL
        else:
            space = ASTAddressSpace.CONST
        return ASTAddressDescriptor(space=space, region=space.value)

    @staticmethod
    def _slot_alias(slot: IRSlot) -> ASTAliasInfo:
        if slot.space is MemSpace.FRAME:
            return ASTAliasInfo(ASTAliasKind.NOALIAS, f"frame_{slot.index:04X}")
        region = "global" if slot.space is MemSpace.GLOBAL else "const"
        return ASTAliasInfo(ASTAliasKind.REGION, f"{region}_{slot.index:04X}")

    def _build_slot_location(self, slot: IRSlot) -> ASTMemoryLocation:
        descriptor = self._slot_descriptor(slot)
        layout = ASTAddressLayout(
            descriptor=descriptor,
            kind=ASTAddressKind.SLOT,
            slot=slot.index,
        )
        return ASTMemoryLocation(layout=layout, alias=self._slot_alias(slot))

    @staticmethod
    def _build_offset_literal(value: int) -> ASTIntegerLiteral:
        magnitude_bits = max(1, abs(value).bit_length())
        bits = ((magnitude_bits + 7) // 8) * 8
        sign = ASTNumericSign.SIGNED if value < 0 else ASTNumericSign.UNSIGNED
        if bits < 8:
            bits = 8
        return ASTIntegerLiteral(value=value, bits=bits, sign=sign)

    @staticmethod
    def _infer_indirect_kind(pointer: ASTExpression | None) -> SSAValueKind:
        if pointer is None:
            return SSAValueKind.UNKNOWN
        pointer_kind = pointer.kind()
        if pointer_kind in {SSAValueKind.BYTE, SSAValueKind.BOOLEAN}:
            return pointer_kind
        return SSAValueKind.WORD

    @staticmethod
    def _memref_descriptor(ref: Optional[MemRef]) -> ASTAddressDescriptor:
        if ref is None:
            return ASTAddressDescriptor(space=ASTAddressSpace.MEMORY, region="mem")
        symbol = ref.symbol
        region = (ref.region or "mem").lower()
        if symbol:
            return ASTAddressDescriptor(
                space=ASTAddressSpace.MEMORY, region=region, symbol=symbol
            )
        if region == "frame":
            return ASTAddressDescriptor(space=ASTAddressSpace.FRAME, region="frame")
        if region.startswith("global"):
            return ASTAddressDescriptor(space=ASTAddressSpace.GLOBAL, region=region)
        if region.startswith("const"):
            return ASTAddressDescriptor(space=ASTAddressSpace.CONST, region=region)
        if region.startswith("io"):
            return ASTAddressDescriptor(space=ASTAddressSpace.IO, region=region)
        return ASTAddressDescriptor(space=ASTAddressSpace.MEMORY, region=region)

    @staticmethod
    def _memref_alias(ref: Optional[MemRef]) -> ASTAliasInfo:
        if ref is None:
            return ASTAliasInfo(ASTAliasKind.UNKNOWN)
        region = (ref.region or "mem").lower()
        if region == "frame":
            return ASTAliasInfo(ASTAliasKind.NOALIAS, "frame")
        if ref.bank is not None and region.startswith("mem"):
            return ASTAliasInfo(ASTAliasKind.REGION, f"bank_{ref.bank:04X}")
        return ASTAliasInfo(ASTAliasKind.REGION, region)

    def _memref_layout(
        self,
        ref: Optional[MemRef],
        kind: ASTAddressKind,
        *,
        slot: Optional[int] = None,
        page_register: Optional[int] = None,
        page_value: Optional[int] = None,
        offset_override: Optional[int] = None,
    ) -> ASTAddressLayout:
        descriptor = self._memref_descriptor(ref)
        bank = ref.bank if ref is not None else None
        page_alias = ref.page_alias if ref is not None else None
        page = ref.page if ref is not None else None
        base = ref.base if ref is not None else None
        offset = ref.offset if ref is not None else None
        if offset_override is not None:
            offset = offset_override
        if page_value is not None:
            page = page_value
            page_alias = page_alias
        layout = ASTAddressLayout(
            descriptor=descriptor,
            kind=kind,
            bank=bank,
            page=page,
            page_alias=page_alias,
            page_register=page_register,
            base=base,
            offset=offset,
            slot=slot,
        )
        return layout

    def _build_banked_location(
        self,
        node: IRBankedLoad | IRBankedStore,
        pointer: Optional[ASTExpression],
        offset: Optional[ASTExpression],
    ) -> ASTMemoryLocation:
        alias = self._memref_alias(node.ref)
        pointer_expr = pointer if pointer is not None and not isinstance(pointer, ASTUnknown) else None
        offset_literal: Optional[int] = None
        index_expr: Optional[ASTExpression] = None
        if offset is not None and not isinstance(offset, ASTUnknown):
            if isinstance(offset, ASTIntegerLiteral):
                offset_literal = offset.value
            else:
                index_expr = offset
        elif node.ref is not None and node.ref.offset is not None:
            offset_literal = node.ref.offset
        layout = self._memref_layout(
            node.ref,
            ASTAddressKind.BANKED,
            page_register=None if node.register_value is not None else node.register,
            page_value=node.register_value,
            offset_override=offset_literal,
        )
        return ASTMemoryLocation(
            layout=layout,
            alias=alias,
            pointer=pointer_expr,
            index=index_expr,
        )

    def _build_indirect_location(
        self,
        node: IRIndirectLoad | IRIndirectStore,
        pointer: ASTExpression,
        offset: ASTExpression,
    ) -> ASTMemoryLocation:
        alias = self._memref_alias(node.ref)
        pointer_expr = pointer if not isinstance(pointer, ASTUnknown) else None
        index_expr: Optional[ASTExpression] = None
        offset_literal: Optional[int] = None
        if isinstance(offset, ASTIntegerLiteral) and not isinstance(offset, ASTUnknown):
            offset_literal = offset.value
        else:
            index_expr = offset
        layout = self._memref_layout(
            node.ref,
            ASTAddressKind.POINTER,
            offset_override=offset_literal,
        )
        return ASTMemoryLocation(
            layout=layout,
            alias=alias,
            pointer=pointer_expr,
            index=index_expr,
        )

    @staticmethod
    def _stack_effect_operand(step: IRStackEffect) -> Optional[int]:
        include_operand = bool(step.operand_role or step.operand_alias)
        if not include_operand:
            include_operand = bool(step.operand) or step.mnemonic not in {"stack_teardown"}
        return step.operand if include_operand else None

    @staticmethod
    def _bitfield(value: int, alias: Optional[str], default_width: int = 16) -> ASTBitField:
        width = default_width
        if value:
            width = max(default_width, ((value.bit_length() + 7) // 8) * 8)
        alias_key = alias.upper() if alias else ""
        if alias_key in {"RET_MASK", "FANOUT_FLAGS", "FANOUT_FLAGS_A", "FANOUT_FLAGS_B"}:
            width = max(width, 16)
        if width == 0:
            width = default_width
        mask = (1 << width) - 1
        return ASTBitField(width=width, value=value & mask, alias=alias)

    @staticmethod
    def _mnemonic_opcode(mnemonic: str) -> Optional[int]:
        if not mnemonic.startswith("op_"):
            return None
        try:
            return int(mnemonic[3:5], 16)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _effect_channel(alias: Optional[str], operand: Optional[int]) -> str:
        if alias:
            return str(alias)
        if operand is not None:
            return f"0x{operand:04X}"
        return "?"

    def _epilogue_step_kind(self, step: IRStackEffect) -> str:
        mnemonic = step.mnemonic
        alias = step.operand_alias
        operand = step.operand

        direct = _DIRECT_EPILOGUE_KIND_MAP.get(mnemonic)
        if direct is not None:
            return direct

        alias_text = str(alias) if alias is not None else None
        if (
            mnemonic in _MASK_STEP_MNEMONICS
            or operand in _MASK_OPERANDS
            or (alias_text is not None and alias_text in _MASK_ALIASES)
        ):
            return "frame.return_mask"

        opcode = self._mnemonic_opcode(mnemonic)
        if opcode in _MASK_OPCODE_FALLBACK:
            return "frame.return_mask"

        if mnemonic in _FRAME_DROP_MNEMONICS or opcode in _DROP_OPCODE_FALLBACK:
            if step.pops:
                return "frame.drop"

        if mnemonic in _IO_BRIDGE_MNEMONICS or opcode in _BRIDGE_OPCODE_FALLBACK:
            return "io.bridge"

        if (
            mnemonic in _IO_STEP_MNEMONICS
            or opcode in _IO_OPCODE_FALLBACK
            or (alias_text is not None and alias_text == "ChatOut")
        ):
            return "io.step"

        if step.pops and step.mnemonic != "stack_teardown":
            return "frame.drop"

        return "frame.effect"

    def _classify_frame_effect_kind(
        self,
        step: IRStackEffect,
        operand: Optional[int],
        alias: Optional[str],
    ) -> str:
        alias_text = alias.lower() if alias else ""
        opcode = self._mnemonic_opcode(step.mnemonic)

        if alias_text:
            if alias_text in _FRAME_CLEAR_ALIASES or alias_text.endswith(".reset"):
                return "frame.reset"
            if any(alias_text.startswith(prefix) for prefix in _FORMAT_PREFIXES):
                return "helpers.format"
            if alias_text.startswith("scheduler."):
                return "frame.scheduler"
            if alias_text.startswith("page."):
                return "frame.page_select"
            if alias_text.startswith("io."):
                return "io.step"

        if operand in _FRAME_OPERAND_KIND_OVERRIDES:
            return _FRAME_OPERAND_KIND_OVERRIDES[operand]

        if operand is None:
            return "frame.write"

        if operand == 0:
            return "frame.reset"

        if opcode in _BRIDGE_OPCODE_FALLBACK:
            return "io.bridge"

        return "frame.write"

    def _effect_from_kind(
        self,
        kind: str,
        operand: Optional[int],
        alias: Optional[str],
        pops: int = 0,
    ) -> Optional[ASTEffect]:
        if kind.startswith("frame."):
            action = kind.split(".", 1)[1]
            if action == "protocol":
                return None
            bitfield: Optional[ASTBitField] = (
                self._bitfield(operand, alias) if operand is not None else None
            )
            if action == "return_mask" and bitfield is not None:
                return ASTFrameMaskEffect(mask=bitfield)
            if action == "write":
                channel_name = alias or (f"0x{operand:04X}" if operand is not None else "frame")
                return ASTFrameWriteEffect(channel=channel_name, mask=bitfield)
            if action == "reset":
                return ASTFrameResetEffect(channel=alias, mask=bitfield)
            if action == "page_select":
                selector = alias or (f"0x{operand:04X}" if operand is not None else "page")
                return ASTFramePageSelectEffect(selector=selector, mask=bitfield)
            if action == "teardown":
                return ASTFrameTeardownEffect(pops=pops)
            if action == "drop":
                return ASTFrameDropEffect(pops=pops or 1)
            label = alias or action
            return ASTFrameCleanupEffect(label=label, mask=bitfield)
        if kind.startswith("helpers."):
            action = kind.split(".", 1)[1]
            op_map = {
                "invoke": ASTHelperOperation.INVOKE,
                "fanout": ASTHelperOperation.FANOUT,
                "mask_low": ASTHelperOperation.MASK_LOW,
                "mask_high": ASTHelperOperation.MASK_HIGH,
                "dispatch": ASTHelperOperation.DISPATCH,
                "reduce": ASTHelperOperation.REDUCE,
                "wrapper": ASTHelperOperation.WRAPPER,
                "format": ASTHelperOperation.FORMAT,
            }
            operation = op_map.get(action, ASTHelperOperation.INVOKE)
            mask = None
            if operation in {ASTHelperOperation.MASK_LOW, ASTHelperOperation.MASK_HIGH} and operand is not None:
                mask = self._bitfield(operand, alias)
            symbol = alias if alias and not mask else None
            return ASTHelperEffect(operation=operation, symbol=symbol, mask=mask)
        if kind.startswith("io."):
            action = kind.split(".", 1)[1]
            op_map = {
                "write": ASTIOOperation.WRITE,
                "read": ASTIOOperation.READ,
                "step": ASTIOOperation.STEP,
                "bridge": ASTIOOperation.BRIDGE,
                "handshake": ASTIOOperation.BRIDGE,
            }
            operation = op_map.get(action, ASTIOOperation.STEP)
            port = self._effect_channel(alias, operand)
            mask = None
            if operation is ASTIOOperation.WRITE and operand is not None:
                mask = self._bitfield(operand, alias)
            return ASTIOEffect(operation=operation, port=port, mask=mask)
        if kind.startswith("abi."):
            action = kind.split(".", 1)[1]
            if action == "return_mask" and operand is not None:
                return ASTFrameMaskEffect(mask=self._bitfield(operand, alias))
            symbol = alias or action
            return ASTHelperEffect(operation=ASTHelperOperation.INVOKE, symbol=symbol)
        return None

    def _effects_from_call_step(self, step: IRStackEffect) -> Tuple[ASTEffect, ...]:
        if step.mnemonic == "stack_shuffle":
            return tuple()
        operand = self._stack_effect_operand(step)
        alias = str(step.operand_alias) if step.operand_alias is not None else None
        kind = self._classify_frame_effect_kind(step, operand, alias)
        effect = self._effect_from_kind(kind, operand, alias, pops=step.pops)
        return (effect,) if effect else tuple()

    def _build_call_abi(
        self,
        node: Any,
        arg_exprs: Sequence[ASTExpression],
    ) -> Optional[ASTCallABI]:
        steps = list(self._pending_call_frame)
        self._pending_call_frame.clear()
        effects: List[ASTEffect] = []
        for step in steps:
            effects.extend(self._effects_from_call_step(step))
        arity = getattr(node, "arity", None)
        slot_count = arity or len(arg_exprs)
        slots: List[ASTCallArgumentSlot] = []
        if slot_count:
            values = list(arg_exprs)
            if len(values) < slot_count:
                deficit = slot_count - len(values)
                values.extend(ASTUnknown(f"slot_{index}") for index in range(len(values), len(values) + deficit))
            for index in range(slot_count):
                slots.append(ASTCallArgumentSlot(index=index, value=values[index]))
                token = f"slot_{index}"
                value = values[index]
                if not isinstance(value, ASTUnknown):
                    self._call_arg_values[token] = value
        live_mask = getattr(node, "cleanup_mask", None)
        return_tokens = getattr(node, "returns", ())
        return_slots: List[ASTCallReturnSlot] = []
        for index, token in enumerate(return_tokens):
            kind = self._infer_kind(token)
            name = self._canonical_placeholder_name(token, index, kind)
            return_slots.append(ASTCallReturnSlot(index=index, kind=kind, name=name))
        abi_effects = getattr(node, "abi_effects", ())
        for spec in abi_effects:
            alias = str(spec.alias) if spec.alias is not None else None
            if spec.kind == "return_mask":
                converted = self._effect_from_kind(
                    "frame.return_mask", spec.operand, alias
                )
            else:
                converted = self._effect_from_kind(spec.kind, spec.operand, alias)
            if converted:
                effects.append(converted)
        if live_mask is not None:
            mask_effect = ASTFrameMaskEffect(mask=self._bitfield(live_mask, None))
            if all(
                not (
                    isinstance(effect, ASTFrameMaskEffect)
                    and effect.mask.value == mask_effect.mask.value
                    and effect.mask.alias == mask_effect.mask.alias
                )
                for effect in effects
            ):
                effects.append(mask_effect)
        tail_allowed = bool(getattr(node, "tail", False) or isinstance(node, IRTailCall))
        if (
            not slots
            and not effects
            and not return_slots
            and not tail_allowed
        ):
            return None
        return ASTCallABI(
            slots=tuple(slots),
            returns=tuple(return_slots),
            effects=tuple(effects),
            tail=tail_allowed,
        )

    def _build_return_identifier(self, name: str, index: int) -> ASTIdentifier:
        kind = self._infer_kind(name)
        canonical = self._canonical_placeholder_name(name, index, kind)
        return ASTIdentifier(canonical, kind)

    def _canonicalise_return_expr(
        self, index: int, expr: ASTExpression
    ) -> ASTExpression:
        if isinstance(expr, ASTIdentifier):
            kind = expr.kind()
            canonical = self._canonical_placeholder_name(expr.name, index, kind)
            if canonical != expr.name:
                return ASTIdentifier(canonical, kind)
        if isinstance(expr, ASTUnknown):
            token = expr.token
            if token and token.startswith("ret"):
                canonical = self._canonical_placeholder_name(
                    token, index, SSAValueKind.UNKNOWN
                )
                return ASTIdentifier(canonical, SSAValueKind.UNKNOWN)
        return expr

    def _build_return_payload(
        self,
        node: IRReturn,
        value_state: Mapping[str, ASTExpression],
    ) -> ASTReturnPayload:
        values = tuple(
            self._canonicalise_return_expr(
                index, self._resolve_expr(name, value_state)
            )
            for index, name in enumerate(node.values)
        )
        return ASTReturnPayload(values=values, varargs=node.varargs)

    def _policy_effects(self, policy: _FramePolicySummary) -> List[ASTEffect]:
        summary: List[ASTEffect] = []
        for value, alias in policy.masks:
            if value is None:
                continue
            summary.append(
                ASTFrameMaskEffect(mask=self._bitfield(value, alias))
            )
        if policy.teardown:
            summary.append(ASTFrameTeardownEffect(pops=policy.teardown))
        if policy.drops:
            summary.append(ASTFrameDropEffect(pops=policy.drops))
        return summary

    def _build_epilogue_effects(
        self,
        cleanup_steps: Sequence[IRStackEffect],
        abi_effects: Sequence[IRAbiEffect],
    ) -> Tuple[ASTEffect, ...]:
        combined = list(self._pending_epilogue)
        combined.extend(cleanup_steps)
        self._pending_epilogue = []

        policy = _FramePolicySummary()
        effects: List[ASTEffect] = []

        for step in combined:
            kind = self._epilogue_step_kind(step)
            operand = self._stack_effect_operand(step)
            alias = str(step.operand_alias) if step.operand_alias is not None else None
            if kind == "frame.teardown":
                policy.teardown += step.pops
                continue
            if kind == "frame.drop":
                policy.drops += step.pops or 1
                continue
            if kind == "frame.return_mask":
                policy.add_mask(operand, alias)
                continue
            if kind == "frame.effect":
                kind = self._classify_frame_effect_kind(step, operand, alias)
            effect = self._effect_from_kind(kind, operand, alias, pops=step.pops)
            if effect:
                effects.append(effect)

        for effect in abi_effects:
            if effect.kind == "return_mask":
                alias = str(effect.alias) if effect.alias is not None else None
                policy.add_mask(effect.operand, alias)
                continue
            converted = self._effect_from_kind(
                f"abi.{effect.kind}", effect.operand, str(effect.alias) if effect.alias else None
            )
            if converted:
                effects.append(converted)

        effects.extend(self._policy_effects(policy))

        if policy.has_effects():
            masks = tuple(
                self._bitfield(value, alias)
                for value, alias in policy.masks
                if value is not None
            )
            effects.append(
                ASTFrameProtocolEffect(
                    masks=masks,
                    teardown=policy.teardown,
                    drops=policy.drops,
                )
            )

        unique: Dict[Tuple[Any, ...], ASTEffect] = {}
        for effect in effects:
            key = self._effect_identity(effect)
            unique[key] = effect
        ordered = sorted(unique.values(), key=lambda eff: eff.order_key())
        return tuple(ordered)

    def _convert_call(
        self,
        target: int,
        args: Sequence[str],
        symbol: Optional[str],
        tail: bool,
        varargs: bool,
        value_state: Mapping[str, ASTExpression],
    ) -> Tuple[ASTCallExpr, Tuple[ASTExpression, ...]]:
        expressions = tuple(self._resolve_expr(arg, value_state) for arg in args)
        operands: List[ASTCallOperand] = []
        for token, expr in zip(args, expressions):
            operand = self._build_call_operand(token, expr)
            operands.append(operand)
            if token and operand.is_resolved() and operand.expression is not None:
                self._call_arg_values[token] = operand.expression
        call_expr = ASTCallExpr(
            target=target,
            operands=tuple(operands),
            symbol=symbol,
            tail=tail,
            varargs=varargs,
        )
        return call_expr, expressions

    @staticmethod
    def _parse_slot_token(token: str) -> Optional[int]:
        if token.startswith("slot(") and token.endswith(")"):
            try:
                return int(token[5:-1], 16)
            except ValueError:
                return None
        return None

    def _build_call_operand(
        self, token: Optional[str], expr: ASTExpression
    ) -> ASTCallOperand:
        if not token:
            return ASTCallOperand(ASTCallOperandKind.VALUE, expression=expr)
        token_stripped = token.strip()
        slot_index = self._parse_slot_token(token_stripped)
        if slot_index is not None:
            return ASTCallOperand(
                ASTCallOperandKind.STACK,
                expression=expr,
                token=token,
                slot_index=slot_index,
            )
        if token_stripped == "stack_top":
            return ASTCallOperand(
                ASTCallOperandKind.STACK,
                expression=expr,
                token=token,
                slot_index=None,
            )
        if ":" in token_stripped and "@0x" in token_stripped:
            return ASTCallOperand(ASTCallOperandKind.TRACE, expression=None, token=token)
        if token_stripped.startswith("lit(") or self._is_hex_literal(token_stripped):
            return ASTCallOperand(ASTCallOperandKind.IMMEDIATE, expression=expr, token=token)
        return ASTCallOperand(ASTCallOperandKind.VALUE, expression=expr, token=token)

    def _resolve_expr(self, token: Optional[str], value_state: Mapping[str, ASTExpression]) -> ASTExpression:
        if not token:
            return ASTUnknown("")
        if token in value_state:
            return value_state[token]
        if token in self._call_arg_values:
            return self._call_arg_values[token]
        if token in self._expression_lookup:
            return self._expression_lookup[token]
        if " & " in token:
            head, mask = token.rsplit(" & ", 1)
            if self._is_hex_literal(mask.strip()):
                return self._resolve_expr(head.strip(), value_state)
        if token.startswith("lit(") and token.endswith(")"):
            literal = token[4:-1]
            try:
                value = int(literal, 16)
            except ValueError:
                return ASTUnknown(token)
            return self._build_offset_literal(value)
        if token.startswith("slot(") and token.endswith(")"):
            try:
                index = int(token[5:-1], 16)
            except ValueError:
                return ASTUnknown(token)
            slot = self._build_slot(index)
            location = self._build_slot_location(slot)
            return ASTMemoryRead(location=location, value_kind=SSAValueKind.POINTER)
        return ASTIdentifier(token, self._infer_kind(token))

    @staticmethod
    def _build_slot(index: int) -> IRSlot:
        if index < 0x1000:
            space = MemSpace.FRAME
        elif index < 0x8000:
            space = MemSpace.GLOBAL
        else:
            space = MemSpace.CONST
        return IRSlot(space=space, index=index)

    def _infer_kind(self, name: str) -> SSAValueKind:
        lowered = name.lower()
        if lowered.startswith("bool"):
            return SSAValueKind.BOOLEAN
        if lowered.startswith("word"):
            return SSAValueKind.WORD
        if lowered.startswith("byte"):
            return SSAValueKind.BYTE
        if lowered.startswith("ptr"):
            return SSAValueKind.POINTER
        if lowered.startswith("page"):
            return SSAValueKind.PAGE_REGISTER
        if lowered.startswith("io"):
            return SSAValueKind.IO
        if lowered.startswith("id"):
            return SSAValueKind.IDENTIFIER
        return SSAValueKind.UNKNOWN

    @staticmethod
    def _ret_prefix(kind: SSAValueKind) -> str:
        mapping = {
            SSAValueKind.BOOLEAN: "flag",
            SSAValueKind.BYTE: "byte",
            SSAValueKind.WORD: "word",
            SSAValueKind.POINTER: "ptr",
            SSAValueKind.IO: "io",
            SSAValueKind.PAGE_REGISTER: "page",
            SSAValueKind.IDENTIFIER: "id",
        }
        return mapping.get(kind, "value")

    def _canonical_placeholder_name(
        self, token: str, index: int, kind: SSAValueKind
    ) -> str:
        if token.startswith("ret"):
            prefix = self._ret_prefix(kind)
            return f"{prefix}{index}"
        return token

    def _describe_branch_target(self, origin_offset: int, target_offset: int) -> str:
        if target_offset in self._current_block_labels:
            return self._current_block_labels[target_offset]
        origin_analysis = self._current_analyses.get(origin_offset)
        if origin_analysis and origin_analysis.fallthrough == target_offset:
            return "fallthrough"
        exit_hint = self._current_exit_hints.get(target_offset)
        if exit_hint:
            return exit_hint
        entry_reasons = self._current_entry_reasons.get(target_offset)
        if entry_reasons:
            joined = ",".join(entry_reasons) or "unspecified"
            return f"entry({joined})"
        return "fallthrough"

    def _format_exit_hint(self, exit_reasons: Tuple[str, ...]) -> str:
        if not exit_reasons:
            return ""
        mapping = {"return": "return", "tail_call": "tail_call"}
        return "|".join(mapping.get(reason, reason) for reason in exit_reasons)

    def _resolve_target(self, target: int) -> int:
        redirects = self._current_redirects
        seen: Set[int] = set()
        while target in redirects and target not in seen:
            seen.add(target)
            target = redirects[target]
        return target

    @staticmethod
    def _is_hex_literal(value: str) -> bool:
        try:
            if value.lower().startswith("0x"):
                int(value, 16)
                return True
        except ValueError:
            return False
        return False

    def _clear_context(self) -> None:
        self._current_analyses = {}
        self._current_entry_reasons = {}
        self._current_block_labels = {}
        self._current_exit_hints = {}
        self._current_redirects = {}
        self._pending_call_frame = []
        self._pending_epilogue = []
        self._segment_declared_enum_keys = []
        self._segment_declared_enum_set = set()
        self._current_segment_index = -1
        self._call_arg_values = {}
        self._expression_lookup = {}


__all__ = ["ASTBuilder"]
