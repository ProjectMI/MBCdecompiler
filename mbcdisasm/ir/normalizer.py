"""Normalisation pipeline that converts raw instructions into IR nodes."""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union, cast

from ..constants import (
    CALL_SHUFFLE_STANDARD,
    IO_PORT_NAME,
    IO_SLOT,
    IO_SLOT_ALIASES,
    PAGE_REGISTER,
    MEMORY_BANK_ALIASES,
    MEMORY_PAGE_ALIASES,
    RET_MASK,
)
from ..analyzer.instruction_profile import InstructionKind, InstructionProfile
from ..analyzer.stack import StackEvent, StackTracker, StackValueType
from ..instruction import read_instructions
from ..knowledge import CallSignature, CallSignatureEffect, CallSignaturePattern, KnowledgeBase
from ..mbc import MbcContainer, Segment
from .model import (
    IRAsciiFinalize,
    IRAsciiHeader,
    IRAsciiPreamble,
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    CallPredicate,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRTailCall,
    IRTailcallReturn,
    IRConditionMask,
    IRFlagCheck,
    IRFunctionPrologue,
    IRLiteral,
    IRLiteralBlock,
    IRLiteralChunk,
    IRPageRegister,
    IRLoad,
    IRNode,
    IRProgram,
    IRRaw,
    IRReturn,
    IRSegment,
    IRSlot,
    MemRef,
    IRStackEffect,
    IRStore,
    IRStackDuplicate,
    IRStackDrop,
    IRStringConstant,
    IRTablePatch,
    IRTableBuilderBegin,
    IRTableBuilderEmit,
    IRTableBuilderCommit,
    IRDispatchCase,
    IRSwitchDispatch,
    IRTailcallFrame,
    IRTestSetBranch,
    IRIf,
    IRIndirectLoad,
    IRIndirectStore,
    IRIORead,
    IRIOWrite,
    MemSpace,
    SSAValueKind,
    NormalizerMetrics,
)


ANNOTATION_MNEMONICS = {"literal_marker"}
RETURN_NIBBLE_MODES = {0x29, 0x2C, 0x32, 0x41, 0x65, 0x69, 0x6C}


CALL_PREPARATION_PREFIXES = {"stack_shuffle", "fanout", "op_59_FE"}
CALL_PREPARATION_MNEMONICS = {
    "op_01_2C",
    "op_02_2A",
    "op_02_F0",
    "op_02_F1",
    "op_09_29",
    "op_0A_F1",
    "op_0F_00",
    "op_0C_2C",
    "op_10_DC",
    "op_11_B4",
    "op_28_10",
    "op_3C_02",
    "op_3D_30",
    "op_4F_01",
    "op_4F_02",
    "op_4D_30",
    "op_5C_08",
    "op_60_04",
    "op_60_08",
    "op_74_08",
    "op_AC_01",
    "op_4B_91",
    "op_72_23",
}
CALL_CLEANUP_MNEMONICS = {
    "call_helpers",
    "fanout",
    "op_01_2C",
    "op_01_2E",
    "op_01_6C",
    "op_05_00",
    "op_10_0E",
    "op_10_12",
    "op_10_5C",
    "op_10_8C",
    "op_10_DC",
    "op_10_E8",
    "op_10_32",
    "op_14_07",
    "op_0C_00",
    "op_0F_00",
    "op_02_F0",
    "op_11_B4",
    "op_32_29",
    "op_4F_01",
    "op_4F_02",
    "op_52_05",
    "op_58_08",
    "op_5E_29",
    "op_64_20",
    "op_65_30",
    "op_6C_01",
    "op_C4_06",
    "op_D0_04",
    "op_D0_06",
    "op_D8_04",
    "op_F0_4B",
    "op_FD_4A",
    "op_05_F0",
    "stack_shuffle",
    "op_4B_91",
    "op_E4_01",
}
CALL_CLEANUP_PREFIXES = ("stack_teardown_", "op_4A_", "op_95_FE")
CALL_PREDICATE_SKIP_MNEMONICS = {
    "op_06_66",
    "op_10_8C",
    "op_29_10",
    "op_64_20",
    "op_65_30",
    "op_70_29",
}

TAILCALL_HELPERS = {
    0x003E,
    0x00ED,
    0x00F0,
    0x013D,
    0x01EC,
    0x01F1,
    0x032C,
    0x0BF0,
    0x0FF0,
    0x16F0,
}

ASCII_HELPER_IDS = {0xF172, 0x7223, 0x3D30}


LITERAL_MARKER_HINTS: Dict[int, str] = {
    0x0067: "literal_hint",
    0x6704: "literal_hint",
    0x0400: "literal_hint",
    0x0110: "literal_hint",
}


def _io_slot_bytes() -> Set[int]:
    values: Set[int] = set()
    for alias in IO_SLOT_ALIASES:
        values.add(alias & 0xFF)
        values.add((alias >> 8) & 0xFF)
    return values


def _io_prefixes() -> Set[int]:
    prefixes = {value & 0xF0 for value in _io_slot_bytes() if value & 0xF0}
    prefixes.update({0x40, 0xF0})
    return prefixes


def _io_low_nibbles() -> Set[int]:
    nibbles = {value & 0x0F for value in _io_slot_bytes()}
    nibbles.update({0x0, 0x4, 0x8, 0x9, 0xC})
    return nibbles


def _io_mask_values() -> Set[int]:
    masks: Set[int] = set()
    for prefix in _io_prefixes():
        for nibble in _io_low_nibbles():
            value = prefix | nibble
            if value:
                masks.add(value & 0xFF)
    masks.discard((RET_MASK >> 8) & 0xFF)
    port_code = IO_SLOT & 0xFF
    for value in _io_slot_bytes():
        if value != port_code:
            masks.discard(value & 0xFF)
    masks = {value for value in masks if (value & 0xF0) != 0x30}
    return masks


def _mnemonics_for_pairs(
    lhs: Set[int], rhs: Set[int], *, mirror: bool = False
) -> Set[str]:
    mnemonics: Set[str] = set()
    for left in lhs:
        for right in rhs:
            mnemonics.add(f"op_{left:02X}_{right:02X}")
            if mirror:
                mnemonics.add(f"op_{right:02X}_{left:02X}")
    return mnemonics


_IO_PORT_CODES = {IO_SLOT & 0xFF}
_IO_MASK_VALUES = _io_mask_values()
_IO_READ_PORT_CODES = {min(_IO_PORT_CODES) & 0xFF, (min(_IO_PORT_CODES) + 1) & 0xFF}
_IO_READ_MASK_PREFIXES = {prefix for prefix in _io_prefixes() if prefix in {0x20, 0x30}}
_IO_READ_MASKS = {prefix | 0x08 for prefix in _IO_READ_MASK_PREFIXES}

IO_READ_MNEMONICS = _mnemonics_for_pairs(_IO_READ_PORT_CODES, _IO_READ_MASKS)
IO_FACADE_WRITE_MNEMONICS = {"op_13_00", "op_10_E4"}
IO_WRITE_MNEMONICS = _mnemonics_for_pairs(_IO_PORT_CODES, _IO_MASK_VALUES, mirror=True)
IO_WRITE_MNEMONICS.update(IO_FACADE_WRITE_MNEMONICS)
IO_ACCEPTED_OPERANDS = {0} | set(IO_SLOT_ALIASES)

_IO_BRIDGE_FAMILIES = {
    0x3D: {0x01, 0xF1},
    0x00: {0x38, 0x4C},
    0x08: {0x5C},
    0x1C: {0x0C},
    0x1B: {0xF0, 0xA4},
    0x10: {0x61},
    0xE8: {0xF0},
}

IO_BRIDGE_MNEMONICS = set().union(
    *(
        _mnemonics_for_pairs({opcode}, {mode}, mirror=True)
        for mode, opcodes in _IO_BRIDGE_FAMILIES.items()
        for opcode in opcodes
    )
)
IO_BRIDGE_NODE_TYPES = (
    IRCall,
    IRCallCleanup,
    IRCallPreparation,
    IRCallReturn,
    IRReturn,
    IRTailCall,
    IRTailcallReturn,
    IRSwitchDispatch,
    IRTailcallFrame,
)

EPILOGUE_ALLOWED_NODE_TYPES = (
    IRCallCleanup,
    IRCallReturn,
    IRCallPreparation,
    IRConditionMask,
    IRLiteral,
    IRLiteralChunk,
    IRStringConstant,
    IRAsciiPreamble,
    IRAsciiFinalize,
    IRAsciiHeader,
)

IO_HELPER_MNEMONICS = {"call_helpers", "op_F0_4B", "op_4A_10"}
CALL_HELPER_FACADE_MNEMONICS = {"op_05_F0", "op_FD_4A"}
FANOUT_FACADE_MNEMONICS = {"op_10_32"}

ASCII_NEIGHBOR_NODE_TYPES = (
    IRLiteralChunk,
    IRAsciiPreamble,
    IRAsciiFinalize,
    IRAsciiHeader,
)

STACK_NEUTRAL_CONTROL_KINDS = {
    InstructionKind.CONTROL,
    InstructionKind.TERMINATOR,
    InstructionKind.BRANCH,
    InstructionKind.RETURN,
    InstructionKind.TEST,
    InstructionKind.CALL,
    InstructionKind.TAILCALL,
}

MASK_OPERAND_ALIASES = {"RET_MASK", "IO_SLOT", "FANOUT_FLAGS"}

SIDE_EFFECT_KIND_HINTS = {
    InstructionKind.INDIRECT,
    InstructionKind.INDIRECT_STORE,
    InstructionKind.TABLE_LOOKUP,
}

SIDE_EFFECT_KEYWORDS = ("io", "port", "memory", "store", "write", "page", "mode", "status", "flag")


INDIRECT_PAGE_REGISTER_MNEMONICS = {
    "op_D4_06": 0x06D4,
    "op_C8_06": 0x06C8,
    "op_BC_06": 0x06BC,
}

INDIRECT_CONFIGURATION_BRIDGES = {"op_3D_30", "op_43_30"}

INDIRECT_MASK_MNEMONICS = {"op_01_DC"}


@dataclass(frozen=True)
class RawInstruction:
    """Wrapper that couples a profile with stack tracking details."""

    profile: InstructionProfile
    event: StackEvent
    annotations: Tuple[str, ...]
    ssa_values: Tuple[str, ...]
    ssa_kinds: Tuple[SSAValueKind, ...]

    @property
    def mnemonic(self) -> str:
        return self.profile.mnemonic

    @property
    def operand(self) -> int:
        return self.profile.word.operand

    @property
    def offset(self) -> int:
        return self.profile.word.offset

    def pushes_value(self) -> bool:
        return bool(self.event.pushed_types)

    def describe_source(self) -> str:
        return f"{self.profile.label}@0x{self.offset:06X}"


@dataclass(frozen=True)
class RawBlock:
    """Sequence of executable instructions terminated by control flow."""

    index: int
    start_offset: int
    instructions: Tuple[RawInstruction, ...]


@dataclass(frozen=True)
class InstructionTemplate:
    """Simple pattern used to classify raw instructions."""

    name: str
    mnemonics: Tuple[str, ...] = tuple()
    prefixes: Tuple[str, ...] = tuple()
    kinds: Tuple[InstructionKind, ...] = tuple()
    min_delta: int = -999
    max_delta: int = 999

    def matches(self, instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if self.mnemonics and mnemonic not in self.mnemonics:
            return False
        if self.prefixes and not any(mnemonic.startswith(prefix) for prefix in self.prefixes):
            return False

        kind = instruction.event.kind
        if self.kinds and kind not in self.kinds:
            return False

        delta = instruction.event.delta
        if delta < self.min_delta or delta > self.max_delta:
            return False
        return True


CALL_PREPARATION_TEMPLATES: Tuple[InstructionTemplate, ...] = (
    InstructionTemplate(
        name="prep_teardown",
        kinds=(InstructionKind.STACK_TEARDOWN,),
        max_delta=-1,
    ),
    InstructionTemplate(
        name="prep_mnemonics",
        mnemonics=tuple(CALL_PREPARATION_MNEMONICS),
    ),
    InstructionTemplate(
        name="prep_prefix",
        prefixes=tuple(CALL_PREPARATION_PREFIXES),
    ),
)


CALL_CLEANUP_TEMPLATES: Tuple[InstructionTemplate, ...] = (
    InstructionTemplate(
        name="cleanup_teardown",
        kinds=(InstructionKind.STACK_TEARDOWN,),
        max_delta=-1,
    ),
    InstructionTemplate(
        name="cleanup_mnemonics",
        mnemonics=tuple(CALL_CLEANUP_MNEMONICS),
    ),
    InstructionTemplate(
        name="cleanup_prefix",
        prefixes=tuple(CALL_CLEANUP_PREFIXES),
    ),
)


CallLike = Union[IRCall, IRCallReturn, IRTailcallReturn, IRTailCall]


class _ItemList:
    """Mutable wrapper around a block during normalisation passes."""

    def __init__(self, items: Sequence[Union[RawInstruction, IRNode]]):
        self._items: List[Union[RawInstruction, IRNode]] = list(items)

    def __iter__(self) -> Iterator[Union[RawInstruction, IRNode]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Union[RawInstruction, IRNode]:
        return self._items[index]

    def replace_slice(
        self, start: int, end: int, replacement: Sequence[Union[RawInstruction, IRNode]]
    ) -> None:
        self._items[start:end] = list(replacement)

    def insert(self, index: int, value: Union[RawInstruction, IRNode]) -> None:
        self._items.insert(index, value)

    def pop(self, index: int) -> Union[RawInstruction, IRNode]:
        return self._items.pop(index)

    def to_tuple(self) -> Tuple[Union[RawInstruction, IRNode], ...]:
        return tuple(self._items)


@dataclass
class _TableBuilderState:
    """Book-keeping for an active table builder sequence."""

    begin_index: int
    parameters: List[str]


class IRNormalizer:
    """Drive the multi-pass IR normalisation pipeline."""

    _SSA_PREFIX = {
        SSAValueKind.UNKNOWN: "ssa",
        SSAValueKind.BYTE: "byte",
        SSAValueKind.WORD: "word",
        SSAValueKind.POINTER: "ptr",
        SSAValueKind.IO: "io",
        SSAValueKind.PAGE_REGISTER: "page",
        SSAValueKind.BOOLEAN: "bool",
        SSAValueKind.IDENTIFIER: "id",
    }
    _OPCODE_TABLE_MIN_RUN = 4
    _OPCODE_TABLE_MAX_AFFIX = 2
    _OPCODE_TABLE_MODES = {
        0x00,
        0x01,
        0x02,
        0x2A,
        0x2B,
        0x32,
        0x33,
        0x42,
        0x43,
        0x44,
        0x45,
        0x46,
        0x47,
        0x48,
        0x4E,
        0x4F,
        0x50,
        0x51,
    }
    _OPCODE_TABLE_BODY_OPERANDS = {0x0000, 0x0008}
    _OPCODE_TABLE_AFFIX_MNEMONICS = {"op_08_00"}
    _OPCODE_TABLE_AFFIX_OPERANDS = {
        "reduce_pair": {0x0000},
        "op_04_02": {0x0000},
        "op_08_03": {0x0000},
        "op_35_45": {0x0000},
        "op_03_46": {0x0000},
        "op_2C_46": {0x0000},
        "op_2B_47": {0x0000},
    }
    _OPCODE_TABLE_NONTRIVIAL_AFFIX = {
        "reduce_pair",
        "op_04_02",
        "op_08_03",
        "op_35_45",
        "op_03_46",
        "op_2C_46",
        "op_2B_47",
    }
    _TABLE_PATCH_EXTRA_MNEMONICS = {
        "fanout": None,
        "stack_teardown_4": None,
        "stack_teardown_5": None,
        "reduce_pair": {0x0000},
        "op_04_02": {0x0000},
        "op_08_03": {0x0000},
        "op_35_45": {0x0000},
        "op_03_46": {0x0000},
        "op_2C_46": {0x0000},
        "op_2B_47": {0x0000},
    }
    _ADAPTIVE_TABLE_MIN_RUN = 8
    _ADAPTIVE_TABLE_KINDS = {
        InstructionKind.UNKNOWN,
        InstructionKind.TABLE_LOOKUP,
        InstructionKind.META,
    }

    _SSA_PRIORITY = {
        SSAValueKind.UNKNOWN: 0,
        SSAValueKind.BYTE: 1,
        SSAValueKind.WORD: 2,
        SSAValueKind.IDENTIFIER: 2,
        SSAValueKind.POINTER: 3,
        SSAValueKind.IO: 4,
        SSAValueKind.PAGE_REGISTER: 5,
        SSAValueKind.BOOLEAN: 6,
    }

    _AGGREGATE_PARALLEL_THRESHOLD = 16

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._annotation_offsets: Set[int] = set()
        self._ssa_bindings: Dict[int, Tuple[str, ...]] = {}
        self._ssa_types: Dict[str, SSAValueKind] = {}
        self._ssa_aliases: Dict[str, str] = {}
        self._ssa_counters: Dict[str, int] = defaultdict(int)
        self._memref_regions: Dict[int, str] = {}
        self._memref_symbols: Dict[Tuple[str, Optional[int], Optional[int], Optional[int]], str] = {}
        self._memref_symbol_counters: Dict[str, int] = defaultdict(int)
        self._pending_tail_targets: Dict[int, List[int]] = defaultdict(list)
        self._string_pool: Dict[bytes, IRStringConstant] = {}
        self._string_pool_order: List[IRStringConstant] = []

    @staticmethod
    def _aggregate_parallel_workers() -> int:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, 32))

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def normalise_container(
        self,
        container: MbcContainer,
        *,
        segment_indices: Optional[Sequence[int]] = None,
    ) -> IRProgram:
        segments: List[IRSegment] = []
        aggregate_metrics = NormalizerMetrics()
        selection = set(segment_indices or [])
        self._string_pool.clear()
        self._string_pool_order.clear()

        for segment in container.segments():
            if selection and segment.index not in selection:
                continue
            normalised = self.normalise_segment(segment)
            segments.append(normalised)
            aggregate_metrics.observe(normalised.metrics)

        return IRProgram(
            segments=tuple(segments),
            metrics=aggregate_metrics,
            string_pool=tuple(self._string_pool_order),
        )

    def normalise_segment(self, segment: Segment) -> IRSegment:
        raw_blocks = self._parse_segment(segment)
        blocks: List[IRBlock] = []
        metrics = NormalizerMetrics()
        self._pending_tail_targets.clear()

        for block in raw_blocks:
            ir_block, block_metrics = self._normalise_block(block)
            blocks.append(ir_block)
            metrics.observe(block_metrics)
            self._update_tail_helper_hints(block)

        return IRSegment(
            index=segment.index,
            start=segment.start,
            length=segment.length,
            blocks=tuple(blocks),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # parsing helpers
    # ------------------------------------------------------------------
    def _parse_segment(self, segment: Segment) -> Tuple[RawBlock, ...]:
        instructions, _ = read_instructions(segment.data, segment.start)
        if not instructions:
            return tuple()

        profiles = [InstructionProfile.from_word(word, self.knowledge) for word in instructions]
        executable: List[InstructionProfile] = []
        annotations: Dict[int, List[str]] = {}
        pending: List[str] = []

        for profile in profiles:
            if profile.mnemonic in ANNOTATION_MNEMONICS or (
                profile.is_literal_marker() and profile.mnemonic.startswith("op_")
            ):
                pending.append(profile.mnemonic)
                continue
            index = len(executable)
            if pending:
                annotations.setdefault(index, []).extend(pending)
                pending.clear()
            executable.append(profile)

        tracker = StackTracker()
        events = tracker.process_sequence(executable)

        raw_instructions: List[RawInstruction] = []
        stack_names: List[str] = []
        next_ssa = 0
        forced_push_kinds = {InstructionKind.REDUCE}
        for index, (profile, event) in enumerate(zip(executable, events)):
            notes = tuple(annotations.get(index, ()))
            pops = len(event.popped_types)
            if pops:
                del stack_names[-pops:]
            pushes = len(event.pushed_types)
            if pushes == 0 and profile.kind in forced_push_kinds:
                baseline = max(event.depth_before - pops, 0)
                inferred = event.depth_after - baseline
                pushes = max(inferred, 1)
            pushed_names: List[str] = []
            pushed_kinds: List[SSAValueKind] = []
            for _ in range(pushes):
                name = f"ssa{next_ssa}"
                next_ssa += 1
                pushed_names.append(name)
            if event.pushed_types:
                for value_type in event.pushed_types[:pushes]:
                    pushed_kinds.append(self._map_stack_type(value_type))
            while len(pushed_kinds) < len(pushed_names):
                pushed_kinds.append(SSAValueKind.UNKNOWN)
            if pushed_names:
                stack_names.extend(pushed_names)
            raw_instructions.append(
                RawInstruction(
                    profile=profile,
                    event=event,
                    annotations=notes,
                    ssa_values=tuple(pushed_names),
                    ssa_kinds=tuple(pushed_kinds),
                )
            )

        blocks: List[RawBlock] = []
        current: List[RawInstruction] = []
        block_index = 0
        block_start = raw_instructions[0].offset if raw_instructions else segment.start

        for instruction in raw_instructions:
            if not current:
                block_start = instruction.offset
            current.append(instruction)
            if self._is_block_terminator(instruction):
                blocks.append(
                    RawBlock(index=block_index, start_offset=block_start, instructions=tuple(current))
                )
                block_index += 1
                current = []

        if current:
            blocks.append(
                RawBlock(index=block_index, start_offset=block_start, instructions=tuple(current))
            )

        return tuple(blocks)

    @staticmethod
    def _is_block_terminator(instruction: RawInstruction) -> bool:
        profile = instruction.profile
        if profile.kind in {
            InstructionKind.BRANCH,
            InstructionKind.RETURN,
            InstructionKind.TERMINATOR,
            InstructionKind.TAILCALL,
        }:
            return True
        control = (profile.control_flow or "").lower()
        return any(token in control for token in {"return", "jump", "stop"})

    # ------------------------------------------------------------------
    # normalisation passes
    # ------------------------------------------------------------------
    def _normalise_block(self, block: RawBlock) -> Tuple[IRBlock, NormalizerMetrics]:
        self._annotation_offsets.clear()
        self._ssa_bindings.clear()
        self._ssa_types.clear()
        self._ssa_aliases.clear()
        self._ssa_counters.clear()
        items = _ItemList(block.instructions)
        metrics = NormalizerMetrics()

        self._pass_literals(items, metrics)
        self._pass_ascii_glue(items, metrics)
        self._pass_ascii_runs(items, metrics)
        self._pass_stack_manipulation(items, metrics)
        self._pass_calls_and_returns(items, metrics)
        self._pass_aggregates(items, metrics)
        self._pass_literal_blocks(items)
        self._pass_literal_block_reducers(items, metrics)
        self._pass_reduce_pair_constants(items, metrics)
        self._pass_ascii_preamble(items)
        self._collapse_call_wrapper_sequences(items)
        self._pass_call_preparation(items)
        self._pass_call_cleanup(items)
        self._pass_io_operations(items)
        self._pass_io_facade(items)
        self._pass_call_conventions(items)
        self._pass_tailcall_frames(items)
        self._pass_opcode_tables(items)
        self._pass_table_patches(items)
        self._pass_table_dispatch(items)
        self._pass_dispatch_wrappers(items)
        self._pass_ascii_finalize(items)
        self._pass_tail_helpers(items)
        self._pass_tailcall_returns(items)
        self._pass_assign_ssa_names(items)
        self._pass_testset_branches(items, metrics)
        self._pass_branches(items, metrics)
        self._pass_table_builders(items)
        self._pass_flag_checks(items)
        self._pass_function_prologues(items)
        self._pass_ascii_headers(items)
        self._pass_call_contracts(items)
        self._pass_condition_masks(items)
        self._pass_call_predicates(items)
        self._pass_prune_testset_duplicates(items)
        self._pass_call_return_templates(items)
        self._pass_tailcall_returns(items)
        self._pass_page_registers(items)
        self._pass_indirect_access(items, metrics)
        self._pass_epilogue_prologue_compaction(items)

        final_items = list(items)
        nodes: List[IRNode] = []
        block_annotations: List[str] = []
        for index, item in enumerate(final_items):
            if isinstance(item, RawInstruction):
                if self._is_annotation_only(item) or self._is_stack_neutral_bridge(
                    item, final_items, index
                ):
                    annotation = self._format_annotation(item)
                    if annotation:
                        block_annotations.append(annotation)
                    continue
                metrics.raw_remaining += 1
                nodes.append(
                    IRRaw(
                        mnemonic=item.mnemonic,
                        operand=item.operand,
                        operand_role=item.profile.operand_role(),
                        operand_alias=item.profile.operand_alias(),
                        annotations=item.annotations,
                    )
                )
            else:
                nodes.append(item)

        ir_block = IRBlock(
            label=f"block_{block.index}",
            start_offset=block.start_offset,
            nodes=tuple(nodes),
            annotations=tuple(block_annotations),
        )
        return ir_block, metrics

    # ------------------------------------------------------------------
    # SSA helpers
    # ------------------------------------------------------------------
    def _record_ssa(
        self,
        item: Union[RawInstruction, IRNode],
        values: Sequence[str],
        *,
        kinds: Optional[Sequence[SSAValueKind]] = None,
    ) -> None:
        if values:
            stored = tuple(values)
            self._ssa_bindings[id(item)] = stored
            if kinds:
                for name, kind in zip(stored, kinds):
                    self._set_ssa_kind(name, kind)
        else:
            self._ssa_bindings.pop(id(item), None)

    def _transfer_ssa(
        self,
        source: Union[RawInstruction, IRNode],
        target: Union[RawInstruction, IRNode],
    ) -> None:
        values: Sequence[str] = ()
        kinds: Optional[Sequence[SSAValueKind]] = None
        if isinstance(source, RawInstruction):
            values = source.ssa_values
            kinds = source.ssa_kinds
        stored = self._ssa_bindings.get(id(source))
        if stored:
            values = stored
            kinds = None
        if values:
            self._record_ssa(target, values, kinds=kinds)
        self._ssa_bindings.pop(id(source), None)

    def _ssa_value(
        self,
        item: Union[RawInstruction, IRNode],
        index: int = -1,
        *,
        raw: bool = False,
    ) -> Optional[str]:
        values = self._ssa_bindings.get(id(item))
        if not values:
            return None
        if index < 0:
            index += len(values)
        if not (0 <= index < len(values)):
            return None
        name = values[index]
        if raw:
            return name
        return self._render_ssa(name)

    def _render_ssa(self, name: str) -> str:
        kind = self._ssa_types.get(name, SSAValueKind.UNKNOWN)
        prefix = self._SSA_PREFIX.get(kind, "ssa")
        alias = self._ssa_aliases.get(name)
        if alias and alias.startswith(prefix):
            return alias
        alias = f"{prefix}{self._ssa_counters[prefix]}"
        self._ssa_counters[prefix] += 1
        self._ssa_aliases[name] = alias
        return alias

    def _set_ssa_kind(self, name: str, kind: SSAValueKind) -> None:
        current = self._ssa_types.get(name)
        if current is None or self._SSA_PRIORITY.get(kind, 0) > self._SSA_PRIORITY.get(current, 0):
            self._ssa_types[name] = kind
            self._ssa_aliases.pop(name, None)

    def _promote_ssa_kind(self, name: Optional[str], kind: SSAValueKind) -> None:
        if not name:
            return
        current = self._ssa_types.get(name)
        if current is None or self._SSA_PRIORITY.get(kind, 0) > self._SSA_PRIORITY.get(current, 0):
            self._ssa_types[name] = kind
            self._ssa_aliases.pop(name, None)

    def _map_stack_type(self, value_type: StackValueType) -> SSAValueKind:
        if value_type is StackValueType.SLOT:
            return SSAValueKind.POINTER
        if value_type is StackValueType.NUMBER:
            return SSAValueKind.WORD
        if value_type is StackValueType.IDENTIFIER:
            return SSAValueKind.IDENTIFIER
        return SSAValueKind.UNKNOWN

    def _stack_sources(
        self, items: _ItemList, index: int, count: int
    ) -> List[Tuple[str, Union[RawInstruction, IRNode]]]:
        sources: List[Tuple[str, Union[RawInstruction, IRNode]]] = []
        scan = index - 1
        while scan >= 0 and len(sources) < count:
            candidate = items[scan]
            values = self._ssa_bindings.get(id(candidate))
            if values:
                for name in reversed(values):
                    sources.append((name, candidate))
                    if len(sources) == count:
                        break
            scan -= 1
        return sources

    # ------------------------------------------------------------------
    # individual pass implementations
    # ------------------------------------------------------------------
    def _pass_literals(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            literal = self._literal_from_instruction(item)
            if literal is None:
                index += 1
                continue

            if isinstance(literal, IRLiteral):
                metrics.literals += 1
            elif isinstance(literal, IRLiteralChunk):
                metrics.literal_chunks += 1

            self._transfer_ssa(item, literal)
            if isinstance(literal, IRLiteral):
                self._annotate_literal_ssa(literal)
            items.replace_slice(index, index + 1, [literal])
            index += 1

    def _annotate_literal_ssa(self, node: IRLiteral) -> None:
        names = self._ssa_bindings.get(id(node))
        if not names:
            return
        kind = self._literal_ssa_kind(node)
        if kind is None:
            return
        for name in names:
            self._promote_ssa_kind(name, kind)

    def _literal_ssa_kind(self, node: IRLiteral) -> Optional[SSAValueKind]:
        value = node.value & 0xFFFF
        if value == PAGE_REGISTER:
            return SSAValueKind.PAGE_REGISTER
        if self._operand_is_io_slot(value):
            return SSAValueKind.IO
        if value <= 0xFF:
            return SSAValueKind.BYTE
        return SSAValueKind.WORD

    def _pass_page_registers(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.operand != PAGE_REGISTER:
                index += 1
                continue

            if item.event.delta > 0:
                index += 1
                continue

            value_name: Optional[str] = None
            value_repr: Optional[str] = None
            literal_value: Optional[int] = None

            sources = self._stack_sources(items, index, 1)
            if sources:
                value_name, source_node = sources[0]
                self._promote_ssa_kind(value_name, SSAValueKind.PAGE_REGISTER)
                if isinstance(source_node, IRLiteral):
                    value_repr = source_node.describe()
                    literal_value = source_node.value & 0xFFFF
                elif isinstance(source_node, IRLiteralChunk):
                    value_repr = source_node.describe()
                else:
                    value_repr = self._render_ssa(value_name)
            else:
                previous = items[index - 1] if index > 0 else None
                if isinstance(previous, IRLiteral):
                    value_repr = previous.describe()
                    literal_value = previous.value & 0xFFFF

            node = IRPageRegister(
                register=item.operand,
                value=value_repr,
                literal=literal_value,
            )
            self._transfer_ssa(item, node)
            items.replace_slice(index, index + 1, [node])
            continue

    def _pass_stack_manipulation(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            mnemonic = item.mnemonic
            if mnemonic == "op_02_66":
                value = self._describe_stack_top(items, index)
                copies = max(1, item.event.depth_after - item.event.depth_before)
                node = IRStackDuplicate(value=value, copies=copies + 1)
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                continue

            if mnemonic == "op_01_66":
                value = self._describe_stack_top(items, index)
                node = IRStackDrop(value=value)
                items.replace_slice(index, index + 1, [node])
                continue

            if mnemonic == "op_03_66":
                slot = self._classify_slot(item.operand)
                node = IRLoad(slot=slot)
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                metrics.loads += 1
                continue

            index += 1

    def _intern_string_constant(self, data: bytes, source: str) -> IRStringConstant:
        constant = self._string_pool.get(data)
        if constant is not None:
            return constant

        if data:
            segments = (data,)
        else:
            segments = (b"",)
        name = f"str_{len(self._string_pool_order):04d}"
        constant = IRStringConstant(name=name, data=data, segments=segments, source=source)
        self._string_pool[data] = constant
        self._string_pool_order.append(constant)
        return constant

    def _make_literal_chunk(
        self, data: bytes, source: str, annotations: Sequence[str]
    ) -> IRLiteralChunk:
        constant = self._intern_string_constant(data, source)
        return IRLiteralChunk(
            data=data,
            source=source,
            annotations=tuple(annotations),
            symbol=constant.name,
        )

    def _literal_from_instruction(self, instruction: RawInstruction) -> Optional[IRNode]:
        profile = instruction.profile

        if profile.is_literal_marker():
            self._annotation_offsets.add(instruction.offset)
            hint = LITERAL_MARKER_HINTS.get(instruction.operand)
            if hint is not None:
                return IRLiteral(
                    value=instruction.operand,
                    mode=profile.mode,
                    source=hint,
                    annotations=instruction.annotations,
                )
            return None

        if profile.kind is InstructionKind.ASCII_CHUNK or profile.mnemonic.startswith(
            "inline_ascii_chunk"
        ):
            data = instruction.profile.word.raw.to_bytes(4, "big")
            return self._make_literal_chunk(data, profile.mnemonic, instruction.annotations)

        if profile.kind is InstructionKind.LITERAL and instruction.pushes_value():
            return IRLiteral(
                value=instruction.operand,
                mode=profile.mode,
                source=profile.mnemonic,
                annotations=instruction.annotations,
            )

        return None

    def _pass_calls_and_returns(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            mnemonic = item.mnemonic
            if mnemonic in {"call_dispatch", "tailcall_dispatch"}:
                args, start = self._collect_call_arguments(items, index)
                target = item.operand
                symbol = self.knowledge.lookup_address(target)
                if mnemonic == "tailcall_dispatch":
                    inline_target = self._extract_tail_dispatch_target(items, index)
                    if inline_target is not None:
                        target = inline_target
                        symbol = self.knowledge.lookup_address(target)
                        if args:
                            args = args[:-1]
                        if not args and start > 0:
                            literal_index = start - 1
                            while literal_index > 0 and isinstance(
                                items[literal_index], RawInstruction
                            ) and items[literal_index].mnemonic == "op_4A_05":
                                literal_index -= 1
                            prior = items[literal_index]
                            if isinstance(prior, IRLiteral):
                                source = getattr(prior, "source", "")
                                if source in {"op_00_52", "push_literal"} or prior.value == 0:
                                    start = literal_index
                call = IRCall(
                    target=target,
                    args=tuple(args),
                    tail=mnemonic == "tailcall_dispatch",
                    symbol=symbol,
                )
                metrics.calls += 1
                if call.tail:
                    metrics.tail_calls += 1
                items.replace_slice(start, index + 1, [call])
                index = start
                if call.tail:
                    self._collapse_tail_return(items, index, metrics)
                continue

            if mnemonic == "return_values":
                count, varargs = self._resolve_return_signature(items, index)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
                if varargs and not values:
                    values = ("ret*",)
                items.replace_slice(index, index + 1, [IRReturn(values=values, varargs=varargs)])
                metrics.returns += 1
                continue

            index += 1

    def _pass_ascii_runs(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            start = index
            data_parts: List[bytes] = []
            annotations: List[str] = []

            while index < len(items):
                candidate = items[index]
                if isinstance(candidate, IRLiteralChunk):
                    data_parts.append(candidate.data)
                    annotations.extend(candidate.annotations)
                    index += 1
                    continue

                break

            if not data_parts:
                index += 1
                continue

            if len(data_parts) <= 1:
                index = start + 1
                continue

            chunk = self._make_literal_chunk(
                b"".join(data_parts), "ascii_run", annotations
            )
            items.replace_slice(start, index, [chunk])
            index = start + 1

    def _pass_ascii_glue(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            if not isinstance(items[index], IRLiteralChunk):
                index += 1
                continue

            start = index
            scan = index
            chunk_nodes: List[IRLiteralChunk] = []

            while scan < len(items) and isinstance(items[scan], IRLiteralChunk):
                chunk = items[scan]
                assert isinstance(chunk, IRLiteralChunk)
                chunk_nodes.append(chunk)
                scan += 1

            reducers: List[RawInstruction] = []

            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and candidate.mnemonic == "reduce_pair":
                    reducers.append(candidate)
                    scan += 1
                    if scan < len(items) and isinstance(items[scan], IRLiteralChunk):
                        next_chunk = items[scan]
                        assert isinstance(next_chunk, IRLiteralChunk)
                        chunk_nodes.append(next_chunk)
                        scan += 1
                        continue
                    break
                break

            if not reducers or len(chunk_nodes) != len(reducers) + 1:
                index += 1
                continue

            removal_end = scan
            removed_items = [items[pos] for pos in range(start, removal_end)]
            source = reducers[-1]

            merged_annotations: List[str] = []
            seen_annotations: set[str] = set()
            for chunk in chunk_nodes:
                for note in chunk.annotations:
                    if note in seen_annotations:
                        continue
                    seen_annotations.add(note)
                    merged_annotations.append(note)

            data = b"".join(chunk.data for chunk in chunk_nodes)
            combined = self._make_literal_chunk(data, "ascii_glue", merged_annotations)
            self._transfer_ssa(source, combined)

            for removed in removed_items:
                if removed is source:
                    continue
                self._ssa_bindings.pop(id(removed), None)

            items.replace_slice(start, removal_end, [combined])
            metrics.reduce_replaced += len(reducers)
            index = start + 1

    def _collect_call_arguments(
        self, items: _ItemList, call_index: int
    ) -> Tuple[List[str], int]:
        args: List[str] = []
        start = call_index
        scan = call_index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                name = self._ssa_value(candidate)
                args.append(name or candidate.describe())
                scan -= 1
                continue
            if isinstance(candidate, IRStackDuplicate):
                args.append(candidate.value)
                scan -= 1
                continue
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    args.append(self._describe_value(candidate))
                    scan -= 1
                    continue
                break
            else:
                break
        args.reverse()
        start = scan + 1
        return args, start

    def _extract_tail_dispatch_target(
        self, items: _ItemList, call_index: int
    ) -> Optional[int]:
        if call_index <= 0:
            return None
        candidate = items[call_index - 1]
        if isinstance(candidate, RawInstruction) and candidate.pushes_value():
            if candidate.mnemonic in {"op_03_00", "push_literal"}:
                return candidate.operand & 0xFFFF
        if isinstance(candidate, IRLiteral):
            return candidate.value & 0xFFFF
        return None

    def _collapse_tail_return(self, items: _ItemList, call_index: int, metrics: NormalizerMetrics) -> None:
        index = call_index + 1
        collected_cleanup: List[IRStackEffect] = []
        while index < len(items):
            item = items[index]
            if isinstance(item, RawInstruction) and item.mnemonic == "return_values":
                count, varargs = self._resolve_return_signature(items, index)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
                if varargs and not values:
                    values = ("ret*",)
                items.replace_slice(
                    index,
                    index + 1,
                    [IRReturn(values=values, varargs=varargs, cleanup=tuple(collected_cleanup))],
                )
                metrics.returns += 1
                return
            if isinstance(item, RawInstruction) and item.profile.kind in {
                InstructionKind.STACK_TEARDOWN,
                InstructionKind.META,
            }:
                if item.profile.kind is InstructionKind.STACK_TEARDOWN:
                    collected_cleanup.append(self._call_cleanup_effect(item))
                items.pop(index)
                continue
            break

    def _resolve_return_signature(self, items: _ItemList, index: int) -> Tuple[int, bool]:
        instruction = items[index]
        assert isinstance(instruction, RawInstruction)

        operand = instruction.operand
        lo = operand & 0xFF
        hi = (operand >> 8) & 0xFF
        mode = instruction.profile.mode
        nibble = lo & 0x0F

        if mode in RETURN_NIBBLE_MODES:
            if nibble:
                return nibble, False
            hint = self._stack_teardown_hint(items, index)
            if hint is not None:
                return hint, False
            base = hi & 0x1F
            if base:
                return base, False
            return 0, True

        if lo:
            if lo > 0x3F:
                narrowed = lo & 0x0F
                if narrowed:
                    return narrowed, False
            return lo, False
        if hi:
            return hi, False

        hint = self._stack_teardown_hint(items, index)
        if hint is not None:
            return hint, False

        return 1, False

    def _stack_teardown_hint(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index - 1
        total = 0
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                kind = candidate.profile.kind
                if kind is InstructionKind.STACK_TEARDOWN:
                    delta = -candidate.event.delta
                    if delta > 0:
                        total += delta
                    scan -= 1
                    continue
                if kind in {InstructionKind.RETURN, InstructionKind.BRANCH, InstructionKind.TERMINATOR}:
                    break
            else:
                break
            scan -= 1
        if total:
            return total
        return None

    def _pass_aggregates(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        executor: Optional[ThreadPoolExecutor] = None
        try:
            while index < len(items):
                literal = self._literal_at(items, index)
                if literal is None:
                    index += 1
                    continue

                literal_nodes: List[IRLiteral] = [literal]
                reducers: List[RawInstruction] = []
                spacers: List[RawInstruction] = []
                scan = index + 1
                added_literal_since_reduce = True

                while scan < len(items):
                    candidate_literal = self._literal_at(items, scan)
                    if candidate_literal is not None:
                        literal_nodes.append(candidate_literal)
                        scan += 1
                        added_literal_since_reduce = True
                        continue

                    candidate = items[scan]
                    if isinstance(candidate, RawInstruction) and self._is_annotation_only(candidate):
                        spacers.append(candidate)
                        scan += 1
                        continue

                    if isinstance(candidate, RawInstruction) and candidate.mnemonic.startswith("reduce"):
                        if not added_literal_since_reduce:
                            self._annotation_offsets.add(candidate.offset)
                            spacers.append(candidate)
                            scan += 1
                            continue
                        reducers.append(candidate)
                        scan += 1
                        added_literal_since_reduce = False
                        continue

                    break

                if not reducers:
                    index += 1
                    continue

                literal_count = len(literal_nodes)
                if literal_count >= self._AGGREGATE_PARALLEL_THRESHOLD:
                    if executor is None:
                        executor = ThreadPoolExecutor(
                            max_workers=self._aggregate_parallel_workers()
                        )
                    literals = list(executor.map(self._literal_repr, literal_nodes))
                else:
                    literals = [self._literal_repr(node) for node in literal_nodes]
                replacement: IRNode
                if len(literals) >= 2 and len(literals) == 2 * len(reducers):
                    entries = []
                    for pos in range(0, len(literals), 2):
                        entries.append((literals[pos], literals[pos + 1]))
                    replacement = IRBuildMap(entries=tuple(entries))
                elif len(literals) == len(reducers) + 1:
                    replacement = IRBuildArray(elements=tuple(literals))
                else:
                    replacement = IRBuildTuple(elements=tuple(literals))

                metrics.aggregates += 1
                metrics.reduce_replaced += len(reducers)
                replacement_sequence: List[Union[RawInstruction, IRNode]] = [replacement]
                if reducers:
                    self._transfer_ssa(reducers[-1], replacement)
                if spacers:
                    replacement_sequence.extend(spacers)
                items.replace_slice(index, scan, replacement_sequence)
                index += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def _pass_literal_blocks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRBuildTuple):
                values = self._parse_literal_list(item.elements)
                block = self._literal_block_from_values(values)
                if block is not None:
                    items.replace_slice(index, index + 1, [block])
                    continue
            if isinstance(item, IRLiteral):
                values, end = self._collect_literal_block_literals(items, index)
                block = self._literal_block_from_values(values)
                if block is not None:
                    items.replace_slice(index, end, [block])
                    continue
            index += 1

    def _literal_block_from_values(
        self,
        values: Sequence[int],
        *,
        reducer: Optional[str] = None,
        operand: Optional[int] = None,
    ) -> Optional[IRLiteralBlock]:
        if not values:
            return None

        normalized = self._normalize_literal_block(values)
        if normalized is None:
            return None

        triplets, tail = normalized
        return IRLiteralBlock(
            triplets=triplets,
            reducer=reducer,
            reducer_operand=operand,
            tail=tail,
        )

    def _pass_literal_block_reducers(
        self, items: _ItemList, metrics: NormalizerMetrics
    ) -> None:
        index = 0
        while index < len(items) - 1:
            first = items[index]
            second = items[index + 1]
            if isinstance(first, RawInstruction) and first.mnemonic.startswith("reduce"):
                if isinstance(second, IRLiteralBlock):
                    if second.reducer is None:
                        updated = IRLiteralBlock(
                            triplets=second.triplets,
                            reducer=first.mnemonic,
                            reducer_operand=first.operand,
                            tail=second.tail,
                        )
                        self._transfer_ssa(first, updated)
                        items.replace_slice(index, index + 2, [updated])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRBuildTuple):
                    values = self._parse_literal_list(second.elements)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
                        self._transfer_ssa(first, block)
                        items.replace_slice(index, index + 2, [block])
                        metrics.reduce_replaced += 1
                        continue
                if isinstance(second, IRLiteral):
                    values, end = self._collect_literal_block_literals(items, index + 1)
                    block = self._literal_block_from_values(
                        values, reducer=first.mnemonic, operand=first.operand
                    )
                    if block is not None:
                        self._transfer_ssa(first, block)
                        items.replace_slice(index, end, [block])
                        metrics.reduce_replaced += 1
                        continue
            index += 1

    def _pass_reduce_pair_constants(
        self, items: _ItemList, metrics: NormalizerMetrics
    ) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not (isinstance(item, RawInstruction) and item.mnemonic == "reduce_pair"):
                index += 1
                continue

            constants: List[str] = []
            removal_start = index
            scan = index - 1
            while scan >= 0 and len(constants) < 2:
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and self._is_annotation_only(candidate):
                    scan -= 1
                    continue

                rendered = self._constant_aggregate_repr(candidate)
                if rendered is None:
                    break

                constants.insert(0, rendered)
                removal_start = scan
                scan -= 1

            if len(constants) != 2:
                index += 1
                continue

            for pos in range(removal_start, index):
                removed = items[pos]
                self._ssa_bindings.pop(id(removed), None)

            tuple_node = IRBuildTuple(elements=tuple(constants))
            self._transfer_ssa(item, tuple_node)
            items.replace_slice(removal_start, index + 1, [tuple_node])
            metrics.aggregates += 1
            metrics.reduce_replaced += 1
            index = removal_start + 1

    def _constant_aggregate_repr(
        self, item: Union[RawInstruction, IRNode]
    ) -> Optional[str]:
        if isinstance(item, IRLiteral):
            return item.describe()
        if isinstance(item, IRLiteralChunk):
            return item.describe()
        if isinstance(item, IRLiteralBlock):
            return item.describe()
        if isinstance(item, IRBuildTuple):
            return item.describe()
        if isinstance(item, IRBuildArray):
            return item.describe()
        if isinstance(item, IRBuildMap):
            return item.describe()
        return None

    def _pass_ascii_preamble(self, items: _ItemList) -> None:
        index = 0
        while index <= len(items) - 3:
            first, second, third = items[index : index + 3]
            if (
                isinstance(first, RawInstruction)
                and isinstance(second, RawInstruction)
                and isinstance(third, RawInstruction)
                and first.mnemonic == "op_72_23"
                and second.mnemonic == "op_31_30"
                and third.mnemonic == "stack_shuffle"
                and third.operand == CALL_SHUFFLE_STANDARD
            ):
                node = IRAsciiPreamble(
                    loader_operand=first.operand,
                    mode_operand=second.operand,
                    shuffle_operand=third.operand,
                )
                items.replace_slice(index, index + 3, [node])
                continue
            index += 1

    def _collapse_call_wrapper_sequences(self, items: _ItemList) -> None:
        self._collapse_call_preparation_sequences(items)

    def _collapse_call_preparation_sequences(self, items: _ItemList) -> None:
        prefixes = tuple(CALL_PREPARATION_PREFIXES)
        index = 0
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and any(item.mnemonic.startswith(prefix) for prefix in prefixes)
            ):
                index += 1
                continue

            start = index
            consumed: List[RawInstruction] = []
            steps: List[Tuple[str, int]] = []
            while index < len(items):
                candidate = items[index]
                if not (
                    isinstance(candidate, RawInstruction)
                    and any(candidate.mnemonic.startswith(prefix) for prefix in prefixes)
                ):
                    break
                consumed.append(candidate)
                steps.append(self._call_preparation_step(candidate))
                index += 1

            if not steps:
                index = start + 1
                continue

            if not any(
                step[0] == "stack_shuffle" or step[0].startswith("op_59_FE")
                for step in steps
            ):
                index = start + 1
                continue

            node = IRCallPreparation(steps=tuple(steps))
            for source in consumed:
                self._transfer_ssa(source, node)
            items.replace_slice(start, index, [node])
            index = start + 1

    def _pass_call_preparation(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRCall):
                index += 1
                continue

            start = index
            steps: List[Tuple[str, int]] = []
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, IRCallPreparation):
                    steps = list(candidate.steps) + steps
                    start = scan
                    break
                if isinstance(candidate, RawInstruction) and self._is_call_preparation_instruction(candidate):
                    steps.insert(0, self._call_preparation_step(candidate))
                    start = scan
                    scan -= 1
                    continue
                break

            if steps:
                items.replace_slice(start, index, [IRCallPreparation(steps=tuple(steps))])
                index = start + 2
                continue

            index += 1

    def _pass_call_cleanup(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or not self._is_call_cleanup_instruction(item):
                index += 1
                continue

            steps: List[IRStackEffect] = []
            start = index
            end = index
            scan = index
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction) and self._is_call_cleanup_instruction(candidate):
                    steps.append(self._call_cleanup_effect(candidate))
                    end = scan + 1
                    scan += 1
                    continue
                break

            if not steps:
                index += 1
                continue

            steps = self._coalesce_epilogue_steps(steps)

            prev_index = start - 1
            while prev_index >= 0 and isinstance(
                items[prev_index], (IRLiteral, IRLiteralChunk, IRStringConstant)
            ):
                prev_index -= 1

            if prev_index >= 0 and isinstance(items[prev_index], IRCall):
                items.replace_slice(start, end, [IRCallCleanup(steps=tuple(steps))])
                index = prev_index + 2
                continue

            next_index = end
            while next_index < len(items) and isinstance(
                items[next_index], (IRLiteral, IRLiteralChunk, IRStringConstant)
            ):
                next_index += 1

            if next_index < len(items) and isinstance(items[next_index], IRReturn):
                return_node = items[next_index]
                assert isinstance(return_node, IRReturn)
                combined = return_node.cleanup + tuple(steps)
                updated = IRReturn(
                    values=return_node.values,
                    varargs=return_node.varargs,
                    cleanup=combined,
                )
                self._transfer_ssa(return_node, updated)
                items.replace_slice(start, end, [])
                next_index -= len(steps)
                items.replace_slice(next_index, next_index + 1, [updated])
                index = next_index + 1
                continue

            items.replace_slice(start, end, [IRCallCleanup(steps=tuple(steps))])
            index = start + 1

    def _pass_io_operations(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and self._is_io_handshake_instruction(item)
            ):
                index += 1
                continue

            candidate_index = self._find_io_candidate(items, index)
            if candidate_index is not None:
                candidate = items[candidate_index]
                assert isinstance(candidate, RawInstruction)

                node = self._build_io_node(items, candidate_index, candidate, allow_prefix=True)
                if node is None:
                    index += 1
                    continue

                self._transfer_ssa(candidate, node)
                self._transfer_ssa(item, node)
                start = min(index, candidate_index)
                end = max(index, candidate_index) + 1
                items.replace_slice(start, end, [node])
                index = start + 1
                continue

            node = self._build_io_node(items, index, item)
            if node is not None:
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                continue

            index += 1

        index = 0
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and (
                    item.mnemonic in IO_READ_MNEMONICS
                    or item.mnemonic in IO_WRITE_MNEMONICS
                    or (
                        item.mnemonic.startswith("op_10_")
                        and self._operand_is_io_slot(item.operand)
                    )
                )
            ):
                index += 1
                continue

            if self._find_io_handshake(items, index) is not None:
                index += 1
                continue

            node = self._build_io_node(items, index, item)
            if node is None:
                index += 1
                continue

            self._transfer_ssa(item, node)
            items.replace_slice(index, index + 1, [node])
            continue

    def _pass_io_facade(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            node = items[index]
            if not isinstance(node, (IRIORead, IRIOWrite)):
                index += 1
                continue

            pre_helpers = list(getattr(node, "pre_helpers", tuple()))
            post_helpers = list(getattr(node, "post_helpers", tuple()))
            changed = False

            prev_index = self._io_facade_neighbor_index(items, index, direction=-1)
            if prev_index is not None:
                helpers, remaining = self._extract_io_helper_steps(items[prev_index], trailing=True)
                if helpers:
                    pre_helpers = list(helpers) + pre_helpers
                    changed = True
                    if remaining:
                        items.replace_slice(prev_index, prev_index + 1, [IRCallCleanup(steps=remaining)])
                    else:
                        items.pop(prev_index)
                        index -= 1

            next_index = self._io_facade_neighbor_index(items, index, direction=1)
            if next_index is not None:
                helpers, remaining = self._extract_io_helper_steps(items[next_index], trailing=False)
                if helpers:
                    post_helpers.extend(helpers)
                    changed = True
                    if remaining:
                        items.replace_slice(next_index, next_index + 1, [IRCallCleanup(steps=remaining)])
                    else:
                        items.pop(next_index)

            if changed:
                updated = replace(
                    node,
                    pre_helpers=tuple(pre_helpers),
                    post_helpers=tuple(post_helpers),
                )
                self._transfer_ssa(node, updated)
                items.replace_slice(index, index + 1, [updated])
            index += 1

    def _extract_io_helper_steps(
        self, cleanup: IRCallCleanup, *, trailing: bool
    ) -> Tuple[Tuple[IRStackEffect, ...], Tuple[IRStackEffect, ...]]:
        steps = list(cleanup.steps)
        extracted: List[IRStackEffect] = []
        if trailing:
            while steps and self._is_io_helper_step(steps[-1]):
                extracted.insert(0, steps.pop())
        else:
            while steps and self._is_io_helper_step(steps[0]):
                extracted.append(steps.pop(0))
        if not extracted:
            return tuple(), tuple(cleanup.steps)
        return tuple(extracted), tuple(steps)

    @staticmethod
    def _is_io_helper_step(step: IRStackEffect) -> bool:
        return step.mnemonic in IO_HELPER_MNEMONICS

    def _io_facade_neighbor_index(
        self, items: _ItemList, index: int, *, direction: int
    ) -> Optional[int]:
        scan = index + direction
        while 0 <= scan < len(items):
            candidate = items[scan]
            if isinstance(candidate, IRCallCleanup):
                return scan
            if isinstance(candidate, (IRLiteral, IRLiteralChunk, IRStringConstant)):
                scan += direction
                continue
            break
        return None

    def _find_io_candidate(self, items: _ItemList, handshake_index: int) -> Optional[int]:
        for direction in (-1, 1):
            scan = handshake_index + direction
            steps = 0
            while 0 <= scan < len(items) and steps < 12:
                node = items[scan]
                if isinstance(node, RawInstruction):
                    if self._is_io_write_candidate(node):
                        return scan
                    if self._is_io_bridge_instruction(node):
                        scan += direction
                        steps += 1
                        continue
                    break
                if isinstance(node, (IRLiteral, IRLiteralChunk)):
                    scan += direction
                    steps += 1
                    continue
                if isinstance(node, IRStringConstant):
                    scan += direction
                    steps += 1
                    continue
                if isinstance(node, IO_BRIDGE_NODE_TYPES):
                    scan += direction
                    steps += 1
                    continue
                if isinstance(node, (IRPageRegister, IRConditionMask)):
                    scan += direction
                    steps += 1
                    continue
                break
        return None

    def _is_io_write_candidate(self, instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic in IO_WRITE_MNEMONICS:
            return True
        return mnemonic.startswith("op_10_")

    def _build_io_node(
        self,
        items: _ItemList,
        index: int,
        instruction: RawInstruction,
        *,
        allow_prefix: bool = False,
    ) -> Optional[IRNode]:
        mnemonic = instruction.mnemonic
        if self._is_io_handshake_instruction(instruction):
            mask = self._io_mask_value(items, index)
            return IRIOWrite(mask=mask, port=IO_PORT_NAME)
        if mnemonic in IO_READ_MNEMONICS:
            return IRIORead(port=IO_PORT_NAME)
        if mnemonic in IO_WRITE_MNEMONICS or mnemonic.startswith("op_10_"):
            if (
                mnemonic not in IO_WRITE_MNEMONICS
                and not self._operand_is_io_slot(instruction.operand)
                and not allow_prefix
            ):
                return None
            mask = self._io_mask_value(items, index)
            if mask is None and instruction.operand not in IO_ACCEPTED_OPERANDS:
                mask = instruction.operand
            return IRIOWrite(mask=mask, port=IO_PORT_NAME)
        return None

    @staticmethod
    def _operand_is_io_slot(operand: int) -> bool:
        return operand in IO_SLOT_ALIASES

    def _io_mask_value(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index - 1
        steps = 0
        while scan >= 0 and steps < 32:
            node = items[scan]
            if isinstance(node, IRLiteral):
                return node.value
            if isinstance(node, RawInstruction):
                if self._is_io_bridge_instruction(node) or self._is_io_handshake_instruction(node):
                    scan -= 1
                    steps += 1
                    continue
                if node.mnemonic.startswith("op_10_"):
                    break
            elif isinstance(node, IRLiteralChunk):
                scan -= 1
                steps += 1
                continue
            elif isinstance(node, IRStringConstant):
                scan -= 1
                steps += 1
                continue
            elif isinstance(node, IO_BRIDGE_NODE_TYPES):
                scan -= 1
                steps += 1
                continue
            elif isinstance(node, (IRStackEffect, IRStackDrop, IRStackDuplicate)):
                scan -= 1
                steps += 1
                continue
            elif isinstance(node, (IRPageRegister, IRConditionMask)):
                scan -= 1
                steps += 1
                continue
            else:
                break
            scan -= 1
            steps += 1
        return None

    def _pass_call_conventions(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRCall):
                index += 1
                continue

            convention = item.convention
            arity = item.arity
            cleanup_mask = item.cleanup_mask
            cleanup_steps = item.cleanup

            consumed: List[int] = []
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, IRCallPreparation):
                    for mnemonic, operand in candidate.steps:
                        if mnemonic == "stack_shuffle":
                            convention = self._call_convention_effect(operand)
                    consumed.append(scan)
                    scan -= 1
                    continue
                if isinstance(candidate, IRLiteral) and arity is None:
                    decoded = self._decode_call_arity(candidate.value)
                    if decoded is not None:
                        arity = decoded
                        consumed.append(scan)
                        scan -= 1
                        continue
                if isinstance(candidate, IRCallCleanup):
                    shuffle_operand = self._extract_call_shuffle(candidate)
                    if shuffle_operand is not None:
                        convention = self._call_convention_effect(shuffle_operand)
                        consumed.append(scan)
                        scan -= 1
                        continue
                break

            post_index = index + 1
            while post_index < len(items) and isinstance(items[post_index], IRLiteralChunk):
                post_index += 1

            cleanup_index: Optional[int] = None
            if post_index < len(items) and isinstance(items[post_index], IRCallCleanup):
                cleanup_index = post_index
                cleanup_node = items[cleanup_index]
                cleanup_steps = cleanup_steps + cleanup_node.steps
                if cleanup_mask is None:
                    cleanup_mask = self._extract_cleanup_mask(cleanup_node.steps)

            if cleanup_index is not None:
                items.pop(cleanup_index)
                if cleanup_index < index:
                    index -= 1

            for position in sorted(consumed, reverse=True):
                items.pop(position)
                if position < index:
                    index -= 1

            call = items[index]
            assert isinstance(call, IRCall)

            tail = call.tail
            if not tail and cleanup_mask == RET_MASK:
                tail = True

            updated = IRCall(
                target=call.target,
                args=call.args,
                tail=tail,
                arity=arity,
                convention=convention,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup_steps,
                symbol=call.symbol,
            )
            self._transfer_ssa(call, updated)
            items.replace_slice(index, index + 1, [updated])
            index += 1

    def _pass_tailcall_frames(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction) or item.mnemonic != "op_3D_30":
                index += 1
                continue

            steps: List[Tuple[str, int]] = [(item.mnemonic, item.operand)]
            scan = index + 1
            end = index + 1
            while scan < len(items) and scan - index <= 6:
                candidate = items[scan]
                if not isinstance(candidate, RawInstruction):
                    break
                steps.append((candidate.mnemonic, candidate.operand))
                if candidate.mnemonic == "op_F0_4B":
                    end = scan + 1
                    break
                scan += 1

            if steps[-1][0] == "op_F0_4B":
                items.replace_slice(index, end, [IRTailcallFrame(steps=tuple(steps))])
                continue

            index += 1

    def _pass_opcode_tables(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if self._is_opcode_table_seed(item):
                assert isinstance(item, RawInstruction)
                mode = item.profile.mode
                start = index
                scan = index
                body_ops: List[Tuple[str, int]] = []
                body_annotations: List[str] = []

                while scan < len(items):
                    candidate = items[scan]
                    if self._is_opcode_table_body(candidate, mode):
                        assert isinstance(candidate, RawInstruction)
                        body_ops.append((candidate.mnemonic, candidate.operand))
                        body_annotations.extend(candidate.annotations)
                        scan += 1
                        continue
                    break

                if len(body_ops) < self._OPCODE_TABLE_MIN_RUN:
                    index += 1
                    continue

                prefix = start
                prefix_ops: List[Tuple[str, int]] = []
                prefix_annotations: List[str] = []
                affix = 0
                while prefix > 0 and affix < self._OPCODE_TABLE_MAX_AFFIX:
                    candidate = items[prefix - 1]
                    if not self._is_opcode_table_affix(candidate, mode):
                        break
                    assert isinstance(candidate, RawInstruction)
                    prefix -= 1
                    prefix_ops.insert(0, (candidate.mnemonic, candidate.operand))
                    prefix_annotations.extend(candidate.annotations)
                    affix += 1

                suffix = scan
                suffix_ops: List[Tuple[str, int]] = []
                suffix_annotations: List[str] = []
                affix = 0
                while suffix < len(items) and affix < self._OPCODE_TABLE_MAX_AFFIX:
                    candidate = items[suffix]
                    if not self._is_opcode_table_affix(candidate, mode):
                        break
                    assert isinstance(candidate, RawInstruction)
                    suffix_ops.append((candidate.mnemonic, candidate.operand))
                    suffix_annotations.extend(candidate.annotations)
                    suffix += 1
                    affix += 1

                operations = prefix_ops + body_ops + suffix_ops
                annotations = ["opcode_table", f"mode=0x{mode:02X}"]
                seen_annotations = set(annotations)
                for pool in (prefix_annotations, body_annotations, suffix_annotations):
                    for note in pool:
                        if not note or note in seen_annotations:
                            continue
                        annotations.append(note)
                        seen_annotations.add(note)

                for position in range(prefix, suffix):
                    removed = items[position]
                    self._ssa_bindings.pop(id(removed), None)

                items.replace_slice(
                    prefix,
                    suffix,
                    [
                        IRTablePatch(
                            operations=tuple(operations),
                            annotations=tuple(annotations),
                        )
                    ],
                )
                index = prefix + 1
                continue

            adaptive = self._match_adaptive_table_run(items, index)
            if adaptive is not None:
                start, end, operations, annotations = adaptive
                for position in range(start, end):
                    removed = items[position]
                    self._ssa_bindings.pop(id(removed), None)
                items.replace_slice(
                    start,
                    end,
                    [
                        IRTablePatch(
                            operations=tuple(operations),
                            annotations=annotations,
                        )
                    ],
                )
                index = start + 1
                continue

            index += 1

    def _is_opcode_table_seed(self, item: Union[RawInstruction, IRNode]) -> bool:
        if not isinstance(item, RawInstruction):
            return False
        mode = item.profile.mode
        if mode not in self._OPCODE_TABLE_MODES:
            return False
        return self._is_opcode_table_body(item, mode)

    def _is_opcode_table_body(
        self, item: Union[RawInstruction, IRNode], mode: int
    ) -> bool:
        if not isinstance(item, RawInstruction):
            return False
        if item.profile.mode != mode:
            return False
        if item.operand not in self._OPCODE_TABLE_BODY_OPERANDS:
            return False
        if self._is_annotation_only(item):
            return False
        return self._has_trivial_stack_effect(item)

    def _is_opcode_table_affix(
        self, item: Union[RawInstruction, IRNode], mode: int
    ) -> bool:
        if not isinstance(item, RawInstruction):
            return False
        if item.profile.mode != mode:
            allowed_operands = self._OPCODE_TABLE_AFFIX_OPERANDS.get(item.mnemonic)
            if allowed_operands is not None:
                if item.operand not in allowed_operands:
                    return False
            else:
                if item.mnemonic not in self._OPCODE_TABLE_AFFIX_MNEMONICS:
                    return False
                if item.operand not in self._OPCODE_TABLE_BODY_OPERANDS:
                    return False
        if self._is_annotation_only(item):
            return False
        if item.mnemonic in self._OPCODE_TABLE_NONTRIVIAL_AFFIX:
            return True
        return self._has_trivial_stack_effect(item)

    def _match_adaptive_table_run(
        self, items: _ItemList, index: int
    ) -> Optional[Tuple[int, int, Tuple[Tuple[str, int], ...], Tuple[str, ...]]]:
        seed = items[index]
        if not self._is_adaptive_table_seed(seed):
            return None
        assert isinstance(seed, RawInstruction)
        mode = seed.profile.mode
        kind = seed.profile.kind
        start = index
        scan = index
        operations: List[Tuple[str, int]] = []
        annotations = ["adaptive_table", f"mode=0x{mode:02X}", f"kind={kind.name.lower()}"]
        seen_annotations = set(annotations)

        while scan < len(items):
            candidate = items[scan]
            if not self._is_adaptive_table_body(candidate, mode, kind):
                break
            assert isinstance(candidate, RawInstruction)
            operations.append((candidate.mnemonic, candidate.operand))
            for note in candidate.annotations:
                if not note or note in seen_annotations:
                    continue
                annotations.append(note)
                seen_annotations.add(note)
            scan += 1

        if len(operations) < self._ADAPTIVE_TABLE_MIN_RUN:
            return None

        return start, scan, tuple(operations), tuple(annotations)

    def _is_adaptive_table_seed(self, item: Union[RawInstruction, IRNode]) -> bool:
        if not isinstance(item, RawInstruction):
            return False
        if item.profile.mode in self._OPCODE_TABLE_MODES:
            return False
        if item.profile.kind not in self._ADAPTIVE_TABLE_KINDS:
            return False
        return self._is_adaptive_table_body(
            item, item.profile.mode, item.profile.kind
        )

    def _is_adaptive_table_body(
        self, item: Union[RawInstruction, IRNode], mode: int, kind: InstructionKind
    ) -> bool:
        if not isinstance(item, RawInstruction):
            return False
        if item.profile.mode != mode:
            return False
        if item.profile.kind != kind:
            return False
        if not item.profile.mnemonic.startswith("op_"):
            return False
        if self._is_annotation_only(item):
            return False
        return self._has_trivial_stack_effect(item)

    @staticmethod
    def _has_trivial_stack_effect(instruction: RawInstruction) -> bool:
        event = instruction.event
        if event.delta != 0:
            return False
        if event.depth_before != event.depth_after:
            return False
        if event.minimum != event.depth_before:
            return False
        if event.maximum != event.depth_before:
            return False
        if event.popped_types or event.pushed_types:
            return False
        return True

    def _pass_table_patches(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not (
                isinstance(item, RawInstruction)
                and item.mnemonic.startswith("op_2C_")
                and 0x6600 <= item.operand <= 0x66FF
            ):
                index += 1
                continue

            operations: List[Tuple[str, int]] = [(item.mnemonic, item.operand)]
            scan = index + 1
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction):
                    if (
                        candidate.mnemonic.startswith("op_2C_")
                        and 0x6600 <= candidate.operand <= 0x66FF
                    ):
                        operations.append((candidate.mnemonic, candidate.operand))
                        scan += 1
                        continue
                    extra = self._TABLE_PATCH_EXTRA_MNEMONICS.get(candidate.mnemonic)
                    if extra is not None:
                        if extra and candidate.operand not in extra:
                            break
                        operations.append((candidate.mnemonic, candidate.operand))
                        scan += 1
                        continue
                break

            items.replace_slice(index, scan, [IRTablePatch(operations=tuple(operations))])
            index += 1

    def _pass_table_dispatch(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, IRTablePatch):
                index += 1
                continue

            helper_target: Optional[int] = None
            helper_symbol: Optional[str] = None
            follow = index + 1
            while follow < len(items) and isinstance(items[follow], IRLiteralChunk):
                follow += 1
            if follow < len(items):
                candidate = items[follow]
                if isinstance(candidate, CallLike):
                    target = getattr(candidate, "target", None)
                    if isinstance(target, int) and target in {0x6623, 0x6624}:
                        helper_target = target
                        helper_symbol = self.knowledge.lookup_address(target)

            cases, default = self._extract_dispatch_cases(item.operations)
            if not cases:
                index += 1
                continue

            dispatch = IRSwitchDispatch(
                cases=tuple(sorted(cases, key=lambda entry: entry.key)),
                helper=helper_target,
                helper_symbol=helper_symbol,
                default=default,
            )
            items.replace_slice(index, index + 1, [dispatch])
            index += 1

    def _pass_dispatch_wrappers(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            node = items[index]
            if not isinstance(node, IRSwitchDispatch):
                index += 1
                continue

            prefix_start = index
            while prefix_start > 0:
                candidate = items[prefix_start - 1]
                if (
                    isinstance(candidate, RawInstruction)
                    and self._is_dispatch_wrapper_instruction(candidate)
                ):
                    prefix_start -= 1
                    continue
                if isinstance(candidate, IRCallCleanup):
                    prefix_start -= 1
                    continue
                break

            if wrapper_positions := [
                pos for pos in range(prefix_start, index) if isinstance(items[pos], (RawInstruction, IRCallCleanup))
            ]:
                steps: List[IRStackEffect] = []
                for position in sorted(wrapper_positions):
                    entry = items[position]
                    if isinstance(entry, RawInstruction):
                        steps.append(self._call_cleanup_effect(entry))
                    elif isinstance(entry, IRCallCleanup):
                        steps.extend(entry.steps)
                if steps:
                    cleanup = IRCallCleanup(steps=tuple(steps))
                    first_index = min(wrapper_positions)
                    wrapper_set = set(wrapper_positions)
                    others = [
                        items[position]
                        for position in range(first_index, index)
                        if position not in wrapper_set
                    ]
                    replacement = [cleanup] + others
                    items.replace_slice(first_index, index, replacement)
                    index = first_index + len(replacement)
                    node = items[index]

            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if (
                    isinstance(candidate, RawInstruction)
                    and self._is_dispatch_wrapper_instruction(candidate)
                ):
                    cleanup = IRCallCleanup(steps=(self._call_cleanup_effect(candidate),))
                    items.replace_slice(scan, scan + 1, [cleanup])
                    scan -= 1
                    continue
                if isinstance(candidate, IRCallCleanup):
                    scan -= 1
                    continue
                if isinstance(candidate, (IRLiteral, IRLiteralChunk, IRStringConstant)):
                    scan -= 1
                    continue
                break

            node = items[index]

            follow = index + 1
            suffix_end = follow
            while suffix_end < len(items):
                candidate = items[suffix_end]
                if (
                    isinstance(candidate, RawInstruction)
                    and self._is_dispatch_wrapper_instruction(candidate)
                ):
                    suffix_end += 1
                    continue
                if isinstance(candidate, IRCallCleanup):
                    suffix_end += 1
                    continue
                break

            if wrapper_positions := [
                pos for pos in range(follow, suffix_end) if isinstance(items[pos], (RawInstruction, IRCallCleanup))
            ]:
                steps: List[IRStackEffect] = []
                for position in sorted(wrapper_positions):
                    entry = items[position]
                    if isinstance(entry, RawInstruction):
                        steps.append(self._call_cleanup_effect(entry))
                    elif isinstance(entry, IRCallCleanup):
                        steps.extend(entry.steps)
                if steps:
                    cleanup = IRCallCleanup(steps=tuple(steps))
                    wrapper_set = set(wrapper_positions)
                    others = [
                        items[position]
                        for position in range(follow, suffix_end)
                        if position not in wrapper_set
                    ]
                    replacement = others + [cleanup]
                    items.replace_slice(follow, suffix_end, replacement)
                    index = follow + len(replacement)
                    continue

            scan = index
            while scan < len(items):
                candidate = items[scan]
                if (
                    isinstance(candidate, RawInstruction)
                    and self._is_dispatch_wrapper_instruction(candidate)
                ):
                    cleanup = IRCallCleanup(steps=(self._call_cleanup_effect(candidate),))
                    items.replace_slice(scan, scan + 1, [cleanup])
                    scan += 1
                    continue
                if isinstance(candidate, IRCallCleanup):
                    scan += 1
                    continue
                if isinstance(candidate, (IRLiteral, IRLiteralChunk, IRStringConstant)):
                    scan += 1
                    continue
                break

            index = follow

    def _pass_table_builders(self, items: _ItemList) -> None:
        state: Optional[_TableBuilderState] = None
        index = 0
        while index < len(items):
            if state is None:
                prologue_span = self._match_table_builder_prologue(items, index)
                if not prologue_span:
                    index += 1
                    continue

                prologue_nodes = [items[index + offset] for offset in range(prologue_span)]
                mode = cast(RawInstruction, prologue_nodes[0]).profile.mode
                prologue_ops = [
                    (cast(RawInstruction, node).mnemonic, cast(RawInstruction, node).operand)
                    for node in prologue_nodes
                ]
                annotations: List[str] = []
                seen: Set[str] = set()
                for node in prologue_nodes:
                    raw = cast(RawInstruction, node)
                    self._ssa_bindings.pop(id(raw), None)
                    for note in raw.annotations:
                        if not note or note in seen:
                            continue
                        annotations.append(note)
                        seen.add(note)

                begin = IRTableBuilderBegin(
                    mode=mode,
                    prologue=tuple(prologue_ops),
                    annotations=tuple(annotations),
                )
                items.replace_slice(index, index + prologue_span, [begin])
                state = _TableBuilderState(begin_index=index, parameters=[])
                index += 1
                continue

            node = items[index]
            commit = self._build_table_builder_commit(node)
            if commit is not None:
                self._ssa_bindings.pop(id(node), None)
                if isinstance(node, IRTestSetBranch) and index + 1 < len(items):
                    follower = items[index + 1]
                    if isinstance(follower, IRIf) and (
                        follower.then_target == commit.then_target
                        and follower.else_target == commit.else_target
                    ):
                        self._ssa_bindings.pop(id(follower), None)
                        items.pop(index + 1)
                items.replace_slice(index, index + 1, [commit])
                state = None
                index += 1
                continue

            if isinstance(node, IRTablePatch):
                emit = self._table_builder_emit(node, state.parameters)
                self._ssa_bindings.pop(id(node), None)
                items.replace_slice(index, index + 1, [emit])
                state.parameters = []
                index += 1
                continue

            if self._is_table_builder_parameter(node):
                descriptor = self._describe_table_builder_parameter(node)
                state.parameters.append(descriptor)
                self._ssa_bindings.pop(id(node), None)
                items.pop(index)
                continue

            if isinstance(node, IRTableBuilderBegin):
                state = _TableBuilderState(begin_index=index, parameters=[])
                index += 1
                continue

            state = None
            index += 1

    def _match_table_builder_prologue(self, items: _ItemList, index: int) -> int:
        if index >= len(items):
            return 0
        node = items[index]
        if not isinstance(node, RawInstruction):
            return 0
        if not self._is_table_builder_prologue(node):
            return 0

        lookahead = min(len(items), index + 12)
        has_table = False
        scan = index + 1
        while scan < lookahead:
            candidate = items[scan]
            if isinstance(candidate, IRTablePatch):
                has_table = True
                break
            if isinstance(candidate, IRTableBuilderBegin):
                break
            if isinstance(candidate, RawInstruction) and candidate.profile.kind is InstructionKind.BRANCH:
                break
            scan += 1

        if not has_table:
            return 0

        span = index + 1
        while span < len(items):
            candidate = items[span]
            if not isinstance(candidate, RawInstruction):
                break
            if not self._is_table_builder_prologue(candidate):
                break
            span += 1
        return span - index

    def _is_table_builder_prologue(self, item: RawInstruction) -> bool:
        if not item.mnemonic.startswith("op_"):
            return False
        if item.operand != 0:
            return False
        if item.profile.kind not in {InstructionKind.UNKNOWN, InstructionKind.META}:
            return False
        if item.profile.mode == 0:
            return False
        return self._has_trivial_stack_effect(item)

    def _is_table_builder_parameter(self, node: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(node, (IRLiteral, IRLiteralChunk, IRLoad))

    def _describe_table_builder_parameter(self, node: IRNode) -> str:
        if isinstance(node, IRLiteralChunk) and node.symbol:
            return f"str({node.symbol})"
        return node.describe()

    def _table_builder_emit(
        self, table: IRTablePatch, parameters: Sequence[str]
    ) -> IRTableBuilderEmit:
        annotations = tuple(table.annotations)
        mode = self._table_builder_mode(table)
        kind = annotations[0] if annotations else "table_patch"
        return IRTableBuilderEmit(
            mode=mode,
            kind=kind,
            operations=table.operations,
            annotations=annotations,
            parameters=tuple(parameters),
        )

    def _table_builder_mode(self, table: IRTablePatch) -> int:
        for note in table.annotations:
            if note.startswith("mode=0x"):
                try:
                    return int(note[6:], 16)
                except ValueError:
                    continue
        if table.operations:
            mnemonic = table.operations[0][0]
            if len(mnemonic) >= 8:
                try:
                    return int(mnemonic[6:8], 16)
                except ValueError:
                    pass
        return 0

    def _build_table_builder_commit(
        self, node: Union[RawInstruction, IRNode]
    ) -> Optional[IRTableBuilderCommit]:
        if isinstance(node, IRTestSetBranch):
            predicate = f"{node.var}={node.expr}"
            if "table" in node.expr:
                return IRTableBuilderCommit(
                    predicate=predicate,
                    then_target=node.then_target,
                    else_target=node.else_target,
                )
        if isinstance(node, IRIf):
            if "table" in node.condition:
                return IRTableBuilderCommit(
                    predicate=node.condition,
                    then_target=node.then_target,
                    else_target=node.else_target,
                )
        return None

    def _extract_dispatch_cases(
        self, operations: Sequence[Tuple[str, int]]
    ) -> Tuple[List[IRDispatchCase], Optional[int]]:
        cases: List[IRDispatchCase] = []
        default_target: Optional[int] = None
        for mnemonic, operand in operations:
            if mnemonic.startswith("op_2C_"):
                suffix = mnemonic.split("_")[-1]
                try:
                    key = int(suffix, 16)
                except ValueError:
                    continue
                target = operand & 0xFFFF
                symbol = self.knowledge.lookup_address(target)
                cases.append(IRDispatchCase(key=key, target=target, symbol=symbol))
                continue
            if mnemonic == "fanout":
                default_target = operand & 0xFFFF
        return cases, default_target

    def _pass_tail_helpers(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, CallLike):
                index += 1
                continue

            helper_target = getattr(item, "target", None)
            if helper_target not in {0x0072, 0x003D, 0x00F0}:
                index += 1
                continue

            new_target = self._extract_tail_helper_target(items, index, helper_target)
            if new_target is None:
                index += 1
                continue

            if helper_target == 0x0072:
                self._rewrite_tail_helper_72(items, index, item, new_target)
                continue

            if helper_target in {0x003D, 0x00F0}:
                self._rewrite_tail_helper_io(items, index, item, helper_target, new_target)
                continue

            index += 1

    def _pass_tailcall_returns(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRTailCall):
                cleanup_steps: List[IRStackEffect] = list(item.cleanup)
                mask = item.cleanup_mask

                pre = index - 1
                while pre >= 0 and isinstance(items[pre], (IRCallCleanup, IRConditionMask)):
                    prefix = items[pre]
                    if isinstance(prefix, IRCallCleanup):
                        cleanup_steps = list(prefix.steps) + cleanup_steps
                        mask = mask or self._extract_cleanup_mask(prefix.steps)
                    else:
                        mask = mask or prefix.mask
                    items.pop(pre)
                    index -= 1
                    pre -= 1

                follow = index + 1
                while follow < len(items) and isinstance(items[follow], (IRCallCleanup, IRConditionMask)):
                    suffix = items[follow]
                    if isinstance(suffix, IRCallCleanup):
                        cleanup_steps.extend(suffix.steps)
                        mask = mask or self._extract_cleanup_mask(suffix.steps)
                    else:
                        mask = mask or suffix.mask
                    items.pop(follow)

                cleanup_steps = self._coalesce_epilogue_steps(cleanup_steps)
                mask = mask or item.cleanup_mask

                updated_call = IRCall(
                    target=item.target,
                    args=item.args,
                    tail=True,
                    arity=item.arity,
                    convention=item.convention,
                    cleanup_mask=mask,
                    cleanup=item.call.cleanup,
                    symbol=item.symbol,
                    predicate=item.predicate,
                )
                updated = IRTailCall(
                    call=updated_call,
                    returns=item.returns,
                    varargs=item.varargs,
                    cleanup=tuple(cleanup_steps),
                    cleanup_mask=mask,
                )
                self._transfer_ssa(item, updated)
                items.replace_slice(index, index + 1, [updated])
                index += 1
                continue

            if not isinstance(item, IRTailcallReturn):
                index += 1
                continue

            cleanup_steps: List[IRStackEffect] = list(item.cleanup)
            mask = item.cleanup_mask

            pre = index - 1
            while pre >= 0 and isinstance(items[pre], (IRCallCleanup, IRConditionMask)):
                prefix = items[pre]
                if isinstance(prefix, IRCallCleanup):
                    cleanup_steps = list(prefix.steps) + cleanup_steps
                    mask = mask or self._extract_cleanup_mask(prefix.steps)
                else:
                    mask = mask or prefix.mask
                items.pop(pre)
                index -= 1
                pre -= 1

            follow = index + 1
            while follow < len(items) and isinstance(items[follow], (IRCallCleanup, IRConditionMask)):
                suffix = items[follow]
                if isinstance(suffix, IRCallCleanup):
                    cleanup_steps.extend(suffix.steps)
                    mask = mask or self._extract_cleanup_mask(suffix.steps)
                else:
                    mask = mask or suffix.mask
                items.pop(follow)

            cleanup_steps = self._coalesce_epilogue_steps(cleanup_steps)
            mask = mask or item.cleanup_mask

            returns = item.returns
            values: Tuple[str, ...]
            if isinstance(returns, tuple):
                values = returns
            else:
                count = max(int(returns), 0)
                values = tuple(f"ret{i}" for i in range(count)) if count else tuple()
            if item.varargs and not values:
                values = ("ret*",)

            call = IRCall(
                target=item.target,
                args=item.args,
                tail=True,
                arity=item.arity,
                convention=item.convention,
                cleanup_mask=mask,
                cleanup=tuple(),
                symbol=item.symbol,
                predicate=item.predicate,
            )

            tail_call = IRTailCall(
                call=call,
                returns=values,
                varargs=item.varargs,
                cleanup=tuple(cleanup_steps),
                cleanup_mask=mask,
            )
            self._transfer_ssa(item, tail_call)
            items.replace_slice(index, index + 1, [tail_call])
            index += 1

    def _extract_tail_helper_target(
        self, items: _ItemList, index: int, helper: int
    ) -> Optional[int]:
        if helper == 0x0072:
            scan = index - 1
            while scan >= 0:
                candidate = items[scan]
                if isinstance(candidate, RawInstruction):
                    if candidate.mnemonic in {"op_08_00", "terminator"}:
                        return candidate.operand & 0xFFFF
                    if candidate.mnemonic in {"op_0B_00", "op_01_63"}:
                        scan -= 1
                        continue
                elif isinstance(candidate, IRLiteral):
                    return candidate.value & 0xFFFF
                elif isinstance(candidate, (IRLiteralChunk, IRAsciiPreamble)):
                    scan -= 1
                    continue
                else:
                    break
                scan -= 1
            hints = self._pending_tail_targets.get(helper)
            if hints:
                return hints.pop(0)
            return None

        if helper == 0x003D:
            scan = index + 1
            while scan < len(items):
                candidate = items[scan]
                if isinstance(candidate, IRCallCleanup):
                    for step in candidate.steps:
                        if step.mnemonic == "call_helpers" and step.operand in ASCII_HELPER_IDS:
                            return step.operand & 0xFFFF
                    break
                if isinstance(candidate, IRAsciiFinalize):
                    return candidate.helper & 0xFFFF
                if isinstance(candidate, RawInstruction):
                    if candidate.mnemonic == "call_helpers" and candidate.operand in ASCII_HELPER_IDS:
                        return candidate.operand & 0xFFFF
                    break
                if isinstance(candidate, IRLiteral):
                    scan += 1
                    continue
                break
            hints = self._pending_tail_targets.get(helper)
            if hints:
                return hints.pop(0)
            return 0x3D30

        return None

    def _rewrite_tail_helper_72(
        self,
        items: _ItemList,
        index: int,
        node: CallLike,
        target: int,
    ) -> None:
        args = list(getattr(node, "args", tuple()))
        convention = getattr(node, "convention", None)
        if convention is not None and convention.mnemonic == "stack_shuffle":
            args = self._apply_call_shuffle(args, convention.operand)

        cleanup_mask = getattr(node, "cleanup_mask", None)
        predicate = getattr(node, "predicate", None)
        arity = getattr(node, "arity", None)

        cleanup_effects = list(getattr(node, "cleanup", tuple()))
        teardown: Optional[IRStackEffect] = None
        retained_cleanup: List[IRStackEffect] = []
        for effect in cleanup_effects:
            if (
                teardown is None
                and effect.mnemonic == "stack_teardown"
                and effect.pops in {0, 1}
            ):
                pops = effect.pops if effect.pops else 1
                teardown = IRStackEffect(
                    mnemonic="stack_teardown",
                    operand=effect.operand,
                    pops=pops,
                    operand_role=effect.operand_role,
                    operand_alias=effect.operand_alias,
                )
                continue
            retained_cleanup.append(effect)

        next_node: Optional[IRNode] = items[index + 1] if index + 1 < len(items) else None
        returns_node: Optional[IRReturn] = None
        if isinstance(next_node, IRReturn):
            returns_node = next_node
            items.pop(index + 1)

        symbol = self.knowledge.lookup_address(target)

        if cleanup_mask == RET_MASK and returns_node is not None:
            cleanup_chain = list(returns_node.cleanup)
            if teardown is not None:
                cleanup_chain.append(teardown)
                teardown = None
            new_return = IRReturn(
                values=returns_node.values,
                varargs=returns_node.varargs,
                cleanup=tuple(cleanup_chain),
            )
            self._transfer_ssa(node, new_return)
            items.replace_slice(index, index + 1, [new_return])
            return

        returns = 0
        varargs = False
        if returns_node is not None:
            returns = len(returns_node.values)
            varargs = returns_node.varargs

        tailcall = IRTailcallReturn(
            target=target,
            args=tuple(args),
            returns=returns,
            varargs=varargs,
            cleanup=tuple(retained_cleanup),
            tail=True,
            arity=arity,
            convention=None,
            cleanup_mask=None,
            symbol=symbol,
            predicate=predicate,
        )
        self._transfer_ssa(node, tailcall)
        items.replace_slice(index, index + 1, [tailcall])

        if teardown is not None:
            self._attach_tail_helper_cleanup(items, index, teardown)

    def _rewrite_tail_helper_io(
        self,
        items: _ItemList,
        index: int,
        node: CallLike,
        helper: int,
        target: int,
    ) -> None:
        handshake_index = self._find_io_handshake(items, index)
        if handshake_index is None:
            symbol = self.knowledge.lookup_address(target)
            replacement: CallLike
            cleanup: Tuple[IRStackEffect, ...] = tuple()
            args = getattr(node, "args", tuple())
            predicate = getattr(node, "predicate", None)
            arity = getattr(node, "arity", None)
            if isinstance(node, IRCall):
                replacement = IRCall(
                    target=target,
                    args=args,
                    tail=getattr(node, "tail", False),
                    arity=arity,
                    convention=None,
                    cleanup=cleanup,
                    cleanup_mask=None,
                    symbol=symbol,
                    predicate=predicate,
                )
            elif isinstance(node, IRCallReturn):
                replacement = IRCallReturn(
                    target=target,
                    args=args,
                    tail=getattr(node, "tail", False),
                    returns=getattr(node, "returns", tuple()),
                    varargs=getattr(node, "varargs", False),
                    cleanup=cleanup,
                    arity=arity,
                    convention=None,
                    cleanup_mask=None,
                    symbol=symbol,
                    predicate=predicate,
                )
            elif isinstance(node, IRCallReturn):
                replacement = IRCallReturn(
                    target=target,
                    args=args,
                    tail=getattr(node, "tail", False),
                    returns=getattr(node, "returns", tuple()),
                    varargs=getattr(node, "varargs", False),
                    cleanup=cleanup,
                    arity=arity,
                    convention=None,
                    cleanup_mask=None,
                    symbol=symbol,
                    predicate=predicate,
                )
            elif isinstance(node, IRTailCall):
                updated_call = IRCall(
                    target=target,
                    args=args,
                    tail=True,
                    arity=arity,
                    convention=None,
                    cleanup_mask=None,
                    cleanup=tuple(),
                    symbol=symbol,
                    predicate=predicate,
                )
                replacement = IRTailCall(
                    call=updated_call,
                    returns=getattr(node, "returns", tuple()),
                    varargs=getattr(node, "varargs", False),
                    cleanup=cleanup,
                    cleanup_mask=None,
                )
            else:
                replacement = IRTailcallReturn(
                    target=target,
                    args=args,
                    returns=getattr(node, "returns", 0),
                    varargs=getattr(node, "varargs", False),
                    cleanup=cleanup,
                    tail=getattr(node, "tail", True),
                    arity=arity,
                    convention=None,
                    cleanup_mask=None,
                    symbol=symbol,
                    predicate=predicate,
                )
            self._transfer_ssa(node, replacement)
            items.replace_slice(index, index + 1, [replacement])
            return

        mask = self._io_mask_value(items, handshake_index)
        if helper == 0x00F0:
            io_node: IRNode = IRIORead(port=IO_PORT_NAME)
        else:
            io_node = IRIOWrite(mask=mask, port=IO_PORT_NAME)
        port_node = items[handshake_index]
        self._transfer_ssa(port_node, io_node)
        self._transfer_ssa(node, io_node)

        slice_end = index + 1
        if slice_end < len(items) and isinstance(items[slice_end], IRReturn):
            slice_end += 1
        items.replace_slice(handshake_index, slice_end, [io_node])

    @staticmethod
    def _apply_call_shuffle(args: Sequence[str], operand: int) -> List[str]:
        if not args:
            return list(args)
        if operand == CALL_SHUFFLE_STANDARD and len(args) >= 2:
            reordered = list(args)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            return reordered
        return list(args)

    def _attach_tail_helper_cleanup(
        self, items: _ItemList, index: int, effect: IRStackEffect
    ) -> None:
        # Prefer attaching to an existing cleanup node to avoid splitting blocks.
        for offset in (1, -1):
            pos = index + offset
            if 0 <= pos < len(items) and isinstance(items[pos], IRCallCleanup):
                cleanup = items[pos]
                assert isinstance(cleanup, IRCallCleanup)
                items.replace_slice(
                    pos,
                    pos + 1,
                    [IRCallCleanup(steps=cleanup.steps + (effect,))],
                )
                return
        items.insert(index + 1, IRCallCleanup(steps=(effect,)))

    def _find_io_handshake(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if self._is_io_handshake_instruction(candidate):
                    return scan
                if candidate.mnemonic.startswith("op_10_"):
                    break
                if self._is_io_bridge_instruction(candidate):
                    scan -= 1
                    continue
            elif isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                scan -= 1
                continue
            elif isinstance(candidate, IO_BRIDGE_NODE_TYPES):
                scan -= 1
                continue
            else:
                break
            scan -= 1
        return None

    def _is_io_handshake_instruction(self, instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        return (
            mnemonic.startswith("op_")
            and mnemonic.endswith("_30")
            and self._operand_is_io_slot(instruction.operand)
        )

    def _is_io_bridge_instruction(self, instruction: RawInstruction) -> bool:
        if instruction.mnemonic in IO_WRITE_MNEMONICS or self._is_io_handshake_instruction(instruction):
            return False
        if instruction.mnemonic in IO_BRIDGE_MNEMONICS:
            return True
        event = instruction.event
        if event.delta != 0:
            return False
        if event.popped_types or event.pushed_types:
            return False
        kind = instruction.profile.kind
        if kind in {
            InstructionKind.BRANCH,
            InstructionKind.RETURN,
            InstructionKind.TERMINATOR,
            InstructionKind.CALL,
            InstructionKind.TAILCALL,
            InstructionKind.TEST,
            InstructionKind.CONTROL,
        }:
            return False
        return True

    def _update_tail_helper_hints(self, block: RawBlock) -> None:
        if not block.instructions:
            return

        for instruction in reversed(block.instructions):
            if instruction.mnemonic == "terminator":
                self._pending_tail_targets[0x0072].append(instruction.operand & 0xFFFF)
                break
            if instruction.mnemonic == "op_08_00":
                self._pending_tail_targets[0x0072].append(instruction.operand & 0xFFFF)
                break
            if instruction.mnemonic == "call_helpers" and instruction.operand in ASCII_HELPER_IDS:
                self._pending_tail_targets[0x003D].append(instruction.operand & 0xFFFF)
                break

    def _pass_ascii_finalize(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            helper_operand: Optional[int] = None
            if isinstance(item, RawInstruction):
                if item.mnemonic == "call_helpers" and item.operand in ASCII_HELPER_IDS:
                    helper_operand = item.operand
            elif isinstance(item, IRCallCleanup):
                ascii_steps = [
                    step
                    for step in item.steps
                    if step.mnemonic == "call_helpers" and step.operand in ASCII_HELPER_IDS
                ]
                if ascii_steps:
                    helper_operand = ascii_steps[0].operand
            if helper_operand is None:
                index += 1
                continue

            summary = ""
            if index > 0:
                previous = items[index - 1]
                if isinstance(previous, IRLiteralChunk):
                    summary = previous.describe()
                elif isinstance(previous, IRLiteral):
                    summary = previous.describe()
                elif isinstance(previous, IRAsciiPreamble):
                    summary = previous.describe()

            if not summary:
                index += 1
                continue

            finalize = IRAsciiFinalize(helper=helper_operand, summary=summary or "ascii")
            self._transfer_ssa(item, finalize)

            if isinstance(item, IRCallCleanup):
                remaining_steps = tuple(
                    step
                    for step in item.steps
                    if not (step.mnemonic == "call_helpers" and step.operand in ASCII_HELPER_IDS)
                )
                if remaining_steps:
                    updated_cleanup = IRCallCleanup(steps=remaining_steps)
                    items.replace_slice(index, index + 1, [finalize, updated_cleanup])
                    index += 2
                else:
                    items.replace_slice(index, index + 1, [finalize])
                    index += 1
                continue

            items.replace_slice(index, index + 1, [finalize])
            index += 1

    def _pass_flag_checks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRIf):
                flag = self._flag_literal_value(item.condition)
                if flag is not None:
                    node = IRFlagCheck(
                        flag=flag,
                        then_target=item.then_target,
                        else_target=item.else_target,
                    )
                    items.replace_slice(index, index + 1, [node])
                    index += 1
                    continue
            index += 1

    def _pass_function_prologues(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRTestSetBranch):
                if index == 0 or all(
                    self._is_prologue_prefix(items[pos]) for pos in range(index)
                ):
                    if item.var.startswith("slot("):
                        node = IRFunctionPrologue(
                            var=item.var,
                            expr=item.expr,
                            then_target=item.then_target,
                            else_target=item.else_target,
                        )
                        self._transfer_ssa(item, node)
                        items.replace_slice(index, index + 1, [node])
                        index += 1
                        continue
            index += 1

    @staticmethod
    def _is_prologue_prefix(item: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(
            item,
            (
                IRAsciiHeader,
                IRLiteralChunk,
                IRLiteral,
                IRCallPreparation,
                IRCallCleanup,
            ),
        )

    def _pass_ascii_headers(self, items: _ItemList) -> None:
        if not items:
            return
        chunk_nodes: List[IRLiteralChunk] = []
        index = 0
        while index < len(items) and isinstance(items[index], IRLiteralChunk):
            chunk_nodes.append(items[index])
            index += 1
        if len(chunk_nodes) >= 2:
            names = [chunk.symbol or chunk.describe() for chunk in chunk_nodes]
            items.replace_slice(0, index, [IRAsciiHeader(chunks=tuple(names))])
            return

        if len(chunk_nodes) == 1:
            literal = chunk_nodes[0]
            symbol = literal.symbol
            if symbol:
                items.replace_slice(0, 1, [IRAsciiHeader(chunks=(symbol,))])
                return
            if isinstance(literal, IRLiteralChunk) and len(literal.data) >= 8:
                if len(literal.data) % 4 == 0:
                    parts = []
                    for pos in range(0, len(literal.data), 4):
                        segment = literal.data[pos : pos + 4]
                        piece = self._make_literal_chunk(
                            segment, literal.source, literal.annotations
                        )
                        parts.append(piece.symbol or piece.describe())
                    if len(parts) >= 2:
                        items.replace_slice(0, 1, [IRAsciiHeader(chunks=tuple(parts))])

    def _pass_call_return_templates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items) - 1:
            call = items[index]
            if isinstance(call, IRCall):
                offset = index + 1
                cleanup_steps: List[IRStackEffect] = list(call.cleanup)
                cleanup_mask = call.cleanup_mask
                base_tail = call.tail
                consumed = 0

                while offset < len(items):
                    candidate = items[offset]
                    if isinstance(candidate, IRCallCleanup):
                        cleanup_steps.extend(candidate.steps)
                        offset += 1
                        consumed += 1
                        continue
                    if isinstance(candidate, IRConditionMask):
                        effect = IRStackEffect(
                            mnemonic=candidate.source,
                            operand=candidate.mask,
                            operand_role="mask",
                        )
                        cleanup_steps.append(effect)
                        cleanup_mask = candidate.mask
                        offset += 1
                        consumed += 1
                        continue
                    if (
                        isinstance(candidate, RawInstruction)
                        and self._is_neutral_cleanup_step(candidate)
                    ):
                        cleanup_steps.append(self._call_cleanup_effect(candidate))
                        if self._uses_ret_mask(candidate):
                            cleanup_mask = RET_MASK
                        offset += 1
                        consumed += 1
                        continue
                    if (
                        isinstance(candidate, RawInstruction)
                        and call.target in TAILCALL_HELPERS
                        and candidate.event.delta <= 0
                        and not candidate.event.pushed_types
                    ):
                        cleanup_steps.append(self._call_cleanup_effect(candidate))
                        if self._uses_ret_mask(candidate):
                            cleanup_mask = RET_MASK
                        offset += 1
                        consumed += 1
                        continue
                    if (
                        call.target in TAILCALL_HELPERS
                        and isinstance(candidate, (IRTestSetBranch, IRIf))
                    ):
                        offset += 1
                        consumed += 1
                        continue
                    break

                tail_hint = base_tail
                if cleanup_mask == RET_MASK:
                    tail_hint = True

                if offset < len(items) and isinstance(items[offset], IRReturn):
                    return_node = items[offset]
                    combined_cleanup = tuple(cleanup_steps + list(return_node.cleanup))
                    varargs = return_node.varargs
                    return_count = len(return_node.values)
                    should_bundle = tail_hint and (
                        base_tail or (return_count == 0 and not varargs)
                    )

                    if should_bundle:
                        node = IRTailcallReturn(
                            target=call.target,
                            args=call.args,
                            returns=return_count,
                            varargs=varargs,
                            cleanup=combined_cleanup,
                            tail=True,
                            arity=call.arity,
                            convention=call.convention,
                            cleanup_mask=cleanup_mask,
                            symbol=call.symbol,
                            predicate=call.predicate,
                        )
                    else:
                        tail = base_tail and consumed == 0 and not return_node.cleanup
                        if not tail and cleanup_mask == RET_MASK and consumed == 0 and not return_node.cleanup:
                            tail = True
                        node = IRCallReturn(
                            target=call.target,
                            args=call.args,
                            tail=tail,
                            returns=return_node.values,
                            varargs=varargs,
                            cleanup=combined_cleanup,
                            arity=call.arity,
                            convention=call.convention,
                            cleanup_mask=cleanup_mask,
                            symbol=call.symbol,
                            predicate=call.predicate,
                        )
                    self._transfer_ssa(call, node)
                    end = offset + 1
                    items.replace_slice(index, end, [node])
                    continue
            index += 1

    def _pass_condition_masks(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRCallCleanup) and len(item.steps) == 1:
                step = item.steps[0]
                if step.mnemonic == "fanout" and step.operand == RET_MASK:
                    node = IRConditionMask(source="fanout", mask=step.operand)
                    items.replace_slice(index, index + 1, [node])
                    continue
            if isinstance(item, RawInstruction):
                if item.operand == RET_MASK and item.mnemonic in {"terminator", "op_29_10"}:
                    node = IRConditionMask(source=item.mnemonic, mask=item.operand)
                    items.replace_slice(index, index + 1, [node])
                    continue
            index += 1

    def _pass_call_predicates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items) - 1:
            call = items[index]
            if not isinstance(call, CallLike):
                index += 1
                continue

            if getattr(call, "predicate", None) is not None:
                index += 1
                continue

            extracted = self._extract_call_predicate(items, index)
            if extracted is None:
                index += 1
                continue

            predicate, branch_index, alias = extracted
            updated = self._call_with_predicate(call, predicate)
            self._transfer_ssa(call, updated)
            items.replace_slice(index, index + 1, [updated])

            branch = items[branch_index]
            if isinstance(branch, IRTestSetBranch) and branch.expr != alias:
                updated_branch = IRTestSetBranch(
                    var=branch.var,
                    expr=alias,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                )
                items.replace_slice(branch_index, branch_index + 1, [updated_branch])
            elif isinstance(branch, IRIf) and branch.condition != alias:
                updated_branch = IRIf(
                    condition=alias,
                    then_target=branch.then_target,
                    else_target=branch.else_target,
                )
                items.replace_slice(branch_index, branch_index + 1, [updated_branch])

            index += 1

    def _pass_prune_testset_duplicates(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, IRTestSetBranch):
                scan = index + 1
                while scan < len(items) and isinstance(items[scan], IRLiteralChunk):
                    scan += 1
                if scan < len(items):
                    candidate = items[scan]
                    if isinstance(candidate, IRIf):
                        if item.var in candidate.condition or item.expr in candidate.condition:
                            items.pop(scan)
                            continue
            index += 1

    def _extract_call_predicate(
        self, items: _ItemList, index: int
    ) -> Optional[Tuple[CallPredicate, int, str]]:
        scan = index + 1
        while scan < len(items):
            candidate = items[scan]
            if isinstance(candidate, (IRCallCleanup, IRConditionMask)):
                scan += 1
                continue
            if isinstance(candidate, RawInstruction) and candidate.mnemonic in CALL_PREDICATE_SKIP_MNEMONICS:
                scan += 1
                continue
            break

        if scan >= len(items):
            return None

        branch = items[scan]
        call = items[index]
        alias = self._ssa_value(call)
        if not alias or alias == "stack_top":
            raw_name = f"call_predicate_{id(call)}"
            self._record_ssa(call, (raw_name,))
            self._promote_ssa_kind(raw_name, SSAValueKind.BOOLEAN)
            alias = self._render_ssa(raw_name)

        if isinstance(branch, IRIf):
            predicate = CallPredicate(
                kind="if",
                expr=alias,
                then_target=branch.then_target,
                else_target=branch.else_target,
            )
            return predicate, scan, alias

        if isinstance(branch, IRTestSetBranch):
            predicate = CallPredicate(
                kind="testset",
                var=branch.var,
                expr=alias,
                then_target=branch.then_target,
                else_target=branch.else_target,
            )
            return predicate, scan, alias

        return None

    def _call_with_predicate(
        self,
        node: CallLike,
        predicate: CallPredicate,
    ) -> CallLike:
        if isinstance(node, IRCall):
            return IRCall(
                target=node.target,
                args=node.args,
                tail=node.tail,
                arity=node.arity,
                convention=node.convention,
                cleanup_mask=node.cleanup_mask,
                cleanup=node.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRCallReturn):
            return IRCallReturn(
                target=node.target,
                args=node.args,
                tail=node.tail,
                returns=node.returns,
                varargs=node.varargs,
                cleanup=node.cleanup,
                arity=node.arity,
                convention=node.convention,
                cleanup_mask=node.cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailCall):
            updated_call = IRCall(
                target=node.target,
                args=node.args,
                tail=True,
                arity=node.arity,
                convention=node.convention,
                cleanup_mask=node.cleanup_mask,
                cleanup=node.call.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )
            return IRTailCall(
                call=updated_call,
                returns=node.returns,
                varargs=node.varargs,
                cleanup=node.cleanup,
                cleanup_mask=node.cleanup_mask,
            )

        if isinstance(node, IRTailcallReturn):
            return IRTailcallReturn(
                target=node.target,
                args=node.args,
                returns=node.returns,
                varargs=node.varargs,
                cleanup=node.cleanup,
                tail=node.tail,
                arity=node.arity,
                convention=node.convention,
                cleanup_mask=node.cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        return node

    def _pass_call_contracts(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            node = items[index]
            if not isinstance(node, CallLike):
                index += 1
                continue

            signature = self.knowledge.call_signature(getattr(node, "target", -1))
            if signature is None:
                index += 1
                continue

            updated_index = self._apply_call_signature(items, index, node, signature)
            index = updated_index + 1

    def _apply_call_signature(
        self,
        items: _ItemList,
        index: int,
        node: CallLike,
        signature: CallSignature,
    ) -> int:
        current_index = index
        target = getattr(node, "target", -1)

        tail = getattr(node, "tail", False)
        arity = getattr(node, "arity", None)
        convention = getattr(node, "convention", None)
        cleanup_mask = getattr(node, "cleanup_mask", None)
        predicate = getattr(node, "predicate", None)
        existing_cleanup = list(getattr(node, "cleanup", tuple()))

        if signature.tail is not None:
            tail = signature.tail or tail
        if signature.arity is not None:
            arity = signature.arity
        if signature.cleanup_mask is not None and cleanup_mask is None:
            cleanup_mask = signature.cleanup_mask
        if signature.shuffle is not None:
            convention = self._call_convention_effect(signature.shuffle)
        elif signature.shuffle_options:
            current_operand = convention.operand if convention is not None else None
            if current_operand not in signature.shuffle_options:
                convention = self._call_convention_effect(signature.shuffle_options[0])

        prefix_effects: List[IRStackEffect] = []
        suffix_effects: List[IRStackEffect] = []

        for pattern in signature.prelude:
            matched, effects, mask, tail_flag, predicate_update, consumed = self._consume_call_pattern_before(
                items, current_index, pattern
            )
            if not matched:
                if pattern.optional:
                    continue
            else:
                prefix_effects = effects + prefix_effects
                if mask is not None:
                    cleanup_mask = mask
                if tail_flag:
                    tail = True
                if predicate_update is not None:
                    predicate = predicate_update
                current_index -= consumed
                continue
            # required pattern not matched: keep scanning without consuming

        node = items[current_index]

        for pattern in signature.postlude:
            matched, effects, mask, tail_flag, predicate_update = self._consume_call_pattern_after(
                items, current_index, pattern
            )
            if not matched:
                if pattern.optional:
                    continue
            else:
                suffix_effects.extend(effects)
                if mask is not None:
                    cleanup_mask = mask
                if tail_flag:
                    tail = True
                if predicate_update is not None:
                    predicate = predicate_update
                continue
            # required pattern not matched: leave remaining nodes untouched

        if cleanup_mask is None:
            cleanup_mask = signature.cleanup_mask

        signature_cleanup = self._convert_signature_effects(signature.cleanup)
        combined_cleanup = tuple(prefix_effects + signature_cleanup + existing_cleanup + suffix_effects)

        updated = self._rebuild_call_node(
            node,
            tail=tail,
            arity=arity,
            convention=convention,
            cleanup_mask=cleanup_mask,
            cleanup=combined_cleanup,
            predicate=predicate,
            target=target,
            signature=signature,
        )
        self._transfer_ssa(node, updated)
        items.replace_slice(current_index, current_index + 1, [updated])
        return current_index

    def _consume_call_pattern_before(
        self,
        items: _ItemList,
        index: int,
        pattern: CallSignaturePattern,
    ) -> Tuple[bool, List[IRStackEffect], Optional[int], bool, Optional[CallPredicate], int]:
        position = index - 1
        if position < 0:
            return False, [], None, False, None, 0

        candidate = items[position]
        if pattern.kind == "raw" and isinstance(candidate, RawInstruction):
            if pattern.mnemonic and candidate.mnemonic != pattern.mnemonic:
                return False, [], None, False, None, 0
            if pattern.operand is not None and candidate.operand != pattern.operand:
                return False, [], None, False, None, 0
            effects: List[IRStackEffect] = []
            if pattern.effect is not None:
                effects.append(self._stack_effect_from_signature(pattern.effect, candidate))
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            predicate: Optional[CallPredicate] = None
            items.pop(position)
            return True, effects, cleanup_mask, tail, predicate, 1

        return False, [], None, False, None, 0

    def _consume_call_pattern_after(
        self,
        items: _ItemList,
        index: int,
        pattern: CallSignaturePattern,
    ) -> Tuple[bool, List[IRStackEffect], Optional[int], bool, Optional[CallPredicate]]:
        if index + 1 >= len(items):
            return False, [], None, False, None

        candidate = items[index + 1]

        if pattern.kind == "raw" and isinstance(candidate, RawInstruction):
            if pattern.mnemonic and candidate.mnemonic != pattern.mnemonic:
                return False, [], None, False, None
            if pattern.operand is not None and candidate.operand != pattern.operand:
                return False, [], None, False, None
            effects: List[IRStackEffect] = []
            if pattern.effect is not None:
                effects.append(self._stack_effect_from_signature(pattern.effect, candidate))
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            predicate: Optional[CallPredicate] = None
            items.pop(index + 1)
            return True, effects, cleanup_mask, tail, predicate

        if pattern.kind == "testset" and isinstance(candidate, IRTestSetBranch):
            predicate = CallPredicate(
                kind="testset",
                var=candidate.var,
                expr=candidate.expr,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        if pattern.kind == "flag" and isinstance(candidate, IRFlagCheck):
            predicate = CallPredicate(
                kind="flag",
                flag=candidate.flag,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        if pattern.kind == "if" and isinstance(candidate, IRIf):
            predicate = CallPredicate(
                kind="if",
                expr=candidate.condition,
                then_target=candidate.then_target,
                else_target=candidate.else_target,
            )
            cleanup_mask = pattern.cleanup_mask
            tail = pattern.tail
            items.pop(index + 1)
            return True, [], cleanup_mask, tail, predicate

        return False, [], None, False, None

    def _convert_signature_effects(
        self, specs: Sequence[CallSignatureEffect]
    ) -> List[IRStackEffect]:
        effects: List[IRStackEffect] = []
        for spec in specs:
            effects.append(self._stack_effect_from_signature(spec, None))
        return effects

    def _stack_effect_from_signature(
        self, spec: CallSignatureEffect, instruction: Optional[RawInstruction]
    ) -> IRStackEffect:
        operand = spec.operand
        operand_role = spec.operand_role
        operand_alias = spec.operand_alias
        if instruction is not None:
            if operand is None or spec.inherit_operand:
                operand = instruction.operand
            if operand_role is None:
                operand_role = instruction.profile.operand_role()
            if spec.inherit_alias or operand_alias is None:
                alias = instruction.profile.operand_alias()
                if alias is not None:
                    operand_alias = alias
        if operand is None:
            operand = 0

        pops = spec.pops
        if instruction is not None and pops == 0:
            if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
                delta = -instruction.event.delta
                if delta > 0:
                    pops = delta

        return IRStackEffect(
            mnemonic=spec.mnemonic,
            operand=operand,
            pops=pops,
            operand_role=operand_role,
            operand_alias=operand_alias,
        )

    def _rebuild_call_node(
        self,
        node: CallLike,
        *,
        tail: bool,
        arity: Optional[int],
        convention: Optional[IRStackEffect],
        cleanup_mask: Optional[int],
        cleanup: Tuple[IRStackEffect, ...],
        predicate: Optional[CallPredicate],
        target: int,
        signature: CallSignature,
    ) -> CallLike:
        if isinstance(node, IRCall):
            return IRCall(
                target=target,
                args=node.args,
                tail=tail,
                arity=arity,
                convention=convention,
                cleanup_mask=cleanup_mask,
                cleanup=cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRCallReturn):
            returns = node.returns
            if signature.returns is not None and not node.varargs:
                count = max(signature.returns, 0)
                returns = tuple(f"ret{i}" for i in range(count))
            return IRCallReturn(
                target=target,
                args=node.args,
                tail=tail,
                returns=returns,
                varargs=node.varargs,
                cleanup=cleanup,
                arity=arity,
                convention=convention,
                cleanup_mask=cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        if isinstance(node, IRTailCall):
            returns = node.returns
            if signature.returns is not None and not node.varargs:
                count = max(signature.returns, 0)
                returns = tuple(f"ret{i}" for i in range(count)) if count else tuple()
            updated_call = IRCall(
                target=target,
                args=node.args,
                tail=True,
                arity=arity,
                convention=convention,
                cleanup_mask=cleanup_mask,
                cleanup=node.call.cleanup,
                symbol=node.symbol,
                predicate=predicate,
            )
            return IRTailCall(
                call=updated_call,
                returns=returns,
                varargs=node.varargs,
                cleanup=cleanup,
                cleanup_mask=cleanup_mask,
            )

        if isinstance(node, IRTailcallReturn):
            returns = node.returns
            if signature.returns is not None and not node.varargs:
                returns = max(signature.returns, 0)
            return IRTailcallReturn(
                target=target,
                args=node.args,
                returns=returns,
                varargs=node.varargs,
                cleanup=cleanup,
                tail=tail,
                arity=arity,
                convention=convention,
                cleanup_mask=cleanup_mask,
                symbol=node.symbol,
                predicate=predicate,
            )

        return node

    @staticmethod
    def _matches_templates(
        instruction: RawInstruction, templates: Sequence[InstructionTemplate]
    ) -> bool:
        return any(template.matches(instruction) for template in templates)

    @staticmethod
    def _uses_ret_mask(instruction: RawInstruction) -> bool:
        alias = instruction.profile.operand_alias()
        if alias == "RET_MASK":
            return True
        return instruction.operand == RET_MASK

    @staticmethod
    def _has_mask_operand(instruction: RawInstruction) -> bool:
        alias = instruction.profile.operand_alias()
        if alias and alias in MASK_OPERAND_ALIASES:
            return True
        operand = instruction.operand
        if operand == RET_MASK or operand in IO_SLOT_ALIASES:
            return True
        role = instruction.profile.operand_role()
        if role:
            lowered = role.lower()
            if "mask" in lowered or "flag" in lowered:
                return True
        return False

    @staticmethod
    def _is_stack_teardown_step(instruction: RawInstruction) -> bool:
        event = instruction.event
        if event.kind is InstructionKind.STACK_TEARDOWN:
            return True
        profile_kind = instruction.profile.kind
        if profile_kind is InstructionKind.STACK_TEARDOWN:
            return True
        return event.delta < 0

    def _is_neutral_cleanup_step(self, instruction: RawInstruction) -> bool:
        event = instruction.event
        if event.delta > 0:
            return False
        if event.pushed_types:
            return False
        if self._is_stack_teardown_step(instruction):
            return True
        return self._has_mask_operand(instruction)

    def _is_call_cleanup_instruction(self, instruction: RawInstruction) -> bool:
        mnemonic = instruction.mnemonic
        if mnemonic.startswith("op_10_") and (
            mnemonic in IO_WRITE_MNEMONICS or mnemonic in IO_READ_MNEMONICS
        ):
            return False
        return self._matches_templates(instruction, CALL_CLEANUP_TEMPLATES)

    def _is_dispatch_wrapper_instruction(self, instruction: RawInstruction) -> bool:
        if self._is_annotation_only(instruction):
            return False

        event = instruction.event
        if event.pushed_types or event.delta > 0:
            return False

        kind = instruction.profile.kind
        if kind in {
            InstructionKind.BRANCH,
            InstructionKind.CALL,
            InstructionKind.RETURN,
            InstructionKind.TAILCALL,
        }:
            return False

        mnemonic = instruction.mnemonic
        if mnemonic.startswith("op_"):
            return True
        if self._is_stack_teardown_step(instruction):
            return True
        return False

    @staticmethod
    def _call_preparation_step(instruction: RawInstruction) -> Tuple[str, int]:
        mnemonic = instruction.mnemonic
        if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
            pops = -instruction.event.delta
            if pops > 0:
                return ("stack_teardown", pops)
        return (mnemonic, instruction.operand)

    def _is_call_preparation_instruction(self, instruction: RawInstruction) -> bool:
        if instruction.mnemonic == "op_4A_05":
            return True
        return self._matches_templates(instruction, CALL_PREPARATION_TEMPLATES)

    def _call_convention_effect(self, operand: int) -> IRStackEffect:
        alias = "CALL_SHUFFLE_STD" if operand == CALL_SHUFFLE_STANDARD else None
        return IRStackEffect(
            mnemonic="stack_shuffle",
            operand=operand,
            operand_alias=alias,
        )

    def _call_cleanup_effect(self, instruction: RawInstruction) -> IRStackEffect:
        mnemonic = instruction.mnemonic
        operand = instruction.operand
        pops = 0
        if mnemonic in CALL_HELPER_FACADE_MNEMONICS:
            mnemonic = "call_helpers"
        elif mnemonic in FANOUT_FACADE_MNEMONICS:
            mnemonic = "fanout"
        if mnemonic == "op_6C_01" and operand == PAGE_REGISTER:
            mnemonic = "page_register"
        if instruction.profile.kind is InstructionKind.STACK_TEARDOWN:
            pops = -instruction.event.delta
            if mnemonic.startswith("stack_teardown"):
                mnemonic = "stack_teardown"
        return IRStackEffect(
            mnemonic=mnemonic,
            operand=operand,
            pops=pops,
            operand_role=instruction.profile.operand_role(),
            operand_alias=instruction.profile.operand_alias(),
        )

    @staticmethod
    def _coalesce_epilogue_steps(steps: Sequence[IRStackEffect]) -> List[IRStackEffect]:
        if not steps:
            return list(steps)

        combined: List[IRStackEffect] = []
        index = 0
        while index < len(steps):
            step = steps[index]
            next_step = steps[index + 1] if index + 1 < len(steps) else None
            if (
                step.mnemonic == "op_52_05"
                and next_step is not None
                and next_step.mnemonic == "op_32_29"
                and step.operand == next_step.operand
            ):
                operand_role = step.operand_role or next_step.operand_role
                operand_alias = step.operand_alias or next_step.operand_alias
                combined.append(
                    IRStackEffect(
                        mnemonic="epilogue",
                        operand=step.operand,
                        pops=step.pops + next_step.pops,
                        operand_role=operand_role,
                        operand_alias=operand_alias,
                    )
                )
                index += 2
                continue
            if (
                step.mnemonic.startswith("op_4A_")
                and next_step is not None
                and next_step.mnemonic == "op_32_29"
                and step.operand == next_step.operand
            ):
                operand_role = step.operand_role or next_step.operand_role
                operand_alias = step.operand_alias or next_step.operand_alias
                combined.append(
                    IRStackEffect(
                        mnemonic="epilogue",
                        operand=step.operand,
                        pops=step.pops + next_step.pops,
                        operand_role=operand_role,
                        operand_alias=operand_alias,
                    )
                )
                index += 2
                continue
            combined.append(step)
            index += 1
        return combined

    @staticmethod
    def _extract_call_shuffle(cleanup: IRCallCleanup) -> Optional[int]:
        if len(cleanup.steps) != 1:
            return None
        step = cleanup.steps[0]
        if step.mnemonic != "stack_shuffle":
            return None
        return step.operand

    @staticmethod
    def _decode_call_arity(value: int) -> Optional[int]:
        if value <= 0:
            return None
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        if high and low == 0:
            return high
        if high:
            return high
        if value <= 0x3F:
            return value
        return None

    @staticmethod
    def _extract_cleanup_mask(steps: Sequence[IRStackEffect]) -> Optional[int]:
        for mnemonic in ("epilogue", "op_52_05", "op_32_29", "fanout"):
            for step in steps:
                if step.mnemonic == mnemonic:
                    return step.operand
        return None

    def _literal_at(self, items: _ItemList, index: int) -> Optional[IRLiteral]:
        item = items[index]
        if isinstance(item, IRLiteral):
            return item
        if isinstance(item, RawInstruction):
            if item.profile.kind is InstructionKind.LITERAL and item.pushes_value():
                literal = IRLiteral(
                    value=item.operand,
                    mode=item.profile.mode,
                    source=item.profile.mnemonic,
                    annotations=item.annotations,
                )
                items.replace_slice(index, index + 1, [literal])
                return literal
        return None

    @staticmethod
    def _literal_repr(node: IRLiteral) -> str:
        return node.describe()

    def _parse_literal_list(self, elements: Sequence[str]) -> Tuple[int, ...]:
        values: List[int] = []
        for text in elements:
            value = self._parse_literal_text(text)
            if value is None:
                return tuple()
            values.append(value)
        return tuple(values)

    @staticmethod
    def _parse_literal_text(text: str) -> Optional[int]:
        if text.startswith("lit(0x") and text.endswith(")"):
            try:
                return int(text[4:-1], 16)
            except ValueError:
                return None
        return None

    def _collect_literal_block_literals(
        self, items: _ItemList, start: int
    ) -> Tuple[Tuple[int, ...], int]:
        values: List[int] = []
        index = start
        while index < len(items):
            item = items[index]
            if isinstance(item, IRLiteral):
                values.append(item.value)
                index += 1
                continue
            break
        sequence = tuple(values)
        if sequence:
            return sequence, index
        return tuple(), start

    @staticmethod
    def _is_literal_block(self, values: Sequence[int]) -> bool:
        normalized = self._normalize_literal_block(values)
        if normalized is None:
            return False
        _, tail = normalized
        return not tail

    def _normalize_literal_block(
        self, values: Sequence[int]
    ) -> Optional[Tuple[Tuple[Tuple[int, int, int], ...], Tuple[int, ...]]]:
        if len(values) < 3:
            return None

        triplets: List[Tuple[int, int, int]] = []
        pos = 0
        while pos + 2 < len(values):
            chunk = tuple(values[pos : pos + 3])
            normalized = self._normalize_literal_triplet(chunk)
            if normalized is None:
                break
            triplets.append(normalized)
            pos += 3

        if not triplets:
            return None

        tail = tuple(values[pos:])
        return tuple(triplets), tail

    def _normalize_literal_triplet(
        self, chunk: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int]]:
        if len(chunk) != 3:
            return None

        marker_values = {0x0400, 0x0110}
        has_hint = 0x0067 in chunk
        marker = next((value for value in chunk if value in marker_values), None)
        if not has_hint or marker is None:
            return None

        anchor_candidates = [
            value for value in chunk if value not in {marker, 0x0067}
        ]
        if len(anchor_candidates) != 1:
            return None

        anchor = anchor_candidates[0]
        if anchor not in {0x6704, 0x0000}:
            return None

        return (0x0067, marker, anchor)

    def _pass_assign_ssa_names(self, items: _ItemList) -> None:
        for item in items:
            if isinstance(item, RawInstruction) and item.ssa_values:
                self._record_ssa(item, item.ssa_values, kinds=item.ssa_kinds)

    def _pass_testset_branches(
        self, items: _ItemList, metrics: NormalizerMetrics
    ) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.mnemonic == "testset_branch":
                expr = self._describe_condition(items, index, skip_literals=True)
                then_target = self._branch_target(item)
                else_target = self._fallthrough_target(item)
                node = IRTestSetBranch(
                    var=self._format_testset_var(item),
                    expr=expr,
                    then_target=then_target,
                    else_target=else_target,
                )
                self._transfer_ssa(item, node)
                replacement: List[Union[RawInstruction, IRNode]] = [node]
                if item.profile.mode == 0x33:
                    branch = IRIf(
                        condition=node.var,
                        then_target=then_target,
                        else_target=else_target,
                    )
                    replacement.append(branch)
                    metrics.if_branches += 1
                items.replace_slice(index, index + 1, replacement)
                metrics.testset_branches += 1
                continue

            index += 1

    def _pass_branches(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            if item.profile.kind is InstructionKind.BRANCH:
                node = IRIf(
                    condition=self._describe_condition(items, index),
                    then_target=self._branch_target(item),
                    else_target=self._fallthrough_target(item),
                )
                items.replace_slice(index, index + 1, [node])
                metrics.if_branches += 1
                continue

            index += 1

    def _pass_indirect_access(self, items: _ItemList, metrics: NormalizerMetrics) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            kind = item.event.kind
            if kind in {InstructionKind.INDIRECT_LOAD, InstructionKind.INDIRECT}:
                base_sources = self._stack_sources(items, index, 1)
                base_slot: Optional[IRSlot] = None
                base_alias = "stack"
                if base_sources:
                    base_name, base_node = base_sources[0]
                    self._promote_ssa_kind(base_name, SSAValueKind.POINTER)
                    base_alias = self._render_ssa(base_name)
                    if isinstance(base_node, IRLoad):
                        base_slot = base_node.slot
                target_name = item.ssa_values[0] if item.ssa_values else None
                target_alias = self._render_ssa(target_name) if target_name else "stack"
                base_name = base_sources[0][0] if base_sources else None
                memref, index = self._collect_memref(items, index, base_slot, base_name, item)
                if base_name is not None:
                    base_alias = self._render_ssa(base_name)
                pointer_alias = base_alias
                node = IRIndirectLoad(
                    base=base_alias,
                    offset=item.operand,
                    target=target_alias,
                    base_slot=base_slot,
                    ref=memref,
                    pointer=pointer_alias,
                )
                self._transfer_ssa(item, node)
                items.replace_slice(index, index + 1, [node])
                self._consume_indirect_configuration(items, index, node)
                metrics.loads += 1
                continue
            if kind is InstructionKind.INDIRECT_STORE:
                base_sources = self._stack_sources(items, index, 2)
                if base_sources:
                    base_name, base_node = base_sources[0]
                    value_name = base_sources[1][0] if len(base_sources) > 1 else None
                else:
                    base_name = None
                    base_node = None
                    value_name = None
                base_slot = base_node.slot if isinstance(base_node, IRLoad) else None
                if base_name:
                    self._promote_ssa_kind(base_name, SSAValueKind.POINTER)
                base_alias = self._render_ssa(base_name) if base_name else "stack"
                value_alias = self._render_ssa(value_name) if value_name else "stack"
                memref, index = self._collect_memref(items, index, base_slot, base_name, item)
                if base_name:
                    base_alias = self._render_ssa(base_name)
                pointer_alias = base_alias
                node = IRIndirectStore(
                    base=base_alias,
                    value=value_alias,
                    offset=item.operand,
                    base_slot=base_slot,
                    ref=memref,
                    pointer=pointer_alias,
                )
                items.replace_slice(index, index + 1, [node])
                self._consume_indirect_configuration(items, index, node)
                metrics.stores += 1
                continue

            index += 1

        self._sweep_indirect_configuration(items)

    def _pass_epilogue_prologue_compaction(self, items: _ItemList) -> None:
        index = 0
        while index < len(items):
            item = items[index]
            if not isinstance(item, RawInstruction):
                index += 1
                continue

            target_index = self._locate_epilogue_target(items, index)
            if target_index is None:
                index += 1
                continue

            target = items[target_index]
            raw_positions: List[int] = []
            safe = True
            for scan in range(index, target_index):
                candidate = items[scan]
                if isinstance(candidate, RawInstruction):
                    if isinstance(target, IRFunctionPrologue) and candidate.event.delta != 0:
                        safe = False
                        break
                    raw_positions.append(scan)
                    continue
                if isinstance(candidate, EPILOGUE_ALLOWED_NODE_TYPES):
                    continue
                safe = False
                break

            if not safe or not raw_positions:
                index += 1
                continue

            effects = [self._stack_effect_from_instruction(items[pos]) for pos in raw_positions]
            updated = self._append_epilogue_effects(target, effects)
            if updated is None:
                index += 1
                continue

            self._transfer_ssa(target, updated)
            for pos in reversed(raw_positions):
                raw = items[pos]
                assert isinstance(raw, RawInstruction)
                self._ssa_bindings.pop(id(raw), None)
                items.pop(pos)
                if pos < target_index:
                    target_index -= 1
            items.replace_slice(target_index, target_index + 1, [updated])
            continue

    def _locate_epilogue_target(self, items: _ItemList, index: int) -> Optional[int]:
        scan = index + 1
        while scan < len(items):
            node = items[scan]
            if isinstance(node, (IRReturn, IRTailCall, IRTailcallReturn, IRFunctionPrologue)):
                return scan
            if isinstance(node, RawInstruction):
                scan += 1
                continue
            if isinstance(node, EPILOGUE_ALLOWED_NODE_TYPES):
                scan += 1
                continue
            return None
        return None

    def _stack_effect_from_instruction(self, instruction: RawInstruction) -> IRStackEffect:
        pops = 0
        delta = instruction.event.delta
        if delta < 0:
            pops = -delta
        return IRStackEffect(
            mnemonic=instruction.mnemonic,
            operand=instruction.operand,
            pops=pops,
            operand_role=instruction.profile.operand_role(),
            operand_alias=instruction.profile.operand_alias(),
        )

    def _append_epilogue_effects(
        self,
        target: IRNode,
        effects: Sequence[IRStackEffect],
    ) -> Optional[IRNode]:
        if not effects:
            return target
        if isinstance(target, IRReturn):
            cleanup = list(effects) + list(target.cleanup)
            combined = tuple(self._coalesce_epilogue_steps(cleanup))
            return IRReturn(
                values=target.values,
                varargs=target.varargs,
                cleanup=combined,
                mask=target.mask,
            )
        if isinstance(target, IRTailCall):
            cleanup = list(effects) + list(target.cleanup)
            combined = tuple(self._coalesce_epilogue_steps(cleanup))
            return replace(target, cleanup=combined)
        if isinstance(target, IRTailcallReturn):
            cleanup = list(effects) + list(target.cleanup)
            combined = tuple(self._coalesce_epilogue_steps(cleanup))
            return replace(target, cleanup=combined)
        if isinstance(target, IRFunctionPrologue):
            return target
        return None

    # ------------------------------------------------------------------
    # description helpers
    # ------------------------------------------------------------------
    def _describe_stack_top(self, items: _ItemList, index: int) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    name = self._ssa_value(candidate)
                    if name:
                        return name
                    return self._describe_value(candidate)
            elif isinstance(candidate, IRLiteral):
                name = self._ssa_value(candidate)
                if name:
                    return name
                return candidate.describe()
            elif isinstance(candidate, IRLiteralChunk):
                name = self._ssa_value(candidate)
                if name:
                    return name
                return candidate.describe()
            elif isinstance(candidate, IRStackDuplicate):
                return candidate.value
            elif isinstance(candidate, IRNode):
                name = self._ssa_value(candidate)
                if name:
                    return name
                describe = getattr(candidate, "describe", None)
                if callable(describe):
                    return describe()
            scan -= 1
        return "stack_top"

    def _describe_value(self, instruction: RawInstruction) -> str:
        mapped = self._ssa_value(instruction)
        if mapped is not None:
            return mapped
        mnemonic = instruction.mnemonic
        operand = instruction.operand
        if mnemonic == "push_literal":
            return f"lit(0x{operand:04X})"
        if mnemonic.startswith("reduce"):
            return f"reduce(0x{operand:04X})"
        if "slot" in mnemonic:
            return f"slot(0x{operand:04X})"
        if instruction.profile.kind is InstructionKind.LITERAL:
            return f"lit(0x{operand:04X})"
        return instruction.describe_source()

    def _describe_condition(self, items: _ItemList, index: int, *, skip_literals: bool = False) -> str:
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                if candidate.pushes_value():
                    if skip_literals and candidate.mnemonic == "push_literal":
                        mapped = self._ssa_value(candidate, raw=True)
                        if mapped is not None:
                            self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                            return self._render_ssa(mapped)
                        scan -= 1
                        continue
                    mapped = self._ssa_value(candidate, raw=True)
                    if mapped is not None:
                        self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                        return self._render_ssa(mapped)
                    return self._describe_value(candidate)
                if skip_literals:
                    mapped = self._ssa_value(candidate, raw=True)
                    if mapped is not None:
                        self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                        return self._render_ssa(mapped)
                    return self._describe_value(candidate)
            if isinstance(candidate, IRLiteral):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                if skip_literals:
                    scan -= 1
                    continue
                return candidate.describe()
            if isinstance(candidate, IRLiteralChunk):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                if skip_literals:
                    scan -= 1
                    continue
                return candidate.describe()
            if isinstance(candidate, IRStackDuplicate):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                return candidate.value
            if isinstance(candidate, IRNode):
                mapped = self._ssa_value(candidate, raw=True)
                if mapped is not None:
                    self._promote_ssa_kind(mapped, SSAValueKind.BOOLEAN)
                    return self._render_ssa(mapped)
                return getattr(candidate, "describe", lambda: "expr()")()
            scan -= 1
        return "stack_top"

    @staticmethod
    def _branch_target(instruction: RawInstruction) -> int:
        return instruction.operand

    @staticmethod
    def _fallthrough_target(instruction: RawInstruction) -> int:
        return instruction.offset + 4

    @staticmethod
    def _classify_slot(operand: int) -> IRSlot:
        if operand < 0x1000:
            space = MemSpace.FRAME
        elif operand < 0x8000:
            space = MemSpace.GLOBAL
        else:
            space = MemSpace.CONST
        return IRSlot(space=space, index=operand)

    def _collect_memref(
        self,
        items: _ItemList,
        index: int,
        base_slot: Optional[IRSlot],
        base_name: Optional[str],
        instruction: RawInstruction,
    ) -> Tuple[Optional[MemRef], int]:
        components: List[Tuple[int, int, bool]] = []
        scan = index - 1
        while scan >= 0:
            candidate = items[scan]
            if isinstance(candidate, RawInstruction):
                value = self._memref_component(candidate)
                if value is None:
                    break
                components.append((scan, value, True))
                scan -= 1
                continue
            if isinstance(candidate, IRPageRegister):
                value = self._page_register_component(candidate)
                if value is not None:
                    components.append((scan, value, False))
                scan -= 1
                continue
            if self._is_memref_bridge(candidate):
                scan -= 1
                continue
            break

        removal_indexes = [position for position, _, remove in components if remove]
        removal_indexes.sort(reverse=True)
        for position in removal_indexes:
            removed = items.pop(position)
            self._ssa_bindings.pop(id(removed), None)
        index -= len(removal_indexes)

        components.reverse()
        values = [value for _, value, _ in components]
        bank = values[0] if values else None
        base_value = values[1] if len(values) > 1 else None
        page = instruction.operand >> 8
        offset = instruction.operand & 0xFF
        region_override: Optional[str] = None
        page_alias_override: Optional[str] = None
        symbol_override: Optional[str] = None
        if instruction.profile.opcode == 0x69 and bank is None and base_slot is None:
            if self._operand_is_io_slot(instruction.operand):
                region_override = "io"
                page_alias_override = IO_PORT_NAME
                symbol_override = IO_PORT_NAME

        region, page_alias = self._memref_region(base_slot, bank, page)
        if region_override is not None:
            region = region_override
        if page_alias_override is not None:
            page_alias = page_alias_override
        symbol = symbol_override or self._memref_symbol(region, bank, page, offset)
        memref = MemRef(
            region=region,
            bank=bank,
            base=base_value,
            page=page,
            offset=offset,
            symbol=symbol,
            page_alias=page_alias,
        )

        if base_name and symbol:
            self._ssa_aliases[base_name] = symbol

        return memref, index

    def _consume_indirect_configuration(
        self,
        items: _ItemList,
        index: int,
        node: Union[IRIndirectLoad, IRIndirectStore],
    ) -> None:
        scan = index + 1
        pending_literal_index: Optional[int] = None
        pending_literal: Optional[IRNode] = None
        pending_value: Optional[str] = None
        pending_literal_value: Optional[int] = None
        pending_ssa: Optional[Tuple[str, ...]] = None
        pending_kinds: Optional[Tuple[SSAValueKind, ...]] = None

        while scan < len(items):
            candidate = items[scan]

            if isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                pending_literal_index = scan
                pending_literal = candidate
                pending_value = candidate.describe()
                pending_literal_value = candidate.value & 0xFFFF if isinstance(candidate, IRLiteral) else None
                stored = self._ssa_bindings.get(id(candidate))
                if stored:
                    pending_ssa = stored
                    pending_kinds = tuple(self._ssa_types.get(name, SSAValueKind.UNKNOWN) for name in stored)
                else:
                    pending_ssa = None
                    pending_kinds = None
                scan += 1
                continue

            if isinstance(candidate, IRPageRegister):
                pending_value = candidate.value
                pending_literal_value = candidate.literal
                pending_literal_index = None
                pending_literal = None
                pending_ssa = self._ssa_bindings.get(id(candidate))
                if pending_ssa:
                    pending_kinds = tuple(self._ssa_types.get(name, SSAValueKind.UNKNOWN) for name in pending_ssa)
                else:
                    pending_kinds = None
                scan += 1
                continue

            if isinstance(candidate, IO_BRIDGE_NODE_TYPES):
                scan += 1
                continue

            if isinstance(candidate, IRConditionMask):
                scan += 1
                continue

            if isinstance(candidate, RawInstruction):
                mnemonic = candidate.mnemonic

                if mnemonic in INDIRECT_PAGE_REGISTER_MNEMONICS:
                    register = INDIRECT_PAGE_REGISTER_MNEMONICS[mnemonic]
                    value_repr = pending_value
                    literal_value = pending_literal_value
                    literal_names = pending_ssa
                    literal_kinds = pending_kinds

                    if pending_literal is not None and pending_literal_index is not None:
                        removed = items.pop(pending_literal_index)
                        self._ssa_bindings.pop(id(removed), None)
                        if pending_literal_index < scan:
                            scan -= 1
                        pending_literal_index = None
                        pending_literal = None
                        pending_ssa = None
                        pending_kinds = None

                    page_node = IRPageRegister(
                        register=register,
                        value=value_repr,
                        literal=literal_value,
                    )

                    if literal_names:
                        self._record_ssa(page_node, literal_names, kinds=literal_kinds)
                    else:
                        names = self._ssa_bindings.get(id(candidate)) or candidate.ssa_values
                        if names:
                            kinds = tuple(
                                self._ssa_types.get(name, SSAValueKind.UNKNOWN)
                                for name in names
                            )
                            self._record_ssa(page_node, names, kinds=kinds)
                        else:
                            self._record_ssa(page_node, tuple())

                    self._ssa_bindings.pop(id(candidate), None)
                    items.replace_slice(scan, scan + 1, [page_node])
                    pending_value = value_repr
                    pending_literal_value = literal_value
                    pending_ssa = None
                    pending_kinds = None
                    scan += 1
                    continue

                if mnemonic in INDIRECT_MASK_MNEMONICS:
                    mask_node = IRConditionMask(source=mnemonic, mask=candidate.operand)
                    self._transfer_ssa(candidate, mask_node)
                    items.replace_slice(scan, scan + 1, [mask_node])
                    scan += 1
                    continue

                if mnemonic in INDIRECT_CONFIGURATION_BRIDGES:
                    pending_literal_index = None
                    pending_literal = None
                    pending_ssa = None
                    pending_kinds = None
                    pending_value = f"lit(0x{candidate.operand:04X})"
                    pending_literal_value = candidate.operand & 0xFFFF
                    scan += 1
                    continue

            break

    def _sweep_indirect_configuration(self, items: _ItemList) -> None:
        index = 0
        pending_literal_index: Optional[int] = None
        pending_literal: Optional[IRNode] = None
        pending_value: Optional[str] = None
        pending_literal_value: Optional[int] = None
        pending_ssa: Optional[Tuple[str, ...]] = None
        pending_kinds: Optional[Tuple[SSAValueKind, ...]] = None

        while index < len(items):
            candidate = items[index]

            if isinstance(candidate, (IRLiteral, IRLiteralChunk)):
                pending_literal_index = index
                pending_literal = candidate
                pending_value = candidate.describe()
                pending_literal_value = candidate.value & 0xFFFF if isinstance(candidate, IRLiteral) else None
                stored = self._ssa_bindings.get(id(candidate))
                if stored:
                    pending_ssa = stored
                    pending_kinds = tuple(self._ssa_types.get(name, SSAValueKind.UNKNOWN) for name in stored)
                else:
                    pending_ssa = None
                    pending_kinds = None
                index += 1
                continue

            if isinstance(candidate, IRPageRegister):
                pending_value = candidate.value
                pending_literal_value = candidate.literal
                pending_literal_index = None
                pending_literal = None
                pending_ssa = self._ssa_bindings.get(id(candidate))
                if pending_ssa:
                    pending_kinds = tuple(self._ssa_types.get(name, SSAValueKind.UNKNOWN) for name in pending_ssa)
                else:
                    pending_kinds = None
                index += 1
                continue

            if isinstance(candidate, RawInstruction):
                mnemonic = candidate.mnemonic

                if mnemonic in INDIRECT_PAGE_REGISTER_MNEMONICS:
                    register = INDIRECT_PAGE_REGISTER_MNEMONICS[mnemonic]
                    value_repr = pending_value
                    literal_value = pending_literal_value
                    literal_names = pending_ssa
                    literal_kinds = pending_kinds

                    if pending_literal is not None and pending_literal_index is not None:
                        removed = items.pop(pending_literal_index)
                        self._ssa_bindings.pop(id(removed), None)
                        if pending_literal_index < index:
                            index -= 1
                        pending_literal_index = None
                        pending_literal = None
                        pending_ssa = None
                        pending_kinds = None

                    page_node = IRPageRegister(
                        register=register,
                        value=value_repr,
                        literal=literal_value,
                    )

                    if literal_names:
                        self._record_ssa(page_node, literal_names, kinds=literal_kinds)
                    else:
                        names = self._ssa_bindings.get(id(candidate)) or candidate.ssa_values
                        if names:
                            kinds = tuple(
                                self._ssa_types.get(name, SSAValueKind.UNKNOWN)
                                for name in names
                            )
                            self._record_ssa(page_node, names, kinds=kinds)
                        else:
                            self._record_ssa(page_node, tuple())

                    self._ssa_bindings.pop(id(candidate), None)
                    items.replace_slice(index, index + 1, [page_node])
                    pending_value = value_repr
                    pending_literal_value = literal_value
                    pending_ssa = None
                    pending_kinds = None
                    index += 1
                    continue

                if mnemonic in INDIRECT_MASK_MNEMONICS:
                    mask_node = IRConditionMask(source=mnemonic, mask=candidate.operand)
                    self._transfer_ssa(candidate, mask_node)
                    items.replace_slice(index, index + 1, [mask_node])
                    index += 1
                    continue

                if mnemonic in INDIRECT_CONFIGURATION_BRIDGES:
                    pending_literal_index = None
                    pending_literal = None
                    pending_ssa = None
                    pending_kinds = None
                    pending_value = f"lit(0x{candidate.operand:04X})"
                    pending_literal_value = candidate.operand & 0xFFFF
                    index += 1
                    continue

            pending_literal_index = None
            pending_literal = None
            pending_value = None
            pending_literal_value = None
            pending_ssa = None
            pending_kinds = None
            index += 1

    @staticmethod
    def _is_memref_bridge(node: Union[RawInstruction, IRNode]) -> bool:
        return isinstance(node, (IRLoad, IRStackDuplicate))

    def _memref_component(self, instruction: RawInstruction) -> Optional[int]:
        if instruction.event.delta != 0:
            return None
        if instruction.event.popped_types or instruction.event.pushed_types:
            return None
        mnemonic = instruction.mnemonic
        if not mnemonic.startswith("op_"):
            return None
        parts = mnemonic.split("_")
        if len(parts) != 3:
            return None
        try:
            high = int(parts[1], 16)
            low = int(parts[2], 16)
        except ValueError:
            return None
        return (high << 8) | low

    def _page_register_component(self, node: IRPageRegister) -> Optional[int]:
        if node.register != PAGE_REGISTER:
            return None
        if node.literal is not None:
            return node.literal
        if node.value:
            parsed = self._flag_literal_value(node.value)
            if parsed is not None:
                return parsed
        return None

    def _memref_region(
        self, base_slot: Optional[IRSlot], bank: Optional[int], page: Optional[int]
    ) -> Tuple[str, Optional[str]]:
        if base_slot is not None:
            return base_slot.space.name.lower(), None
        if bank is None:
            return "mem", None

        normalized = bank & 0xFFF0
        alias = MEMORY_BANK_ALIASES.get(normalized)
        page_alias = self._memref_page_alias(normalized, page)
        if alias is not None:
            return alias, page_alias

        label = self._memref_regions.get(bank)
        if label is None:
            label = f"bank_{bank:04X}"
            self._memref_regions[bank] = label
        return label, page_alias

    def _memref_page_alias(
        self, bank: Optional[int], page: Optional[int]
    ) -> Optional[str]:
        if page is None:
            return None
        if bank is not None:
            alias = MEMORY_PAGE_ALIASES.get((bank, page))
            if alias is not None:
                return alias
            alias = MEMORY_PAGE_ALIASES.get((bank & 0xFFF0, page))
            if alias is not None:
                return alias
        return MEMORY_PAGE_ALIASES.get((None, page))

    def _memref_symbol(
        self, region: str, bank: Optional[int], page: Optional[int], offset: Optional[int]
    ) -> Optional[str]:
        if bank is None:
            return None
        key = (region, bank, page, offset)
        alias = self._memref_symbols.get(key)
        if alias is not None:
            return alias
        prefix = region[:1].upper() if region else "M"
        if not prefix.isalpha():
            prefix = "M"
        alias_base = f"{prefix}_{bank:04X}"
        if page is not None:
            alias_base += f"_{page:02X}"
        if offset is not None:
            alias_base += f"_{offset:02X}"
        counter = self._memref_symbol_counters[alias_base]
        self._memref_symbol_counters[alias_base] += 1
        alias = alias_base if counter == 0 else f"{alias_base}_{counter}"
        self._memref_symbols[key] = alias
        return alias

    @staticmethod
    def _flag_literal_value(text: str) -> Optional[int]:
        if text.startswith("lit(0x") and text.endswith(")"):
            try:
                value = int(text[4:-1], 16)
            except ValueError:
                return None
            if value in {0x0166, 0x0266}:
                return value
        return None

    # ------------------------------------------------------------------
    # annotation helpers
    # ------------------------------------------------------------------
    def _is_annotation_only(self, instruction: RawInstruction) -> bool:
        if instruction.offset in self._annotation_offsets:
            return True
        if instruction.profile.is_literal_marker():
            return True
        if not instruction.annotations:
            return False
        for note in instruction.annotations:
            if note in ANNOTATION_MNEMONICS:
                return True
            if note.startswith("literal_marker") or note.startswith("inline_ascii_chunk"):
                return True
            if note.startswith("op_") and instruction.profile.kind is InstructionKind.LITERAL:
                return True
        return False

    def _is_stack_neutral_bridge(
        self,
        instruction: RawInstruction,
        items: Sequence[Union[RawInstruction, IRNode]],
        index: int,
    ) -> bool:
        event = instruction.event
        if event.delta != 0:
            return False
        if instruction.profile.kind in STACK_NEUTRAL_CONTROL_KINDS:
            return False
        if instruction.profile.is_control():
            return False
        if self._has_profile_side_effects(instruction):
            return False
        if self._is_bridge_edge_position(items, index):
            return False
        if self._is_io_bridge_instruction(instruction):
            return True
        return self._has_ascii_neighbor(items, index)

    def _has_profile_side_effects(self, instruction: RawInstruction) -> bool:
        profile = instruction.profile
        if profile.kind in SIDE_EFFECT_KIND_HINTS:
            return True
        operand = instruction.operand
        if operand in IO_SLOT_ALIASES or operand == PAGE_REGISTER:
            return True
        alias = profile.operand_alias()
        if alias and any(keyword in alias.lower() for keyword in SIDE_EFFECT_KEYWORDS):
            return True
        role = profile.operand_role()
        if role and any(keyword in role.lower() for keyword in SIDE_EFFECT_KEYWORDS):
            return True
        category = profile.category
        if category and any(keyword in category.lower() for keyword in SIDE_EFFECT_KEYWORDS):
            return True
        summary = profile.summary
        if summary and any(keyword in summary.lower() for keyword in SIDE_EFFECT_KEYWORDS):
            return True
        mnemonic = profile.mnemonic.lower()
        if any(keyword in mnemonic for keyword in SIDE_EFFECT_KEYWORDS):
            return True
        return False

    def _is_bridge_edge_position(
        self,
        items: Sequence[Union[RawInstruction, IRNode]],
        index: int,
    ) -> bool:
        if index == 0 or index == len(items) - 1:
            return True
        next_item = items[index + 1]
        if isinstance(next_item, RawInstruction):
            next_kind = next_item.profile.kind
            if next_kind in STACK_NEUTRAL_CONTROL_KINDS or next_item.profile.is_control():
                return True
        return False

    def _has_ascii_neighbor(
        self,
        items: Sequence[Union[RawInstruction, IRNode]],
        index: int,
    ) -> bool:
        if index > 0 and self._is_ascii_related(items[index - 1]):
            return True
        if index + 1 < len(items) and self._is_ascii_related(items[index + 1]):
            return True
        return False

    def _is_ascii_related(self, item: Union[RawInstruction, IRNode]) -> bool:
        if isinstance(item, ASCII_NEIGHBOR_NODE_TYPES):
            return True
        if isinstance(item, IRLiteral):
            return any("ascii" in note for note in item.annotations)
        if isinstance(item, RawInstruction):
            profile = item.profile
            if profile.kind is InstructionKind.ASCII_CHUNK:
                return True
            if profile.mnemonic.startswith("inline_ascii_chunk"):
                return True
            return any("ascii" in note for note in item.annotations)
        return False

    def _format_annotation(self, instruction: RawInstruction) -> str:
        parts = [f"0x{instruction.offset:06X}"]
        if instruction.annotations:
            parts.extend(instruction.annotations)
        else:
            parts.append(instruction.mnemonic)
        return " ".join(parts)

    @staticmethod
    def _format_testset_var(instruction: RawInstruction) -> str:
        operand = instruction.operand
        return f"slot(0x{operand:04X})"


__all__ = ["IRNormalizer", "RawInstruction", "RawBlock"]
