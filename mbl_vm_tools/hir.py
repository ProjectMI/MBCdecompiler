from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from mbl_vm_tools.IR import IRFunction, IRNode, build_function_ir
from mbl_vm_tools.parser import MBCModule


HIR_CONTRACT_VERSION = "hir-v11"


def _coerce_hir_block(payload: dict[str, Any]) -> "HIRBlock":
    return HIRBlock(
        id=payload.get("id", ""),
        index=int(payload.get("index", 0)),
        start_offset=int(payload.get("start_offset", 0)),
        end_offset=int(payload.get("end_offset", 0)),
        instruction_indices=list(payload.get("instruction_indices", [])),
        entry_stack=list(payload.get("entry_stack", [])),
        exit_stack=list(payload.get("exit_stack", [])),
        phi_bindings=list(payload.get("phi_bindings", [])),
        block_params=list(payload.get("block_params", [])),
        incoming_args={str(key): list(value) for key, value in dict(payload.get("incoming_args", {})).items()},
        statements=list(payload.get("statements", [])),
        terminator=dict(payload.get("terminator", {})),
        branch_target=payload.get("branch_target"),
        fallthrough_target=payload.get("fallthrough_target"),
        successors=list(payload.get("successors", [])),
        predecessors=list(payload.get("predecessors", [])),
        flags=list(payload.get("flags", [])),
    )



@dataclass
class CanonicalInstruction:
    index: int
    origin_node_index: int
    offset: int
    size: int
    raw_kind: str
    terminal_kind: str
    macro_kind: Optional[str]
    opcode: str
    semantic_op: str
    display_op: str
    operands: dict[str, Any]
    confidence: float
    stack_inputs_required: int
    stack_outputs: int
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    control: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataflowValue:
    name: str
    producer_instruction: int
    expr: str
    use_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HIRBlock:
    id: str
    index: int
    start_offset: int
    end_offset: int
    instruction_indices: list[int]
    entry_stack: list[str]
    exit_stack: list[str]
    phi_bindings: list[str]
    block_params: list[str]
    incoming_args: dict[str, list[str]]
    statements: list[str]
    terminator: dict[str, Any]
    branch_target: Optional[str]
    fallthrough_target: Optional[str]
    successors: list[str]
    predecessors: list[str]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HIRFunction:
    name: str
    span: dict[str, int]
    slice_mode: str
    summary: dict[str, Any]
    cfg: dict[str, Any]
    canonical_instructions: list[CanonicalInstruction]
    core_dataflow_values: list[DataflowValue]
    core_hir_blocks: list[HIRBlock]
    normalized_hir_blocks: list[HIRBlock]
    analysis_hints: dict[str, Any]
    report: dict[str, Any]
    body_selection: dict[str, Any]

    def to_dict(self, include_canonical: bool = True, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "span": self.span,
            "slice_mode": self.slice_mode,
            "summary": self.summary,
            "cfg": self.cfg,
            "body_selection": self.body_selection,
            "input_contract": HIR_CONTRACT_VERSION,
            "core_hir": {
                "dataflow_values": [value.to_dict() for value in self.core_dataflow_values],
                "hir_blocks": [block.to_dict() for block in self.core_hir_blocks],
            },
            "normalized_hir": {
                "hir_blocks": [block.to_dict() for block in self.normalized_hir_blocks],
            },
            "analysis_hints": self.analysis_hints,
            "report": self.report,
        }
        if include_canonical:
            payload["canonical_instructions"] = [inst.to_dict() for inst in self.canonical_instructions]
        return payload


@dataclass
class _StackPlanBlock:
    entry_depth: int
    exit_depth: int
    output_used: dict[int, bool]


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_placeholder_value(name: str) -> bool:
    return isinstance(name, str) and name.startswith(("in_", "undef_"))


_TEMP_LIKE_RE = re.compile(r"\b(?:t\d+|phi_[A-Za-z0-9_]+|m_[A-Za-z0-9_]+|arg\d+|in_[A-Za-z0-9_]+|undef_[A-Za-z0-9_]+)\b")
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def _replace_vars(text: Optional[str], mapping: dict[str, str]) -> Optional[str]:
    updated = text
    for old_name, new_name in mapping.items():
        updated = _replace_var(updated, old_name, new_name)
    return updated


def _iter_temp_like_refs(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return _TEMP_LIKE_RE.findall(text)


def _match_assignment(stmt: str) -> Optional[tuple[str, str]]:
    match = _ASSIGNMENT_RE.match(stmt)
    if match is None:
        return None
    return match.group(1), match.group(2)


# --- canonical lowering ---------------------------------------------------


def _slot_name(ref: Any, mode: Any = None) -> str:
    if ref is None:
        return "slot<?>"
    if mode is None:
        return f"slot_{ref}"
    return f"slot_{ref}@{mode}"


def _format_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _format_call_target(inst: CanonicalInstruction) -> str:
    operands = inst.operands
    if inst.semantic_op == "syscall":
        opid = operands.get("opid")
        family = operands.get("family")
        if opid is not None:
            return f"syscall_{opid}"
        if family:
            return family.lower()
        return "syscall"
    rel = operands.get("rel")
    if rel is not None:
        return f"call_rel_{rel}"
    family = operands.get("family")
    if family:
        return family.lower()
    return "call"


def _format_record_constructor(inst: CanonicalInstruction) -> str:
    parts: list[str] = []
    for key in sorted(inst.operands.keys()):
        if key in {"family", "macro_family", "prefix_chain", "record_kind", "stack_inputs_required"}:
            continue
        parts.append(f"{key}={inst.operands[key]}")
    joined = ", ".join(parts)
    record_kind = inst.operands.get("record_kind")
    if isinstance(record_kind, str) and record_kind:
        suffix = record_kind.lower()
    else:
        family = inst.operands.get("family") or inst.operands.get("macro_family")
        family_text = str(family or "")
        if family_text.endswith("REC41"):
            suffix = "rec41"
        elif family_text.endswith("REC62"):
            suffix = "rec62"
        elif family_text.endswith("REC61"):
            suffix = "rec61"
        else:
            suffix = inst.terminal_kind.lower()
    return f"{suffix}({joined})"


def _format_aggregate(inst: CanonicalInstruction) -> str:
    children = inst.operands.get("children") or []
    if children:
        child_bits = []
        for child in children:
            tag = child.get("tag")
            ref = child.get("ref")
            child_bits.append(f"({tag}, {ref})")
        return f"aggregate_{inst.operands.get('arity', len(children))}({', '.join(child_bits)})"
    arity = inst.operands.get("arity")
    return f"aggregate_{arity if arity is not None else '?'}()"


def _canonical_expr(inst: CanonicalInstruction, args: list[str]) -> str:
    if inst.semantic_op == "const":
        value = inst.operands.get("value")
        if value is None:
            value = inst.operands.get("imm")
        return _format_literal(value)
    if inst.semantic_op == "load":
        if inst.operands.get("family"):
            return f"load_{str(inst.operands['family']).lower()}({', '.join(args)})" if args else f"load_{str(inst.operands['family']).lower()}()"
        return _slot_name(inst.operands.get("ref"), inst.operands.get("mode"))
    if inst.semantic_op == "read_field":
        base = args[-1] if args else _slot_name(inst.operands.get("ref"), inst.operands.get("mode"))
        field_name = inst.operands.get("family") or inst.operands.get("field") or "field"
        return f"{base}.{str(field_name).lower()}"
    if inst.semantic_op == "make_record":
        return _format_record_constructor(inst)
    if inst.semantic_op == "aggregate":
        return _format_aggregate(inst)
    if inst.semantic_op in {"call", "syscall", "opaque_call"}:
        return f"{_format_call_target(inst)}({', '.join(args)})"
    if inst.semantic_op == "cmp":
        return f"cmp({', '.join(args)})"
    if inst.semantic_op.startswith("opaque_"):
        raw = inst.macro_kind or inst.raw_kind
        return f"{inst.semantic_op}<{raw}>({', '.join(args)})"
    return f"{inst.semantic_op}({', '.join(args)})"


def _infer_stack_io(inst: CanonicalInstruction) -> tuple[int, int]:
    op = inst.semantic_op
    operands = inst.operands
    if "stack_inputs_required" in operands:
        forced_in = int(operands.get("stack_inputs_required") or 0)
        forced_out = int(operands.get("stack_outputs", 0) or 0)
        return forced_in, forced_out
    if op in {"const", "load", "read_field", "make_record", "aggregate", "call", "syscall", "opaque_call", "cmp", "opaque_const", "opaque_load", "opaque_record", "opaque_aggregate", "opaque_op"}:
        if op in {"call", "syscall", "opaque_call"}:
            return int(operands.get("argc", 0) or 0), 1
        if op == "cmp":
            return 2, 1
        if op == "read_field":
            return 1, 1
        return 0, 1
    if op in {"store", "write_field", "opaque_store"}:
        return 2, 0
    if op == "branch":
        prefix_chain = operands.get("prefix_chain") or []
        family = str(operands.get("family") or "")
        if prefix_chain or family.startswith("PAD"):
            return 2, 0
        return 1, 0
    if op == "return":
        return 1, 0
    return 0, 0


def _effect_lists(inst: CanonicalInstruction) -> tuple[list[str], list[str], list[str]]:
    op = inst.semantic_op
    operands = inst.operands
    reads: list[str] = []
    writes: list[str] = []
    effects: list[str] = []
    if op in {"load", "read_field"}:
        reads.append(_slot_name(operands.get("ref"), operands.get("mode")))
    if op in {"store", "write_field"}:
        writes.append(_slot_name(operands.get("ref"), operands.get("mode")))
    if op in {"call", "opaque_call"}:
        effects.append("call")
    if op == "syscall":
        effects.extend(["call", "native_call"])
    if op == "make_record":
        effects.append("alloc")
    if op == "aggregate":
        effects.append("aggregate")
    if op == "branch":
        effects.append("control")
    if op == "return":
        effects.extend(["control", "return"])
    return reads, writes, effects


def _make_inst(*, inst_index: int, node: IRNode, semantic_op: str, display_op: str, operands: dict[str, Any], macro_kind: Optional[str] = None) -> CanonicalInstruction:
    dummy = CanonicalInstruction(
        index=inst_index,
        origin_node_index=node.index,
        offset=node.offset,
        size=node.size,
        raw_kind=node.raw_kind,
        terminal_kind=node.terminal_kind,
        macro_kind=macro_kind,
        opcode=node.opcode,
        semantic_op=semantic_op,
        display_op=display_op,
        operands=operands,
        confidence=node.confidence,
        stack_inputs_required=0,
        stack_outputs=0,
        control={},
    )
    stack_in, stack_out = _infer_stack_io(dummy)
    reads, writes, effects = _effect_lists(dummy)
    dummy.stack_inputs_required = stack_in
    dummy.stack_outputs = stack_out
    dummy.reads = reads
    dummy.writes = writes
    dummy.effects = effects
    return dummy


def _byte_literal(value: Any) -> Any:
    if isinstance(value, int) and 0 <= value <= 0xFF:
        return f"0x{value:02X}"
    return value


def _signature_family_name(node: IRNode) -> str:
    family = node.raw_kind if node.raw_kind.startswith("SIG_") else node.terminal_kind if node.terminal_kind.startswith("SIG_") else node.raw_kind
    return family[4:] if family.startswith("SIG_") else family


def _signature_const_step(family: str, value: Any, *, suffix: str | None = None) -> tuple[str, str, dict[str, Any]] | None:
    if value is None:
        return None
    tagged_family = family if suffix is None else f"{family}.{suffix}"
    return ("const", "const", {"value": value, "family": tagged_family})


def _signature_load_step(
    family: str,
    operands: dict[str, Any],
    *,
    ref_key: str = "ref",
    mode_key: str = "ref_mode",
    op_key: str = "ref_op",
    ref_width: int | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    ref = operands.get(ref_key)
    if ref is None:
        return None
    load_operands: dict[str, Any] = {"ref": ref, "macro_family": family}
    mode = operands.get(mode_key)
    if mode is not None:
        load_operands["mode"] = _byte_literal(mode)
    source_op = operands.get(op_key)
    if source_op is not None:
        load_operands["source_op"] = _byte_literal(source_op)
    if ref_width is not None:
        load_operands["ref_width"] = ref_width
    return ("load", "load", load_operands)


def _signature_return_step(family: str, operands: dict[str, Any], *, tail_form: str | None = None) -> tuple[str, str, dict[str, Any]]:
    return (
        "return",
        "return",
        {
            "family": family,
            "tail_form": tail_form if tail_form is not None else operands.get("tail_form"),
        },
    )


def _signature_expand_nested(family: str, nested_kind: Any, nested_payload: Any) -> list[tuple[str, str, dict[str, Any]]]:
    if not isinstance(nested_kind, str) or not isinstance(nested_payload, dict):
        return []
    if nested_kind == "PAIR72_23":
        return [_signature_return_step(family, nested_payload, tail_form="pair72_23")]
    if nested_kind == "SIG_RETURN_TAIL":
        steps: list[tuple[str, str, dict[str, Any]]] = []
        const_step = _signature_const_step(family, nested_payload.get("imm"), suffix="tail")
        if const_step is not None:
            steps.append(const_step)
        steps.append(_signature_return_step(family, nested_payload))
        return steps
    if nested_kind == "BR":
        return [
            (
                "branch",
                "branch",
                {
                    "family": family,
                    "branch_op": _byte_literal(nested_payload.get("op")),
                    "offset": nested_payload.get("off"),
                },
            )
        ]
    return []


def _signature_to_steps(node: IRNode) -> list[tuple[str, str, dict[str, Any]]]:
    family = _signature_family_name(node)
    operands = dict(node.operands)

    if family == "MAIN_PARSECOMMAND_TAIL":
        steps: list[tuple[str, str, dict[str, Any]]] = []
        cmp_value = operands.get("cmp_value")
        if cmp_value is not None:
            steps.append(("const", "const", {"value": cmp_value, "family": family}))
        steps.append(
            (
                "branch",
                "branch",
                {
                    "family": family,
                    "branch_op": operands.get("branch_op"),
                    "offset": operands.get("offset"),
                    # This tail reuses the immediately preceding loaded value and
                    # compares it against the embedded literal before branching.
                    "stack_inputs_required": 2,
                },
            )
        )
        return steps

    if family == "RETURN_TAIL":
        steps: list[tuple[str, str, dict[str, Any]]] = []
        const_step = _signature_const_step(family, operands.get("imm"))
        if const_step is not None:
            steps.append(const_step)
        steps.append(_signature_return_step(family, operands))
        return steps

    if family == "U32_U8_CALL66_TAIL":
        steps: list[tuple[str, str, dict[str, Any]]] = []
        const_u32 = _signature_const_step(family, operands.get("value"), suffix="u32")
        if const_u32 is not None:
            steps.append(const_u32)
        const_u8 = _signature_const_step(family, operands.get("arg"), suffix="u8")
        if const_u8 is not None:
            steps.append(const_u8)
        steps.append(("syscall", "syscall", {"family": family, "opid": 0x27, "argc": 2, "immediate_args": True}))
        return steps

    if family == "INPUTDONE_SHORT":
        steps: list[tuple[str, str, dict[str, Any]]] = []
        load_step = _signature_load_step(family, operands)
        if load_step is not None:
            steps.append(load_step)
        steps.append(_signature_return_step(family, operands, tail_form="pair72_23"))
        return steps

    if family.startswith("CONST_U32") or family in {"CONST_0100", "CONST_U32_TRAILER"}:
        steps: list[tuple[str, str, dict[str, Any]]] = []
        const_step = _signature_const_step(family, operands.get("value"), suffix="u32")
        if const_step is not None:
            steps.append(const_step)

        if family in {"CONST_U32_REF", "CONST_U32_PFX_3D_REF", "CONST_U32_PFX_3D_30_REF", "CONST_U32_PFX_5E_REF", "CONST_U32_26_REF"}:
            load_step = _signature_load_step(family, operands)
            if load_step is not None:
                steps.append(load_step)
            return steps or [(node.semantic_op, node.semantic_op, dict(operands))]

        if family == "CONST_U32_REF16":
            load_step = _signature_load_step(family, operands, mode_key="mode", op_key="op", ref_width=16)
            if load_step is None:
                load_step = ("load", "load", {"ref": operands.get("ref"), "mode": "0x10", "ref_width": 16, "macro_family": family})
            steps.append(load_step)
            return steps

        if family in {"CONST_U32_IMM", "CONST_U32_5E_IMM"}:
            imm_step = _signature_const_step(family, operands.get("imm"), suffix="imm")
            if imm_step is not None:
                steps.append(imm_step)
            return steps or [(node.semantic_op, node.semantic_op, dict(operands))]

        if family == "CONST_U32_IMM16":
            imm_step = _signature_const_step(family, operands.get("imm16"), suffix="imm16")
            if imm_step is not None:
                steps.append(imm_step)
            return steps or [(node.semantic_op, node.semantic_op, dict(operands))]

        if family in {"CONST_U32_CALL66", "CONST_U32_26_CALL66"}:
            steps.append(("syscall", "syscall", {"family": family, "argc": operands.get("argc", 0), "opid": operands.get("opid")}))
            return steps

        if family == "CONST_U32_CALL63A":
            steps.append(("call", "call", {"family": family, "argc": operands.get("argc", 0), "rel": operands.get("rel")}))
            return steps

        if family in {"CONST_U32_REC41", "CONST_U32_REC62"}:
            record_operands = {key: value for key, value in operands.items() if key not in {"family", "value", "prefix_chain"}}
            record_operands["record_kind"] = "REC41" if family.endswith("REC41") else "REC62"
            steps.append(("make_record", "make_record", record_operands))
            return steps

        if family == "CONST_U32_PFX72":
            nested_steps = _signature_expand_nested(family, operands.get("nested_kind"), operands.get("nested"))
            if nested_steps:
                steps.extend(nested_steps)
                return steps

        if family == "CONST_U32_PFX_3D_PAIR72_23":
            steps.append(_signature_return_step(family, operands, tail_form="pair72_23"))
            return steps

        if family == "CONST_0100":
            if not steps:
                steps.append(("const", "const", {"value": 0x100, "family": family}))
            return steps

        if family == "CONST_U32_TRAILER":
            return steps or [("const", "const", {"value": operands.get("value"), "family": family})]

        if steps:
            return steps

    if node.semantic_op == "return":
        return [("return", "return", {"family": family})]
    if node.semantic_op == "branch":
        return [("branch", "branch", dict(operands))]
    if node.semantic_op in {"call", "syscall", "opaque_call", "const", "load", "read_field", "make_record", "aggregate", "store", "write_field", "cmp"}:
        return [(node.semantic_op, node.semantic_op, dict(operands))]
    if node.semantic_op.startswith("opaque_"):
        return [(node.semantic_op, node.semantic_op, dict(operands))]
    return [(node.semantic_op, node.semantic_op, dict(operands))]


def lower_ir_nodes(nodes: list[IRNode]) -> tuple[list[CanonicalInstruction], dict[int, list[int]]]:
    instructions: list[CanonicalInstruction] = []
    node_to_instruction_indices: dict[int, list[int]] = defaultdict(list)
    inst_index = 0
    for node in nodes:
        if node.raw_kind.startswith("SIG_") or node.terminal_kind.startswith("SIG_"):
            steps = _signature_to_steps(node)
            for semantic_op, display_op, operands in steps:
                inst = _make_inst(
                    inst_index=inst_index,
                    node=node,
                    semantic_op=semantic_op,
                    display_op=display_op,
                    operands=dict(operands),
                    macro_kind=node.raw_kind if node.raw_kind.startswith("SIG_") else node.terminal_kind,
                )
                instructions.append(inst)
                node_to_instruction_indices[node.index].append(inst_index)
                inst_index += 1
            continue

        inst = _make_inst(
            inst_index=inst_index,
            node=node,
            semantic_op=node.semantic_op,
            display_op=node.opcode,
            operands=dict(node.operands),
            macro_kind=None,
        )
        instructions.append(inst)
        node_to_instruction_indices[node.index].append(inst_index)
        inst_index += 1
    return instructions, node_to_instruction_indices


# --- canonical CFG --------------------------------------------------------


def _build_instruction_control(instructions: list[CanonicalInstruction], nodes: list[IRNode]) -> None:
    origin_by_index = {node.index: node for node in nodes}
    offset_to_first_inst: dict[int, int] = {}
    for inst in instructions:
        offset_to_first_inst.setdefault(inst.offset, inst.index)

    for inst in instructions:
        origin = origin_by_index[inst.origin_node_index]
        control = {
            "kind": None,
            "is_terminator": False,
            "fallthrough": True,
            "target_offset": None,
            "target_instruction_index": None,
        }
        if inst.semantic_op == "return":
            control.update({"kind": "return", "is_terminator": True, "fallthrough": False})
        elif inst.semantic_op == "branch":
            target_offset = inst.operands.get("target_offset")
            if target_offset is None:
                target_offset = origin.control.get("target_offset")
            control.update(
                {
                    "kind": "branch",
                    "is_terminator": True,
                    "fallthrough": True,
                    "target_offset": target_offset,
                    "target_instruction_index": offset_to_first_inst.get(target_offset),
                    "branch_op": inst.operands.get("branch_op") or origin.operands.get("branch_op"),
                    "resolved": target_offset in offset_to_first_inst,
                }
            )
        elif inst.semantic_op in {"call", "syscall"} and inst.operands.get("rel") is not None:
            target_offset = origin.control.get("resolved_target_offset")
            control.update(
                {
                    "kind": "call",
                    "resolved_target_offset": target_offset,
                    "resolved_target_instruction_index": offset_to_first_inst.get(target_offset),
                    "fallthrough": True,
                }
            )
        inst.control = control


def _compute_canonical_blocks(instructions: list[CanonicalInstruction]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not instructions:
        cfg = {
            "entry_block": None,
            "blocks": [],
            "edges": [],
            "anomalies": [],
            "stats": {"block_count": 0, "edge_count": 0, "unreachable_blocks": 0, "unresolved_targets": 0},
        }
        return [], cfg

    leaders: set[int] = {0}
    for idx, inst in enumerate(instructions):
        ctrl = inst.control
        if ctrl.get("is_terminator"):
            target_index = ctrl.get("target_instruction_index")
            if target_index is not None:
                leaders.add(target_index)
            if ctrl.get("fallthrough") and idx + 1 < len(instructions):
                leaders.add(idx + 1)

    sorted_leaders = sorted(leaders)
    blocks: list[dict[str, Any]] = []
    inst_to_block: dict[int, str] = {}
    for block_idx, start_idx in enumerate(sorted_leaders):
        end_idx = sorted_leaders[block_idx + 1] - 1 if block_idx + 1 < len(sorted_leaders) else len(instructions) - 1
        block_id = f"bb{block_idx}"
        inst_indices = list(range(start_idx, end_idx + 1))
        for inst_index in inst_indices:
            inst_to_block[inst_index] = block_id
        blocks.append(
            {
                "id": block_id,
                "index": block_idx,
                "start_offset": instructions[start_idx].offset,
                "end_offset": instructions[end_idx].offset + instructions[end_idx].size,
                "instruction_indices": inst_indices,
                "terminator_index": inst_indices[-1],
                "terminator_op": instructions[inst_indices[-1]].semantic_op,
                "successors": [],
                "predecessors": [],
                "flags": [],
            }
        )

    block_index = {block["id"]: idx for idx, block in enumerate(blocks)}
    edges: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    def add_edge(src: str, dst: Optional[str], kind: str, target_offset: Optional[int] = None) -> None:
        edge = {"src": src, "dst": dst, "kind": kind}
        if target_offset is not None:
            edge["target_offset"] = target_offset
        edges.append(edge)
        if dst is None:
            return
        src_block = blocks[block_index[src]]
        if dst not in src_block["successors"]:
            src_block["successors"].append(dst)
        dst_block = blocks[block_index[dst]]
        if src not in dst_block["predecessors"]:
            dst_block["predecessors"].append(src)

    for idx, block in enumerate(blocks):
        terminator = instructions[block["terminator_index"]]
        ctrl = terminator.control
        next_block_id = blocks[idx + 1]["id"] if idx + 1 < len(blocks) else None

        if ctrl.get("kind") == "return":
            block["flags"].append("returns")
            continue

        if ctrl.get("kind") == "branch":
            target_inst = ctrl.get("target_instruction_index")
            target_block = inst_to_block.get(target_inst) if target_inst is not None else None
            if target_block is not None:
                add_edge(block["id"], target_block, "branch", ctrl.get("target_offset"))
            else:
                anomalies.append(
                    {
                        "kind": "unresolved_branch_target",
                        "block_id": block["id"],
                        "instruction_index": terminator.index,
                        "offset": terminator.offset,
                        "target_offset": ctrl.get("target_offset"),
                    }
                )
                block["flags"].append("unresolved_branch_target")
                add_edge(block["id"], None, "branch_unresolved", ctrl.get("target_offset"))
            if next_block_id is not None and ctrl.get("fallthrough"):
                add_edge(block["id"], next_block_id, "fallthrough")
            elif next_block_id is None:
                block["flags"].append("open_exit")
            continue

        if next_block_id is not None:
            add_edge(block["id"], next_block_id, "fallthrough")
        else:
            block["flags"].append("open_exit")

    reachable: set[str] = set()
    if blocks:
        queue = deque([blocks[0]["id"]])
        while queue:
            block_id = queue.popleft()
            if block_id in reachable:
                continue
            reachable.add(block_id)
            for succ in blocks[block_index[block_id]]["successors"]:
                if succ not in reachable:
                    queue.append(succ)

    for block in blocks:
        if block["id"] not in reachable:
            block["flags"].append("unreachable")
            anomalies.append({"kind": "unreachable_block", "block_id": block["id"]})

    cfg = {
        "entry_block": blocks[0]["id"],
        "blocks": [dict(block) for block in blocks],
        "edges": edges,
        "anomalies": anomalies,
        "stats": {
            "block_count": len(blocks),
            "edge_count": len(edges),
            "unreachable_blocks": sum(1 for block in blocks if "unreachable" in block["flags"]),
            "unresolved_targets": sum(1 for item in anomalies if item["kind"] == "unresolved_branch_target"),
        },
    }
    return blocks, cfg


# --- dataflow -------------------------------------------------------------


def _compute_stack_plans(blocks: list[dict[str, Any]], instructions: list[CanonicalInstruction]) -> dict[str, _StackPlanBlock]:
    if not blocks:
        return {}
    block_index = {block["id"]: block["index"] for block in blocks}
    entry_depth: dict[str, int] = {block["id"]: 0 for block in blocks}
    exit_depth: dict[str, int] = {block["id"]: 0 for block in blocks}

    # Use a forward-edge requirement pass. Back-edges are intentionally ignored here;
    # otherwise any mis-modeled positive stack effect inside a loop explodes into
    # fake function arguments and phi towers.
    for _ in range(max(4, len(blocks) * 4)):
        changed = False
        for block in reversed(blocks):
            forward_succs = [succ for succ in block["successors"] if block_index.get(succ, 10 ** 6) > block["index"]]
            succ_depths = [entry_depth[succ] for succ in forward_succs if succ in entry_depth]
            new_exit = max(succ_depths) if succ_depths else 0
            needed = new_exit
            for inst_index in reversed(block["instruction_indices"]):
                inst = instructions[inst_index]
                uses_output = inst.stack_outputs > 0 and needed > 0
                needed = needed - (1 if uses_output else 0) + inst.stack_inputs_required
            if new_exit != exit_depth[block["id"]] or needed != entry_depth[block["id"]]:
                exit_depth[block["id"]] = new_exit
                entry_depth[block["id"]] = needed
                changed = True
        if not changed:
            break

    plans: dict[str, _StackPlanBlock] = {}
    for block in blocks:
        needed = exit_depth[block["id"]]
        output_used: dict[int, bool] = {}
        for inst_index in reversed(block["instruction_indices"]):
            inst = instructions[inst_index]
            uses_output = inst.stack_outputs > 0 and needed > 0
            output_used[inst_index] = uses_output
            needed = needed - (1 if uses_output else 0) + inst.stack_inputs_required
        plans[block["id"]] = _StackPlanBlock(entry_depth=entry_depth[block["id"]], exit_depth=exit_depth[block["id"]], output_used=output_used)
    return plans


def _pop_args(stack: list[str], required: int, block_id: str, inst_index: int) -> list[str]:
    if required <= 0:
        return []
    available = min(required, len(stack))
    taken = stack[-available:] if available else []
    if available:
        del stack[-available:]
    missing = required - available
    if missing <= 0:
        return taken
    prefix = [f"undef_{block_id}_{inst_index}_{i}" for i in range(missing)]
    return prefix + taken


def _simulate_block_stack(block: dict[str, Any], instructions: list[CanonicalInstruction], plan: _StackPlanBlock, entry_stack: list[str]) -> tuple[list[str], dict[int, list[str]], dict[int, list[str]]]:
    stack = list(entry_stack)
    arg_map: dict[int, list[str]] = {}
    out_map: dict[int, list[str]] = {}
    for inst_index in block["instruction_indices"]:
        inst = instructions[inst_index]
        args = _pop_args(stack, inst.stack_inputs_required, block["id"], inst.index)
        arg_map[inst_index] = list(args)
        outputs: list[str] = []
        if inst.stack_outputs > 0 and plan.output_used.get(inst_index, False):
            out_name = f"t{inst.index}"
            stack.append(out_name)
            outputs = [out_name]
        out_map[inst_index] = outputs
        if inst.control.get("is_terminator"):
            break
    exit_depth = plan.exit_depth
    if exit_depth <= 0:
        return [], arg_map, out_map
    if len(stack) >= exit_depth:
        return list(stack[-exit_depth:]), arg_map, out_map
    padding = [f"in_{block['id']}_{i}" for i in range(exit_depth - len(stack))]
    return padding + stack, arg_map, out_map


def _compose_entry_stack(
    block_id: str,
    depth: int,
    incoming_by_pred: dict[str, list[str]],
    pred_order: dict[str, int],
    is_entry: bool,
    entry_seed: Optional[list[str]] = None,
) -> tuple[list[str], dict[str, list[str]]]:
    effective_incoming = dict(incoming_by_pred)
    if is_entry and entry_seed is not None:
        effective_incoming = {"__entry__": list(entry_seed[-depth:]) if depth else [], **effective_incoming}

    if is_entry and not effective_incoming:
        return [f"arg{i}" for i in range(depth)], {}
    if not effective_incoming:
        return [f"in_{block_id}_{i}" for i in range(depth)], {}

    entry: list[str] = []
    phi_defs: dict[str, list[str]] = {}
    ordered_items = sorted(effective_incoming.items(), key=lambda item: -1 if item[0] == "__entry__" else pred_order.get(item[0], 10 ** 6))
    for pos in range(depth):
        values: list[str] = []
        for pred_id, stack in ordered_items:
            if pos < len(stack):
                values.append(stack[pos])
            elif is_entry and pred_id == "__entry__":
                values.append(f"arg{pos}")
            else:
                values.append(f"in_{block_id}_{pos}")
        phi_name = f"phi_{block_id}_{pos}"
        unique = _ordered_unique([value for value in values if value != phi_name] or values)
        if len(unique) == 1:
            entry.append(unique[0])
        else:
            entry.append(phi_name)
            phi_defs[phi_name] = unique
    return entry, phi_defs


def build_hir_blocks(
    blocks: list[dict[str, Any]],
    cfg: dict[str, Any],
    instructions: list[CanonicalInstruction],
    entry_seed: Optional[list[str]] = None,
) -> tuple[list[HIRBlock], list[DataflowValue], dict[str, int]]:
    plans = _compute_stack_plans(blocks, instructions)
    if not blocks:
        return [], [], {"placeholder_count": 0, "phi_count": 0}

    by_id = {block["id"]: block for block in blocks}
    pred_order = {block["id"]: block["index"] for block in blocks}
    edge_kinds: dict[str, dict[str, Optional[str]]] = defaultdict(lambda: {"branch": None, "fallthrough": None})
    for edge in cfg.get("edges", []):
        src = edge.get("src")
        kind = edge.get("kind")
        if kind == "branch":
            edge_kinds[src]["branch"] = edge.get("dst")
        elif kind == "fallthrough":
            edge_kinds[src]["fallthrough"] = edge.get("dst")
    entry_block_id = cfg.get("entry_block")

    incoming: dict[str, dict[str, list[str]]] = defaultdict(dict)
    entry_stacks: dict[str, list[str]] = {}
    phi_defs: dict[str, dict[str, list[str]]] = defaultdict(dict)
    exit_stacks: dict[str, list[str]] = {}
    work = deque([entry_block_id] if entry_block_id else [])

    entry_plan = plans.get(entry_block_id) if entry_block_id else None
    if entry_block_id and entry_plan is not None:
        if entry_seed:
            entry_stacks[entry_block_id] = list(entry_seed[-entry_plan.entry_depth:]) if entry_plan.entry_depth else []
        else:
            entry_stacks[entry_block_id] = [f"arg{i}" for i in range(entry_plan.entry_depth)]

    while work:
        block_id = work.popleft()
        block = by_id[block_id]
        plan = plans[block_id]
        current_entry, current_phi = _compose_entry_stack(
            block_id,
            plan.entry_depth,
            incoming[block_id],
            pred_order,
            block_id == entry_block_id,
            entry_seed=entry_seed if block_id == entry_block_id else None,
        )
        entry_changed = entry_stacks.get(block_id) != current_entry
        if entry_changed:
            entry_stacks[block_id] = current_entry
            for pred in block.get("predecessors", []):
                if pred_order.get(pred, -1) >= pred_order.get(block_id, 10 ** 6):
                    work.append(pred)
        phi_defs[block_id] = current_phi

        exit_stack, _, _ = _simulate_block_stack(block, instructions, plan, entry_stacks[block_id])
        if exit_stacks.get(block_id) == exit_stack:
            pass
        else:
            exit_stacks[block_id] = exit_stack
        for succ in block["successors"]:
            succ_depth = plans[succ].entry_depth
            propagated = list(exit_stack[-succ_depth:]) if succ_depth else []
            if len(propagated) < succ_depth:
                missing = succ_depth - len(propagated)
                padding: list[str] = []
                if succ == block_id and entry_stacks.get(block_id):
                    loop_tail = list(entry_stacks[block_id][-succ_depth:]) if succ_depth else []
                    padding.extend(loop_tail[:missing])
                elif pred_order.get(succ, 10 ** 6) <= pred_order.get(block_id, -1) and entry_stacks.get(succ):
                    carry_tail = list(entry_stacks[succ][-succ_depth:]) if succ_depth else []
                    padding.extend(carry_tail[:missing])
                elif succ == entry_block_id and entry_seed:
                    seed_tail = list(entry_seed[-succ_depth:]) if succ_depth else []
                    padding.extend(seed_tail[:missing])
                if len(padding) < missing:
                    padding.extend(f"in_{succ}_{i}" for i in range(len(padding), missing))
                propagated = padding + propagated
            if incoming[succ].get(block_id) != propagated:
                incoming[succ][block_id] = propagated
                work.append(succ)

    for block in blocks:
        block_id = block["id"]
        if block_id in entry_stacks:
            continue
        plan = plans[block_id]
        entry_stacks[block_id], phi_defs[block_id] = _compose_entry_stack(
            block_id,
            plan.entry_depth,
            incoming[block_id],
            pred_order,
            False,
            entry_seed=None,
        )
        exit_stacks[block_id], _, _ = _simulate_block_stack(block, instructions, plan, entry_stacks[block_id])

    temp_uses: Counter[str] = Counter()
    value_map: dict[str, DataflowValue] = {}
    hir_blocks: list[HIRBlock] = []
    placeholder_count = 0

    for block in blocks:
        block_id = block["id"]
        plan = plans[block_id]
        entry_stack = entry_stacks.get(block_id, [])
        exit_stack, arg_map, out_map = _simulate_block_stack(block, instructions, plan, entry_stack)

        phi_bindings: list[str] = []
        for name, sources in sorted(phi_defs.get(block_id, {}).items()):
            phi_bindings.append(f"{name} = phi({', '.join(sources)})")
            value_map[name] = DataflowValue(name=name, producer_instruction=-1, expr=f"phi({', '.join(sources)})")
            for source in sources:
                if source.startswith(("t", "phi_", "arg", "in_", "undef_")):
                    temp_uses[source] += 1
                    if _is_placeholder_value(source):
                        placeholder_count += 1

        param_positions = [pos for pos, name in enumerate(entry_stack) if name in phi_defs.get(block_id, {})]
        block_params = [entry_stack[pos] for pos in param_positions]
        incoming_args_map = {
            pred_id: [stack[pos] if pos < len(stack) else f"in_{block_id}_{pos}" for pos in param_positions]
            for pred_id, stack in incoming.get(block_id, {}).items()
        }

        statements: list[str] = []
        terminator: dict[str, Any] = {"kind": "fallthrough", "text": None, "condition": None}

        for inst_index in block["instruction_indices"]:
            inst = instructions[inst_index]
            args = list(arg_map.get(inst_index, []))
            outputs = list(out_map.get(inst_index, []))
            inst.inputs = args
            inst.outputs = outputs
            for arg in args:
                if arg.startswith(("t", "phi_", "arg", "in_", "undef_")):
                    temp_uses[arg] += 1
                    if _is_placeholder_value(arg):
                        placeholder_count += 1
            expr = _canonical_expr(inst, args)

            if inst.semantic_op in {"const", "load", "read_field", "make_record", "aggregate", "call", "syscall", "opaque_call", "cmp", "opaque_const", "opaque_load", "opaque_record", "opaque_aggregate", "opaque_op"}:
                if outputs:
                    out_name = outputs[0]
                    value_map[out_name] = DataflowValue(name=out_name, producer_instruction=inst.index, expr=expr)
                    statements.append(f"{out_name} = {expr}")
                elif inst.effects or inst.semantic_op in {"call", "syscall", "opaque_call"}:
                    statements.append(expr)
                continue

            if inst.semantic_op in {"store", "write_field", "opaque_store"}:
                target = args[0] if len(args) >= 1 else _slot_name(inst.operands.get("ref"), inst.operands.get("mode"))
                value = args[1] if len(args) >= 2 else "<?>"
                if inst.semantic_op == "write_field":
                    field_name = inst.operands.get("family") or inst.operands.get("field") or "field"
                    statements.append(f"{target}.{str(field_name).lower()} = {value}")
                else:
                    statements.append(f"{target} = {value}")
                continue

            if inst.semantic_op == "return":
                value = args[-1] if args else None
                terminator = {
                    "kind": "return",
                    "text": f"return {value}" if value is not None else "return",
                    "condition": None,
                    "instruction_index": inst.index,
                }
                statements.append(terminator["text"])
                break

            if inst.semantic_op == "branch":
                if len(args) >= 2:
                    cond_value = f"cmp({', '.join(args)})"
                elif args:
                    cond_value = args[-1]
                else:
                    cond_value = f"cond_{block_id}"
                branch_op = inst.control.get("branch_op") or inst.operands.get("branch_op") or "0x??"
                cond_text = f"cond[{branch_op}]({cond_value})"
                terminator = {
                    "kind": "branch",
                    "text": cond_text,
                    "condition": cond_text,
                    "branch_op": branch_op,
                    "instruction_index": inst.index,
                }
                break

            if inst.semantic_op == "data":
                statements.append(f"data({json.dumps(inst.operands, ensure_ascii=False)})")
                continue

        hir_blocks.append(
            HIRBlock(
                id=block_id,
                index=block["index"],
                start_offset=block["start_offset"],
                end_offset=block["end_offset"],
                instruction_indices=list(block["instruction_indices"]),
                entry_stack=list(entry_stack),
                exit_stack=list(exit_stack),
                phi_bindings=phi_bindings,
                block_params=block_params,
                incoming_args=incoming_args_map,
                statements=statements,
                terminator=terminator,
                branch_target=edge_kinds[block_id].get("branch"),
                fallthrough_target=edge_kinds[block_id].get("fallthrough"),
                successors=list(block["successors"]),
                predecessors=list(block["predecessors"]),
                flags=list(block["flags"]),
            )
        )

    for value in value_map.values():
        value.use_count = temp_uses.get(value.name, 0)

    dataflow_values = sorted(value_map.values(), key=lambda item: (item.producer_instruction < 0, item.producer_instruction, item.name))
    metrics = {
        "placeholder_count": placeholder_count,
        "phi_count": sum(1 for name in value_map if name.startswith("phi_")),
    }
    return hir_blocks, dataflow_values, metrics


def _replace_var(text: Optional[str], name: str, replacement: str) -> Optional[str]:
    if text is None or name not in text:
        return text
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    return re.sub(pattern, replacement, text)


def _substitute_block_values(block: HIRBlock, mapping: dict[str, str], *, replace_block_params: bool = False) -> HIRBlock:
    incoming_args = {
        pred: [_replace_vars(arg, mapping) or arg for arg in args]
        for pred, args in block.incoming_args.items()
    }
    statements = [_replace_vars(stmt, mapping) or stmt for stmt in block.statements]
    terminator = dict(block.terminator)
    terminator["text"] = _replace_vars(terminator.get("text"), mapping)
    terminator["condition"] = _replace_vars(terminator.get("condition"), mapping)
    updates: dict[str, Any] = {
        "entry_stack": [_replace_vars(item, mapping) or item for item in block.entry_stack],
        "exit_stack": [_replace_vars(item, mapping) or item for item in block.exit_stack],
        "phi_bindings": [_replace_vars(item, mapping) or item for item in block.phi_bindings],
        "incoming_args": incoming_args,
        "statements": statements,
        "terminator": terminator,
    }
    if replace_block_params:
        updates["block_params"] = [mapping.get(item, item) for item in block.block_params]
    return _clone_block(block, **updates)


def _substitute_blocks(hir_blocks: list[HIRBlock], mapping: dict[str, str], *, replace_block_params: bool = False) -> list[HIRBlock]:
    if not mapping:
        return list(hir_blocks)
    return [_substitute_block_values(block, mapping, replace_block_params=replace_block_params) for block in hir_blocks]


def _find_assignment_expr(hir_blocks: list[HIRBlock], name: str) -> Optional[str]:
    prefix = f"{name} = "
    for block in hir_blocks:
        for stmt in block.statements:
            if stmt.startswith(prefix):
                return stmt[len(prefix):]
    return None


def _drop_assignment_statement(block: HIRBlock, name: str) -> HIRBlock:
    prefix = f"{name} = "
    kept = [stmt for stmt in block.statements if not stmt.startswith(prefix)]
    if len(kept) == len(block.statements):
        return block
    return _clone_block(block, statements=kept)


def _substitute_text_map(text: Optional[str], mapping: dict[str, str]) -> Optional[str]:
    if text is None or not mapping:
        return text
    refs = _iter_temp_like_refs(text)
    if not refs:
        return text
    seen = set(refs)
    if not any(ref in mapping for ref in seen):
        return text
    return _TEMP_LIKE_RE.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def _expand_inline_expr(name: str, inline_candidates: dict[str, str], *, max_len: int = 96, cache: Optional[dict[str, str]] = None, active: Optional[set[str]] = None) -> Optional[str]:
    if cache is None:
        cache = {}
    if active is None:
        active = set()
    if name in cache:
        return cache[name]
    expr = inline_candidates.get(name)
    if expr is None:
        return None
    if len(expr) > max_len or name in expr:
        return None
    if name in active:
        return None
    active.add(name)
    mapping: dict[str, str] = {}
    for ref in _ordered_unique(_iter_temp_like_refs(expr)):
        if ref == name or ref not in inline_candidates:
            continue
        replacement = _expand_inline_expr(ref, inline_candidates, max_len=max_len, cache=cache, active=active)
        if replacement is None:
            continue
        candidate = _replace_var(expr, ref, replacement)
        if candidate is None or len(candidate) > max_len:
            continue
        expr = candidate
        mapping[ref] = replacement
    active.remove(name)
    if len(expr) > max_len or name in expr:
        return None
    cache[name] = expr
    return expr


def _collect_surface_assignments(blocks: list[HIRBlock]) -> tuple[dict[str, str], dict[str, str]]:
    expr_by_name: dict[str, str] = {}
    owner_by_name: dict[str, str] = {}
    for block in blocks:
        for stmt in block.statements:
            matched = _match_assignment(stmt)
            if matched is None:
                continue
            name, expr = matched
            if name in expr_by_name:
                continue
            expr_by_name[name] = expr
            owner_by_name[name] = block.id
    return expr_by_name, owner_by_name


def _cleanup_surface_hir(
    hir_blocks: list[HIRBlock],
    canonical: list[CanonicalInstruction],
    values: list[DataflowValue],
) -> tuple[list[HIRBlock], list[DataflowValue], dict[str, int]]:
    producer_semantic: dict[str, str] = {}
    for inst in canonical:
        for out in inst.outputs:
            producer_semantic[out] = inst.semantic_op

    def is_inlinable(value: DataflowValue, current_expr: Optional[str]) -> bool:
        if not value.name.startswith("t"):
            return False
        if value.use_count != 1:
            return False
        if current_expr is None or current_expr == value.name or value.name in current_expr:
            return False
        semantic = producer_semantic.get(value.name)
        if semantic not in {"const", "load", "read_field", "make_record", "aggregate", "cmp", "opaque_const", "opaque_load", "opaque_record", "opaque_aggregate"}:
            return False
        if len(current_expr) > 96:
            return False
        return True

    blocks = list(hir_blocks)
    expr_by_name, _ = _collect_surface_assignments(blocks)
    inline_candidates = {
        value.name: expr_by_name[value.name]
        for value in sorted(values, key=lambda item: (item.producer_instruction < 0, item.producer_instruction, item.name))
        if value.name in expr_by_name and is_inlinable(value, expr_by_name[value.name])
    }
    expanded_cache: dict[str, str] = {}
    inline_map = {
        name: expanded
        for name in inline_candidates
        if (expanded := _expand_inline_expr(name, inline_candidates, cache=expanded_cache)) is not None
    }
    if not inline_map:
        return blocks, values, {"inlined_value_count": 0, "removed_assignment_count": 0}

    inlined_count = 0
    removed_assignments = 0
    cleaned: list[HIRBlock] = []
    for block in blocks:
        statements: list[str] = []
        dropped = 0
        for stmt in block.statements:
            matched = _match_assignment(stmt)
            if matched is not None and matched[0] in inline_map:
                dropped += 1
                continue
            statements.append(_substitute_text_map(stmt, inline_map) or stmt)
        if dropped:
            removed_assignments += dropped
        incoming_args = {
            pred: [_substitute_text_map(arg, inline_map) or arg for arg in args]
            for pred, args in block.incoming_args.items()
        }
        terminator = dict(block.terminator)
        terminator["text"] = _substitute_text_map(terminator.get("text"), inline_map)
        terminator["condition"] = _substitute_text_map(terminator.get("condition"), inline_map)
        cleaned.append(
            _clone_block(
                block,
                entry_stack=[_substitute_text_map(item, inline_map) or item for item in block.entry_stack],
                exit_stack=[_substitute_text_map(item, inline_map) or item for item in block.exit_stack],
                phi_bindings=[_substitute_text_map(item, inline_map) or item for item in block.phi_bindings],
                incoming_args=incoming_args,
                statements=statements,
                terminator=terminator,
            )
        )

    blocks = cleaned
    inlined_count = len(inline_map)

    debug = {"inlined_value_count": inlined_count, "removed_assignment_count": removed_assignments}
    return blocks, values, debug




# --- structuring ----------------------------------------------------------


def _clone_block(block: HIRBlock, **updates: Any) -> HIRBlock:
    payload = {
        "id": block.id,
        "index": block.index,
        "start_offset": block.start_offset,
        "end_offset": block.end_offset,
        "instruction_indices": list(block.instruction_indices),
        "entry_stack": list(block.entry_stack),
        "exit_stack": list(block.exit_stack),
        "phi_bindings": list(block.phi_bindings),
        "block_params": list(block.block_params),
        "incoming_args": {pred: list(args) for pred, args in block.incoming_args.items()},
        "statements": list(block.statements),
        "terminator": dict(block.terminator),
        "branch_target": block.branch_target,
        "fallthrough_target": block.fallthrough_target,
        "successors": list(block.successors),
        "predecessors": list(block.predecessors),
        "flags": list(block.flags),
    }
    payload.update(updates)
    return HIRBlock(**payload)


def _block_maps(hir_blocks: list[HIRBlock]) -> tuple[dict[str, HIRBlock], dict[str, int], dict[str, list[str]], dict[str, list[str]]]:
    by_id = {block.id: block for block in hir_blocks}
    index_by_id = {block.id: block.index for block in hir_blocks}
    succs = {block.id: list(block.successors) for block in hir_blocks}
    preds = {block.id: list(block.predecessors) for block in hir_blocks}
    return by_id, index_by_id, succs, preds


def _reindex_hir_blocks(hir_blocks: list[HIRBlock]) -> list[HIRBlock]:
    return [_clone_block(block, index=index) for index, block in enumerate(hir_blocks)]


def _sanitize_hir_blocks(hir_blocks: list[HIRBlock]) -> list[HIRBlock]:
    valid_ids = {block.id for block in hir_blocks}

    def _clean_edge_list(items: list[str]) -> list[str]:
        return [item for item in _ordered_unique(items) if item in valid_ids]

    cleaned = [
        _clone_block(
            block,
            branch_target=block.branch_target if block.branch_target in valid_ids else None,
            fallthrough_target=block.fallthrough_target if block.fallthrough_target in valid_ids else None,
            successors=_clean_edge_list(list(block.successors)),
            predecessors=[],
        )
        for block in hir_blocks
    ]
    predecessor_map: dict[str, list[str]] = defaultdict(list)
    for block in cleaned:
        for succ in block.successors:
            predecessor_map[succ].append(block.id)
    return _reindex_hir_blocks(
        [
            _clone_block(block, predecessors=_ordered_unique(predecessor_map.get(block.id, [])))
            for block in cleaned
        ]
    )


def _collect_hir_defs(blocks: list[HIRBlock]) -> set[str]:
    defs: set[str] = set()
    if blocks:
        defs.update(item for item in blocks[0].entry_stack if isinstance(item, str) and item.startswith("arg"))
    for block in blocks:
        defs.update(item for item in block.block_params if isinstance(item, str))
        for binding in block.phi_bindings:
            matched = _match_assignment(binding)
            if matched is not None:
                defs.add(matched[0])
        for stmt in block.statements:
            matched = _match_assignment(stmt)
            if matched is not None:
                defs.add(matched[0])
    return defs


def validate_hir_blocks(blocks: list[HIRBlock], values: list[DataflowValue], stage: str) -> dict[str, Any]:
    valid_ids = {block.id for block in blocks}
    _, _, succs, preds = _block_maps(blocks) if blocks else ({}, {}, {}, {})
    known_defs = _collect_hir_defs(blocks)
    value_names = {value.name for value in values}
    known_defs.update(name for name in value_names if name.startswith(("arg", "phi_", "m_", "t")))
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def push(kind: str, block_id: Optional[str], detail: str) -> None:
        counts[kind] += 1
        if len(errors) < 16:
            item = {"kind": kind, "detail": detail}
            if block_id is not None:
                item["block_id"] = block_id
            errors.append(item)

    seen_ids: set[str] = set()
    for expected_index, block in enumerate(blocks):
        if block.id in seen_ids:
            push("duplicate_block_id", block.id, f"duplicate block id {block.id}")
        seen_ids.add(block.id)
        if block.index != expected_index:
            push("non_contiguous_index", block.id, f"expected index {expected_index}, got {block.index}")
        if block.branch_target is not None and block.branch_target not in valid_ids:
            push("dangling_branch_target", block.id, f"unknown branch target {block.branch_target}")
        if block.fallthrough_target is not None and block.fallthrough_target not in valid_ids:
            push("dangling_fallthrough_target", block.id, f"unknown fallthrough target {block.fallthrough_target}")
        for succ in block.successors:
            if succ not in valid_ids:
                push("dangling_successor", block.id, f"unknown successor {succ}")
                continue
            if block.id not in preds.get(succ, []):
                push("cfg_asymmetry", block.id, f"{succ} missing predecessor {block.id}")
        for pred in block.predecessors:
            if pred not in valid_ids:
                push("dangling_predecessor", block.id, f"unknown predecessor {pred}")
                continue
            if block.id not in succs.get(pred, []):
                push("cfg_asymmetry", block.id, f"{pred} missing successor {block.id}")
        if set(block.incoming_args.keys()) - set(block.predecessors):
            extra = sorted(set(block.incoming_args.keys()) - set(block.predecessors))
            push("incoming_without_pred", block.id, f"incoming args for non-predecessors: {extra}")
        if block.block_params:
            missing = [pred for pred in block.predecessors if pred not in block.incoming_args]
            if missing:
                push("missing_incoming_args", block.id, f"missing incoming args for predecessors: {missing}")
        for pred, args in block.incoming_args.items():
            if len(args) != len(block.block_params):
                push("incoming_arity_mismatch", block.id, f"{pred} arity {len(args)} != params {len(block.block_params)}")

    def check_text_refs(block: HIRBlock, text: Optional[str], *, label: str, strip_lhs: bool = False) -> None:
        if not text:
            return
        source = text
        if strip_lhs:
            matched = _match_assignment(text)
            if matched is not None:
                source = matched[1]
        for ref in _iter_temp_like_refs(source):
            if ref in known_defs:
                continue
            if ref.startswith(("arg", "in_", "undef_")):
                continue
            push("dangling_ref", block.id, f"{label} references undefined {ref}")

    for block in blocks:
        for item in block.entry_stack:
            check_text_refs(block, item, label="entry_stack")
        for item in block.exit_stack:
            check_text_refs(block, item, label="exit_stack")
        for item in block.phi_bindings:
            check_text_refs(block, item, label="phi_binding", strip_lhs=True)
        for pred, args in block.incoming_args.items():
            for arg in args:
                check_text_refs(block, arg, label=f"incoming[{pred}]")
        for stmt in block.statements:
            check_text_refs(block, stmt, label="statement", strip_lhs=True)
        check_text_refs(block, block.terminator.get("text"), label="terminator")
        check_text_refs(block, block.terminator.get("condition"), label="condition")

    return {
        "stage": stage,
        "ok": not errors,
        "error_count": sum(counts.values()),
        "kind_histogram": dict(counts),
        "samples": errors,
    }


def _compose_forwarded_incoming_args(block: HIRBlock, succ: HIRBlock) -> Optional[dict[str, list[str]]]:
    forwarded_args = list(succ.incoming_args.get(block.id, []))
    param_names = list(block.block_params)
    incoming = {pred: list(args) for pred, args in succ.incoming_args.items() if pred != block.id}
    for pred_id in block.predecessors:
        if pred_id == succ.id or pred_id in incoming:
            return None
        pred_args = list(block.incoming_args.get(pred_id, []))
        if param_names and len(pred_args) != len(param_names):
            return None
        composed: list[str] = []
        for arg in forwarded_args:
            rewritten = arg
            for idx, param_name in enumerate(param_names):
                replacement = pred_args[idx] if idx < len(pred_args) else param_name
                rewritten = _replace_var(rewritten, param_name, replacement) or rewritten
            composed.append(rewritten)
        incoming[pred_id] = composed
    return incoming


def _rewrite_empty_fallthrough(blocks: list[HIRBlock]) -> Optional[list[HIRBlock]]:
    by_id = {block.id: block for block in blocks}
    for block in blocks:
        if block.index == 0:
            continue
        if block.statements or block.phi_bindings or block.block_params:
            continue
        if block.entry_stack or block.exit_stack:
            continue
        if any(args for args in block.incoming_args.values()):
            continue
        if block.terminator.get("kind") != "fallthrough":
            continue
        if len(block.successors) != 1:
            continue
        succ_id = block.successors[0]
        succ = by_id.get(succ_id)
        if succ is None or succ.id == block.id:
            continue

        forwarded_incoming = {pred: list(args) for pred, args in succ.incoming_args.items() if pred != block.id}
        forwarded_tail = list(succ.incoming_args.get(block.id, []))
        for pred_id in block.predecessors:
            if pred_id == succ.id or pred_id in forwarded_incoming:
                forwarded_incoming = {}
                break
            forwarded_incoming[pred_id] = list(forwarded_tail)
        if not forwarded_incoming and (block.predecessors or succ.incoming_args.get(block.id)):
            continue

        rewritten: list[HIRBlock] = []
        for current in blocks:
            if current.id == block.id:
                continue
            preds = [succ_id if item == block.id else item for item in current.predecessors]
            succs = [succ_id if item == block.id else item for item in current.successors]
            incoming_args = {pred: list(args) for pred, args in current.incoming_args.items()}
            if current.id == succ_id:
                preds = list(block.predecessors) + [item for item in preds if item != block.id]
                incoming_args = forwarded_incoming
            else:
                incoming_args.pop(block.id, None)
            rewritten.append(
                _clone_block(
                    current,
                    branch_target=succ_id if current.branch_target == block.id else current.branch_target,
                    fallthrough_target=succ_id if current.fallthrough_target == block.id else current.fallthrough_target,
                    successors=_ordered_unique(succs),
                    predecessors=_ordered_unique(preds),
                    incoming_args=incoming_args,
                )
            )
        return _sanitize_hir_blocks(rewritten)
    return None


def _rewrite_linear_fallthrough(blocks: list[HIRBlock]) -> Optional[list[HIRBlock]]:
    by_id = {block.id: block for block in blocks}
    for block in blocks:
        if block.terminator.get("kind") != "fallthrough" or len(block.successors) != 1:
            continue
        succ_id = block.successors[0]
        succ = by_id.get(succ_id)
        if succ is None or succ.id == block.id:
            continue
        if succ.predecessors != [block.id]:
            continue

        succ_ready = succ
        if succ_ready.block_params:
            incoming_args = list(succ_ready.incoming_args.get(block.id, []))
            if len(incoming_args) != len(succ_ready.block_params):
                continue
            mapping = {name: incoming_args[idx] for idx, name in enumerate(succ_ready.block_params)}
            succ_ready = _substitute_block_values(succ_ready, mapping)
            succ_ready = _clone_block(succ_ready, block_params=[], phi_bindings=[], incoming_args={})

        if succ_ready.entry_stack:
            if len(succ_ready.entry_stack) != len(block.exit_stack):
                continue
            positional_mapping = {
                succ_name: block_name
                for succ_name, block_name in zip(succ_ready.entry_stack, block.exit_stack)
                if succ_name != block_name
            }
            if positional_mapping:
                succ_ready = _substitute_block_values(succ_ready, positional_mapping)

        merged = _clone_block(
            block,
            end_offset=succ_ready.end_offset,
            instruction_indices=list(block.instruction_indices) + list(succ_ready.instruction_indices),
            exit_stack=list(succ_ready.exit_stack),
            phi_bindings=list(block.phi_bindings) + list(succ_ready.phi_bindings),
            statements=list(block.statements) + list(succ_ready.statements),
            terminator=dict(succ_ready.terminator),
            branch_target=succ_ready.branch_target,
            fallthrough_target=succ_ready.fallthrough_target,
            successors=list(succ_ready.successors),
            flags=_ordered_unique(list(block.flags) + list(succ_ready.flags)),
        )

        rewritten: list[HIRBlock] = []
        for current in blocks:
            if current.id == succ.id:
                continue
            if current.id == block.id:
                rewritten.append(merged)
                continue
            rewritten.append(
                _clone_block(
                    current,
                    branch_target=block.id if current.branch_target == succ.id else current.branch_target,
                    fallthrough_target=block.id if current.fallthrough_target == succ.id else current.fallthrough_target,
                    successors=[block.id if item == succ.id else item for item in current.successors],
                    predecessors=[block.id if item == succ.id else item for item in current.predecessors],
                )
            )
        return _sanitize_hir_blocks(rewritten)
    return None


def _normalize_hir_blocks(hir_blocks: list[HIRBlock]) -> list[HIRBlock]:
    blocks = _sanitize_hir_blocks(hir_blocks)
    while True:
        rewritten = _rewrite_empty_fallthrough(blocks)
        if rewritten is None:
            rewritten = _rewrite_linear_fallthrough(blocks)
        if rewritten is None:
            return blocks
        blocks = rewritten


def _rename_merge_params(hir_blocks: list[HIRBlock]) -> list[HIRBlock]:
    mapping: dict[str, str] = {}
    for block in hir_blocks:
        for name in block.block_params:
            if name.startswith("phi_"):
                mapping[name] = f"m_{name[4:]}"
    if not mapping:
        return hir_blocks
    return _substitute_blocks(hir_blocks, mapping, replace_block_params=True)



def _compute_dominators(hir_blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not hir_blocks:
        return {}
    ids = [block.id for block in hir_blocks]
    _, _, _, preds = _block_maps(hir_blocks)
    entry = hir_blocks[0].id
    dom: dict[str, set[str]] = {block.id: set(ids) for block in hir_blocks}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in hir_blocks[1:]:
            pred_sets = [dom[pred] for pred in preds[block.id] if pred in dom]
            new_dom = {block.id}
            if pred_sets:
                new_dom |= set.intersection(*pred_sets)
            if new_dom != dom[block.id]:
                dom[block.id] = new_dom
                changed = True
    return dom


def _compute_postdominators(hir_blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not hir_blocks:
        return {}
    ids = [block.id for block in hir_blocks]
    _, _, succs, _ = _block_maps(hir_blocks)
    exits = [block.id for block in hir_blocks if not succs[block.id]]
    postdom: dict[str, set[str]] = {block.id: set(ids) for block in hir_blocks}
    for exit_id in exits:
        postdom[exit_id] = {exit_id}
    changed = True
    while changed:
        changed = False
        for block in hir_blocks:
            if block.id in exits:
                continue
            succ_sets = [postdom[succ] for succ in succs[block.id] if succ in postdom]
            new_postdom = {block.id}
            if succ_sets:
                new_postdom |= set.intersection(*succ_sets)
            if new_postdom != postdom[block.id]:
                postdom[block.id] = new_postdom
                changed = True
    return postdom


def _immediate_postdom(block_id: str, postdom: dict[str, set[str]], index_by_id: dict[str, int]) -> Optional[str]:
    candidates = list(postdom.get(block_id, set()) - {block_id})
    if not candidates:
        return None
    candidates.sort(key=lambda item: index_by_id.get(item, 10 ** 6))
    for candidate in candidates:
        if all(candidate not in postdom.get(other, set()) for other in candidates if other != candidate):
            return candidate
    return candidates[0]


def _edge_targets(block: HIRBlock, index_by_id: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    if block.terminator.get("kind") != "branch":
        return None, block.fallthrough_target or (block.successors[0] if block.successors else None)
    branch_target = block.branch_target
    fallthrough_target = block.fallthrough_target
    if branch_target is None and block.successors:
        ordered = sorted(block.successors, key=lambda item: index_by_id.get(item, 10 ** 6))
        branch_target = ordered[0]
        fallthrough_target = ordered[1] if len(ordered) > 1 else None
    return branch_target, fallthrough_target


def _natural_loops(hir_blocks: list[HIRBlock], dom: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    _, index_by_id, succs, preds = _block_maps(hir_blocks)
    loops: dict[str, dict[str, Any]] = {}
    for block in hir_blocks:
        for succ in succs[block.id]:
            if succ in dom.get(block.id, set()) and index_by_id.get(succ, 10 ** 6) <= block.index:
                loop = loops.setdefault(succ, {"header": succ, "latches": set(), "nodes": {succ}})
                loop["latches"].add(block.id)
                work = [block.id]
                while work:
                    node = work.pop()
                    if node in loop["nodes"]:
                        continue
                    loop["nodes"].add(node)
                    for pred in preds.get(node, []):
                        if pred not in loop["nodes"]:
                            work.append(pred)
    for header_id, loop in loops.items():
        node_indices = sorted(index_by_id[node] for node in loop["nodes"])
        loop["index_range"] = (node_indices[0], node_indices[-1]) if node_indices else (index_by_id[header_id], index_by_id[header_id])
        body_succs = [succ for succ in succs[header_id] if succ in loop["nodes"]]
        exit_succs = [succ for succ in succs[header_id] if succ not in loop["nodes"]]
        loop["body_succ"] = body_succs[0] if len(body_succs) == 1 else None
        loop["exit_succ"] = exit_succs[0] if len(exit_succs) == 1 else None
        idx0, idx1 = loop["index_range"]
        contiguous_ids = {hir_blocks[idx].id for idx in range(idx0, idx1 + 1)}
        loop["contiguous"] = contiguous_ids == set(loop["nodes"])
        exit_idx = index_by_id.get(loop["exit_succ"], -1) if loop["exit_succ"] is not None else -1
        body_idx = index_by_id.get(loop["body_succ"], -1) if loop["body_succ"] is not None else -1
        loop["safe"] = (
            hir_blocks[index_by_id[header_id]].terminator.get("kind") == "branch"
            and loop["body_succ"] is not None
            and loop["exit_succ"] is not None
            and loop["contiguous"]
            and body_idx >= idx0
            and exit_idx > idx1
        )
    return loops



def _collect_switch_like(start_index: int, hir_blocks: list[HIRBlock], index_by_id: dict[str, int], ipdom: dict[str, Optional[str]]) -> Optional[dict[str, Any]]:
    chain: list[HIRBlock] = []
    join: Optional[str] = None
    i = start_index
    while i < len(hir_blocks):
        block = hir_blocks[i]
        if block.terminator.get("kind") != "branch" or block.statements:
            break
        block_join = ipdom.get(block.id)
        branch_target, fallthrough_target = _edge_targets(block, index_by_id)
        ordered = [target for target in [branch_target, fallthrough_target] if target is not None]
        if len(ordered) != 2 or fallthrough_target != (hir_blocks[i + 1].id if i + 1 < len(hir_blocks) else None):
            break
        if join is None:
            join = block_join
        elif join != block_join:
            break
        chain.append(block)
        i += 1
        if join is not None and ordered[0] == join:
            break
    if len(chain) < 3 or join is None:
        return None
    return {"blocks": chain, "join": join, "end_index": i}



def _function_summary(
    canonical: list[CanonicalInstruction],
    core_values: list[DataflowValue],
    core_blocks: list[HIRBlock],
    normalized_blocks: list[HIRBlock],
    cfg: dict[str, Any],
    analysis_hints: dict[str, Any],
    dataflow_metrics: dict[str, int],
    report: dict[str, Any],
) -> dict[str, Any]:
    canonical_hist = Counter(inst.semantic_op for inst in canonical)
    validation = report.get("validation", {})
    core_validation = validation.get("core_raw", {})
    normalized_validation = validation.get("normalized_final", {})
    return {
        "canonical_instruction_count": len(canonical),
        "basic_block_count": len(core_blocks),
        "normalized_basic_block_count": len(normalized_blocks),
        "value_count": len(core_values),
        "macro_lowered_count": sum(1 for inst in canonical if inst.macro_kind),
        "call_count": canonical_hist.get("call", 0) + canonical_hist.get("syscall", 0) + canonical_hist.get("opaque_call", 0),
        "branch_count": canonical_hist.get("branch", 0),
        "return_count": canonical_hist.get("return", 0),
        "cfg_anomaly_count": len(cfg.get("anomalies", [])),
        "unresolved_branch_count": int(cfg.get("stats", {}).get("unresolved_targets", 0)),
        "mean_confidence": (sum(inst.confidence for inst in canonical) / len(canonical)) if canonical else 0.0,
        "placeholder_count": dataflow_metrics.get("placeholder_count", 0),
        "phi_count": dataflow_metrics.get("phi_count", 0),
        "loop_hint_count": len(analysis_hints.get("loops", [])),
        "branch_region_hint_count": len(analysis_hints.get("branch_regions", [])),
        "switch_candidate_count": len(analysis_hints.get("switch_candidates", [])),
        "core_validation_error_count": int(core_validation.get("error_count", 0)),
        "validation_error_count": int(normalized_validation.get("error_count", 0)),
    }


def _looks_like_aggregate_prologue(nodes: list[IRNode]) -> bool:
    if not nodes:
        return False
    first = nodes[0]
    if first.offset != 0:
        return False
    if first.semantic_op != "aggregate" or first.raw_kind not in {"AGG0", "AGG"}:
        return False
    if len(nodes) == 1:
        return True
    next_node = nodes[1]
    return next_node.semantic_op in {"const", "load", "read_field", "call", "syscall", "opaque_call", "branch", "return", "cmp", "make_record"}



def _build_hir_analysis_hints(blocks: list[HIRBlock]) -> dict[str, Any]:
    if not blocks:
        return {
            "entry_block": None,
            "exit_blocks": [],
            "rpo": [],
            "dominators": {},
            "postdominators": {},
            "loops": [],
            "branch_regions": [],
            "switch_candidates": [],
        }

    by_id, index_by_id, _, _ = _block_maps(blocks)
    dom = _compute_dominators(blocks)
    postdom = _compute_postdominators(blocks)
    ipdom = {block.id: _immediate_postdom(block.id, postdom, index_by_id) for block in blocks}
    loops = _natural_loops(blocks, dom)
    switch_candidates = []
    for idx in range(len(blocks)):
        info = _collect_switch_like(idx, blocks, index_by_id, ipdom)
        if info is None:
            continue
        switch_candidates.append(
            {
                "dispatch_head": blocks[idx].id,
                "join_block": info.get("join"),
                "block_ids": [item.id for item in info.get("blocks", [])],
                "end_index": info.get("end_index"),
            }
        )

    branch_regions = []
    loops_by_header = set(loops.keys())
    for block in blocks:
        if block.terminator.get("kind") != "branch":
            continue
        if block.id in loops_by_header:
            continue
        succ_a, succ_b = _edge_targets(block, index_by_id)
        branch_regions.append(
            {
                "header": block.id,
                "condition": block.terminator.get("condition"),
                "true_target": succ_a,
                "false_target": succ_b,
                "merge": ipdom.get(block.id),
            }
        )

    def _rpo_order() -> list[str]:
        visited: set[str] = set()
        order: list[str] = []
        entry = blocks[0].id

        def dfs(block_id: str) -> None:
            if block_id in visited or block_id not in by_id:
                return
            visited.add(block_id)
            for succ in by_id[block_id].successors:
                dfs(succ)
            order.append(block_id)

        dfs(entry)
        for block in blocks:
            dfs(block.id)
        order.reverse()
        return order

    loop_hints = []
    for loop in loops.values():
        loop_hints.append(
            {
                "header": loop.get("header"),
                "body_succ": loop.get("body_succ"),
                "exit_succ": loop.get("exit_succ"),
                "nodes": list(loop.get("nodes") or []),
                "safe": bool(loop.get("safe")),
                "index_range": list(loop.get("index_range") or []),
            }
        )

    exit_blocks = [block.id for block in blocks if not block.successors or block.terminator.get("kind") == "return"]
    return {
        "entry_block": blocks[0].id,
        "exit_blocks": exit_blocks,
        "rpo": _rpo_order(),
        "dominators": {block_id: sorted(items, key=lambda item: index_by_id.get(item, 10 ** 9)) for block_id, items in dom.items()},
        "postdominators": {block_id: sorted(items, key=lambda item: index_by_id.get(item, 10 ** 9)) for block_id, items in postdom.items()},
        "loops": sorted(loop_hints, key=lambda item: index_by_id.get(item.get("header"), 10 ** 9)),
        "branch_regions": branch_regions,
        "switch_candidates": switch_candidates,
    }


def build_function_hir(mod: MBCModule, export_name: str, include_canonical: bool = True, include_text: bool = True) -> HIRFunction:
    t0 = time.perf_counter()
    ir_function = build_function_ir(mod, export_name)
    hir_nodes = list(ir_function.nodes)
    prologue_meta: dict[str, Any] | None = None
    entry_seed: list[str] | None = None
    if _looks_like_aggregate_prologue(hir_nodes):
        first = hir_nodes[0]
        prologue_meta = {
            "kind": "aggregate_signature",
            "raw_kind": first.raw_kind,
            "arity": first.operands.get("arity"),
            "children": list(first.operands.get("children") or []),
        }
        entry_seed = [_slot_name(child.get("ref")) for child in first.operands.get("children") or [] if child.get("ref") is not None] or None
        hir_nodes = hir_nodes[1:]

    t_lower0 = time.perf_counter()
    canonical, _ = lower_ir_nodes(hir_nodes)
    _build_instruction_control(canonical, hir_nodes)
    canonical_blocks, canonical_cfg = _compute_canonical_blocks(canonical)
    t_lower1 = time.perf_counter()

    t_core0 = time.perf_counter()
    core_hir_blocks, core_values, dataflow_metrics = build_hir_blocks(canonical_blocks, canonical_cfg, canonical, entry_seed=entry_seed)
    t_core1 = time.perf_counter()

    run_validation = include_canonical or include_text
    validation: dict[str, Any] = {}
    if run_validation:
        validation["core_raw"] = validate_hir_blocks(core_hir_blocks, core_values, "core_raw")

    t_norm0 = time.perf_counter()
    normalized_seed_blocks = [_clone_block(block) for block in core_hir_blocks]
    normalized_cleaned_blocks, _, cleanup_debug = _cleanup_surface_hir(normalized_seed_blocks, canonical, core_values)
    if run_validation:
        validation["normalized_cleaned"] = validate_hir_blocks(normalized_cleaned_blocks, core_values, "normalized_cleaned")
    normalized_blocks = _rename_merge_params(_normalize_hir_blocks(normalized_cleaned_blocks))
    if run_validation:
        validation["normalized_final"] = validate_hir_blocks(normalized_blocks, core_values, "normalized_final")
    t_norm1 = time.perf_counter()

    t_hints0 = time.perf_counter()
    analysis_hints = _build_hir_analysis_hints(normalized_blocks)
    if prologue_meta is not None:
        analysis_hints["prologue"] = prologue_meta
    t_hints1 = time.perf_counter()

    pipeline = {
        "core_block_count": len(core_hir_blocks),
        "normalized_block_count": len(normalized_blocks),
        **dataflow_metrics,
        **cleanup_debug,
    }
    timings_ms = {
        "total": round((time.perf_counter() - t0) * 1000.0, 3),
        "canonical_lowering": round((t_lower1 - t_lower0) * 1000.0, 3),
        "core_hir": round((t_core1 - t_core0) * 1000.0, 3),
        "normalization": round((t_norm1 - t_norm0) * 1000.0, 3),
        "analysis_hints": round((t_hints1 - t_hints0) * 1000.0, 3),
    }
    report = {
        "validation": validation,
        "pipeline": pipeline,
        "timings_ms": timings_ms,
    }
    summary = _function_summary(canonical, core_values, core_hir_blocks, normalized_blocks, canonical_cfg, analysis_hints, dataflow_metrics, report)

    return HIRFunction(
        name=export_name,
        span=ir_function.span,
        slice_mode=ir_function.slice_mode,
        summary=summary,
        cfg=canonical_cfg,
        canonical_instructions=canonical if include_canonical else [],
        core_dataflow_values=core_values,
        core_hir_blocks=core_hir_blocks,
        normalized_hir_blocks=normalized_blocks,
        analysis_hints=analysis_hints,
        report=report,
        body_selection=ir_function.body_selection,
    )


def _module_summary(functions: list[HIRFunction]) -> dict[str, Any]:
    total_canonical = sum(fn.summary["canonical_instruction_count"] for fn in functions)
    return {
        "export_count": len(functions),
        "total_canonical_instructions": total_canonical,
        "avg_canonical_instructions_per_export": (total_canonical / len(functions)) if functions else 0.0,
        "total_values": sum(fn.summary["value_count"] for fn in functions),
        "total_cfg_anomalies": sum(fn.summary["cfg_anomaly_count"] for fn in functions),
        "total_unresolved_branches": sum(fn.summary["unresolved_branch_count"] for fn in functions),
        "total_macro_lowered": sum(fn.summary["macro_lowered_count"] for fn in functions),
        "total_placeholders": sum(fn.summary.get("placeholder_count", 0) for fn in functions),
        "total_phi": sum(fn.summary.get("phi_count", 0) for fn in functions),
        "total_loop_hints": sum(fn.summary.get("loop_hint_count", 0) for fn in functions),
        "total_branch_region_hints": sum(fn.summary.get("branch_region_hint_count", 0) for fn in functions),
        "total_switch_candidates": sum(fn.summary.get("switch_candidate_count", 0) for fn in functions),
        "total_core_validation_errors": sum(fn.summary.get("core_validation_error_count", 0) for fn in functions),
        "total_validation_errors": sum(fn.summary.get("validation_error_count", 0) for fn in functions),
    }


def build_module_hir(path: str | Path, include_canonical: bool = True, include_text: bool = True) -> dict[str, Any]:
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    functions = [build_function_hir(mod, name, include_canonical=include_canonical, include_text=False) for name in mod.export_names()]
    summary_only_layout = (not include_canonical) and (not include_text)
    functions_payload = [
        {"name": fn.name, "summary": fn.summary, "analysis_hints": fn.analysis_hints}
        if summary_only_layout
        else fn.to_dict(include_canonical=include_canonical, include_text=False)
        for fn in functions
    ]
    return {
        "contract": {
            "version": HIR_CONTRACT_VERSION,
            "layers": [
                "token_stream",
                "normalized_ir_nodes",
                "canonical_instructions",
                "canonical_cfg",
                "core_hir",
                "normalized_hir",
                "analysis_hints",
                "report",
            ],
            "notes": [
                "HIR ends at a normalized CFG-like IR and does not try to be the final source-shaped tree",
                "CFG is rebuilt after canonical lowering, not reused from pre-lowered IR nodes",
                "macro signatures are lowered conservatively to avoid inventing false returns",
                "branch predicates are rendered as cond[opcode](value) to avoid inventing exact VM truth semantics",
                "normalized_hir applies only semantics-safe cleanup, fallthrough collapse, and merge-parameter renaming",
                "analysis_hints provides dominators, postdominators, loop candidates, branch regions, and switch candidates for AST",
                "final structuring and semantic lifting now belong to ast.py, not hir.py",
            ],
        },
        "path": str(path),
        "script_name": path.name,
        "has_magic_header": mod.has_magic_header,
        "code_base": mod.code_base,
        "code_size": mod.code_size,
        "data_blob_size": mod.data_blob_size,
        "definition_count": len(mod.definitions),
        "globals_count": len(mod.globals),
        "exports_count": len(mod.exports),
        "summary": _module_summary(functions),
        "functions": functions_payload,
    }


def summarize_corpus(module_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    module_count = len(module_payloads)
    export_count = 0
    total_instructions = 0
    unresolved = 0
    cfg_anomalies = 0
    macro_lowered = 0
    placeholders = 0
    phi_total = 0
    total_loop_hints = 0
    total_branch_region_hints = 0
    total_switch_candidates = 0
    heaviest_functions: list[dict[str, Any]] = []

    for module in module_payloads:
        summary = module.get("summary", {})
        export_count += int(summary.get("export_count", 0))
        total_instructions += int(summary.get("total_canonical_instructions", 0))
        unresolved += int(summary.get("total_unresolved_branches", 0))
        cfg_anomalies += int(summary.get("total_cfg_anomalies", 0))
        macro_lowered += int(summary.get("total_macro_lowered", 0))
        placeholders += int(summary.get("total_placeholders", 0))
        phi_total += int(summary.get("total_phi", 0))
        total_loop_hints += int(summary.get("total_loop_hints", 0))
        total_branch_region_hints += int(summary.get("total_branch_region_hints", 0))
        total_switch_candidates += int(summary.get("total_switch_candidates", 0))
        for fn in module.get("functions", []):
            fn_summary = fn.get("summary", {})
            heaviest_functions.append(
                {
                    "script_name": module.get("script_name"),
                    "function": fn.get("name"),
                    "canonical_instruction_count": int(fn_summary.get("canonical_instruction_count", 0)),
                    "cfg_anomaly_count": int(fn_summary.get("cfg_anomaly_count", 0)),
                    "placeholder_count": int(fn_summary.get("placeholder_count", 0)),
                    "loop_hint_count": int(fn_summary.get("loop_hint_count", 0)),
                }
            )

    heaviest_functions.sort(key=lambda item: (-item["canonical_instruction_count"], item["placeholder_count"], item["script_name"], item["function"]))
    return {
        "summary": {
            "module_count": module_count,
            "export_count": export_count,
            "total_canonical_instructions": total_instructions,
            "avg_canonical_instructions_per_export": (total_instructions / export_count) if export_count else 0.0,
            "total_cfg_anomalies": cfg_anomalies,
            "total_unresolved_branches": unresolved,
            "total_macro_lowered": macro_lowered,
            "total_placeholders": placeholders,
            "total_phi": phi_total,
            "total_loop_hints": total_loop_hints,
            "total_branch_region_hints": total_branch_region_hints,
            "total_switch_candidates": total_switch_candidates,
            "total_core_validation_errors": sum(int(module.get("summary", {}).get("total_core_validation_errors", 0)) for module in module_payloads),
            "total_validation_errors": sum(int(module.get("summary", {}).get("total_validation_errors", 0)) for module in module_payloads),
        },
        "heaviest_functions": heaviest_functions[:32],
    }


def write_json(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_text(text: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
