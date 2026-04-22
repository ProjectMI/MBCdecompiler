from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from mbl_vm_tools.IR import IRFunction, IRNode, build_function_ir
from mbl_vm_tools.parser import MBCModule


HIR_CONTRACT_VERSION = "hir-v2"


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
    dataflow_values: list[DataflowValue]
    hir_blocks: list[HIRBlock]
    structured_hir: dict[str, Any]
    hir_text: str
    body_selection: dict[str, Any]

    def to_dict(self, include_canonical: bool = True, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "span": self.span,
            "slice_mode": self.slice_mode,
            "summary": self.summary,
            "cfg": self.cfg,
            "dataflow_values": [value.to_dict() for value in self.dataflow_values],
            "hir_blocks": [block.to_dict() for block in self.hir_blocks],
            "structured_hir": self.structured_hir,
            "body_selection": self.body_selection,
        }
        if include_canonical:
            payload["canonical_instructions"] = [inst.to_dict() for inst in self.canonical_instructions]
        if include_text:
            payload["hir_text"] = self.hir_text
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

        statements: list[str] = []
        terminator: dict[str, Any] = {"kind": "fallthrough", "text": None, "condition": None}
        statements.extend(phi_bindings)

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


# --- structuring ----------------------------------------------------------


def _block_maps(hir_blocks: list[HIRBlock]) -> tuple[dict[str, HIRBlock], dict[str, int], dict[str, list[str]], dict[str, list[str]]]:
    by_id = {block.id: block for block in hir_blocks}
    index_by_id = {block.id: block.index for block in hir_blocks}
    succs = {block.id: list(block.successors) for block in hir_blocks}
    preds = {block.id: list(block.predecessors) for block in hir_blocks}
    return by_id, index_by_id, succs, preds


def _indent(lines: list[str], level: int) -> list[str]:
    prefix = "    " * level
    return [prefix + line if line else "" for line in lines]


def _compute_dominators(hir_blocks: list[HIRBlock]) -> dict[str, set[str]]:
    if not hir_blocks:
        return {}
    ids = [block.id for block in hir_blocks]
    by_id, _, _, preds = _block_maps(hir_blocks)
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
        for block in reversed(hir_blocks):
            if block.id in exits:
                continue
            succ_sets = [postdom[succ] for succ in succs[block.id] if succ in postdom]
            new_post = {block.id}
            if succ_sets:
                new_post |= set.intersection(*succ_sets)
            if new_post != postdom[block.id]:
                postdom[block.id] = new_post
                changed = True
    return postdom


def _immediate_postdom(block_id: str, postdom: dict[str, set[str]], index_by_id: dict[str, int]) -> Optional[str]:
    candidates = list(postdom.get(block_id, set()) - {block_id})
    if not candidates:
        return None
    candidates.sort(key=lambda item: index_by_id.get(item, 10 ** 6))
    for candidate in candidates:
        if all(candidate == other or candidate not in postdom.get(other, set()) for other in candidates):
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
    by_id, index_by_id, succs, preds = _block_maps(hir_blocks)
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
        header = by_id[header_id]
        body_succs = [succ for succ in succs[header_id] if succ in loop["nodes"]]
        exit_succs = [succ for succ in succs[header_id] if succ not in loop["nodes"]]
        loop["body_succ"] = body_succs[0] if len(body_succs) == 1 else None
        loop["exit_succ"] = exit_succs[0] if len(exit_succs) == 1 else None
        idx0, idx1 = loop["index_range"]
        contiguous_ids = {hir_blocks[idx].id for idx in range(idx0, idx1 + 1)}
        loop["contiguous"] = contiguous_ids == set(loop["nodes"])
        loop["safe"] = header.terminator.get("kind") == "branch" and loop["body_succ"] is not None and loop["exit_succ"] is not None and loop["contiguous"]
    return loops


def _fallback_block_lines(block: HIRBlock, index_by_id: dict[str, int]) -> list[str]:
    lines = [f"{block.id}:" ]
    for stmt in block.statements:
        lines.append(f"    {stmt}")
    if block.terminator.get("kind") == "branch":
        true_succ = block.branch_target or "<unresolved>"
        false_succ = block.fallthrough_target
        lines.append(f"    if {block.terminator['condition']}: goto {true_succ}")
        if false_succ:
            lines.append(f"    else: goto {false_succ}")
    return lines


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


def _render_structured(blocks: list[HIRBlock]) -> tuple[str, dict[str, Any]]:
    if not blocks:
        return "", {"constructs": {}, "fallback_block_count": 0}

    by_id, index_by_id, succs, _ = _block_maps(blocks)
    dom = _compute_dominators(blocks)
    postdom = _compute_postdominators(blocks)
    ipdom = {block.id: _immediate_postdom(block.id, postdom, index_by_id) for block in blocks}
    loops = _natural_loops(blocks, dom)
    constructs: Counter[str] = Counter()
    visited: set[str] = set()
    fallback_blocks = 0

    switches: dict[int, dict[str, Any]] = {}
    for idx in range(len(blocks)):
        info = _collect_switch_like(idx, blocks, index_by_id, ipdom)
        if info:
            switches[idx] = info

    def render_range(start_idx: int, stop_idx: Optional[int], indent: int, allowed_ids: Optional[set[str]] = None) -> list[str]:
        nonlocal fallback_blocks
        lines: list[str] = []
        limit = stop_idx if stop_idx is not None else len(blocks)
        i = start_idx
        while i < limit and i < len(blocks):
            block = blocks[i]
            if allowed_ids is not None and block.id not in allowed_ids:
                i += 1
                continue
            if block.id in visited:
                i += 1
                continue

            loop = loops.get(block.id)
            if loop and loop.get("safe") and loop["index_range"][1] < limit and (allowed_ids is None or set(loop["nodes"]).issubset(allowed_ids | {block.id})):
                constructs["while"] += 1
                visited.add(block.id)
                header_lines = [line for line in block.statements]
                cond = block.terminator.get("condition") or f"cond_{block.id}"
                body_ids = set(loop["nodes"]) - {block.id}
                body_start = index_by_id.get(loop["body_succ"], i + 1) if loop.get("body_succ") else i + 1
                body_end = loop["index_range"][1] + 1
                if header_lines:
                    lines.extend(_indent(["while (true) {"], indent))
                    lines.extend(_indent(header_lines, indent + 1))
                    lines.extend(_indent([f"if (!({cond})) break"], indent + 1))
                    lines.extend(render_range(body_start, body_end, indent + 1, body_ids))
                    lines.extend(_indent(["}"], indent))
                else:
                    lines.extend(_indent([f"while ({cond}) {{"], indent))
                    lines.extend(render_range(body_start, body_end, indent + 1, body_ids))
                    lines.extend(_indent(["}"], indent))
                visited.update(body_ids)
                i = loop["index_range"][1] + 1
                continue

            switch_info = switches.get(i)
            if switch_info and switch_info["end_index"] <= limit:
                constructs["switch_like"] += 1
                lines.extend(_indent(["switch_like {"], indent))
                for case_block in switch_info["blocks"]:
                    visited.add(case_block.id)
                    case_target, _ = _edge_targets(case_block, index_by_id)
                    lines.extend(_indent([f"when ({case_block.terminator.get('condition')}) -> {case_target or '<unresolved>'} {{"], indent + 1))
                    if case_target in by_id and case_target not in visited:
                        target_block = by_id[case_target]
                        case_lines = [line for line in target_block.statements]
                        if target_block.terminator.get("kind") == "return":
                            case_lines.append(target_block.terminator["text"])
                            visited.add(case_target)
                        lines.extend(_indent(case_lines, indent + 2))
                    lines.extend(_indent(["}"], indent + 1))
                lines.extend(_indent(["default: { /* dispatch fallthrough */ }", "}"], indent))
                i = switch_info["end_index"]
                continue

            join = ipdom.get(block.id)
            succ_a, succ_b = _edge_targets(block, index_by_id)
            if block.terminator.get("kind") == "branch" and join and join in index_by_id and index_by_id[join] < limit and block.id not in loops:
                join_idx = index_by_id[join]
                before_join = [candidate.id for candidate in blocks[i + 1:join_idx] if allowed_ids is None or candidate.id in allowed_ids]
                arm_a = [bid for bid in before_join if succ_a and succ_a in dom.get(bid, set()) and not (succ_b and succ_b in dom.get(bid, set()))]
                arm_b = [bid for bid in before_join if succ_b and succ_b in dom.get(bid, set()) and not (succ_a and succ_a in dom.get(bid, set()))]
                arm_a_idx = [index_by_id[bid] for bid in arm_a]
                arm_b_idx = [index_by_id[bid] for bid in arm_b]
                arm_a_contig = not arm_a_idx or arm_a_idx == list(range(min(arm_a_idx), max(arm_a_idx) + 1))
                arm_b_contig = not arm_b_idx or arm_b_idx == list(range(min(arm_b_idx), max(arm_b_idx) + 1))
                if succ_b == join and arm_a and arm_a_contig:
                    constructs["if"] += 1
                    visited.add(block.id)
                    lines.extend(_indent(block.statements, indent))
                    lines.extend(_indent([f"if ({block.terminator['condition']}) {{"], indent))
                    lines.extend(render_range(min(arm_a_idx), max(arm_a_idx) + 1, indent + 1, set(arm_a)))
                    lines.extend(_indent(["}"], indent))
                    visited.update(arm_a)
                    i = join_idx
                    continue
                if succ_a == join and arm_b and arm_b_contig:
                    constructs["if"] += 1
                    visited.add(block.id)
                    lines.extend(_indent(block.statements, indent))
                    lines.extend(_indent([f"if (!({block.terminator['condition']})) {{"], indent))
                    lines.extend(render_range(min(arm_b_idx), max(arm_b_idx) + 1, indent + 1, set(arm_b)))
                    lines.extend(_indent(["}"], indent))
                    visited.update(arm_b)
                    i = join_idx
                    continue
                if arm_a and arm_b and arm_a_contig and arm_b_contig and set(arm_a).isdisjoint(arm_b):
                    constructs["if_else"] += 1
                    visited.add(block.id)
                    lines.extend(_indent(block.statements, indent))
                    lines.extend(_indent([f"if ({block.terminator['condition']}) {{"], indent))
                    lines.extend(render_range(min(arm_a_idx), max(arm_a_idx) + 1, indent + 1, set(arm_a)))
                    lines.extend(_indent(["} else {"], indent))
                    lines.extend(render_range(min(arm_b_idx), max(arm_b_idx) + 1, indent + 1, set(arm_b)))
                    lines.extend(_indent(["}"], indent))
                    visited.update(arm_a)
                    visited.update(arm_b)
                    i = join_idx
                    continue

            visited.add(block.id)
            fallback_blocks += 1
            lines.extend(_indent(_fallback_block_lines(block, index_by_id), indent))
            i += 1
        return lines

    body_lines = render_range(0, None, 1)
    text = "\n".join(body_lines)
    meta = {
        "constructs": dict(constructs),
        "fallback_block_count": fallback_blocks,
        "loop_header_count": sum(1 for loop in loops.values() if loop.get("safe")),
    }
    return text, meta


# --- public builders ------------------------------------------------------


def _function_summary(canonical: list[CanonicalInstruction], values: list[DataflowValue], blocks: list[HIRBlock], cfg: dict[str, Any], structured_meta: dict[str, Any], dataflow_metrics: dict[str, int]) -> dict[str, Any]:
    canonical_hist = Counter(inst.semantic_op for inst in canonical)
    return {
        "canonical_instruction_count": len(canonical),
        "basic_block_count": len(blocks),
        "value_count": len(values),
        "macro_lowered_count": sum(1 for inst in canonical if inst.macro_kind),
        "call_count": canonical_hist.get("call", 0) + canonical_hist.get("syscall", 0) + canonical_hist.get("opaque_call", 0),
        "branch_count": canonical_hist.get("branch", 0),
        "return_count": canonical_hist.get("return", 0),
        "constructs": structured_meta.get("constructs", {}),
        "fallback_block_count": structured_meta.get("fallback_block_count", 0),
        "cfg_anomaly_count": len(cfg.get("anomalies", [])),
        "unresolved_branch_count": int(cfg.get("stats", {}).get("unresolved_targets", 0)),
        "mean_confidence": (sum(inst.confidence for inst in canonical) / len(canonical)) if canonical else 0.0,
        "placeholder_count": dataflow_metrics.get("placeholder_count", 0),
        "phi_count": dataflow_metrics.get("phi_count", 0),
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


def build_function_hir(mod: MBCModule, export_name: str, include_canonical: bool = True, include_text: bool = True) -> HIRFunction:
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

    canonical, _ = lower_ir_nodes(hir_nodes)
    _build_instruction_control(canonical, hir_nodes)
    canonical_blocks, canonical_cfg = _compute_canonical_blocks(canonical)
    hir_blocks, values, dataflow_metrics = build_hir_blocks(canonical_blocks, canonical_cfg, canonical, entry_seed=entry_seed)
    hir_text_body, structured_meta = _render_structured(hir_blocks)
    structured_meta = dict(structured_meta)
    if prologue_meta is not None:
        structured_meta["prologue"] = prologue_meta
    summary = _function_summary(canonical, values, hir_blocks, canonical_cfg, structured_meta, dataflow_metrics)

    entry_args: list[str] = hir_blocks[0].entry_stack if hir_blocks else []
    function_header = f"function {export_name}({', '.join(entry_args)}) {{" if entry_args else f"function {export_name}() {{"
    function_footer = "}"
    hir_text = "\n".join([function_header, hir_text_body, function_footer]) if include_text else ""

    return HIRFunction(
        name=export_name,
        span=ir_function.span,
        slice_mode=ir_function.slice_mode,
        summary=summary,
        cfg=canonical_cfg,
        canonical_instructions=canonical if include_canonical else [],
        dataflow_values=values,
        hir_blocks=hir_blocks,
        structured_hir=structured_meta,
        hir_text=hir_text,
        body_selection=ir_function.validation.get("body_selection", {}),
    )


def _module_summary(functions: list[HIRFunction]) -> dict[str, Any]:
    total_canonical = sum(fn.summary["canonical_instruction_count"] for fn in functions)
    constructs = Counter()
    for fn in functions:
        constructs.update(fn.summary.get("constructs", {}))
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
        "construct_histogram": dict(constructs),
    }


def build_module_hir(path: str | Path, include_canonical: bool = True, include_text: bool = True) -> dict[str, Any]:
    path = Path(path)
    mod = MBCModule(path, collect_auxiliary=False)
    functions = [build_function_hir(mod, name, include_canonical=include_canonical, include_text=include_text) for name in mod.export_names()]
    return {
        "contract": {
            "version": HIR_CONTRACT_VERSION,
            "layers": [
                "token_stream",
                "normalized_ir_nodes",
                "canonical_instructions",
                "canonical_cfg",
                "stack_dataflow",
                "structured_hir",
            ],
            "notes": [
                "CFG is rebuilt after canonical lowering, not reused from pre-lowered IR nodes",
                "macro signatures are lowered conservatively to avoid inventing false returns",
                "branch predicates are rendered as cond[opcode](value) to avoid inventing exact VM truth semantics",
                "structuring is conservative and falls back to labeled CFG blocks when a safe reconstruction is unavailable",
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
        "functions": [fn.to_dict(include_canonical=include_canonical, include_text=include_text) for fn in functions],
    }


def summarize_corpus(module_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    module_count = len(module_payloads)
    export_count = 0
    total_instructions = 0
    construct_hist = Counter()
    unresolved = 0
    cfg_anomalies = 0
    macro_lowered = 0
    placeholders = 0
    phi_total = 0
    heaviest_functions: list[dict[str, Any]] = []

    for module in module_payloads:
        summary = module.get("summary", {})
        export_count += int(summary.get("export_count", 0))
        total_instructions += int(summary.get("total_canonical_instructions", 0))
        construct_hist.update(summary.get("construct_histogram", {}))
        unresolved += int(summary.get("total_unresolved_branches", 0))
        cfg_anomalies += int(summary.get("total_cfg_anomalies", 0))
        macro_lowered += int(summary.get("total_macro_lowered", 0))
        placeholders += int(summary.get("total_placeholders", 0))
        phi_total += int(summary.get("total_phi", 0))
        for fn in module.get("functions", []):
            fn_summary = fn.get("summary", {})
            heaviest_functions.append(
                {
                    "script_name": module.get("script_name"),
                    "function": fn.get("name"),
                    "canonical_instruction_count": int(fn_summary.get("canonical_instruction_count", 0)),
                    "constructs": fn_summary.get("constructs", {}),
                    "cfg_anomaly_count": int(fn_summary.get("cfg_anomaly_count", 0)),
                    "placeholder_count": int(fn_summary.get("placeholder_count", 0)),
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
            "construct_histogram": dict(construct_hist),
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
