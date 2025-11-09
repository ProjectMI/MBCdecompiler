"""Construction of the higher-level AST from the normalised IR."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Optional, Sequence, Set, Tuple

from ..ir.cfg import analyse_segments
from ..ir.model import (
    IRBlock,
    IRCallReturn,
    IRFlagCheck,
    IRFunctionCfg,
    IRFunctionPrologue,
    IRIf,
    IRProgram,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRNode,
    IRCfgBlock,
)
from .model import ASTBlock, ASTDominatorInfo, ASTEdge, ASTFunction, ASTLoop, ASTProgram, ASTTerminatorView

_VIRTUAL_EXIT = "__virtual_exit__"


@dataclass
class _MutableEdge:
    kind: str
    target: str


@dataclass
class _MutableTerminator:
    kind: str
    text: str
    edges: List[_MutableEdge] = field(default_factory=list)
    data: Dict[str, object] = field(default_factory=dict)

    def successors(self) -> Tuple[str, ...]:
        return tuple(edge.target for edge in self.edges)

    def replace_target(self, old: str, new: str) -> None:
        updated = False
        for edge in self.edges:
            if edge.target == old:
                edge.target = new
                updated = True
        if updated:
            self.canonicalise()

    def canonicalise(self) -> None:
        if self.kind == "goto":
            target = self.edges[0].target if self.edges else "?"
            self.text = f"goto {target}"
        elif self.kind == "branch":
            form = str(self.data.get("form", "if"))
            condition = str(self.data.get("condition", "cond"))
            then_target = next((edge.target for edge in self.edges if edge.kind == "then"), None)
            else_target = next((edge.target for edge in self.edges if edge.kind == "else"), None)
            if form == "if":
                self.text = f"if {condition} then {then_target} else {else_target}"
            elif form == "testset":
                self.text = f"testset {condition} then {then_target} else {else_target}"
            elif form == "flag":
                self.text = f"flag {condition} then {then_target} else {else_target}"
            elif form == "prologue":
                self.text = f"prologue {condition} then {then_target} else {else_target}"
            else:
                self.text = f"{form} {condition} then {then_target} else {else_target}"
        elif self.kind == "switch":
            parts: List[str] = []
            default = None
            for edge in self.edges:
                if edge.kind.startswith("case "):
                    parts.append(f"{edge.kind[5:]}->{edge.target}")
                elif edge.kind == "default":
                    default = edge.target
            helper = self.data.get("helper")
            index = self.data.get("index")
            description = "switch [" + ", ".join(parts) + "]"
            if default is not None:
                description += f" default={default}"
            if helper:
                description += f" helper={helper}"
            if index:
                description += f" index={index}"
            self.text = description
        elif self.kind in {"call", "tailcall", "tailcall_return", "return", "raw"}:
            description = self.data.get("description")
            if isinstance(description, str):
                self.text = description


@dataclass
class _MutableBlock:
    label: str
    offset: Optional[int]
    statements: List[IRNode]
    terminator: _MutableTerminator
    annotations: Tuple[str, ...] = tuple()
    synthetic: bool = False
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)

    def refresh_successors(self) -> None:
        self.successors = list(self.terminator.successors())


class ASTBuilder:
    """Recover a higher-level AST representation from the IR programme."""

    def __init__(self) -> None:
        self._critical_counter = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, program: IRProgram) -> ASTProgram:
        cfg = program.cfg
        if cfg is None:
            cfg, _ = analyse_segments(program.segments)
        functions = [self._build_function(function_cfg, program.segments) for function_cfg in cfg.functions]
        return ASTProgram(functions=tuple(functions))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_function(self, function_cfg: IRFunctionCfg, segments: Sequence[IRSegment]) -> ASTFunction:
        block_lookup = self._build_block_lookup(segments)
        blocks: MutableMapping[str, _MutableBlock] = {}
        order: List[str] = []

        for cfg_block in function_cfg.blocks:
            ir_block = block_lookup.get(cfg_block.label)
            mutable = self._block_from_ir(cfg_block, ir_block)
            blocks[cfg_block.label] = mutable
            order.append(cfg_block.label)

        self._prune_unreachable(blocks, function_cfg.entry_block)
        order = [label for label in order if label in blocks]

        self._recompute_predecessors(blocks)
        self._fold_trivial_branches(blocks)
        self._split_critical_edges(blocks, order, function_cfg.entry_block)
        self._merge_degenerate_gotos(blocks, order, function_cfg.entry_block)
        self._split_critical_edges(blocks, order, function_cfg.entry_block)
        self._canonicalise(blocks)
        self._recompute_predecessors(blocks)

        dominators = self._compute_dominators(blocks, function_cfg.entry_block)
        post_dominators = self._compute_post_dominators(blocks, function_cfg.entry_block)
        loops = self._discover_loops(blocks, dominators)

        ast_blocks = self._freeze_blocks(blocks, order)
        dom_infos = self._freeze_dominators(dominators)
        post_dom_infos = self._freeze_dominators(post_dominators)
        loop_infos = self._freeze_loops(loops)

        return ASTFunction(
            name=function_cfg.name,
            segment_index=function_cfg.segment_index,
            entry_block=function_cfg.entry_block,
            entry_offset=function_cfg.entry_offset,
            blocks=ast_blocks,
            dominators=dom_infos,
            post_dominators=post_dom_infos,
            loops=loop_infos,
        )

    def _build_block_lookup(self, segments: Sequence[IRSegment]) -> Dict[str, IRBlock]:
        lookup: Dict[str, IRBlock] = {}
        for segment in segments:
            for block in segment.blocks:
                lookup[block.label] = block
        return lookup

    def _block_from_ir(self, cfg_block: IRCfgBlock, ir_block: Optional[IRBlock]) -> _MutableBlock:
        nodes: Tuple[IRNode, ...] = tuple()
        annotations: Tuple[str, ...] = tuple()
        if ir_block is not None:
            nodes = ir_block.nodes
            annotations = ir_block.annotations
        statements: List[IRNode] = list(nodes[:-1]) if nodes else []
        terminator_node: Optional[IRNode]
        terminator_node = nodes[-1] if nodes else None
        if terminator_node is not None and not isinstance(
            terminator_node,
            (IRReturn, IRTailCall, IRTailcallReturn, IRCallReturn, IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue, IRSwitchDispatch, IRTerminator),
        ):
            # Non-terminator node at the end, treat as statement.
            statements.append(terminator_node)
            terminator_node = None

        terminator = self._make_terminator(cfg_block, terminator_node)
        block = _MutableBlock(
            label=cfg_block.label,
            offset=ir_block.start_offset if ir_block is not None else None,
            statements=statements,
            terminator=terminator,
            annotations=annotations,
        )
        block.refresh_successors()
        return block

    def _make_terminator(self, cfg_block: IRCfgBlock, node: Optional[IRNode]) -> _MutableTerminator:
        edges = [_MutableEdge(edge.kind, edge.target) for edge in cfg_block.edges]
        if isinstance(node, IRReturn):
            term = _MutableTerminator("return", node.describe(), edges, {"description": node.describe()})
        elif isinstance(node, IRTailCall):
            term = _MutableTerminator("tailcall", node.describe(), edges, {"description": node.describe()})
        elif isinstance(node, IRTailcallReturn):
            term = _MutableTerminator("tailcall_return", node.describe(), edges, {"description": node.describe()})
        elif isinstance(node, IRCallReturn):
            term = _MutableTerminator("call", node.describe(), edges, {"description": node.describe()})
        elif isinstance(node, IRIf):
            data = {"form": "if", "condition": node.condition}
            term = _MutableTerminator("branch", node.describe(), edges, data)
        elif isinstance(node, IRTestSetBranch):
            cond = f"{node.var}={node.expr}"
            term = _MutableTerminator("branch", node.describe(), edges, {"form": "testset", "condition": cond})
        elif isinstance(node, IRFlagCheck):
            cond = f"0x{node.flag:04X}"
            term = _MutableTerminator("branch", node.describe(), edges, {"form": "flag", "condition": cond})
        elif isinstance(node, IRFunctionPrologue):
            cond = f"{node.var}={node.expr}"
            term = _MutableTerminator("branch", node.describe(), edges, {"form": "prologue", "condition": cond})
        elif isinstance(node, IRSwitchDispatch):
            helper = None
            if node.helper is not None:
                helper = f"0x{node.helper:04X}"
                if node.helper_symbol:
                    helper = f"{node.helper_symbol}({helper})"
            index = node.index.describe() if node.index is not None else None
            data = {"helper": helper, "index": index, "description": node.describe()}
            term = _MutableTerminator("switch", node.describe(), edges, data)
        elif isinstance(node, IRTerminator):
            term = _MutableTerminator("goto", node.describe(), edges, {})
        elif node is not None:
            term = _MutableTerminator("raw", node.describe(), edges, {"description": node.describe()})
        else:
            if len(edges) == 1:
                term = _MutableTerminator("goto", f"goto {edges[0].target}", edges, {})
            elif len(edges) == 2:
                term = _MutableTerminator("branch", cfg_block.terminator, edges, {"form": "unknown", "condition": cfg_block.terminator or "cond"})
            else:
                term = _MutableTerminator("raw", cfg_block.terminator or "", edges, {"description": cfg_block.terminator or ""})
        term.canonicalise()
        return term

    def _prune_unreachable(self, blocks: MutableMapping[str, _MutableBlock], entry: str) -> None:
        reachable: Set[str] = set()
        queue: deque[str] = deque([entry])
        while queue:
            label = queue.popleft()
            if label in reachable:
                continue
            if label not in blocks:
                continue
            reachable.add(label)
            block = blocks[label]
            for target in block.successors:
                if target not in reachable:
                    queue.append(target)
        unreachable = set(blocks) - reachable
        for label in unreachable:
            del blocks[label]

    def _recompute_predecessors(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        for block in blocks.values():
            block.predecessors.clear()
            block.refresh_successors()
        for block in blocks.values():
            for target in block.successors:
                succ = blocks.get(target)
                if succ is None:
                    continue
                if block.label not in succ.predecessors:
                    succ.predecessors.append(block.label)
        for block in blocks.values():
            block.predecessors.sort()
            block.refresh_successors()

    def _split_critical_edges(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        order: List[str],
        entry: str,
    ) -> None:
        changed = True
        while changed:
            changed = False
            for block in list(blocks.values()):
                if len(block.successors) <= 1:
                    continue
                for target in list(block.successors):
                    if not any(edge.target == target for edge in block.terminator.edges):
                        continue
                    succ = blocks.get(target)
                    if succ is None:
                        continue
                    if len(succ.predecessors) <= 1:
                        continue
                    label = self._new_synthetic_label(block.label, target)
                    bridge = _MutableBlock(
                        label=label,
                        offset=None,
                        statements=[],
                        terminator=_MutableTerminator("goto", f"goto {target}", [_MutableEdge("goto", target)], {}),
                        annotations=tuple(),
                        synthetic=True,
                    )
                    bridge.refresh_successors()
                    blocks[label] = bridge
                    order.append(label)
                    block.terminator.replace_target(target, label)
                    block.refresh_successors()
                    bridge.predecessors.append(block.label)
                    succ.predecessors = [p if p != block.label else label for p in succ.predecessors]
                    changed = True
            if changed:
                self._recompute_predecessors(blocks)

    def _merge_degenerate_gotos(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        order: List[str],
        entry: str,
    ) -> None:
        changed = True
        while changed:
            changed = False
            for label in list(blocks.keys()):
                if label == entry:
                    continue
                block = blocks[label]
                if block.synthetic:
                    continue
                if block.statements:
                    continue
                if block.terminator.kind != "goto":
                    continue
                if len(block.successors) != 1:
                    continue
                target = block.successors[0]
                succ = blocks.get(target)
                if succ is None:
                    continue
                if self._has_path(blocks, target, label):
                    continue
                if len(block.predecessors) != 1:
                    continue
                pred_label = block.predecessors[0]
                pred = blocks.get(pred_label)
                if pred is None:
                    continue
                pred.terminator.replace_target(label, target)
                pred.refresh_successors()
                succ.predecessors = [pred_label if p == label else p for p in succ.predecessors]
                del blocks[label]
                if label in order:
                    order.remove(label)
                changed = True
                break
            if changed:
                self._recompute_predecessors(blocks)

    def _fold_trivial_branches(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        for block in blocks.values():
            if block.terminator.kind != "branch":
                continue
            successors = block.successors
            if not successors:
                continue
            targets = set(successors)
            if len(targets) == 1:
                target = successors[0]
                block.terminator = _MutableTerminator("goto", f"goto {target}", [_MutableEdge("goto", target)], {})
                block.refresh_successors()

    def _canonicalise(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        for block in blocks.values():
            block.terminator.canonicalise()
            block.refresh_successors()

    def _compute_dominators(
        self, blocks: MutableMapping[str, _MutableBlock], entry: str
    ) -> Dict[str, Set[str]]:
        nodes = set(blocks)
        dom: Dict[str, Set[str]] = {label: set(nodes) for label in nodes}
        if entry not in dom:
            return dom
        dom[entry] = {entry}
        changed = True
        while changed:
            changed = False
            for label in nodes:
                if label == entry:
                    continue
                block = blocks[label]
                preds = [p for p in block.predecessors if p in nodes]
                if not preds:
                    new_set = {label}
                else:
                    intersect = set(nodes)
                    for pred in preds:
                        intersect &= dom[pred]
                    new_set = {label} | intersect
                if new_set != dom[label]:
                    dom[label] = new_set
                    changed = True
        return dom

    def _compute_post_dominators(
        self, blocks: MutableMapping[str, _MutableBlock], entry: str
    ) -> Dict[str, Set[str]]:
        nodes = set(blocks)
        post_dom: Dict[str, Set[str]] = {label: set(nodes) | {_VIRTUAL_EXIT} for label in nodes}
        post_dom[_VIRTUAL_EXIT] = {_VIRTUAL_EXIT}
        changed = True
        while changed:
            changed = False
            for label in nodes:
                block = blocks[label]
                succs = block.successors or [_VIRTUAL_EXIT]
                intersect = set(nodes) | {_VIRTUAL_EXIT}
                for succ in succs:
                    intersect &= post_dom.get(succ, {_VIRTUAL_EXIT})
                new_set = {label} | intersect
                if new_set != post_dom[label]:
                    post_dom[label] = new_set
                    changed = True
        for label in list(post_dom):
            if label == _VIRTUAL_EXIT:
                continue
            post_dom[label].discard(_VIRTUAL_EXIT)
        return post_dom

    def _discover_loops(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        dominators: Dict[str, Set[str]],
    ) -> Dict[str, Dict[str, Set[str]]]:
        loops: Dict[str, Dict[str, Set[str]]] = {}
        for block in blocks.values():
            for target in block.successors:
                if target not in dominators:
                    continue
                if target not in dominators.get(block.label, set()):
                    continue
                info = loops.setdefault(target, {"nodes": set(), "latches": set()})
                loop_nodes = self._collect_loop_nodes(blocks, block.label, target)
                info["nodes"].update(loop_nodes)
                info["latches"].add(block.label)
        return loops

    def _collect_loop_nodes(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        latch: str,
        header: str,
    ) -> Set[str]:
        loop_nodes: Set[str] = {header}
        worklist: deque[str] = deque([latch])
        while worklist:
            label = worklist.pop()
            if label in loop_nodes:
                continue
            loop_nodes.add(label)
            block = blocks.get(label)
            if block is None:
                continue
            for pred in block.predecessors:
                if pred not in loop_nodes:
                    worklist.append(pred)
        return loop_nodes

    def _has_path(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        start: str,
        goal: str,
    ) -> bool:
        if start == goal:
            return True
        seen: Set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            label = queue.popleft()
            if label == goal:
                return True
            if label in seen:
                continue
            seen.add(label)
            block = blocks.get(label)
            if block is None:
                continue
            for succ in block.successors:
                if succ not in seen:
                    queue.append(succ)
        return False

    def _freeze_blocks(
        self,
        blocks: MutableMapping[str, _MutableBlock],
        order: List[str],
    ) -> Tuple[ASTBlock, ...]:
        frozen: List[ASTBlock] = []
        ordered_labels = [label for label in order if label in blocks]
        additional = [label for label in blocks if label not in ordered_labels]
        for label in ordered_labels + sorted(additional):
            block = blocks[label]
            terminator = ASTTerminatorView(
                kind=block.terminator.kind,
                text=block.terminator.text,
                edges=tuple(ASTEdge(edge.kind, edge.target) for edge in block.terminator.edges),
            )
            frozen.append(
                ASTBlock(
                    label=block.label,
                    offset=block.offset,
                    statements=tuple(block.statements),
                    terminator=terminator,
                    successors=tuple(block.successors),
                    predecessors=tuple(block.predecessors),
                    annotations=block.annotations,
                    synthetic=block.synthetic,
                )
            )
        return tuple(frozen)

    def _freeze_dominators(self, dominators: Dict[str, Set[str]]) -> Tuple[ASTDominatorInfo, ...]:
        infos = []
        for label in sorted(dominators):
            members = tuple(sorted(dominators[label]))
            infos.append(ASTDominatorInfo(label=label, members=members))
        return tuple(infos)

    def _freeze_loops(self, loops: Dict[str, Dict[str, Set[str]]]) -> Tuple[ASTLoop, ...]:
        frozen: List[ASTLoop] = []
        for header in sorted(loops):
            info = loops[header]
            nodes = tuple(sorted(info.get("nodes", set())))
            latches = tuple(sorted(info.get("latches", set())))
            frozen.append(ASTLoop(header=header, nodes=nodes, latches=latches))
        return tuple(frozen)

    def _new_synthetic_label(self, pred: str, target: str) -> str:
        self._critical_counter += 1
        base = f"{pred}_to_{target}_crit{self._critical_counter}"
        return base


__all__ = ["ASTBuilder"]
