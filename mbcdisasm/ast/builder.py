"""Construction of the high level AST from the normalised IR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..ir.model import (
    IRBlock,
    IRCallReturn,
    IRFunctionCfg,
    IRIf,
    IRReturn,
    IRSegment,
    IRSwitchDispatch,
    IRTestSetBranch,
    IRFlagCheck,
    IRFunctionPrologue,
    IRTailCall,
    IRTailcallReturn,
    IRTerminator,
    IRProgram,
    IRNode,
)
from ..ir.cfg import analyse_segments
from .model import (
    ASTBlock,
    ASTFunction,
    ASTFunctionAlias,
    ASTLoop,
    ASTProgram,
    ASTTerminator,
    DominatorInfo,
)


@dataclass
class _Edge:
    kind: str
    target: str


@dataclass
class _GotoTerminator:
    target: str


@dataclass
class _MutableBlock:
    label: str
    start_offset: int
    statements: List[IRNode]
    terminator: Optional[object]
    edges: List[_Edge]
    predecessors: Set[str]
    synthetic: bool = False


_TERMINATOR_TYPES = (
    IRReturn,
    IRTailCall,
    IRTailcallReturn,
    IRCallReturn,
    IRIf,
    IRTestSetBranch,
    IRFlagCheck,
    IRFunctionPrologue,
    IRSwitchDispatch,
    IRTerminator,
)


class ASTBuilder:
    """Build an :class:`ASTProgram` from a normalised :class:`IRProgram`."""

    def build(self, program: IRProgram) -> ASTProgram:
        if program.cfg is None:
            cfg, _ = analyse_segments(program.segments)
        else:
            cfg = program.cfg

        block_lookup = self._build_block_lookup(program.segments)
        functions: List[ASTFunction] = []

        for function_cfg in cfg.functions:
            ast_function = self._build_function(function_cfg, block_lookup)
            functions.append(ast_function)

        functions = self._collapse_auto_trampoline_templates(functions)

        return ASTProgram(functions=tuple(functions))

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_block_lookup(self, segments: Sequence[IRSegment]) -> Mapping[str, IRBlock]:
        lookup: Dict[str, IRBlock] = {}
        for segment in segments:
            for block in segment.blocks:
                lookup[block.label] = block
        return lookup

    def _build_function(
        self,
        function_cfg: IRFunctionCfg,
        block_lookup: Mapping[str, IRBlock],
    ) -> ASTFunction:
        blocks = self._initialise_blocks(function_cfg, block_lookup)
        if function_cfg.entry_block not in blocks:
            raise ValueError(f"entry block {function_cfg.entry_block} missing from CFG")

        self._prune_unreachable(function_cfg.entry_block, blocks)
        self._merge_degenerate_blocks(function_cfg.entry_block, blocks)
        self._remove_critical_edges(blocks)
        self._fold_redundant_branches(blocks)
        self._prune_unreachable(function_cfg.entry_block, blocks)

        dominators = self._compute_dominator_info(function_cfg.entry_block, blocks)
        post_dominators = self._compute_post_dominator_info(
            function_cfg.entry_block, blocks
        )
        loops = self._identify_loops(dominators, blocks)

        ast_blocks = self._freeze_blocks(blocks)

        return ASTFunction(
            segment_index=function_cfg.segment_index,
            name=function_cfg.name,
            entry_block=function_cfg.entry_block,
            entry_offset=function_cfg.entry_offset,
            blocks=ast_blocks,
            dominators=dominators,
            post_dominators=post_dominators,
            loops=loops,
        )

    def _initialise_blocks(
        self,
        function_cfg: IRFunctionCfg,
        block_lookup: Mapping[str, IRBlock],
    ) -> MutableMapping[str, _MutableBlock]:
        blocks: Dict[str, _MutableBlock] = {}
        for cfg_block in function_cfg.blocks:
            ir_block = block_lookup.get(cfg_block.label)
            if ir_block is None:
                raise ValueError(f"missing IR block for label {cfg_block.label}")
            statements, terminator = self._split_block(ir_block)
            edges = [_Edge(edge.kind, edge.target) for edge in cfg_block.edges]
            blocks[cfg_block.label] = _MutableBlock(
                label=cfg_block.label,
                start_offset=cfg_block.start_offset,
                statements=statements,
                terminator=terminator,
                edges=edges,
                predecessors=set(),
                synthetic=False,
            )
        for block in blocks.values():
            for edge in block.edges:
                target = blocks.get(edge.target)
                if target is not None:
                    target.predecessors.add(block.label)
        return blocks

    def _split_block(self, block: IRBlock) -> Tuple[List[IRNode], Optional[IRNode]]:
        if not block.nodes:
            return [], None
        statements = list(block.nodes)
        terminator: Optional[IRNode] = None
        for index in range(len(statements) - 1, -1, -1):
            node = statements[index]
            if isinstance(node, _TERMINATOR_TYPES):
                terminator = node
                del statements[index]
                break
        return statements, terminator

    # ------------------------------------------------------------------
    # graph normalisation passes
    # ------------------------------------------------------------------
    def _collapse_auto_trampoline_templates(
        self, functions: Sequence[ASTFunction]
    ) -> List[ASTFunction]:
        groups: Dict[Tuple[Tuple[str, ...], str], List[int]] = {}
        for index, function in enumerate(functions):
            if not self._is_auto_trampoline_candidate(function):
                continue
            signature = self._auto_trampoline_signature(function)
            groups.setdefault(signature, []).append(index)

        if not groups:
            return list(functions)

        updated: Dict[int, ASTFunction] = {}
        removed: Set[int] = set()
        template_index = 0

        for indices in groups.values():
            if len(indices) <= 1:
                continue

            functions_for_group = [functions[index] for index in indices]
            aliases = tuple(
                ASTFunctionAlias(
                    name=item.name,
                    entry_block=item.entry_block,
                    entry_offset=item.entry_offset,
                )
                for item in sorted(
                    functions_for_group,
                    key=lambda func: (func.segment_index, func.entry_offset),
                )
            )

            template_name = self._auto_trampoline_template_name(template_index)
            template_index += 1

            primary_index = indices[0]
            primary = functions[primary_index]
            updated[primary_index] = replace(
                primary, name=template_name, aliases=aliases
            )
            for extra_index in indices[1:]:
                removed.add(extra_index)

        if not updated and not removed:
            return list(functions)

        collapsed: List[ASTFunction] = []
        for index, function in enumerate(functions):
            if index in removed:
                continue
            collapsed.append(updated.get(index, function))
        return collapsed

    def _is_auto_trampoline_candidate(self, function: ASTFunction) -> bool:
        if not function.name.startswith("auto_"):
            return False
        if len(function.blocks) != 1:
            return False
        block = function.blocks[0]
        if block.successors:
            return False
        return block.terminator.kind == "tailcall"

    def _auto_trampoline_signature(self, function: ASTFunction) -> Tuple[Tuple[str, ...], str]:
        block = function.blocks[0]
        statements = tuple(
            getattr(statement, "describe", lambda: repr(statement))()
            for statement in block.statements
        )
        terminator_detail = block.terminator.describe()
        return statements, terminator_detail

    def _auto_trampoline_template_name(self, index: int) -> str:
        if index == 0:
            return "template.bank_init_trampoline"
        return f"template.bank_init_trampoline_{index + 1}"

    def _prune_unreachable(
        self, entry: str, blocks: MutableMapping[str, _MutableBlock]
    ) -> None:
        reachable = set(self._dfs(entry, blocks))
        removed = True
        while removed:
            removed = False
            for label in list(blocks.keys()):
                if label not in reachable:
                    self._delete_block(label, blocks)
                    removed = True
            if removed:
                reachable = set(self._dfs(entry, blocks))

    def _dfs(self, start: str, blocks: Mapping[str, _MutableBlock]) -> Iterable[str]:
        if start not in blocks:
            return []
        visited: Set[str] = set()
        order: List[str] = []
        stack = [start]
        while stack:
            label = stack.pop()
            if label in visited:
                continue
            visited.add(label)
            order.append(label)
            block = blocks[label]
            for edge in reversed(block.edges):
                if edge.target in blocks:
                    stack.append(edge.target)
        return order

    def _delete_block(
        self, label: str, blocks: MutableMapping[str, _MutableBlock]
    ) -> None:
        block = blocks.pop(label, None)
        if block is None:
            return
        for predecessor_label in list(block.predecessors):
            predecessor = blocks.get(predecessor_label)
            if predecessor is None:
                continue
            predecessor.edges = [
                edge for edge in predecessor.edges if edge.target != label
            ]
        for edge in block.edges:
            successor = blocks.get(edge.target)
            if successor is None:
                continue
            successor.predecessors.discard(label)

    def _merge_degenerate_blocks(
        self, entry: str, blocks: MutableMapping[str, _MutableBlock]
    ) -> None:
        changed = True
        while changed:
            changed = False
            for label, block in list(blocks.items()):
                if label == entry:
                    continue
                if block.synthetic:
                    continue
                if block.statements:
                    continue
                if len(block.edges) != 1:
                    continue
                target_label = block.edges[0].target
                if target_label == label:
                    continue
                target = blocks.get(target_label)
                if target is None:
                    continue
                if isinstance(block.terminator, (IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue, IRSwitchDispatch)):
                    continue
                # rewrite predecessors to jump directly to the successor
                for predecessor_label in list(block.predecessors):
                    predecessor = blocks.get(predecessor_label)
                    if predecessor is None:
                        continue
                    updated = False
                    for edge in predecessor.edges:
                        if edge.target == label:
                            edge.target = target_label
                            updated = True
                    if updated:
                        target.predecessors.add(predecessor_label)
                    target.predecessors.discard(label)
                self._delete_block(label, blocks)
                changed = True
                break

    def _remove_critical_edges(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        counter = 0
        while True:
            updated = False
            for label, block in list(blocks.items()):
                if len(block.edges) <= 1:
                    continue
                for edge in list(block.edges):
                    target = blocks.get(edge.target)
                    if target is None:
                        continue
                    if len(target.predecessors) <= 1:
                        continue
                    counter += 1
                    new_label = f"{target.label}_split_{counter}"
                    while new_label in blocks:
                        counter += 1
                        new_label = f"{target.label}_split_{counter}"
                    split_block = _MutableBlock(
                        label=new_label,
                        start_offset=target.start_offset,
                        statements=[],
                        terminator=_GotoTerminator(target=target.label),
                        edges=[_Edge("goto", target.label)],
                        predecessors={block.label},
                        synthetic=True,
                    )
                    blocks[new_label] = split_block
                    edge.target = new_label
                    target.predecessors.discard(block.label)
                    target.predecessors.add(new_label)
                    updated = True
                if updated:
                    break
            if not updated:
                break

    def _fold_redundant_branches(self, blocks: MutableMapping[str, _MutableBlock]) -> None:
        for block in blocks.values():
            if not isinstance(
                block.terminator,
                (IRIf, IRTestSetBranch, IRFlagCheck, IRFunctionPrologue),
            ):
                continue
            then_target = None
            else_target = None
            for edge in block.edges:
                if edge.kind.endswith("then"):
                    then_target = edge.target
                elif edge.kind.endswith("else"):
                    else_target = edge.target
            if then_target is None or else_target is None:
                continue
            if then_target != else_target:
                continue
            block.edges = [_Edge("goto", then_target)]
            block.terminator = _GotoTerminator(target=then_target)

    # ------------------------------------------------------------------
    # dominator analysis
    # ------------------------------------------------------------------
    def _compute_dominator_info(
        self, entry: str, blocks: Mapping[str, _MutableBlock]
    ) -> DominatorInfo:
        succs, preds = self._build_succ_pred_maps(blocks)
        order = self._reverse_postorder(entry, succs)
        idom = self._compute_idoms(entry, order, preds)
        dominators = self._dominators_from_idom(idom)
        return DominatorInfo.freeze(entry, idom, dominators)

    def _compute_post_dominator_info(
        self, entry: str, blocks: Mapping[str, _MutableBlock]
    ) -> DominatorInfo:
        succs, preds = self._build_succ_pred_maps(blocks)
        exits = [label for label, block in blocks.items() if not block.edges]
        if not exits:
            exits = [entry]
        if len(exits) == 1:
            root = exits[0]
            rev_succs = {label: list(preds[label]) for label in blocks}
            rev_preds = {label: set() for label in rev_succs}
            for src, targets in rev_succs.items():
                for target in targets:
                    rev_preds[target].add(src)
            order = self._reverse_postorder(root, rev_succs)
            idom = self._compute_idoms(root, order, rev_preds)
            dominators = self._dominators_from_idom(idom)
            return DominatorInfo.freeze(root, idom, dominators)
        sink = "__post_exit__"
        rev_succs = {label: list(preds[label]) for label in blocks}
        rev_succs[sink] = exits
        rev_preds: Dict[str, Set[str]] = {label: set() for label in rev_succs}
        for src, targets in rev_succs.items():
            for target in targets:
                rev_preds.setdefault(target, set()).add(src)
        order = self._reverse_postorder(sink, rev_succs)
        idom = self._compute_idoms(sink, order, rev_preds)
        if sink in idom:
            del idom[sink]
        cleaned_idom: Dict[str, Optional[str]] = {}
        for label, parent in idom.items():
            if parent == sink:
                cleaned_idom[label] = None
            else:
                cleaned_idom[label] = parent
        dominators = self._dominators_from_idom(cleaned_idom)
        return DominatorInfo.freeze(sink, cleaned_idom, dominators)

    def _build_succ_pred_maps(
        self, blocks: Mapping[str, _MutableBlock]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
        succs: Dict[str, List[str]] = {}
        preds: Dict[str, Set[str]] = {}
        for label, block in blocks.items():
            succs[label] = [edge.target for edge in block.edges]
            preds.setdefault(label, set())
            for edge in block.edges:
                preds.setdefault(edge.target, set()).add(label)
        return succs, preds

    def _reverse_postorder(
        self, root: str, succs: Mapping[str, Sequence[str]]
    ) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for target in succs.get(node, []):
                dfs(target)
            order.append(node)

        dfs(root)
        order.reverse()
        return order

    def _compute_idoms(
        self,
        root: str,
        order: Sequence[str],
        preds: Mapping[str, Set[str]],
    ) -> Dict[str, Optional[str]]:
        idom: Dict[str, Optional[str]] = {root: None}
        for node in order:
            if node == root:
                continue
            idom[node] = None
        changed = True
        rpo = [node for node in order if node in idom]
        while changed:
            changed = False
            for node in rpo:
                if node == root:
                    continue
                candidates = [
                    pred
                    for pred in preds.get(node, set())
                    if pred in idom and (idom[pred] is not None or pred == root)
                ]
                if not candidates:
                    continue
                new_idom = candidates[0]
                for pred in candidates[1:]:
                    new_idom = self._intersect(pred, new_idom, idom)
                if idom.get(node) != new_idom:
                    idom[node] = new_idom
                    changed = True
        return idom

    def _intersect(
        self, finger_a: str, finger_b: str, idom: Mapping[str, Optional[str]]
    ) -> str:
        visited: Set[str] = set()
        a = finger_a
        while a is not None:
            visited.add(a)
            a = idom.get(a)
        b = finger_b
        while b is not None and b not in visited:
            b = idom.get(b)
        if b is None:
            return finger_b
        return b

    def _dominators_from_idom(
        self, idom: Mapping[str, Optional[str]]
    ) -> Dict[str, List[str]]:
        doms: Dict[str, List[str]] = {}
        for node in idom:
            doms[node] = []
        for node in idom:
            current = node
            while current is not None:
                doms[node].append(current)
                current = idom.get(current)
        return doms

    # ------------------------------------------------------------------
    # loop detection
    # ------------------------------------------------------------------
    def _identify_loops(
        self, dominators: DominatorInfo, blocks: Mapping[str, _MutableBlock]
    ) -> Tuple[ASTLoop, ...]:
        idom = dominators.immediate
        preds: Dict[str, Set[str]] = {}
        succs: Dict[str, List[str]] = {}
        for label, block in blocks.items():
            succs[label] = [edge.target for edge in block.edges]
            for edge in block.edges:
                preds.setdefault(edge.target, set()).add(label)
        loop_latches: Dict[Tuple[str, Tuple[str, ...]], Set[str]] = {}
        dom_sets = dominators.dominators
        for header in dominators.dominators:
            for pred in preds.get(header, set()):
                if header == pred:
                    continue
                pred_doms = dominators.dominators.get(pred, ())  # type: ignore[arg-type]
                if header not in pred_doms:
                    continue
                loop_blocks = tuple(
                    sorted(self._collect_loop(header, pred, preds, dom_sets))
                )
                key = (header, loop_blocks)
                loop_latches.setdefault(key, set()).add(pred)
        loops: List[ASTLoop] = []
        for (header, blocks_tuple), latches in loop_latches.items():
            loops.append(
                ASTLoop(
                    header=header,
                    latches=tuple(sorted(latches)),
                    blocks=blocks_tuple,
                )
            )
        return tuple(sorted(loops, key=lambda loop: (loop.header, loop.blocks)))

    def _collect_loop(
        self,
        header: str,
        latch: str,
        preds: Mapping[str, Set[str]],
        dom_sets: Mapping[str, Sequence[str]],
    ) -> List[str]:
        loop_nodes: Set[str] = {header}
        stack = [latch]
        while stack:
            node = stack.pop()
            if node in loop_nodes:
                continue
            loop_nodes.add(node)
            for pred in preds.get(node, set()):
                if pred == header:
                    loop_nodes.add(pred)
                    continue
                if header not in dom_sets.get(pred, ()):  # type: ignore[arg-type]
                    continue
                if pred not in loop_nodes:
                    stack.append(pred)
        return sorted(loop_nodes)

    # ------------------------------------------------------------------
    # freezing helpers
    # ------------------------------------------------------------------
    def _freeze_blocks(self, blocks: Mapping[str, _MutableBlock]) -> Tuple[ASTBlock, ...]:
        frozen: List[ASTBlock] = []
        for label in sorted(blocks.keys()):
            block = blocks[label]
            terminator = self._canonical_terminator(block)
            frozen.append(
                ASTBlock(
                    label=block.label,
                    start_offset=block.start_offset,
                    statements=tuple(block.statements),
                    terminator=terminator,
                    predecessors=tuple(sorted(block.predecessors)),
                    successors=tuple(sorted(edge.target for edge in block.edges)),
                    synthetic=block.synthetic,
                )
            )
        return tuple(frozen)

    def _canonical_terminator(self, block: _MutableBlock) -> ASTTerminator:
        targets = tuple(edge.target for edge in block.edges)
        term = block.terminator
        if isinstance(term, _GotoTerminator):
            return ASTTerminator(kind="goto", targets=targets)
        if isinstance(term, IRIf):
            detail = {"condition": term.condition}
            then_target = None
            else_target = None
            for edge in block.edges:
                if edge.kind.endswith("then"):
                    then_target = edge.target
                elif edge.kind.endswith("else"):
                    else_target = edge.target
            if then_target is not None and else_target is not None and then_target != else_target:
                return ASTTerminator(
                    kind="if",
                    targets=(then_target, else_target),
                    detail=detail,
                )
            if then_target is not None:
                return ASTTerminator(kind="goto", targets=(then_target,), detail=detail)
        if isinstance(term, IRTestSetBranch):
            detail = {
                "var": term.var,
                "expr": term.expr,
            }
            then_target = None
            else_target = None
            for edge in block.edges:
                if edge.kind.endswith("then"):
                    then_target = edge.target
                elif edge.kind.endswith("else"):
                    else_target = edge.target
            if then_target is not None and else_target is not None and then_target != else_target:
                return ASTTerminator(
                    kind="testset",
                    targets=(then_target, else_target),
                    detail=detail,
                )
            if then_target is not None:
                return ASTTerminator(kind="goto", targets=(then_target,), detail=detail)
        if isinstance(term, IRFlagCheck):
            detail = {"flag": term.flag}
            then_target = None
            else_target = None
            for edge in block.edges:
                if edge.kind.endswith("then"):
                    then_target = edge.target
                elif edge.kind.endswith("else"):
                    else_target = edge.target
            if then_target is not None and else_target is not None and then_target != else_target:
                return ASTTerminator(
                    kind="flag_check",
                    targets=(then_target, else_target),
                    detail=detail,
                )
            if then_target is not None:
                return ASTTerminator(kind="goto", targets=(then_target,), detail=detail)
        if isinstance(term, IRFunctionPrologue):
            detail = {"var": term.var, "expr": term.expr}
            then_target = None
            else_target = None
            for edge in block.edges:
                if edge.kind.endswith("then"):
                    then_target = edge.target
                elif edge.kind.endswith("else"):
                    else_target = edge.target
            if then_target is not None and else_target is not None and then_target != else_target:
                return ASTTerminator(
                    kind="function_prologue",
                    targets=(then_target, else_target),
                    detail=detail,
                )
            if then_target is not None:
                return ASTTerminator(kind="goto", targets=(then_target,), detail=detail)
        if isinstance(term, IRSwitchDispatch):
            return ASTTerminator(kind="switch", targets=targets, detail=term)
        if isinstance(term, IRReturn):
            return ASTTerminator(kind="return", targets=(), detail=term)
        if isinstance(term, IRTailcallReturn):
            return ASTTerminator(kind="tailcall_return", targets=(), detail=term)
        if isinstance(term, IRTailCall):
            return ASTTerminator(kind="tailcall", targets=targets, detail=term)
        if isinstance(term, IRCallReturn):
            predicate = getattr(term, "predicate", None)
            detail = term
            if predicate is not None:
                detail = predicate.describe()
                return ASTTerminator(
                    kind="predicate",
                    targets=targets,
                    detail=detail,
                )
            return ASTTerminator(kind="call_return", targets=targets, detail=term)
        if isinstance(term, IRTerminator):
            if len(targets) == 1:
                return ASTTerminator(kind="goto", targets=targets, detail=term)
            return ASTTerminator(kind="terminator", targets=targets, detail=term)
        if not targets:
            return ASTTerminator(kind="halt", targets=targets, detail=term)
        return ASTTerminator(kind="goto", targets=targets, detail=term)

