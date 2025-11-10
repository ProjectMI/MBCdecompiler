"""Construction of the high level AST from the normalised IR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..ir.model import (
    IRBlock,
    IRCall,
    IRCallReturn,
    IRCallCleanup,
    IRFunctionCfg,
    IRIf,
    IRLiteral,
    IRLiteralChunk,
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
    IRIndirectStore,
    IRAbiEffect,
)
from ..ir.cfg import analyse_segments
from ..constants import IO_SLOT, RET_MASK
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

        self._fold_function_templates(functions)

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
    # template folding
    # ------------------------------------------------------------------
    def _fold_function_templates(self, functions: List[ASTFunction]) -> None:
        template_groups: Dict[Tuple[str, Tuple[Tuple, ...]], List[int]] = {}
        for index, function in enumerate(functions):
            template_name = self._identify_bank_initialisation_template(function)
            if template_name is None:
                continue
            signature = self._function_signature(function)
            template_groups.setdefault((template_name, signature), []).append(index)

        removal_indices: List[int] = []

        for (template_name, _signature), indices in template_groups.items():
            if len(indices) <= 1:
                continue
            canonical_index = indices[0]
            canonical = functions[canonical_index]
            alias_entries: List[ASTFunctionAlias] = []
            canonical_alias = ASTFunctionAlias(
                name=canonical.name,
                segment_index=canonical.segment_index,
                entry_block=canonical.entry_block,
                entry_offset=canonical.entry_offset,
            )
            clone_aliases: List[ASTFunctionAlias] = []
            for clone_index in indices[1:]:
                clone = functions[clone_index]
                clone_aliases.append(
                    ASTFunctionAlias(
                        name=clone.name,
                        segment_index=clone.segment_index,
                        entry_block=clone.entry_block,
                        entry_offset=clone.entry_offset,
                    )
                )
                removal_indices.append(clone_index)
            clone_aliases.sort(
                key=lambda alias: (alias.segment_index, alias.entry_offset, alias.name)
            )
            alias_entries.append(canonical_alias)
            alias_entries.extend(clone_aliases)
            updated = replace(
                canonical,
                name=template_name,
                aliases=tuple(alias_entries),
            )
            functions[canonical_index] = updated

        for index in sorted(set(removal_indices), reverse=True):
            del functions[index]

    def _function_signature(self, function: ASTFunction) -> Tuple[Tuple, ...]:
        signature: List[Tuple] = []
        for block in function.blocks:
            statements = []
            for statement in block.statements:
                describe = getattr(statement, "describe", None)
                if callable(describe):
                    statements.append((type(statement).__name__, describe()))
                else:
                    statements.append((type(statement).__name__, repr(statement)))
            terminator_detail = block.terminator.describe()
            signature.append(
                (
                    len(signature),
                    tuple(statements),
                    block.terminator.kind,
                    terminator_detail,
                )
            )
        return tuple(signature)

    def _identify_bank_initialisation_template(
        self, function: ASTFunction
    ) -> Optional[str]:
        if function.name != "auto_0":
            return None
        if len(function.blocks) != 1:
            return None
        block = function.blocks[0]
        if block.successors:
            return None
        statements = block.statements
        if len(statements) != 9:
            return None
        if not self._matches_frame_reset(statements[0]):
            return None
        if not self._matches_bank_helper_call(statements[1], target=0x04F0):
            return None
        if not self._matches_frame_write(statements[2], operand=0x4AA2):
            return None
        if not isinstance(statements[3], IRLiteral) or statements[3].value != IO_SLOT:
            return None
        if not self._matches_frame_reset(statements[4]):
            return None
        if not self._matches_bank_helper_call(statements[5], target=0x0EF0):
            return None
        if not isinstance(statements[6], IRLiteralChunk):
            return None
        if not self._matches_bank_flag_store(statements[7]):
            return None
        if not isinstance(statements[8], IRLiteral) or statements[8].value != RET_MASK:
            return None
        if block.terminator.kind != "tailcall":
            return None
        detail = block.terminator.detail
        if not isinstance(detail, IRTailCall):
            return None
        call = detail.call
        if call.target != IO_SLOT:
            return None
        if not self._call_has_return_mask(call, RET_MASK):
            return None
        return "template.bank_init_trampoline"

    def _matches_frame_reset(self, node: IRNode) -> bool:
        if not isinstance(node, IRCallCleanup):
            return False
        return any(step.category == "frame.reset" and step.operand == 0 for step in node.steps)

    def _matches_frame_write(self, node: IRNode, operand: int) -> bool:
        if not isinstance(node, IRCallCleanup):
            return False
        return any(step.category == "frame.write" and step.operand == operand for step in node.steps)

    def _matches_bank_helper_call(self, node: IRNode, target: int) -> bool:
        if not isinstance(node, IRCall):
            return False
        if not node.tail:
            return False
        if node.target != target:
            return False
        return self._call_has_return_mask(node, 0x0030)

    def _matches_bank_flag_store(self, node: IRNode) -> bool:
        if not isinstance(node, IRIndirectStore):
            return False
        if node.value != "bool1":
            return False
        if node.ref is None:
            return False
        if node.ref.page != 0xE4 or node.ref.offset != 0x01:
            return False
        return True

    def _call_has_return_mask(self, node: IRCall, mask: int) -> bool:
        for effect in node.abi_effects:
            if isinstance(effect, IRAbiEffect) and effect.kind == "return_mask":
                if effect.operand == mask:
                    return True
        return False

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

