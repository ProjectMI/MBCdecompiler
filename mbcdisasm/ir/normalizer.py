"""Normalisation pipeline that turns raw opcodes into IR nodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, MutableSequence, Optional, Sequence, Tuple, Union

from ..instruction import InstructionWord
from ..knowledge import KnowledgeBase
from .nodes import (
    IRBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRIf,
    IRLiteral,
    IRLoad,
    IRNode,
    IRReturn,
    IRSlot,
    IRStore,
    IRTestSetBranch,
    MemSpace,
)
from .raw import RawInstruction, parse_stream


@dataclass(frozen=True)
class NormalizerMetrics:
    """Summary statistics for a normalisation run."""

    calls: int
    tail_calls: int
    returns: int
    aggregates: int
    testset_branches: int
    if_branches: int
    loads: int
    stores: int
    reduce_replaced: int
    raw_remaining: int


@dataclass(frozen=True)
class NormalizerResult:
    """Final product of a normalisation pass."""

    blocks: Tuple[IRBlock, ...]
    metrics: NormalizerMetrics


Token = Union[IRNode, RawInstruction]


class Normalizer:
    """Build the canonical IR described by the macro signature instructions."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def normalise(
        self, words: Sequence[InstructionWord]
    ) -> NormalizerResult:
        program = parse_stream(words, self.knowledge)
        blocks: List[IRBlock] = []
        metrics_counter = Counter()

        for block in program.blocks:
            tokens: List[Token] = [self._initial_token(inst) for inst in block.instructions]
            tokens = self._merge_calls(tokens, metrics_counter)
            tokens, reduced = self._collapse_literals(tokens, metrics_counter)
            metrics_counter["reduce_replaced"] += reduced
            tokens = self._lift_branches(tokens, metrics_counter)
            tokens = self._map_indirect(tokens, metrics_counter)
            ir_nodes = [token for token in tokens if isinstance(token, IRNode)]
            metrics_counter["raw_remaining"] += sum(
                1 for token in tokens if not isinstance(token, IRNode)
            )
            start_offset = (
                block.instructions[0].word.offset if block.instructions else None
            )
            blocks.append(IRBlock.from_nodes(ir_nodes, start_offset=start_offset))

        metrics = NormalizerMetrics(
            calls=metrics_counter.get("calls", 0),
            tail_calls=metrics_counter.get("tail_calls", 0),
            returns=metrics_counter.get("returns", 0),
            aggregates=metrics_counter.get("aggregates", 0),
            testset_branches=metrics_counter.get("testset_branches", 0),
            if_branches=metrics_counter.get("if_branches", 0),
            loads=metrics_counter.get("loads", 0),
            stores=metrics_counter.get("stores", 0),
            reduce_replaced=metrics_counter.get("reduce_replaced", 0),
            raw_remaining=metrics_counter.get("raw_remaining", 0),
        )
        return NormalizerResult(blocks=tuple(blocks), metrics=metrics)

    # ------------------------------------------------------------------
    # token initialisation
    # ------------------------------------------------------------------
    def _initial_token(self, instruction: RawInstruction) -> Token:
        if instruction.category and instruction.category.lower() in {"literal", "push"}:
            return IRLiteral(instruction.operand)
        return instruction

    # ------------------------------------------------------------------
    # call/return pass
    # ------------------------------------------------------------------
    def _merge_calls(
        self, tokens: Sequence[Token], metrics: Counter
    ) -> List[Token]:
        result: List[Token] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if isinstance(token, RawInstruction) and self._is_call_opcode(token):
                args = self._collect_arguments(result, token)
                call = IRCall.from_args(token.operand, args, tail=False)
                metrics["calls"] += 1
                tail, consumed, ret = self._maybe_tail_return(tokens, idx + 1, metrics)
                if tail:
                    call = IRCall.from_args(token.operand, args, tail=True)
                result.append(call)
                if ret is not None:
                    result.append(ret)
                idx += consumed
                continue
            if isinstance(token, RawInstruction) and self._is_return(token):
                arity = self._return_arity(token)
                metrics["returns"] += 1
                result.append(IRReturn(arity))
                idx += 1
                continue
            result.append(token)
            idx += 1
        return result

    def _collect_arguments(
        self, stack: MutableSequence[Token], token: RawInstruction
    ) -> List[IRNode]:
        arg_count = self._argument_count(token)
        args: List[IRNode] = []
        if arg_count == 0:
            while stack:
                candidate = stack[-1]
                if not isinstance(candidate, IRNode) or not self._is_argument_node(candidate):
                    break
                args.append(candidate)
                stack.pop()
            args.reverse()
            return args
        while stack and arg_count != 0:
            candidate = stack[-1]
            if not isinstance(candidate, IRNode) or not self._is_argument_node(candidate):
                break
            args.append(candidate)
            stack.pop()
            arg_count -= 1
        args.reverse()
        return args

    def _is_argument_node(self, node: IRNode) -> bool:
        return isinstance(
            node,
            (
                IRLiteral,
                IRBuildArray,
                IRBuildMap,
                IRBuildTuple,
                IRLoad,
            ),
        )

    def _argument_count(self, token: RawInstruction) -> int:
        if token.info is not None:
            count = token.info.attributes.get("arg_count") if token.info.attributes else None
            if isinstance(count, int):
                return max(0, count)
        if token.stack_delta < 0:
            return abs(token.stack_delta)
        return 0

    def _maybe_tail_return(
        self, tokens: Sequence[Token], start: int, metrics: Counter
    ) -> Tuple[bool, int, Optional[IRReturn]]:
        idx = start
        consumed = 1
        tail = False
        ret: Optional[IRReturn] = None
        while idx < len(tokens):
            token = tokens[idx]
            if isinstance(token, RawInstruction) and self._is_return(token):
                arity = self._return_arity(token)
                metrics["returns"] += 1
                metrics["tail_calls"] += 1
                tail = True
                ret = IRReturn(arity)
                consumed = idx - start + 2
                break
            if isinstance(token, RawInstruction) and token.category == "service":
                idx += 1
                continue
            break
        return tail, consumed, ret

    def _is_call_opcode(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        if "call" in mnemonic:
            return True
        if token.control_flow == "call":
            return True
        return False

    def _return_arity(self, token: RawInstruction) -> int:
        if token.info is not None and token.info.attributes:
            arity = token.info.attributes.get("return_arity")
            if isinstance(arity, int):
                return max(0, arity)
        if token.stack_delta < 0:
            return abs(token.stack_delta)
        return 0

    def _is_return(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        if mnemonic.startswith("return"):
            return True
        return token.control_flow == "return"

    # ------------------------------------------------------------------
    # literal collapse pass
    # ------------------------------------------------------------------
    def _collapse_literals(
        self, tokens: Sequence[Token], metrics: Counter
    ) -> Tuple[List[Token], int]:
        result: List[Token] = []
        idx = 0
        replaced = 0
        while idx < len(tokens):
            token = tokens[idx]
            if isinstance(token, IRLiteral):
                literals: List[IRLiteral] = [token]
                j = idx + 1
                while j < len(tokens) and isinstance(tokens[j], IRLiteral):
                    literals.append(tokens[j])
                    j += 1
                reduce_tokens: List[RawInstruction] = []
                while j < len(tokens):
                    candidate = tokens[j]
                    if isinstance(candidate, RawInstruction) and self._is_reduce(candidate):
                        reduce_tokens.append(candidate)
                        j += 1
                        continue
                    break
                if reduce_tokens:
                    aggregate = self._build_aggregate(literals, reduce_tokens)
                    result.append(aggregate)
                    metrics["aggregates"] += 1
                    replaced += len(reduce_tokens)
                    idx = j
                    continue
            result.append(token)
            idx += 1
        return result, replaced

    def _is_reduce(self, token: RawInstruction) -> bool:
        if token.category and token.category.lower() == "reduce":
            return True
        return token.mnemonic.lower().startswith("reduce")

    def _build_aggregate(
        self, literals: Sequence[IRLiteral], reduces: Sequence[RawInstruction]
    ) -> IRNode:
        primary = reduces[0].mnemonic.lower()
        if "pair" in primary and len(literals) % 2 == 0:
            pairs = []
            for idx in range(0, len(literals), 2):
                pairs.append((literals[idx], literals[idx + 1]))
            return IRBuildMap.from_pairs(pairs)
        if len(literals) == len(reduces) + 1:
            return IRBuildArray.from_sequence(literals)
        return IRBuildTuple.from_sequence(literals)

    # ------------------------------------------------------------------
    # branch lifting pass
    # ------------------------------------------------------------------
    def _lift_branches(
        self, tokens: Sequence[Token], metrics: Counter
    ) -> List[Token]:
        result: List[Token] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if isinstance(token, RawInstruction) and self._is_testset(token):
                expr = self._pop_last_node(result)
                var = token.info.attributes.get("assign") if token.info and token.info.attributes else None
                name = str(var or f"t{token.operand:04X}")
                branch = IRTestSetBranch(
                    var=name,
                    expr=expr or IRLiteral(0),
                    then_target=self._branch_target(token, "then"),
                    else_target=self._branch_target(token, "else"),
                )
                metrics["testset_branches"] += 1
                result.append(branch)
                idx += 1
                continue
            if isinstance(token, RawInstruction) and self._is_if_branch(token):
                expr = self._pop_last_node(result)
                branch = IRIf(
                    predicate=expr or IRLiteral(0),
                    then_target=self._branch_target(token, "then"),
                    else_target=self._branch_target(token, "else"),
                )
                metrics["if_branches"] += 1
                result.append(branch)
                idx += 1
                continue
            result.append(token)
            idx += 1
        return result

    def _is_testset(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        if "testset" in mnemonic or mnemonic.startswith("test_"):
            return True
        return token.category == "testset"

    def _is_if_branch(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        if mnemonic.startswith("if") or token.control_flow == "branch":
            return True
        return False

    def _branch_target(self, token: RawInstruction, key: str) -> int:
        if token.info and token.info.attributes:
            value = token.info.attributes.get(f"{key}_target")
            if isinstance(value, int):
                return value
        return token.operand

    def _pop_last_node(self, stack: MutableSequence[Token]) -> Optional[IRNode]:
        if stack and isinstance(stack[-1], IRNode):
            node = stack.pop()
            assert isinstance(node, IRNode)
            return node
        return None

    # ------------------------------------------------------------------
    # indirect mapping pass
    # ------------------------------------------------------------------
    def _map_indirect(
        self, tokens: Sequence[Token], metrics: Counter
    ) -> List[Token]:
        result: List[Token] = []
        for token in tokens:
            if isinstance(token, RawInstruction) and self._is_indirect(token):
                slot = IRSlot(space=self._classify_space(token.operand), index=token.operand)
                if self._is_store(token):
                    value = self._pop_last_node(result) or IRLiteral(0)
                    result.append(IRStore(slot=slot, value=value))
                    metrics["stores"] += 1
                else:
                    result.append(IRLoad(slot=slot))
                    metrics["loads"] += 1
                continue
            result.append(token)
        return result

    def _is_indirect(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        if "indirect" in mnemonic:
            return True
        return token.category in {"indirect", "indirect_load", "indirect_store"}

    def _is_store(self, token: RawInstruction) -> bool:
        mnemonic = token.mnemonic.lower()
        return "store" in mnemonic or token.category == "indirect_store"

    def _classify_space(self, operand: int) -> MemSpace:
        if operand < 0x100:
            return MemSpace.FRAME
        if operand < 0x8000:
            return MemSpace.GLOBAL
        return MemSpace.CONST
