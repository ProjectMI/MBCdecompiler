"""Helpers to serialise IR structures for offline analysis."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

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
)
from .normalizer import NormalizerMetrics, NormalizerResult


def serialize_result(result: NormalizerResult) -> Dict[str, Any]:
    """Convert a :class:`NormalizerResult` into a JSON-serialisable mapping."""

    return {"blocks": [serialize_block(block) for block in result.blocks]}


def serialize_block(block: IRBlock) -> Dict[str, Any]:
    """Serialise an :class:`IRBlock` into a dictionary."""

    return {
        "start_offset": block.start_offset,
        "nodes": [serialize_node(node) for node in block.nodes],
    }


def serialize_node(node: IRNode) -> Dict[str, Any]:
    """Serialise an IR node into a dictionary with explicit type tags."""

    if isinstance(node, IRLiteral):
        return {"op": "literal", "value": node.value, "size": node.size}
    if isinstance(node, IRBuildArray):
        return {
            "op": "build_array",
            "elements": [serialize_node(element) for element in node.elements],
        }
    if isinstance(node, IRBuildMap):
        return {
            "op": "build_map",
            "pairs": [
                {"key": serialize_node(key), "value": serialize_node(value)}
                for key, value in node.pairs
            ],
        }
    if isinstance(node, IRBuildTuple):
        return {
            "op": "build_tuple",
            "elements": [serialize_node(element) for element in node.elements],
        }
    if isinstance(node, IRCall):
        return {
            "op": "call",
            "target": node.target,
            "args": [serialize_node(arg) for arg in node.args],
            "tail": node.tail,
        }
    if isinstance(node, IRReturn):
        return {"op": "return", "arity": node.arity}
    if isinstance(node, IRIf):
        return {
            "op": "if",
            "predicate": serialize_node(node.predicate),
            "then_target": node.then_target,
            "else_target": node.else_target,
        }
    if isinstance(node, IRTestSetBranch):
        return {
            "op": "testset_branch",
            "var": node.var,
            "expr": serialize_node(node.expr),
            "then_target": node.then_target,
            "else_target": node.else_target,
        }
    if isinstance(node, IRLoad):
        return {
            "op": "load",
            "slot": serialize_slot(node.slot),
        }
    if isinstance(node, IRStore):
        return {
            "op": "store",
            "slot": serialize_slot(node.slot),
            "value": serialize_node(node.value),
        }
    raise TypeError(f"unsupported IR node type: {type(node)!r}")


def serialize_slot(slot: IRSlot) -> Dict[str, Any]:
    """Serialise an :class:`IRSlot` into a simple mapping."""

    return {"space": slot.space.name.lower(), "index": slot.index}


def serialize_metrics(metrics: NormalizerMetrics) -> Dict[str, int]:
    """Serialise :class:`NormalizerMetrics` into a plain dictionary."""

    return {field.name: getattr(metrics, field.name) for field in fields(NormalizerMetrics)}


__all__ = [
    "serialize_block",
    "serialize_node",
    "serialize_metrics",
    "serialize_result",
    "serialize_slot",
]
