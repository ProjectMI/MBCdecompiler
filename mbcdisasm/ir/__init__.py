"""Convenience exports for the normalisation intermediate representation."""

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
from .normalizer import Normalizer, NormalizerMetrics, NormalizerResult
from .raw import RawBlock, RawInstruction, RawProgram, parse_stream

__all__ = [
    "IRBlock",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRCall",
    "IRIf",
    "IRLiteral",
    "IRLoad",
    "IRNode",
    "IRReturn",
    "IRSlot",
    "IRStore",
    "IRTestSetBranch",
    "MemSpace",
    "Normalizer",
    "NormalizerMetrics",
    "NormalizerResult",
    "RawInstruction",
    "RawBlock",
    "RawProgram",
    "parse_stream",
]
