"""Intermediate representation builder for MBC scripts."""

from .model import (
    IRBasicBlock,
    IRBuildArray,
    IRBuildMap,
    IRBuildTuple,
    IRCall,
    IRIf,
    IRLiteral,
    IRLoad,
    IRProgram,
    IRReturn,
    IRSlot,
    IRStore,
    IRTemp,
    IRTestSetBranch,
    MemSpace,
    NormalizerMetrics,
)
from .pipeline import build_segment_ir, render_program

__all__ = [
    "IRBasicBlock",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRCall",
    "IRIf",
    "IRLiteral",
    "IRLoad",
    "IRProgram",
    "IRReturn",
    "IRSlot",
    "IRStore",
    "IRTemp",
    "IRTestSetBranch",
    "MemSpace",
    "NormalizerMetrics",
    "build_segment_ir",
    "render_program",
]
