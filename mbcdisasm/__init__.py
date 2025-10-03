"""Public package exports for the stripped-down MBC disassembler."""

from .adb import SegmentDescriptor, SegmentIndex
from .disassembler import Disassembler
from .instruction import InstructionWord
from .knowledge import KnowledgeBase, OpcodeInfo
from .ir import (
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
    Normalizer,
    NormalizerMetrics,
    NormalizerResult,
)
from .mbc import MbcContainer, Segment

__all__ = [
    "SegmentDescriptor",
    "SegmentIndex",
    "InstructionWord",
    "Disassembler",
    "KnowledgeBase",
    "OpcodeInfo",
    "MbcContainer",
    "Segment",
    "IRNode",
    "IRLiteral",
    "IRCall",
    "IRReturn",
    "IRBuildArray",
    "IRBuildMap",
    "IRBuildTuple",
    "IRIf",
    "IRTestSetBranch",
    "IRBlock",
    "IRLoad",
    "IRStore",
    "IRSlot",
    "MemSpace",
    "Normalizer",
    "NormalizerResult",
    "NormalizerMetrics",
]
