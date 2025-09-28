"""High-level package entry point for the Sphere MBC disassembler toolkit."""

from .adb import SegmentIndex
from .mbc import MbcContainer, Segment
from .instruction import InstructionWord
from .disassembler import Disassembler
from .analysis import Analyzer, EmulationQualityMetrics
from .knowledge import (
    AnnotationUpdate,
    KnowledgeBase,
    MergeReport,
    ProfileAssessment,
    ReviewTask,
    SemanticNamingReport,
    StackObservation,
)
from .cfg import ControlFlowGraphBuilder, ControlFlowGraph
from .ir import IRBuilder, IRProgram, write_ir_programs
from .emulator import Emulator, EmulationReport, write_emulation_reports
from .stack_model import StackDeltaEstimate, StackDeltaModeler
from .ast import LuaReconstructor
from .highlevel import HighLevelFunction, HighLevelReconstructor
from .segment_classifier import SegmentClassifier

__all__ = [
    "SegmentIndex",
    "MbcContainer",
    "Segment",
    "InstructionWord",
    "Disassembler",
    "Analyzer",
    "EmulationQualityMetrics",
    "KnowledgeBase",
    "AnnotationUpdate",
    "ProfileAssessment",
    "MergeReport",
    "ReviewTask",
    "SemanticNamingReport",
    "StackObservation",
    "ControlFlowGraphBuilder",
    "ControlFlowGraph",
    "IRBuilder",
    "IRProgram",
    "Emulator",
    "EmulationReport",
    "write_ir_programs",
    "write_emulation_reports",
    "StackDeltaEstimate",
    "StackDeltaModeler",
    "LuaReconstructor",
    "HighLevelFunction",
    "HighLevelReconstructor",
    "SegmentClassifier",
]
