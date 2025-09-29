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
from .highlevel import FunctionMetadata, HighLevelFunction, HighLevelReconstructor
from .literal_sequences import (
    LiteralDescriptor,
    LiteralRun,
    LiteralRunReport,
    LiteralStatistics,
    build_literal_run_report,
    compute_literal_statistics,
    literal_report_to_dict,
    literal_statistics_to_dict,
)
from .lua_formatter import LuaRenderOptions
from .lua_literals import LuaLiteralFormatter
from .segment_classifier import SegmentClassifier
from .manual_semantics import (
    AnnotatedInstruction,
    InstructionSemantics,
    ManualSemanticAnalyzer,
    StackEffect,
)
from .vm_analysis import (
    VMBlockTrace,
    VMInstructionState,
    VMInstructionTrace,
    VMOperation,
    VMProgramTrace,
    VMLifetime,
    VirtualMachineAnalyzer,
    estimate_stack_io,
    analyze_block_lifetimes,
    analyze_program_lifetimes,
    format_vm_block_trace,
    render_value_lifetimes,
    render_vm_program,
    render_vm_traces,
    lifetimes_to_dict,
    lifetimes_to_json,
    vm_block_trace_to_dict,
    vm_block_trace_to_json,
    vm_program_trace_to_dict,
    vm_program_trace_to_json,
    summarise_program,
    count_operations,
)

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
    "FunctionMetadata",
    "LiteralDescriptor",
    "LiteralRun",
    "LiteralRunReport",
    "LiteralStatistics",
    "compute_literal_statistics",
    "build_literal_run_report",
    "literal_report_to_dict",
    "literal_statistics_to_dict",
    "LuaRenderOptions",
    "SegmentClassifier",
    "ManualSemanticAnalyzer",
    "InstructionSemantics",
    "AnnotatedInstruction",
    "StackEffect",
    "VirtualMachineAnalyzer",
    "VMOperation",
    "VMInstructionTrace",
    "VMInstructionState",
    "VMBlockTrace",
    "VMProgramTrace",
    "VMLifetime",
    "estimate_stack_io",
    "analyze_block_lifetimes",
    "analyze_program_lifetimes",
    "LuaLiteralFormatter",
    "format_vm_block_trace",
    "render_value_lifetimes",
    "render_vm_program",
    "render_vm_traces",
    "lifetimes_to_dict",
    "lifetimes_to_json",
    "vm_block_trace_to_dict",
    "vm_block_trace_to_json",
    "vm_program_trace_to_dict",
    "vm_program_trace_to_json",
    "summarise_program",
    "count_operations",
]
