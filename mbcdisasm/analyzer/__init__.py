"""High level pipeline analysis toolkit.

The :mod:`mbcdisasm.analyzer` package provides the infrastructure required to
convert a raw instruction stream into an annotated collection of *pipeline
blocks*.  A pipeline block is a short sequence of instructions – typically
between two and six words – that together implement a well defined operation in
Sphere's Lua-inspired virtual machine.  Historically the project offered only a
single-pass disassembler that printed annotated mnemonics.  While helpful for
manual inspection, the output did not expose the higher level structure needed
for automated rewriting or decompilation.  The new analyzer closes that gap by
extracting common execution forms such as literal loaders, push chains, tail
call preparation and control flow tests.

The package is intentionally opinionated: it encodes the heuristics that were
observed in multiple megabytes of real `.mbc` bytecode and combines them with a
stack-aware scoring system.  The resulting analysis is deterministic and does
not rely on speculative execution.  Users can feed the pipeline analyser with
instruction words produced by :func:`mbcdisasm.instruction.read_instructions`
and will receive structured blocks that are ready for higher level translation.

The public surface of the module is intentionally small and revolves around the
:class:`~mbcdisasm.analyzer.pipeline.PipelineAnalyzer` class.  A typical workflow
looks as follows::

    from mbcdisasm import KnowledgeBase
    from mbcdisasm.analyzer import PipelineAnalyzer
    from mbcdisasm.instruction import read_instructions

    instructions, _ = read_instructions(segment_bytes, base_offset)
    knowledge = KnowledgeBase.load(manual_annotations_path)
    analyzer = PipelineAnalyzer(knowledge)
    report = analyzer.analyse_segment(instructions)

    for block in report.blocks:
        print(block.describe())

The :class:`~mbcdisasm.analyzer.report.PipelineReport` returned by the analyser
contains a curated view of the segment.  Each block records its stack delta,
confidence, dominant instruction kinds and context.  Consumers may use this
information to build higher level control flow graphs or to detect suspicious
patterns that need manual review.
"""

from .pipeline import PipelineAnalyzer
from .report import PipelineBlock, PipelineReport

__all__ = ["PipelineAnalyzer", "PipelineBlock", "PipelineReport"]
