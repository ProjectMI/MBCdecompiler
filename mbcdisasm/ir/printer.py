"""Utilities for serialising the normalised IR into a text format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .model import IRAbiFunctionReport, IRBlock, IRCfgBlock, IRProgram, IRSegment


class IRTextRenderer:
    """Render :class:`IRProgram` instances into a stable textual form."""

    def render(self, program: IRProgram) -> str:
        lines: List[str] = []
        lines.extend(self._render_string_pool(program))
        lines.extend(self._render_metrics(program))
        lines.extend(self._render_cfg(program))
        for segment in program.segments:
            lines.extend(self._render_segment(segment))
        return "\n".join(lines) + "\n"

    def write(self, program: IRProgram, output_path: Path) -> None:
        output_path.write_text(self.render(program), "utf-8")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _render_string_pool(self, program: IRProgram) -> Iterable[str]:
        yield "; string pool"
        if program.string_pool:
            for const in program.string_pool:
                yield const.describe()
        else:
            yield ";   (empty)"
        yield ""

    def _render_metrics(self, program: IRProgram) -> Iterable[str]:
        yield "; metrics"
        yield f";   normalizer: {program.metrics.describe()}"
        abi_metrics = program.abi_metrics
        if abi_metrics is None:
            yield ";   abi: unavailable"
            yield ""
            return

        coverage = f"{abi_metrics.return_mask_coverage:.2f}%"
        coverage_detail = f"{abi_metrics.masked_exits}/{abi_metrics.total_exits}"
        yield (
            f";   abi.return_mask.coverage={coverage}"
            f" ({coverage_detail})"
        )

        consistency = f"{abi_metrics.mask_consistency:.2f}%"
        total_functions = abi_metrics.total_functions
        consistent = total_functions - abi_metrics.inconsistent_functions
        yield (
            f";   abi.return_mask.consistency={consistency}"
            f" ({consistent}/{total_functions})"
        )

        yield from self._render_metric_violations(
            "abi.return_mask.missing", abi_metrics.missing_mask_functions
        )
        yield from self._render_metric_violations(
            "abi.return_mask.inconsistent", abi_metrics.inconsistent_mask_functions
        )
        yield ""

    def _render_metric_violations(
        self, label: str, reports: Iterable[IRAbiFunctionReport]
    ) -> Iterable[str]:
        reports = list(reports)
        if not reports:
            yield f";   {label}: none"
            return
        yield f";   {label}:"
        for report in reports:
            coverage = (
                f"{report.masked_exits}/{report.total_exits}"
                if report.total_exits
                else "0/0"
            )
            masks = report.describe_masks() or "none"
            yield (
                f";     segment={report.segment_index} entry=0x{report.entry_offset:06X} "
                f"coverage={coverage} masks=[{masks}] name={report.name}"
            )

    def _render_cfg(self, program: IRProgram) -> Iterable[str]:
        yield "; cfg"
        cfg = program.cfg
        if cfg is None or not cfg.functions:
            yield ";   (empty)"
            yield ""
            return
        for function in cfg.functions:
            yield (
                f"; function segment={function.segment_index} name={function.name} "
                f"entry={function.entry_block} offset=0x{function.entry_offset:06X}"
            )
            for block in function.blocks:
                yield from self._render_cfg_block(block)
        yield ""

    def _render_cfg_block(self, block: IRCfgBlock) -> Iterable[str]:
        terminator = block.terminator or "(none)"
        yield (
            f";   block {block.label} offset=0x{block.start_offset:06X} "
            f"terminator={terminator}"
        )
        for edge in block.edges:
            yield f";     {edge.kind} -> {edge.target}"

    def _render_segment(self, segment: IRSegment) -> Iterable[str]:
        header = (
            f"; segment {segment.index} offset=0x{segment.start:06X} "
            f"length={segment.length}"
        )
        yield header
        yield "; metrics: " + segment.metrics.describe(include_teardowns=False)
        for block in segment.blocks:
            yield from self._render_block(block)
        yield ""

    def _render_block(self, block: IRBlock) -> Iterable[str]:
        yield f"block {block.label} offset=0x{block.start_offset:06X}"
        if block.annotations:
            for note in block.annotations:
                yield f"  ; {note}"
        for node in block.nodes:
            describe = getattr(node, "describe", None)
            if callable(describe):
                yield f"  {describe()}"
            else:
                yield f"  {node!r}"


__all__ = ["IRTextRenderer"]
