#!/usr/bin/env python3
"""Command-line interface for the Sphere MBC disassembler."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, List

from mbcdisasm import (
    Analyzer,
    Disassembler,
    KnowledgeBase,
    ManualSemanticAnalyzer,
    MbcContainer,
)
from mbcdisasm.ast import LuaReconstructor
from mbcdisasm.cfg import ControlFlowGraphBuilder, render_cfgs
from mbcdisasm.emulator import Emulator, render_reports, write_emulation_reports
from mbcdisasm.ir import IRBuilder, render_ir_programs, write_ir_programs
from mbcdisasm.branch_analysis import render_branch_report
from mbcdisasm.branch_patterns import analyse_branch_patterns, render_branch_patterns
from mbcdisasm.stack_analysis import render_stack_seed_report
from mbcdisasm.segment_classifier import SegmentClassifier


@dataclass(frozen=True)
class DisassemblyPlan:
    """Derived configuration used to drive the CLI workflow."""

    listing_path: Path
    analysis_path: Optional[Path]
    cfg_path: Optional[Path]
    ir_path: Optional[Path]
    ast_path: Optional[Path]
    emulation_path: Optional[Path]
    branch_report_path: Optional[Path]
    stack_report_path: Optional[Path]
    knowledge_path: Path
    segment_indices: tuple[int, ...]
    missing_segments: tuple[int, ...]
    max_instructions: Optional[int]
    update_knowledge: bool

    @classmethod
    def from_args(cls, container: MbcContainer, args: argparse.Namespace) -> "DisassemblyPlan":
        segment_indices = tuple(args.segments or ())
        available = {segment.index for segment in container.segments()}
        missing = tuple(index for index in segment_indices if index not in available)
        listing_path = args.disasm_out or container.path.with_suffix(".disasm.txt")
        return cls(
            listing_path=listing_path,
            analysis_path=args.analysis_out,
            cfg_path=args.cfg_out,
            ir_path=args.ir_out,
            ast_path=args.ast_out,
            emulation_path=args.emulation_out,
            branch_report_path=args.branch_report_out,
            stack_report_path=args.stack_report_out,
            knowledge_path=args.knowledge_base,
            segment_indices=segment_indices,
            missing_segments=missing,
            max_instructions=args.max_instr,
            update_knowledge=args.update_knowledge,
        )

    def outputs(self) -> Iterable[tuple[str, Path]]:
        """Yield (label, path) pairs for every enabled output."""

        yield ("listing", self.listing_path)
        if self.analysis_path:
            yield ("analysis", self.analysis_path)
        if self.cfg_path:
            yield ("cfg", self.cfg_path)
        if self.ir_path:
            yield ("ir", self.ir_path)
        if self.ast_path:
            yield ("ast", self.ast_path)
        if self.emulation_path:
            yield ("emulation", self.emulation_path)
        if self.branch_report_path:
            yield ("branch_report", self.branch_report_path)
        if self.stack_report_path:
            yield ("stack_report", self.stack_report_path)

    def selection_summary(self) -> str:
        if self.segment_indices:
            return ", ".join(str(index) for index in self.segment_indices)
        return "all code segments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adb", type=Path, help="Path to the companion .adb index file")
    parser.add_argument("mbc", type=Path, help="Path to the .mbc container")
    parser.add_argument(
        "--segment",
        type=int,
        action="append",
        dest="segments",
        help="Restrict disassembly to the selected segment indices",
    )
    parser.add_argument(
        "--max-instr",
        type=int,
        default=None,
        help="Truncate each segment after the specified instruction count",
    )
    parser.add_argument(
        "--opcode-limit",
        type=int,
        default=20,
        help="Maximum number of opcode entries to display in the coverage table",
    )
    parser.add_argument(
        "--disasm-out",
        type=Path,
        default=None,
        help="Override the default <mbc>.disasm.txt output path",
    )
    parser.add_argument(
        "--analysis-out",
        type=Path,
        help="Write a machine-readable JSON analysis report to the provided path",
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
        help="Location of the opcode knowledge base database",
    )
    parser.add_argument(
        "--update-knowledge",
        action="store_true",
        help="Merge the analysis results back into the knowledge base",
    )
    parser.add_argument(
        "--cfg-out",
        type=Path,
        help="Write a textual representation of the control-flow graphs",
    )
    parser.add_argument(
        "--ir-out",
        type=Path,
        help="Write the intermediate representation derived from the CFG",
    )
    parser.add_argument(
        "--ast-out",
        type=Path,
        help="Write the pseudo-Lua reconstruction to the provided path",
    )
    parser.add_argument(
        "--emulation-out",
        type=Path,
        help="Write stack emulation traces to the provided path",
    )
    parser.add_argument(
        "--branch-report-out",
        type=Path,
        help="Write a detailed branch analysis report",
    )
    parser.add_argument(
        "--stack-report-out",
        type=Path,
        help="Write stack seed diagnostics to the provided path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    knowledge = KnowledgeBase.load(args.knowledge_base)
    classifier = SegmentClassifier(knowledge)
    container = MbcContainer.load(args.mbc, args.adb, classifier=classifier)
    plan = DisassemblyPlan.from_args(container, args)

    render_cli_overview(container, plan)

    analyzer = Analyzer(knowledge)
    analysis = analyzer.analyze(container)

    semantic_analyzer = ManualSemanticAnalyzer(knowledge)

    merge_report = None
    semantic_report = None
    preview_count = 0
    if args.update_knowledge:
        merge_report = knowledge.merge_profiles(analysis.iter_profiles())
        updates = merge_report.updates
        if updates:
            preview = updates[:3]
            preview_count = len(preview)
            sample = ", ".join(
                f"{u.key}:{u.new_value} ({u.confidence:.2f})" for u in preview
            )
            if len(updates) > preview_count:
                sample += f", +{len(updates) - preview_count} more"
            print(f"knowledge updated: {sample}")
        conflicts = merge_report.conflicts()
        if conflicts:
            preview = ", ".join(c.key for c in conflicts[:5])
            if len(conflicts) > 5:
                preview += f", +{len(conflicts) - 5} more"
            print(f"stack conflicts detected: {preview}")
        if merge_report.review_tasks:
            preview = ", ".join(
                f"{task.key}:{task.reason}" for task in merge_report.review_tasks[:5]
            )
            if len(merge_report.review_tasks) > 5:
                preview += f", +{len(merge_report.review_tasks) - 5} more"
            print(f"queued review tasks: {preview}")
        semantic_report = knowledge.apply_semantic_annotations()
        if semantic_report.matches:
            semantic_preview = semantic_report.matches[:3]
            summary = ", ".join(
                f"{match.key}->{match.prototype} ({match.score:.2f})"
                for match in semantic_preview
            )
            if len(semantic_report.matches) > len(semantic_preview):
                summary += (
                    f", +{len(semantic_report.matches) - len(semantic_preview)} more"
                )
            print(f"semantic matches applied: {summary}")

    render_cli_summary(container, analysis, opcode_limit=args.opcode_limit, plan=plan)

    if plan.analysis_path:
        plan.analysis_path.write_text(
            json.dumps(serialize_analysis(analysis), indent=2),
            "utf-8",
        )

    Disassembler(knowledge).write_listing(
        container,
        plan.listing_path,
        segment_indices=plan.segment_indices,
        max_instructions=plan.max_instructions,
    )
    print(f"disassembly written to {plan.listing_path}")

    selected_segments = list(_selected_segments(container, plan.segment_indices))

    need_cfg = any(
        (
            plan.cfg_path,
            plan.ir_path,
            plan.ast_path,
            plan.branch_report_path,
            plan.stack_report_path,
        )
    )
    if need_cfg:
        cfg_builder = ControlFlowGraphBuilder(knowledge, semantic_analyzer=semantic_analyzer)
        cfgs = [
            cfg_builder.build(segment, max_instructions=args.max_instr)
            for segment in selected_segments
        ]
        if plan.cfg_path:
            plan.cfg_path.write_text(render_cfgs(cfgs), "utf-8")
            print(f"cfg written to {plan.cfg_path}")

    need_ir = any(
        (
            plan.ir_path,
            plan.ast_path,
            plan.branch_report_path,
            plan.stack_report_path,
        )
    )
    if need_ir:
        if not need_cfg:
            cfg_builder = ControlFlowGraphBuilder(knowledge, semantic_analyzer=semantic_analyzer)
            cfgs = [
                cfg_builder.build(segment, max_instructions=args.max_instr)
                for segment in selected_segments
            ]
        ir_builder = IRBuilder(knowledge)
        programs = [
            ir_builder.from_cfg(segment, cfg)
            for segment, cfg in zip(selected_segments, cfgs)
        ]
        if plan.ir_path:
            write_ir_programs(programs, plan.ir_path)
            print(f"ir written to {plan.ir_path}")
        if plan.ast_path:
            reconstructor = LuaReconstructor()
            lua_blobs = []
            for segment, program in zip(selected_segments, programs):
                function = reconstructor.from_ir(segment.index, program)
                lua_blobs.append(reconstructor.render(function).rstrip())
            plan.ast_path.write_text("\n\n".join(lua_blobs) + "\n", "utf-8")
            print(f"ast written to {plan.ast_path}")
        if plan.branch_report_path:
            report_sections: List[str] = []
            for segment, program in zip(selected_segments, programs):
                registry = program.branch_registry(knowledge)
                structure_report = render_branch_report(registry.structure).rstrip()
                pattern_registry = analyse_branch_patterns(registry)
                pattern_report = render_branch_patterns(pattern_registry).rstrip()
                section = "\n".join(
                    [
                        f"segment {segment.index} branch analysis:",
                        structure_report,
                        pattern_report,
                    ]
                )
                report_sections.append(section.strip())
            plan.branch_report_path.write_text("\n\n".join(report_sections) + "\n", "utf-8")
            print(f"branch report written to {plan.branch_report_path}")
        if plan.stack_report_path:
            stack_sections: List[str] = []
            for segment, program in zip(selected_segments, programs):
                report = render_stack_seed_report(program, knowledge).rstrip()
                stack_sections.append(
                    f"segment {segment.index} stack seed report:\n{report}"
                )
            plan.stack_report_path.write_text("\n\n".join(stack_sections) + "\n", "utf-8")
            print(f"stack seed report written to {plan.stack_report_path}")

    if plan.emulation_path:
        emulator = Emulator(
            knowledge,
            stack_modeler=analyzer.clone_stack_modeler(),
            semantic_analyzer=semantic_analyzer,
        )
        reports = emulator.simulate_container(
            selected_segments,
            max_instructions=plan.max_instructions,
        )
        write_emulation_reports(reports, plan.emulation_path)
        print(f"emulation report written to {plan.emulation_path}")

    detailed_updates: Sequence = ()
    if merge_report:
        detailed_updates = merge_report.updates[preview_count:]
    if semantic_report:
        detailed_updates = tuple(list(detailed_updates) + list(semantic_report.updates))

    if plan.update_knowledge:
        knowledge.save()
        classifier.save_profile()
        print(f"knowledge base updated: {knowledge.path}")
        if merge_report and detailed_updates:
            print("learned stack behaviour:")
            for update in detailed_updates[:10]:
                print(
                    "  "
                    f"{update.key:7s} {update.field}={update.new_value} "
                    f"conf={update.confidence:.2f} samples={update.samples}"
                    f" ({update.reason})"
                )
            if len(detailed_updates) > 10:
                remaining = len(detailed_updates) - 10
                print(f"  ... {remaining} more updates omitted ...")


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.adb, args.mbc):
        if not path.exists():
            raise SystemExit(f"missing input file: {path}")


def render_cli_overview(container: MbcContainer, plan: DisassemblyPlan) -> None:
    """Print a high-level view of the requested operation."""

    header = container.header
    banner = header.get("banner")

    print(f"container: {container.path.name}")
    if banner:
        print(f"  banner: {banner}")

    total_segments = sum(1 for _ in container.segments())
    code_segments = sum(1 for _ in container.iter_code_segments())
    print(f"segments available: {total_segments} total, {code_segments} classified as code")
    print(f"  selection: {plan.selection_summary()}")
    if plan.missing_segments:
        missing = ", ".join(str(index) for index in plan.missing_segments)
        print(f"  missing selections: {missing}")
    if plan.max_instructions is not None:
        print(f"  max instructions per segment: {plan.max_instructions}")

    print(f"knowledge base: {plan.knowledge_path}")
    print(
        "  updates: "
        + ("enabled" if plan.update_knowledge else "disabled"),
    )

    outputs = list(plan.outputs())
    if outputs:
        print("outputs:")
        for label, path in outputs:
            print(f"  {label:12s}: {path}")


def render_cli_summary(
    container: MbcContainer,
    analysis,
    opcode_limit: int,
    *,
    plan: Optional[DisassemblyPlan] = None,
) -> None:
    if plan is None:
        print(f"container: {container.path.name}")
    print(f"segments classified: {analysis.total_segments}")
    print(f"instructions decoded: {analysis.total_instructions}")

    classifications = {}
    for report in analysis.segment_reports:
        classifications.setdefault(report.classification, 0)
        classifications[report.classification] += 1
    for kind, count in sorted(classifications.items()):
        print(f"  {kind:10s}: {count}")

    if analysis.issues:
        print("issues detected:")
        for issue in analysis.issues:
            print(f"  - {issue}")

    hot = analysis.most_common(opcode_limit)
    if hot:
        print("opcode coverage:")
        for key, count in hot:
            print(f"  {key:7s} : {count:6d}")

    breakdown = analysis.confidence_breakdown()
    if breakdown:
        print("knowledge confidence:")
        for status, count in breakdown.items():
            print(f"  {status:12s}: {count:6d}")

    conflicts = analysis.conflicting_profiles()
    if conflicts:
        print("opcode conflicts:")
        for assessment in conflicts[:10]:
            print(
                f"  {assessment.key:7s} {assessment.notes} "
                f"(existing={assessment.existing_count} new={assessment.new_count})"
            )

    unknown_stack = analysis.unknown_stack_profiles()
    if unknown_stack:
        print("stack knowledge gaps:")
        for observation in unknown_stack[:10]:
            ratio = observation.unknown_ratio or 0.0
            print(
                f"  {observation.key:7s} unknown={ratio:.0%} samples={observation.total_samples}"
            )

    def _is_noisy(report) -> bool:
        """Return True when the segment has actionable issues."""

        for issue in report.issues:
            if issue.startswith("trailing_bytes="):
                continue
            return True
        return False

    noisy = [r for r in analysis.segment_reports if _is_noisy(r)]
    if noisy:
        print("problematic segments:")
        for report in noisy[:10]:
            suffix = ", ".join(report.issues)
            print(
                f"  segment {report.index:4d} offset=0x{report.start:06X} len={report.length:6d} ({suffix})"
            )


def serialize_analysis(analysis) -> dict:
    opcode_matrix = analysis.opcode_mode_matrix()
    opcode_entries = [
        {"opcode": opcode, "modes": modes}
        for opcode, modes in sorted(opcode_matrix.items())
    ]
    mode_names = sorted({mode for modes in opcode_matrix.values() for mode in modes})

    return {
        "total_segments": analysis.total_segments,
        "total_instructions": analysis.total_instructions,
        "issues": analysis.issues,
        "segments": [
            {
                "index": report.index,
                "start": report.start,
                "length": report.length,
                "classification": report.classification,
                "instruction_count": report.instruction_count,
                "issues": report.issues,
            }
            for report in analysis.segment_reports
        ],
        "opcode_profiles": {
            "modes": mode_names,
            "opcodes": opcode_entries,
        },
        "opcode_mode_matrix": opcode_matrix,
        "profile_assessments": {
            assessment.key: assessment.to_json()
            for assessment in analysis.profile_assessments
        },
        "stack_observations": {
            key: observation.to_json()
            for key, observation in analysis.stack_observations.items()
            if observation.total_samples
        },
        "emulation_quality": analysis.emulation_quality.to_json(),
    }


def _selected_segments(
    container: MbcContainer,
    segments: Optional[Sequence[int]],
):
    selected = tuple(segments or [])
    if selected:
        index_map = {segment.index: segment for segment in container.segments()}
        for index in selected:
            segment = index_map.get(index)
            if segment is not None:
                yield segment
        return

    yield from container.iter_code_segments()


if __name__ == "__main__":
    main()
