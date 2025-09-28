import contextlib
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from mbc_disasm import main as cli_main, render_cli_summary
from mbcdisasm import (
    Analyzer,
    Disassembler,
    KnowledgeBase,
    MbcContainer,
    StackDeltaModeler,
)
from mbcdisasm.adb import SegmentDescriptor
from mbcdisasm.mbc import Segment
from mbcdisasm.ast import LuaReconstructor
from mbcdisasm.cfg import ControlFlowGraphBuilder
from mbcdisasm.emulator import Emulator, write_emulation_reports
from mbcdisasm.ir import IRBuilder, write_ir_programs
from mbcdisasm.knowledge import OpcodeProfile
from mbcdisasm.instruction import InstructionWord
from mbcdisasm.segment_classifier import SegmentClassifier
from unittest import mock

FIXTURE_MBC = Path("mbc/_main.mbc")
FIXTURE_ADB = Path("mbc/_main.adb")


class AnalysisPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        if not FIXTURE_MBC.exists() or not FIXTURE_ADB.exists():
            self.skipTest("fixtures missing")
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.knowledge_path = Path(self.tempdir.name) / "kb.json"
        self.knowledge = KnowledgeBase.load(self.knowledge_path)
        self.classifier = SegmentClassifier(self.knowledge)
        self.container = MbcContainer.load(
            FIXTURE_MBC,
            FIXTURE_ADB,
            classifier=self.classifier,
        )

    def test_stack_modeler_bitcount_heuristic(self) -> None:
        modeler = StackDeltaModeler()
        push_instr = InstructionWord(0, (0x10 << 24) | (0x0F << 16) | 0x0000)
        estimate = modeler.estimate_instruction(push_instr)
        self.assertTrue(modeler.is_confident(estimate))
        self.assertAlmostEqual(estimate.delta, 4.0)

        # Feeding the instruction through the segment model promotes it to known state.
        segment_estimates = modeler.model_segment([push_instr])
        self.assertTrue(modeler.is_confident(segment_estimates[0]))
        self.assertAlmostEqual(modeler.known_delta(push_instr.label()), 4.0)

        unknown_instr = InstructionWord(4, (0x80 << 24) | (0x00 << 16) | 0x0000)
        combined = modeler.model_segment([unknown_instr, push_instr])
        self.assertTrue(modeler.is_confident(combined[0]))
        self.assertAlmostEqual(combined[0].delta, -4.0)

    def test_analysis_counts_code_segments(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        classes = {report.classification for report in analysis.segment_reports}
        self.assertIn("code", classes)
        self.assertGreater(analysis.total_instructions, 0)

    def test_disassembler_listing_contains_known_mnemonics(self) -> None:
        disasm = Disassembler(self.knowledge)
        listing = disasm.generate_listing(self.container, max_instructions=3)
        self.assertIn("opcode_", listing)
        self.assertIn("segment", listing.splitlines()[0])

    def test_knowledge_update_roundtrip(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        self.knowledge.merge_profiles(analysis.iter_profiles())
        self.knowledge.save()
        reloaded = KnowledgeBase.load(self.knowledge_path)
        self.assertTrue(reloaded.get_profile("00:00").count >= 0)

    def test_segment_classifier_profile_roundtrip(self) -> None:
        profile_path = self.knowledge_path.with_name(
            f"{self.knowledge_path.stem}_segment_classifier.json"
        )
        if profile_path.exists():
            profile_path.unlink()

        self.classifier.weights["ascii_ratio"] = 2.5
        self.classifier.bias = -1.1
        self.classifier._profile_dirty = True  # type: ignore[attr-defined]
        self.classifier.save_profile()

        self.assertTrue(profile_path.exists(), "expected classifier profile to be saved")

        reloaded = SegmentClassifier(self.knowledge)
        self.assertAlmostEqual(reloaded.weights["ascii_ratio"], 2.5)
        self.assertAlmostEqual(reloaded.bias, -1.1)

    def test_profile_assessment_summary(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        self.assertTrue(analysis.profile_assessments)
        statuses = {assessment.status for assessment in analysis.profile_assessments}
        self.assertIn("unknown", statuses)
        breakdown = analysis.confidence_breakdown()
        self.assertGreaterEqual(sum(breakdown.values()), len(analysis.profile_assessments))

    def test_segments_with_trailing_bytes_are_processed(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        trailing = [
            report
            for report in analysis.segment_reports
            if any(issue.startswith("trailing_bytes=") for issue in report.issues)
        ]
        self.assertTrue(trailing)
        for report in trailing:
            self.assertGreater(report.instruction_count, 0)
            self.assertEqual(report.classification, "code")

    def test_data_segments_are_not_flagged_as_problematic(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        for report in analysis.segment_reports:
            if report.classification != "code":
                self.assertFalse(
                    report.issues,
                    msg=f"non-code segment {report.index} should not expose issues",
                )

    def test_cli_summary_omits_trailing_byte_only_segments(self) -> None:
        analysis = Analyzer(self.knowledge).analyze(self.container)
        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            render_cli_summary(self.container, analysis, opcode_limit=5)

        output = stdout.getvalue()
        self.assertNotIn("trailing_bytes=", output)
        self.assertIn("segments with trailing bytes", output)

    def test_merge_profiles_learns_stack_delta(self) -> None:
        kb_path = Path(self.tempdir.name) / "learn.json"
        knowledge = KnowledgeBase.load(kb_path)
        profile = OpcodeProfile("10:20")
        profile.count = 12
        profile.stack_deltas.update({1: 12})

        report = knowledge.merge_profiles([profile], min_samples=4, confidence_threshold=0.7)
        self.assertTrue(report.updates)
        ann = knowledge._annotations.get("10:20")
        self.assertIsNotNone(ann)
        self.assertEqual(ann["stack_delta"], 1)
        self.assertGreaterEqual(ann["stack_confidence"], 0.7)
        self.assertEqual(ann["stack_samples"], 12)
        self.assertAlmostEqual(knowledge.estimate_stack_delta("10:20"), 1.0)

    def test_analysis_derives_stack_delta_samples(self) -> None:
        kb_path = Path(self.tempdir.name) / "fresh.json"
        knowledge = KnowledgeBase.load(kb_path)
        analysis = Analyzer(knowledge).analyze(self.container)

        self.assertTrue(
            any(obs.known_samples > 0 for obs in analysis.stack_observations.values()),
            msg="expected at least one opcode to expose known stack samples",
        )

        report = knowledge.merge_profiles(
            analysis.iter_profiles(),
            min_samples=1,
            confidence_threshold=0.0,
        )

        self.assertTrue(
            any(update.field == "stack_delta" for update in report.updates),
            msg="stack delta modelling should record annotations",
        )

    def test_conflict_profiles_enqueue_review_tasks(self) -> None:
        kb_path = Path(self.tempdir.name) / "conflict.json"
        knowledge = KnowledgeBase.load(kb_path)
        existing = knowledge.get_profile("AA:00")
        existing.count = 8
        existing.stack_deltas.update({1: 8})
        knowledge._data.setdefault("opcode_modes", {})[existing.key] = existing.to_json()

        profile = OpcodeProfile("AA:00")
        profile.count = 4
        profile.stack_deltas.update({-1: 4})

        report = knowledge.merge_profiles([profile], min_samples=1, confidence_threshold=0.0)
        self.assertTrue(report.review_tasks)
        reasons = {task.reason for task in report.review_tasks}
        self.assertIn("stack_conflict", reasons)
        pending = knowledge.pending_review_tasks()
        self.assertTrue(any(task.key == "AA:00" for task in pending))

    def test_merge_profiles_learns_operand_hint(self) -> None:
        kb_path = Path(self.tempdir.name) / "operand.json"
        knowledge = KnowledgeBase.load(kb_path)
        profile = OpcodeProfile("BB:01")
        profile.count = 10
        profile.operand_types.update({"small": 10})

        report = knowledge.merge_profiles([profile], min_samples=2, confidence_threshold=0.6)
        annotations = knowledge._annotations.get("BB:01")
        self.assertIsNotNone(annotations)
        self.assertEqual(annotations["operand_hint"], "small")
        self.assertTrue(any(update.field == "operand_hint" for update in report.updates))

    def test_unknown_profiles_enqueue_review_tasks(self) -> None:
        kb_path = Path(self.tempdir.name) / "review.json"
        knowledge = KnowledgeBase.load(kb_path)
        profile = OpcodeProfile("CC:02")
        profile.count = 5
        profile.stack_deltas.update({"unknown": 5})

        report = knowledge.merge_profiles([profile], min_samples=3, confidence_threshold=0.75)
        self.assertTrue(report.review_tasks)
        task = report.review_tasks[0]
        self.assertEqual(task.reason, "stack_unknown")

    def test_emulator_uses_seeded_stack_deltas(self) -> None:
        raw = (0x10 << 24) | (0x00 << 16) | 0x0001
        data = raw.to_bytes(4, "big") * 2
        descriptor = SegmentDescriptor(index=99, start=0, end=len(data))
        segment = Segment(descriptor, data, "code")

        empty_path = Path(self.tempdir.name) / "emu_empty.json"
        empty_knowledge = KnowledgeBase.load(empty_path)
        empty_emulator = Emulator(empty_knowledge)
        empty_report = empty_emulator.simulate_segment(segment)
        self.assertEqual(
            empty_report.unknown_delta_instructions,
            empty_report.total_instructions,
            msg="expected all instructions to be unknown without annotations",
        )

        seeded_path = Path(self.tempdir.name) / "emu_seeded.json"
        seeded_knowledge = KnowledgeBase.load(seeded_path)
        seeded_knowledge._annotations.setdefault("10:00", {})["stack_delta"] = 2
        seeded_emulator = Emulator(seeded_knowledge)
        seeded_report = seeded_emulator.simulate_segment(segment)

        self.assertEqual(seeded_report.total_instructions, empty_report.total_instructions)
        self.assertLess(
            seeded_report.unknown_delta_instructions,
            empty_report.unknown_delta_instructions,
            msg="seeding from knowledge should reduce unknown delta counts",
        )
        self.assertAlmostEqual(seeded_report.unknown_delta_ratio, 0.0)
        for trace in seeded_report.traces:
            self.assertNotIn("unknown-delta", trace.warnings)

    def test_emulator_seeds_model_with_knowledge(self) -> None:
        kb_path = Path(self.tempdir.name) / "emulator.json"
        knowledge = KnowledgeBase.load(kb_path)
        key = "10:00"
        knowledge._annotations[key] = {"stack_delta": 2}

        descriptor = SegmentDescriptor(index=0, start=0, end=4)
        raw = (0x10 << 24) | (0x00 << 16) | 0x1234
        segment = Segment(descriptor, raw.to_bytes(4, "big"), "code")

        emulator = Emulator(knowledge)
        report = emulator.simulate_segment(segment)

        self.assertEqual(report.unknown_delta_instructions, 0)
        self.assertTrue(report.traces)
        self.assertNotIn("unknown-delta", report.traces[0].warnings)
        self.assertAlmostEqual(report.traces[0].stack_after, 2.0)

    def test_cfg_ir_ast_and_emulator_pipeline(self) -> None:
        segment = next(self.container.iter_code_segments())
        cfg = ControlFlowGraphBuilder(self.knowledge).build(segment, max_instructions=6)
        self.assertTrue(cfg.blocks)
        cfg_text = cfg.to_text()
        self.assertIn("block", cfg_text)

        ir_program = IRBuilder(self.knowledge).from_cfg(segment, cfg)
        self.assertTrue(ir_program.blocks)
        ir_text = ir_program.render_text()
        self.assertIn("IR", ir_text)

        reconstructor = LuaReconstructor()
        lua_function = reconstructor.from_ir(segment.index, ir_program)
        lua_text = reconstructor.render(lua_function)
        self.assertIn("function", lua_text)
        self.assertIn("local stack = {}", lua_text)

        emulator = Emulator(self.knowledge)
        report = emulator.simulate_segment(segment, max_instructions=4)
        self.assertTrue(report.traces)
        self.assertIn("stack", report.to_text())

    def test_ir_and_emulator_outputs_can_be_written_to_files(self) -> None:
        segment = next(self.container.iter_code_segments())
        cfg = ControlFlowGraphBuilder(self.knowledge).build(segment, max_instructions=4)
        program = IRBuilder(self.knowledge).from_cfg(segment, cfg)

        ir_path = Path(self.tempdir.name) / "ir.txt"
        write_ir_programs([program], ir_path)
        ir_contents = ir_path.read_text("utf-8")
        self.assertIn("segment", ir_contents)
        self.assertIn("IR", ir_contents)

        emulator = Emulator(self.knowledge)
        report = emulator.simulate_segment(segment, max_instructions=4)

        emu_path = Path(self.tempdir.name) / "emu.txt"
        write_emulation_reports([report], emu_path)
        emu_contents = emu_path.read_text("utf-8")
        self.assertIn("stack emulation", emu_contents)
        self.assertIn("stack", emu_contents)

    def test_emulator_reports_unknown_delta_ratio(self) -> None:
        kb_path = Path(self.tempdir.name) / "emu.json"
        knowledge = KnowledgeBase.load(kb_path)

        descriptor = SegmentDescriptor(index=0, start=0, end=8)
        segment = Segment(descriptor, b"\x00" * 8, "code")

        emulator = Emulator(knowledge, unknown_ratio_threshold=0.5)
        report = emulator.simulate_segment(segment)

        self.assertEqual(report.total_instructions, 2)
        self.assertEqual(report.unknown_delta_instructions, 2)
        self.assertAlmostEqual(report.unknown_delta_ratio, 1.0)
        self.assertFalse(report.is_reliable)

        summary = report.to_text()
        self.assertIn("unknown-delta=2/2 (100.0%)", summary)
        self.assertIn("reliability=low", summary)

    def test_cli_main_updates_knowledge_without_duplicate_details(self) -> None:
        kb_path = Path(self.tempdir.name) / "cli_kb.json"
        disasm_path = Path(self.tempdir.name) / "cli_listing.txt"
        analysis_path = Path(self.tempdir.name) / "cli_analysis.json"

        argv = [
            "mbc_disasm.py",
            str(FIXTURE_ADB),
            str(FIXTURE_MBC),
            "--update-knowledge",
            "--knowledge-base",
            str(kb_path),
            "--disasm-out",
            str(disasm_path),
            "--analysis-out",
            str(analysis_path),
        ]

        stdout = StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch.object(sys, "argv", argv):
            cli_main()

        output = stdout.getvalue()
        self.assertIn("knowledge base updated:", output)
        self.assertTrue(kb_path.exists())
        self.assertTrue(disasm_path.exists())

        lines = output.splitlines()
        summary_lines = [line for line in lines if line.startswith("knowledge updated:")]
        self.assertTrue(summary_lines, "expected knowledge update summary in CLI output")

        summary_payload = summary_lines[-1].split("knowledge updated: ", 1)[1]
        summary_keys = [
            entry.split(":", 1)[0]
            for entry in summary_payload.split(", ")
            if entry and not entry.startswith("+")
        ]

        detail_keys: list[str] = []
        for index, line in enumerate(lines):
            if line.strip() == "learned stack behaviour:":
                for detail_line in lines[index + 1 :]:
                    if not detail_line.startswith("  "):
                        break
                    stripped = detail_line.strip()
                    if stripped.startswith("..."):
                        continue
                    detail_keys.append(stripped.split()[0])
                break

        if detail_keys:
            self.assertTrue(
                all(key not in summary_keys for key in detail_keys),
                "detailed output should not repeat previewed updates",
            )


if __name__ == "__main__":
    unittest.main()
