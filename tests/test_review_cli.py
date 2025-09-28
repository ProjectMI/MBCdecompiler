import json
import tempfile
import unittest
from pathlib import Path

from mbcdisasm import KnowledgeBase, ReviewTask
from mbcdisasm.review_cli import (
    apply_review_package,
    generate_review_package,
)


class ReviewCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.knowledge_path = Path(self.tempdir.name) / "kb.json"
        self.knowledge = KnowledgeBase.load(self.knowledge_path)

    def test_generate_package_collects_review_tasks(self) -> None:
        profile = self.knowledge.get_profile("AA:10")
        profile.count = 4
        profile.stack_deltas.update({1: 4})
        self.knowledge._data.setdefault("opcode_modes", {})[profile.key] = profile.to_json()

        task = ReviewTask(
            key="AA:10",
            reason="missing_annotations",
            samples=4,
            missing_annotations=["control_flow"],
        )
        self.knowledge._enqueue_review_task(task)
        self.knowledge.save()

        package = generate_review_package(self.knowledge, [])
        self.assertIn("entries", package)
        self.assertEqual(len(package["entries"]), 1)
        entry = package["entries"][0]
        self.assertEqual(entry["key"], "AA:10")
        self.assertEqual(entry["review_tasks"][0]["reason"], "missing_annotations")
        self.assertIn("decision", entry)

    def test_apply_package_updates_manual_annotations(self) -> None:
        profile = self.knowledge.get_profile("BB:20")
        profile.count = 3
        profile.stack_deltas.update({1: 3})
        self.knowledge._data.setdefault("opcode_modes", {})[profile.key] = profile.to_json()

        task = ReviewTask(
            key="BB:20",
            reason="missing_annotations",
            samples=3,
            missing_annotations=["control_flow"],
        )
        self.knowledge._enqueue_review_task(task)
        self.knowledge.save()

        package = {
            "entries": [
                {
                    "key": "BB:20",
                    "decision": {
                        "name": "test_mnemonic",
                        "control_flow": "call",
                        "flow_target": "relative",
                        "stack_delta": 1,
                    },
                    "operator_notes": "verified",
                }
            ]
        }

        manual_path = self.knowledge_path.with_name("manual_annotations.json")
        summary = apply_review_package(
            self.knowledge,
            package,
            manual_path=manual_path,
            resolve_tasks=True,
        )

        self.assertIn("BB:20", summary["updated_keys"])
        manual = json.loads(manual_path.read_text("utf-8"))
        self.assertEqual(manual["BB:20"]["control_flow"], "call")
        self.assertEqual(manual["BB:20"]["flow_target"], "relative")
        self.assertEqual(manual["BB:20"]["stack_delta"], 1)
        self.assertEqual(manual["BB:20"]["notes"], "verified")

        reloaded = KnowledgeBase.load(self.knowledge_path)
        tasks = [task for task in reloaded.pending_review_tasks() if task.key == "BB:20"]
        self.assertFalse(tasks)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
