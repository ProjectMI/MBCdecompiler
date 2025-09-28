import json
from collections import Counter

import pytest

from mbcdisasm.knowledge import KnowledgeBase, OpcodeProfile


def test_load_migrates_legacy_schema(tmp_path):
    legacy_data = {
        "profiles": {
            "01:00": {
                "count": 2,
                "stack_deltas": {"0": 2},
                "operand_types": {"large": 1},
                "preceding": {},
                "following": {},
            }
        }
    }
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(legacy_data), "utf-8")

    knowledge = KnowledgeBase.load(path)

    profile = knowledge.get_profile("01:00")
    assert profile.count == 2
    assert profile.stack_deltas[0] == 2
    assert profile.operand_types["large"] == 1
    assert knowledge._annotations == {}
    assert knowledge._data["schema"] == KnowledgeBase.SCHEMA_VERSION


def test_load_rejects_newer_schema(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(
        json.dumps({"schema": KnowledgeBase.SCHEMA_VERSION + 1}),
        "utf-8",
    )

    with pytest.raises(ValueError, match="newer than supported"):
        KnowledgeBase.load(path)


def test_manual_annotations_are_loaded(tmp_path):
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps({"01:02": {"name": "custom", "summary": "manual override"}}),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)
    metadata = knowledge.instruction_metadata("01:02")

    assert metadata.mnemonic == "custom"
    assert metadata.summary == "manual override"

    manual = knowledge.manual_annotation("01:02")
    assert manual["name"] == "custom"


def _profile_with_stack_delta(key: str, delta: float, samples: int) -> OpcodeProfile:
    profile = OpcodeProfile(key)
    profile.count = samples
    profile.stack_deltas = Counter({delta: samples})
    return profile


def test_render_merge_report_accepts_updates(tmp_path):
    path = tmp_path / "kb.json"
    knowledge = KnowledgeBase.load(path)

    report = knowledge.merge_profiles(
        [_profile_with_stack_delta("01:03", 2, 6)],
        min_samples=3,
        confidence_threshold=0.5,
    )

    summary = knowledge.render_merge_report(
        report,
        min_samples=3,
        confidence_threshold=0.5,
    )

    assert "01:03" in summary
    assert "принять корректировку" in summary


def test_render_merge_report_flags_conflicts(tmp_path):
    path = tmp_path / "kb.json"
    knowledge = KnowledgeBase.load(path)

    initial = _profile_with_stack_delta("02:00", 1, 6)
    knowledge.merge_profiles(
        [initial],
        min_samples=3,
        confidence_threshold=0.5,
    )

    conflict_profile = _profile_with_stack_delta("02:00", 3, 6)
    report = knowledge.merge_profiles(
        [conflict_profile],
        min_samples=3,
        confidence_threshold=0.5,
    )

    summary = knowledge.render_merge_report(
        report,
        min_samples=3,
        confidence_threshold=0.5,
    )

    assert "02:00" in summary
    assert "инициировать ручной просмотр трассировки" in summary


def test_merge_profiles_infers_manual_annotations(tmp_path):
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "AA:00": {
                    "name": "manual_literal",
                    "summary": "Push literal value",
                    "stack_delta": 1,
                    "operand_hint": "small",
                    "control_flow": "fallthrough",
                }
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)

    profile = OpcodeProfile("BB:00")
    profile.count = 8
    profile.stack_deltas = Counter({1: 8})
    profile.operand_types = Counter({"small": 8})
    profile.preceding = Counter({"00:00": 4})
    profile.following = Counter({"00:01": 4})

    report = knowledge.merge_profiles([profile], min_samples=3, confidence_threshold=0.5)
    semantic_report = knowledge.apply_semantic_annotations(min_samples=3)

    metadata = knowledge.instruction_metadata("BB:00")
    assert metadata.mnemonic == "manual_literal"
    assert metadata.summary == "Push literal value"
    assert metadata.stack_delta == 1
    assert metadata.operand_hint == "small"

    update_fields = {
        update.field for update in semantic_report.updates if update.key == "BB:00"
    }
    assert "name" in update_fields
    assert "manual_source" in update_fields


def test_manual_inference_handles_unknown_stack(tmp_path):
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "AA:00": {
                    "name": "manual_literal",
                    "summary": "Push literal value",
                    "stack_delta": 1,
                    "operand_hint": "small",
                    "control_flow": "fallthrough",
                }
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)

    profile = OpcodeProfile("BB:00")
    profile.count = 8
    profile.stack_deltas = Counter({"unknown": 8})
    profile.operand_types = Counter({"small": 8})

    report = knowledge.merge_profiles([profile], min_samples=3, confidence_threshold=0.5)
    semantic_report = knowledge.apply_semantic_annotations(min_samples=3)

    metadata = knowledge.instruction_metadata("BB:00")
    assert metadata.mnemonic == "manual_literal"
    assert metadata.summary == "Push literal value"
    assert metadata.stack_delta == 1
    assert metadata.operand_hint == "small"

    update_fields = {
        update.field for update in semantic_report.updates if update.key == "BB:00"
    }
    assert "manual_source" in update_fields


def test_missing_operand_hint_not_queued_for_review(tmp_path):
    kb_path = tmp_path / "kb.json"
    manual_path = tmp_path / "manual_annotations.json"
    manual_path.write_text(
        json.dumps(
            {
                "CC:00": {
                    "name": "manual_cc",
                    "summary": "Manual without operand hint",
                    "stack_delta": 1,
                }
            }
        ),
        "utf-8",
    )

    knowledge = KnowledgeBase.load(kb_path)

    profile = OpcodeProfile("DD:00")
    profile.count = 6
    profile.stack_deltas = Counter({1: 6})
    profile.preceding = Counter({"00:00": 2})
    profile.following = Counter({"00:01": 2})

    report = knowledge.merge_profiles([profile], min_samples=3, confidence_threshold=0.5)

    for task in report.review_tasks:
        assert "operand_hint" not in task.missing_annotations
    annotations = knowledge._annotations.get("DD:00", {})
    assert "operand_hint" not in annotations
