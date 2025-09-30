"""Utilities for generating and applying manual annotation review packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence

from .analysis import Analyzer
from .auto_analyze import find_pairs
from .knowledge import KnowledgeBase, ReviewTask, StackObservation
from .mbc import MbcContainer
from .segment_classifier import SegmentClassifier

DEFAULT_DECISION_FIELDS: Sequence[str] = (
    "name",
    "summary",
    "control_flow",
    "flow_target",
    "stack_delta",
    "stack_confidence",
    "stack_samples",
    "operand_hint",
    "operand_confidence",
    "notes",
)


def _build_decision_template(annotation: Mapping[str, object]) -> Dict[str, object]:
    template: Dict[str, object] = {}
    for field in DEFAULT_DECISION_FIELDS:
        if field in annotation:
            template[field] = annotation[field]
        else:
            template[field] = None
    return template


def _serialize_review_task(task: ReviewTask) -> Dict[str, object]:
    payload = task.to_json()
    payload["key"] = task.key
    payload["reason"] = task.reason
    payload["samples"] = task.samples
    return payload


def generate_review_package(
    knowledge: KnowledgeBase,
    inputs: Sequence[Path],
    *,
    statuses: Sequence[str] = ("partial", "conflict"),
    min_samples: int = 6,
    confidence_threshold: float = 0.75,
    persist_profile: bool = False,
) -> Dict[str, object]:
    """Assemble a JSON-compatible package of review candidates."""

    status_filter = {status.lower() for status in statuses}

    analyzer = Analyzer(knowledge)
    classifier = SegmentClassifier(knowledge)

    assessments: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    runs: List[Dict[str, object]] = []
    total_instructions = 0

    for adb_path, mbc_path in find_pairs(inputs):
        container = MbcContainer.load(mbc_path, adb_path, classifier=classifier)
        analysis = analyzer.analyze(container)
        total_instructions += analysis.total_instructions

        merge_report = knowledge.merge_profiles(
            analysis.iter_profiles(),
            min_samples=min_samples,
            confidence_threshold=confidence_threshold,
        )
        knowledge.apply_semantic_annotations(
            min_samples=min_samples,
            score_threshold=confidence_threshold,
        )

        status_counts = Counter()
        for assessment in merge_report.assessments:
            status_counts[assessment.status] += 1
            if assessment.status.lower() in status_filter:
                assessments[assessment.key].append(
                    {
                        "status": assessment.status,
                        "existing_count": assessment.existing_count,
                        "new_count": assessment.new_count,
                        "notes": assessment.notes,
                        "source": mbc_path.name,
                    }
                )

        runs.append(
            {
                "adb": str(adb_path),
                "mbc": str(mbc_path),
                "instructions": analysis.total_instructions,
                "assessment_counts": dict(status_counts),
            }
        )

    pending_tasks = knowledge.pending_review_tasks()
    task_map: DefaultDict[str, List[ReviewTask]] = defaultdict(list)
    for task in pending_tasks:
        task_map[task.key].append(task)

    entry_keys = sorted(set(assessments) | set(task_map))

    entries: List[Dict[str, object]] = []
    for key in entry_keys:
        annotation = knowledge._annotations.get(key, {})  # type: ignore[attr-defined]
        profile = knowledge._profiles.get(key)  # type: ignore[attr-defined]
        observation = (
            StackObservation.from_profile(profile).to_json()
            if profile is not None
            else None
        )
        entry = {
            "key": key,
            "current_annotation": dict(sorted(annotation.items())),
            "stack_observation": observation,
            "assessments": assessments.get(key, []),
            "review_tasks": [
                _serialize_review_task(task) for task in task_map.get(key, [])
            ],
            "decision": _build_decision_template(annotation),
            "operator_notes": "",
        }
        entries.append(entry)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "knowledge_base": str(knowledge.path),
        "min_samples": min_samples,
        "confidence_threshold": confidence_threshold,
        "statuses": sorted(status_filter),
        "reference_runs": len(runs),
        "total_instructions": total_instructions,
        "pending_review_tasks": len(pending_tasks),
        "entries": len(entries),
    }

    if persist_profile:
        classifier.save_profile()

    return {"metadata": metadata, "runs": runs, "entries": entries}


def apply_review_package(
    knowledge: KnowledgeBase,
    package: Mapping[str, object],
    *,
    manual_path: Optional[Path] = None,
    decision_fields: Sequence[str] = DEFAULT_DECISION_FIELDS,
    resolve_tasks: bool = True,
) -> Dict[str, object]:
    """Apply operator decisions from a review package to the knowledge base."""

    manual_path = manual_path or knowledge.path.with_name("manual_annotations.json")
    if manual_path.exists():
        manual_data = json.loads(manual_path.read_text("utf-8"))
        if not isinstance(manual_data, dict):
            raise ValueError("manual annotations file must contain a JSON object")
    else:
        manual_data = {}

    overrides = manual_data.setdefault("_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("manual annotations '_overrides' must be a JSON object")

    entries = package.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("package 'entries' must be a list")

    updated_keys: List[str] = []
    resolved = 0

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        key = str(entry.get("key", ""))
        if not key:
            continue
        decision = entry.get("decision", {})
        if not isinstance(decision, Mapping):
            continue

        override_entry = overrides.setdefault(key, {})
        if not isinstance(override_entry, dict):
            raise ValueError(f"manual annotations override for {key} must be a JSON object")

        changes: Dict[str, object] = {}
        for field in decision_fields:
            if field not in decision:
                continue
            value = decision[field]
            if value is None:
                continue
            override_entry[field] = value
            changes[field] = value

        operator_notes = entry.get("operator_notes")
        if isinstance(operator_notes, str) and operator_notes:
            override_entry["notes"] = operator_notes
            changes["notes"] = operator_notes

        if not changes:
            continue

        knowledge.record_annotation(key, **override_entry)
        updated_keys.append(key)
        if resolve_tasks:
            resolved += knowledge.resolve_review_task(key)

    manual_path.parent.mkdir(parents=True, exist_ok=True)
    manual_path.write_text(json.dumps(manual_data, indent=2, sort_keys=True), "utf-8")

    knowledge.save()

    return {
        "updated_keys": sorted(set(updated_keys)),
        "manual_path": str(manual_path),
        "resolved_tasks": int(resolved),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="generate review package JSON")
    export.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Paths to directories or files containing .mbc/.adb pairs",
    )
    export.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
        help="Knowledge base database to update",
    )
    export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the review package JSON",
    )
    export.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum stack samples required for automatic updates",
    )
    export.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for stack/operand deductions",
    )
    export.add_argument(
        "--statuses",
        nargs="+",
        default=["partial", "conflict"],
        help="Profile assessment statuses to include in the package",
    )
    export.add_argument(
        "--update-knowledge",
        action="store_true",
        help="Persist the evolved knowledge base after generation",
    )
    export.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the emitted JSON payload",
    )

    apply = subparsers.add_parser("apply", help="apply decisions from a package")
    apply.add_argument(
        "package",
        type=Path,
        help="Path to the JSON package produced by the export command",
    )
    apply.add_argument(
        "--knowledge-base",
        type=Path,
        default=Path("knowledge/opcode_profiles.json"),
        help="Knowledge base database to update",
    )
    apply.add_argument(
        "--manual-path",
        type=Path,
        default=None,
        help="Override the manual annotations JSON path",
    )
    apply.add_argument(
        "--keep-review-tasks",
        action="store_true",
        help="Do not remove review tasks for updated entries",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "export":
        knowledge = KnowledgeBase.load(args.knowledge_base)
        package = generate_review_package(
            knowledge,
            args.inputs,
            statuses=args.statuses,
            min_samples=args.min_samples,
            confidence_threshold=args.confidence_threshold,
            persist_profile=args.update_knowledge,
        )
        if args.update_knowledge:
            knowledge.save()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dumps = json.dumps(package, indent=2 if args.pretty else None, sort_keys=args.pretty)
        args.output.write_text(dumps, "utf-8")
    else:
        knowledge = KnowledgeBase.load(args.knowledge_base)
        package_data = json.loads(args.package.read_text("utf-8"))
        summary = apply_review_package(
            knowledge,
            package_data,
            manual_path=args.manual_path,
            resolve_tasks=not args.keep_review_tasks,
        )
        print(
            "updated {count} entries (manual annotations at {path}, resolved {resolved} review tasks)".format(
                count=len(summary["updated_keys"]),
                path=summary["manual_path"],
                resolved=summary["resolved_tasks"],
            )
        )


if __name__ == "__main__":
    main()
