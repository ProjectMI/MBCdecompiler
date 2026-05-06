from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .parser import MBCModule
from .vm_spec import VMWord, decode_words, is_lower_operand_atom


NATIVE_CONTRACT_VERSION = "vm-native-v4"
NATIVE_POLICY = (
    "Native-call analysis consumes the decoded VM token stream. Native opid is "
    "module-local evidence only and is not used as a semantic key. CALL_NATIVE "
    "is classified by its lower operand-frame category: encoded argc consumes "
    "the argv suffix, the older frame prefix is owned by the call as contextual "
    "operator/effect input, and a call result is a demand-bound value rather "
    "than a persistent stack push."
)

CATEGORY_NAMES: dict[str, str] = {
    "reference_scalar": "reference + scalar operation",
    "reference": "reference operation",
    "record_reference_resource": "record/reference resource operation",
    "record_resource": "record resource operation",
    "scalar": "scalar operation",
    "result_forward": "result forwarding / tuple expansion",
    "nullary": "nullary effect or query",
    "context_reference_effect": "contextual reference effect",
    "context_record_resource_effect": "contextual record/resource effect",
    "context_effect": "contextual effect",
    "callback_binding": "callback/control binding",
    "context_callback_effect": "contextual callback/control effect",
    "prefixed": "prefixed native operation",
    "mixed": "mixed native operation",
    "unresolved_frame": "unresolved operand-frame binding",
}


@dataclass(frozen=True)
class VMNativeOperand:
    role: str
    index: int
    category: str
    shape: str
    terminal_kind: Optional[str] = None
    word_index: Optional[int] = None
    offset: Optional[int] = None
    prefixes_hex: list[str] = field(default_factory=list)
    decoder_rule: Optional[str] = None
    mode: Optional[int] = None
    literal_class: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMNativeCallFact:
    id: str
    module: str
    function: str
    offset: int
    word_index: Optional[int]
    encoded_argc: int
    opid: Optional[int]
    category: str
    category_name: str
    frame_shape: str
    call_prefixes_hex: list[str]
    argv: list[VMNativeOperand] = field(default_factory=list)
    non_argv: list[VMNativeOperand] = field(default_factory=list)
    auxiliary: list[VMNativeOperand] = field(default_factory=list)
    pending_result_argv_count: int = 0
    argc_deficit: int = 0
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["argv"] = [item.to_dict() for item in self.argv]
        payload["non_argv"] = [item.to_dict() for item in self.non_argv]
        payload["auxiliary"] = [item.to_dict() for item in self.auxiliary]
        return payload


@dataclass(frozen=True)
class VMNativeCategoryFact:
    id: str
    category: str
    category_name: str
    call_count: int
    coverage: float
    module_count: int
    modules: list[str]
    argc_histogram: dict[str, int]
    frame_shape_histogram_top: dict[str, int]
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMNativeReport:
    contract: str
    summary: dict[str, Any]
    categories: list[VMNativeCategoryFact]
    calls: list[VMNativeCallFact] = field(default_factory=list)

    def to_dict(self, *, include_calls: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": self.contract,
            "summary": dict(self.summary),
            "categories": [item.to_dict() for item in self.categories],
        }
        if include_calls:
            payload["calls"] = [call.to_dict() for call in self.calls]
        return payload


# ---------------------------------------------------------------------------
# Stable helpers


def _stable_sorted(values: Iterable[str]) -> list[str]:
    return sorted((str(v) for v in values), key=lambda value: (len(value), value))


def _safe_id_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _short_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _counter_dict(counter: Counter[Any], *, limit: Optional[int] = None) -> dict[str, int]:
    items = counter.most_common(limit) if limit is not None else sorted(counter.items(), key=lambda kv: str(kv[0]))
    return {str(key): int(value) for key, value in items}


def _literal_class(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if isinstance(value, float):
            return "float_zero" if value == 0.0 else "float"
        ivalue = int(value)
    except Exception:
        return "literal"
    if ivalue == 0:
        return "zero"
    if ivalue == 1:
        return "one"
    if ivalue == -1:
        return "minus_one"
    if -255 <= ivalue <= 255:
        return "small_int"
    return "int"


def _prefixes_hex(word: VMWord) -> list[str]:
    return [f"0x{int(prefix):02X}" for prefix in word.prefixes]


def _prefix_class(prefixes_hex: Iterable[str]) -> str:
    prefixes = tuple(str(prefix) for prefix in prefixes_hex)
    if not prefixes:
        return "-"
    classes: list[str] = []
    if "0x30" in prefixes:
        classes.append("frame")
    if "0x3D" in prefixes:
        classes.append("ctx3d")
    if "0x26" in prefixes:
        classes.append("ctx26")
    if "0x72" in prefixes:
        classes.append("ctx72")
    if not classes:
        classes.extend(prefixes)
    return "+".join(classes)


def _word_literal_value(word: VMWord) -> Any:
    for key in ("value", "u16", "imm", "ref", "opid", "op"):
        if key in word.operands:
            return word.operands.get(key)
    return None


def _operand_category(word: VMWord) -> str:
    terminal_kind = str(word.terminal_kind)
    if terminal_kind in {"REF", "REF16"}:
        return "ref"
    if terminal_kind in {"REC41", "REC61", "REC62"}:
        return "record"
    if terminal_kind == "CODE_REF":
        return "code_ref"
    if terminal_kind == "F32":
        return "number"
    if terminal_kind in {"IMM8", "IMM16", "IMM24S", "IMM24U", "IMM24Z", "IMM32", "U16"}:
        return "zero" if _literal_class(_word_literal_value(word)) == "zero" else "number"
    return "other"


def _operand_shape(word: VMWord, category: str) -> str:
    prefix_class = _prefix_class(_prefixes_hex(word))
    terminal_kind = str(word.terminal_kind)
    if category == "ref":
        return f"{prefix_class}:ref:{terminal_kind}:mode={word.operands.get('mode')}"
    if category == "record":
        return f"{prefix_class}:record:{terminal_kind}:mode={word.operands.get('mode')}"
    if category == "number":
        return f"{prefix_class}:number:{terminal_kind}:{_literal_class(_word_literal_value(word)) or 'number'}"
    if category == "zero":
        return f"{prefix_class}:zero:{terminal_kind}"
    if category == "code_ref":
        return f"{prefix_class}:code_ref:{terminal_kind}"
    return f"{prefix_class}:{category}:{terminal_kind}"


def _operand_from_word(word: VMWord, *, role: str, index: int, auxiliary: Optional[list[VMNativeOperand]] = None) -> VMNativeOperand:
    category = _operand_category(word)
    prefixes_hex = _prefixes_hex(word)
    evidence = {
        key: word.operands.get(key)
        for key in ("value", "ref", "op", "mode", "u16", "width", "signed")
        if word.operands.get(key) is not None
    }
    if auxiliary:
        evidence["auxiliary"] = [item.to_dict() for item in auxiliary]
    return VMNativeOperand(
        role=role,
        index=index,
        category=category,
        shape=_operand_shape(word, category),
        terminal_kind=word.terminal_kind,
        word_index=int(word.index) if int(word.index) >= 0 else None,
        offset=int(word.offset),
        prefixes_hex=prefixes_hex,
        decoder_rule=word.decoder_rule,
        mode=int(word.operands["mode"]) if word.operands.get("mode") is not None else None,
        literal_class=_literal_class(_word_literal_value(word)),
        evidence=evidence,
    )


def _auxiliary_operand(word: VMWord, *, index: int) -> VMNativeOperand:
    value = _word_literal_value(word)
    prefixes_hex = _prefixes_hex(word)
    return VMNativeOperand(
        role="auxiliary",
        index=index,
        category="auxiliary_literal",
        shape=f"{_prefix_class(prefixes_hex)}:auxiliary_literal:{word.terminal_kind}",
        terminal_kind=word.terminal_kind,
        word_index=int(word.index) if int(word.index) >= 0 else None,
        offset=int(word.offset),
        prefixes_hex=prefixes_hex,
        decoder_rule=word.decoder_rule,
        literal_class=_literal_class(value),
        evidence={"value": value} if value is not None else {},
    )


def _pending_result_operand(*, index: int, slot_count: int, producer_kind: Optional[str]) -> VMNativeOperand:
    return VMNativeOperand(
        role="argv_pending_result",
        index=index,
        category="call_result",
        shape=f"pending_call_result:slots={int(slot_count)}",
        terminal_kind=producer_kind,
        word_index=None,
        offset=None,
        evidence={
            "slot_count": int(slot_count),
            "producer_terminal_kind": producer_kind,
            "binding_rule": "demand_bound_call_result_supplies_missing_argv_slots",
        },
    )


def _category_for_call(
    *,
    encoded_argc: int,
    argv_categories: list[str],
    non_argv_categories: list[str],
    pending_result_argv_count: int,
    argc_deficit: int,
    call_prefixes_hex: list[str],
) -> str:
    categories = [*argv_categories, *non_argv_categories]
    category_set = set(categories)
    real_argv_categories = [item for item in argv_categories if item != "call_result"]
    if argc_deficit > 0:
        return "unresolved_frame"
    if pending_result_argv_count and not real_argv_categories:
        return "result_forward"
    if encoded_argc == 0 and not argv_categories and not non_argv_categories:
        return "nullary"
    if not argv_categories and non_argv_categories:
        if "code_ref" in category_set:
            return "context_callback_effect"
        if "record" in category_set:
            return "context_record_resource_effect"
        if "ref" in category_set:
            return "context_reference_effect"
        return "context_effect"
    if "code_ref" in category_set:
        return "callback_binding"
    if "record" in category_set and "ref" in category_set:
        return "record_reference_resource"
    if "record" in category_set:
        return "record_resource"
    if "ref" in category_set and any(item in {"number", "zero", "call_result"} for item in category_set):
        return "reference_scalar"
    if "ref" in category_set:
        return "reference"
    if argv_categories and all(item in {"number", "zero", "call_result"} for item in argv_categories):
        return "scalar"
    if call_prefixes_hex:
        return "prefixed"
    return "mixed"


def _category_confidence(category: str, *, argc_deficit: int) -> float:
    if argc_deficit > 0 or category == "unresolved_frame":
        return 0.25
    if category in {"mixed", "prefixed"}:
        return 0.65
    return 1.0


def _frame_shape(
    *,
    encoded_argc: int,
    argv: list[VMNativeOperand],
    non_argv: list[VMNativeOperand],
    auxiliary: list[VMNativeOperand],
    call_prefixes_hex: list[str],
) -> str:
    return (
        f"argc={encoded_argc};call_prefix={_prefix_class(call_prefixes_hex)};"
        f"argv=[{','.join(item.shape for item in argv)}];"
        f"non_argv=[{','.join(item.shape for item in non_argv)}];"
        f"aux=[{','.join(item.shape for item in auxiliary)}]"
    )


def _flatten_clauses(clauses: list[list[VMNativeOperand]]) -> list[VMNativeOperand]:
    out: list[VMNativeOperand] = []
    for clause in clauses:
        out.extend(clause)
    return out


def _reindexed_operands(items: list[VMNativeOperand], *, role: str, start: int = 0) -> list[VMNativeOperand]:
    out: list[VMNativeOperand] = []
    for idx, item in enumerate(items, start=start):
        out.append(VMNativeOperand(
            role=role,
            index=idx,
            category=item.category,
            shape=item.shape,
            terminal_kind=item.terminal_kind,
            word_index=item.word_index,
            offset=item.offset,
            prefixes_hex=list(item.prefixes_hex),
            decoder_rule=item.decoder_rule,
            mode=item.mode,
            literal_class=item.literal_class,
            evidence=dict(item.evidence),
        ))
    return out


class _TokenFrame:
    def __init__(self) -> None:
        self.clauses: list[list[VMNativeOperand]] = []
        self.pending_aux: list[VMNativeOperand] = []
        self.pending_call_results: list[dict[str, Any]] = []
        self.max_frame_depth = 0
        self.max_clause_count = 0
        self.result_binding_histogram: Counter[str] = Counter()

    def start_clause(self) -> None:
        self.clauses.append([])

    def ensure_clause(self) -> list[VMNativeOperand]:
        if not self.clauses:
            self.start_clause()
        return self.clauses[-1]

    def clear_frame(self) -> None:
        self.clauses.clear()
        self.pending_aux.clear()

    def update_max(self) -> None:
        depth = sum(len(clause) for clause in self.clauses)
        clauses = sum(1 for clause in self.clauses if clause)
        self.max_frame_depth = max(self.max_frame_depth, depth)
        self.max_clause_count = max(self.max_clause_count, clauses)

    def bind_pending_results(self, binding: str) -> None:
        if self.pending_call_results:
            self.result_binding_histogram[binding] += len(self.pending_call_results)
        self.pending_call_results.clear()

    def consume_pending_result_deficit(self, count: int) -> tuple[int, Optional[str]]:
        if count <= 0 or not self.pending_call_results:
            return 0, None
        result = self.pending_call_results.pop()
        self.result_binding_histogram["argv_deficit_result"] += 1
        return count, str(result.get("terminal_kind")) if result.get("terminal_kind") is not None else None

    def add_operand_atom(self, word: VMWord) -> None:
        if 0x30 in [int(prefix) for prefix in word.prefixes]:
            self.start_clause()
        operand = _operand_from_word(word, role="frame", index=sum(len(clause) for clause in self.clauses), auxiliary=self.pending_aux)
        self.pending_aux = []
        self.ensure_clause().append(operand)
        self.update_max()

    def add_auxiliary(self, word: VMWord) -> None:
        self.pending_aux.append(_auxiliary_operand(word, index=len(self.pending_aux)))

    def consume_call_frame(self, word: VMWord, *, role: str) -> tuple[list[VMNativeOperand], list[VMNativeOperand], list[VMNativeOperand], int, int, Optional[str], dict[str, Any]]:
        encoded_argc = int(word.operands.get("argc", 0) or 0)
        frame_before = _flatten_clauses(self.clauses)
        available_argc = min(encoded_argc, len(frame_before))
        real_argv = frame_before[len(frame_before) - available_argc:] if available_argc else []
        non_argv = frame_before[:len(frame_before) - available_argc] if available_argc else list(frame_before)
        raw_deficit = max(0, encoded_argc - len(real_argv))
        pending_result_argv_count, pending_result_kind = self.consume_pending_result_deficit(raw_deficit)
        argc_deficit = max(0, raw_deficit - pending_result_argv_count)
        pending_argv: list[VMNativeOperand] = []
        if pending_result_argv_count:
            pending_argv.append(_pending_result_operand(index=0, slot_count=pending_result_argv_count, producer_kind=pending_result_kind))
        argv = [*pending_argv, *_reindexed_operands(real_argv, role="argv", start=len(pending_argv))]
        non_argv = _reindexed_operands(non_argv, role="non_argv")
        auxiliary = _reindexed_operands(self.pending_aux, role="auxiliary")
        evidence = {
            "frame_atom_count_before_call": len(frame_before),
            "frame_atom_pop": len(real_argv),
            "frame_non_arg_pop": len(non_argv),
            "frame_pending_call_result_argc_pop": pending_result_argv_count,
            "frame_raw_argc_deficit": raw_deficit,
            "frame_argc_deficit": argc_deficit,
            "call_argument_binding_rule": "encoded_argc_consumes_lower_operand_frame_suffix",
            "call_non_arg_binding_rule": "remaining_lower_frame_prefix_is_call_context",
        }
        self.clear_frame()
        self.pending_call_results.append({
            "terminal_kind": word.terminal_kind,
            "role": role,
            "offset": int(word.offset),
            "word_index": int(word.index) if int(word.index) >= 0 else None,
        })
        return auxiliary, non_argv, argv, pending_result_argv_count, argc_deficit, pending_result_kind, evidence


# ---------------------------------------------------------------------------
# Public classification API


def analyze_function_native_calls(
    *,
    module_name: str,
    function_name: str,
    words: list[VMWord],
) -> list[VMNativeCallFact]:
    """Classify top-level CALL_NATIVE tokens in one function body.

    The pass is intentionally token-stream based. It uses the lower operand-frame
    transfer rule, but does not build CFG, does not inspect raw bytes, and does
    not infer source-level native names.
    """

    frame = _TokenFrame()
    calls: list[VMNativeCallFact] = []
    for ordinal, word in enumerate(words):
        if word.terminal_kind == "BARE_U32":
            frame.add_auxiliary(word)
            continue
        if is_lower_operand_atom(word):
            frame.add_operand_atom(word)
            continue
        if word.terminal_kind in {"CALL_NATIVE", "CALL_SCRIPT"}:
            auxiliary, non_argv, argv, pending_result_argv_count, argc_deficit, pending_result_kind, frame_evidence = frame.consume_call_frame(word, role=word.terminal_kind)
            if word.terminal_kind != "CALL_NATIVE":
                continue
            encoded_argc = int(word.operands.get("argc", 0) or 0)
            opid_raw = word.operands.get("opid")
            try:
                opid = int(opid_raw) if opid_raw is not None else None
            except Exception:
                opid = None
            call_prefixes_hex = _prefixes_hex(word)
            argv_categories = [operand.category for operand in argv]
            non_argv_categories = [operand.category for operand in non_argv]
            category = _category_for_call(
                encoded_argc=encoded_argc,
                argv_categories=argv_categories,
                non_argv_categories=non_argv_categories,
                pending_result_argv_count=pending_result_argv_count,
                argc_deficit=argc_deficit,
                call_prefixes_hex=call_prefixes_hex,
            )
            shape = _frame_shape(
                encoded_argc=encoded_argc,
                argv=argv,
                non_argv=non_argv,
                auxiliary=auxiliary,
                call_prefixes_hex=call_prefixes_hex,
            )
            call_id_payload = [module_name, function_name, int(word.index), int(word.offset), encoded_argc, call_prefixes_hex, shape]
            evidence = {
                "classifier": "native_token_frame_v4",
                "opid_is_module_local_evidence": True,
                "pending_result_source_kind": pending_result_kind,
                **frame_evidence,
            }
            calls.append(VMNativeCallFact(
                id=f"native_call_{_short_hash(call_id_payload)}",
                module=module_name,
                function=function_name,
                offset=int(word.offset),
                word_index=int(word.index) if int(word.index) >= 0 else None,
                encoded_argc=encoded_argc,
                opid=opid,
                category=category,
                category_name=CATEGORY_NAMES.get(category, category),
                frame_shape=shape,
                call_prefixes_hex=call_prefixes_hex,
                argv=argv,
                non_argv=non_argv,
                auxiliary=auxiliary,
                pending_result_argv_count=pending_result_argv_count,
                argc_deficit=argc_deficit,
                confidence=_category_confidence(category, argc_deficit=argc_deficit),
                evidence=evidence,
            ))
            continue
        if word.terminal_kind == "BR":
            frame.bind_pending_results("next_control_or_predicate_consumer")
            frame.clear_frame()
            continue
        if word.terminal_kind in {"RETURN_PAIR", "END"}:
            frame.bind_pending_results("next_return_consumer")
            frame.clear_frame()
            continue
        if word.terminal_kind not in {"NOP", "MARK"}:
            frame.bind_pending_results("next_structural_consumer")

    frame.bind_pending_results("open_at_function_exit")
    calls.sort(key=lambda call: (-1 if call.word_index is None else int(call.word_index), int(call.offset), call.id))
    return calls


def native_call_fact_for_word(
    *,
    module_name: str,
    function_name: str,
    words: list[VMWord],
) -> dict[int, VMNativeCallFact]:
    return {int(call.word_index): call for call in analyze_function_native_calls(module_name=module_name, function_name=function_name, words=words) if call.word_index is not None}


def analyze_module_native_calls(
    module: MBCModule | str | Path,
    *,
    function: Optional[str] = None,
    limit_functions: Optional[int] = None,
) -> list[VMNativeCallFact]:
    from .ir import select_function_body_vmir

    mod = module if isinstance(module, MBCModule) else MBCModule(module)
    module_name = Path(mod.path).name
    entries = [mod.get_function_entry(function)] if function else mod.function_entries(include_definitions=True, include_exports=True, dedupe=True)
    if limit_functions is not None:
        entries = entries[: max(0, int(limit_functions))]
    calls: list[VMNativeCallFact] = []
    for entry in entries:
        raw, _selection = select_function_body_vmir(mod, entry)
        words = decode_words(raw)
        calls.extend(analyze_function_native_calls(module_name=module_name, function_name=entry.name, words=words))
    return calls


# ---------------------------------------------------------------------------
# Aggregation / validation


def build_native_report(
    modules: Iterable[str | Path],
    *,
    function: Optional[str] = None,
    limit_functions: Optional[int] = None,
    top_frame_shapes_per_category: int = 12,
) -> VMNativeReport:
    module_paths = list(modules)
    all_calls: list[VMNativeCallFact] = []
    for module_path in module_paths:
        all_calls.extend(analyze_module_native_calls(module_path, function=function, limit_functions=limit_functions))

    total = len(all_calls)
    category_groups: dict[str, list[VMNativeCallFact]] = defaultdict(list)
    for call in all_calls:
        category_groups[call.category].append(call)

    categories: list[VMNativeCategoryFact] = []
    for category, calls in category_groups.items():
        modules_seen = _stable_sorted({call.module for call in calls})
        argc_hist = Counter(str(call.encoded_argc) for call in calls)
        shape_hist = Counter(call.frame_shape for call in calls)
        categories.append(VMNativeCategoryFact(
            id=f"native_category_{_safe_id_part(category)}",
            category=category,
            category_name=CATEGORY_NAMES.get(category, category),
            call_count=len(calls),
            coverage=(len(calls) / total) if total else 0.0,
            module_count=len(modules_seen),
            modules=modules_seen,
            argc_histogram=_counter_dict(argc_hist),
            frame_shape_histogram_top=_counter_dict(shape_hist, limit=top_frame_shapes_per_category),
            confidence=min((call.confidence for call in calls), default=1.0),
            evidence={
                "category_rule": "category is derived from lower operand-frame token categories and demand-bound result use",
                "opid_keyed": False,
                "frame_shape_count": len(shape_hist),
            },
        ))
    categories.sort(key=lambda item: (-item.call_count, item.id))

    module_hist = Counter(call.module for call in all_calls)
    category_hist = Counter(call.category for call in all_calls)
    argc_hist = Counter(str(call.encoded_argc) for call in all_calls)
    frame_shape_hist = Counter(call.frame_shape for call in all_calls)
    unresolved_count = category_hist.get("unresolved_frame", 0)
    classified_count = total - unresolved_count
    summary = {
        "native_contract": NATIVE_CONTRACT_VERSION,
        "module_count": len({Path(module).name for module in module_paths}),
        "native_call_count": total,
        "classified_native_call_count": classified_count,
        "unresolved_frame_call_count": unresolved_count,
        "category_count": len(categories),
        "category_coverage": (classified_count / total) if total else 0.0,
        "frame_shape_count": len(frame_shape_hist),
        "top_16_frame_shape_coverage": (sum(count for _shape, count in frame_shape_hist.most_common(16)) / total) if total else 0.0,
        "encoded_argc_histogram": _counter_dict(argc_hist),
        "native_category_histogram": _counter_dict(category_hist),
        "module_native_call_histogram": _counter_dict(module_hist),
        "policy": NATIVE_POLICY,
    }
    return VMNativeReport(NATIVE_CONTRACT_VERSION, summary, categories, all_calls)


def validate_native_report(report: VMNativeReport) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    total = int(report.summary.get("native_call_count", 0) or 0)
    category_total = sum(int(category.call_count) for category in report.categories)
    if total != category_total:
        errors.append({"kind": "category_count_mismatch", "native_call_count": total, "category_call_count": category_total})
    if total != len(report.calls):
        errors.append({"kind": "call_list_count_mismatch", "native_call_count": total, "call_fact_count": len(report.calls)})
    for call in report.calls:
        if call.category not in CATEGORY_NAMES:
            errors.append({"kind": "unknown_native_category", "call": call.id, "category": call.category})
        if call.opid is not None and f"opid={call.opid}" in call.frame_shape:
            errors.append({"kind": "opid_leaked_into_frame_shape", "call": call.id, "frame_shape": call.frame_shape})
        if call.encoded_argc < 0:
            errors.append({"kind": "negative_argc", "call": call.id, "encoded_argc": call.encoded_argc})
        if call.argc_deficit < 0:
            errors.append({"kind": "negative_argc_deficit", "call": call.id, "argc_deficit": call.argc_deficit})
    return errors


# ---------------------------------------------------------------------------
# CLI


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze CALL_NATIVE categories in MBL .mbc modules")
    parser.add_argument("modules", type=Path, nargs="+", help="one or more .mbc modules")
    parser.add_argument("--json", type=Path, default=None, help="write native analysis JSON")
    parser.add_argument("--function", default=None, help="only analyze one function name in each module")
    parser.add_argument("--limit-functions", type=int, default=None)
    parser.add_argument("--include-calls", action="store_true", help="include per-call native facts in JSON")
    parser.add_argument("--top-frame-shapes-per-category", type=int, default=12)
    parser.add_argument("--strict", action="store_true", help="return non-zero if report invariants fail")
    args = parser.parse_args(argv)

    report = build_native_report(
        args.modules,
        function=args.function,
        limit_functions=args.limit_functions,
        top_frame_shapes_per_category=args.top_frame_shapes_per_category,
    )
    validation_errors = validate_native_report(report)
    payload = report.to_dict(include_calls=args.include_calls)
    payload["validation"] = {"error_count": len(validation_errors), "errors": validation_errors}
    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2 if args.strict and validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(_main())
