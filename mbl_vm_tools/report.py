from __future__ import annotations

from pathlib import Path
import json
from collections import Counter, defaultdict

from .parser import MBCModule, MAGIC_HEADER
from .tokenizer import Token, tokenize_stream, coverage


SUSPICIOUS_PREFIX_DEF_NAMES = {"ITSOFF", "MBS", "ie"}
DATA_LIKE_EXPORT_NAMES = {"pClanName", "pUSTATE", "GetTableTime"}
TRIVIAL_STUB_EXPORT_NAMES = {"empty", "halt", "thalt", "CallLink", "CallEnd", "M"}

SEMANTIC_TOKEN_KINDS = {
    "OP", "AGG", "AGG0", "REF", "REF16", "REC41", "REC61", "REC62",
    "CALL66", "CALL63A", "CALL63B", "IMM", "IMM16", "IMM24S", "IMM24U",
    "IMM32", "BR", "OPU16", "DWBLOB", "SIG_CALL66_REFPAIR_HEAD",
    "SIG_CALL66_SMALLIMM", "SIG_CONST_U32_TRAILER", "SIG_SLOT_CONST",
    "SIG_PADDED_CHECKPUT", "SIG_USEOWNER_HEAD", "SIG_USECLIENT_HEAD",
    "SIG_UNIQUEGEN_HEAD", "SIG_USEOFF_HEAD", "SIG_INPUTDONE_HEAD",
    "SIG_U32_U8_CALL66_TAIL", "SIG_AGG2_PARTIAL_HEAD", "SIG_AGG1_PARTIAL_HEAD",
    "SIG_USECLIENT_ALT_HEAD", "SIG_SETOSST_HEAD", "SIG_GETPLAYERID_HEAD",
    "SIG_PAD17", "SIG_PAD11_BR", "SIG_GETMODIFIERS_PADTAIL",
}


def _vm_export_slices(mod: MBCModule, code_base: int, name: str, next_head_bytes: int = 16) -> tuple[bytes, bytes] | None:
    defs = sorted(mod.definitions, key=lambda r: r.a)
    if not defs:
        return None
    by_name = {r.name: r for r in defs}
    rec = by_name.get(name)
    if rec is None:
        return None

    start_addr = rec.a
    end_addr = rec.b + 1
    start = code_base + start_addr
    end = code_base + end_addr
    if start < 0 or end < 0 or end > len(mod.data) or start >= end:
        return None

    raw = mod.data[start:end]
    stitched = raw
    idx = next((i for i, r in enumerate(defs) if r.name == name), None)
    if idx is not None and idx + 1 < len(defs):
        nxt = defs[idx + 1]
        nstart = code_base + nxt.a
        if 0 <= nstart < len(mod.data):
            stitched = raw + mod.data[nstart:nstart + next_head_bytes]
    return raw, stitched


def _infer_vm_code_base(mod: MBCModule) -> int | None:
    if not mod.definitions or not mod.exports:
        return None
    if not any(r.a < len(MAGIC_HEADER) for r in mod.exports):
        return None

    boundary = min(
        [
            c.start
            for c in (mod.definition_table, mod.globals_table, mod.exports_table)
            if c is not None
        ]
        + [len(mod.data)]
    )

    defs = sorted(mod.definitions, key=lambda r: r.a)
    exported = [r.name for r in mod.exports]
    max_addr = max((rec.b for rec in defs if rec.name in exported), default=None)
    if max_addr is None:
        return None

    start_base = 0x20
    max_base = boundary - (max_addr + 1)
    if max_base <= start_base:
        return None

    best = None
    best_score = -1.0
    for base in range(start_base, max_base + 1):
        total = 0
        covered = 0
        ok = 0
        for name in exported:
            pair = _vm_export_slices(mod, base, name, next_head_bytes=2)
            if pair is None:
                continue
            _, stitched = pair
            toks = tokenize_stream(stitched)
            cov = coverage(toks, len(stitched))
            total += cov["total_bytes"]
            covered += cov["covered_bytes"]
            if cov["coverage_ratio"] >= 0.75:
                ok += 1
        if total == 0:
            continue
        score = (covered / total) + (ok * 0.01)
        if score > best_score:
            best_score = score
            best = base

    if best is None or best_score < 0.85:
        return None
    return best


def classify_layout(mod: MBCModule) -> str:
    if not mod.definitions and not mod.globals and not mod.exports:
        return "empty_or_unparsed"
    if mod.definitions and not mod.globals and not mod.exports:
        return "definitions_only"
    if mod.embedded_import_like_exports:
        names = {r.name for r in mod.embedded_import_like_exports}
        if {"InvOn", "InvOff", "InvTrig"} <= names:
            if {"InitAI", "InvertAI", "SetRespRadius"} <= names:
                return "split_interface_with_toggle_and_ai_splice"
            return "split_interface_with_toggle_splice"
        return "split_interface_with_embedded_import_block"
    return "clean_split_interface"


def classify_interface_signature(mod: MBCModule) -> str | None:
    emb = tuple(x.name for x in mod.embedded_import_like_exports)
    if emb:
        s = set(emb)
        if emb == ("GetModifiers", "GetModifs", "GetModifsArr", "ModifsLoaded", "GetParent", "GetCaster"):
            return "modifier_context_block"
        if emb == ("GetModifiers", "GetParent"):
            return "modifier_context_minimal_block"
        if emb == ("InvOn", "InvOff", "InvTrig"):
            return "toggle_splice"
        if emb == ("InitAI", "InvertAI", "SetRespRadius"):
            return "ai_splice"
        if emb == ("InvOn", "InvOff", "InvTrig", "InitAI", "InvertAI", "SetRespRadius"):
            return "toggle_and_ai_splice"
        if emb == ("TradeTrig", "TradeOn", "TradeOff", "InvOn", "InvOff", "InvTrig"):
            return "trade_toggle_splice"
        if {"TableOn", "TableOff", "TableTrig", "PutMoney", "GetMoney"} <= s:
            return "table_money_splice"
        if {"PlayerName", "Model", "NumToModel", "Paused", "Getxyz", "Getabg", "ShowWebShopWindow", "QueryShowWebShop"} <= s:
            return "player_webshop_context"
        if {"GetSlotsNum", "getPictName", "GetRootParent", "TestIt"} <= s:
            return "inventory_container_context"
        if {"LoadMsgGroup", "UnloadMsgGroup", "SetItemGroup", "gMsg"} <= s and "GetPictsPointer" in s:
            return "trade_ui_context"
        if {"ShowWebShopWindow", "QueryShowWebShop", "HideWebShopWindow", "IsWebShopWindowOn", "InvTrig", "InvOn", "InvOff"} <= s:
            return "webshop_toggle_context"
        if {"PlayerName", "Model", "NumToModel", "Paused", "Getxyz", "Getabg", "GetP"} <= s and "CopyAb" in s:
            return "mission_player_context"
        if {"CreateObj", "CreateObjWait", "DestroyObj", "CenterObj", "SetTrig", "CountAnim", "SpeedObj", "DirectOfObj"} <= s:
            return "character_runtime_context"
        if emb == ("readstr", "AskForFile", "WaitForAsk", "IsReceived", "PercentReceived"):
            return "file_dialog_context"
        if emb == ("WaitForAsk", "LoadMsgGroup", "UnloadMsgGroup", "SetItemGroup", "gMsg"):
            return "message_io_context"
        if emb == ("Params", "pEnemies", "FlyWeapon", "Pet"):
            return "combat_core_context"
        return "embedded_import_block_unknown_signature"

    exports = {x.name for x in mod.exports}
    defs = [x.name for x in mod.definitions]
    if {"CallLink", "CallEnd", "CallPict"} <= exports:
        if "CPar" in exports:
            return "micro_dispatch_with_params"
        return "micro_dispatch"
    if {"LoadMsgGroup", "UnloadMsgGroup", "SetItemGroup", "gMsg"} <= exports:
        return "message_group_service"
    if {"ShowWebShopWindow", "HideWebShopWindow", "IsWebShopWindowOn"} <= exports:
        return "webshop_window_service"
    if {"CheckLinking", "getBLkType", "getBLkID", "getLinkingInfo"} <= exports:
        return "linking_object_service"
    if defs and defs[0] in SUSPICIOUS_PREFIX_DEF_NAMES:
        return "prefix_contaminated_definitions"
    return None


def _coverage_for_kinds(tokens: list[Token], total_size: int, include_kinds: set[str]) -> dict:
    covered = sum(tok.size for tok in tokens if tok.kind in include_kinds)
    return {
        "covered_bytes": covered,
        "total_bytes": total_size,
        "coverage_ratio": (covered / total_size) if total_size else 0.0,
        "token_counts": dict(Counter(tok.kind for tok in tokens if tok.kind in include_kinds)),
    }


def _semantic_coverage(tokens: list[Token], total_size: int) -> dict:
    return _coverage_for_kinds(tokens, total_size, SEMANTIC_TOKEN_KINDS)


def _repair_export_tail(mod: MBCModule, name: str, raw: bytes, max_borrow: int = 8) -> tuple[bytes, list[Token], int] | None:
    names = mod.export_names()
    idx = names.index(name)
    if idx + 1 >= len(names):
        return None

    next_head = mod.get_export_body(names[idx + 1])[:max_borrow]
    if not next_head:
        return None

    raw_tokens = tokenize_stream(raw)
    raw_cov = coverage(raw_tokens, len(raw))["coverage_ratio"]
    best: tuple[float, int, bytes, list[Token]] | None = None
    raw_len = len(raw)

    for borrow in range(1, len(next_head) + 1):
        data = raw + next_head[:borrow]
        toks = tokenize_stream(data)

        if any(tok.offset >= raw_len for tok in toks):
            continue

        crossing = [tok for tok in toks if tok.offset < raw_len < (tok.offset + tok.size) and tok.kind != "UNK"]
        if not crossing:
            continue

        if not all(any(tok.offset <= pos < (tok.offset + tok.size) for tok in crossing) for pos in range(raw_len, len(data))):
            continue

        cov = coverage(toks, len(data))["coverage_ratio"]
        if cov <= raw_cov:
            continue

        if best is None or cov > best[0] or (cov == best[0] and borrow < best[1]):
            best = (cov, borrow, data, toks)

    if best is None:
        return None
    return best[2], best[3], best[1]


def _data_like_export_reason(name: str, best_cov: float) -> bool:
    if name in DATA_LIKE_EXPORT_NAMES and best_cov < 0.65:
        return True
    if name.startswith("p") and len(name) > 1 and name[1].isupper() and best_cov < 0.25:
        return True
    return False


def _best_size(entry: dict) -> int:
    if entry["best_mode"] == "repaired" and entry.get("repaired_size") is not None:
        return entry["repaired_size"]
    return entry["stitched_size"] if entry["best_mode"] == "stitched" else entry["raw_size"]


def _weighted_average(entries: list[dict], key: str) -> float | None:
    if not entries:
        return None
    denom = sum(_best_size(x) for x in entries)
    if not denom:
        return None
    num = sum(_best_size(x) * x[key] for x in entries)
    return num / denom


def _dominant_byte_info(data: bytes) -> tuple[int | None, float]:
    if not data:
        return None, 0.0
    value, count = Counter(data).most_common(1)[0]
    return value, count / len(data)


def _is_padding_like_export(entry: dict) -> bool:
    dominant_ratio = entry.get("dominant_byte_ratio", 0.0)
    dominant_byte = entry.get("dominant_byte")
    if dominant_byte is None:
        return False
    if entry.get("best_coverage_ratio", 0.0) >= 0.95:
        return False
    if entry["raw_size"] <= 64 and dominant_ratio >= 0.80:
        return True
    if dominant_byte in {0x7C, 0x00} and entry["raw_size"] <= 64 and dominant_ratio >= 0.65 and entry["best_coverage_ratio"] < 0.90:
        return True
    return False


def _is_stub_like(entry: dict) -> bool:
    if entry.get("header_overlap"):
        return True
    if entry.get("padding_like"):
        return True
    if entry.get("name") in TRIVIAL_STUB_EXPORT_NAMES:
        return True
    if _best_size(entry) <= 32 and entry.get("best_coverage_ratio", 0.0) < 0.85:
        return True
    return False


def _infer_micro_semantic_kind(tokens: list[Token]) -> str | None:
    kinds = [tok.kind for tok in tokens if tok.kind not in {"PAD", "ASCII"}]
    if not kinds:
        return None
    first = kinds[0]
    if first == "SIG_USEOFF_HEAD":
        return "useoff_wrapper"
    if first == "SIG_USECLIENT_HEAD":
        return "useclient_wrapper"
    if first == "SIG_USEOWNER_HEAD":
        return "useowner_wrapper"
    if first == "SIG_UNIQUEGEN_HEAD":
        return "unique_generation_wrapper_head"
    if first == "SIG_PADDED_CHECKPUT":
        return "padded_checkput_stub"
    if first == "SIG_INPUTDONE_HEAD":
        return "inputdone_wrapper_head"
    if first == "SIG_U32_U8_CALL66_TAIL":
        return "u32_call66_tail_wrapper"
    if first == "SIG_AGG2_PARTIAL_HEAD":
        return "agg2_partial_head"
    if first == "SIG_AGG1_PARTIAL_HEAD":
        return "agg1_partial_head"
    if first == "SIG_USECLIENT_ALT_HEAD":
        return "useclient_alt_wrapper_head"
    if first == "SIG_CALL66_SMALLIMM":
        return "call66_smallimm_wrapper"
    if first == "SIG_CALL66_REFPAIR_HEAD":
        return "call66_refpair_wrapper"
    if first == "SIG_CONST_U32_TRAILER":
        return "u32_const_trailer"
    if first == "SIG_SLOT_CONST":
        return "slot_const_wrapper"
    if first == "SIG_SETOSST_HEAD":
        return "setosst_wrapper"
    if first == "SIG_GETPLAYERID_HEAD":
        return "getplayerid_wrapper"
    if first == "SIG_PAD17":
        return "pad17_stub"
    if first == "SIG_PAD11_BR":
        return "pad11_branch_stub"
    if first == "SIG_GETMODIFIERS_PADTAIL":
        return "getmodifiers_padtail_stub"
    if first == "DWBLOB":
        return "dword_blob_segment"
    if kinds == ["AGG"] or kinds == ["AGG0"]:
        return "aggregate_wrapper"
    if all(kind == "REF" for kind in kinds):
        return "ref_chain"
    return None


def _classify_ir_readiness(entry: dict) -> str:
    if entry.get("padding_like"):
        return "padding_or_noise"
    if entry.get("data_like"):
        return "data_like"
    sem = entry.get("best_semantic_coverage_ratio", 0.0)
    if sem >= 0.95:
        return "ir_ready"
    if sem >= 0.75:
        return "mostly_ir_ready"
    if sem >= 0.35:
        return "partial_semantic"
    return "opaque_or_unresolved"


def module_attention_reasons(result: dict) -> list[str]:
    reasons: list[str] = []
    if result.get("layout_type") == "empty_or_unparsed":
        reasons.append("empty_or_unparsed")
    elif result.get("layout_type") == "definitions_only" and result.get("exports_count", 0) == 0:
        reasons.append("definitions_only")

    if result.get("interface_signature_type") == "prefix_contaminated_definitions":
        reasons.append("prefix_contamination")

    if (result.get("adjusted_avg_semantic_coverage") or 0.0) < 0.80:
        reasons.append("low_adjusted_semantic_coverage")
    if (result.get("adjusted_avg_semantic_gap") or 0.0) >= 0.15:
        reasons.append("large_semantic_gap")

    opaque = [
        x["name"]
        for x in result.get("export_analysis", [])
        if x["ir_readiness"] == "opaque_or_unresolved"
        and not x.get("data_like")
        and not x.get("padding_like")
    ]
    if opaque:
        reasons.append("opaque_exports")

    raw_header_overlap = [] if result.get("export_address_mode") == "vmaddr" else (result.get("header_overlap_exports") or [])
    if raw_header_overlap:
        reasons.append("header_overlap_exports")

    return sorted(set(reasons))


def _export_entry(path: str, layout_type: str, interface_signature_type: str | None, export: dict) -> dict:
    return {
        "path": path,
        "layout_type": layout_type,
        "interface_signature_type": interface_signature_type,
        "name": export["name"],
        "ir_readiness": export["ir_readiness"],
        "best_mode": export["best_mode"],
        "best_coverage_ratio": export["best_coverage_ratio"],
        "best_semantic_coverage_ratio": export["best_semantic_coverage_ratio"],
        "semantic_gap_ratio": export["semantic_gap_ratio"],
        "raw_size": export["raw_size"],
        "stitched_size": export["stitched_size"],
        "start_offset": export["start_offset"],
    }


def _group_modules_by_adb_signature(modules: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for module in modules:
        adb = module.get("adb_info") or {}
        sig = adb.get("family_signature") or adb.get("shape_signature") or adb.get("exact_signature")
        if adb.get("present") and sig:
            groups[sig].append(module)
    return groups


def _export_name_tuple(module: dict) -> tuple[str, ...]:
    return tuple(x["name"] for x in module.get("export_analysis", []))


def _annotate_adb_consensus(modules: list[dict]) -> dict:
    groups = _group_modules_by_adb_signature(modules)
    summary_clusters = []
    resolved_count = 0
    annotated_count = 0

    for sig, members in groups.items():
        if len(members) < 2:
            continue

        interface_counts = Counter(m.get("interface_signature_type") for m in members if m.get("interface_signature_type"))
        layout_counts = Counter(m.get("layout_type") for m in members if m.get("layout_type"))
        export_name_counts = Counter(_export_name_tuple(m) for m in members)
        export_count_set = sorted({m.get("exports_count", 0) for m in members})
        same_export_order = len(export_name_counts) == 1

        top_interface = interface_counts.most_common(1)[0][0] if interface_counts else None
        top_layout = layout_counts.most_common(1)[0][0] if layout_counts else None
        unanimous_interface = bool(interface_counts) and len(interface_counts) == 1
        strong_cluster = unanimous_interface and same_export_order and len(export_count_set) == 1

        sample_paths = [m["path"] for m in members[:8]]
        short_sig = sig[:12]

        if strong_cluster:
            summary_clusters.append({
                "adb_signature": short_sig,
                "module_count": len(members),
                "interface_signature_type": top_interface,
                "layout_type": top_layout,
                "exports_count": export_count_set[0] if export_count_set else None,
                "sample_paths": sample_paths,
            })

        for module in members:
            module["adb_cluster_id"] = short_sig
            module["adb_cluster_size"] = len(members)
            module["adb_peer_examples"] = sample_paths
            module["adb_consensus"] = {
                "strong_family_cluster": strong_cluster,
                "interface_signature_type": top_interface,
                "layout_type": top_layout,
                "same_export_order": same_export_order,
                "export_counts": export_count_set,
            }
            module["resolved_interface_signature_type"] = module.get("interface_signature_type")
            module["resolved_interface_source"] = "native" if module.get("interface_signature_type") else None
            annotated_count += 1

            if strong_cluster and not module.get("interface_signature_type") and top_interface:
                module["resolved_interface_signature_type"] = top_interface
                module["resolved_interface_source"] = "adb_family_cluster"
                resolved_count += 1

    summary_clusters.sort(key=lambda x: (-x["module_count"], x["interface_signature_type"] or "", x["adb_signature"]))
    return {
        "group_count": len(groups),
        "annotated_module_count": annotated_count,
        "resolved_interface_module_count": resolved_count,
        "top_family_clusters": summary_clusters[:25],
    }


def summarize_many(modules: list[dict]) -> dict:
    layout_counts = Counter(m["layout_type"] for m in modules)
    interface_signature_counts = Counter(m["interface_signature_type"] for m in modules if m.get("interface_signature_type"))
    resolved_interface_signature_counts = Counter(m["resolved_interface_signature_type"] for m in modules if m.get("resolved_interface_signature_type"))
    best_mode_counts = Counter()
    ir_readiness_counts = Counter()
    micro_semantic_counts = Counter()

    module_count_nonempty = 0
    export_count = 0
    sum_simple_best = 0.0
    sum_simple_sem = 0.0
    sum_adjusted_best = 0.0
    sum_adjusted_sem = 0.0
    adjusted_count = 0

    low_semantic_modules = []
    low_semantic_exports = []
    high_gap_exports = []
    modules_requiring_attention = []
    unresolved_signature_counts = Counter()

    weighted_entries: list[dict] = []

    for m in modules:
        exports = m.get("export_analysis", [])
        if m.get("simple_avg_best_coverage") is not None:
            module_count_nonempty += 1
            sum_simple_best += m["simple_avg_best_coverage"]
            sum_simple_sem += m["simple_avg_semantic_coverage"]
            low_semantic_modules.append({
                "path": m["path"],
                "layout_type": m["layout_type"],
                "interface_signature_type": m.get("interface_signature_type"),
                "adjusted_avg_best_coverage": m.get("adjusted_avg_best_coverage"),
                "adjusted_avg_semantic_coverage": m.get("adjusted_avg_semantic_coverage"),
                "adjusted_avg_semantic_gap": m.get("adjusted_avg_semantic_gap"),
                "exports_count": m.get("exports_count", 0),
            })
        if m.get("adjusted_avg_best_coverage") is not None:
            adjusted_count += 1
            sum_adjusted_best += m["adjusted_avg_best_coverage"]
            sum_adjusted_sem += m["adjusted_avg_semantic_coverage"]

        weighted_entries.extend(exports)
        export_count += len(exports)

        for x in exports:
            best_mode_counts[x["best_mode"]] += 1
            ir_readiness_counts[x["ir_readiness"]] += 1
            if x.get("micro_semantic_kind"):
                micro_semantic_counts[x["micro_semantic_kind"]] += 1
            if x["best_semantic_coverage_ratio"] < 0.75 and not x.get("data_like") and not x.get("padding_like"):
                low_semantic_exports.append(_export_entry(m["path"], m["layout_type"], m.get("interface_signature_type"), x))
                sig = x.get("raw_signature_hex")
                if sig:
                    unresolved_signature_counts[(x.get("micro_semantic_kind"), sig)] += 1
            if x["semantic_gap_ratio"] >= 0.10 and not x.get("data_like") and not x.get("padding_like"):
                high_gap_exports.append(_export_entry(m["path"], m["layout_type"], m.get("interface_signature_type"), x))

        reasons = m.get("attention_reasons", [])
        if reasons:
            modules_requiring_attention.append({
                "path": m["path"],
                "layout_type": m["layout_type"],
                "interface_signature_type": m.get("interface_signature_type"),
                "attention_reasons": reasons,
                "adjusted_avg_best_coverage": m.get("adjusted_avg_best_coverage"),
                "adjusted_avg_semantic_coverage": m.get("adjusted_avg_semantic_coverage"),
                "adjusted_avg_semantic_gap": m.get("adjusted_avg_semantic_gap"),
            })

    low_semantic_modules.sort(key=lambda x: (
        x["adjusted_avg_semantic_coverage"] if x["adjusted_avg_semantic_coverage"] is not None else 999,
        x["adjusted_avg_semantic_gap"] if x["adjusted_avg_semantic_gap"] is not None else -999,
        x["path"],
    ))
    low_semantic_exports.sort(key=lambda x: (x["best_semantic_coverage_ratio"], -x["semantic_gap_ratio"], x["path"], x["name"]))
    high_gap_exports.sort(key=lambda x: (-x["semantic_gap_ratio"], x["best_semantic_coverage_ratio"], x["path"], x["name"]))
    modules_requiring_attention.sort(key=lambda x: (
        -len(x["attention_reasons"]),
        x["adjusted_avg_semantic_coverage"] if x["adjusted_avg_semantic_coverage"] is not None else 999,
        x["path"],
    ))

    simple_avg_best = (sum_simple_best / module_count_nonempty) if module_count_nonempty else None
    simple_avg_sem = (sum_simple_sem / module_count_nonempty) if module_count_nonempty else None
    adjusted_avg_best = (sum_adjusted_best / adjusted_count) if adjusted_count else None
    adjusted_avg_sem = (sum_adjusted_sem / adjusted_count) if adjusted_count else None
    weighted_best = _weighted_average(weighted_entries, "best_coverage_ratio")
    weighted_sem = _weighted_average(weighted_entries, "best_semantic_coverage_ratio")

    return {
        "module_count": len(modules),
        "nonempty_module_count": module_count_nonempty,
        "export_count": export_count,
        "layout_counts": dict(layout_counts),
        "interface_signature_counts": dict(interface_signature_counts),
        "resolved_interface_signature_counts": dict(resolved_interface_signature_counts),
        "best_mode_counts": dict(best_mode_counts),
        "ir_readiness_counts": dict(ir_readiness_counts),
        "micro_semantic_counts": dict(micro_semantic_counts),
        "overall": {
            "simple_avg_best_coverage": simple_avg_best,
            "weighted_best_coverage": weighted_best,
            "adjusted_avg_best_coverage": adjusted_avg_best,
            "simple_avg_semantic_coverage": simple_avg_sem,
            "weighted_semantic_coverage": weighted_sem,
            "adjusted_avg_semantic_coverage": adjusted_avg_sem,
            "simple_avg_semantic_gap": (simple_avg_best - simple_avg_sem) if simple_avg_best is not None and simple_avg_sem is not None else None,
            "weighted_semantic_gap": (weighted_best - weighted_sem) if weighted_best is not None and weighted_sem is not None else None,
            "adjusted_avg_semantic_gap": (adjusted_avg_best - adjusted_avg_sem) if adjusted_avg_best is not None and adjusted_avg_sem is not None else None,
            "exports_below_0_75_best": sum(1 for x in weighted_entries if x["best_coverage_ratio"] < 0.75),
            "exports_below_0_75_semantic": sum(1 for x in weighted_entries if x["best_semantic_coverage_ratio"] < 0.75),
            "exports_semantic_gap_ge_0_10": sum(1 for x in weighted_entries if x["semantic_gap_ratio"] >= 0.10),
            "exports_semantic_gap_ge_0_25": sum(1 for x in weighted_entries if x["semantic_gap_ratio"] >= 0.25),
        },
        "lowest_semantic_modules": low_semantic_modules[:25],
        "lowest_semantic_exports": low_semantic_exports[:50],
        "highest_semantic_gap_exports": high_gap_exports[:50],
        "modules_requiring_attention": modules_requiring_attention[:100],
        "top_unresolved_signatures": [
            {"micro_semantic_kind": kind, "raw_signature_hex": sig, "count": count}
            for (kind, sig), count in unresolved_signature_counts.most_common(25)
        ],
    }


def analyze_module(path: str | Path, overrides: dict | None = None) -> dict:
    mod = MBCModule(path, overrides=overrides)
    vm_code_base = _infer_vm_code_base(mod)
    header_overlap_names = {r.name for r in mod.header_overlap_exports(len(MAGIC_HEADER))}

    export_analysis = []
    for idx, name in enumerate(mod.export_names()):
        if vm_code_base is not None:
            pair = _vm_export_slices(mod, vm_code_base, name, next_head_bytes=16)
            if pair is None:
                raw = mod.get_export_body(name)
                stitched = mod.stitch_export_body(name, next_head_bytes=16)
                addr_mode = "file_fallback"
            else:
                raw, stitched = pair
                addr_mode = "vmaddr"
        else:
            raw = mod.get_export_body(name)
            stitched = mod.stitch_export_body(name, next_head_bytes=16)
            addr_mode = "file_offset"

        raw_tokens = tokenize_stream(raw)
        stitched_tokens = tokenize_stream(stitched)
        raw_cov = coverage(raw_tokens, len(raw))
        raw_sem = _semantic_coverage(raw_tokens, len(raw))
        stitched_cov = coverage(stitched_tokens, len(stitched))
        stitched_sem = _semantic_coverage(stitched_tokens, len(stitched))

        repaired_cov = None
        repaired_sem = None
        repaired_size = None
        repaired_borrow = 0
        repaired = _repair_export_tail(mod, name, raw)
        if repaired is not None:
            repaired_bytes, repaired_tokens, repaired_borrow = repaired
            repaired_cov = coverage(repaired_tokens, len(repaired_bytes))
            repaired_sem = _semantic_coverage(repaired_tokens, len(repaired_bytes))
            repaired_size = len(repaired_bytes)

        candidates = [("raw", raw_cov, raw_sem, raw_tokens)]
        if repaired_cov is not None and repaired_sem is not None and repaired is not None:
            candidates.append(("repaired", repaired_cov, repaired_sem, repaired_tokens))
        candidates.append(("stitched", stitched_cov, stitched_sem, stitched_tokens))
        best_mode, best_cov, best_sem, best_tokens = max(candidates, key=lambda item: item[1]["coverage_ratio"])

        dominant_byte, dominant_ratio = _dominant_byte_info(raw)
        entry = {
            "name": name,
            "start_offset": mod.exports[idx].a,
            "addr_mode": addr_mode,
            "header_overlap": (name in header_overlap_names) and vm_code_base is None,
            "raw_size": len(raw),
            "repaired_size": repaired_size,
            "repaired_borrowed_bytes": repaired_borrow,
            "stitched_size": len(stitched),
            "raw_coverage_ratio": raw_cov["coverage_ratio"],
            "raw_semantic_coverage_ratio": raw_sem["coverage_ratio"],
            "best_mode": best_mode,
            "best_coverage_ratio": best_cov["coverage_ratio"],
            "best_semantic_coverage_ratio": best_sem["coverage_ratio"],
            "dominant_byte": dominant_byte,
            "dominant_byte_ratio": dominant_ratio,
            "raw_signature_hex": raw[:32].hex(" "),
        }
        entry["micro_semantic_kind"] = _infer_micro_semantic_kind(best_tokens)
        entry["data_like"] = _data_like_export_reason(entry["name"], entry["best_coverage_ratio"])
        entry["padding_like"] = _is_padding_like_export(entry)
        entry["semantic_gap_ratio"] = entry["best_coverage_ratio"] - entry["best_semantic_coverage_ratio"]
        entry["ir_readiness"] = _classify_ir_readiness(entry)
        export_analysis.append(entry)

    result = {
        "path": str(Path(path)),
        "has_magic_header": mod.has_magic_header,
        "adb_info": mod.adb_info.to_dict(),
        "definition_count": len(mod.definitions),
        "globals_count": len(mod.globals),
        "exports_count": len(mod.exports),
        "embedded_import_like_export_count": len(mod.embedded_import_like_exports),
        "layout_type": classify_layout(mod),
        "interface_signature_type": classify_interface_signature(mod),
        "resolved_interface_signature_type": classify_interface_signature(mod),
        "resolved_interface_source": "native" if classify_interface_signature(mod) else None,
        "export_address_mode": "vmaddr" if vm_code_base is not None else "file_offset",
        "vm_code_base": vm_code_base,
        "header_overlap_exports": [name for name in mod.export_names() if name in header_overlap_names and vm_code_base is None],
        "export_analysis": export_analysis,
    }

    result["simple_avg_best_coverage"] = (
        sum(x["best_coverage_ratio"] for x in export_analysis) / len(export_analysis)
        if export_analysis else None
    )
    result["weighted_best_coverage"] = _weighted_average(export_analysis, "best_coverage_ratio")
    result["simple_avg_semantic_coverage"] = (
        sum(x["best_semantic_coverage_ratio"] for x in export_analysis) / len(export_analysis)
        if export_analysis else None
    )
    result["weighted_semantic_coverage"] = _weighted_average(export_analysis, "best_semantic_coverage_ratio")

    adjusted_entries = [x for x in export_analysis if not _is_stub_like(x)]
    result["adjusted_avg_best_coverage"] = (
        sum(x["best_coverage_ratio"] for x in adjusted_entries) / len(adjusted_entries)
        if adjusted_entries else None
    )
    result["adjusted_avg_semantic_coverage"] = (
        sum(x["best_semantic_coverage_ratio"] for x in adjusted_entries) / len(adjusted_entries)
        if adjusted_entries else None
    )
    result["simple_avg_semantic_gap"] = (
        result["simple_avg_best_coverage"] - result["simple_avg_semantic_coverage"]
        if result["simple_avg_best_coverage"] is not None and result["simple_avg_semantic_coverage"] is not None else None
    )
    result["weighted_semantic_gap"] = (
        result["weighted_best_coverage"] - result["weighted_semantic_coverage"]
        if result["weighted_best_coverage"] is not None and result["weighted_semantic_coverage"] is not None else None
    )
    result["adjusted_avg_semantic_gap"] = (
        result["adjusted_avg_best_coverage"] - result["adjusted_avg_semantic_coverage"]
        if result["adjusted_avg_best_coverage"] is not None and result["adjusted_avg_semantic_coverage"] is not None else None
    )
    result["attention_reasons"] = module_attention_reasons(result)
    return result


def analyze_modules(paths: list[str | Path], overrides: dict | None = None) -> list[dict]:
    return [analyze_module(path, overrides=overrides) for path in paths]


def analyze_many(paths: list[str | Path], overrides: dict | None = None) -> dict:
    modules = analyze_modules(paths, overrides=overrides)
    adb_summary = _annotate_adb_consensus(modules)
    summary = summarize_many(modules)
    summary["adb_consensus"] = adb_summary
    return {
        "summary": summary,
        "modules": modules,
    }


def dump_json(obj: dict, out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
