from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math

from .parser import MBCModule
from .tokenizer import Token, tokenize_stream, coverage

# These are token kinds that provide relatively strong evidence that we matched a real
# bytecode structure, not just filler, plain data, or a stream of trivially accepted ops.
HARD_SEMANTIC_TOKEN_KINDS = {
    "AGG", "AGG0",
    "REF", "REF16",
    "REC41", "REC61", "REC62",
    "CALL66", "CALL63A", "CALL63B",
    "IMM", "IMM16", "IMM24S", "IMM24U", "IMM32",
    "F32", "BR", "OPU16",
    "SIG_USEOWNER_HEAD", "SIG_USECLIENT_HEAD", "SIG_UNIQUEGEN_HEAD",
    "SIG_USEOFF_HEAD", "SIG_INPUTDONE_HEAD",
    "SIG_U32_U8_CALL66_TAIL", "SIG_AGG2_PARTIAL_HEAD", "SIG_AGG1_PARTIAL_HEAD",
    "SIG_USECLIENT_ALT_HEAD", "SIG_CALL66_REFPAIR_HEAD", "SIG_CALL66_SMALLIMM",
    "SIG_CONST_U32_TRAILER", "SIG_SLOT_CONST", "SIG_SETOSST_HEAD",
    "SIG_GETPLAYERID_HEAD", "SIG_USEOFF_CONST_CHAIN", "SIG_GETCASTLENUM_HEAD",
}

PAD_DRIVEN_TOKEN_KINDS = {
    "PAD",
    "SIG_PAD17", "SIG_PAD11_BR", "SIG_PADRUN_BR", "SIG_PADRUN_OPREF",
    "SIG_PADDED_CHECKPUT", "SIG_GETMODIFIERS_PADTAIL",
}

DATA_TOKEN_KINDS = {"ASCII", "DWBLOB"}

SUSPICIOUS_PREFIX_DEF_NAMES = {"ITSOFF", "MBS", "ie"}

DATA_LIKE_EXPORT_NAMES = {"pClanName", "pUSTATE", "GetTableTime"}

SAFE_MICRO_KIND_BY_FIRST_TOKEN = {
    "SIG_USEOFF_HEAD": "useoff_wrapper",
    "SIG_USECLIENT_HEAD": "useclient_wrapper",
    "SIG_USEOWNER_HEAD": "useowner_wrapper",
    "SIG_UNIQUEGEN_HEAD": "unique_generation_wrapper_head",
    "SIG_INPUTDONE_HEAD": "inputdone_wrapper_head",
    "SIG_U32_U8_CALL66_TAIL": "u32_call66_tail_wrapper",
    "SIG_AGG2_PARTIAL_HEAD": "agg2_partial_head",
    "SIG_AGG1_PARTIAL_HEAD": "agg1_partial_head",
    "SIG_USECLIENT_ALT_HEAD": "useclient_alt_wrapper_head",
    "SIG_CALL66_SMALLIMM": "call66_smallimm_wrapper",
    "SIG_CALL66_REFPAIR_HEAD": "call66_refpair_wrapper",
    "SIG_CONST_U32_TRAILER": "u32_const_trailer",
    "SIG_SLOT_CONST": "slot_const_wrapper",
    "SIG_SETOSST_HEAD": "setosst_wrapper",
    "SIG_GETPLAYERID_HEAD": "getplayerid_wrapper",
    "SIG_USEOFF_CONST_CHAIN": "useoff_const_chain",
    "SIG_GETCASTLENUM_HEAD": "getcastlenum_wrapper",
}


def _ratio(part: int, whole: int) -> float:
    return (part / whole) if whole else 0.0


def _token_bytes(tokens: list[Token], include: set[str]) -> int:
    return sum(tok.size for tok in tokens if tok.kind in include)



def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = Counter(data)
    n = len(data)
    acc = 0.0
    for count in freq.values():
        p = count / n
        acc -= p * math.log2(p)
    return acc



def _dominant_byte_info(data: bytes) -> tuple[int | None, float]:
    if not data:
        return None, 0.0
    value, count = Counter(data).most_common(1)[0]
    return value, count / len(data)



def _op_stats(tokens: list[Token]) -> dict:
    op_values = [tok.payload.get("op") for tok in tokens if tok.kind == "OP"]
    op_values = [x for x in op_values if isinstance(x, int)]
    if not op_values:
        return {
            "op_token_count": 0,
            "unique_op_values": 0,
            "dominant_op_value": None,
            "dominant_op_ratio": 0.0,
        }

    counts = Counter(op_values)
    value, count = counts.most_common(1)[0]
    return {
        "op_token_count": len(op_values),
        "unique_op_values": len(counts),
        "dominant_op_value": value,
        "dominant_op_ratio": count / len(op_values),
    }



def _infer_micro_semantic_kind(tokens: list[Token]) -> str | None:
    filtered = [
        tok.kind for tok in tokens
        if tok.kind not in PAD_DRIVEN_TOKEN_KINDS
        and tok.kind not in DATA_TOKEN_KINDS
        and tok.kind not in {"OP", "UNK"}
    ]
    if not filtered:
        return None
    first = filtered[0]
    if first in SAFE_MICRO_KIND_BY_FIRST_TOKEN:
        return SAFE_MICRO_KIND_BY_FIRST_TOKEN[first]
    if filtered == ["AGG"] or filtered == ["AGG0"]:
        return "aggregate_wrapper"
    if filtered == ["AGG0", "IMM"]:
        return "agg0_const_wrapper"
    if filtered == ["AGG0", "REF"] or filtered == ["AGG0", "REF16"]:
        return "agg0_ref_wrapper"
    if filtered in (["AGG0", "OPU16"], ["AGG0", "OPU16", "IMM"], ["AGG0", "OPU16", "REF"], ["AGG0", "OPU16", "REF16"]):
        return "agg0_u16_wrapper"
    if first == "AGG0" and all(kind in {"IMM", "REF", "REF16", "OPU16", "CALL66", "CALL63A", "CALL63B", "BR", "SIG_U32_U8_CALL66_TAIL"} for kind in filtered[1:]):
        return "agg0_head_wrapper"
    if filtered and all(kind == "REF" for kind in filtered):
        return "ref_chain"
    return None



def _byte_profile(data: bytes) -> dict:
    dominant, dominant_ratio = _dominant_byte_info(data)
    size = len(data)
    return {
        "raw_size": size,
        "zero_byte_ratio": _ratio(data.count(0x00), size),
        "pad7c_byte_ratio": _ratio(data.count(0x7C), size),
        "ff_byte_ratio": _ratio(data.count(0xFF), size),
        "dominant_byte": dominant,
        "dominant_byte_ratio": dominant_ratio,
        "entropy_bits_per_byte": _entropy(data),
    }



def _filler_reasons(profile: dict, op_stats: dict, hard_sem_ratio: float, pad_ratio: float) -> list[str]:
    reasons: list[str] = []
    if profile["raw_size"] == 0:
        reasons.append("empty_slice")
        return reasons

    dominant = profile.get("dominant_byte")
    dominant_ratio = profile.get("dominant_byte_ratio", 0.0)
    if dominant in (0x00, 0x7C, 0xFF) and dominant_ratio >= 0.60:
        reasons.append("dominant_filler_byte")
    if profile.get("zero_byte_ratio", 0.0) >= 0.50:
        reasons.append("zero_dominated")
    if profile.get("pad7c_byte_ratio", 0.0) >= 0.40:
        reasons.append("pad7c_dominated")
    if profile.get("ff_byte_ratio", 0.0) >= 0.40:
        reasons.append("ff_dominated")
    if op_stats["op_token_count"] > 0 and op_stats["unique_op_values"] <= 2 and op_stats["dominant_op_value"] in (0x00, 0x7C, 0xFF) and op_stats["dominant_op_ratio"] >= 0.80:
        reasons.append("degenerate_op_stream")
    if pad_ratio >= 0.35 and hard_sem_ratio < 0.30:
        reasons.append("pad_driven_match")
    if profile.get("entropy_bits_per_byte", 0.0) <= 1.0 and hard_sem_ratio < 0.30:
        reasons.append("very_low_entropy")
    return sorted(set(reasons))



def _classify_evidence_level(slice_status: str, hard_sem_ratio: float, recognized_nonpadding_ratio: float, suspicious_reasons: list[str]) -> str:
    if hard_sem_ratio >= 0.80 and not suspicious_reasons:
        return "strong"
    if hard_sem_ratio >= 0.50 and len(suspicious_reasons) <= 1:
        return "moderate"
    if hard_sem_ratio >= 0.20 or recognized_nonpadding_ratio >= 0.50:
        return "weak"
    return "unresolved"



def _classify_ir_readiness(slice_status: str, evidence_level: str, suspicious_reasons: list[str], hard_sem_ratio: float) -> str:
    if evidence_level == "strong":
        return "candidate_ir"
    if evidence_level == "moderate":
        return "review_required"
    if suspicious_reasons and hard_sem_ratio < 0.30:
        return "not_ready"
    if evidence_level == "weak":
        return "not_ready"
    return "not_ready"



def _data_like_export_reason(name: str, data_ratio: float, dominant_byte_ratio: float) -> bool:
    if name in DATA_LIKE_EXPORT_NAMES and data_ratio >= 0.30:
        return True
    if name.startswith("p") and len(name) > 1 and name[1].isupper() and dominant_byte_ratio >= 0.40:
        return True
    return False



def _raw_export_span(mod: MBCModule, name: str) -> tuple[int, int, int]:
    names = mod.export_names()
    idx = names.index(name)
    start = mod.exports[idx].a
    if idx + 1 < len(mod.exports):
        end = mod.exports[idx + 1].a
    elif mod.code_size:
        end = mod.code_size
    else:
        end = max(start, len(mod.data) - mod.code_base)
    return idx, start, end



def _definition_backed_file_slice(mod: MBCModule, name: str) -> tuple[bytes, dict] | None:
    idx, start, raw_end = _raw_export_span(mod, name)
    defs_by_name = {r.name: r for r in mod.definitions}
    rec = defs_by_name.get(name)
    if rec is None:
        return None
    if rec.a != start or rec.b < rec.a:
        return None

    code_end = rec.b + 1
    if mod.code_size and code_end > mod.code_size:
        return None
    if code_end > raw_end:
        return None

    file_start = mod.code_base + start
    file_end = mod.code_base + code_end
    if file_start < 0 or file_end > len(mod.data) or file_start >= file_end:
        return None

    return mod.data[file_start:file_end], {
        "slice_status": "definition_exact",
        "slice_proof": "definition_table_code_relative",
        "raw_export_span_size": raw_end - start,
        "code_body_size": code_end - start,
        "trailing_data_size": raw_end - code_end,
        "slice_start": file_start,
        "slice_end": file_end,
        "code_offset_start": start,
        "code_offset_end": code_end,
        "export_index": idx,
    }


def _override_vm_slice(mod: MBCModule, name: str, code_base: int) -> tuple[bytes, dict] | None:
    defs_by_name = {r.name: r for r in mod.definitions}
    rec = defs_by_name.get(name)
    if rec is None or rec.b < rec.a:
        return None

    start = code_base + rec.a
    end = code_base + rec.b + 1
    limit = code_base + (mod.code_size or max(rec.b + 1, 0))
    if start < 0 or end > min(limit, len(mod.data)) or start >= end:
        return None

    idx = mod.export_names().index(name)
    return mod.data[start:end], {
        "slice_status": "vm_definition_exact",
        "slice_proof": "override_code_base",
        "raw_export_span_size": None,
        "code_body_size": end - start,
        "trailing_data_size": None,
        "slice_start": start,
        "slice_end": end,
        "code_offset_start": rec.a,
        "code_offset_end": rec.b + 1,
        "export_index": idx,
    }


def _unverified_file_span(mod: MBCModule, name: str) -> tuple[bytes, dict] | None:
    idx, start, end = _raw_export_span(mod, name)
    file_start = mod.code_base + start
    file_end = mod.code_base + end
    code_limit = mod.code_base + (mod.code_size or (len(mod.data) - mod.code_base))
    if file_start < 0 or file_end > min(code_limit, len(mod.data)) or file_start >= file_end:
        return None
    return mod.data[file_start:file_end], {
        "slice_status": "export_span_unverified",
        "slice_proof": "export_table_code_relative",
        "raw_export_span_size": end - start,
        "code_body_size": end - start,
        "trailing_data_size": None,
        "slice_start": file_start,
        "slice_end": file_end,
        "code_offset_start": start,
        "code_offset_end": end,
        "export_index": idx,
    }


def _select_export_slice(mod: MBCModule, name: str, override_entry: dict | None) -> tuple[bytes, dict]:
    if override_entry and "code_base" in override_entry:
        vm = _override_vm_slice(mod, name, int(override_entry["code_base"]))
        if vm is not None:
            return vm

    idx, start, end = _raw_export_span(mod, name)

    exact = _definition_backed_file_slice(mod, name)
    if exact is not None:
        return exact

    span = _unverified_file_span(mod, name)
    if span is not None:
        return span

    return b"", {
        "slice_status": "invalid_export_span",
        "slice_proof": None,
        "raw_export_span_size": max(0, end - start),
        "code_body_size": None,
        "trailing_data_size": None,
        "slice_start": None,
        "slice_end": None,
        "code_offset_start": start,
        "code_offset_end": end,
        "export_index": idx,
    }


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



def _analyze_export(mod: MBCModule, name: str, override_entry: dict | None) -> dict:
    raw, meta = _select_export_slice(mod, name, override_entry)
    tokens = tokenize_stream(raw) if raw else []
    raw_cov = coverage(tokens, len(raw)) if raw else {"coverage_ratio": 0.0, "covered_bytes": 0, "total_bytes": len(raw), "token_counts": {}}

    hard_sem_bytes = _token_bytes(tokens, HARD_SEMANTIC_TOKEN_KINDS)
    pad_bytes = _token_bytes(tokens, PAD_DRIVEN_TOKEN_KINDS)
    data_bytes = _token_bytes(tokens, DATA_TOKEN_KINDS)
    op_bytes = _token_bytes(tokens, {"OP"})
    recognized_nonpadding_bytes = sum(tok.size for tok in tokens if tok.kind not in PAD_DRIVEN_TOKEN_KINDS and tok.kind != "UNK")

    hard_sem_ratio = _ratio(hard_sem_bytes, len(raw))
    pad_ratio = _ratio(pad_bytes, len(raw))
    data_ratio = _ratio(data_bytes, len(raw))
    op_ratio = _ratio(op_bytes, len(raw))
    recognized_nonpadding_ratio = _ratio(recognized_nonpadding_bytes, len(raw))

    profile = _byte_profile(raw)
    op_stats = _op_stats(tokens)
    suspicious_reasons = _filler_reasons(profile, op_stats, hard_sem_ratio, pad_ratio)
    evidence_level = _classify_evidence_level(meta["slice_status"], hard_sem_ratio, recognized_nonpadding_ratio, suspicious_reasons)
    ir_readiness = _classify_ir_readiness(meta["slice_status"], evidence_level, suspicious_reasons, hard_sem_ratio)

    entry = {
        "name": name,
        "start_offset": mod.exports[meta["export_index"]].a,
        "slice_status": meta["slice_status"],
        "slice_proof": meta["slice_proof"],
        "slice_start": meta["slice_start"],
        "slice_end": meta["slice_end"],
        "code_offset_start": meta.get("code_offset_start"),
        "code_offset_end": meta.get("code_offset_end"),
        "raw_export_span_size": meta["raw_export_span_size"],
        "code_body_size": meta["code_body_size"],
        "trailing_data_size": meta["trailing_data_size"],
        "raw_size": len(raw),
        "tokenizer_recognition_ratio": raw_cov["coverage_ratio"],
        "recognized_nonpadding_ratio": recognized_nonpadding_ratio,
        "hard_semantic_ratio": hard_sem_ratio,
        "opcode_ratio": op_ratio,
        "pad_ratio": pad_ratio,
        "data_ratio": data_ratio,
        "token_counts": raw_cov["token_counts"],
        "dominant_byte": profile["dominant_byte"],
        "dominant_byte_ratio": profile["dominant_byte_ratio"],
        "zero_byte_ratio": profile["zero_byte_ratio"],
        "pad7c_byte_ratio": profile["pad7c_byte_ratio"],
        "ff_byte_ratio": profile["ff_byte_ratio"],
        "entropy_bits_per_byte": profile["entropy_bits_per_byte"],
        "op_token_count": op_stats["op_token_count"],
        "unique_op_values": op_stats["unique_op_values"],
        "dominant_op_value": op_stats["dominant_op_value"],
        "dominant_op_ratio": op_stats["dominant_op_ratio"],
        "micro_semantic_kind": _infer_micro_semantic_kind(tokens),
        "data_like": _data_like_export_reason(name, data_ratio, profile["dominant_byte_ratio"]),
        "evidence_level": evidence_level,
        "ir_readiness": ir_readiness,
        "suspicious_reasons": suspicious_reasons,
        "raw_signature_hex": raw[:32].hex(" ") if raw else "",
    }
    return entry



def _weighted_average(entries: list[dict], key: str, weight_key: str = "raw_size") -> float | None:
    weighted = [(entry.get(weight_key, 0), entry.get(key)) for entry in entries if entry.get(key) is not None and entry.get(weight_key, 0) > 0]
    if not weighted:
        return None
    total_w = sum(w for w, _ in weighted)
    return sum(w * v for w, v in weighted) / total_w if total_w else None



def module_attention_reasons(result: dict) -> list[str]:
    reasons: list[str] = []
    if result.get("layout_type") == "empty_or_unparsed":
        reasons.append("empty_or_unparsed")
    elif result.get("layout_type") == "definitions_only" and result.get("exports_count", 0) == 0:
        reasons.append("definitions_only")

    if result.get("interface_signature_type") == "prefix_contaminated_definitions":
        reasons.append("prefix_contamination")

    if (result.get("strong_export_ratio") or 0.0) < 0.30 and result.get("exports_count", 0) > 0:
        reasons.append("low_strong_evidence")

    if (result.get("weighted_opcode_ratio") or 0.0) >= 0.60 and (result.get("weighted_hard_semantic_ratio") or 0.0) < 0.30:
        reasons.append("opcode_dominated")

    if (result.get("weighted_pad_ratio") or 0.0) >= 0.25:
        reasons.append("padding_dominated")

    if (result.get("weighted_zero_byte_ratio") or 0.0) >= 0.40:
        reasons.append("zero_dominated")

    suspicious = [x["name"] for x in result.get("export_analysis", []) if x.get("suspicious_reasons")]
    if suspicious:
        reasons.append("suspicious_exports")

    return sorted(set(reasons))



def _export_entry(path: str, layout_type: str, interface_signature_type: str | None, export: dict) -> dict:
    return {
        "path": path,
        "layout_type": layout_type,
        "interface_signature_type": interface_signature_type,
        "name": export["name"],
        "evidence_level": export["evidence_level"],
        "ir_readiness": export["ir_readiness"],
        "hard_semantic_ratio": export["hard_semantic_ratio"],
        "opcode_ratio": export["opcode_ratio"],
        "pad_ratio": export["pad_ratio"],
        "data_ratio": export["data_ratio"],
        "raw_size": export["raw_size"],
        "start_offset": export["start_offset"],
        "slice_status": export["slice_status"],
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
        stable_cluster = same_export_order and len(export_count_set) == 1

        sample_paths = [m["path"] for m in members[:8]]
        short_sig = sig[:12]

        if stable_cluster:
            summary_clusters.append({
                "adb_signature": short_sig,
                "module_count": len(members),
                "interface_name_pattern_hint": top_interface,
                "layout_type_hint": top_layout,
                "exports_count": export_count_set[0] if export_count_set else None,
                "sample_paths": sample_paths,
            })

        for module in members:
            module["adb_cluster_id"] = short_sig
            module["adb_cluster_size"] = len(members)
            module["adb_peer_examples"] = sample_paths
            module["adb_consensus"] = {
                "stable_family_shape": stable_cluster,
                "interface_name_pattern_hint": top_interface,
                "layout_type_hint": top_layout,
                "same_export_order": same_export_order,
                "export_counts": export_count_set,
            }
            # Deliberately do not auto-resolve semantics from adb clustering.
            module["resolved_interface_signature_type"] = module.get("interface_signature_type")
            module["resolved_interface_source"] = module.get("resolved_interface_source")
            annotated_count += 1

    summary_clusters.sort(key=lambda x: (-x["module_count"], x["interface_name_pattern_hint"] or "", x["adb_signature"]))
    return {
        "group_count": len(groups),
        "annotated_module_count": annotated_count,
        "resolved_interface_module_count": 0,
        "top_family_clusters": summary_clusters[:25],
    }



def summarize_many(modules: list[dict]) -> dict:
    layout_counts = Counter(m.get("layout_type") for m in modules)
    interface_signature_counts = Counter(m.get("interface_signature_type") for m in modules if m.get("interface_signature_type"))
    resolved_interface_signature_counts = Counter(m.get("resolved_interface_signature_type") for m in modules if m.get("resolved_interface_signature_type"))
    evidence_level_counts = Counter()
    ir_readiness_counts = Counter()
    suspicious_reason_counts = Counter()
    modules_requiring_attention = []
    low_evidence_modules = []
    weighted_entries: list[dict] = []

    for m in modules:
        exports = m.get("export_analysis", [])
        for e in exports:
            evidence_level_counts[e["evidence_level"]] += 1
            ir_readiness_counts[e["ir_readiness"]] += 1
            suspicious_reason_counts.update(e.get("suspicious_reasons", []))
            weighted_entries.append(_export_entry(m["path"], m.get("layout_type"), m.get("interface_signature_type"), e))

        low_evidence_modules.append({
            "path": m["path"],
            "layout_type": m.get("layout_type"),
            "interface_signature_type": m.get("interface_signature_type"),
            "strong_export_ratio": m.get("strong_export_ratio"),
            "weighted_hard_semantic_ratio": m.get("weighted_hard_semantic_ratio"),
            "weighted_opcode_ratio": m.get("weighted_opcode_ratio"),
            "weighted_pad_ratio": m.get("weighted_pad_ratio"),
            "exports_count": m.get("exports_count", 0),
        })
        if m.get("attention_reasons"):
            modules_requiring_attention.append({
                "path": m["path"],
                "layout_type": m.get("layout_type"),
                "interface_signature_type": m.get("interface_signature_type"),
                "attention_reasons": m.get("attention_reasons"),
                "strong_export_ratio": m.get("strong_export_ratio"),
                "weighted_hard_semantic_ratio": m.get("weighted_hard_semantic_ratio"),
                "exports_count": m.get("exports_count", 0),
            })

    low_evidence_modules.sort(key=lambda x: (x.get("strong_export_ratio") or 0.0, x.get("weighted_hard_semantic_ratio") or 0.0, -(x.get("exports_count") or 0), x["path"]))
    modules_requiring_attention.sort(key=lambda x: (-len(x["attention_reasons"]), x.get("strong_export_ratio") or 0.0, x["path"]))

    weighted_hard = _weighted_average([{"raw_size": e["raw_size"], "value": e["hard_semantic_ratio"]} for e in weighted_entries], "value")
    weighted_op = _weighted_average([{"raw_size": e["raw_size"], "value": e["opcode_ratio"]} for e in weighted_entries], "value")
    weighted_pad = _weighted_average([{"raw_size": e["raw_size"], "value": e["pad_ratio"]} for e in weighted_entries], "value")
    weighted_data = _weighted_average([{"raw_size": e["raw_size"], "value": e["data_ratio"]} for e in weighted_entries], "value")

    return {
        "module_count": len(modules),
        "layout_counts": dict(layout_counts),
        "interface_signature_counts": dict(interface_signature_counts),
        "resolved_interface_signature_counts": dict(resolved_interface_signature_counts),
        "evidence_level_counts": dict(evidence_level_counts),
        "ir_readiness_counts": dict(ir_readiness_counts),
        "suspicious_reason_counts": dict(suspicious_reason_counts),
        "weighted_hard_semantic_ratio": weighted_hard,
        "weighted_opcode_ratio": weighted_op,
        "weighted_pad_ratio": weighted_pad,
        "weighted_data_ratio": weighted_data,
        "lowest_evidence_modules": low_evidence_modules[:25],
        "modules_requiring_attention": modules_requiring_attention[:25],
    }



def analyze_module(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    override_entry = None
    if overrides:
        override_entry = overrides.get(path.name)

    mod = MBCModule(path, overrides=overrides)
    layout_type = classify_layout(mod)
    interface_signature_type = classify_interface_signature(mod)

    export_analysis = [_analyze_export(mod, name, override_entry) for name in mod.export_names()]

    requested_override_code_base = int(override_entry["code_base"]) if override_entry and "code_base" in override_entry else None
    slice_proofs = {x.get("slice_proof") for x in export_analysis if x.get("slice_proof")}
    if slice_proofs == {"override_code_base"}:
        export_address_mode = "override_vm_code_base"
        vm_code_base = requested_override_code_base
    elif "override_code_base" in slice_proofs:
        export_address_mode = "mixed_override_and_header_relative"
        vm_code_base = requested_override_code_base
    else:
        export_address_mode = "header_code_relative"
        vm_code_base = mod.code_base

    result = {
        "path": str(path),
        "has_magic_header": mod.has_magic_header,
        "adb_info": mod.adb_info.to_dict(),
        "definition_count": len(mod.definitions),
        "globals_count": len(mod.globals),
        "exports_count": len(mod.exports),
        "embedded_import_like_export_count": len(mod.embedded_import_like_exports),
        "layout_type": layout_type,
        "interface_signature_type": interface_signature_type,
        "resolved_interface_signature_type": interface_signature_type,
        "resolved_interface_source": "names_only_pattern" if interface_signature_type else None,
        "export_address_mode": export_address_mode,
        "requested_override_code_base": requested_override_code_base,
        "vm_code_base": vm_code_base,
        "code_base": mod.code_base,
        "code_size": mod.code_size,
        "data_blob_size": mod.data_blob_size,
        "code_prefix_exports": [r.name for r in mod.exports if r.a < 16],
        "export_analysis": export_analysis,
    }

    result["weighted_tokenizer_recognition_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["tokenizer_recognition_ratio"]} for x in export_analysis], "value")
    result["weighted_hard_semantic_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["hard_semantic_ratio"]} for x in export_analysis], "value")
    result["weighted_opcode_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["opcode_ratio"]} for x in export_analysis], "value")
    result["weighted_pad_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["pad_ratio"]} for x in export_analysis], "value")
    result["weighted_data_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["data_ratio"]} for x in export_analysis], "value")
    result["weighted_zero_byte_ratio"] = _weighted_average([{"raw_size": x["raw_size"], "value": x["zero_byte_ratio"]} for x in export_analysis], "value")

    strong = [x for x in export_analysis if x["evidence_level"] == "strong"]
    moderate = [x for x in export_analysis if x["evidence_level"] == "moderate"]
    weak = [x for x in export_analysis if x["evidence_level"] == "weak"]
    unresolved = [x for x in export_analysis if x["evidence_level"] == "unresolved"]
    suspicious = [x for x in export_analysis if x.get("suspicious_reasons")]

    total_exports = len(export_analysis)
    result["strong_export_ratio"] = _ratio(len(strong), total_exports)
    result["moderate_export_ratio"] = _ratio(len(moderate), total_exports)
    result["weak_export_ratio"] = _ratio(len(weak), total_exports)
    result["unresolved_export_ratio"] = _ratio(len(unresolved), total_exports)
    result["suspicious_export_ratio"] = _ratio(len(suspicious), total_exports)

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
