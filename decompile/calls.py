from __future__ import annotations

"""Runtime call catalog and stack-effect policy.

This module owns both native/builtin API specifications and the decompiler's
call-effect model.  Keeping these together avoids a split where one file names
a native call and another nearby file decides whether that same call returns a
value, mutates state, or should be emitted as a statement.
"""

from dataclasses import dataclass, replace
import re
from typing import Any

from mbc_format.common import TYPE_FLOAT, TYPE_INT, TYPE_SLICE, TYPE_STRING
from mbc_format.opcodes import BUILTINS

@dataclass(frozen=True)
class NativeCallSpec:
    name: str
    arity: int | None = None
    arg_types: tuple[str, ...] = ()
    return_type: str = "void"
    return_type_id: int | None = None
    pushes: int | None = None
    side_effects: tuple[str, ...] = ()
    confidence: str = "inferred"
    note: str = ""
    layer: str = "native"
    selector: int | None = None
    source: str = ""

    @property
    def returns_value(self) -> bool:
        pushes = self.pushes if self.pushes is not None else (0 if self.return_type == "void" else 1)
        return pushes > 0 and self.return_type != "void"

    @property
    def push_count(self) -> int:
        if self.pushes is not None:
            return self.pushes
        return 1 if self.return_type != "void" else 0

    def render_signature(self) -> str:
        if self.arity is None:
            parts = [f"{typ} arg{i}" for i, typ in enumerate(self.arg_types)]
            parts.append("...")
        else:
            parts = []
            for i in range(self.arity):
                typ = self.arg_types[i] if i < len(self.arg_types) else "unknown"
                parts.append(f"{typ} arg{i}")
        ret = "" if self.return_type == "void" else f" -> {self.return_type}"
        return f"({', '.join(parts)}){ret}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arity": self.arity,
            "arg_types": list(self.arg_types),
            "return_type": self.return_type,
            "return_type_id": self.return_type_id,
            "pushes": self.push_count,
            "side_effects": list(self.side_effects),
            "confidence": self.confidence,
            "note": self.note,
            "layer": self.layer,
            "selector": self.selector,
            "source": self.source,
        }


def _spec(
    name: str,
    *,
    arity: int | None = None,
    arg_types: tuple[str, ...] = (),
    return_type: str = "void",
    return_type_id: int | None = None,
    pushes: int | None = None,
    side_effects: tuple[str, ...] = ("runtime_call",),
    confidence: str = "recovered",
    note: str = "",
    layer: str = "native",
    selector: int | None = None,
    source: str = "",
) -> NativeCallSpec:
    if pushes is None:
        pushes = 0 if return_type == "void" else 1
    return NativeCallSpec(
        name=name,
        arity=arity,
        arg_types=arg_types,
        return_type=return_type,
        return_type_id=return_type_id,
        pushes=pushes,
        side_effects=side_effects,
        confidence=confidence,
        note=note,
        layer=layer,
        selector=selector,
        source=source,
    )


def _int(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="int32", return_type_id=TYPE_INT, pushes=1, **kw)


def _float(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="float32", return_type_id=TYPE_FLOAT, pushes=1, **kw)


def _slice(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="slice/span", return_type_id=TYPE_SLICE, pushes=1, **kw)


def _string(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="span/string", return_type_id=TYPE_STRING, pushes=1, **kw)


def _void(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="void", return_type_id=None, pushes=0, **kw)


def _unknown_value(name: str, **kw: Any) -> NativeCallSpec:
    return _spec(name, return_type="unknown", return_type_id=None, pushes=1, **kw)


ENGINE_NATIVE_IMPORTS: dict[str, NativeCallSpec] = {
    "CreateObj": _int(
        "native.CreateObj",
        arity=2,
        arg_types=("span/string", "int32"),
        side_effects=("engine_object", "process", "runtime_call"),
        confidence="engine-native",
        note="Function-table import has no MBC provider; engine object factory, observed as handle-returning in the corpus.",
        layer="engine_import",
    ),
    "CreateObjWait": _int(
        "native.CreateObjWait",
        arity=2,
        arg_types=("span/string", "int32"),
        side_effects=("engine_object", "process", "runtime_call"),
        confidence="engine-native",
        note="Engine object factory variant; observed with the same two-argument shape as CreateObj and a returned handle.",
        layer="engine_import",
    ),
    "DestroyObj": _void(
        "native.DestroyObj",
        arity=None,
        side_effects=("engine_object", "runtime_call"),
        confidence="engine-native",
        note="Engine object destroy/release import; no neighbouring MBC provider exists.",
        layer="engine_import",
    ),
    "CenterObj": _void(
        "native.CenterObj",
        arity=6,
        arg_types=("unknown", "unknown", "unknown", "unknown", "unknown", "unknown"),
        side_effects=("engine_object", "ui", "runtime_call"),
        confidence="engine-native",
        note="Engine object centering/placement helper; observed corpus calls use six arguments and no returned value.",
        layer="engine_import",
    ),
    "SetTrig": _void(
        "native.SetTrig",
        arity=1,
        arg_types=("int32",),
        side_effects=("engine_object", "process", "runtime_call"),
        confidence="engine-native",
        note="Engine trigger/state setter; observed corpus calls use one argument.",
        layer="engine_import",
    ),
    "CountAnim": _int(
        "native.CountAnim",
        arity=0,
        side_effects=("engine_object", "runtime_call"),
        confidence="engine-native",
        note="Animation-count query; observed as zero-argument value-returning import.",
        layer="engine_import",
    ),
    "SpeedObj": _float(
        "native.SpeedObj",
        arity=0,
        side_effects=("engine_object", "runtime_call"),
        confidence="engine-native",
        note="Object speed query; observed as zero-argument value-returning import.",
        layer="engine_import",
    ),
    "DirectOfObj": _int(
        "native.DirectOfObj",
        arity=0,
        side_effects=("engine_object", "runtime_call"),
        confidence="engine-native",
        note="Object direction query; observed as zero-argument value-returning import.",
        layer="engine_import",
    ),
    "InitAI": _void(
        "native.InitAI",
        arity=None,
        side_effects=("ai", "engine_object", "process", "runtime_call"),
        confidence="engine-native",
        note="AI initialization import. Arity is left call-site-driven because the native body is outside MBC bytecode.",
        layer="engine_import",
    ),
    "InvertAI": _void(
        "native.InvertAI",
        arity=None,
        side_effects=("ai", "engine_object", "runtime_call"),
        confidence="engine-native",
        note="AI state modifier import; native body is not present in the MBC corpus.",
        layer="engine_import",
    ),
    "SetRespRadius": _void(
        "native.SetRespRadius",
        arity=None,
        side_effects=("ai", "engine_object", "runtime_call"),
        confidence="engine-native",
        note="AI/respawn radius setter import; native body is not present in the MBC corpus.",
        layer="engine_import",
    ),
    "QueryShowWebShop": _int(
        "native.QueryShowWebShop",
        arity=None,
        side_effects=("ui", "runtime_call"),
        confidence="engine-native",
        note="UI/shop native query; no neighbouring MBC provider exists.",
        layer="engine_import",
    ),
    "WaitForAsk": _int(
        "native.WaitForAsk",
        arity=None,
        side_effects=("ui", "process", "runtime_call"),
        confidence="engine-native",
        note="Modal/question wait helper; native body is outside MBC bytecode.",
        layer="engine_import",
    ),
    "GetPictsPointer": _string(
        "native.GetPictsPointer",
        arity=None,
        side_effects=("ui", "resources", "runtime_call"),
        confidence="engine-native",
        note="Picture/resource pointer helper; no neighbouring MBC provider exists.",
        layer="engine_import",
    ),
    "PutMoney": _void(
        "native.PutMoney",
        arity=None,
        side_effects=("ui", "process", "runtime_call"),
        confidence="engine-native",
        note="Money/transaction UI helper; native body is not present in the MBC corpus.",
        layer="engine_import",
    ),
    "GetMoney": _int(
        "native.GetMoney",
        arity=None,
        side_effects=("ui", "process", "runtime_call"),
        confidence="engine-native-unobserved",
        note="Money/transaction query helper. It appears as the native counterpart of PutMoney in cs_knot/cs_table/town_table function tables; no MBC provider and no direct corpus call-site were observed.",
        layer="engine_import",
    ),
}

# Selectors recovered from the nested switch under builtin subopcode 0x67 in
# sub_477500.  For opaque cases, the push/no-push classification follows visible
# sub_47ACD0 paths in the C decompilation and is intentionally conservative.
_FFSYS_CASES = {
    0: ("g_ground", True), 1: ("set_object_flag_274", False), 2: ("set_global_float_5602D0", False),
    3: ("set_object_flag_276", False), 4: ("set_current_object", False), 5: ("set_game_time", False),
    6: ("get_game_time", True), 7: ("set_runtime_value_4a8b62c", False), 8: ("reset_runtime_value_4a8b62c", False),
    9: ("get_time_field", True), 10: ("select_object_context", False), 11: ("get_or_set_runtime_var_4a8b624", True),
    12: ("get_error_message", False), 14: ("query_runtime_ready", True), 18: ("set_object_motion_state", False),
    19: ("find_first", True), 20: ("find_next", True), 21: ("find_close", False), 22: ("reset_global_counter", False),
    23: ("consume_one_arg", False), 24: ("get_global_6fd15c", True), 25: ("get_global_6fd160", True),
    26: ("get_global_565764", True), 27: ("get_global_6a2fac", True), 28: ("get_global_567710", True),
    29: ("set_visible_or_runtime_state", True), 32: ("cursor_or_runtime_point", False), 33: ("get_mouse_xy", False),
    34: ("get_global_4a2ab30", True), 35: ("sel_035", False), 36: ("get_current_dir", False),
    37: ("query_runtime_flag", True), 38: ("set_runtime_flag", False), 39: ("clear_runtime_flag", False),
    41: ("sel_041", False), 43: ("object_query_or_lookup", True), 44: ("object_state_guard", True),
    45: ("link_on", False), 46: ("link_off_or_update", False), 47: ("runtime_link_state", False),
    53: ("get_absolute_time", True), 57: ("time_api", True), 58: ("set_global_6fd168", False),
    59: ("set_global_6fd164", False), 60: ("get_global_6fd164", True), 61: ("unload_main_or_log", False),
    63: ("sel_063", False), 64: ("play_music", False), 65: ("stop_music_or_sound", False),
    66: ("gz_pack", True), 67: ("gz_unpack", True), 68: ("map_load", False), 69: ("map_save", False),
    70: ("map_pic_size", False), 71: ("sel_071", False), 72: ("spawn_or_runtime_create", False),
    73: ("runtime_status_or_wait", True), 74: ("sel_074", False), 75: ("set_runtime_timer_300ms", False),
    77: ("sel_077", False), 78: ("runtime_object_action", False), 79: ("sel_079", False),
    80: ("object_transform_query", True), 81: ("object_name_or_state_query", True), 82: ("sel_082", False),
    83: ("g_plant", True), 84: ("plant_or_map_query", True), 85: ("map_object_write", True),
    86: ("map_object_query_pair", True), 87: ("map_object_query", True), 88: ("map_global_query", True),
    89: ("map_or_object_update", False), 92: ("load_mbc_or_script_path", False), 93: ("sel_093", False),
    94: ("sel_094", False), 95: ("object_collision_or_region_test", True), 96: ("object_query_bool", True),
    98: ("runtime_pattern_set_short", False), 99: ("runtime_pattern_patch", False), 100: ("runtime_vector_query", True),
    103: ("write_version", False), 106: ("runtime_string_command", False), 107: ("set_runtime_flag_4a9a824", False),
    109: ("process_bind_strings", False), 110: ("process_step_move_or_query", False), 111: ("process_state_reset", False), 112: ("process_state_query", True),
    113: ("sel_113", False), 114: ("get_root_drive", True), 115: ("path_exists", True), 116: ("resource_load_by_name", True),
    119: ("sel_119", False), 120: ("set_runtime_handle_state", False), 121: ("runtime_pointer_query", True),
    122: ("u64_to_string", True), 123: ("u64_add_i32_checked", True), 124: ("u64_sub_i32_checked", True), 125: ("u64_compare_i32", True),
    126: ("u64_mul_float_to_i32", True), 127: ("u64_mul_i32_checked", True), 128: ("u64_compare", True), 131: ("format_date_time", True),
    132: ("shell_execute", True), 133: ("push_zero_reserved_133", True), 134: ("push_zero_reserved_134", True), 136: ("path_is_directory", True),
    138: ("push_zero_reserved_138", True), 140: ("chat_set_user_name", False), 150: ("connection_lost_notification", False),
    152: ("runtime_ui_flush", False), 206: ("world_map_bitmap", True), 207: ("world_map_state", True), 212: ("get_global_5610B8", True),
    213: ("runtime_set_4FA240", False), 214: ("runtime_set_4FA2D0", False), 215: ("process_find_state_by_name", True), 216: ("process_compare_state_strings", True),
    218: ("runtime_flag_save_and_clear", False), 219: ("lookup_runtime_table_entry", True), 220: ("lookup_runtime_table_global", True), 221: ("runtime_flag_restore", False),
    224: ("object_set_extra_vec3", False), 225: ("write_localtime_struct", False), 226: ("get_tick_count", True), 227: ("normalize_vec3_in_place", True),
    228: ("query_login_cache", True), 229: ("slice_copy_runtime", True), 508: ("script_function_find_next", True), 509: ("set_global_55F760", False),
    510: ("get_global_55F760", True),
}

# Human-readable notes from the IDA ASM slices and matching C decompilation.
# They are deliberately descriptive rather than pretending every helper has a
# final game-facing name.  The important part for the VM model is whether the
# selector pushes a value and what broad memory/process/UI effect it has.
_FFSYS_NOTES: dict[int, str] = {
    75: "pops enable flag, calls sub_46AE00 and manages dword_55F764 timer/update handle",
    98: "pops six scalar fields and updates runtime pattern/state table through sub_45C230",
    99: "pops extended scalar pattern fields and updates runtime pattern/state table",
    103: "G_VERSION: pops destination pointer and writes version value 3; no stack push",
    106: "pops string data offset and scalar mode, then dispatches sub_46DED0",
    107: "pops scalar and stores boolean runtime flag dword_4A9A824",
    109: "pops two string offsets and binds process/runtime strings through sub_45E070",
    110: "process step/query helper; writes output data slots and may reset process state",
    111: "process state reset via sub_45E000",
    112: "process state query via sub_45EAC0; pushes int32",
    44: "corpus/stack-verified: selector result is immediately stored; must push int32/status",
    85: "corpus/stack-verified: map/object write returns status consumed by comparison",
    115: "path/existence check; shared labels push 0 or 1",
    116: "resource lookup/load by name; pushes handle or 0",
    123: "signed 64-bit add with overflow check; mutates qword and pushes status",
    124: "signed 64-bit subtract with overflow check; mutates qword and pushes status",
    125: "signed 64-bit compare against int32; pushes comparison boolean/status",
    126: "multiplies qword low part by float and pushes int32 result",
    127: "signed 64-bit multiply by int32; mutates qword and pushes result",
    128: "signed 64-bit compare between two qwords; pushes comparison boolean/status",
    133: "reserved/default case that falls through to LABEL_1851 and pushes 0",
    134: "reserved/default case that falls through to LABEL_1851 and pushes 0",
    136: "directory test using findfirst64i32; pushes -1/0/1 style status",
    138: "reserved/default case that falls through to LABEL_1851 and pushes 0",
    152: "UI/runtime flush/update through sub_4FC870; no stack push",
    212: "pushes global dword_5610B8",
    213: "passes scalar to sub_4FA240; no stack push",
    214: "passes scalar to sub_4FA2D0; no stack push",
    215: "process named-state lookup through sub_45E2D0; pushes handle/status",
    216: "process string-pair lookup/compare through sub_45E3A0; pushes handle/status",
    218: "saves runtime flag byte_5610A4 into byte_5610A5 and clears it",
    219: "runtime table lookup through sub_4BE140; pushes handle/status",
    220: "global table lookup through sub_4BE1A0; pushes handle/status",
    221: "restores byte_5610A4 from byte_5610A5",
    224: "object handle plus three floats; writes object fields +0x288/+0x28C/+0x290",
    225: "writes localtime64 structure into pointer destination",
    226: "GetTickCount wrapper; pushes tick count",
    227: "normalizes vector stored in data section and writes length to destination; pushes status",
    228: "copies cached login/session string when available and pushes readiness/status",
    229: "copies runtime slice/string via sub_459690 and falls through to push 0",
    508: "enumerates script function-table names and pushes next index or -1",
    509: "stores scalar into dword_55F760",
    510: "pushes dword_55F760",
}


def _ffsys_spec(selector: int, short_name: str, returns: bool) -> NativeCallSpec:
    name = f"ffsys.{short_name}"
    side_effects = ("memory", "process", "ui", "runtime_call")
    extra = _FFSYS_NOTES.get(selector)
    note = f"builtin 0x67 selector {selector}; recovered from sub_477500 nested ffsys switch"
    if extra:
        note = f"{note}. {extra}"
    if returns:
        return _int(name, side_effects=side_effects, confidence="ffsys-selector", note=note, layer="ffsys", selector=selector, source="sub_477500/case_0x67")
    return _void(name, side_effects=side_effects, confidence="ffsys-selector", note=note, layer="ffsys", selector=selector, source="sub_477500/case_0x67")


FFSYS_SELECTORS: dict[int, NativeCallSpec] = {
    selector: _ffsys_spec(selector, short_name, returns)
    for selector, (short_name, returns) in _FFSYS_CASES.items()
}

CONFIG_API_SELECTORS: dict[int, NativeCallSpec] = {
    # sub_486A60 is reached from sub_477500 case 117 (builtin 0x75).  It first
    # consumes the selector with sub_47AAF0, then switches on selector - 13.
    # Cases not listed here fall to the default branch and return without push.
    13: _int(
        "native_config.set_typed_value",
        arity=None,
        arg_types=("span/string", "unknown"),
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 13: key string plus typed value; serializes int/float/string/bit-array forms and pushes status 0",
        layer="native_config",
        selector=13,
        source="sub_486A60/case_13",
    ),
    14: _int(
        "native_config.get_typed_value",
        arity=None,
        arg_types=("pointer", "pointer", "int32"),
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 14: key pointer, destination pointer and optional mode; writes int/float/string/bit-array destination and pushes 0 on success or -1 on failure",
        layer="native_config",
        selector=14,
        source="sub_486A60/case_14",
    ),
    15: _int(
        "native_config.find_key",
        arity=None,
        arg_types=("pointer",),
        side_effects=("config", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 15: key lookup through sub_441990; pushes 0 on success or -1 on miss/failure",
        layer="native_config",
        selector=15,
        source="sub_486A60/case_15",
    ),
    16: _int(
        "native_config.apply_mode_0",
        arity=0,
        side_effects=("config", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 16: calls sub_441AC0 with mode 0 and pushes 0/-1 status",
        layer="native_config",
        selector=16,
        source="sub_486A60/case_16",
    ),
    17: _void(
        "native_config.load_source_text",
        arity=None,
        arg_types=("pointer",),
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 17: source pointer; calls sub_441700 and returns without pushing",
        layer="native_config",
        selector=17,
        source="sub_486A60/case_17",
    ),
    30: _int(
        "native_config.save_to_path",
        arity=None,
        arg_types=("pointer",),
        side_effects=("config", "file", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 30: path pointer; calls sub_441730 and pushes status 0",
        layer="native_config",
        selector=30,
        source="sub_486A60/case_30",
    ),
    54: _int(
        "native_config.consume_current_script_blob",
        arity=0,
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 54: consumes [current_context+0xB0] blob into config storage; pushes 0, or -1 when absent",
        layer="native_config",
        selector=54,
        source="sub_486A60/case_54",
    ),
    55: _int(
        "native_config.apply_mode_1",
        arity=0,
        side_effects=("config", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 55: calls sub_441AC0 with mode 1 and pushes 0/-1 status",
        layer="native_config",
        selector=55,
        source="sub_486A60/case_55",
    ),
    56: _int(
        "native_config.read_blob_into",
        arity=None,
        arg_types=("pointer", "int32"),
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 56: destination pointer plus size/count; calls sub_441B70 and pushes its int result",
        layer="native_config",
        selector=56,
        source="sub_486A60/case_56",
    ),
    57: _int(
        "native_config.get_size_or_state",
        arity=0,
        side_effects=("config", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 57: pushes sub_4415F0 size/state result",
        layer="native_config",
        selector=57,
        source="sub_486A60/case_57",
    ),
    62: _void(
        "native_config.set_source_text",
        arity=None,
        arg_types=("pointer",),
        side_effects=("config", "memory", "runtime_call"),
        confidence="asm-verified",
        note="sub_486A60 selector 62: source text pointer; calls sub_441570 and returns without pushing",
        layer="native_config",
        selector=62,
        source="sub_486A60/case_62",
    ),
}

def engine_native_import(name: str) -> NativeCallSpec | None:
    return ENGINE_NATIVE_IMPORTS.get(name)


def native_api_spec(name: str) -> NativeCallSpec | None:
    """Backwards-compatible alias for engine-level function-table imports."""
    return engine_native_import(name)


def is_engine_native_import(name: str) -> bool:
    return name in ENGINE_NATIVE_IMPORTS


def selector_from_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 0)
        except ValueError:
            return None
    return None


def selector_from_slot(slot: Any) -> int | None:
    value = getattr(slot, "value", None)
    selector = selector_from_value(value)
    if selector is not None:
        return selector
    expr = getattr(slot, "expr", None)
    if isinstance(expr, str):
        text = expr.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text, 10)
        if re.fullmatch(r"0x[0-9a-fA-F]+", text):
            return int(text, 16)
    return None


def selector_from_args(args: list[Any] | tuple[Any, ...]) -> int | None:
    if not args:
        return None
    return selector_from_slot(args[0])


def builtin_api_spec(subopcode: int, selector: int) -> NativeCallSpec | None:
    if subopcode == 0x67:
        spec = FFSYS_SELECTORS.get(selector)
        if spec is not None:
            return spec
        return _void(
            f"ffsys.noop_selector_{selector:03d}",
            side_effects=(),
            confidence="ffsys-selector-default-no-push",
            note=f"builtin 0x67 selector {selector}; sub_477500 nested switch default path restores state and returns without pushing or side effects",
            layer="ffsys",
            selector=selector,
            source="sub_477500/case_0x67/default",
        )
    if subopcode == 0x75:
        spec = CONFIG_API_SELECTORS.get(selector)
        if spec is not None:
            return spec
        return _void(
            f"native_config.sel_{selector:03d}",
            side_effects=("config", "memory", "runtime_call"),
            confidence="native-config-selector-default",
            note=f"builtin 0x75 selector {selector}; sub_486A60 default branch returns without pushing",
            layer="native_config",
            selector=selector,
            source="sub_486A60/default",
        )
    return None


def specialize_builtin_api(subopcode: int, args: list[Any]) -> tuple[NativeCallSpec, list[Any]] | None:
    if subopcode not in (0x67, 0x75):
        return None
    selector = selector_from_args(args)
    if selector is None:
        return None
    spec = builtin_api_spec(subopcode, selector)
    if spec is None:
        return None
    return spec, list(args[1:])

@dataclass(frozen=True)
class CallEffect:
    name: str
    arity: int | None = None
    arg_types: tuple[str, ...] = ()
    return_type: str = "void"
    return_type_id: int | None = None
    consumes: int | None = None
    pushes: int = 0
    side_effects: tuple[str, ...] = ()
    confidence: str = "inferred"
    statement: bool = False
    note: str = ""

    @property
    def returns_value(self) -> bool:
        return self.pushes > 0 and self.return_type != "void"

    @property
    def is_pure(self) -> bool:
        return self.returns_value and not self.side_effects and not self.statement

    def with_call_arity(self, argc: int | None) -> "CallEffect":
        if argc is None:
            return self
        return replace(self, arity=argc, consumes=argc)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arity": self.arity,
            "arg_types": list(self.arg_types),
            "return_type": self.return_type,
            "return_type_id": self.return_type_id,
            "consumes": self.consumes,
            "pushes": self.pushes,
            "side_effects": list(self.side_effects),
            "confidence": self.confidence,
            "statement": self.statement,
            "note": self.note,
        }


_FLOAT_UNARY = {"sin", "cos", "sqrt_abs_float", "exp", "abs_float", "logf_math"}
_INT_UNARY = {"abs_int", "identity_int", "bit_not"}
_FLOAT_RETURNS = {
    "sin", "cos", "sqrt_abs_float", "exp", "rand_float", "object_get_x", "object_get_y", "object_get_z",
    "object_get_vec14_x", "object_get_vec14_y", "object_get_vec14_z", "get_float_field_0x294",
}
_INT_RETURNS_BY_NAME = {
    "push_vm_tick", "push_runtime_handle", "push_runtime_flag_byte", "ffprc_load", "ffprc_link", "ffprc_state",
    "last_process_result", "arg_count", "current_process_state", "push_zero", "push_zero_alias", "ffprc_id",
    "send_to_process_id", "send_to_process_zero", "send_message_marshaled", "receive_message_marshaled", "window_api",
    "ffsys_api", "text_api", "native_config_api", "effect_attach", "item_inventory_api", "entity_ref_api", "resource_handle_api", "sscanf", "prc_name", "file_remove", "reserved_noop_7a",
    "push_context_id_or_zero", "lookup_process_by_name", "push_current_flags_mask_4", "strlen_checked", "strcmp",
    "current_process_id", "file_create", "file_open", "file_close", "file_read", "file_read_line", "file_write", "identity_int", "object_create",
    "object_get_or_set_flag_0x278", "push_runtime_constant_pair", "find_effect_id", "assoc_array_get", "file_seek",
    "file_length", "file_stat_time_field", "file_lookup_476310", "stricmp", "strncmp", "strnicmp", "current_sender_id",
    "bit_and", "bit_or", "bit_xor", "bit_not", "shift_left", "shift_right", "bit_clear", "bit_set", "bit_test",
    "memcmp", "binary_search_i32", "buffer_hash_or_checksum", "push_runtime_slot", "distance_or_distance_sq",
    "angle_delta", "push_minus_one", "pack_rgb24", "object_create", "sprite_create_or_update",
    "typed_load_width_1", "typed_load_width_2", "typed_load_width_3", "typed_load_width_4",
}
_SLICE_RETURNS_BY_NAME = {
    "alloc_span", "strstr", "strchr", "stristr", "push_static_word_span", "parse_api",
    "typed_store_width_1", "typed_store_width_2", "typed_store_width_3", "typed_store_width_4", "process_translate_ptr", "span_write_float", "span_write_cstring", "ptr_store_i32_from_ptr", "ptr_copy_cstring",
}
_VOID_BY_NAME = {
    "debug_print_float", "debug_print_float_alias", "print_string_or_exit",
    "ffprc_unload",
    "strcpy_checked", "strcat_checked", "log_event_dispatch",
    "object_set_pos_xyz", "object_add_pos_xyz", "view_set_pos_xyz", "view_set_z", "object_set_vec14_xyz",
    "global_vector_set", "object_delete_type0", "object_release_type4", "text_color",
    "process_memcpy", "memcpy", "memset", "thisname", "ffmempcpy_alt", "copy_effect_name_by_id",
    "dmalloc_free", "dmalloc", "assoc_array_set", "file_lock",
    "snprintf", "sprintf", "file_rename", "file_truncate", "file_set_time",
    "object_set_flag_0x141", "object_get_norm_vec3", "object_get_position_vec3", "object_get_abg_vec3",
    "chat_utility_api", "editor_get_click_point", "raw_arg_read", "external_runtime_update_473730", "reserved_noop_82", "reserved_noop_85",
}

_EXPLICIT_ARITY: dict[str, int] = {
    "sin": 1,
    "cos": 1,
    "sqrt_abs_float": 1,
    "exp": 1,
    "abs_float": 1,
    "abs_int": 1,
    "identity_int": 1,
    "identity_float": 1,
    "atan2": 2,
    "pack_rgb24": 3,
    "push_vm_tick": 0,
    "last_process_result": 0,
    "arg_count": 0,
    "current_process_state": 0,
    "current_process_id": 0,
    "current_sender_id": 0,
    "push_zero": 0,
    "push_zero_alias": 0,
    "push_minus_one": 0,
    "rand_float": 0,
    "push_static_word_span": 0,
}


def _side_effects_for(name: str, semantic: str, returns: bool) -> tuple[str, ...]:
    s = f"{name} {semantic}".lower()
    effects: list[str] = []
    if any(token in s for token in ("write", "copy", "memcpy", "memset", "strcpy", "strcat", "alloc", "free", "store", "destination", "buffer")):
        effects.append("memory")
    if any(token in s for token in ("process", "prc", "send", "receive", "message", "interpreter", "runtime")):
        effects.append("process")
    if any(token in s for token in ("window", "ui", "object", "view", "sprite", "text", "chat", "editor", "item", "effect")):
        effects.append("ui")
    if "file" in s or any(token in s for token in ("open", "close", "read", "write", "rename", "remove", "seek", "truncate")):
        effects.append("file")
    if "log" in s or "debug" in s or "print" in s:
        effects.append("diagnostic")
    if not effects and not returns:
        effects.append("runtime")
    return tuple(dict.fromkeys(effects))


def _return_for(name: str, semantic: str) -> tuple[str, int | None, int]:
    s = semantic.lower()
    if name in _VOID_BY_NAME:
        return "void", None, 0
    if name in _FLOAT_RETURNS or ("pushes" in s and "float" in s and "string" not in s):
        return "float32", TYPE_FLOAT, 1
    if name in _SLICE_RETURNS_BY_NAME or ("pushes" in s and ("slice descriptor" in s or "span" in s) and "writes" not in s):
        return "slice", TYPE_SLICE, 1
    if name in _INT_RETURNS_BY_NAME:
        return "int32", TYPE_INT, 1
    if "pushes" in s or name.startswith("push_"):
        if "string" in s and "length" not in s:
            return "span/string", TYPE_STRING, 1
        return "int32", TYPE_INT, 1
    return "void", None, 0


def _arg_types_for(name: str, arity: int | None) -> tuple[str, ...]:
    if arity is None:
        return ()
    if name in {"sin", "cos", "sqrt_abs_float", "exp", "abs_float"}:
        return tuple("float32" for _ in range(arity))
    if name == "atan2":
        return ("float32", "float32")
    if name == "identity_float":
        return ("float32",)
    return tuple("unknown" for _ in range(arity))


# A value-returning call can be rendered as an expression when its value is
# consumed, but the call still has to survive when the VM later discards the
# returned slot.  This list/token model is intentionally conservative for
# operations that mutate memory, process state, files, UI, objects or config.
_VALUE_CALL_STATEMENT_NAMES = {
    "send_to_process_id", "send_to_process_zero", "send_message_marshaled", "receive_message_marshaled",
    "effect_attach", "object_get_or_set_flag_0x278", "object_create", "sprite_create_or_update",
    "file_create", "file_open", "file_close", "file_read", "file_read_line", "file_write", "file_seek", "file_remove",
    "sscanf", "typed_store_width_1", "typed_store_width_2", "typed_store_width_3", "typed_store_width_4",
    "process_translate_ptr", "span_write_float", "span_write_cstring", "ptr_store_i32_from_ptr", "ptr_copy_cstring",
    "alloc_span", "text_api", "window_api", "native_config_api", "item_inventory_api", "entity_ref_api",
    "resource_handle_api",
}

_VALUE_CALL_STATEMENT_TOKENS = (
    "set", "write", "store", "send", "receive", "create", "delete", "remove", "open", "close",
    "load", "save", "copy", "memcpy", "memset", "alloc", "free", "attach", "update", "patch",
    "flush", "rename", "truncate", "translate", "wait", "lock", "unlock", "consume", "apply",
)

_READONLY_VALUE_CALL_NAMES = {
    "current_process_id", "current_process_state", "current_sender_id", "last_process_result", "arg_count",
    "ffprc_state", "object_state_query", "object_get_x", "object_get_y", "object_get_z",
    "object_get_vec14_x", "object_get_vec14_y", "object_get_vec14_z", "get_float_field_0x294",
    "strlen_checked", "strcmp", "stricmp", "strncmp", "strnicmp", "memcmp",
    "binary_search_i32", "buffer_hash_or_checksum", "distance_or_distance_sq", "angle_delta",
    "file_length", "file_stat_time_field", "file_lookup_476310", "find_effect_id", "assoc_array_get",
    "CountAnim", "native.CountAnim", "native.SpeedObj", "native.DirectOfObj",
    "native.QueryShowWebShop", "native_config.get_size_or_state",
}


def _value_call_needs_statement(name: str, semantic: str, side_effects: tuple[str, ...]) -> bool:
    if name in _READONLY_VALUE_CALL_NAMES:
        return False
    if name in _VALUE_CALL_STATEMENT_NAMES:
        return True
    text = f"{name} {semantic}".lower()
    if any(token in text for token in _VALUE_CALL_STATEMENT_TOKENS):
        return True
    # Unknown runtime/native calls are safer to preserve when their return value
    # is discarded; losing a call is worse than an occasional noisy statement.
    if any(effect in side_effects for effect in ("memory", "file", "ui", "config", "engine_object", "ai", "native_or_unresolved_call")):
        return True
    return False


def _statement_for_call(name: str, semantic: str, returns: bool, side_effects: tuple[str, ...]) -> bool:
    if not side_effects:
        return False
    if not returns:
        return True
    return _value_call_needs_statement(name, semantic, side_effects)


def builtin_effect(subopcode: int, *, argc: int | None = None) -> CallEffect:
    builtin = BUILTINS[subopcode]
    name = builtin.mnemonic
    semantic = builtin.semantic
    return_type, return_type_id, pushes = _return_for(name, semantic)
    arity = argc if argc is not None else _EXPLICIT_ARITY.get(name)
    returns = pushes > 0 and return_type != "void"
    side_effects = _side_effects_for(name, semantic, returns)
    # `statement` means “must be preserved if the returned VM slot is later
    # discarded”.  Value-returning queries may be inlined or dropped; mutating
    # value-returning calls must not disappear.
    statement = _statement_for_call(name, semantic, returns, side_effects)
    effect = CallEffect(
        name=name,
        arity=arity,
        arg_types=_arg_types_for(name, arity),
        return_type=return_type,
        return_type_id=return_type_id,
        consumes=arity,
        pushes=pushes,
        side_effects=side_effects,
        confidence=builtin.confidence,
        statement=statement,
        note=semantic,
    )
    return effect.with_call_arity(argc)



def effect_from_native_spec(spec: NativeCallSpec, *, argc: int | None = None, name: str | None = None) -> CallEffect:
    arity = argc if argc is not None else spec.arity
    if arity is not None and spec.arg_types and len(spec.arg_types) == arity:
        arg_types = spec.arg_types
    elif arity is not None:
        arg_types = tuple("unknown" for _ in range(max(0, arity)))
    else:
        arg_types = spec.arg_types
    return CallEffect(
        name=name or spec.name,
        arity=arity,
        arg_types=arg_types,
        return_type=spec.return_type,
        return_type_id=spec.return_type_id,
        consumes=arity,
        pushes=spec.push_count,
        side_effects=spec.side_effects,
        confidence=spec.confidence,
        # Value-returning native calls are still rendered as expressions when
        # consumed, but mutating ones must be emitted if their result is dropped.
        statement=_statement_for_call(name or spec.name, spec.note, spec.returns_value, spec.side_effects),
        note=spec.note,
    )


def native_import_effect(name: str, *, argc: int | None = None) -> CallEffect | None:
    spec = engine_native_import(name)
    if spec is None and name.startswith("native."):
        spec = engine_native_import(name.split(".", 1)[1])
    if spec is None:
        return None
    return effect_from_native_spec(spec, argc=argc)


def specialize_builtin_call(subopcode: int, args: list[object], base_effect: CallEffect) -> tuple[str, list[object], CallEffect]:
    specialized = specialize_builtin_api(subopcode, args)
    if specialized is None:
        return base_effect.name, args, base_effect
    spec, call_args = specialized
    effect = effect_from_native_spec(spec, argc=len(call_args), name=spec.name)
    return spec.name, call_args, effect

def function_effect(name: str, signature: FunctionSignature, *, argc: int | None = None, linked: bool = False) -> CallEffect:
    arity = argc if argc is not None else (signature.arity if signature.source != "unknown" else None)
    qnote = "runtime-linked MBC function" if linked else "MBC function"
    return CallEffect(
        name=name,
        arity=arity,
        arg_types=tuple(arg.type_name for arg in signature.args),
        return_type=signature.return_type,
        return_type_id=signature.return_type_id,
        consumes=arity,
        pushes=1 if signature.return_type == "unknown" else (0 if signature.return_type == "void" else 1),
        side_effects=("runtime_call",),
        confidence=signature.source,
        statement=True,
        note=qnote,
    )


def unresolved_import_effect(name: str, *, argc: int | None = None) -> CallEffect:
    native = native_import_effect(name, argc=argc)
    if native is not None:
        return native
    return CallEffect(
        name=name,
        arity=argc,
        consumes=argc,
        pushes=1,
        return_type="unknown",
        side_effects=("native_or_unresolved_call",),
        confidence="unresolved",
        statement=True,
        note="No matching MBC provider or interpreter-native API entry was found.",
    )
