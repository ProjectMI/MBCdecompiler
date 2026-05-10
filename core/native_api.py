from __future__ import annotations

"""Interpreter-native API layer for calls that are not backed by .mbc code.

There are two different runtime-linking surfaces in the Sphere client:

* function-table imports with ``program_index == -1`` and 0x67 stubs.  The VM
  first tries to patch these against neighbouring MBC scripts (see
  ``sub_4784C0`` in the C decompilation).  Names that still have no MBC provider
  are engine-native functions such as ``CreateObj`` and ``DestroyObj``.
* builtin subopcode 0x67/0x75 dispatchers.  Their first VM argument is a native
  selector: 0x67 enters the large ``ffsys`` switch in ``sub_477500``; 0x75 jumps
  into the config/native switch recovered as ``sub_486A60``.

This module keeps those native surfaces explicit so the decompiler does not mark
known interpreter APIs as unresolved imports or opaque ``external_native_*``
helpers.
"""

from dataclasses import dataclass
import re
from typing import Any

from .vm_stack import TYPE_FLOAT, TYPE_INT, TYPE_SLICE, TYPE_STRING


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
    41: ("sel_041", False), 43: ("object_query_or_lookup", True), 44: ("object_state_guard", False),
    45: ("link_on", False), 46: ("link_off_or_update", False), 47: ("runtime_link_state", False),
    53: ("get_absolute_time", True), 57: ("time_api", True), 58: ("set_global_6fd168", False),
    59: ("set_global_6fd164", False), 60: ("get_global_6fd164", True), 61: ("unload_main_or_log", False),
    63: ("sel_063", False), 64: ("play_music", False), 65: ("stop_music_or_sound", False),
    66: ("gz_pack", True), 67: ("gz_unpack", True), 68: ("map_load", False), 69: ("map_save", False),
    70: ("map_pic_size", False), 71: ("sel_071", False), 72: ("spawn_or_runtime_create", False),
    73: ("runtime_status_or_wait", True), 74: ("sel_074", False), 75: ("set_runtime_timer_300ms", False),
    77: ("sel_077", False), 78: ("runtime_object_action", False), 79: ("sel_079", False),
    80: ("object_transform_query", True), 81: ("object_name_or_state_query", True), 82: ("sel_082", False),
    83: ("g_plant", True), 84: ("plant_or_map_query", True), 85: ("map_object_write", False),
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
    13: _int("native_config.set_value", side_effects=("config", "memory", "runtime_call"), confidence="recovered", note="sub_486A60 selector 13: set/replace key value and push status 0", layer="native_config", selector=13, source="sub_486A60"),
    14: _int("native_config.get_value", side_effects=("config", "memory", "runtime_call"), confidence="recovered", note="sub_486A60 selector 14: cfg_get, writes typed output and pushes status", layer="native_config", selector=14, source="sub_486A60"),
    15: _int("native_config.find_key", side_effects=("config", "runtime_call"), confidence="recovered", note="sub_486A60 selector 15: key lookup, pushes status/result", layer="native_config", selector=15, source="sub_486A60"),
    16: _int("native_config.reset_cursor", side_effects=("config", "runtime_call"), confidence="recovered", note="sub_486A60 selector 16: cursor/state reset, pushes status", layer="native_config", selector=16, source="sub_486A60"),
    17: _void("native_config.load_file", side_effects=("config", "file", "runtime_call"), confidence="recovered", note="sub_486A60 selector 17: load config text/file", layer="native_config", selector=17, source="sub_486A60"),
    30: _int("native_config.save_file", side_effects=("config", "file", "runtime_call"), confidence="recovered", note="sub_486A60 selector 30: save config file and push status 0", layer="native_config", selector=30, source="sub_486A60"),
    54: _int("native_config.load_current_script_blob", side_effects=("config", "memory", "runtime_call"), confidence="recovered", note="sub_486A60 selector 54: import current script config blob and push status", layer="native_config", selector=54, source="sub_486A60"),
    55: _int("native_config.next_entry", side_effects=("config", "runtime_call"), confidence="recovered", note="sub_486A60 selector 55: advance/query config cursor, pushes status", layer="native_config", selector=55, source="sub_486A60"),
    56: _int("native_config.read_blob", side_effects=("config", "memory", "runtime_call"), confidence="recovered", note="sub_486A60 selector 56: read config blob into destination and push result", layer="native_config", selector=56, source="sub_486A60"),
    57: _int("native_config.get_size_or_state", side_effects=("config", "runtime_call"), confidence="recovered", note="sub_486A60 selector 57: pushes config size/state", layer="native_config", selector=57, source="sub_486A60"),
    62: _void("native_config.set_source_text", side_effects=("config", "memory", "runtime_call"), confidence="recovered", note="sub_486A60 selector 62: set/parse source text", layer="native_config", selector=62, source="sub_486A60"),
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
        return _unknown_value(
            f"ffsys.sel_{selector:03d}",
            side_effects=("memory", "process", "ui", "runtime_call"),
            confidence="ffsys-selector-unknown",
            note=f"builtin 0x67 selector {selector}; selector observed in bytecode but not named in the current registry",
            layer="ffsys",
            selector=selector,
            source="sub_477500/case_0x67",
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
