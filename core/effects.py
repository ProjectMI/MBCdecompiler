from __future__ import annotations

"""Stack and side-effect model for function and builtin calls.

The bytecode uses the same call surface for pure expressions, statement-like
runtime operations and native/builtin helpers.  A decompiler cannot decide
whether ``foo(...)`` should be pushed as an rvalue or emitted as a statement
without an effect model.  This module provides a conservative effect record for
all recovered builtins and for linked MBC functions.
"""

from dataclasses import dataclass, replace
from typing import Iterable, Optional

from .linker import FunctionSignature
from .native_api import NativeCallSpec, engine_native_import, specialize_builtin_api
from .opcodes import BUILTINS
from .vm_stack import TYPE_FLOAT, TYPE_INT, TYPE_SLICE, TYPE_STRING


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
    "current_process_id", "file_create", "file_open", "file_close", "file_read", "file_write", "identity_int", "object_create",
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
    "snprintf", "file_read_line", "sprintf", "file_rename", "file_truncate", "file_set_time",
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
    "file_create", "file_open", "file_close", "file_read", "file_write", "file_seek", "file_remove",
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
