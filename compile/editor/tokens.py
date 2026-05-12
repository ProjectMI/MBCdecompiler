from __future__ import annotations

import re

LOC_RE = re.compile(r"loc_([0-9A-Fa-f]{8})")
FUNCTION_RE = re.compile(r"^\s*function\s+([A-Za-z_][\w.]*)")
LOCAL_HELPER_RE = re.compile(r"^local_([0-9A-Fa-f]{8})$")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VARIABLE_RE = re.compile(
    r"\b(?:arg\d+|(?:g_)?(?:v|buf|arr|span|rec)_[0-9A-Fa-f]{4,8}|g_[0-9A-Fa-f]{4,8})\b"
)
RESERVED_WORDS = {
    "function", "globals", "if", "else", "switch", "case", "default", "while", "for",
    "return", "goto", "break", "continue", "yield_program", "program_restart", "halt_interpreter",
    "int", "float", "char", "string", "record", "int_ref", "float_ref", "native",
}
KEYWORD_RE = re.compile(
    r"\b(function|globals|if|else|switch|case|default|while|for|return|goto|break|continue|yield_program|program_restart|halt_interpreter)\b"
)
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
NUMBER_RE = re.compile(r"\b(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)\b")
SIGNED_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?)(?![A-Za-z0-9_])")
COMMENT_RE = re.compile(r"//.*")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*(?=\()")
DATA_COMMENT_RE = re.compile(r"data\[0x([0-9A-Fa-f]+)\]")
DATA_SYMBOL_RE = re.compile(r"\b(?:(?:g_)?(?:v|buf|arr|span|rec)_([0-9A-Fa-f]{4,8})|g_([0-9A-Fa-f]{4,8}))\b")
LINE_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
PROJECT_SUFFIX = ".mbcproj.json"
DEFAULT_DECOMPILED_DIR = "mbc_decompiled"
TYPED_IMMEDIATE_OPCODES = {0x39, 0x28, 0x29}
FIXED_SPAN_OPCODES = {0x41, 0x65, 0x6C}
FIXED_INTEGER_OPERAND_OPCODES = {0x2C, 0x43, 0x50, 0x52, 0x53, 0x55, 0x5B, 0x5D, 0xCF, 0xD3, 0xD6, 0xD7}
SCOPED_RENAME_SYMBOL_RE = re.compile(r"\b(?:arg\d+|(?:v|buf|arr|span|rec)_[0-9A-Fa-f]{4,8})\b")
GLOBAL_RENAME_SYMBOL_RE = re.compile(r"\b(?:(?:g_)?(?:v|buf|arr|span|rec)_[0-9A-Fa-f]{4,8}|g_[0-9A-Fa-f]{4,8})\b")
GLOBAL_CANON_SYMBOL_RE = re.compile(r"\b(?:g_(?:v|buf|arr|span|rec)_[0-9A-Fa-f]{4,8}|g_[0-9A-Fa-f]{4,8})\b")

