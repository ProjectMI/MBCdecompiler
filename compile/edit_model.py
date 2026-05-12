from __future__ import annotations

"""Safe edit policy for the MBC AST/source editor.

This module is deliberately separate from the Tk UI.  It describes which edits
are currently accepted by the compiler side and why.  The source editor can ask
this model before offering an action; unsupported edits are reported as blocked
until a lowering/relocation pass exists.
"""

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from compile.lossless_ir import find_instruction

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
DISPLAY_SYMBOL_RE = re.compile(
    r"(?:arg\d+|(?:g_)?(?:v|buf|arr|span|rec)_[0-9A-Fa-f]{4,8}|g_[0-9A-Fa-f]{4,8}|[A-Za-z_][A-Za-z0-9_]*)\Z"
)
TYPED_IMMEDIATE_OPCODES = {0x39, 0x28, 0x29}
FIXED_DATA_SPAN_OPCODES = {0x41, 0x65, 0x6C}
FIXED_INTEGER_OPERAND_OPCODES = {0x2C, 0x43, 0x50, 0x52, 0x53, 0x55, 0x5B, 0x5D, 0xCF, 0xD3, 0xD6, 0xD7}


@dataclass(frozen=True)
class EditCapability:
    """One possible editor action for a selected object."""

    key: str
    title: str
    status: str
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class EditPolicy:
    """High-level capability set for the current compiler/editor generation."""

    allow_display_renames: bool = True
    allow_fixed_width_immediates: bool = True
    allow_fixed_data_patches: bool = True
    allow_same_width_raw_instruction_patches: bool = True
    allow_length_changing_code: bool = False
    allow_ast_relowering: bool = False

    def summary_lines(self) -> list[str]:
        return [
            "Supported now:",
            "  - editable pseudo-source drafts in the GUI",
            "  - display-only variable/function renames saved in .mbcproj.json",
            "  - line-local source replacements of existing fixed-width numeric literals",
            "  - same-width integer operand edits for supported non-CFG rows such as program_i16/u16/u8",
            "  - line-local source replacements of existing fixed-capacity string/span literals",
            "  - fixed-range data byte/string/scalar patches through compile.lossless_ir helpers",
            "  - same-length raw instruction replacement through compile.lossless_ir helpers",
            "Blocked until lowering/relocation exists:",
            "  - source edits that insert/delete lines or statements",
            "  - condition rewrites that change instruction count/width",
            "  - inserting/deleting calls",
            "  - moving labels/basic blocks or changing CFG shape",
        ]


DEFAULT_POLICY = EditPolicy()


@dataclass(frozen=True)
class EditContext:
    """Selection context passed from UI to policy."""

    source: str
    offset: int | None = None
    identifier: str | None = None
    instruction: Mapping[str, Any] | None = None


def validate_identifier(name: str) -> bool:
    return bool(IDENT_RE.fullmatch(name))


def is_display_symbol(name: str) -> bool:
    return bool(DISPLAY_SYMBOL_RE.fullmatch(name))


def capabilities_for_context(context: EditContext, *, policy: EditPolicy = DEFAULT_POLICY) -> list[EditCapability]:
    caps: list[EditCapability] = []

    if context.identifier:
        ok = policy.allow_display_renames and is_display_symbol(context.identifier)
        caps.append(EditCapability(
            key="display_rename",
            title="Rename display symbol",
            status="allowed" if ok else "blocked",
            supported=ok,
            reason="Saved as editor annotation; does not alter bytecode." if ok else "Selection is not a safe display-level symbol.",
        ))

    row = context.instruction
    if row is not None:
        opcode = int(row.get("opcode", -1))
        is_imm = opcode in TYPED_IMMEDIATE_OPCODES
        caps.append(EditCapability(
            key="typed_immediate",
            title="Edit immediate/value constant",
            status="allowed" if policy.allow_fixed_width_immediates and is_imm else "blocked",
            supported=policy.allow_fixed_width_immediates and is_imm,
            reason="Instruction width is preserved." if is_imm else "Selected instruction is not push_imm32/push_imm_u16/push_imm_i8.",
        ))
        is_span = opcode in FIXED_DATA_SPAN_OPCODES
        caps.append(EditCapability(
            key="inline_span_text",
            title="Edit fixed-capacity inline string/span data",
            status="allowed" if policy.allow_fixed_data_patches and is_span else "blocked",
            supported=policy.allow_fixed_data_patches and is_span,
            reason="The data span capacity is fixed; bytecode width is unchanged." if is_span else "Selected instruction is not a fixed data span producer.",
        ))
        is_fixed_int = opcode in FIXED_INTEGER_OPERAND_OPCODES
        caps.append(EditCapability(
            key="fixed_integer_operand",
            title="Edit same-width integer operand",
            status="allowed" if policy.allow_fixed_width_immediates and is_fixed_int else "blocked",
            supported=policy.allow_fixed_width_immediates and is_fixed_int,
            reason="Operand width is preserved and the row is not a relative CFG edge." if is_fixed_int else "Selected instruction is not a supported same-width integer operand row.",
        ))
        caps.append(EditCapability(
            key="same_width_raw_instruction",
            title="Replace raw instruction bytes with same length",
            status="allowed" if policy.allow_same_width_raw_instruction_patches else "blocked",
            supported=policy.allow_same_width_raw_instruction_patches,
            reason="Opcode row length must stay unchanged; use helper API, not free text.",
        ))

    caps.append(EditCapability(
        key="ast_relower",
        title="Relower pretty AST block",
        status="blocked",
        supported=False,
        reason="Requires relocation/lowering pass for branches, program/function tables and metadata references.",
    ))
    return caps


def capabilities_for_offset(ir: Mapping[str, Any], offset: int | None, *, identifier: str | None = None) -> list[EditCapability]:
    row: Mapping[str, Any] | None = None
    if offset is not None:
        try:
            row = find_instruction(ir, offset)
        except KeyError:
            row = None
    return capabilities_for_context(EditContext(source="ui", offset=offset, identifier=identifier, instruction=row))


def concise_status(capabilities: Iterable[EditCapability]) -> str:
    allowed = [cap.title for cap in capabilities if cap.supported]
    if not allowed:
        return "no safe edit at selection"
    return "safe edits: " + "; ".join(allowed[:3])
