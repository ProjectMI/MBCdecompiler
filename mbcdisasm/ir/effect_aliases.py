"""Shared helpers for classifying stack cleanup side effects."""

from __future__ import annotations

from typing import Optional

from ..constants import FANOUT_FLAGS_A, FANOUT_FLAGS_B, IO_SLOT_ALIASES, RET_MASK

DIRECT_EPILOGUE_KIND_MAP = {
    "call_helpers": "helpers.invoke",
    "fanout": "helpers.fanout",
    "page_register": "frame.page_select",
    "stack_teardown": "frame.teardown",
    "op_6C_01": "frame.page_select",
    "op_08_00": "helpers.dispatch",
    "op_72_23": "helpers.wrapper",
    "op_52_06": "io.handshake",
    "op_0B_00": "io.handshake",
    "op_0B_E1": "io.handshake",
    "op_76_41": "io.bridge",
    "op_04_EC": "io.bridge",
    "op_2D_01": "frame.cleanup",
    "reduce_passthrough": "helpers.reduce",
}

MASK_STEP_MNEMONICS = {
    "epilogue",
    "op_29_10",
    "op_31_52",
    "op_32_29",
    "op_4B_0B",
    "op_4B_45",
    "op_4B_CC",
    "op_4F_01",
    "op_4F_02",
    "op_4F_03",
    "op_52_05",
    "op_5E_29",
    "op_70_29",
    "op_72_4A",
}

MASK_OPERANDS = {RET_MASK, FANOUT_FLAGS_A, FANOUT_FLAGS_B}
MASK_ALIAS_HINTS = {"RET_MASK", "FANOUT_FLAGS"}
MASK_ALIASES = MASK_ALIAS_HINTS

FRAME_DROP_MNEMONICS = {
    "op_01_0C",
    "op_01_30",
    "op_01_78",
    "op_01_C0",
    "op_2D_01",
}

IO_BRIDGE_MNEMONICS = {
    "op_3A_01",
    "op_3E_4B",
    "op_E1_4D",
    "op_E8_4B",
    "op_ED_4B",
    "op_F0_4B",
}

IO_STEP_MNEMONICS = {
    "op_10_08",
    "op_10_0A",
    "op_10_0E",
    "op_10_18",
    "op_10_3C",
    "op_10_69",
    "op_10_C5",
    "op_10_CC",
    "op_10_CD",
    "op_10_E3",
    "op_10_E5",
    "op_15_4A",
    "op_18_2A",
    "op_20_8C",
    "op_21_94",
    "op_21_AC",
    "op_28_6C",
    "op_52_06",
    "op_78_00",
    "op_89_01",
    "op_A0_03",
    "op_C0_00",
    "op_C3_01",
}

IO_OPCODE_FALLBACK = {
    0x10,
    0x11,
    0x14,
    0x15,
    0x18,
    0x1C,
    0x1D,
    0x20,
    0x21,
    0x28,
    0x52,
    0x78,
    0x89,
    0xA0,
    0xA2,
    0xC0,
    0xC3,
}

MASK_OPCODE_FALLBACK = {0x29, 0x31, 0x32, 0x4B, 0x4F, 0x52, 0x5E, 0x70, 0x72}
DROP_OPCODE_FALLBACK = {0x01, 0x2D}
BRIDGE_OPCODE_FALLBACK = {0x3A, 0x3E, 0x04, 0x4A, 0x4B, 0x76, 0xE1, 0xE8, 0xED, 0xF0}

FRAME_CLEAR_ALIASES = {"fmt.buffer_reset"}
FORMAT_PREFIXES = ("fmt.", "text.")
FRAME_CLEAR_ALIASES_LOWER = {alias.lower() for alias in FRAME_CLEAR_ALIASES}

FRAME_OPERAND_KIND_OVERRIDES = {
    0x0000: "frame.reset",
    0x0020: "helpers.format",
    0x0029: "frame.scheduler",
    0x002C: "frame.scheduler",
    0x0041: "frame.scheduler",
    0x0069: "frame.scheduler",
    0x006C: "frame.page_select",
    0x10E1: "io.bridge",
    0x2C04: "helpers.format",
    0x2DF0: "io.step",
    0x2EF0: "io.step",
    0x3100: "helpers.format",
    0x3E4B: "helpers.format",
    0x5B01: "helpers.format",
    0xED4D: "frame.page_select",
    0xF0EB: "io.step",
}

CHATOUT_ALIAS = "ChatOut"


def mnemonic_opcode(mnemonic: str) -> Optional[int]:
    """Extract the opcode nibble from an instruction mnemonic."""

    if not mnemonic.startswith("op_"):
        return None
    try:
        return int(mnemonic[3:5], 16)
    except (ValueError, IndexError):
        return None


def epilogue_step_kind(
    mnemonic: str,
    operand: Optional[int],
    alias: Optional[str],
    pops: int = 0,
    opcode: Optional[int] = None,
) -> str:
    """Return a coarse classification for the cleanup stack effect."""

    direct = DIRECT_EPILOGUE_KIND_MAP.get(mnemonic)
    if direct is not None:
        return direct

    alias_text = alias or ""
    if alias_text == CHATOUT_ALIAS:
        return "io.step"

    if operand is not None and operand in IO_SLOT_ALIASES:
        return "io.step"

    alias_upper = alias_text.upper()

    if mnemonic in MASK_STEP_MNEMONICS:
        return "frame.return_mask"
    if operand is not None and operand in MASK_OPERANDS:
        return "frame.return_mask"
    if alias_upper in MASK_ALIAS_HINTS:
        return "frame.return_mask"

    if opcode is None:
        opcode = mnemonic_opcode(mnemonic)
    if opcode in MASK_OPCODE_FALLBACK:
        return "frame.return_mask"

    if mnemonic in FRAME_DROP_MNEMONICS or opcode in DROP_OPCODE_FALLBACK:
        if pops:
            return "frame.drop"

    if mnemonic in IO_BRIDGE_MNEMONICS or opcode in BRIDGE_OPCODE_FALLBACK:
        return "io.bridge"

    if (
        mnemonic in IO_STEP_MNEMONICS
        or opcode in IO_OPCODE_FALLBACK
        or (alias_text == CHATOUT_ALIAS)
    ):
        return "io.step"

    if pops and mnemonic != "stack_teardown":
        return "frame.drop"

    return "frame.effect"


def classify_frame_effect_kind(
    mnemonic: str,
    operand: Optional[int],
    alias: Optional[str],
) -> str:
    """Refine frame effects into channel specific categories."""

    alias_text = alias.lower() if alias else ""

    if alias_text:
        if alias_text in FRAME_CLEAR_ALIASES_LOWER or alias_text.endswith(".reset"):
            return "frame.reset"
        if any(alias_text.startswith(prefix) for prefix in FORMAT_PREFIXES):
            return "helpers.format"
        if alias_text.startswith("scheduler."):
            return "frame.scheduler"
        if alias_text.startswith("page."):
            return "frame.page_select"
        if alias_text.startswith("io."):
            return "io.step"

    if operand is not None and operand in FRAME_OPERAND_KIND_OVERRIDES:
        return FRAME_OPERAND_KIND_OVERRIDES[operand]

    if operand is None:
        return "frame.write"

    if operand == 0:
        return "frame.reset"

    opcode = mnemonic_opcode(mnemonic)
    if opcode in BRIDGE_OPCODE_FALLBACK:
        return "io.bridge"

    return "frame.write"


def cleanup_category(
    mnemonic: str,
    operand: Optional[int],
    alias: Optional[str],
    pops: int = 0,
    opcode: Optional[int] = None,
) -> str:
    """Return the final category assigned to a cleanup stack step."""

    coarse = epilogue_step_kind(mnemonic, operand, alias, pops=pops, opcode=opcode)
    if coarse == "frame.effect":
        return classify_frame_effect_kind(mnemonic, operand, alias)
    return coarse
