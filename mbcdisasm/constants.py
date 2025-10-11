"""Common opcode operand constants discovered during reversing sessions."""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Shared operand values
# ---------------------------------------------------------------------------

# Frequently observed helper operands.  Naming them once keeps the rest of the
# codebase readable – the raw hexadecimal numbers migrate between patterns
# which makes it hard to reason about them in isolation.
RET_MASK = 0x2910
IO_SLOT = 0x6910
IO_SLOT_ALIASES = frozenset({IO_SLOT, 0x6901, 0x3069, 0x2961})
IO_PORT_NAME = "io.port_6910"
PAGE_REGISTER = 0x6C01
FANOUT_FLAGS_A = 0x2C02
FANOUT_FLAGS_B = 0x2C03
CALL_SHUFFLE_STANDARD = 0x4B08


LITERAL_MARKER_HINTS: Dict[int, str] = {
    0x0067: "literal_hint",
    0x6704: "literal_hint",
    0x0400: "literal_hint",
    0x0110: "literal_hint",
}


OPERAND_ALIASES: Dict[int, str] = {
    RET_MASK: "RET_MASK",
    IO_SLOT: "IO_SLOT",
    0x6901: "IO_SLOT",
    0x3069: "IO_SLOT",
    0x2961: "IO_SLOT",
    PAGE_REGISTER: "PAGE_REG",
    FANOUT_FLAGS_A: "FANOUT_FLAGS",
    FANOUT_FLAGS_B: "FANOUT_FLAGS",
    CALL_SHUFFLE_STANDARD: "CALL_SHUFFLE_STD",
}


# ---------------------------------------------------------------------------
# Memory layout aliases
# ---------------------------------------------------------------------------

# Selected helper banks that appear throughout the control helpers.  The low
# nybble in the runtime stream is frequently repurposed for sub-banks.  To keep
# the output stable we normalise the value before applying the alias.
MEMORY_BANK_ALIASES = {
    0x4B10: "sys.helper",
    0x3D30: "io.helpers",
}

# Frequently accessed pages get a shorter alias to make the rendered memory
# references easier to read in complex table dispatch blocks.
MEMORY_PAGE_ALIASES = {
    (0x3D30, 0xDC): "io",
}


__all__ = [
    "RET_MASK",
    "IO_SLOT",
    "IO_SLOT_ALIASES",
    "IO_PORT_NAME",
    "PAGE_REGISTER",
    "FANOUT_FLAGS_A",
    "FANOUT_FLAGS_B",
    "CALL_SHUFFLE_STANDARD",
    "LITERAL_MARKER_HINTS",
    "OPERAND_ALIASES",
    "MEMORY_BANK_ALIASES",
    "MEMORY_PAGE_ALIASES",
]
