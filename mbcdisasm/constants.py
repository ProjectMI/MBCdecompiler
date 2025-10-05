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
PAGE_REGISTER = 0x6C01
FANOUT_FLAGS_A = 0x2C02
FANOUT_FLAGS_B = 0x2C03
CALL_SHUFFLE_STANDARD = 0x4B08


OPERAND_ALIASES: Dict[int, str] = {
    RET_MASK: "RET_MASK",
    IO_SLOT: "IO_SLOT",
    PAGE_REGISTER: "PAGE_REG",
    FANOUT_FLAGS_A: "FANOUT_FLAGS",
    FANOUT_FLAGS_B: "FANOUT_FLAGS",
    CALL_SHUFFLE_STANDARD: "CALL_SHUFFLE_STD",
}


__all__ = [
    "RET_MASK",
    "IO_SLOT",
    "PAGE_REGISTER",
    "FANOUT_FLAGS_A",
    "FANOUT_FLAGS_B",
    "CALL_SHUFFLE_STANDARD",
    "OPERAND_ALIASES",
]
