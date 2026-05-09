from __future__ import annotations

"""Recovered MBC opcode model.

This module deliberately keeps opcode semantics out of the CFG interpreter.  The
upper dispatcher table is the byte_55F7F0 / funcs_48983F table recovered from the
client, while the lower table is the jpt_47754E table reached by opcode 0x66
(`f`).  Lengths are expressed in bytecode bytes and include the opcode byte.

Important PC rule from sub_489410: the VM increments PC before invoking an
opcode handler.  Relative branch targets are therefore computed from
``pc_after_opcode`` (``off + 1``), not from the end of the encoded instruction.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import struct

CODE_FILE_OFFSET = 0x20

TYPE_NAMES: Dict[int, str] = {
    0: "i8/char",
    1: "span/string",
    16: "int32",
    17: "int32_ref_or_span",
    32: "float32",
    33: "float32_ref_or_span",
    48: "slice_descriptor",
}


@dataclass(frozen=True)
class OpcodeSpec:
    opcode: int
    char: str
    table_ea: int
    handler_ea: int
    handler_name: str
    mnemonic: str
    format: str
    semantic: str

    @property
    def opcode_hex(self) -> str:
        return f"0x{self.opcode:02X}"


@dataclass(frozen=True)
class BuiltinSpec:
    subopcode: int
    table_ea: int
    target_ea: int
    target_name: str
    mnemonic: str
    semantic: str

    @property
    def subopcode_hex(self) -> str:
        return f"0x{self.subopcode:02X}"


@dataclass(frozen=True)
class DecodedEdge:
    kind: str
    dst: Optional[int]
    note: str = ""


@dataclass(frozen=True)
class DecodedOpcode:
    mnemonic: str
    length: int
    operands: Dict[str, Any]
    terminal: bool
    known: bool
    edges: List[DecodedEdge]


def _s8(value: int) -> int:
    return value - 0x100 if value >= 0x80 else value


def _s16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]


def _u16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _s32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<i", buf, off)[0]


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _f32_from_u32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def safe_chr(value: int) -> str:
    if 32 <= value < 127:
        return chr(value)
    return "."


def type_name(type_id: int) -> str:
    return TYPE_NAMES.get(type_id, f"type_{type_id}")


_TOP_OPCODE_ROWS: Dict[int, Dict[str, Any]] = {0: {'char': '\\x00',
     'format': 'trap',
     'handler_ea': 4690992,
     'handler_name': 'sub_479430',
     'mnemonic': 'unknown_opcode_trap',
     'opcode': 0,
     'semantic': 'dispatcher trap; handler decrements PC and reports unknown script code',
     'table_ea': 5634584},
 33: {'char': '!',
      'format': 'none',
      'handler_ea': 4692432,
      'handler_name': 'sub_4799D0',
      'mnemonic': 'logical_not',
      'opcode': 33,
      'semantic': 'unary logical not',
      'table_ea': 5634272},
 34: {'char': '"',
      'format': 'none',
      'handler_ea': 4683456,
      'handler_name': 'sub_4776C0',
      'mnemonic': 'to_int_prev',
      'opcode': 34,
      'semantic': 'converts previous float value to integer',
      'table_ea': 5634456},
 37: {'char': '%',
      'format': 'none',
      'handler_ea': 4681504,
      'handler_name': 'sub_476F20',
      'mnemonic': 'mod',
      'opcode': 37,
      'semantic': 'integer modulo',
      'table_ea': 5634168},
 38: {'char': '&',
      'format': 'none',
      'handler_ea': 4681168,
      'handler_name': 'sub_476DD0',
      'mnemonic': 'address_of',
      'opcode': 38,
      'semantic': 'turns stack value metadata into pointer/slice descriptor',
      'table_ea': 5634392},
 40: {'char': '(',
      'format': 'typed_imm_u16',
      'handler_ea': 4679776,
      'handler_name': 'sub_476860',
      'mnemonic': 'push_imm_u16',
      'opcode': 40,
      'semantic': 'push typed immediate: type:u8, value:u16',
      'table_ea': 5634512},
 41: {'char': ')',
      'format': 'typed_imm_i8',
      'handler_ea': 4679872,
      'handler_name': 'sub_4768C0',
      'mnemonic': 'push_imm_i8',
      'opcode': 41,
      'semantic': 'push typed immediate: type:u8, value:i8',
      'table_ea': 5634520},
 42: {'char': '*',
      'format': 'none',
      'handler_ea': 4681408,
      'handler_name': 'sub_476EC0',
      'mnemonic': 'mul',
      'opcode': 42,
      'semantic': 'binary numeric multiplication',
      'table_ea': 5634152},
 43: {'char': '+',
      'format': 'none',
      'handler_ea': 4681248,
      'handler_name': 'sub_476E20',
      'mnemonic': 'add',
      'opcode': 43,
      'semantic': 'binary numeric addition',
      'table_ea': 5634136},
 44: {'char': ',',
      'format': 'u8',
      'handler_ea': 4679616,
      'handler_name': 'sub_4767C0',
      'mnemonic': 'set_arg_count',
      'opcode': 44,
      'semantic': 'sets VM argument count dword_86232C',
      'table_ea': 5634048},
 45: {'char': '-',
      'format': 'none',
      'handler_ea': 4681328,
      'handler_name': 'sub_476E70',
      'mnemonic': 'sub',
      'opcode': 45,
      'semantic': 'binary numeric subtraction',
      'table_ea': 5634144},
 46: {'char': '.',
      'format': 'none',
      'handler_ea': 4683104,
      'handler_name': 'sub_477560',
      'mnemonic': 'to_float',
      'opcode': 46,
      'semantic': 'converts current integer slot to float',
      'table_ea': 5634408},
 47: {'char': '/',
      'format': 'none',
      'handler_ea': 4691440,
      'handler_name': 'sub_4795F0',
      'mnemonic': 'div',
      'opcode': 47,
      'semantic': 'binary numeric division; integer path checks division by zero',
      'table_ea': 5634160},
 48: {'char': '0',
      'format': 'none',
      'handler_ea': 4679648,
      'handler_name': 'sub_4767E0',
      'mnemonic': 'stack_frame_reset',
      'opcode': 48,
      'semantic': 'main-loop special: reset stack pointer to current frame base',
      'table_ea': 5634056},
 49: {'char': '1',
      'format': 'none',
      'handler_ea': 4691056,
      'handler_name': 'sub_479470',
      'mnemonic': 'push_stack_frame',
      'opcode': 49,
      'semantic': 'pushes current stack base on VM stack-of-stacks',
      'table_ea': 5634176},
 50: {'char': '2',
      'format': 'none',
      'handler_ea': 4691120,
      'handler_name': 'sub_4794B0',
      'mnemonic': 'pop_stack_frame',
      'opcode': 50,
      'semantic': 'pops VM stack-of-stacks',
      'table_ea': 5634184},
 57: {'char': '9',
      'format': 'typed_imm32',
      'handler_ea': 4679680,
      'handler_name': 'sub_476800',
      'mnemonic': 'push_imm32',
      'opcode': 57,
      'semantic': 'push typed immediate: type:u8, value:u32',
      'table_ea': 5634072},
 58: {'char': ':',
      'format': 'none',
      'handler_ea': 4683136,
      'handler_name': 'sub_477580',
      'mnemonic': 'to_float_prev',
      'opcode': 58,
      'semantic': 'converts previous stack value to float',
      'table_ea': 5634416},
 59: {'char': ';',
      'format': 'none',
      'handler_ea': 4683760,
      'handler_name': 'sub_4777F0',
      'mnemonic': 'force_two_ints',
      'opcode': 59,
      'semantic': 'marks two stack cells as integer/bool type',
      'table_ea': 5634488},
 60: {'char': '<',
      'format': 'none',
      'handler_ea': 4681968,
      'handler_name': 'sub_4770F0',
      'mnemonic': 'lt',
      'opcode': 60,
      'semantic': 'binary less-than comparison',
      'table_ea': 5634216},
 61: {'char': '=',
      'format': 'none',
      'handler_ea': 4680736,
      'handler_name': 'sub_476C20',
      'mnemonic': 'store',
      'opcode': 61,
      'semantic': 'stores top value into lvalue below it',
      'table_ea': 5634128},
 62: {'char': '>',
      'format': 'none',
      'handler_ea': 4681840,
      'handler_name': 'sub_477070',
      'mnemonic': 'gt',
      'opcode': 62,
      'semantic': 'binary greater-than comparison',
      'table_ea': 5634208},
 65: {'char': 'A',
      'format': 'inline_span',
      'handler_ea': 4679968,
      'handler_name': 'sub_476920',
      'mnemonic': 'push_inline_span',
      'opcode': 65,
      'semantic': 'pushes inline span/slice: data_offset:u32, length:u16',
      'table_ea': 5634368},
 67: {'char': 'C',
      'format': 'program_i16',
      'handler_ea': 4680592,
      'handler_name': 'sub_476B90',
      'mnemonic': 'program_activate',
      'opcode': 67,
      'semantic': 'activates target program',
      'table_ea': 5634104},
 71: {'char': 'G',
      'format': 'rel32',
      'handler_ea': 4679584,
      'handler_name': 'sub_4767A0',
      'mnemonic': 'jmp_rel32',
      'opcode': 71,
      'semantic': 'unconditional relative jump; target is pc_after_opcode + rel32',
      'table_ea': 5634040},
 72: {'char': 'H',
      'format': 'none',
      'handler_ea': 4683088,
      'handler_name': 'sub_477550',
      'mnemonic': 'halt_interpreter',
      'opcode': 72,
      'semantic': 'sets interpreter stop flag',
      'table_ea': 5634400},
 73: {'char': 'I',
      'format': 'jfalse_rel32',
      'handler_ea': 4697904,
      'handler_name': 'sub_47AF30',
      'mnemonic': 'jfalse_rel32',
      'opcode': 73,
      'semantic': 'pop/test condition; false jumps by rel32, true skips operand',
      'table_ea': 5634064},
 74: {'char': 'J',
      'format': 'rel16',
      'handler_ea': 4679600,
      'handler_name': 'sub_4767B0',
      'mnemonic': 'jmp_rel16',
      'opcode': 74,
      'semantic': 'unconditional relative short jump; target is pc_after_opcode + rel16',
      'table_ea': 5634496},
 75: {'char': 'K',
      'format': 'jfalse_rel16',
      'handler_ea': 4697936,
      'handler_name': 'sub_47AF50',
      'mnemonic': 'jfalse_rel16',
      'opcode': 75,
      'semantic': 'pop/test condition; false jumps by rel16, true skips operand',
      'table_ea': 5634504},
 76: {'char': 'L',
      'format': 'logical_or_rel16',
      'handler_ea': 4682352,
      'handler_name': 'sub_477270',
      'mnemonic': 'logical_or_rel16',
      'opcode': 76,
      'semantic': 'short-circuit OR: true keeps bool true and jumps, false pops and falls through',
      'table_ea': 5634568},
 77: {'char': 'M',
      'format': 'logical_and_rel16',
      'handler_ea': 4682432,
      'handler_name': 'sub_4772C0',
      'mnemonic': 'logical_and_rel16',
      'opcode': 77,
      'semantic': 'short-circuit AND: false keeps bool false and jumps, true pops and falls through',
      'table_ea': 5634576},
 79: {'char': 'O',
      'format': 'prologue',
      'handler_ea': 4691664,
      'handler_name': 'sub_4796D0',
      'mnemonic': 'program_prologue',
      'opcode': 79,
      'semantic': 'binds call parameters into local data slots; count:i8 then abs(count) descriptors of type:u8 + '
                  'dst_offset:u32',
      'table_ea': 5634248},
 80: {'char': 'P',
      'format': 'program_i16',
      'handler_ea': 4680544,
      'handler_name': 'sub_476B60',
      'mnemonic': 'program_stop',
      'opcode': 80,
      'semantic': 'sets target program inactive',
      'table_ea': 5634384},
 82: {'char': 'R',
      'format': 'program_i16',
      'handler_ea': 4680304,
      'handler_name': 'sub_476A70',
      'mnemonic': 'program_restart',
      'opcode': 82,
      'semantic': 'sets target program PC to its primary entry and activates it; parent is reset to -1',
      'table_ea': 5634088},
 83: {'char': 'S',
      'format': 'program_i16',
      'handler_ea': 4680496,
      'handler_name': 'sub_476B30',
      'mnemonic': 'program_reset_alt_pc',
      'opcode': 83,
      'semantic': 'sets target program PC to alternate saved entry',
      'table_ea': 5634096},
 85: {'char': 'U',
      'format': 'program_i16',
      'handler_ea': 4680400,
      'handler_name': 'sub_476AD0',
      'mnemonic': 'program_restart_child',
      'opcode': 85,
      'semantic': 'sets target program PC to primary entry and records current program as parent',
      'table_ea': 5634296},
 91: {'char': '[',
      'format': 'u16',
      'handler_ea': 4683248,
      'handler_name': 'sub_4775F0',
      'mnemonic': 'ptr_add_scaled_u16',
      'opcode': 91,
      'semantic': 'pops index/pointer pair and adds immediate-scaled offset',
      'table_ea': 5634432},
 93: {'char': ']',
      'format': 'u16',
      'handler_ea': 4683328,
      'handler_name': 'sub_477640',
      'mnemonic': 'ptr_sub_scaled_u16',
      'opcode': 93,
      'semantic': 'pops index/pointer pair and subtracts immediate-scaled offset',
      'table_ea': 5634440},
 94: {'char': '^',
      'format': 'none',
      'handler_ea': 4680912,
      'handler_name': 'sub_476CD0',
      'mnemonic': 'deref',
      'opcode': 94,
      'semantic': 'dereferences pointer/slice descriptor on stack',
      'table_ea': 5634376},
 96: {'char': '`',
      'format': 'none',
      'handler_ea': 4683408,
      'handler_name': 'sub_477690',
      'mnemonic': 'to_int',
      'opcode': 96,
      'semantic': 'converts current float slot to integer',
      'table_ea': 5634448},
 97: {'char': 'a',
      'format': 'array_abs',
      'handler_ea': 4698000,
      'handler_name': 'sub_47AF90',
      'mnemonic': 'array_index_abs',
      'opcode': 97,
      'semantic': 'indexed absolute array reference: type:u8, elem_size:u16, base:u32, span:u32, count:i32; index is '
                  'taken from stack',
      'table_ea': 5634080},
 98: {'char': 'b',
      'format': 'array2_checked',
      'handler_ea': 4698400,
      'handler_name': 'sub_47B120',
      'mnemonic': 'array2_index_checked',
      'opcode': 98,
      'semantic': 'indexed relative array/slice reference with explicit count; type:u8, elem_size:u16, count:i32; '
                  'base/index from stack',
      'table_ea': 5634344},
 99: {'char': 'c',
      'format': 'call_rel32',
      'handler_ea': 4699776,
      'handler_name': 'sub_47B680',
      'mnemonic': 'call_rel32',
      'opcode': 99,
      'semantic': 'pushes return address then jumps by rel32',
      'table_ea': 5634112},
 100: {'char': 'd',
       'format': 'slice_offset_ref',
       'handler_ea': 4699152,
       'handler_name': 'sub_47B410',
       'mnemonic': 'slice_offset_ref',
       'opcode': 100,
       'semantic': 'relative slice reference: type:u8, offset:u16; type 48 has extra length:u32',
       'table_ea': 5634544},
 101: {'char': 'e',
       'format': 'typed_span_ref',
       'handler_ea': 4680080,
       'handler_name': 'sub_476990',
       'mnemonic': 'push_typed_span_ref',
       'opcode': 101,
       'semantic': 'push typed span reference: type:u8, data_offset:u32, length:u32',
       'table_ea': 5634536},
 102: {'char': 'f',
       'format': 'builtin',
       'handler_ea': 4683008,
       'handler_name': 'sub_477500',
       'mnemonic': 'builtin_call',
       'opcode': 102,
       'semantic': 'second-level dispatcher; subopcode:u8 indexes jpt_47754E / low_mbc_opcode_table',
       'table_ea': 5634304},
 103: {'char': 'g',
       'format': 'none',
       'handler_ea': 4691040,
       'handler_name': 'sub_479460',
       'mnemonic': 'unlinked_call_error',
       'opcode': 103,
       'semantic': 'runtime error path: unlinked function was called',
       'table_ea': 5634360},
 104: {'char': 'h',
       'format': 'slice_offset_span',
       'handler_ea': 4699536,
       'handler_name': 'sub_47B590',
       'mnemonic': 'slice_offset_span',
       'opcode': 104,
       'semantic': 'relative slice/span reference: type:u8, offset:u16, length:u32',
       'table_ea': 5634552},
 105: {'char': 'i',
       'format': 'data_ref',
       'handler_ea': 4679392,
       'handler_name': 'sub_4766E0',
       'mnemonic': 'push_data_ref',
       'opcode': 105,
       'semantic': 'push typed lvalue/value from data section: type:u8, data_offset:u32',
       'table_ea': 5634032},
 108: {'char': 'l',
       'format': 'typed_span_inline',
       'handler_ea': 4680192,
       'handler_name': 'sub_476A00',
       'mnemonic': 'push_inline_typed_span',
       'opcode': 108,
       'semantic': 'push typed inline span: type:u8, data_offset:u32, length:u32',
       'table_ea': 5634560},
 109: {'char': 'm',
       'format': 'array2',
       'handler_ea': 4698816,
       'handler_name': 'sub_47B2C0',
       'mnemonic': 'array2_index',
       'opcode': 109,
       'semantic': 'indexed relative array/slice reference; type:u8, elem_size:u16; base/index from stack',
       'table_ea': 5634352},
 114: {'char': 'r',
       'format': 'none',
       'handler_ea': 4691152,
       'handler_name': 'sub_4794D0',
       'mnemonic': 'return',
       'opcode': 114,
       'semantic': 'returns through VM return stack / process stack',
       'table_ea': 5634120},
 116: {'char': 't',
       'format': 'none',
       'handler_ea': 4680656,
       'handler_name': 'sub_476BD0',
       'mnemonic': 'return_local',
       'opcode': 116,
       'semantic': 'returns to local return stack entry or terminates program',
       'table_ea': 5634464},
 126: {'char': '~',
       'format': 'none',
       'handler_ea': 4683168,
       'handler_name': 'sub_4775A0',
       'mnemonic': 'swap',
       'opcode': 126,
       'semantic': 'swaps top two VM stack cells',
       'table_ea': 5634424},
 201: {'char': '\\xC9',
       'format': 'none',
       'handler_ea': 4700176,
       'handler_name': 'sub_47B810',
       'mnemonic': 'call_linked_function',
       'opcode': 201,
       'semantic': 'runtime-linked function call by pending name; no inline operand',
       'table_ea': 5634528},
 207: {'char': '\\xCF',
       'format': 'u16',
       'handler_ea': 4683504,
       'handler_name': 'sub_4776F0',
       'mnemonic': 'ptr_add_assign_u16',
       'opcode': 207,
       'semantic': 'adds immediate u16 to pointer/value and writes it back',
       'table_ea': 5634472},
 211: {'char': '\\xD3',
       'format': 'u16',
       'handler_ea': 4683568,
       'handler_name': 'sub_477730',
       'mnemonic': 'ptr_sub_assign_u16',
       'opcode': 211,
       'semantic': 'subtracts immediate u16 from pointer/value and writes it back',
       'table_ea': 5634480},
 214: {'char': '\\xD6',
       'format': 'u16',
       'handler_ea': 4683632,
       'handler_name': 'sub_477770',
       'mnemonic': 'add_assign_u16',
       'opcode': 214,
       'semantic': 'adds immediate u16 to integer lvalue in data memory',
       'table_ea': 5634328},
 215: {'char': '\\xD7',
       'format': 'u16',
       'handler_ea': 4683696,
       'handler_name': 'sub_4777B0',
       'mnemonic': 'sub_assign_u16',
       'opcode': 215,
       'semantic': 'subtracts immediate u16 from integer lvalue in data memory',
       'table_ea': 5634336},
 225: {'char': '\\xE1',
       'format': 'none',
       'handler_ea': 4682096,
       'handler_name': 'sub_477170',
       'mnemonic': 'ge',
       'opcode': 225,
       'semantic': 'binary greater-or-equal comparison',
       'table_ea': 5634224},
 232: {'char': '\\xE8',
       'format': 'none',
       'handler_ea': 4682512,
       'handler_name': 'sub_477310',
       'mnemonic': 'force_int_type_alt',
       'opcode': 232,
       'semantic': 'same handler as 0xEB; marks current stack slot as integer/bool type',
       'table_ea': 5634264},
 235: {'char': '\\xEB',
       'format': 'none',
       'handler_ea': 4682512,
       'handler_name': 'sub_477310',
       'mnemonic': 'force_int_type',
       'opcode': 235,
       'semantic': 'marks current stack slot as integer/bool type',
       'table_ea': 5634256},
 236: {'char': '\\xEC',
       'format': 'none',
       'handler_ea': 4682224,
       'handler_name': 'sub_4771F0',
       'mnemonic': 'le',
       'opcode': 236,
       'semantic': 'binary less-or-equal comparison',
       'table_ea': 5634232},
 237: {'char': '\\xED',
       'format': 'none',
       'handler_ea': 4681712,
       'handler_name': 'sub_476FF0',
       'mnemonic': 'ne',
       'opcode': 237,
       'semantic': 'binary inequality comparison',
       'table_ea': 5634200},
 239: {'char': '\\xEF',
       'format': 'none',
       'handler_ea': 4682560,
       'handler_name': 'sub_477340',
       'mnemonic': 'pre_inc',
       'opcode': 239,
       'semantic': 'pre-increment lvalue and keep updated value',
       'table_ea': 5634280},
 240: {'char': '\\xF0',
       'format': 'none',
       'handler_ea': 4681584,
       'handler_name': 'sub_476F70',
       'mnemonic': 'eq',
       'opcode': 240,
       'semantic': 'binary equality comparison',
       'table_ea': 5634192},
 241: {'char': '\\xF1',
       'format': 'none',
       'handler_ea': 4691552,
       'handler_name': 'sub_479660',
       'mnemonic': 'neg',
       'opcode': 241,
       'semantic': 'unary numeric negation',
       'table_ea': 5634240},
 243: {'char': '\\xF3',
       'format': 'none',
       'handler_ea': 4682672,
       'handler_name': 'sub_4773B0',
       'mnemonic': 'pre_dec',
       'opcode': 243,
       'semantic': 'pre-decrement lvalue and keep updated value',
       'table_ea': 5634288},
 246: {'char': '\\xF6',
       'format': 'none',
       'handler_ea': 4682784,
       'handler_name': 'sub_477420',
       'mnemonic': 'post_inc',
       'opcode': 246,
       'semantic': 'post-increment lvalue; memory is updated while stack keeps old value',
       'table_ea': 5634312},
 247: {'char': '\\xF7',
       'format': 'none',
       'handler_ea': 4682896,
       'handler_name': 'sub_477490',
       'mnemonic': 'post_dec',
       'opcode': 247,
       'semantic': 'post-decrement lvalue; memory is updated while stack keeps old value',
       'table_ea': 5634320}}
_LOW_OPCODE_ROWS: Dict[int, Dict[str, Any]] = {0: {'mnemonic': 'debug_print_float',
     'semantic': 'debug print float; falls through to print string in original switch',
     'subopcode': 0,
     'table_ea': 5634856,
     'target_ea': 4700464,
     'target_name': 'loc_47B930'},
 1: {'mnemonic': 'debug_print_float_alias',
     'semantic': 'alias of debug print float; same target as subopcode 0',
     'subopcode': 1,
     'table_ea': 5634860,
     'target_ea': 4700464,
     'target_name': 'loc_47B930'},
 2: {'mnemonic': 'print_string_or_exit',
     'semantic': 'prints string argument, exits when called with zero args',
     'subopcode': 2,
     'table_ea': 5634864,
     'target_ea': 4700544,
     'target_name': 'loc_47B980'},
 3: {'mnemonic': 'sin',
     'semantic': 'pushes sin(float_arg)',
     'subopcode': 3,
     'table_ea': 5634868,
     'target_ea': 4700608,
     'target_name': 'loc_47B9C0'},
 4: {'mnemonic': 'cos',
     'semantic': 'pushes cos(float_arg)',
     'subopcode': 4,
     'table_ea': 5634872,
     'target_ea': 4700656,
     'target_name': 'loc_47B9F0'},
 5: {'mnemonic': 'abs_float',
     'semantic': 'pushes absolute value of float_arg',
     'subopcode': 5,
     'table_ea': 5634876,
     'target_ea': 4700912,
     'target_name': 'loc_47BAF0'},
 6: {'mnemonic': 'abs_int',
     'semantic': 'pushes absolute value of int_arg',
     'subopcode': 6,
     'table_ea': 5634880,
     'target_ea': 4700816,
     'target_name': 'loc_47BA90'},
 7: {'mnemonic': 'atan2',
     'semantic': 'pushes atan2(y, x)',
     'subopcode': 7,
     'table_ea': 5634884,
     'target_ea': 4700752,
     'target_name': 'loc_47BA50'},
 8: {'mnemonic': 'push_code_base',
     'semantic': 'pushes current code base pointer/offset marker',
     'subopcode': 8,
     'table_ea': 5634888,
     'target_ea': 4700960,
     'target_name': 'loc_47BB20'},
 9: {'mnemonic': 'sscanf',
     'semantic': 'sscanf wrapper; argument count controls destination count',
     'subopcode': 9,
     'table_ea': 5634892,
     'target_ea': 4715952,
     'target_name': 'loc_47F5B0'},
 10: {'mnemonic': 'window_api',
      'semantic': 'nested window/UI API dispatcher',
      'subopcode': 10,
      'table_ea': 5634896,
      'target_ea': 4719760,
      'target_name': 'loc_480490'},
 11: {'mnemonic': 'builtin_0B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 11,
      'table_ea': 5634900,
      'target_ea': 4701600,
      'target_name': 'loc_47BDA0'},
 12: {'mnemonic': 'builtin_0C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 12,
      'table_ea': 5634904,
      'target_ea': 4701968,
      'target_name': 'loc_47BF10'},
 13: {'mnemonic': 'builtin_0D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 13,
      'table_ea': 5634908,
      'target_ea': 4702032,
      'target_name': 'loc_47BF50'},
 14: {'mnemonic': 'builtin_0E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 14,
      'table_ea': 5634912,
      'target_ea': 4702112,
      'target_name': 'loc_47BFA0'},
 15: {'mnemonic': 'builtin_0F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 15,
      'table_ea': 5634916,
      'target_ea': 4726048,
      'target_name': 'loc_481D20'},
 16: {'mnemonic': 'builtin_10',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 16,
      'table_ea': 5634920,
      'target_ea': 4702160,
      'target_name': 'loc_47BFD0'},
 17: {'mnemonic': 'builtin_11',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 17,
      'table_ea': 5634924,
      'target_ea': 4702256,
      'target_name': 'loc_47C030'},
 18: {'mnemonic': 'builtin_12',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 18,
      'table_ea': 5634928,
      'target_ea': 4702400,
      'target_name': 'loc_47C0C0'},
 19: {'mnemonic': 'builtin_13',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 19,
      'table_ea': 5634932,
      'target_ea': 4702544,
      'target_name': 'loc_47C150'},
 20: {'mnemonic': 'builtin_14',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 20,
      'table_ea': 5634936,
      'target_ea': 4703600,
      'target_name': 'loc_47C570'},
 21: {'mnemonic': 'builtin_15',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 21,
      'table_ea': 5634940,
      'target_ea': 4703680,
      'target_name': 'loc_47C5C0'},
 22: {'mnemonic': 'builtin_16',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 22,
      'table_ea': 5634944,
      'target_ea': 4703712,
      'target_name': 'loc_47C5E0'},
 23: {'mnemonic': 'builtin_17',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 23,
      'table_ea': 5634948,
      'target_ea': 4703808,
      'target_name': 'loc_47C640'},
 24: {'mnemonic': 'builtin_18',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 24,
      'table_ea': 5634952,
      'target_ea': 4703904,
      'target_name': 'loc_47C6A0'},
 25: {'mnemonic': 'builtin_19',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 25,
      'table_ea': 5634956,
      'target_ea': 4704128,
      'target_name': 'loc_47C780'},
 26: {'mnemonic': 'builtin_1A',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 26,
      'table_ea': 5634960,
      'target_ea': 4704224,
      'target_name': 'loc_47C7E0'},
 27: {'mnemonic': 'builtin_1B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 27,
      'table_ea': 5634964,
      'target_ea': 4704128,
      'target_name': 'loc_47C780'},
 28: {'mnemonic': 'builtin_1C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 28,
      'table_ea': 5634968,
      'target_ea': 4702624,
      'target_name': 'loc_47C1A0'},
 29: {'mnemonic': 'builtin_1D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 29,
      'table_ea': 5634972,
      'target_ea': 4706960,
      'target_name': 'loc_47D290'},
 30: {'mnemonic': 'builtin_1E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 30,
      'table_ea': 5634976,
      'target_ea': 4707008,
      'target_name': 'loc_47D2C0'},
 31: {'mnemonic': 'builtin_1F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 31,
      'table_ea': 5634980,
      'target_ea': 4684816,
      'target_name': 'loc_477C10'},
 32: {'mnemonic': 'builtin_20',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 32,
      'table_ea': 5634984,
      'target_ea': 4707152,
      'target_name': 'loc_47D350'},
 33: {'mnemonic': 'builtin_21',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 33,
      'table_ea': 5634988,
      'target_ea': 4705056,
      'target_name': 'loc_47CB20'},
 34: {'mnemonic': 'builtin_22',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 34,
      'table_ea': 5634992,
      'target_ea': 4707248,
      'target_name': 'loc_47D3B0'},
 35: {'mnemonic': 'builtin_23',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 35,
      'table_ea': 5634996,
      'target_ea': 4707776,
      'target_name': 'loc_47D5C0'},
 36: {'mnemonic': 'builtin_24',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 36,
      'table_ea': 5635000,
      'target_ea': 4708496,
      'target_name': 'loc_47D890'},
 37: {'mnemonic': 'builtin_25',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 37,
      'table_ea': 5635004,
      'target_ea': 4708800,
      'target_name': 'loc_47D9C0'},
 38: {'mnemonic': 'builtin_26',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 38,
      'table_ea': 5635008,
      'target_ea': 4709344,
      'target_name': 'loc_47DBE0'},
 39: {'mnemonic': 'builtin_27',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 39,
      'table_ea': 5635012,
      'target_ea': 4704016,
      'target_name': 'loc_47C710'},
 40: {'mnemonic': 'builtin_28',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 40,
      'table_ea': 5635016,
      'target_ea': 4711680,
      'target_name': 'loc_47E500'},
 41: {'mnemonic': 'builtin_29',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 41,
      'table_ea': 5635020,
      'target_ea': 4711824,
      'target_name': 'loc_47E590'},
 42: {'mnemonic': 'builtin_2A',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 42,
      'table_ea': 5635024,
      'target_ea': 4712096,
      'target_name': 'loc_47E6A0'},
 43: {'mnemonic': 'builtin_2B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 43,
      'table_ea': 5635028,
      'target_ea': 4712144,
      'target_name': 'loc_47E6D0'},
 44: {'mnemonic': 'builtin_2C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 44,
      'table_ea': 5635032,
      'target_ea': 4712272,
      'target_name': 'loc_47E750'},
 45: {'mnemonic': 'builtin_2D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 45,
      'table_ea': 5635036,
      'target_ea': 4713056,
      'target_name': 'loc_47EA60'},
 46: {'mnemonic': 'builtin_2E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 46,
      'table_ea': 5635040,
      'target_ea': 4713152,
      'target_name': 'loc_47EAC0'},
 47: {'mnemonic': 'builtin_2F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 47,
      'table_ea': 5635044,
      'target_ea': 4713280,
      'target_name': 'loc_47EB40'},
 48: {'mnemonic': 'builtin_30',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 48,
      'table_ea': 5635048,
      'target_ea': 4713552,
      'target_name': 'loc_47EC50'},
 49: {'mnemonic': 'builtin_31',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 49,
      'table_ea': 5635052,
      'target_ea': 4714656,
      'target_name': 'loc_47F0A0'},
 50: {'mnemonic': 'builtin_32',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 50,
      'table_ea': 5635056,
      'target_ea': 4714784,
      'target_name': 'loc_47F120'},
 51: {'mnemonic': 'builtin_33',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 51,
      'table_ea': 5635060,
      'target_ea': 4714880,
      'target_name': 'loc_47F180'},
 52: {'mnemonic': 'builtin_34',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 52,
      'table_ea': 5635064,
      'target_ea': 4714160,
      'target_name': 'loc_47EEB0'},
 53: {'mnemonic': 'builtin_35',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 53,
      'table_ea': 5635068,
      'target_ea': 4714944,
      'target_name': 'loc_47F1C0'},
 54: {'mnemonic': 'builtin_36',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 54,
      'table_ea': 5635072,
      'target_ea': 4713728,
      'target_name': 'loc_47ED00'},
 55: {'mnemonic': 'builtin_37',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 55,
      'table_ea': 5635076,
      'target_ea': 4713792,
      'target_name': 'loc_47ED40'},
 56: {'mnemonic': 'builtin_38',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 56,
      'table_ea': 5635080,
      'target_ea': 4713856,
      'target_name': 'loc_47ED80'},
 57: {'mnemonic': 'builtin_39',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 57,
      'table_ea': 5635084,
      'target_ea': 4713920,
      'target_name': 'loc_47EDC0'},
 58: {'mnemonic': 'builtin_3A',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 58,
      'table_ea': 5635088,
      'target_ea': 4713984,
      'target_name': 'loc_47EE00'},
 59: {'mnemonic': 'builtin_3B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 59,
      'table_ea': 5635092,
      'target_ea': 4714048,
      'target_name': 'loc_47EE40'},
 60: {'mnemonic': 'builtin_3C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 60,
      'table_ea': 5635096,
      'target_ea': 4714112,
      'target_name': 'loc_47EE80'},
 61: {'mnemonic': 'builtin_3D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 61,
      'table_ea': 5635100,
      'target_ea': 4715120,
      'target_name': 'loc_47F270'},
 62: {'mnemonic': 'builtin_3E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 62,
      'table_ea': 5635104,
      'target_ea': 4715504,
      'target_name': 'loc_47F3F0'},
 63: {'mnemonic': 'builtin_3F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 63,
      'table_ea': 5635108,
      'target_ea': 4715552,
      'target_name': 'loc_47F420'},
 64: {'mnemonic': 'builtin_40',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 64,
      'table_ea': 5635112,
      'target_ea': 4713168,
      'target_name': 'loc_47EAD0'},
 65: {'mnemonic': 'builtin_41',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 65,
      'table_ea': 5635116,
      'target_ea': 4712560,
      'target_name': 'loc_47E870'},
 66: {'mnemonic': 'builtin_42',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 66,
      'table_ea': 5635120,
      'target_ea': 4712640,
      'target_name': 'loc_47E8C0'},
 67: {'mnemonic': 'builtin_43',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 67,
      'table_ea': 5635124,
      'target_ea': 4716320,
      'target_name': 'loc_47F720'},
 68: {'mnemonic': 'builtin_44',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 68,
      'table_ea': 5635128,
      'target_ea': 4712944,
      'target_name': 'loc_47E9F0'},
 69: {'mnemonic': 'builtin_45',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 69,
      'table_ea': 5635132,
      'target_ea': 4712896,
      'target_name': 'loc_47E9C0'},
 70: {'mnemonic': 'builtin_46',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 70,
      'table_ea': 5635136,
      'target_ea': 4712672,
      'target_name': 'loc_47E8E0'},
 71: {'mnemonic': 'builtin_47',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 71,
      'table_ea': 5635140,
      'target_ea': 4712752,
      'target_name': 'loc_47E930'},
 72: {'mnemonic': 'builtin_48',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 72,
      'table_ea': 5635144,
      'target_ea': 4712800,
      'target_name': 'loc_47E960'},
 73: {'mnemonic': 'builtin_49',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 73,
      'table_ea': 5635148,
      'target_ea': 4715632,
      'target_name': 'loc_47F470'},
 74: {'mnemonic': 'builtin_4A',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 74,
      'table_ea': 5635152,
      'target_ea': 4713008,
      'target_name': 'loc_47EA30'},
 75: {'mnemonic': 'builtin_4B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 75,
      'table_ea': 5635156,
      'target_ea': 4708960,
      'target_name': 'loc_47DA60'},
 76: {'mnemonic': 'builtin_4C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 76,
      'table_ea': 5635160,
      'target_ea': 4709088,
      'target_name': 'loc_47DAE0'},
 77: {'mnemonic': 'builtin_4D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 77,
      'table_ea': 5635164,
      'target_ea': 4717008,
      'target_name': 'loc_47F9D0'},
 78: {'mnemonic': 'builtin_4E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 78,
      'table_ea': 5635168,
      'target_ea': 4717552,
      'target_name': 'loc_47FBF0'},
 79: {'mnemonic': 'builtin_4F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 79,
      'table_ea': 5635172,
      'target_ea': 4717776,
      'target_name': 'loc_47FCD0'},
 80: {'mnemonic': 'builtin_50',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 80,
      'table_ea': 5635176,
      'target_ea': 4706816,
      'target_name': 'loc_47D200'},
 81: {'mnemonic': 'builtin_51',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 81,
      'table_ea': 5635180,
      'target_ea': 4701104,
      'target_name': 'loc_47BBB0'},
 82: {'mnemonic': 'builtin_52',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 82,
      'table_ea': 5635184,
      'target_ea': 4717888,
      'target_name': 'loc_47FD40'},
 83: {'mnemonic': 'builtin_53',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 83,
      'table_ea': 5635188,
      'target_ea': 4714288,
      'target_name': 'loc_47EF30'},
 84: {'mnemonic': 'builtin_54',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 84,
      'table_ea': 5635192,
      'target_ea': 4714592,
      'target_name': 'loc_47F060'},
 85: {'mnemonic': 'builtin_55',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 85,
      'table_ea': 5635196,
      'target_ea': 4717936,
      'target_name': 'loc_47FD70'},
 86: {'mnemonic': 'builtin_56',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 86,
      'table_ea': 5635200,
      'target_ea': 4711632,
      'target_name': 'loc_47E4D0'},
 87: {'mnemonic': 'builtin_57',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 87,
      'table_ea': 5635204,
      'target_ea': 4718032,
      'target_name': 'loc_47FDD0'},
 88: {'mnemonic': 'builtin_58',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 88,
      'table_ea': 5635208,
      'target_ea': 4718128,
      'target_name': 'loc_47FE30'},
 89: {'mnemonic': 'builtin_59',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 89,
      'table_ea': 5635212,
      'target_ea': 4718224,
      'target_name': 'loc_47FE90'},
 90: {'mnemonic': 'builtin_5A',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 90,
      'table_ea': 5635216,
      'target_ea': 4718272,
      'target_name': 'loc_47FEC0'},
 91: {'mnemonic': 'builtin_5B',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 91,
      'table_ea': 5635220,
      'target_ea': 4711200,
      'target_name': 'loc_47E320'},
 92: {'mnemonic': 'builtin_5C',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 92,
      'table_ea': 5635224,
      'target_ea': 4718320,
      'target_name': 'loc_47FEF0'},
 93: {'mnemonic': 'builtin_5D',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 93,
      'table_ea': 5635228,
      'target_ea': 4718496,
      'target_name': 'loc_47FFA0'},
 94: {'mnemonic': 'builtin_5E',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 94,
      'table_ea': 5635232,
      'target_ea': 4718608,
      'target_name': 'loc_480010'},
 95: {'mnemonic': 'builtin_5F',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 95,
      'table_ea': 5635236,
      'target_ea': 4708272,
      'target_name': 'loc_47D7B0'},
 96: {'mnemonic': 'builtin_60',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 96,
      'table_ea': 5635240,
      'target_ea': 4717664,
      'target_name': 'loc_47FC60'},
 97: {'mnemonic': 'builtin_61',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 97,
      'table_ea': 5635244,
      'target_ea': 4718704,
      'target_name': 'loc_480070'},
 98: {'mnemonic': 'builtin_62',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 98,
      'table_ea': 5635248,
      'target_ea': 4718832,
      'target_name': 'loc_4800F0'},
 99: {'mnemonic': 'builtin_63',
      'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
      'subopcode': 99,
      'table_ea': 5635252,
      'target_ea': 4715040,
      'target_name': 'loc_47F220'},
 100: {'mnemonic': 'builtin_64',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 100,
       'table_ea': 5635256,
       'target_ea': 4718928,
       'target_name': 'loc_480150'},
 101: {'mnemonic': 'builtin_65',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 101,
       'table_ea': 5635260,
       'target_ea': 4719136,
       'target_name': 'loc_480220'},
 102: {'mnemonic': 'builtin_66',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 102,
       'table_ea': 5635264,
       'target_ea': 4719296,
       'target_name': 'loc_4802C0'},
 103: {'mnemonic': 'builtin_67',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 103,
       'table_ea': 5635268,
       'target_ea': 4734064,
       'target_name': 'loc_483C70'},
 104: {'mnemonic': 'builtin_68',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 104,
       'table_ea': 5635272,
       'target_ea': 4725200,
       'target_name': 'loc_4819D0'},
 105: {'mnemonic': 'builtin_69',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 105,
       'table_ea': 5635276,
       'target_ea': 4715904,
       'target_name': 'loc_47F580'},
 106: {'mnemonic': 'builtin_6A',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 106,
       'table_ea': 5635280,
       'target_ea': 4701056,
       'target_name': 'loc_47BB80'},
 107: {'mnemonic': 'builtin_6B',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 107,
       'table_ea': 5635284,
       'target_ea': 4725280,
       'target_name': 'loc_481A20'},
 108: {'mnemonic': 'builtin_6C',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 108,
       'table_ea': 5635288,
       'target_ea': 4717296,
       'target_name': 'loc_47FAF0'},
 109: {'mnemonic': 'builtin_6D',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 109,
       'table_ea': 5635292,
       'target_ea': 4725408,
       'target_name': 'loc_481AA0'},
 110: {'mnemonic': 'builtin_6E',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 110,
       'table_ea': 5635296,
       'target_ea': 4725520,
       'target_name': 'loc_481B10'},
 111: {'mnemonic': 'builtin_6F',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 111,
       'table_ea': 5635300,
       'target_ea': 4725696,
       'target_name': 'loc_481BC0'},
 112: {'mnemonic': 'builtin_70',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 112,
       'table_ea': 5635304,
       'target_ea': 4726400,
       'target_name': 'loc_481E80'},
 113: {'mnemonic': 'builtin_71',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 113,
       'table_ea': 5635308,
       'target_ea': 4726128,
       'target_name': 'loc_481D70'},
 114: {'mnemonic': 'builtin_72',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 114,
       'table_ea': 5635312,
       'target_ea': 4726624,
       'target_name': 'loc_481F60'},
 115: {'mnemonic': 'builtin_73',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 115,
       'table_ea': 5635316,
       'target_ea': 4727008,
       'target_name': 'loc_4820E0'},
 116: {'mnemonic': 'builtin_74',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 116,
       'table_ea': 5635320,
       'target_ea': 4712016,
       'target_name': 'loc_47E650'},
 117: {'mnemonic': 'builtin_75',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 117,
       'table_ea': 5635324,
       'target_ea': 4745824,
       'target_name': 'sub_486A60'},
 118: {'mnemonic': 'builtin_76',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 118,
       'table_ea': 5635328,
       'target_ea': 4727472,
       'target_name': 'loc_4822B0'},
 119: {'mnemonic': 'builtin_77',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 119,
       'table_ea': 5635332,
       'target_ea': 4727520,
       'target_name': 'loc_4822E0'},
 120: {'mnemonic': 'builtin_78',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 120,
       'table_ea': 5635336,
       'target_ea': 4727600,
       'target_name': 'sub_482330'},
 121: {'mnemonic': 'builtin_79',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 121,
       'table_ea': 5635340,
       'target_ea': 4716576,
       'target_name': 'loc_47F820'},
 122: {'mnemonic': 'builtin_7A',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 122,
       'table_ea': 5635344,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 123: {'mnemonic': 'builtin_7B',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 123,
       'table_ea': 5635348,
       'target_ea': 4719424,
       'target_name': 'loc_480340'},
 124: {'mnemonic': 'builtin_7C',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 124,
       'table_ea': 5635352,
       'target_ea': 4712384,
       'target_name': 'loc_47E7C0'},
 125: {'mnemonic': 'builtin_7D',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 125,
       'table_ea': 5635356,
       'target_ea': 4700704,
       'target_name': 'loc_47BA20'},
 126: {'mnemonic': 'builtin_7E',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 126,
       'table_ea': 5635360,
       'target_ea': 4710448,
       'target_name': 'loc_47E030'},
 127: {'mnemonic': 'builtin_7F',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 127,
       'table_ea': 5635364,
       'target_ea': 4716464,
       'target_name': 'loc_47F7B0'},
 128: {'mnemonic': 'builtin_80',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 128,
       'table_ea': 5635368,
       'target_ea': 4733952,
       'target_name': 'loc_483C00'},
 129: {'mnemonic': 'builtin_81',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 129,
       'table_ea': 5635372,
       'target_ea': 4763216,
       'target_name': 'loc_48AE50'},
 130: {'mnemonic': 'builtin_82',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 130,
       'table_ea': 5635376,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 131: {'mnemonic': 'builtin_83',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 131,
       'table_ea': 5635380,
       'target_ea': 4719584,
       'target_name': 'loc_4803E0'},
 132: {'mnemonic': 'builtin_84',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 132,
       'table_ea': 5635384,
       'target_ea': 4758544,
       'target_name': 'loc_489C10'},
 133: {'mnemonic': 'builtin_85',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 133,
       'table_ea': 5635388,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 134: {'mnemonic': 'builtin_86',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 134,
       'table_ea': 5635392,
       'target_ea': 4701744,
       'target_name': 'loc_47BE30'},
 135: {'mnemonic': 'builtin_87',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 135,
       'table_ea': 5635396,
       'target_ea': 4708384,
       'target_name': 'loc_47D820'},
 136: {'mnemonic': 'builtin_88',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 136,
       'table_ea': 5635400,
       'target_ea': 4707504,
       'target_name': 'loc_47D4B0'},
 137: {'mnemonic': 'builtin_89',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 137,
       'table_ea': 5635404,
       'target_ea': 4709216,
       'target_name': 'loc_47DB60'},
 138: {'mnemonic': 'builtin_8A',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 138,
       'table_ea': 5635408,
       'target_ea': 4727616,
       'target_name': 'loc_482340'},
 139: {'mnemonic': 'builtin_8B',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 139,
       'table_ea': 5635412,
       'target_ea': 4727728,
       'target_name': 'loc_4823B0'},
 140: {'mnemonic': 'builtin_8C',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 140,
       'table_ea': 5635416,
       'target_ea': 4727840,
       'target_name': 'loc_482420'},
 141: {'mnemonic': 'builtin_8D',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 141,
       'table_ea': 5635420,
       'target_ea': 4727952,
       'target_name': 'loc_482490'},
 142: {'mnemonic': 'builtin_8E',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 142,
       'table_ea': 5635424,
       'target_ea': 4728048,
       'target_name': 'loc_4824F0'},
 143: {'mnemonic': 'builtin_8F',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 143,
       'table_ea': 5635428,
       'target_ea': 4728160,
       'target_name': 'loc_482560'},
 144: {'mnemonic': 'builtin_90',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 144,
       'table_ea': 5635432,
       'target_ea': 4728272,
       'target_name': 'loc_4825D0'},
 145: {'mnemonic': 'builtin_91',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 145,
       'table_ea': 5635436,
       'target_ea': 4728400,
       'target_name': 'loc_482650'},
 146: {'mnemonic': 'builtin_92',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 146,
       'table_ea': 5635440,
       'target_ea': 4728528,
       'target_name': 'loc_4826D0'},
 147: {'mnemonic': 'builtin_93',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 147,
       'table_ea': 5635444,
       'target_ea': 4748432,
       'target_name': 'loc_487490'},
 148: {'mnemonic': 'builtin_94',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 148,
       'table_ea': 5635448,
       'target_ea': 4729008,
       'target_name': 'loc_4828B0'},
 149: {'mnemonic': 'builtin_95',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 149,
       'table_ea': 5635452,
       'target_ea': 4729024,
       'target_name': 'loc_4828C0'},
 150: {'mnemonic': 'builtin_96',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 150,
       'table_ea': 5635456,
       'target_ea': 4729040,
       'target_name': 'loc_4828D0'},
 151: {'mnemonic': 'builtin_97',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 151,
       'table_ea': 5635460,
       'target_ea': 4729056,
       'target_name': 'loc_4828E0'},
 152: {'mnemonic': 'builtin_98',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 152,
       'table_ea': 5635464,
       'target_ea': 4729072,
       'target_name': 'loc_4828F0'},
 153: {'mnemonic': 'builtin_99',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 153,
       'table_ea': 5635468,
       'target_ea': 4729232,
       'target_name': 'loc_482990'},
 154: {'mnemonic': 'builtin_9A',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 154,
       'table_ea': 5635472,
       'target_ea': 4729552,
       'target_name': 'loc_482AD0'},
 155: {'mnemonic': 'builtin_9B',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 155,
       'table_ea': 5635476,
       'target_ea': 4729568,
       'target_name': 'loc_482AE0'},
 156: {'mnemonic': 'builtin_9C',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 156,
       'table_ea': 5635480,
       'target_ea': 4729584,
       'target_name': 'loc_482AF0'},
 157: {'mnemonic': 'builtin_9D',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 157,
       'table_ea': 5635484,
       'target_ea': 4729600,
       'target_name': 'loc_482B00'},
 158: {'mnemonic': 'builtin_9E',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 158,
       'table_ea': 5635488,
       'target_ea': 4729616,
       'target_name': 'loc_482B10'},
 159: {'mnemonic': 'builtin_9F',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 159,
       'table_ea': 5635492,
       'target_ea': 4729808,
       'target_name': 'loc_482BD0'},
 160: {'mnemonic': 'builtin_A0',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 160,
       'table_ea': 5635496,
       'target_ea': 4730048,
       'target_name': 'loc_482CC0'},
 161: {'mnemonic': 'builtin_A1',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 161,
       'table_ea': 5635500,
       'target_ea': 4540288,
       'target_name': 'loc_454780'},
 162: {'mnemonic': 'builtin_A2',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 162,
       'table_ea': 5635504,
       'target_ea': 4464704,
       'target_name': 'loc_442040'},
 163: {'mnemonic': 'builtin_A3',
       'semantic': 'second-level builtin/native handler; exact side effects remain runtime-specific',
       'subopcode': 163,
       'table_ea': 5635508,
       'target_ea': 4730272,
       'target_name': 'loc_482DA0'}}

OPCODES: Dict[int, OpcodeSpec] = {op: OpcodeSpec(**row) for op, row in _TOP_OPCODE_ROWS.items()}
BUILTINS: Dict[int, BuiltinSpec] = {op: BuiltinSpec(**row) for op, row in _LOW_OPCODE_ROWS.items()}

BINARY_AST_OPS = {
    0x2B: "+", 0x2D: "-", 0x2A: "*", 0x2F: "/", 0x25: "%",
    0xF0: "==", 0xED: "!=", 0x3E: ">", 0x3C: "<", 0xE1: ">=", 0xEC: "<=",
}
UNARY_AST_OPS = {0xF1: "-", 0x21: "!"}


def opcode_to_dict() -> Dict[str, Dict[str, Any]]:
    return {f"0x{op:02X}": dict(row) for op, row in _TOP_OPCODE_ROWS.items()}


def builtin_to_dict() -> Dict[str, Dict[str, Any]]:
    return {f"0x{op:02X}": dict(row) for op, row in _LOW_OPCODE_ROWS.items()}


def decode_opcode(buf: bytes, off: int) -> DecodedOpcode:
    """Decode one bytecode instruction according to the recovered opcode specs."""
    if off >= len(buf):
        return DecodedOpcode("eof", 0, {}, True, False, [])

    opcode = buf[off]

    # These are handled directly by sub_489410 before the funcs_48983F table.
    if opcode == 0x23:  # '#': end of program body
        return DecodedOpcode(
            mnemonic="end_program",
            length=1,
            operands={"semantic": "main-loop special: finish current program"},
            terminal=True,
            known=True,
            edges=[DecodedEdge("end_program", None)],
        )
    if opcode == 0x7C:  # '|': yield / suspend current program
        return DecodedOpcode(
            mnemonic="yield_program",
            length=1,
            operands={"semantic": "main-loop special: save current PC and yield/suspend"},
            terminal=True,
            known=True,
            edges=[DecodedEdge("yield", None)],
        )

    spec = OPCODES.get(opcode)
    if spec is None:
        return DecodedOpcode(
            mnemonic=f"unknown_{opcode:02X}",
            length=1,
            operands={"opcode_char": safe_chr(opcode)},
            terminal=False,
            known=False,
            edges=[],
        )

    def have(n: int) -> bool:
        return off + n <= len(buf)

    def base_operands() -> Dict[str, Any]:
        return {
            "handler": spec.handler_name,
            "handler_ea": f"0x{spec.handler_ea:08X}",
            "semantic": spec.semantic,
        }

    fmt = spec.format
    operands = base_operands()
    edges: List[DecodedEdge] = []
    length = 1
    terminal = False
    known = True
    mnemonic = spec.mnemonic

    try:
        if fmt == "none":
            length = 1

        elif fmt == "trap":
            length = 1
            known = False
            terminal = True
            edges.append(DecodedEdge("trap", None, "unknown opcode trap in recovered dispatcher"))

        elif fmt == "u8":
            if not have(2):
                raise ValueError("truncated u8 operand")
            length = 2
            operands["value"] = buf[off + 1]

        elif fmt == "u16":
            if not have(3):
                raise ValueError("truncated u16 operand")
            length = 3
            operands["value"] = _u16(buf, off + 1)

        elif fmt == "program_i16":
            if not have(3):
                raise ValueError("truncated program operand")
            length = 3
            idx = _s16(buf, off + 1)
            operands["program_index"] = idx
            operands["program_index_u16"] = _u16(buf, off + 1)

        elif fmt == "rel16":
            if not have(3):
                raise ValueError("truncated rel16 operand")
            length = 3
            rel = _s16(buf, off + 1)
            target = off + 1 + rel
            operands.update({"rel": rel, "target": target, "target_file": target + CODE_FILE_OFFSET})
            edges.append(DecodedEdge("jmp", target))
            terminal = True

        elif fmt == "rel32":
            if not have(5):
                raise ValueError("truncated rel32 operand")
            length = 5
            rel = _s32(buf, off + 1)
            target = off + 1 + rel
            operands.update({"rel": rel, "target": target, "target_file": target + CODE_FILE_OFFSET})
            edges.append(DecodedEdge("jmp", target))
            terminal = True

        elif fmt == "jfalse_rel16":
            if not have(3):
                raise ValueError("truncated conditional rel16 operand")
            length = 3
            rel = _s16(buf, off + 1)
            target = off + 1 + rel
            fallthrough = off + length
            operands.update({
                "rel": rel,
                "target": target,
                "target_file": target + CODE_FILE_OFFSET,
                "fallthrough": fallthrough,
                "fallthrough_file": fallthrough + CODE_FILE_OFFSET,
            })
            edges.append(DecodedEdge("jfalse", target, "condition is false/zero"))
            edges.append(DecodedEdge("jtrue_fallthrough", fallthrough, "condition is true/non-zero"))

        elif fmt == "jfalse_rel32":
            if not have(5):
                raise ValueError("truncated conditional rel32 operand")
            length = 5
            rel = _s32(buf, off + 1)
            target = off + 1 + rel
            fallthrough = off + length
            operands.update({
                "rel": rel,
                "target": target,
                "target_file": target + CODE_FILE_OFFSET,
                "fallthrough": fallthrough,
                "fallthrough_file": fallthrough + CODE_FILE_OFFSET,
            })
            edges.append(DecodedEdge("jfalse", target, "condition is false/zero"))
            edges.append(DecodedEdge("jtrue_fallthrough", fallthrough, "condition is true/non-zero"))

        elif fmt == "logical_or_rel16":
            if not have(3):
                raise ValueError("truncated logical OR rel16 operand")
            length = 3
            rel = _s16(buf, off + 1)
            target = off + 1 + rel
            fallthrough = off + length
            operands.update({
                "rel": rel,
                "target": target,
                "target_file": target + CODE_FILE_OFFSET,
                "fallthrough": fallthrough,
                "fallthrough_file": fallthrough + CODE_FILE_OFFSET,
            })
            edges.append(DecodedEdge("jtrue", target, "short-circuit OR: true jumps with result=true"))
            edges.append(DecodedEdge("jfalse_fallthrough", fallthrough, "false falls through to evaluate RHS"))

        elif fmt == "logical_and_rel16":
            if not have(3):
                raise ValueError("truncated logical AND rel16 operand")
            length = 3
            rel = _s16(buf, off + 1)
            target = off + 1 + rel
            fallthrough = off + length
            operands.update({
                "rel": rel,
                "target": target,
                "target_file": target + CODE_FILE_OFFSET,
                "fallthrough": fallthrough,
                "fallthrough_file": fallthrough + CODE_FILE_OFFSET,
            })
            edges.append(DecodedEdge("jfalse", target, "short-circuit AND: false jumps with result=false"))
            edges.append(DecodedEdge("jtrue_fallthrough", fallthrough, "true falls through to evaluate RHS"))

        elif fmt == "call_rel32":
            if not have(5):
                raise ValueError("truncated call rel32 operand")
            length = 5
            rel = _s32(buf, off + 1)
            target = off + 1 + rel
            ret = off + length
            operands.update({"rel": rel, "target": target, "target_file": target + CODE_FILE_OFFSET, "return": ret, "return_file": ret + CODE_FILE_OFFSET})
            edges.append(DecodedEdge("call_rel32", target))
            edges.append(DecodedEdge("call_return", ret))

        elif fmt == "data_ref":
            if not have(6):
                raise ValueError("truncated data_ref operand")
            length = 6
            typ = buf[off + 1]
            ptr = _u32(buf, off + 2)
            operands.update({"type": typ, "type_name": type_name(typ), "data_offset": ptr})

        elif fmt == "typed_imm32":
            if not have(6):
                raise ValueError("truncated typed imm32 operand")
            length = 6
            typ = buf[off + 1]
            raw = _u32(buf, off + 2)
            operands.update({"type": typ, "type_name": type_name(typ), "value_u32": raw, "value_i32": _s32(buf, off + 2)})
            if typ == 32:
                operands["value_float"] = _f32_from_u32(raw)

        elif fmt == "typed_imm_u16":
            if not have(4):
                raise ValueError("truncated typed imm16 operand")
            length = 4
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "value": _u16(buf, off + 2)})

        elif fmt == "typed_imm_i8":
            if not have(3):
                raise ValueError("truncated typed imm8 operand")
            length = 3
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "value": _s8(buf[off + 2]), "value_u8": buf[off + 2]})

        elif fmt == "inline_span":
            if not have(7):
                raise ValueError("truncated inline span operand")
            length = 7
            operands.update({"data_offset": _u32(buf, off + 1), "length": _u16(buf, off + 5)})

        elif fmt == "typed_span_ref":
            if not have(10):
                raise ValueError("truncated typed span ref operand")
            length = 10
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "data_offset": _u32(buf, off + 2), "length": _u32(buf, off + 6)})

        elif fmt == "typed_span_inline":
            if not have(10):
                raise ValueError("truncated typed span inline operand")
            length = 10
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "data_offset": _u32(buf, off + 2), "length": _u32(buf, off + 6)})

        elif fmt == "array_abs":
            if not have(16):
                raise ValueError("truncated absolute array operand")
            length = 16
            typ = buf[off + 1]
            operands.update({
                "type": typ,
                "type_name": type_name(typ),
                "element_size": _u16(buf, off + 2),
                "base": _u32(buf, off + 4),
                "span": _u32(buf, off + 8),
                "count": _s32(buf, off + 12),
            })

        elif fmt == "array2_checked":
            if not have(8):
                raise ValueError("truncated relative array checked operand")
            length = 8
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "element_size": _u16(buf, off + 2), "count": _s32(buf, off + 4)})

        elif fmt == "array2":
            if not have(4):
                raise ValueError("truncated relative array operand")
            length = 4
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "element_size": _u16(buf, off + 2)})

        elif fmt == "slice_offset_ref":
            if not have(4):
                raise ValueError("truncated slice offset operand")
            typ = buf[off + 1]
            length = 8 if typ == 48 else 4
            if not have(length):
                raise ValueError("truncated slice descriptor extension")
            operands.update({"type": typ, "type_name": type_name(typ), "offset": _u16(buf, off + 2)})
            if typ == 48:
                operands["length"] = _u32(buf, off + 4)

        elif fmt == "slice_offset_span":
            if not have(8):
                raise ValueError("truncated slice span operand")
            length = 8
            typ = buf[off + 1]
            operands.update({"type": typ, "type_name": type_name(typ), "offset": _u16(buf, off + 2), "length": _u32(buf, off + 4)})

        elif fmt == "prologue":
            if not have(2):
                raise ValueError("truncated prologue count")
            signed_count = _s8(buf[off + 1])
            count = abs(signed_count)
            length = 2 + count * 5
            if not have(length):
                raise ValueError("truncated prologue descriptors")
            descriptors = []
            p = off + 2
            for _ in range(count):
                typ = buf[p]
                dst = _u32(buf, p + 1)
                descriptors.append({"type": typ, "type_name": type_name(typ), "data_offset": dst})
                p += 5
            operands.update({"signed_count": signed_count, "descriptor_count": count, "descriptors": descriptors})

        elif fmt == "builtin":
            if not have(2):
                raise ValueError("truncated builtin subopcode")
            length = 2
            sub = buf[off + 1]
            builtin = BUILTINS.get(sub)
            if builtin is None:
                operands.update({"subopcode": sub, "subopcode_hex": f"0x{sub:02X}"})
                mnemonic = f"builtin_{sub:02X}"
                known = False
            else:
                mnemonic = builtin.mnemonic
                operands.update({
                    "subopcode": sub,
                    "subopcode_hex": builtin.subopcode_hex,
                    "target_ea": f"0x{builtin.target_ea:08X}",
                    "target_name": builtin.target_name,
                    "builtin_semantic": builtin.semantic,
                })

        else:
            # Spec exists but the decoder has not been taught the operand layout yet.
            length = 1
            known = False
            operands["undecoded_format"] = fmt

    except (struct.error, ValueError) as exc:
        length = 1
        known = False
        operands["decode_error"] = str(exc)
        mnemonic = f"{spec.mnemonic}_truncated"
        edges = []
        terminal = False

    if opcode in (0x72, 0x74):
        terminal = True
        edges.append(DecodedEdge("return", None, spec.semantic))
    elif opcode == 0x48:
        terminal = True
        edges.append(DecodedEdge("halt", None, spec.semantic))
    elif opcode == 0x67:
        terminal = True
        edges.append(DecodedEdge("error", None, spec.semantic))

    return DecodedOpcode(mnemonic, length, operands, terminal, known, edges)
