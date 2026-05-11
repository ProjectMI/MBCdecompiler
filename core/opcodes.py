from __future__ import annotations

"""Recovered MBC opcode model.

This module owns recovered opcode tables and operand decoding only. The
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

from .common import (
    CODE_FILE_OFFSET,
    TYPE_BASE_NAMES,
    TYPE_NAMES,
    f32_from_u32 as _f32_from_u32,
    safe_chr,
    s8 as _s8,
    s16 as _s16,
    s32 as _s32,
    type_name,
    u16 as _u16,
    u32 as _u32,
)


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
    confidence: str = "unverified"

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
       'format': 'import_stub_u32',
       'handler_ea': 4691040,
       'handler_name': 'sub_479460',
       'mnemonic': 'import_stub_u32',
       'opcode': 103,
       'semantic': '5-byte import/unlinked-call stub: opcode 0x67 plus a u32 payload; terminal error path until linker/import resolution patches it',
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
_LOW_OPCODE_ROWS: Dict[int, Dict[str, Any]] = {0: {'confidence': 'exact',
     'mnemonic': 'debug_print_float',
     'semantic': 'debug UI fatal-print of a float argument; falls through to string-print path in the original switch',
     'subopcode': 0,
     'table_ea': 5634856,
     'target_ea': 4700464,
     'target_name': 'loc_47B930'},
 1: {'confidence': 'exact',
     'mnemonic': 'debug_print_float_alias',
     'semantic': 'alias of debug_print_float; same handler target as subopcode 0',
     'subopcode': 1,
     'table_ea': 5634860,
     'target_ea': 4700464,
     'target_name': 'loc_47B930'},
 2: {'confidence': 'exact',
     'mnemonic': 'print_string_or_exit',
     'semantic': 'prints string argument; with zero arguments tears down and exits the interpreter process',
     'subopcode': 2,
     'table_ea': 5634864,
     'target_ea': 4700544,
     'target_name': 'loc_47B980'},
 3: {'confidence': 'exact',
     'mnemonic': 'sin',
     'semantic': 'pushes sin(float_arg)',
     'subopcode': 3,
     'table_ea': 5634868,
     'target_ea': 4700608,
     'target_name': 'loc_47B9C0'},
 4: {'confidence': 'exact',
     'mnemonic': 'cos',
     'semantic': 'pushes cos(float_arg)',
     'subopcode': 4,
     'table_ea': 5634872,
     'target_ea': 4700656,
     'target_name': 'loc_47B9F0'},
 5: {'confidence': 'exact',
     'mnemonic': 'abs_float',
     'semantic': 'pushes absolute value of float_arg',
     'subopcode': 5,
     'table_ea': 5634876,
     'target_ea': 4700912,
     'target_name': 'loc_47BAF0'},
 6: {'confidence': 'exact',
     'mnemonic': 'abs_int',
     'semantic': 'pushes absolute value of int_arg',
     'subopcode': 6,
     'table_ea': 5634880,
     'target_ea': 4700816,
     'target_name': 'loc_47BA90'},
 7: {'confidence': 'exact',
     'mnemonic': 'atan2',
     'semantic': 'pushes atan2(y, x) from two float arguments',
     'subopcode': 7,
     'table_ea': 5634884,
     'target_ea': 4700752,
     'target_name': 'loc_47BA50'},
 8: {'confidence': 'recovered',
     'mnemonic': 'push_vm_tick',
     'semantic': 'pushes dword_3E5E1B8, the VM/interpreter loop tick counter incremented by the dispatcher',
     'subopcode': 8,
     'table_ea': 5634888,
     'target_ea': 4700960,
     'target_name': 'loc_47BB20'},
 9: {'confidence': 'exact',
     'mnemonic': 'sscanf',
     'semantic': 'sscanf wrapper; argument count controls the number and kind of destination pointers',
     'subopcode': 9,
     'table_ea': 5634892,
     'target_ea': 4715952,
     'target_name': 'loc_47F5B0'},
 10: {'confidence': 'partial',
      'mnemonic': 'window_api',
      'semantic': 'ASM-verified nested window/UI dispatcher: first integer selector 0..0x4F is remapped through '
                  'byte_481980 into jpt_4804BA; visible submodes include create/destroy/query dimensions, event/status '
                  'extraction and window object operations. Many callee bodies are engine UI internals, so submode '
                  'names remain partial.',
      'subopcode': 10,
      'table_ea': 5634896,
      'target_ea': 4719760,
      'target_name': 'loc_480490'},
 11: {'confidence': 'recovered',
      'mnemonic': 'pack_rgb24',
      'semantic': 'pops three integer byte/color components and pushes a packed 24-bit color value',
      'subopcode': 11,
      'table_ea': 5634900,
      'target_ea': 4701600,
      'target_name': 'loc_47BDA0'},
 12: {'confidence': 'exact',
      'mnemonic': 'sqrt_abs_float',
      'semantic': 'pushes sqrt(abs(float_arg))',
      'subopcode': 12,
      'table_ea': 5634904,
      'target_ea': 4701968,
      'target_name': 'loc_47BF10'},
 13: {'confidence': 'exact',
      'mnemonic': 'push_runtime_handle',
      'semantic': 'pushes runtime/global handle by selector: no args or selector 0 => dword_4A46338; selector 1 => '
                  'dword_4AE50E4; any other selector => dword_4ABF1AC',
      'subopcode': 13,
      'table_ea': 5634908,
      'target_ea': 4702032,
      'target_name': 'loc_47BF50'},
 14: {'confidence': 'exact',
      'mnemonic': 'push_runtime_flag_byte',
      'semantic': 'pops index and pushes sign-extended byte_4ABF0A8[index] when dword_4A8B62C is nonzero; otherwise '
                  'pushes 0',
      'subopcode': 14,
      'table_ea': 5634912,
      'target_ea': 4702112,
      'target_name': 'loc_47BFA0'},
 15: {'confidence': 'recovered',
      'mnemonic': 'alloc_span',
      'semantic': 'allocates a VM memory span of the requested byte size and pushes a slice descriptor for it',
      'subopcode': 15,
      'table_ea': 5634916,
      'target_ea': 4726048,
      'target_name': 'loc_481D20'},
 16: {'confidence': 'recovered',
      'mnemonic': 'ffprc_load',
      'semantic': 'ffprc_load: loads/starts a process by script name plus optional parameter and pushes resulting '
                  'process id/status',
      'subopcode': 16,
      'table_ea': 5634920,
      'target_ea': 4702160,
      'target_name': 'loc_47BFD0'},
 17: {'confidence': 'recovered',
      'mnemonic': 'ffprc_unload',
      'semantic': 'ffprc_unload: validates process id and unloads/stops it, rejecting the current/_main process',
      'subopcode': 17,
      'table_ea': 5634924,
      'target_ea': 4702256,
      'target_name': 'loc_47C030'},
 18: {'confidence': 'recovered',
      'mnemonic': 'ffprc_link',
      'semantic': 'ffprc_link: links/resolves a process/script name and pushes the resolved id/status',
      'subopcode': 18,
      'table_ea': 5634928,
      'target_ea': 4702400,
      'target_name': 'loc_47C0C0'},
 19: {'confidence': 'recovered',
      'mnemonic': 'ffprc_state',
      'semantic': 'validates process id and pushes its state field dword_9C65B4[210*pid]',
      'subopcode': 19,
      'table_ea': 5634932,
      'target_ea': 4702544,
      'target_name': 'loc_47C150'},
 20: {'confidence': 'recovered',
      'mnemonic': 'send_to_process_id',
      'semantic': 'validates destination process id (pid <= 0xFFFF, dword_9C6624[210*pid] == pid, state >= 0, pid > 0) '
                  'and tailcalls sub_47C300(pid); underflow/invalid path tailcalls sub_47AE00(-1)',
      'subopcode': 20,
      'table_ea': 5634936,
      'target_ea': 4703600,
      'target_name': 'loc_47C570'},
 21: {'confidence': 'recovered',
      'mnemonic': 'send_to_process_zero',
      'semantic': 'validates that at least one payload argument remains, then tailcalls sub_47C300(0); underflow path '
                  'tailcalls sub_47AE00(-1)',
      'subopcode': 21,
      'table_ea': 5634940,
      'target_ea': 4703680,
      'target_name': 'loc_47C5C0'},
 22: {'confidence': 'recovered',
      'mnemonic': 'last_process_result',
      'semantic': 'pushes dword_9C643C, the last process/native result slot',
      'subopcode': 22,
      'table_ea': 5634944,
      'target_ea': 4703712,
      'target_name': 'loc_47C5E0'},
 23: {'confidence': 'exact',
      'mnemonic': 'arg_count',
      'semantic': 'pushes dword_86232C, the current builtin argument count',
      'subopcode': 23,
      'table_ea': 5634948,
      'target_ea': 4703808,
      'target_name': 'loc_47C640'},
 24: {'confidence': 'recovered',
      'mnemonic': 'current_process_state',
      'semantic': 'pushes state of the current process dword_9C65B4[210*current_pid]',
      'subopcode': 24,
      'table_ea': 5634952,
      'target_ea': 4703904,
      'target_name': 'loc_47C6A0'},
 25: {'confidence': 'exact',
      'mnemonic': 'push_zero',
      'semantic': 'pushes integer zero; same target as subopcode 0x1B',
      'subopcode': 25,
      'table_ea': 5634956,
      'target_ea': 4704128,
      'target_name': 'loc_47C780'},
 26: {'confidence': 'partial',
      'mnemonic': 'send_message_marshaled',
      'semantic': 'ASM-verified VM message-send marshaller: consumes message/region/count arguments, validates VM '
                  'buffers and flags, then invokes the runtime send path. Exact high-level message API names are not '
                  'present in the asm slice.',
      'subopcode': 26,
      'table_ea': 5634960,
      'target_ea': 4704224,
      'target_name': 'loc_47C7E0'},
 27: {'confidence': 'exact',
      'mnemonic': 'push_zero_alias',
      'semantic': 'alias of push_zero; same target as subopcode 0x19',
      'subopcode': 27,
      'table_ea': 5634964,
      'target_ea': 4704128,
      'target_name': 'loc_47C780'},
 28: {'confidence': 'recovered',
      'mnemonic': 'ffprc_id',
      'semantic': 'ffprc_id: resolves a process id by script/process name, with optional parent/current-process lookup '
                  'path',
      'subopcode': 28,
      'table_ea': 5634968,
      'target_ea': 4702624,
      'target_name': 'loc_47C1A0'},
 29: {'confidence': 'exact',
      'mnemonic': 'push_context_id_or_zero',
      'semantic': 'pops selector; when no VM error is pending, selector < 0 pushes dword_6FD10C, selector >= 0 pushes '
                  '0',
      'subopcode': 29,
      'table_ea': 5634972,
      'target_ea': 4706960,
      'target_name': 'loc_47D290'},
 30: {'confidence': 'recovered',
      'mnemonic': 'lookup_process_by_name',
      'semantic': 'reads process/name string arguments, optionally consumes an integer selector, calls '
                  'sub_473240(name, selector), and pushes the returned process/runtime id as an int stack slot',
      'subopcode': 30,
      'table_ea': 5634976,
      'target_ea': 4707008,
      'target_name': 'loc_47D2C0'},
 31: {'confidence': 'external',
      'mnemonic': 'external_runtime_update_473730',
      'semantic': 'tail-jumps to sub_473730; the target body is not included in the supplied asm slice, so only '
                  'dispatch/target identity is verified',
      'subopcode': 31,
      'table_ea': 5634980,
      'target_ea': 4684816,
      'target_name': 'loc_477C10'},
 32: {'confidence': 'exact',
      'mnemonic': 'push_current_flags_mask_4',
      'semantic': 'pushes ([dword_48C06E4 + 0x94] & 4) from the current runtime/context structure',
      'subopcode': 32,
      'table_ea': 5634984,
      'target_ea': 4707152,
      'target_name': 'loc_47D350'},
 33: {'confidence': 'partial',
      'mnemonic': 'receive_message_marshaled',
      'semantic': 'ASM-verified VM receive/unmarshal path: validates requested receive fields and writes/pushes '
                  'message data from the current runtime message context. Full message schema remains engine-specific.',
      'subopcode': 33,
      'table_ea': 5634988,
      'target_ea': 4705056,
      'target_name': 'loc_47CB20'},
 34: {'confidence': 'recovered',
      'mnemonic': 'strcpy_checked',
      'semantic': 'bounded string copy into a VM slice; optional third argument limits copied length and destination '
                  'slice is resized/checked',
      'subopcode': 34,
      'table_ea': 5634992,
      'target_ea': 4707248,
      'target_name': 'loc_47D3B0'},
 35: {'confidence': 'exact',
      'mnemonic': 'strcat_checked',
      'semantic': 'bounded strcat into a VM slice; optional third argument is maximum allowed resulting size',
      'subopcode': 35,
      'table_ea': 5634996,
      'target_ea': 4707776,
      'target_name': 'loc_47D5C0'},
 36: {'confidence': 'exact',
      'mnemonic': 'strlen_checked',
      'semantic': 'ffstrlen: pushes string length; optional max buffer length bounds the scan',
      'subopcode': 36,
      'table_ea': 5635000,
      'target_ea': 4708496,
      'target_name': 'loc_47D890'},
 37: {'confidence': 'exact',
      'mnemonic': 'strcmp',
      'semantic': 'pushes strcmp(a, b)',
      'subopcode': 37,
      'table_ea': 5635004,
      'target_ea': 4708800,
      'target_name': 'loc_47D9C0'},
 38: {'confidence': 'partial',
      'mnemonic': 'log_event_dispatch',
      'semantic': 'ASM-verified logging/statistics dispatcher gated by current stack/context flags; consumes mode and '
                  'payload arguments and routes into several runtime log/event sinks. Sink semantics remain '
                  'engine-specific.',
      'subopcode': 38,
      'table_ea': 5635008,
      'target_ea': 4709344,
      'target_name': 'loc_47DBE0'},
 39: {'confidence': 'recovered',
      'mnemonic': 'current_process_id',
      'semantic': 'pushes current process id dword_9C6624[210*current_pid]',
      'subopcode': 39,
      'table_ea': 5635012,
      'target_ea': 4704016,
      'target_name': 'loc_47C710'},
 40: {'confidence': 'recovered',
      'mnemonic': 'file_create',
      'semantic': 'creates/truncates a file via chmod/open and pushes file descriptor/status',
      'subopcode': 40,
      'table_ea': 5635016,
      'target_ea': 4711680,
      'target_name': 'loc_47E500'},
 41: {'confidence': 'recovered',
      'mnemonic': 'file_open',
      'semantic': 'opens a file via open/sopen and pushes file descriptor/status',
      'subopcode': 41,
      'table_ea': 5635020,
      'target_ea': 4711824,
      'target_name': 'loc_47E590'},
 42: {'confidence': 'exact',
      'mnemonic': 'file_close',
      'semantic': 'closes a file descriptor and pushes/records close result',
      'subopcode': 42,
      'table_ea': 5635024,
      'target_ea': 4712096,
      'target_name': 'loc_47E6A0'},
 43: {'confidence': 'exact',
      'mnemonic': 'file_write',
      'semantic': 'writes a VM buffer to a file descriptor',
      'subopcode': 43,
      'table_ea': 5635028,
      'target_ea': 4712144,
      'target_name': 'loc_47E6D0'},
 44: {'confidence': 'exact',
      'mnemonic': 'file_read',
      'semantic': 'reads from a file descriptor into a VM buffer',
      'subopcode': 44,
      'table_ea': 5635032,
      'target_ea': 4712272,
      'target_name': 'loc_47E750'},
 45: {'confidence': 'exact',
      'mnemonic': 'identity_int',
      'semantic': 'pops an integer and pushes the same integer value',
      'subopcode': 45,
      'table_ea': 5635036,
      'target_ea': 4713056,
      'target_name': 'loc_47EA60'},
 46: {'confidence': 'exact',
      'mnemonic': 'identity_float',
      'semantic': 'pops a float and pushes the same float value',
      'subopcode': 46,
      'table_ea': 5635040,
      'target_ea': 4713152,
      'target_name': 'loc_47EAC0'},
 47: {'confidence': 'recovered',
      'mnemonic': 'object_create',
      'semantic': 'creates an engine object from name/resource/type arguments via sub_4B9430, pushes the object '
                  'handle, registers it as handle type 4, and initializes object flags +0x274..+0x277 from '
                  'dword_55FF98/55FFC4/55FFF0/56001C',
      'subopcode': 47,
      'table_ea': 5635044,
      'target_ea': 4713280,
      'target_name': 'loc_47EB40'},
 48: {'confidence': 'recovered',
      'mnemonic': 'object_set_pos_xyz',
      'semantic': 'sets object fields +0x08/+0x0C/+0x10 to x/y/z floats after resolving handle via sub_494C60; '
                  'optional extra args call sub_49EA40 and may update field +0x20',
      'subopcode': 48,
      'table_ea': 5635048,
      'target_ea': 4713552,
      'target_name': 'loc_47EC50'},
 49: {'confidence': 'exact',
      'mnemonic': 'object_add_pos_xyz',
      'semantic': 'adds x/y/z float deltas to object fields +0x08/+0x0C/+0x10 after resolving handle via sub_494C60',
      'subopcode': 49,
      'table_ea': 5635052,
      'target_ea': 4714656,
      'target_name': 'loc_47F0A0'},
 50: {'confidence': 'recovered',
      'mnemonic': 'view_set_pos_xyz',
      'semantic': 'pops selector/handle plus three floats and calls sub_4AB350(selector, x/y/z as stack floats); exact '
                  'engine meaning of the selector is external',
      'subopcode': 50,
      'table_ea': 5635056,
      'target_ea': 4714784,
      'target_name': 'loc_47F120'},
 51: {'confidence': 'recovered',
      'mnemonic': 'view_set_z',
      'semantic': 'pops selector/handle and one float, then calls sub_4AB350(selector, 0, 0, z-like component)',
      'subopcode': 51,
      'table_ea': 5635060,
      'target_ea': 4714880,
      'target_name': 'loc_47F180'},
 52: {'confidence': 'exact',
      'mnemonic': 'object_set_vec14_xyz',
      'semantic': 'sets object fields +0x14/+0x18/+0x1C to three float arguments after resolving handle via sub_494C60',
      'subopcode': 52,
      'table_ea': 5635064,
      'target_ea': 4714160,
      'target_name': 'loc_47EEB0'},
 53: {'confidence': 'recovered',
      'mnemonic': 'global_vector_set',
      'semantic': 'pops selector/handle plus three floats and calls engine global-vector setter sub_4A31E0',
      'subopcode': 53,
      'table_ea': 5635068,
      'target_ea': 4714944,
      'target_name': 'loc_47F1C0'},
 54: {'confidence': 'recovered',
      'mnemonic': 'object_get_x',
      'semantic': 'pushes first float component of object position/vector via sub_494B20(handle)',
      'subopcode': 54,
      'table_ea': 5635072,
      'target_ea': 4713728,
      'target_name': 'loc_47ED00'},
 55: {'confidence': 'recovered',
      'mnemonic': 'object_get_y',
      'semantic': 'pushes second float component of object position/vector via sub_494B20(handle)',
      'subopcode': 55,
      'table_ea': 5635076,
      'target_ea': 4713792,
      'target_name': 'loc_47ED40'},
 56: {'confidence': 'recovered',
      'mnemonic': 'object_get_z',
      'semantic': 'pushes third float component of object position/vector via sub_494B20(handle)',
      'subopcode': 56,
      'table_ea': 5635080,
      'target_ea': 4713856,
      'target_name': 'loc_47ED80'},
 57: {'confidence': 'exact',
      'mnemonic': 'object_get_vec14_x',
      'semantic': 'pushes float [sub_494BC0(handle)+0x00], or 0.0 when the resolved vector pointer is null',
      'subopcode': 57,
      'table_ea': 5635084,
      'target_ea': 4713920,
      'target_name': 'loc_47EDC0'},
 58: {'confidence': 'exact',
      'mnemonic': 'object_get_vec14_y',
      'semantic': 'pushes float [sub_494BC0(handle)+0x04], or 0.0 when the resolved vector pointer is null',
      'subopcode': 58,
      'table_ea': 5635088,
      'target_ea': 4713984,
      'target_name': 'loc_47EE00'},
 59: {'confidence': 'exact',
      'mnemonic': 'object_get_vec14_z',
      'semantic': 'pushes float [sub_494BC0(handle)+0x08], or 0.0 when the resolved vector pointer is null',
      'subopcode': 59,
      'table_ea': 5635092,
      'target_ea': 4714048,
      'target_name': 'loc_47EE40'},
 60: {'confidence': 'recovered',
      'mnemonic': 'object_delete_type0',
      'semantic': 'if handle != -1 calls sub_498E70(handle), then unregisters/clears the handle via sub_477910(handle, '
                  'type 0, value 0)',
      'subopcode': 60,
      'table_ea': 5635096,
      'target_ea': 4714112,
      'target_name': 'loc_47EE80'},
 61: {'confidence': 'partial',
      'mnemonic': 'text_api',
      'semantic': 'ASM-verified text object API: supports direct text-property updates and text object creation '
                  'through sub_4A2D20; validates window/text handles and writes text fields +0x6DA8..+0x6DB4. Some '
                  'engine text internals remain partial.',
      'subopcode': 61,
      'table_ea': 5635100,
      'target_ea': 4715120,
      'target_name': 'loc_47F270'},
 62: {'confidence': 'recovered',
      'mnemonic': 'object_release_type4',
      'semantic': 'if handle >= 0 calls sub_497060(handle), then unregisters/clears the handle via sub_477910(handle, '
                  'type 4, value 0)',
      'subopcode': 62,
      'table_ea': 5635104,
      'target_ea': 4715504,
      'target_name': 'loc_47F3F0'},
 63: {'confidence': 'recovered',
      'mnemonic': 'text_color',
      'semantic': 'sets text object color field (+40); errors on wrong text handle',
      'subopcode': 63,
      'table_ea': 5635108,
      'target_ea': 4715552,
      'target_name': 'loc_47F420'},
 64: {'confidence': 'exact',
      'mnemonic': 'push_runtime_slot',
      'semantic': 'pushes dword_474C218[dword_4A2AB30] and increments dword_4A2AB30 only when the value is nonzero',
      'subopcode': 64,
      'table_ea': 5635112,
      'target_ea': 4713168,
      'target_name': 'loc_47EAD0'},
 65: {'confidence': 'exact',
      'mnemonic': 'file_seek',
      'semantic': 'lseek(fd, offset, whence) wrapper',
      'subopcode': 65,
      'table_ea': 5635116,
      'target_ea': 4712560,
      'target_name': 'loc_47E870'},
 66: {'confidence': 'exact',
      'mnemonic': 'file_length',
      'semantic': 'pushes filelength(fd)',
      'subopcode': 66,
      'table_ea': 5635120,
      'target_ea': 4712640,
      'target_name': 'loc_47E8C0'},
 67: {'confidence': 'exact',
      'mnemonic': 'sprintf',
      'semantic': 'ffsprintf: formats into a destination VM string buffer using vsprintf-style varargs',
      'subopcode': 67,
      'table_ea': 5635124,
      'target_ea': 4716320,
      'target_name': 'loc_47F720'},
 68: {'confidence': 'exact',
      'mnemonic': 'file_rename',
      'semantic': 'rename(old, new) wrapper',
      'subopcode': 68,
      'table_ea': 5635128,
      'target_ea': 4712944,
      'target_name': 'loc_47E9F0'},
 69: {'confidence': 'exact',
      'mnemonic': 'file_remove',
      'semantic': 'remove(path) wrapper',
      'subopcode': 69,
      'table_ea': 5635132,
      'target_ea': 4712896,
      'target_name': 'loc_47E9C0'},
 70: {'confidence': 'recovered',
      'mnemonic': 'file_stat_time_field',
      'semantic': 'calls _fstat64i32(fd, &stat) and pushes the recovered time dword read from the local stat buffer; '
                  'exact stat member name depends on MSVC struct layout naming in the asm export',
      'subopcode': 70,
      'table_ea': 5635136,
      'target_ea': 4712672,
      'target_name': 'loc_47E8E0'},
 71: {'confidence': 'exact',
      'mnemonic': 'file_truncate',
      'semantic': 'chsize(fd, size) wrapper',
      'subopcode': 71,
      'table_ea': 5635140,
      'target_ea': 4712752,
      'target_name': 'loc_47E930'},
 72: {'confidence': 'exact',
      'mnemonic': 'file_set_time',
      'semantic': 'sets file access/modify time using futime(fd, {time,time})',
      'subopcode': 72,
      'table_ea': 5635144,
      'target_ea': 4712800,
      'target_name': 'loc_47E960'},
 73: {'confidence': 'recovered',
      'mnemonic': 'sprite_create_or_update',
      'semantic': 'with two args calls sub_49E880(a,b) and pushes 0; otherwise creates a sprite/resource through '
                  'sub_4A79F0, pushes the new handle, and registers it as handle type 5',
      'subopcode': 73,
      'table_ea': 5635148,
      'target_ea': 4715632,
      'target_name': 'loc_47F470'},
 74: {'confidence': 'recovered',
      'mnemonic': 'file_lookup_476310',
      'semantic': 'calls sub_476310(path) for a VM string path and pushes dword_3E6AA80 result/status',
      'subopcode': 74,
      'table_ea': 5635152,
      'target_ea': 4713008,
      'target_name': 'loc_47EA30'},
 75: {'confidence': 'exact',
      'mnemonic': 'stricmp',
      'semantic': 'pushes case-insensitive strcmp/stricmp result',
      'subopcode': 75,
      'table_ea': 5635156,
      'target_ea': 4708960,
      'target_name': 'loc_47DA60'},
 76: {'confidence': 'exact',
      'mnemonic': 'strncmp',
      'semantic': 'pushes strncmp(a, b, n) result',
      'subopcode': 76,
      'table_ea': 5635160,
      'target_ea': 4709088,
      'target_name': 'loc_47DAE0'},
 77: {'confidence': 'recovered',
      'mnemonic': 'process_memcpy',
      'semantic': 'ffmempcpy: copies memory between process address spaces after validating process ids and VM slices',
      'subopcode': 77,
      'table_ea': 5635164,
      'target_ea': 4717008,
      'target_name': 'loc_47F9D0'},
 78: {'confidence': 'exact',
      'mnemonic': 'memcpy',
      'semantic': 'memcpy(dst, src, len) with VM slice bounds checks/resizing',
      'subopcode': 78,
      'table_ea': 5635168,
      'target_ea': 4717552,
      'target_name': 'loc_47FBF0'},
 79: {'confidence': 'exact',
      'mnemonic': 'memset',
      'semantic': 'memset(dst, value, len) with VM slice bounds checks/resizing',
      'subopcode': 79,
      'table_ea': 5635172,
      'target_ea': 4717776,
      'target_name': 'loc_47FCD0'},
 80: {'confidence': 'recovered',
      'mnemonic': 'angle_delta',
      'semantic': 'computes a wrapped angular delta using the engine angle domain (0x3840/14400 range)',
      'subopcode': 80,
      'table_ea': 5635176,
      'target_ea': 4706816,
      'target_name': 'loc_47D200'},
 81: {'confidence': 'exact',
      'mnemonic': 'distance_or_distance_sq',
      'semantic': 'computes distance-like values: with two 12-byte vector buffers pushes 3D distance; with 4 args '
                  'pushes 2D distance; with 5 args pushes squared 2D distance; with 6 args pushes 3D distance, '
                  'otherwise squared 3D term',
      'subopcode': 81,
      'table_ea': 5635180,
      'target_ea': 4701104,
      'target_name': 'loc_47BBB0'},
 82: {'confidence': 'recovered',
      'mnemonic': 'request_halt',
      'semantic': 'requests interpreter/application halt by setting a global exit flag',
      'subopcode': 82,
      'table_ea': 5635184,
      'target_ea': 4717888,
      'target_name': 'loc_47FD40'},
 83: {'confidence': 'recovered',
      'mnemonic': 'object_state_query',
      'semantic': 'queries object state via sub_4B2CC0/sub_49C850. One-arg form pushes sub_4B2CC0(handle,0); five-arg '
                  'form reads fields +0x2A0..+0x2B0 and writes status/vector/float outputs into VM buffers',
      'subopcode': 83,
      'table_ea': 5635188,
      'target_ea': 4714288,
      'target_name': 'loc_47EF30'},
 84: {'confidence': 'exact',
      'mnemonic': 'object_get_field_0xB4',
      'semantic': 'pushes [sub_49C950(handle)+0xB4] or -1 when handle resolution fails',
      'subopcode': 84,
      'table_ea': 5635192,
      'target_ea': 4714592,
      'target_name': 'loc_47F060'},
 85: {'confidence': 'recovered',
      'mnemonic': 'push_global_55F73C',
      'semantic': 'pushes global integer dword_55F73C',
      'subopcode': 85,
      'table_ea': 5635196,
      'target_ea': 4717936,
      'target_name': 'loc_47FD70'},
 86: {'confidence': 'exact',
      'mnemonic': 'discard_value',
      'semantic': 'pops/discards one argument, choosing int or float pop by stack type metadata',
      'subopcode': 86,
      'table_ea': 5635200,
      'target_ea': 4711632,
      'target_name': 'loc_47E4D0'},
 87: {'confidence': 'recovered',
      'mnemonic': 'object_set_int_property_a',
      'semantic': 'sets an int through property pointer returned by sub_49CC20(handle) or sub_49D0E0(handle) depending '
                  'on optional third argument',
      'subopcode': 87,
      'table_ea': 5635204,
      'target_ea': 4718032,
      'target_name': 'loc_47FDD0'},
 88: {'confidence': 'recovered',
      'mnemonic': 'object_set_int_property_b',
      'semantic': 'sets an int through property pointer returned by sub_49CD40(handle) or sub_49CFA0(handle) depending '
                  'on optional third argument',
      'subopcode': 88,
      'table_ea': 5635208,
      'target_ea': 4718128,
      'target_name': 'loc_47FE30'},
 89: {'confidence': 'recovered',
      'mnemonic': 'object_relation_query',
      'semantic': 'pushes result of sub_49CA90(arg0,arg1)',
      'subopcode': 89,
      'table_ea': 5635212,
      'target_ea': 4718224,
      'target_name': 'loc_47FE90'},
 90: {'confidence': 'recovered',
      'mnemonic': 'object_set_float_property',
      'semantic': 'sets *sub_49CE60(handle) to a float argument when the property pointer is non-null',
      'subopcode': 90,
      'table_ea': 5635216,
      'target_ea': 4718272,
      'target_name': 'loc_47FEC0'},
 91: {'confidence': 'exact',
      'mnemonic': 'formatted_file_log',
      'semantic': 'ffflogf: appends timestamped formatted text to logs/<filename>',
      'subopcode': 91,
      'table_ea': 5635220,
      'target_ea': 4711200,
      'target_name': 'loc_47E320'},
 92: {'confidence': 'exact',
      'mnemonic': 'object_set_float_fields_0x27C_0x280_0x284',
      'semantic': 'sets engine object fields +0x27C and +0x284 from two floats; when arg_count == 4 also sets +0x280 '
                  'from an additional float',
      'subopcode': 92,
      'table_ea': 5635224,
      'target_ea': 4718320,
      'target_name': 'loc_47FEF0'},
 93: {'confidence': 'exact',
      'mnemonic': 'object_set_float_field_0x28C',
      'semantic': 'sets engine object field +0x28C to either a float argument or sub_491670(int_arg)-converted float '
                  'in the three-argument form',
      'subopcode': 93,
      'table_ea': 5635228,
      'target_ea': 4718496,
      'target_name': 'loc_47FFA0'},
 94: {'confidence': 'exact',
      'mnemonic': 'object_set_float_field_0x294',
      'semantic': 'sets engine object field +0x294 to a float argument',
      'subopcode': 94,
      'table_ea': 5635232,
      'target_ea': 4718608,
      'target_name': 'loc_480010'},
 95: {'confidence': 'exact',
      'mnemonic': 'strstr',
      'semantic': 'pushes slice descriptor at strstr(haystack, needle), or null span when not found',
      'subopcode': 95,
      'table_ea': 5635236,
      'target_ea': 4708272,
      'target_name': 'loc_47D7B0'},
 96: {'confidence': 'exact',
      'mnemonic': 'memmove',
      'semantic': 'memmove(dst, src, len) with VM slice bounds checks/resizing',
      'subopcode': 96,
      'table_ea': 5635240,
      'target_ea': 4717664,
      'target_name': 'loc_47FC60'},
 97: {'confidence': 'exact',
      'mnemonic': 'object_get_or_set_flag_0x278',
      'semantic': 'reads or sets engine object flag field +0x278: second arg -1 pushes whether the flag equals 1; '
                  'otherwise sets the flag to 1 and pushes 0',
      'subopcode': 97,
      'table_ea': 5635244,
      'target_ea': 4718704,
      'target_name': 'loc_480070'},
 98: {'confidence': 'exact',
      'mnemonic': 'push_runtime_constant_pair',
      'semantic': 'pops two selectors and pushes globals: (-2,-2)=>dword_4A43F04, (-1,*)=>dword_4A43EF8, '
                  '(*,-1)=>dword_4A43EFC, otherwise 0',
      'subopcode': 98,
      'table_ea': 5635248,
      'target_ea': 4718832,
      'target_name': 'loc_4800F0'},
 99: {'confidence': 'exact',
      'mnemonic': 'object_set_flag_0x141',
      'semantic': 'sets byte [sub_49C850(sub_494C60(handle), line 0x21D3)+0x141] to bool(value)',
      'subopcode': 99,
      'table_ea': 5635252,
      'target_ea': 4715040,
      'target_name': 'loc_47F220'},
 100: {'confidence': 'exact',
       'mnemonic': 'object_get_norm_vec3',
       'semantic': 'ffg_norm: validates a float-buffer descriptor, then writes [sub_49C850(...)+0x14C/+0x150/+0x154] '
                   'into the destination 3-float VM buffer',
       'subopcode': 100,
       'table_ea': 5635256,
       'target_ea': 4718928,
       'target_name': 'loc_480150'},
 101: {'confidence': 'recovered',
       'mnemonic': 'object_get_position_vec3',
       'semantic': 'writes object position vector (+8/+12/+16) into a 12-byte VM buffer',
       'subopcode': 101,
       'table_ea': 5635260,
       'target_ea': 4719136,
       'target_name': 'loc_480220'},
 102: {'confidence': 'exact',
       'mnemonic': 'object_get_abg_vec3',
       'semantic': 'ffg_abg: writes object fields +0x14/+0x18/+0x1C into a destination 3-float VM buffer',
       'subopcode': 102,
       'table_ea': 5635264,
       'target_ea': 4719296,
       'target_name': 'loc_4802C0'},
 103: {'confidence': 'partial',
       'mnemonic': 'ffsys_api',
       'semantic': 'builtin 0x67 enters the large ffsys selector switch in sub_477500; first VM arg is the selector and later args are selector-specific.',
       'subopcode': 103,
       'table_ea': 5635268,
       'target_ea': 4734064,
       'target_name': 'loc_483C70'},
 104: {'confidence': 'recovered',
       'mnemonic': 'thisname',
       'semantic': 'copies current script/object name from current runtime context into destination string buffer',
       'subopcode': 104,
       'table_ea': 5635272,
       'target_ea': 4725200,
       'target_name': 'loc_4819D0'},
 105: {'confidence': 'recovered',
       'mnemonic': 'object_remove_type5',
       'semantic': 'if handle >= 0 calls sub_495910(handle), then unregisters/clears the handle via sub_477910(handle, '
                   'type 5, value 0)',
       'subopcode': 105,
       'table_ea': 5635276,
       'target_ea': 4715904,
       'target_name': 'loc_47F580'},
 106: {'confidence': 'exact',
       'mnemonic': 'rand_float',
       'semantic': 'pushes random float in approximately [0, 1) using rand()*1/32768',
       'subopcode': 106,
       'table_ea': 5635280,
       'target_ea': 4701056,
       'target_name': 'loc_47BB80'},
 107: {'confidence': 'recovered',
       'mnemonic': 'prc_name',
       'semantic': 'copies process name for a process id into destination buffer and pushes zero/status',
       'subopcode': 107,
       'table_ea': 5635284,
       'target_ea': 4725280,
       'target_name': 'loc_481A20'},
 108: {'confidence': 'recovered',
       'mnemonic': 'ffmempcpy_alt',
       'semantic': 'alternate ffmempcpy/process-memory copy path with explicit source/destination process ids',
       'subopcode': 108,
       'table_ea': 5635288,
       'target_ea': 4717296,
       'target_name': 'loc_47FAF0'},
 109: {'confidence': 'exact',
       'mnemonic': 'copy_effect_name_by_id',
       'semantic': 'copies effect/name table entry byte_3E73208[index*0x3C] into destination VM string buffer when '
                   'index <= 0x0FFF; otherwise writes an empty string',
       'subopcode': 109,
       'table_ea': 5635292,
       'target_ea': 4725408,
       'target_name': 'loc_481AA0'},
 110: {'confidence': 'recovered',
       'mnemonic': 'find_effect_id',
       'semantic': 'case-insensitive lookup in the effect/name table; pushes matching index or -1',
       'subopcode': 110,
       'table_ea': 5635296,
       'target_ea': 4725520,
       'target_name': 'loc_481B10'},
 111: {'confidence': 'recovered',
       'mnemonic': 'effect_attach',
       'semantic': 'four-argument effect attach path: validates nonzero handle, calls sub_46AB20 with '
                   'effect/handler/name args and pushes returned effect id/status; emits error on zero handle',
       'subopcode': 111,
       'table_ea': 5635300,
       'target_ea': 4725696,
       'target_name': 'loc_481BC0'},
 112: {'confidence': 'recovered',
       'mnemonic': 'dmalloc_free',
       'semantic': 'frees a dynamic VM allocation referenced by a pointer slice and clears its descriptor',
       'subopcode': 112,
       'table_ea': 5635304,
       'target_ea': 4726400,
       'target_name': 'loc_481E80'},
 113: {'confidence': 'recovered',
       'mnemonic': 'dmalloc',
       'semantic': 'allocates dynamic VM memory for a pointer slice descriptor; errors if descriptor already points to '
                   'memory',
       'subopcode': 113,
       'table_ea': 5635308,
       'target_ea': 4726128,
       'target_name': 'loc_481D70'},
 114: {'confidence': 'recovered',
       'mnemonic': 'assoc_array_set',
       'semantic': 'stores a value into the named runtime associative array dword_4A2AB64, growing/creating entries as '
                   'needed and using dword_6FD414 as allocation state',
       'subopcode': 114,
       'table_ea': 5635312,
       'target_ea': 4726624,
       'target_name': 'loc_481F60'},
 115: {'confidence': 'recovered',
       'mnemonic': 'assoc_array_get',
       'semantic': 'reads from the named runtime associative array dword_4A2AB64: supports count/query and indexed '
                   'retrieval forms, returning 0 for missing entries',
       'subopcode': 115,
       'table_ea': 5635316,
       'target_ea': 4727008,
       'target_name': 'loc_4820E0'},
 116: {'confidence': 'exact',
       'mnemonic': 'file_lock',
       'semantic': 'locking(fd, lock_or_unlock, len) wrapper',
       'subopcode': 116,
       'table_ea': 5635320,
       'target_ea': 4712016,
       'target_name': 'loc_47E650'},
 117: {'confidence': 'recovered',
       'mnemonic': 'native_config_api',
       'semantic': 'lower builtin case 117 / 0x75 enters sub_486A60; first VM arg is a config/native selector and later args are selector-specific. Unknown selectors default to no push.',
       'subopcode': 117,
       'table_ea': 5635324,
       'target_ea': 4745824,
       'target_name': 'sub_486A60'},
 118: {'confidence': 'recovered',
       'mnemonic': 'push_static_word_span',
       'semantic': 'pushes a slice descriptor for static word_4AE09FC buffer',
       'subopcode': 118,
       'table_ea': 5635328,
       'target_ea': 4727472,
       'target_name': 'loc_4822B0'},
 119: {'confidence': 'exact',
       'mnemonic': 'push_minus_one',
       'semantic': 'pushes integer -1',
       'subopcode': 119,
       'table_ea': 5635332,
       'target_ea': 4727520,
       'target_name': 'loc_4822E0'},
 120: {'confidence': 'recovered',
       'mnemonic': 'raw_arg_read',
       'semantic': 'direct entry to sub_482330. Although the body is absent, ASM call-sites use this helper throughout '
                   'nested APIs as the raw/native argument reader returning the next argument in EAX',
       'subopcode': 120,
       'table_ea': 5635336,
       'target_ea': 4727600,
       'target_name': 'sub_482330'},
 121: {'confidence': 'recovered',
       'mnemonic': 'process_translate_ptr',
       'semantic': 'translates a VM slice descriptor into another process memory base; supports implicit sender or '
                   'explicit pid',
       'subopcode': 121,
       'table_ea': 5635340,
       'target_ea': 4716576,
       'target_name': 'loc_47F820'},
 122: {'confidence': 'recovered',
       'mnemonic': 'reserved_noop_7a',
       'semantic': 'reserved/disabled builtin; jump target is IDA nullsub_1 at 0x46B620; no VM stack side effect seen '
                   'in supplied asm',
       'subopcode': 122,
       'table_ea': 5635344,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 123: {'confidence': 'exact',
       'mnemonic': 'snprintf',
       'semantic': 'ffsnprintf: bounded vsnprintf into a destination VM string buffer',
       'subopcode': 123,
       'table_ea': 5635348,
       'target_ea': 4719424,
       'target_name': 'loc_480340'},
 124: {'confidence': 'exact',
       'mnemonic': 'file_read_line',
       'semantic': 'line-oriented read(fd, dst, max_len); stops on newline or NUL and pushes bytes read',
       'subopcode': 124,
       'table_ea': 5635352,
       'target_ea': 4712384,
       'target_name': 'loc_47E7C0'},
 125: {'confidence': 'exact',
       'mnemonic': 'exp',
       'semantic': 'pushes exp(float_arg)',
       'subopcode': 125,
       'table_ea': 5635356,
       'target_ea': 4700704,
       'target_name': 'loc_47BA20'},
 126: {'confidence': 'recovered',
       'mnemonic': 'logf',
       'semantic': 'fflogf-style formatted runtime logging: consumes selector/format/varargs, formats via the VM '
                   'vararg formatter, and routes output according to current context flags. Exact sink naming is '
                   'engine-specific.',
       'subopcode': 126,
       'table_ea': 5635360,
       'target_ea': 4710448,
       'target_name': 'loc_47E030'},
 127: {'confidence': 'recovered',
       'mnemonic': 'current_sender_id',
       'semantic': 'pushes current sender/process id from message context, or -1 when no message context is active',
       'subopcode': 127,
       'table_ea': 5635364,
       'target_ea': 4716464,
       'target_name': 'loc_47F7B0'},
 128: {'confidence': 'recovered',
       'mnemonic': 'parse_api',
       'semantic': 'ffparse: nested parser dispatcher for token/string/identifier/whitespace scanning over VM string '
                   'slices',
       'subopcode': 128,
       'table_ea': 5635368,
       'target_ea': 4733952,
       'target_name': 'loc_483C00'},
 129: {'confidence': 'external',
       'mnemonic': 'chat_utility_api',
       'semantic': 'selector dispatch 0..4 to sub_48A5E0/sub_48A7E0/sub_48A960/sub_48AB00/sub_48ACA0; those target '
                   'bodies are not included in the asm slice',
       'subopcode': 129,
       'table_ea': 5635372,
       'target_ea': 4763216,
       'target_name': 'loc_48AE50'},
 130: {'confidence': 'recovered',
       'mnemonic': 'reserved_noop_82',
       'semantic': 'reserved/disabled builtin; alias of 0x7A to IDA nullsub_1 at 0x46B620',
       'subopcode': 130,
       'table_ea': 5635376,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 131: {'confidence': 'recovered',
       'mnemonic': 'editor_get_click_point',
       'semantic': 'ffeditor selector 0: validates two destination pointers, calls sub_4B2CE0 to retrieve click '
                   'point/status, writes a 3-float vector and one float/status output, then pushes the integer result; '
                   'other selectors return without visible work in this slice',
       'subopcode': 131,
       'table_ea': 5635380,
       'target_ea': 4719584,
       'target_name': 'loc_4803E0'},
 132: {'confidence': 'partial',
       'mnemonic': 'item_inventory_api',
       'semantic': 'large item/inventory dispatcher with selector switch and many item create/load/modify cases '
                   'visible in asm; engine item schema and several callees remain too opaque for exact naming',
       'subopcode': 132,
       'table_ea': 5635384,
       'target_ea': 4758544,
       'target_name': 'loc_489C10'},
 133: {'confidence': 'recovered',
       'mnemonic': 'reserved_noop_85',
       'semantic': 'reserved/disabled builtin; alias of 0x7A to IDA nullsub_1 at 0x46B620',
       'subopcode': 133,
       'table_ea': 5635388,
       'target_ea': 4634144,
       'target_name': 'nullsub_1'},
 134: {'confidence': 'exact',
       'mnemonic': 'strchr',
       'semantic': 'pushes slice descriptor at strchr(string, char), or null span when not found',
       'subopcode': 134,
       'table_ea': 5635392,
       'target_ea': 4701744,
       'target_name': 'loc_47BE30'},
 135: {'confidence': 'recovered',
       'mnemonic': 'stristr',
       'semantic': 'pushes slice descriptor at case-insensitive substring search, or null span when not found',
       'subopcode': 135,
       'table_ea': 5635396,
       'target_ea': 4708384,
       'target_name': 'loc_47D820'},
 136: {'confidence': 'exact',
       'mnemonic': 'strncpy_checked',
       'semantic': 'bounded strncpy into a VM slice; writes terminating NUL and resizes/checks destination',
       'subopcode': 136,
       'table_ea': 5635400,
       'target_ea': 4707504,
       'target_name': 'loc_47D4B0'},
 137: {'confidence': 'exact',
       'mnemonic': 'strnicmp',
       'semantic': 'pushes case-insensitive strnicmp(a,b,n) result',
       'subopcode': 137,
       'table_ea': 5635404,
       'target_ea': 4709216,
       'target_name': 'loc_47DB60'},
 138: {'confidence': 'exact',
       'mnemonic': 'bit_and',
       'semantic': 'pushes integer bitwise AND of two args',
       'subopcode': 138,
       'table_ea': 5635408,
       'target_ea': 4727616,
       'target_name': 'loc_482340'},
 139: {'confidence': 'exact',
       'mnemonic': 'bit_or',
       'semantic': 'pushes integer bitwise OR of two args',
       'subopcode': 139,
       'table_ea': 5635412,
       'target_ea': 4727728,
       'target_name': 'loc_4823B0'},
 140: {'confidence': 'exact',
       'mnemonic': 'bit_xor',
       'semantic': 'pushes integer bitwise XOR of two args',
       'subopcode': 140,
       'table_ea': 5635416,
       'target_ea': 4727840,
       'target_name': 'loc_482420'},
 141: {'confidence': 'exact',
       'mnemonic': 'bit_not',
       'semantic': 'pushes integer bitwise NOT of one arg',
       'subopcode': 141,
       'table_ea': 5635420,
       'target_ea': 4727952,
       'target_name': 'loc_482490'},
 142: {'confidence': 'exact',
       'mnemonic': 'shift_left',
       'semantic': 'pushes integer left shift value << amount',
       'subopcode': 142,
       'table_ea': 5635424,
       'target_ea': 4728048,
       'target_name': 'loc_4824F0'},
 143: {'confidence': 'exact',
       'mnemonic': 'shift_right',
       'semantic': 'pushes integer right shift value >> amount',
       'subopcode': 143,
       'table_ea': 5635428,
       'target_ea': 4728160,
       'target_name': 'loc_482560'},
 144: {'confidence': 'exact',
       'mnemonic': 'bit_clear',
       'semantic': 'pushes value with bit index cleared: value & ~(1 << bit)',
       'subopcode': 144,
       'table_ea': 5635432,
       'target_ea': 4728272,
       'target_name': 'loc_4825D0'},
 145: {'confidence': 'exact',
       'mnemonic': 'bit_set',
       'semantic': 'pushes value with bit index set: value | (1 << bit)',
       'subopcode': 145,
       'table_ea': 5635436,
       'target_ea': 4728400,
       'target_name': 'loc_482650'},
 146: {'confidence': 'exact',
       'mnemonic': 'bit_test',
       'semantic': 'pushes bit test result: (value & (1 << bit)) >> bit',
       'subopcode': 146,
       'table_ea': 5635440,
       'target_ea': 4728528,
       'target_name': 'loc_4826D0'},
 147: {'confidence': 'exact',
       'mnemonic': 'memcmp',
       'semantic': 'compares two byte buffers over n bytes and pushes -1/0/1 style comparison result',
       'subopcode': 147,
       'table_ea': 5635444,
       'target_ea': 4748432,
       'target_name': 'loc_487490'},
 148: {'confidence': 'external',
       'mnemonic': 'typed_load_width_1',
       'semantic': 'thin wrapper: sets ecx=1 and tail-jumps to sub_482740; helper body is not included in the asm '
                   'slice',
       'subopcode': 148,
       'table_ea': 5635448,
       'target_ea': 4729008,
       'target_name': 'loc_4828B0'},
 149: {'confidence': 'external',
       'mnemonic': 'typed_load_width_2',
       'semantic': 'thin wrapper: sets ecx=2 and tail-jumps to sub_482740; helper body is not included in the asm '
                   'slice',
       'subopcode': 149,
       'table_ea': 5635452,
       'target_ea': 4729024,
       'target_name': 'loc_4828C0'},
 150: {'confidence': 'external',
       'mnemonic': 'typed_load_width_3',
       'semantic': 'thin wrapper: sets ecx=3 and tail-jumps to sub_482740; helper body is not included in the asm '
                   'slice',
       'subopcode': 150,
       'table_ea': 5635456,
       'target_ea': 4729040,
       'target_name': 'loc_4828D0'},
 151: {'confidence': 'external',
       'mnemonic': 'typed_load_width_4',
       'semantic': 'thin wrapper: sets ecx=4 and tail-jumps to sub_482740; helper body is not included in the asm '
                   'slice',
       'subopcode': 151,
       'table_ea': 5635460,
       'target_ea': 4729056,
       'target_name': 'loc_4828E0'},
 152: {'confidence': 'recovered',
       'mnemonic': 'span_write_float',
       'semantic': 'writes one float into a destination slice pointer and advances the descriptor by 4 bytes',
       'subopcode': 152,
       'table_ea': 5635464,
       'target_ea': 4729072,
       'target_name': 'loc_4828F0'},
 153: {'confidence': 'recovered',
       'mnemonic': 'span_write_cstring',
       'semantic': 'copies a NUL-terminated string into a destination slice pointer and advances descriptor past the '
                   'string',
       'subopcode': 153,
       'table_ea': 5635468,
       'target_ea': 4729232,
       'target_name': 'loc_482990'},
 154: {'confidence': 'external',
       'mnemonic': 'typed_store_width_1',
       'semantic': 'thin wrapper: sets ecx=1, edx=0 and tail-jumps to sub_4827D0; helper body is not included in the '
                   'asm slice',
       'subopcode': 154,
       'table_ea': 5635472,
       'target_ea': 4729552,
       'target_name': 'loc_482AD0'},
 155: {'confidence': 'external',
       'mnemonic': 'typed_store_width_2',
       'semantic': 'thin wrapper: sets ecx=2, edx=4 and tail-jumps to sub_4827D0; helper body is not included in the '
                   'asm slice',
       'subopcode': 155,
       'table_ea': 5635476,
       'target_ea': 4729568,
       'target_name': 'loc_482AE0'},
 156: {'confidence': 'external',
       'mnemonic': 'typed_store_width_3',
       'semantic': 'thin wrapper: sets ecx=3, edx=4 and tail-jumps to sub_4827D0; helper body is not included in the '
                   'asm slice',
       'subopcode': 156,
       'table_ea': 5635480,
       'target_ea': 4729584,
       'target_name': 'loc_482AF0'},
 157: {'confidence': 'external',
       'mnemonic': 'typed_store_width_4',
       'semantic': 'thin wrapper: sets ecx=4, edx=0 and tail-jumps to sub_4827D0; helper body is not included in the '
                   'asm slice',
       'subopcode': 157,
       'table_ea': 5635484,
       'target_ea': 4729600,
       'target_name': 'loc_482B00'},
 158: {'confidence': 'recovered',
       'mnemonic': 'ptr_store_i32_from_ptr',
       'semantic': 'copies a 32-bit value from source pointer descriptor to destination pointer descriptor and '
                   'advances source',
       'subopcode': 158,
       'table_ea': 5635488,
       'target_ea': 4729616,
       'target_name': 'loc_482B10'},
 159: {'confidence': 'recovered',
       'mnemonic': 'ptr_copy_cstring',
       'semantic': 'copies a NUL-terminated string from source pointer descriptor to destination pointer descriptor '
                   'and advances source',
       'subopcode': 159,
       'table_ea': 5635492,
       'target_ea': 4729808,
       'target_name': 'loc_482BD0'},
 160: {'confidence': 'recovered',
       'mnemonic': 'binary_search_i32',
       'semantic': 'binary-searches a sorted int32 array descriptor for a value and pushes the matching/lower-bound '
                   'index result',
       'subopcode': 160,
       'table_ea': 5635496,
       'target_ea': 4730048,
       'target_name': 'loc_482CC0'},
 161: {'confidence': 'partial',
       'mnemonic': 'resource_handle_api',
       'semantic': 'nested resource-handle dispatcher: reads a raw native argument, validates magic 0x19285, then '
                   'dispatches by handle type/subtype into resource helper calls. Some target helpers are outside the '
                   'asm slice.',
       'subopcode': 161,
       'table_ea': 5635500,
       'target_ea': 4540288,
       'target_name': 'loc_454780'},
 162: {'confidence': 'recovered',
       'mnemonic': 'entity_ref_api',
       'semantic': 'nested entity/reference API: selector 1 creates/looks up refs via '
                   'sub_454C80/sub_4582D0/sub_4562D0/sub_455C30/sub_457AD0; selector 2 validates/registers and calls '
                   'sub_441F10; selectors 3..5 query typed fields. Uses raw_arg_read/sub_482330.',
       'subopcode': 162,
       'table_ea': 5635504,
       'target_ea': 4464704,
       'target_name': 'loc_442040'},
 163: {'confidence': 'recovered',
       'mnemonic': 'buffer_hash_or_checksum',
       'semantic': 'reads buffer pointer descriptor and length, calls sub_4746D0(buffer,len), and pushes dword_3E6AA80 '
                   'result/status; likely hash/checksum based on call pattern',
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
        resume_target = off + 1
        return DecodedOpcode(
            mnemonic="yield_program",
            length=1,
            operands={
                "semantic": "main-loop special: save PC after this opcode, suspend current scheduler slice, and resume later",
                "resume_target": resume_target,
            },
            # Keep this terminal for the *current scheduler slice*.  The explicit
            # yield_resume edge below models the coroutine continuation that the
            # engine reaches after it reloads the saved PC on a later pass.
            terminal=True,
            known=True,
            edges=[DecodedEdge("yield_resume", resume_target, "saved PC after yield; resumed by scheduler")],
        )

    spec = OPCODES.get(opcode)
    if spec is None:
        return DecodedOpcode(
            mnemonic=f"unknown_{opcode:02X}",
            length=1,
            operands={
                "opcode_char": safe_chr(opcode),
                "semantic": "not present in recovered dispatcher table; treated as VM trap/resync stop",
            },
            terminal=True,
            known=False,
            edges=[DecodedEdge("trap", None, "not present in recovered dispatcher table")],
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

        elif fmt == "import_stub_u32":
            if not have(5):
                raise ValueError("truncated import stub payload")
            length = 5
            operands.update({
                "value": _u32(buf, off + 1),
                "payload_u32": _u32(buf, off + 1),
                "stub_note": "0x67 import/unlinked-call stubs are 5 bytes; treating them as length 1 desynchronizes linear decoding",
            })
            terminal = True
            edges.append(DecodedEdge("import_stub", None, spec.semantic))

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
            if sub not in BUILTINS:
                raise AssertionError(f"missing recovered builtin subopcode 0x{sub:02X}")
            builtin = BUILTINS[sub]
            mnemonic = builtin.mnemonic
            operands.update({
                "subopcode": sub,
                "subopcode_hex": builtin.subopcode_hex,
                "target_ea": f"0x{builtin.target_ea:08X}",
                "target_name": builtin.target_name,
                "builtin_semantic": builtin.semantic,
                "builtin_confidence": builtin.confidence,
            })

        else:
            raise AssertionError(f"unhandled recovered opcode format: {fmt!r}")

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
    elif opcode == 0x67 and fmt != "import_stub_u32":
        terminal = True
        edges.append(DecodedEdge("error", None, spec.semantic))

    return DecodedOpcode(mnemonic, length, operands, terminal, known, edges)
