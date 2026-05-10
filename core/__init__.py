"""Compact MBC decompiler core.

Modules are grouped by responsibility:
- common: constants and binary/type helpers
- loader: MBC/project parsing
- opcodes: recovered opcode tables and operand decoding
- bytecode: instruction decoding and control-flow reachability
- linker: symbolic project/module linking
- calls: native/builtin call specs and effect policy
- vm_ast: symbolic VM and pseudo-AST builder
- decompiler: high-level source renderer
"""
