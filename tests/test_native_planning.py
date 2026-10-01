"""Source/planner regressions; these tests never generate the real client corpus."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mbc_format.loader import MbcHeader, MbcProgram, MbcScript, MbcFunction
from mbc_format.metadata import ModuleMetadata
from mbc_format.opcodes import OPCODES
from decompile.linker import MbcStaticLinker
from decompile.source_ir import Block, Expression, Function, Statement, recover_function, DYNAMIC, VOID
from decompile.native_project import Module, Project, Target, build_layout, class_name, cpp_type, is_native_descriptor
from decompile.native_cpp import CppGenerator, Renderer

ROOT = Path(__file__).resolve().parents[2]
OP = {spec.mnemonic: code for code, spec in OPCODES.items()}
OP.update(end_program=35, yield_program=124)
METADATA = struct.pack('<80H7I', *([65535] * 80), *([0] * 7))


def script(name='sample', data=b'\0' * 64):
    return MbcScript(Path(name + '.mbc'), MbcHeader('MBC', 0, 1, 256, len(data)), bytes(256), data,
                     [MbcProgram(0, 'entry', 0, 255, 255, 0, 0)], [], METADATA)


def instruction(index, mnemonic, **operands):
    return SimpleNamespace(offset=index, length=1, opcode=OP.get(mnemonic, 102), mnemonic=mnemonic,
                           operands=operands, known=True, terminal=False, edges=[])


def recover(rows, data=b'\0' * 64):
    image = script(data=data)
    code = [instruction(index, name, **operands) for index, (name, operands) in enumerate(rows)]
    function = recover_function(image, image.programs[0], code, MbcStaticLinker(image))
    return image, function


def conditional(body, tail):
    return [('push_data_ref', {'type': 16, 'data_offset': 0}),
            ('jfalse_rel16', {'target': 2 + len(body)})] + list(body) + list(tail)


def send_current_process():
    return [('set_arg_count', {'value': 0}), ('current_process_id', {'subopcode': 39}),
            ('push_imm_i8', {'type': 16, 'value': 1}), ('set_arg_count', {'value': 2}),
            ('send_to_process_id', {'subopcode': 20})]


def planned(functions, *, data=None):
    modules = []
    for index, function in enumerate(functions):
        image = script('module' + str(index), data[index] if data else b'\0' * 64)
        image.programs[0].name = function.name
        image.programs[0].start = function.entry
        image.functions = [MbcFunction(0, function.name, function.entry, 0, 0)]
        function.aliases.add(function.name)
        module = Module(index, image, [function], cclass=class_name(image.path.stem))
        module.by_entry = {function.entry: function}
        module.by_name = {name: function for name in function.aliases}
        module.layout = build_layout(module)
        modules.append(module)
    return Project(modules)


def constant(value, name='constant', entry=0):
    return Function(name, entry, return_type=16, blocks=[Block(entry, [Statement('return', Expression('number', str(value), (), 16))])])


class NativeTypeTests(unittest.TestCase):
    def test_named_types_keep_existing_cpp_names(self):
        aliases = {VOID: 'void', DYNAMIC: 'Value', None: 'Value', 0: 'std::int8_t',
                   16: 'std::int32_t', 32: 'float', 1: 'String', 2: 'StringRef',
                   17: 'IntRef', 18: 'IntRefRef', 33: 'FloatRef', 34: 'FloatRefRef',
                   48: 'Address', 49: 'AddressRef'}
        for typ, expected in aliases.items():
            with self.subTest(typ=typ):
                self.assertEqual(cpp_type(typ), expected)

    def test_all_descriptor_tags_have_native_cpp_types(self):
        aliases = {1, 2, 17, 18, 33, 34, 48, 49}
        for typ in range(256):
            with self.subTest(typ=typ):
                self.assertEqual(is_native_descriptor(typ), typ not in {0, 16, 32})
                if typ not in aliases | {0, 16, 32}:
                    self.assertEqual(cpp_type(typ), f'Reference<{typ}>')
        for typ in (None, VOID, DYNAMIC):
            self.assertFalse(is_native_descriptor(typ))

    def test_invalid_type_ids_are_not_silently_erased(self):
        for typ in (-3, 256, 4096, '3'):
            with self.subTest(typ=typ):
                with self.assertRaisesRegex(ValueError, 'Unsupported recovered native type'):
                    cpp_type(typ)

    def test_address_of_nested_reference_emits_its_own_type(self):
        for typ in (2, 18, 34, 49, 254):
            with self.subTest(typ=typ):
                _, function = recover([('push_data_ref', {'type': typ, 'data_offset': 0}),
                                       ('address_of', {}), ('return', {})])
                self.assertEqual(function.return_type, typ + 1)
                header, source, _ = CppGenerator(planned([function]), ROOT / 'client').render()
                self.assertIn(f'Reference<{typ + 1}>', header)
                self.assertIn(f'storedValue<Reference<{typ + 1}>>', source)

    def test_reference_types_do_not_merge_different_typed_helpers(self):
        functions = [recover([('push_data_ref', {'type': typ, 'data_offset': 0}),
                              ('return', {})])[1] for typ in (3, 3, 19)]
        project = planned(functions)
        self.assertEqual(len(project.helpers), 2)
        self.assertEqual(sorted(len(helper.instances) for helper in project.helpers), [1, 2])


class RecoveryTests(unittest.TestCase):
    def test_integer_division_stays_integer(self):
        _, f = recover([('push_imm_i8', {'type': 16, 'value': 9}), ('push_imm_i8', {'type': 16, 'value': 2}), ('div', {}), ('return', {})])
        self.assertEqual(f.return_type, 16)

    def test_increment_has_one_write_and_result(self):
        _, f = recover([('push_data_ref', {'type': 16, 'data_offset': 4}), ('push_data_ref', {'type': 16, 'data_offset': 0}), ('pre_inc', {}), ('store', {}), ('return', {})])
        self.assertEqual(sum(s.kind == 'increment' for s in f.statements()), 1)
        self.assertEqual(sum(s.kind == 'store' for s in f.statements()), 1)
        self.assertEqual(list(f.statements())[-1].expression.kind, 'stored')

    def test_yield_count_preserved(self):
        _, f = recover([('yield_program', {}), ('yield_program', {}), ('yield_program', {}), ('end_program', {})])
        self.assertEqual(sum(s.kind == 'yield' for s in f.statements()), 3)
        self.assertTrue(f.asynchronous)

    def test_phi_has_both_branch_values(self):
        _, f = recover([('push_imm_i8', {'type': 16, 'value': 1}), ('jfalse_rel16', {'target': 4}),
                        ('push_imm_i8', {'type': 16, 'value': 10}), ('jmp_rel16', {'target': 5}),
                        ('push_imm_i8', {'type': 16, 'value': 20}), ('force_int_type_alt', {}), ('return', {})])
        values = {s.expression.value for s in f.statements() if s.kind == 'phi'}
        self.assertEqual(values, {'10', '20'})

    def test_dead_result_before_yield_keeps_call_and_every_suspension(self):
        _, function = recover(conditional(send_current_process(), [('yield_program', {})] * 8 + [('end_program', {})]))
        calls = [s for s in function.statements() if s.kind == 'builtin']
        self.assertEqual([s.metadata['subopcode'] for s in calls], [39, 20])
        self.assertIsNone(calls[-1].result)
        self.assertEqual(sum(s.kind == 'yield' for s in function.statements()), 8)
        self.assertFalse(any(name.startswith('merge_') for name in function.variables))

    def test_dead_values_and_frames_before_terminal_boundaries(self):
        for operation in ('yield_program', 'end_program', 'halt_interpreter'):
            with self.subTest(operation=operation):
                rows = conditional([('push_stack_frame', {}), ('push_imm_i8', {'type': 16, 'value': 7})],
                                   [(operation, {}), ('end_program', {})] if operation == 'yield_program' else [(operation, {})])
                _, function = recover(rows)
                self.assertFalse(any(name.startswith('merge_') for name in function.variables))

    def test_dead_result_through_jump_and_program_action(self):
        body = send_current_process()
        join = 2 + len(body)
        rows = conditional(body, [('jmp_rel16', {'target': join + 1}), ('program_restart', {'program_index': 0}),
                                  ('stack_frame_reset', {}), ('push_imm_i8', {'type': 16, 'value': 42}), ('return', {})])
        _, function = recover(rows)
        self.assertEqual(sum(s.kind == 'program' for s in function.statements()), 1)
        self.assertEqual(sum(s.kind == 'builtin' and s.metadata['subopcode'] == 20 for s in function.statements()), 1)

    def test_reset_preserves_outer_frame_values(self):
        rows = [('push_imm_i8', {'type': 16, 'value': 42}), ('push_stack_frame', {})]
        rows += [('push_data_ref', {'type': 16, 'data_offset': 0}), ('jfalse_rel16', {'target': 5}),
                 ('push_imm_i8', {'type': 16, 'value': 99}), ('stack_frame_reset', {}),
                 ('pop_stack_frame', {}), ('return', {})]
        _, function = recover(rows)
        returns = [s for s in function.statements() if s.kind == 'return']
        self.assertEqual(len(returns), 1)
        self.assertIsNotNone(returns[0].expression)
        self.assertFalse(any(node.kind == 'number' and node.value == '99' for node in function.expressions()))

    def test_live_stack_conflict_is_still_rejected(self):
        rows = conditional([('push_imm_i8', {'type': 16, 'value': 99})],
                           [('push_imm_i8', {'type': 16, 'value': 1}), ('add', {}), ('return', {})])
        with self.assertRaisesRegex(ValueError, 'underflow|Incompatible live stack'):
            recover(rows)

    def test_mixed_returns_keep_value_and_empty_paths(self):
        rows = conditional([('push_imm_i8', {'type': 16, 'value': 99})],
                           [('program_restart', {'program_index': 0}), ('return', {})])
        _, function = recover(rows)
        returns = [s.expression for s in function.statements() if s.kind == 'return']
        self.assertEqual(len(returns), 2)
        self.assertEqual(sum(value is None for value in returns), 1)
        self.assertEqual([value.value for value in returns if value is not None], ['99'])
        self.assertEqual(function.return_type, DYNAMIC)

    def test_discard_consumes_argument_count_before_local_call(self):
        rows = conditional([('push_imm_i8', {'type': 16, 'value': 99}), ('set_arg_count', {'value': 1}),
                            ('discard_value', {'subopcode': 86})],
                           [('call_rel32', {'target': 200, 'target_name': 'local_helper'}),
                            ('stack_frame_reset', {}), ('end_program', {})])
        _, function = recover(rows)
        call = next(s for s in function.statements() if s.kind == 'call')
        self.assertEqual(call.expression.children, ())

    def test_zero_argument_call_supplies_return_without_incoming_value(self):
        rows = conditional([('push_imm_i8', {'type': 16, 'value': 99})],
                           [('call_rel32', {'target': 200, 'target_name': 'local_helper'}), ('return_local', {})])
        _, function = recover(rows)
        self.assertEqual(sum(s.kind == 'call' for s in function.statements()), 1)
        self.assertFalse(any(name.startswith('merge_') for name in function.variables))

    def test_system_movement_query_returns_value(self):
        _, function = recover([('push_imm_u16', {'type': 16, 'value': 231}), ('set_arg_count', {'value': 1}),
                               ('ffsys_api', {'subopcode': 103}), ('push_imm_i8', {'type': 16, 'value': 0}),
                               ('ne', {}), ('return', {})])
        call = next(s for s in function.statements() if s.kind == 'builtin')
        self.assertIsNotNone(call.result)
        self.assertEqual(call.metadata['subopcode'], 103)
        self.assertEqual(function.return_type, 16)

    def test_previous_conversion_keeps_top(self):
        _, f = recover([('push_imm_i8', {'type': 16, 'value': 1}), ('push_imm_i8', {'type': 16, 'value': 2}), ('to_float_prev', {}), ('return', {})])
        self.assertEqual(list(f.statements())[-1].expression.value, '2')

    def test_byte_storage_width_survives_binding_collection(self):
        _, f = recover([('push_data_ref', {'type': 0, 'data_offset': 7}), ('return', {})])
        self.assertEqual(f.return_type, 16)
        self.assertTrue(any(b.get('width') == 1 for b in f.bindings.values()))

    def test_invalid_stack_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'underflow'):
            recover([('add', {}), ('return', {})])

    def test_pointer_address_is_separate_from_value(self):
        _, f = recover([('push_data_ref', {'type': 16, 'data_offset': 0}), ('address_of', {}), ('return', {})])
        expression = list(f.statements())[-1].expression
        self.assertEqual(expression.kind, 'address')
        self.assertEqual(expression.children[0].kind, 'storage')


class SharingTests(unittest.TestCase):
    def test_identical_bodies_share_one_non_template_helper(self):
        project = planned([constant(1), constant(1)])
        self.assertEqual(len(project.helpers), 1)
        self.assertEqual(len(project.families), 1)
        generator = CppGenerator(project, ROOT / 'client')
        text = generator.helper_parameters(project.helpers[0])
        self.assertNotIn('Self', text)
        self.assertIn('Module &self', text)

    def test_different_names_and_offsets_share_bodies(self):
        project = planned([constant(4, 'first', 0), constant(4, 'second', 80)])
        self.assertEqual(len(project.helpers), 1)
        generator = CppGenerator(project, ROOT / 'client')
        self.assertIn('FunctionScope context(self, routine);', Renderer(generator, project.helpers[0]).body())

    def test_literal_variants_pass_distinct_settings(self):
        project = planned([constant(0, 'isFlamount'), constant(1, 'isFlamount')])
        self.assertEqual(len(project.helpers), 1)
        helper = project.helpers[0]
        self.assertEqual(helper.literal_parameters, [[0]])
        generator = CppGenerator(project, ROOT / 'client')
        calls = [generator.helper_call(item, 'self', {}, []) for item in helper.instances]
        self.assertNotEqual(calls[0], calls[1])
        self.assertTrue(calls[0].endswith(', 0)'))
        self.assertTrue(calls[1].endswith(', 1)'))

    def test_initial_values_do_not_specialize_code(self):
        project = planned([constant(4), constant(4)], data=[b'\0' * 64, b'\1' + b'\0' * 63])
        self.assertEqual(len(project.families), 1)
        self.assertNotEqual(project.modules[0].layout.fields[0].initial, project.modules[1].layout.fields[0].initial)

    def test_exact_metadata_sequences_are_interned(self):
        generator = CppGenerator(planned([constant(1), constant(1)]), ROOT / 'client')
        first = generator.metadata(generator.modules[0])
        pools = len(generator.pools)
        second = generator.metadata(generator.modules[1])
        self.assertEqual(pools, len(generator.pools))
        self.assertNotEqual(first, second)  # Identity differs; shared metadata does not.

    def test_unique_function_stays_module_method(self):
        project = planned([constant(3)])
        generator = CppGenerator(project, ROOT / 'client')
        self.assertFalse(generator.is_shared(project.helpers[0]))
        self.assertIn('.constant(', generator.helper_call(project.implementations[0], 'self', {}, []))

    def test_side_effects_prevent_false_merging(self):
        a = constant(1)
        b = constant(1)
        b.blocks[0].statements.insert(0, Statement('yield'))
        b.asynchronous = True
        self.assertEqual(len(planned([a, b]).helpers), 2)

    def test_local_names_resolve_once(self):
        image = script()
        image.functions = [MbcFunction(0, 'Get', 10, 0, 0)]
        callee = constant(6, 'Get', 10)
        call = Statement('call', Expression('invoke', 'Get', (), None), 'result', {'target': 10})
        caller = Function('Caller', 0, variables={'result': DYNAMIC}, return_type=DYNAMIC,
                          blocks=[Block(0, [call, Statement('return', Expression('name', 'result'))])])
        module = Module(0, image, [caller, callee], cclass='MbcSample')
        module.by_entry = {0: caller, 10: callee}
        module.by_name = {'Caller': caller, 'Get': callee}
        module.layout = build_layout(module)
        project = Project([module])
        self.assertEqual(project.by_target[Target(0, 0)].targets[id(call)], (Target(0, 10),))
        self.assertEqual(caller.variables['result'], 16)

class TransferTests(unittest.TestCase):
    def test_phi_swap_uses_one_temporary(self):
        from decompile.source_ir import parallel_transfers
        function = Function('swap', 0, variables={'a': 16, 'b': 16})
        copies = [Statement('phi', Expression('name', 'b', (), 16), 'a'),
                  Statement('phi', Expression('name', 'a', (), 16), 'b')]
        result = parallel_transfers(function, copies)
        values = {'a': 1, 'b': 2}
        for statement in result:
            values[statement.result] = values[statement.expression.value]
        self.assertEqual((values['a'], values['b']), (2, 1))
        self.assertEqual(len(result), 3)

    def test_unused_calls_are_not_removed(self):
        from decompile.source_ir import eliminate_dead_values
        call = Statement('call', Expression('invoke', 'effect'), 'unused')
        function = Function('test', 0, variables={'unused': DYNAMIC}, blocks=[Block(0, [call])])
        eliminate_dead_values(function)
        self.assertEqual(len(function.blocks[0].statements), 1)
        self.assertIsNone(call.result)


if __name__ == '__main__':
    unittest.main()
