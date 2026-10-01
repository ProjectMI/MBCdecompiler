"""Compile/run small in-memory fixtures, never the game corpus or client output files."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from test_native_planning import ROOT, CppGenerator, planned, recover, constant, conditional, send_current_process

HOST = r'''
#include <cassert>
using namespace SphereScripts;
class FixtureHost final : public Host
{
    struct Region { std::uint32_t base; void *data; std::size_t size; const void *owner; };
    std::vector<Region> regions;
    std::uint32_t next = 4096;
  public:
    int warnings = 0;
    int processCalls = 0;
    int programActions = 0;
    bool live = true;
    Value invokeEngine(Builtin, std::span<const Argument>) override { return Value(7); }
    Task<Value> callProcess(Module &, bool, std::vector<Argument>) override { ++processCalls; co_return Value(9); }
    Address mapObject(void *data, std::size_t size, const void *owner) override
    {
        for (const auto &region : regions)
            if (region.data == data && region.owner == owner && region.size == size)
                return {region.base, region.base, region.base + std::uint32_t(size) - 1};
        const auto base = next;
        next += std::uint32_t(size) + 16;
        regions.push_back({base, data, size, owner});
        return {base, base, base + std::uint32_t(size) - 1};
    }
    void forgetObject(const void *owner) noexcept override
    {
        std::erase_if(regions, [owner](const auto &region) { return region.owner == owner; });
    }
    std::span<std::byte> memory(Address address, std::size_t size) override
    {
        for (const auto &region : regions)
            if (address.base >= region.base && address.base - region.base <= region.size &&
                size <= region.size - (address.base - region.base))
                return {static_cast<std::byte *>(region.data) + address.base - region.base, size};
        throw std::out_of_range("fixture memory");
    }
    void programAction(Module &, std::string_view, ProgramAction) override { ++programActions; }
    bool alive(const Module &) const noexcept override { return live; }
    Module &moduleInstance(Module &, std::span<const ModuleId>) override
    {
        throw std::logic_error("fixture has no external provider");
    }
    void warning(std::string_view) override { ++warnings; }
    void halt() override { throw std::runtime_error("halt"); }
    Value run(Module &module, int &resumes)
    {
        auto task = module.namedEntry("entry").start({});
        Execution current;
        task.start(current);
        try
        {
            while (!task.done())
            {
                assert(current.suspended);
                activate(current.context, &current);
                current.suspended.resume();
                assert(++resumes < 16);
            }
            const auto result = task.result();
            execution = nullptr;
            return result;
        }
        catch (...)
        {
            execution = nullptr;
            throw;
        }
    }
};
'''


@unittest.skipUnless(shutil.which('g++') or shutil.which('clang++'), 'A C++20 compiler is required')
class EmissionTests(unittest.TestCase):
    def compile_fixture(self, functions, body='', *, data=None, execute=False):
        header, source, _ = CppGenerator(planned(functions, data=data), ROOT / 'client').render()
        unit = header.replace('#pragma once', '') + '\n' + source.replace('#include "script/GeneratedScripts.h"', '')
        compiler = shutil.which('g++') or shutil.which('clang++')
        command = [compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror', '-x', 'c++', '-',
                   '-I' + str(ROOT / 'client/core/public'), '-I' + str(ROOT / 'client/libs/public')]
        with tempfile.TemporaryDirectory(prefix='mbc-native-test-') as directory:
            if execute:
                executable = Path(directory) / 'fixture'
                command += ['-o', str(executable)]
                unit += HOST + '\nint main()\n{\n' + body + '\n}\n'
            else:
                command += ['-fsyntax-only']
            compiled = subprocess.run(command, input=unit, capture_output=True, text=True, timeout=45)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            if execute:
                completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nested_reference_addresses_and_coroutines_execute(self):
        types = (2, 18, 34, 49, 254)
        functions = []
        for asynchronous in (False, True):
            for typ in types:
                rows = [('yield_program', {})] if asynchronous else []
                rows += [('push_data_ref', {'type': typ, 'data_offset': 0}),
                         ('address_of', {}), ('return', {})]
                _, function = recover(rows)
                functions.extend([function, copy.deepcopy(function)])
        data = struct.pack('<III', 41, 17, 93) + bytes(52)
        self.compile_fixture(functions, r'''
    FixtureHost host;
    constexpr int types[] = {3, 19, 35, 50, 255};
    for (int index = 0; index < 20; ++index)
    {
        auto module = createModule(host, "module" + std::to_string(index));
        module->initializeMembers();
        int resumes = 0;
        const auto value = host.run(*module, resumes);
        assert(value.type == types[(index / 2) % 5]);
        assert(value.bits.base != 0 && value.bits.base == value.bits.begin);
        assert(value.bits.end - value.bits.begin == 11);
        const auto stored = host.read<Address>(value.bits);
        assert(stored.base == 41 && stored.begin == 17 && stored.end == 93);
        assert(resumes == (index < 10 ? 1 : 2));
    }
''', data=[data] * len(functions), execute=True)

    def test_nested_reference_load_shift_store_and_deref_execute(self):
        functions, expected = [], []
        for typ in (3, 19, 35, 50, 255):
            for operation in ('load', 'add', 'sub', 'array', 'store_deref'):
                if operation == 'store_deref':
                    rows = [('push_data_ref', {'type': typ, 'data_offset': 24}),
                            ('push_data_ref', {'type': typ - 1, 'data_offset': 0}),
                            ('address_of', {}), ('store', {}), ('stack_frame_reset', {}),
                            ('push_data_ref', {'type': typ, 'data_offset': 24}),
                            ('deref', {})]
                    result = (typ - 1, 41, 17, 93)
                elif operation == 'array':
                    rows = [('push_imm_i8', {'type': 16, 'value': 1}),
                            ('array_index_abs', {'type': typ, 'base': 0, 'element_size': 12,
                                                 'count': -2, 'span': 12})]
                    result = (typ, 74, 60, 101)
                else:
                    rows = [('push_data_ref', {'type': typ, 'data_offset': 0})]
                    displacement = 0
                    if operation != 'load':
                        rows += [('push_imm_i8', {'type': 16, 'value': 2}),
                                 (f'ptr_{operation}_scaled_u16', {'value': 4})]
                        displacement = 8 if operation == 'add' else -8
                    result = (typ, 41 + displacement, 17, 93)
                _, function = recover(rows + [('return', {})])
                functions.append(function)
                expected.append('{' + ', '.join(map(str, result)) + '}')
        body = '    const int expected[][4] = {' + ', '.join(expected) + '};' + r'''
    FixtureHost host;
    for (int index = 0; index < 25; ++index)
    {
        auto module = createModule(host, "module" + std::to_string(index));
        module->initializeMembers();
        int resumes = 0;
        const auto value = host.run(*module, resumes);
        const auto &check = expected[index];
        assert(value.type == check[0]);
        assert(value.bits.base == std::uint32_t(check[1]));
        assert(value.bits.begin == std::uint32_t(check[2]));
        assert(value.bits.end == std::uint32_t(check[3]));
        assert(resumes == 1);
    }
'''
        data = struct.pack('<6I', 41, 17, 93, 74, 60, 101) + bytes(40)
        self.compile_fixture(functions, body, data=[data] * len(functions), execute=True)

    def test_nested_reference_parameter_types_compile(self):
        functions = []
        for typ in (3, 19, 35, 50, 255):
            _, function = recover([('program_prologue', {'signed_count': 1,
                                       'descriptors': [{'type': typ, 'data_offset': 0}]}),
                                   ('yield_program', {}), ('push_data_ref', {'type': typ, 'data_offset': 0}),
                                   ('return', {})])
            functions.extend([function, copy.deepcopy(function)])
        self.compile_fixture(functions)

    def test_unique_and_shared_methods_compile(self):
        for functions in [[constant(3)], [constant(0), constant(1)]]:
            with self.subTest(count=len(functions)):
                self.compile_fixture(functions)

    def test_shared_increment_keeps_instance_state_separate(self):
        _, function = recover([('push_data_ref', {'type': 16, 'data_offset': 0}), ('pre_inc', {}), ('return', {})])
        self.compile_fixture([function, copy.deepcopy(function)], r'''
    FixtureHost host;
    auto first = createModule(host, "MODULE0.mbc");
    auto second = createModule(host, "module1");
    first->initializeMembers();
    second->initializeMembers();
    int resumes = 0;
    assert(integer(host.run(*first, resumes)) == 1);
    assert(integer(host.run(*second, resumes)) == 41);
    assert(integer(host.run(*first, resumes)) == 2);
    assert(resumes == 3);
''', data=[bytes(64), struct.pack('<i', 40) + bytes(60)], execute=True)

    def test_three_yields_require_three_suspensions(self):
        _, function = recover([('yield_program', {}), ('yield_program', {}), ('yield_program', {}),
                               ('push_imm_i8', {'type': 16, 'value': 8}), ('return', {})])
        self.compile_fixture([function], r'''
    FixtureHost host;
    auto module = createModule(host, "module0");
    module->initializeMembers();
    int resumes = 0;
    assert(integer(host.run(*module, resumes)) == 8);
    assert(resumes == 4);
''', execute=True)

    def test_conditional_send_before_eight_yields_executes_both_paths(self):
        rows = conditional(send_current_process(), [('yield_program', {})] * 8 +
                           [('push_imm_i8', {'type': 16, 'value': 42}), ('return', {})])
        _, function = recover(rows)
        self.compile_fixture([function, copy.deepcopy(function)], r'''
    FixtureHost host;
    int path = 0;
    for (const auto name : {"module0", "module1"})
    {
        auto module = createModule(host, name);
        module->initializeMembers();
        int resumes = 0;
        assert(integer(host.run(*module, resumes)) == 42);
        assert(resumes == 9);
        assert(host.processCalls == path++);
    }
''', data=[struct.pack('<i', value) + bytes(60) for value in (0, 1)], execute=True)

    def test_return_epilogue_preserves_optional_value_and_side_effect(self):
        rows = conditional([('push_imm_i8', {'type': 16, 'value': 99})],
                           [('program_restart', {'program_index': 0}), ('return', {})])
        _, function = recover(rows)
        self.compile_fixture([function, copy.deepcopy(function)], r'''
    FixtureHost host;
    int path = 0;
    for (const auto name : {"module0", "module1"})
    {
        auto module = createModule(host, name);
        module->initializeMembers();
        int resumes = 0;
        assert(integer(host.run(*module, resumes)) == 99 * path);
        assert(host.programActions == ++path);
    }
''', data=[struct.pack('<i', value) + bytes(60) for value in (0, 1)], execute=True)

    def test_array_load_and_scalar_descriptor_compile(self):
        for count in [-4, 4]:
            with self.subTest(count=count):
                _, function = recover([('push_imm_i8', {'type': 16, 'value': 1}),
                                       ('array_index_abs', {'type': 16, 'base': 0, 'element_size': 4, 'count': count, 'span': 4}),
                                       ('return', {})])
                self.compile_fixture([function, copy.deepcopy(function)])

    def test_branch_join_compiles_and_executes(self):
        _, function = recover([('push_imm_i8', {'type': 16, 'value': 1}), ('jfalse_rel16', {'target': 4}),
                               ('push_imm_i8', {'type': 16, 'value': 10}), ('jmp_rel16', {'target': 5}),
                               ('push_imm_i8', {'type': 16, 'value': 20}), ('return', {})])
        self.compile_fixture([function], r'''
    FixtureHost host;
    auto module = createModule(host, "module0");
    module->initializeMembers();
    int resumes = 0;
    assert(integer(host.run(*module, resumes)) == 10);
''', execute=True)


if __name__ == '__main__':
    unittest.main()
