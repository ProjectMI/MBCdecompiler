from __future__ import annotations

"""Small smoke checks for the MBC decompiler package.

Run from the repository parent with:
    python -m mbcproj.smoke_tests [path/to/mbc-or-dir]
"""

from pathlib import Path
import sys

from .bytecode import MbcControlFlow, MbcDecoder
from .decompiler import decompile_to_text, load_project_for
from .loader import MbcHeader, MbcProgram, MbcScript, MbcLoader
from .opcodes import decode_opcode
from .vm_ast import AstStatement, build_program_ast, normalize_ast_statements


def _synthetic_script() -> MbcScript:
    code = bytes([0x7C, 0x48, 0x23])  # yield_program; halt_interpreter; end_program
    program = MbcProgram(index=0, name="synthetic_coroutine", start=0, end=len(code) - 1, state_raw=0, queue_id=0, unknown_48=0)
    return MbcScript(
        path=Path("synthetic.mbc"),
        header=MbcHeader("MBL script v4.0", 0, 0, len(code), 0),
        code=code,
        data=b"",
        programs=[program],
        functions=[],
        metadata=b"",
    )


def test_yield_decode() -> None:
    ins = decode_opcode(bytes([0x7C, 0x48]), 0)
    assert ins.mnemonic == "yield_program"
    assert ins.terminal is True
    assert ins.operands["resume_target"] == 1
    assert [(edge.kind, edge.dst) for edge in ins.edges] == [("yield_resume", 1)]


def test_yield_cfg_and_ast() -> None:
    script = _synthetic_script()
    program = script.programs[0]
    instructions = MbcControlFlow(script, decoder=MbcDecoder(script, annotate_linkage=False)).decode_program(program)
    assert [ins.offset for ins in instructions] == [0, 1], [ins.offset for ins in instructions]
    assert [ins.mnemonic for ins in instructions] == ["yield_program", "halt_interpreter"]
    ast = build_program_ast(script, program, instructions, linker=None, scope_map={})
    source = str(ast["source"])
    assert "yield_program(); // suspend; resumes at loc_00000001" in source
    assert "loc_00000001:" in source
    assert "halt_interpreter();" in source


def test_yield_chain_is_coalesced() -> None:
    code = bytes([0x7C, 0x7C, 0x7C, 0x48, 0x23])
    script = MbcScript(
        path=Path("synthetic_yield_chain.mbc"),
        header=MbcHeader("MBL script v4.0", 0, 0, len(code), 0),
        code=code,
        data=b"",
        programs=[MbcProgram(index=0, name="synthetic_yield_chain", start=0, end=len(code) - 1, state_raw=0, queue_id=0, unknown_48=0)],
        functions=[],
        metadata=b"",
    )
    program = script.programs[0]
    instructions = MbcControlFlow(script, decoder=MbcDecoder(script, annotate_linkage=False)).decode_program(program)
    assert [ins.offset for ins in instructions] == [0, 1, 2, 3], [ins.offset for ins in instructions]
    ast = build_program_ast(script, program, instructions, linker=None, scope_map={})
    source = str(ast["source"])
    assert "yield_program(); // suspend x3; resumes at loc_00000003" in source
    assert "loc_00000003:" in source
    assert "loc_00000001:" not in source
    assert "loc_00000002:" not in source
    assert source.count("yield_program();") == 1
    assert "halt_interpreter();" in source




def _stmt(
    offset: int,
    kind: str,
    text: str,
    *,
    target: int | None = None,
    condition: str | None = None,
    branch_when: str | None = None,
    fallthrough: int | None = None,
    short_circuit: str | None = None,
) -> AstStatement:
    operands = {}
    if target is not None:
        operands["target"] = target
    if condition is not None:
        operands["condition"] = condition
    if branch_when is not None:
        operands["branch_when"] = branch_when
    if fallthrough is not None:
        operands["fallthrough"] = fallthrough
    if short_circuit is not None:
        operands["short_circuit"] = short_circuit
    return AstStatement(offset=offset, file_offset=offset + 0x20, kind=kind, text=text, opcode=-1, mnemonic=kind, operands=operands)


def _source_has_adjacent_labels(source: str) -> bool:
    previous_was_label = False
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_label = line.startswith("loc_") and line.endswith(":")
        if is_label and previous_was_label:
            return True
        previous_was_label = is_label
    return False


def test_backward_goto_after_yield_is_not_erased() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(poll())) goto loc_00000014;", target=20, condition="poll()", branch_when="false", fallthrough=4),
        _stmt(4, "yield", "yield_program(); // suspend; resumes at loc_00000005", target=5),
        _stmt(5, "goto", "goto loc_00000000;", target=0),
        _stmt(20, "return", "return 0;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "goto loc_00000000;" in source
    assert "loc_00000005:" in source
    assert "yield_program();" in source


def test_suppressed_target_labels_are_retargeted_to_visible_code() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(x)) goto loc_0000000A;", target=10, condition="x", branch_when="false", fallthrough=4),
        _stmt(4, "yield", "yield_program(); // suspend; resumes at loc_00000007", target=7),
        _stmt(12, "return", "return 1;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "loc_0000000A:" not in source
    assert "goto loc_0000000C" in source
    assert "loc_0000000C:" in source
    assert not _source_has_adjacent_labels(source)


def test_structured_if_else_and_while() -> None:
    if_else = [
        _stmt(0, "if_goto", "if (!(x > 0)) goto loc_00000014;", target=20, condition="x > 0", branch_when="false"),
        _stmt(4, "call", "positive();"),
        _stmt(8, "goto", "goto loc_00000020;", target=32),
        _stmt(20, "call", "non_positive();"),
        _stmt(32, "return", "return 0;"),
    ]
    if_source = "\n".join(stmt.text for stmt in normalize_ast_statements(if_else))
    assert "if (x > 0)" in if_source
    assert "} else {" in if_source
    assert "goto loc_00000020" not in if_source

    loop = [
        _stmt(32, "if_goto", "if (!(i < n)) goto loc_00000050;", target=80, condition="i < n", branch_when="false"),
        _stmt(36, "call", "tick();"),
        _stmt(40, "goto", "goto loc_00000020;", target=32),
        _stmt(80, "return", "return 0;"),
    ]
    loop_source = "\n".join(stmt.text for stmt in normalize_ast_statements(loop))
    assert "while (i < n)" in loop_source
    assert "goto loc_00000020" not in loop_source


def test_structured_switch() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(mode == 1)) goto loc_0000001E;", target=30, condition="mode == 1", branch_when="false"),
        _stmt(10, "call", "one();"),
        _stmt(20, "goto", "goto loc_00000064;", target=100),
        _stmt(30, "if_goto", "if (!(mode == 2)) goto loc_0000003C;", target=60, condition="mode == 2", branch_when="false"),
        _stmt(40, "call", "two();"),
        _stmt(50, "goto", "goto loc_00000064;", target=100),
        _stmt(60, "call", "other();"),
        _stmt(100, "return", "return 0;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "switch (mode)" in source
    assert "case 1:" in source
    assert "case 2:" in source
    assert "default:" in source
    assert "goto loc_00000064" not in source

def test_terminal_switch_chain() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(arg0 == 0)) goto loc_0000001E;", target=30, condition="arg0 == 0", branch_when="false", fallthrough=10),
        _stmt(10, "return", "return 0;"),
        _stmt(30, "if_goto", "if (!(arg0 == 1)) goto loc_0000003C;", target=60, condition="arg0 == 1", branch_when="false", fallthrough=40),
        _stmt(40, "return", "return 0;"),
        _stmt(60, "if_goto", "if (!(arg0 == 2)) goto loc_0000005A;", target=90, condition="arg0 == 2", branch_when="false", fallthrough=70),
        _stmt(70, "return", "return 1;"),
        _stmt(90, "if_goto", "if (!(arg0 == 3)) goto loc_00000078;", target=120, condition="arg0 == 3", branch_when="false", fallthrough=120),
        _stmt(120, "return", "return (-2);"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "switch (arg0)" in source
    assert "case 0:\ncase 1:" in source
    assert "case 2:" in source
    assert "case 3:\ndefault:" in source
    assert "goto loc_" not in source



def test_short_circuit_conditions_are_folded() -> None:
    statements = [
        _stmt(0, "if_goto", "if (a) goto loc_0000000A; // || short-circuit", target=10, condition="a", branch_when="true", fallthrough=1, short_circuit="or"),
        _stmt(10, "if_goto", "if (b) goto loc_00000014; // || short-circuit", target=20, condition="b", branch_when="true", fallthrough=11, short_circuit="or"),
        _stmt(20, "if_goto", "if (!(c)) goto loc_00000032; // && short-circuit", target=50, condition="c", branch_when="false", fallthrough=21, short_circuit="and"),
        _stmt(30, "if_goto", "if (d) goto loc_00000032; // || short-circuit", target=50, condition="d", branch_when="true", fallthrough=31, short_circuit="or"),
        _stmt(50, "if_goto", "if (!(e)) goto loc_0000003C;", target=60, condition="e", branch_when="false", fallthrough=55),
        _stmt(55, "return", "return (-1);"),
        _stmt(60, "return", "return 0;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "short-circuit" not in source
    assert "goto loc_" not in source
    assert "if ((a || b || c) && (d || e))" in source
    assert "return (-1);" in source
    assert "return 0;" in source


def test_inverted_guard_with_nested_if_else() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(cond)) goto loc_00000014;", target=20, condition="cond", branch_when="false", fallthrough=4),
        _stmt(4, "goto", "goto loc_00000064;", target=100),
        _stmt(20, "if_goto", "if (!(x)) goto loc_0000003C;", target=60, condition="x", branch_when="false", fallthrough=30),
        _stmt(30, "call", "a();"),
        _stmt(40, "goto", "goto loc_00000064;", target=100),
        _stmt(60, "call", "b();"),
        _stmt(100, "return", "return 0;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "if (!(cond))" in source
    assert "if (x)" in source
    assert "} else {" in source
    assert "a();" in source and "b();" in source
    assert "goto loc_" not in source



def test_deep_guard_ladder_is_structured() -> None:
    statements = [
        _stmt(0, "if_goto", "if (!(outer)) goto loc_00000064;", target=100, condition="outer", branch_when="false", fallthrough=4),
        _stmt(4, "if_goto", "if (!(a)) goto loc_00000014;", target=20, condition="a", branch_when="false", fallthrough=8),
        _stmt(8, "return", "return 1;"),
        _stmt(20, "if_goto", "if (!(b)) goto loc_00000028;", target=40, condition="b", branch_when="false", fallthrough=24),
        _stmt(24, "return", "return 2;"),
        _stmt(40, "return", "return 3;"),
        _stmt(100, "return", "return 4;"),
    ]
    source = "\n".join(stmt.text for stmt in normalize_ast_statements(statements))
    assert "if (outer)" in source
    assert "if (a)" in source
    assert "if (b)" in source
    assert "goto loc_" not in source
    assert "loc_" not in source

def test_real_mbc(path: Path, *, render_text: bool = True) -> None:
    script = MbcLoader.load(path)
    decoder = MbcDecoder(script, annotate_linkage=False, cache_decodes=True)
    flow = MbcControlFlow(script, decoder=decoder)
    total = 0
    yielded = 0
    for program in script.programs[: min(8, len(script.programs))]:
        instructions = flow.decode_program(program, follow_local_calls=False)
        total += len(instructions)
        yielded += sum(1 for ins in instructions if ins.mnemonic == "yield_program")
        for ins in instructions:
            if ins.mnemonic == "yield_program":
                assert any(edge.kind == "yield_resume" and edge.dst == ins.offset + 1 for edge in ins.edges)
    assert total > 0
    if not render_text:
        return

    # Full text render checks the higher-level orchestration for one file.
    project, script2, project_linker = load_project_for(path)
    text = decompile_to_text(script2, project_linker=project_linker)
    assert "function " in text
    if yielded or b"|" in script.code:
        assert "yield_program();" in text or yielded == 0
    assert not _source_has_adjacent_labels(text)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    test_yield_decode()
    test_yield_cfg_and_ast()
    test_yield_chain_is_coalesced()
    test_backward_goto_after_yield_is_not_erased()
    test_suppressed_target_labels_are_retargeted_to_visible_code()
    test_structured_if_else_and_while()
    test_structured_switch()
    test_terminal_switch_chain()
    test_short_circuit_conditions_are_folded()
    test_inverted_guard_with_nested_if_else()
    test_deep_guard_ladder_is_structured()

    if argv:
        path = Path(argv[0])
        if path.is_dir():
            candidates = sorted(path.glob("*.mbc"))[:10]
        else:
            candidates = [path]
        for idx, candidate in enumerate(candidates):
            test_real_mbc(candidate, render_text=(idx == 0))
    print("smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
