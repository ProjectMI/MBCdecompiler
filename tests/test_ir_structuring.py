from __future__ import annotations

from pathlib import Path

from mbcdisasm.ir import IRNormalizer
from mbcdisasm.ir.model import (
    IRAsciiHeader,
    IRAsciiWrapperCall,
    IRFunctionPrologue,
    IRLiteralBlock,
    IRTailcallReturn,
)
from mbcdisasm.knowledge import KnowledgeBase
from mbcdisasm.mbc import MbcContainer


def _normalise(script: str, segment: int | None = None):
    knowledge = KnowledgeBase.load(Path("knowledge/manual_annotations.json"))
    container = MbcContainer.load(Path(f"mbc/{script}.mbc"), Path(f"mbc/{script}.adb"))
    normalizer = IRNormalizer(knowledge)
    if segment is None:
        return normalizer.normalise_container(container)
    return normalizer.normalise_container(container, segment_indices=[segment])


def test_literal_block_and_tailcall_return() -> None:
    program = _normalise("_table", segment=0)
    segment = program.segments[0]
    block = segment.blocks[0]

    literal_block = block.nodes[0]
    assert isinstance(literal_block, IRLiteralBlock)
    assert literal_block.kind == "reduce_chain"
    assert literal_block.trailer == (0x6910,)
    assert all(group[:2] == (0x6704, 0x0067) for group in literal_block.groups)

    tailcall = next(node for node in block.nodes if isinstance(node, IRTailcallReturn))
    assert tailcall.target == 0x003D
    assert tailcall.values == ("ret0", "ret1", "ret2", "ret3")


def test_ascii_header_collapse() -> None:
    program = _normalise("_char", segment=0)
    block = program.segments[0].blocks[0]
    header = block.nodes[0]
    assert isinstance(header, IRAsciiHeader)
    assert "script" in header.text
    assert header.trailers


def test_function_prologue_and_ascii_wrapper() -> None:
    program = _normalise("_chat", segment=0)
    first_block = program.segments[0].blocks[0]
    assert isinstance(first_block.nodes[0], IRFunctionPrologue)

    wrapper: IRAsciiWrapperCall | None = None
    for block in program.segments[0].blocks:
        for node in block.nodes:
            if isinstance(node, IRAsciiWrapperCall):
                wrapper = node
                break
        if wrapper:
            break

    assert wrapper is not None
    assert wrapper.tail is True
    assert wrapper.ascii_text
    assert wrapper.then_target is not None
    assert wrapper.else_target is not None


def test_ascii_wrapper_branch_targets() -> None:
    program = _normalise("_bank", segment=0)
    block = program.segments[0].blocks[0]
    wrapper = next(node for node in block.nodes if isinstance(node, IRAsciiWrapperCall))
    assert wrapper.ascii_text == "#H#H"
    assert wrapper.then_target == 0x0031
    assert wrapper.else_target == 0x00DC
