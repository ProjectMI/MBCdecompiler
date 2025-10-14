from mbcdisasm.ir import IRCall, IRCallCleanup, IRStackEffect, IRTerminator
from mbcdisasm.optimizer import classify_node_traits


def test_cleanup_traits_marked_structural() -> None:
    cleanup = IRCallCleanup(steps=(IRStackEffect(mnemonic="stack_teardown", pops=3),))
    traits = classify_node_traits(cleanup)
    assert not traits.creates_new_paths
    assert not traits.alias_barrier
    assert not traits.merge_barrier


def test_terminator_traits_match_cleanup() -> None:
    terminator = IRTerminator(operand=0)
    traits = classify_node_traits(terminator)
    assert not traits.creates_new_paths
    assert not traits.alias_barrier
    assert not traits.merge_barrier


def test_regular_call_retains_barrier_traits() -> None:
    call = IRCall(target=0x1234, args=tuple())
    traits = classify_node_traits(call)
    assert traits.creates_new_paths
    assert traits.alias_barrier
    assert traits.merge_barrier
