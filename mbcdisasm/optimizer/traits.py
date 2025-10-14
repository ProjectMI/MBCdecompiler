"""Helpers describing optimisation-relevant traits for IR nodes."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.model import IRCallCleanup, IRTerminator, IRNode


@dataclass(frozen=True)
class IRNodeTraits:
    """Lightweight classification exposing optimiser friendly hints.

    The optimiser cares about three high level aspects when deciding whether
    a node should act as a barrier:

    ``creates_new_paths``
        True when the node potentially introduces additional CFG edges.  Linear
        helpers such as cleanups and epilogues leave this set to ``False``.

    ``alias_barrier``
        True when the node may invalidate alias information (for example calls
        that could write to arbitrary memory).  Epilogue style cleanups merely
        adjust the VM stack and therefore set this flag to ``False`` which
        allows value propagation across them.

    ``merge_barrier``
        True when the node should keep its surrounding basic block intact.
        Structural helpers that are best collapsed into their neighbours keep
        this flag ``False`` so that block compaction can elide them.
    """

    creates_new_paths: bool = True
    alias_barrier: bool = True
    merge_barrier: bool = True


def classify_node_traits(node: IRNode) -> IRNodeTraits:
    """Return optimisation traits for ``node``.

    The helper centralises small pieces of domain knowledge so that other
    passes do not have to duplicate ``isinstance`` checks.  New IR node types
    can extend this function over time without affecting existing callers.
    """

    if isinstance(node, (IRCallCleanup, IRTerminator)):
        # Cleanups and explicit terminators do not influence the surrounding
        # control flow graph and are safe to fold into neighbouring blocks.  They
        # only adjust the call frame, making them equivalent to epilogues for
        # alias analysis purposes.
        return IRNodeTraits(
            creates_new_paths=False,
            alias_barrier=False,
            merge_barrier=False,
        )
    return IRNodeTraits()


__all__ = ["IRNodeTraits", "classify_node_traits"]
