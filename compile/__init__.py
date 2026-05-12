from __future__ import annotations

"""Reverse-compilation / emission package for MBC.

This package owns the lossless editable IR and byte-identical MBC writer. It
must not depend on pretty decompiler decisions; pretty text is only a projection
onto this source-of-truth layer.
"""
