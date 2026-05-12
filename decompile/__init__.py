from __future__ import annotations

"""Pretty/source-level MBC decompilation package.

This package is intentionally one-way: it turns decoded MBC bytecode into
linkage-aware pseudo-source and structured AST/projection data. Binary format
parsing lives in :mod:`mbc_format`; reverse compilation lives in :mod:`compile`.
"""
