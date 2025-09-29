"""Helpers for rendering reconstructed Lua source code.

This module centralises formatting concerns used by the high level
reconstructor.  Historically the renderer inside :mod:`mbcdisasm.highlevel`
emitted plain strings which quickly became unwieldy once more features were
added (helper prelude generation, configurable comment placement, smart blank
line handling).  The new formatter provides a small toolkit that keeps the
rendering decisions encapsulated and ensures that all reconstruction entry
points share the same layout rules.

The design is intentionally opinionated: we favour stable output over raw
performance so that subsequent manual clean-up is predictable.  The helper
registry is similarly focused on human readable stubs that document the known
behaviour of VM helpers without trying to perfectly emulate Sphere's runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


class LuaWriter:
    """Incremental Lua pretty printer.

    The writer keeps track of indentation, blank line management and label
    alignment.  Every component that wants to contribute source lines only
    needs to call :meth:`write_line` or :meth:`write_label` without worrying
    about spacing rules.  The writer mirrors a tiny subset of what a templating
    language would provide while keeping the implementation small and fully
    self-contained.
    """

    def __init__(self, indent: str = "  ") -> None:
        self._indent = 0
        self._indent_unit = indent
        self._lines: List[str] = []
        self._pending_blank = False
        self._saw_content = False

    # ------------------------------------------------------------------
    # basic line emission helpers
    # ------------------------------------------------------------------
    def write_line(self, text: str = "", *, align: bool = True) -> None:
        """Append ``text`` to the output honouring indentation rules.

        Parameters
        ----------
        text:
            Line content to append.  When empty the writer schedules a blank
            line which will materialise once a non-empty line is written.
        align:
            When ``True`` (the default) the current indentation level is
            prepended to the line.  Callers that wish to bypass indentation –
            for example to emit labels – should pass ``align=False``.
        """

        if not text:
            # Consecutive blank lines collapse to a single one.
            if self._saw_content:
                self._pending_blank = True
            return

        if self._pending_blank:
            self._lines.append("")
            self._pending_blank = False

        if align:
            self._lines.append(f"{self._indent_unit * self._indent}{text}")
        else:
            self._lines.append(text)
        self._saw_content = True

    def write_label(self, label: str) -> None:
        """Emit a label without indentation and ensure surrounding spacing."""

        if self._lines and self._lines[-1] != "":
            self.write_line("")
        self.write_line(f"::{label}::", align=False)

    def ensure_blank_line(self) -> None:
        """Request a blank line unless one is already pending."""

        if self._lines and self._lines[-1] == "":
            return
        if not self._saw_content:
            return
        self._pending_blank = True

    def write_comment(self, text: str) -> None:
        """Emit a single comment line with proper indentation."""

        if not text:
            self.write_line("--")
        else:
            self.write_line(f"-- {text}")

    def write_comment_block(self, lines: Sequence[str]) -> None:
        """Emit a multi-line comment block."""

        for line in lines:
            self.write_comment(line)

    # ------------------------------------------------------------------
    # indentation helpers
    # ------------------------------------------------------------------
    @contextmanager
    def indented(self) -> Iterator[None]:
        """Context manager that increases indentation within the ``with`` body."""

        self.indent()
        try:
            yield
        finally:
            self.dedent()

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        if self._indent == 0:
            raise ValueError("indentation underflow")
        self._indent -= 1

    # ------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------
    def render(self) -> str:
        """Return the accumulated Lua source code."""

        return "\n".join(self._lines).rstrip() + "\n"


@dataclass
class LuaRenderOptions:
    """Customisation knobs that influence Lua rendering."""

    max_inline_comment: int = 100
    deduplicate_comments: bool = True
    emit_stub_metadata: bool = True
    emit_enum_metadata: bool = True
    emit_module_summary: bool = True
    emit_literal_report: bool = True


class CommentFormatter:
    """Utility that wraps comment text for Lua output.

    The formatter accepts free-form summaries supplied by the manual
    annotations.  Many of those strings are fairly long and end up overflowing
    the 80 character mark if left untouched.  To keep the generated Lua easy to
    skim we gently reflow the text while honouring sentence boundaries where
    possible.
    """

    def __init__(self, width: int = 78) -> None:
        self._width = width

    def wrap(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = text.split()
        if not tokens:
            return [text]
        lines: List[str] = []
        current: List[str] = []
        current_len = 0
        for token in tokens:
            projected = current_len + len(token) + (1 if current else 0)
            if projected > self._width and current:
                lines.append(" ".join(current))
                current = [token]
                current_len = len(token)
            else:
                current.append(token)
                current_len = projected
        if current:
            lines.append(" ".join(current))
        return lines

    def format_inline(self, text: str) -> str:
        """Return a single line representation suitable for trailing comments."""

        wrapped = self.wrap(text)
        if not wrapped:
            return ""
        if len(wrapped) == 1:
            return wrapped[0]
        return wrapped[0] + " …"


@dataclass
class HelperSignature:
    """Describes the interface of a reconstructed helper function."""

    name: str
    summary: str
    inputs: int
    outputs: int
    uses_operand: bool

    def parameters(self) -> List[str]:
        params = [f"arg{i}" for i in range(1, self.inputs + 1)]
        if self.uses_operand:
            params.append("operand")
        return params

    def return_stub(self) -> Optional[str]:
        if self.outputs <= 0:
            return None
        if self.outputs == 1:
            return "return nil"
        values = ", ".join("nil" for _ in range(self.outputs))
        return f"return {values}"

    def metadata_lines(self) -> List[str]:
        lines = []
        descriptor = [f"inputs={self.inputs}", f"outputs={self.outputs}"]
        if self.uses_operand:
            descriptor.append("uses_operand=yes")
        else:
            descriptor.append("uses_operand=no")
        lines.append("; ".join(descriptor))
        return lines


@dataclass
class MethodSignature(HelperSignature):
    """Specialisation for methods attached to struct contexts."""

    struct: str = "struct"
    method: str = ""


@dataclass
class HelperRegistry:
    """Collect and render helper/method stubs."""

    _functions: Dict[str, HelperSignature] = field(default_factory=dict)
    _methods: Dict[str, Dict[str, MethodSignature]] = field(default_factory=dict)

    def register_function(self, signature: HelperSignature) -> None:
        existing = self._functions.get(signature.name)
        if existing:
            # Preserve the most informative summary we have.
            if not existing.summary and signature.summary:
                existing.summary = signature.summary
            return
        self._functions[signature.name] = signature

    def register_method(self, signature: MethodSignature) -> None:
        bucket = self._methods.setdefault(signature.struct, {})
        existing = bucket.get(signature.method)
        if existing:
            if not existing.summary and signature.summary:
                existing.summary = signature.summary
            return
        bucket[signature.method] = signature

    # ------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------
    def render(
        self,
        writer: LuaWriter,
        comments: CommentFormatter,
        *,
        options: Optional[LuaRenderOptions] = None,
    ) -> None:
        opts = options or LuaRenderOptions()
        for struct_name, methods in sorted(self._methods.items()):
            writer.write_line(f"local {struct_name} = {{}}")
            writer.ensure_blank_line()
            for method_name, signature in sorted(methods.items()):
                params = signature.parameters()
                param_list = ", ".join(params)
                writer.write_line(
                    f"function {signature.struct}:{method_name}({param_list})"
                )
                with writer.indented():
                    for line in comments.wrap(signature.summary):
                        writer.write_comment(line)
                    if opts.emit_stub_metadata:
                        for line in signature.metadata_lines():
                            writer.write_comment(line)
                    stub = signature.return_stub()
                    if stub:
                        writer.write_line(stub)
                writer.write_line("end")
                writer.ensure_blank_line()

        for name, signature in sorted(self._functions.items()):
            params = signature.parameters()
            param_list = ", ".join(params)
            writer.write_line(f"local function {name}({param_list})")
            with writer.indented():
                for line in comments.wrap(signature.summary):
                    writer.write_comment(line)
                if opts.emit_stub_metadata:
                    for line in signature.metadata_lines():
                        writer.write_comment(line)
                stub = signature.return_stub()
                if stub:
                    writer.write_line(stub)
            writer.write_line("end")
            writer.ensure_blank_line()

    def is_empty(self) -> bool:
        return not self._functions and not self._methods

    def function_count(self) -> int:
        return len(self._functions)

    def method_count(self) -> int:
        return sum(len(methods) for methods in self._methods.values())

    def struct_count(self) -> int:
        return len(self._methods)


@dataclass
class EnumNamespace:
    """Container for a single enumeration namespace."""

    name: str
    description: Optional[str] = None
    values: Dict[int, str] = field(default_factory=dict)


class EnumRegistry:
    """Collect enumeration namespaces referenced during reconstruction."""

    def __init__(self) -> None:
        self._namespaces: Dict[str, EnumNamespace] = {}

    def register(
        self,
        namespace: str,
        value: int,
        label: str,
        *,
        description: Optional[str] = None,
    ) -> None:
        bucket = self._namespaces.get(namespace)
        if bucket is None:
            bucket = EnumNamespace(name=namespace, description=description)
            self._namespaces[namespace] = bucket
        if description and not bucket.description:
            bucket.description = description
        bucket.values[value] = label

    def is_empty(self) -> bool:
        return not self._namespaces

    def namespace_count(self) -> int:
        return len(self._namespaces)

    def total_values(self) -> int:
        return sum(len(namespace.values) for namespace in self._namespaces.values())

    def render(
        self,
        writer: LuaWriter,
        *,
        options: Optional[LuaRenderOptions] = None,
    ) -> None:
        opts = options or LuaRenderOptions()
        for namespace in sorted(self._namespaces.values(), key=lambda item: item.name):
            if opts.emit_enum_metadata:
                meta = [f"enum {namespace.name}: {len(namespace.values)} entries"]
                if namespace.description:
                    meta.append(namespace.description)
                writer.write_comment_block(meta)
            writer.write_line(f"local {namespace.name} = {{")
            with writer.indented():
                for value, label in sorted(namespace.values.items()):
                    writer.write_line(f"[{value}] = \"{label}\",")
            writer.write_line("}")
            writer.write_line("")


def ensure_trailing_blank(writer: LuaWriter) -> None:
    """Ensure a blank line at the end of a section."""

    if not writer._lines:
        return
    if writer._lines[-1] != "":
        writer.write_line("")


def format_comment_block(lines: Sequence[str]) -> List[str]:
    """Prefix every line in ``lines`` with Lua comment markers."""

    return [f"-- {line}" if line else "--" for line in lines]


def join_sections(sections: Iterable[str]) -> str:
    """Join multiple rendered sections using blank lines."""

    cleaned = [section.rstrip() for section in sections if section.strip()]
    return "\n\n".join(cleaned) + ("\n" if cleaned else "")

