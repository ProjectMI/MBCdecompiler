"""MBL VM tools with a VM-first IR architecture."""

__all__ = [
    "IR_CONTRACT_VERSION",
    "VMIR_CONTRACT_VERSION",
    "VMFunctionIR",
    "VMModuleIR",
    "build_callable_index",
    "build_facts",
    "build_control_graph",
    "build_function_ir",
    "build_module_ir",
    "render_function_text",
    "render_module_text",
    "resolve_call_target",
]


def __getattr__(name: str):
    if name in __all__:
        if name == "build_facts":
            from .facts import build_facts
            return build_facts
        if name == "build_control_graph":
            from .control import build_control_graph
            return build_control_graph
        from . import ir
        return getattr(ir, name)
    raise AttributeError(name)
