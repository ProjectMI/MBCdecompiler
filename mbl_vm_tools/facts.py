from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .ir import VMModuleIR, build_module_ir


FACTS_CONTRACT_VERSION = "vm-facts-v1"


@dataclass(frozen=True)
class CallableFact:
    name: str
    kind: str
    abi_arity: Optional[int]
    source_kind: Optional[str] = None
    is_exported: bool = False
    span: Optional[dict[str, int]] = None
    observed_argc_histogram: dict[str, int] = field(default_factory=dict)
    call_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CallSiteFact:
    caller: str
    word_index: int
    offset: int
    kind: str
    encoded_argc: int
    target_name: str
    target_kind: str
    resolved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VMFactsReport:
    contract: str
    summary: dict[str, Any]
    callables: list[CallableFact]
    call_sites: list[CallSiteFact]
    abi_mismatches: list[dict[str, Any]]
    external_argc_observations: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "summary": self.summary,
            "callables": [c.to_dict() for c in self.callables],
            "call_sites": [c.to_dict() for c in self.call_sites],
            "abi_mismatches": self.abi_mismatches,
            "external_argc_observations": self.external_argc_observations,
        }


def build_facts(module_ir: VMModuleIR) -> VMFactsReport:
    definitions: dict[str, CallableFact] = {}
    observed: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    call_counts: Counter[tuple[str, str]] = Counter()
    call_sites: list[CallSiteFact] = []
    abi_mismatches: list[dict[str, Any]] = []

    for fn in module_ir.functions:
        definitions[fn.name] = CallableFact(
            name=fn.name,
            kind="definition",
            abi_arity=fn.abi.get("arity"),
            source_kind=fn.source_kind,
            is_exported=fn.is_exported,
            span=dict(fn.span),
        )

    for fn in module_ir.functions:
        for call in fn.calls:
            if call.get("kind") != "script":
                continue
            target = call.get("target") or {}
            target_name = str(target.get("target_name") or f"unresolved@{target.get('absolute_target')}")
            target_kind = str(target.get("target_kind") or "unresolved")
            encoded_argc = int(call.get("encoded_argc", 0) or 0)
            resolved = bool(target.get("resolved"))
            observed[(target_name, target_kind)][encoded_argc] += 1
            call_counts[(target_name, target_kind)] += 1
            fact = CallSiteFact(
                caller=fn.name,
                word_index=int(call.get("word_index", -1)),
                offset=int(call.get("offset", -1)),
                kind="script",
                encoded_argc=encoded_argc,
                target_name=target_name,
                target_kind=target_kind,
                resolved=resolved,
            )
            call_sites.append(fact)
            if target_kind == "definition":
                target_def = definitions.get(target_name)
                abi_arity = target_def.abi_arity if target_def else None
                if abi_arity is not None and abi_arity != encoded_argc:
                    abi_mismatches.append({
                        "caller": fn.name,
                        "offset": fact.offset,
                        "target": target_name,
                        "encoded_argc": encoded_argc,
                        "target_abi_arity": abi_arity,
                        "reason": "CALL_SCRIPT encoded argc differs from target AGG/AGG0 ABI arity",
                    })

    callables: list[CallableFact] = []
    seen: set[tuple[str, str]] = set()
    for name, defn in sorted(definitions.items()):
        key = (name, "definition")
        hist = {str(k): v for k, v in sorted(observed.get(key, Counter()).items())}
        callables.append(
            CallableFact(
                name=defn.name,
                kind=defn.kind,
                abi_arity=defn.abi_arity,
                source_kind=defn.source_kind,
                is_exported=defn.is_exported,
                span=defn.span,
                observed_argc_histogram=hist,
                call_count=call_counts.get(key, 0),
            )
        )
        seen.add(key)

    external_argc_observations: dict[str, dict[str, int]] = {}
    for key, hist_counter in sorted(observed.items()):
        if key in seen:
            continue
        name, kind = key
        hist = {str(k): v for k, v in sorted(hist_counter.items())}
        if kind == "external":
            external_argc_observations[name] = hist
        callables.append(
            CallableFact(
                name=name,
                kind=kind,
                abi_arity=None,
                observed_argc_histogram=hist,
                call_count=call_counts.get(key, 0),
            )
        )

    mixed_external_argc = {
        name: hist for name, hist in external_argc_observations.items() if len(hist) > 1
    }
    unresolved_calls = [site for site in call_sites if not site.resolved]
    summary = {
        "contract": FACTS_CONTRACT_VERSION,
        "module": module_ir.module_path,
        "definition_count": len(definitions),
        "callable_fact_count": len(callables),
        "script_call_site_count": len(call_sites),
        "unresolved_script_call_site_count": len(unresolved_calls),
        "definition_abi_mismatch_count": len(abi_mismatches),
        "external_callable_count": sum(1 for c in callables if c.kind == "external"),
        "mixed_external_argc_count": len(mixed_external_argc),
        "policy": "Facts report compares explicit VM call argc against explicit AGG/AGG0 ABI only; it never infers ABI from stack depth.",
    }
    return VMFactsReport(
        contract=FACTS_CONTRACT_VERSION,
        summary=summary,
        callables=callables,
        call_sites=call_sites,
        abi_mismatches=abi_mismatches,
        external_argc_observations=external_argc_observations,
    )


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build module-level VM fact report from VMIR")
    parser.add_argument("module", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit-functions", type=int, default=None)
    args = parser.parse_args(argv)
    module_ir = build_module_ir(args.module, limit_functions=args.limit_functions)
    report = build_facts(module_ir).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
