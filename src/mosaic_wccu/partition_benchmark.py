from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .segmentation import Segment, Segmentation, default_policies
from .workload import read_jsonl


NONCOMMUTE = {"REBASE", "REVALIDATE", "REVIEW", "REJECT"}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    policy: str
    oracle: str
    shares_segment: bool
    semantic_guard: bool
    false_conflict: bool
    unsafe_accept_physical: bool
    unsafe_accept_guarded: bool
    write_amp_a: float
    write_amp_b: float
    segment_count: int
    storage_factor: float


def _intersects(seg: Segment, span: tuple[int, int]) -> bool:
    start, end = span
    return max(seg.start, start) < min(seg.end, end)


def _touching(segmentation: Segmentation, span: tuple[int, int]) -> list[int]:
    return [i for i, seg in enumerate(segmentation.segments) if _intersects(seg, span)]


def _write_amp(segmentation: Segmentation, span: tuple[int, int]) -> float:
    touched = _touching(segmentation, span)
    modified = max(1, span[1] - span[0])
    versioned = sum(segmentation.segments[i].length for i in touched)
    return versioned / modified


def _semantic_guard(case: dict[str, Any]) -> bool:
    a = case["edit_a"]
    b = case["edit_b"]
    wa = set(a.get("writes", []))
    wb = set(b.get("writes", []))
    da = set(a.get("semantic_dependencies", []))
    db = set(b.get("semantic_dependencies", []))
    if wa & wb or da & wb or db & wa:
        return True
    # Fail closed if a write anchor is ambiguous in the base snapshot.
    for edit in (a, b):
        anchor = edit.get("anchor")
        if isinstance(anchor, dict) and anchor.get("exact_quote"):
            quote = anchor["exact_quote"]
            if case["base_text"].count(quote) != 1:
                return True
    return False


def evaluate_case(case: dict[str, Any], oracle: dict[str, Any], policy_name: str, fn: Callable[[str], Segmentation]) -> CaseResult:
    text = case["base_text"]
    segmentation = fn(text)
    segmentation.validate(text)
    a = tuple(case["edit_a"]["physical_footprint"])
    b = tuple(case["edit_b"]["physical_footprint"])
    touched_a = set(_touching(segmentation, a))
    touched_b = set(_touching(segmentation, b))
    shares = bool(touched_a & touched_b)
    guard = _semantic_guard(case)
    decision = oracle["decision"]
    return CaseResult(
        case_id=case["case_id"],
        policy=policy_name,
        oracle=decision,
        shares_segment=shares,
        semantic_guard=guard,
        false_conflict=(decision == "COMMUTE" and shares),
        unsafe_accept_physical=(decision in NONCOMMUTE and not shares),
        unsafe_accept_guarded=(decision in NONCOMMUTE and not (shares or guard)),
        write_amp_a=_write_amp(segmentation, a),
        write_amp_b=_write_amp(segmentation, b),
        segment_count=len(segmentation.segments),
        storage_factor=segmentation.stored_chars / max(1, len(text)),
    )


def run_benchmark(cases_path: Path, oracle_path: Path, policies: dict[str, Callable[[str], Segmentation]] | None = None, split: str | None = None) -> dict[str, Any]:
    policies = policies or default_policies()
    cases = [c for c in read_jsonl(cases_path) if c.get("case_type") == "concurrency" and (split is None or c.get("split") == split)]
    oracle_rows = {o["case_id"]: o for o in read_jsonl(oracle_path) if o.get("case_type") == "concurrency"}
    rows: list[CaseResult] = []
    for policy_name, fn in policies.items():
        for case in cases:
            rows.append(evaluate_case(case, oracle_rows[case["case_id"]], policy_name, fn))

    summary: dict[str, Any] = {}
    for policy_name in policies:
        subset = [r for r in rows if r.policy == policy_name]
        safe = [r for r in subset if r.oracle == "COMMUTE"]
        unsafe = [r for r in subset if r.oracle in NONCOMMUTE]
        summary[policy_name] = {
            "cases": len(subset),
            "false_conflicts": sum(r.false_conflict for r in subset),
            "false_conflict_rate": (sum(r.false_conflict for r in safe) / len(safe)) if safe else 0.0,
            "unsafe_accepts_physical": sum(r.unsafe_accept_physical for r in subset),
            "unsafe_accept_rate_physical": (sum(r.unsafe_accept_physical for r in unsafe) / len(unsafe)) if unsafe else 0.0,
            "unsafe_accepts_guarded": sum(r.unsafe_accept_guarded for r in subset),
            "unsafe_accept_rate_guarded": (sum(r.unsafe_accept_guarded for r in unsafe) / len(unsafe)) if unsafe else 0.0,
            "mean_write_amplification": mean([r.write_amp_a for r in subset] + [r.write_amp_b for r in subset]),
            "mean_segment_count": mean(r.segment_count for r in subset),
            "mean_storage_factor": mean(r.storage_factor for r in subset),
        }
    return {
        "summary": summary,
        "rows": [r.__dict__ for r in rows],
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    headers = [
        "policy", "cases", "false_conflicts", "false_conflict_rate", "unsafe_accepts_physical",
        "unsafe_accept_rate_physical", "unsafe_accepts_guarded", "unsafe_accept_rate_guarded",
        "mean_write_amplification", "mean_segment_count", "mean_storage_factor",
    ]
    lines = [",".join(headers)]
    for policy, row in result["summary"].items():
        vals = [policy] + [str(row[h]) for h in headers[1:]]
        lines.append(",".join(vals))
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
