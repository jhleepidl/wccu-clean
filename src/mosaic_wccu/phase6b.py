from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3j import paragraph_atoms
from .phase5c import BridgeFootprint
from .segmentation import Segment

PHASE6B_BASE_K = 2
PHASE6B_TRACE_DEPTH = 8


@dataclass(frozen=True)
class BudgetControlResult:
    footprints: tuple[BridgeFootprint, ...]
    target_context_bytes: int
    base_context_bytes: int
    achieved_context_bytes: int
    budget_feasible: bool
    added_count: int
    candidate_count: int
    rule: str


def _fp_text_bytes(doc: AdaptedDocument, fp: BridgeFootprint) -> int:
    text = "\n[...]\n".join(doc.text[r.start:r.end].strip() for r in fp.ranges)
    return len(text.encode("utf-8"))


def footprint_context_bytes(doc: AdaptedDocument, fps: Sequence[BridgeFootprint]) -> int:
    return sum(_fp_text_bytes(doc, fp) for fp in fps)


def physical_trace_footprints(row: Mapping[str, Any], *, limit: int = PHASE6B_TRACE_DEPTH) -> tuple[BridgeFootprint, ...]:
    ranked = row.get("ranked_top")
    if not isinstance(ranked, list) or len(ranked) < PHASE6B_BASE_K:
        raise ValueError("physical RRF trace must contain at least two ranked physical atoms")
    out: list[BridgeFootprint] = []
    seen: set[str] = set()
    for pos, item in enumerate(ranked[:limit], start=1):
        if not isinstance(item, Mapping) or str(item.get("kind")) != "physical_atom":
            raise ValueError("Phase6B physical trace contains a non-physical item")
        tid = str(item.get("tile_id") or "")
        ranges = item.get("ranges")
        if not tid or not isinstance(ranges, list) or len(ranges) != 1:
            raise ValueError("Phase6B requires single-range physical paragraph atoms")
        if tid in seen:
            raise ValueError("duplicate physical tile ID in RRF trace")
        seen.add(tid)
        a, b = ranges[0]
        out.append(BridgeFootprint(tid, (Segment(int(a), int(b)),), pos))
    return tuple(out)


def _base_and_budget(doc: AdaptedDocument, trace: Sequence[BridgeFootprint], target_context_bytes: int):
    if target_context_bytes < 0:
        raise ValueError("target context byte budget must be non-negative")
    base = tuple(trace[:PHASE6B_BASE_K])
    if len(base) != PHASE6B_BASE_K:
        raise ValueError("Phase6B requires the fixed physical top-2 baseline")
    base_bytes = footprint_context_bytes(doc, base)
    feasible = base_bytes <= target_context_bytes
    return base, base_bytes, feasible


def ranked_budget_control(
    doc: AdaptedDocument,
    physical_rrf_row: Mapping[str, Any],
    *,
    target_context_bytes: int,
    trace_depth: int = PHASE6B_TRACE_DEPTH,
) -> BudgetControlResult:
    """Expand physical top-2 using the already-frozen physical RRF rank order.

    Candidates are ranks 3..trace_depth.  A full paragraph is appended if and
    only if the resulting context text remains at or below the per-query target.
    Oversize candidates are skipped and later candidates are still considered.
    If the original top-2 already exceed the target, the original top-2 are kept
    intact and the row is marked budget-infeasible; no truncation is allowed.
    """
    trace = physical_trace_footprints(physical_rrf_row, limit=trace_depth)
    base, base_bytes, feasible = _base_and_budget(doc, trace, target_context_bytes)
    selected = list(base)
    current = base_bytes
    candidates = list(trace[PHASE6B_BASE_K:])
    if feasible:
        for fp in candidates:
            b = _fp_text_bytes(doc, fp)
            if current + b <= target_context_bytes:
                selected.append(fp)
                current += b
    return BudgetControlResult(
        footprints=tuple(selected),
        target_context_bytes=int(target_context_bytes),
        base_context_bytes=base_bytes,
        achieved_context_bytes=current,
        budget_feasible=feasible,
        added_count=max(0, len(selected) - PHASE6B_BASE_K),
        candidate_count=len(candidates),
        rule="physical_rrf_ranks_3_to_8_greedy_whole_paragraph_under_constrained_context_byte_cap",
    )


def _atom_index_by_range(doc: AdaptedDocument) -> tuple[tuple[Segment, ...], dict[tuple[int, int], int]]:
    atoms, _ = paragraph_atoms(doc)
    by_range = {(int(a.start), int(a.end)): i for i, a in enumerate(atoms)}
    if len(by_range) != len(atoms):
        raise ValueError(f"{doc.document_id}: duplicate physical paragraph ranges")
    return atoms, by_range


def adjacency_budget_control(
    doc: AdaptedDocument,
    physical_rrf_row: Mapping[str, Any],
    *,
    target_context_bytes: int,
) -> BudgetControlResult:
    """Expand physical top-2 with source-adjacent paragraphs under the same cap.

    Candidate order is frozen as increasing distance from either baseline atom,
    then source paragraph index.  Each paragraph is considered once.  Full
    paragraphs are greedily appended if they fit the target byte cap.
    """
    trace = physical_trace_footprints(physical_rrf_row, limit=PHASE6B_BASE_K)
    base, base_bytes, feasible = _base_and_budget(doc, trace, target_context_bytes)
    atoms, by_range = _atom_index_by_range(doc)
    base_indices: list[int] = []
    for fp in base:
        r = fp.ranges[0]
        key = (int(r.start), int(r.end))
        if key not in by_range:
            raise ValueError(f"{doc.document_id}: physical RRF base atom is not an exact paragraph atom")
        base_indices.append(by_range[key])
    base_set = set(base_indices)
    candidates = []
    for idx in range(len(atoms)):
        if idx in base_set:
            continue
        distance = min(abs(idx - b) for b in base_indices)
        candidates.append((distance, idx))
    candidates.sort(key=lambda x: (x[0], x[1]))
    selected = list(base)
    current = base_bytes
    if feasible:
        for _, idx in candidates:
            atom = atoms[idx]
            fp = BridgeFootprint(f"adjacent_physical_{idx:05d}", (atom,), idx + 1)
            b = _fp_text_bytes(doc, fp)
            if current + b <= target_context_bytes:
                selected.append(fp)
                current += b
    return BudgetControlResult(
        footprints=tuple(selected),
        target_context_bytes=int(target_context_bytes),
        base_context_bytes=base_bytes,
        achieved_context_bytes=current,
        budget_feasible=feasible,
        added_count=max(0, len(selected) - PHASE6B_BASE_K),
        candidate_count=len(candidates),
        rule="paragraph_distance_from_physical_top2_then_source_order_greedy_whole_paragraph_under_constrained_context_byte_cap",
    )


def attach_budget_metadata(row: dict[str, Any], control: BudgetControlResult) -> dict[str, Any]:
    out = dict(row)
    target = max(1, int(control.target_context_bytes))
    out["phase6b_budget_control"] = {
        "rule": control.rule,
        "target_context_bytes": control.target_context_bytes,
        "base_context_bytes": control.base_context_bytes,
        "achieved_context_bytes": control.achieved_context_bytes,
        "budget_feasible": control.budget_feasible,
        "added_physical_paragraphs": control.added_count,
        "candidate_paragraphs_considered": control.candidate_count,
        "budget_utilization": control.achieved_context_bytes / target,
        "whole_paragraphs_only": True,
        "outcome_dependent_selection": False,
    }
    return out
