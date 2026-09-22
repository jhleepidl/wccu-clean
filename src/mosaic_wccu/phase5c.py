from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .benchmark_adapters import AdaptedDocument, load_hotpotqa_with_audit, load_qasper_with_audit
from .conflict import freshness_decision
from .models import Tile, TileVersion, Update
from .phase3i import deterministic_subsample
from .phase3j import paragraph_atoms
from .phase4a import inventory_tiles, query_text, retrieval_cluster_id
from .phase4f import reciprocal_rank_fuse_tile_ranks, validate_complete_label_ranking
from .phase5a import load_peerqa_fresh_docs
from .segmentation import Segment


PHASE5C_OUTPUT_K = 2
PHASE5C_TRACE_DEPTH = 8
PHASE5C_RRF_K = 60
PHASE5C_VALIDATION_REPEATS = 200


@dataclass(frozen=True)
class BridgeFootprint:
    footprint_id: str
    ranges: tuple[Segment, ...]
    source_rank: int


def load_phase5c_dataset(
    dataset: str,
    *,
    qasper: Path | None = None,
    hotpot: Path | None = None,
    peerqa_qa: Path | None = None,
    peerqa_papers: Path | None = None,
    peerqa_qrels: Path | None = None,
    max_docs: int = 500,
) -> tuple[list[AdaptedDocument], dict[str, Any], dict[str, Any]]:
    """Load the consumed records used by the fixed Phase 5C bridge.

    QASPER/Hotpot use the same deterministic <=500-document sample as Phase 4F.
    PeerQA uses the complete Phase 5A eligible sample; after Phase 5A it is consumed.
    """
    if dataset == "qasper-dev":
        if qasper is None:
            raise ValueError("qasper path is required")
        docs, audit = load_qasper_with_audit(qasper)
        docs = deterministic_subsample(docs, max_docs)
        source = {"qasper": str(qasper)}
    elif dataset == "hotpot-dev":
        if hotpot is None:
            raise ValueError("hotpot path is required")
        docs, audit = load_hotpotqa_with_audit(hotpot, invalid_support_policy="skip_record")
        docs = deterministic_subsample(docs, max_docs)
        source = {"hotpot": str(hotpot)}
    elif dataset == "peerqa":
        if peerqa_qa is None or peerqa_papers is None or peerqa_qrels is None:
            raise ValueError("PeerQA qa/papers/qrels paths are required")
        docs, audit, files = load_peerqa_fresh_docs(peerqa_qa, peerqa_papers, peerqa_qrels)
        source = {"peerqa_files": files}
    else:
        raise ValueError(f"unknown Phase5C dataset: {dataset}")
    return list(docs), audit, source


def physical_inventory(doc: AdaptedDocument):
    """Physical-only paragraph inventory with no semantic overlays."""
    atoms, _ = paragraph_atoms(doc)
    return inventory_tiles(atoms, ())


def canonical_gold_ranges(doc: AdaptedDocument) -> tuple[Segment, ...]:
    """Return the first usable source-order gold alternative.

    Phase 5C freezes this before bridge outcomes are produced.  It avoids any
    method-dependent selection among alternative QASPER annotations.  Hotpot and
    PeerQA already have a single gold footprint per adapted query.
    """
    if not doc.tiles:
        raise ValueError(f"{doc.document_id}: bridge case has no gold support footprint")
    ranges = tuple(doc.tiles[0].ranges)
    if not ranges:
        raise ValueError(f"{doc.document_id}: canonical gold alternative is empty")
    return ranges


def _merged_ranges(ranges: Sequence[Segment]) -> tuple[Segment, ...]:
    ordered = sorted((Segment(int(r.start), int(r.end)) for r in ranges), key=lambda r: (r.start, r.end))
    if not ordered:
        return ()
    out = [ordered[0]]
    for r in ordered[1:]:
        prev = out[-1]
        if r.start <= prev.end:
            out[-1] = Segment(prev.start, max(prev.end, r.end))
        else:
            out.append(r)
    return tuple(out)


def range_is_covered(gold: Segment, footprints: Sequence[BridgeFootprint]) -> bool:
    merged = _merged_ranges([r for fp in footprints for r in fp.ranges])
    return any(r.start <= gold.start and r.end >= gold.end for r in merged)


def footprints_from_rrf_trace(
    row: Mapping[str, Any], *, output_k: int = PHASE5C_OUTPUT_K
) -> tuple[BridgeFootprint, ...]:
    ranked = row.get("ranked_top")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("RRF trace missing ranked_top")
    out: list[BridgeFootprint] = []
    for pos, item in enumerate(ranked[: min(output_k, len(ranked))], start=1):
        if not isinstance(item, Mapping):
            raise ValueError("RRF ranked_top item must be an object")
        ranges = item.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("RRF ranked_top item missing ranges")
        segs = tuple(Segment(int(a), int(b)) for a, b in ranges)
        out.append(BridgeFootprint(str(item.get("tile_id") or f"rrf_{pos:02d}"), segs, pos))
    return tuple(out)


def footprints_from_local_fusion(
    case_row: Mapping[str, Any],
    ranking_row: Mapping[str, Any] | None,
    *,
    output_k: int = PHASE5C_OUTPUT_K,
    fusion_k: int = PHASE5C_RRF_K,
) -> tuple[BridgeFootprint, ...]:
    """Use the already-frozen Phase 4F operational RRF+local-Qwen fusion.

    Missing/failed/malformed model results are fail-closed as an empty witness set,
    matching the historical constrained-reranker evaluators.
    """
    if ranking_row is None or ranking_row.get("status") != "SCORED":
        return ()
    candidates = case_row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ()
    try:
        llm = validate_complete_label_ranking(candidates, ranking_row.get("ranked_labels") or [])
        fused = reciprocal_rank_fuse_tile_ranks(candidates, llm, k=fusion_k)
    except Exception:
        return ()
    by_label = {str(c.get("label")): c for c in candidates if isinstance(c, Mapping)}
    out: list[BridgeFootprint] = []
    for pos, label in enumerate(fused[: min(output_k, len(fused))], start=1):
        c = by_label.get(str(label))
        if not c:
            return ()
        try:
            ranges = tuple(Segment(int(a), int(b)) for a, b in c.get("ranges", []))
        except Exception:
            return ()
        if not ranges:
            return ()
        out.append(BridgeFootprint(str(c.get("tile_id") or f"local_{pos:02d}"), ranges, pos))
    return tuple(out)


def load_jsonl_unique(path: Path, *, id_field: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    malformed = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:
            malformed += 1
            continue
        if not isinstance(row, dict):
            malformed += 1
            continue
        rid = str(row.get(id_field) or "")
        if not rid:
            malformed += 1
            continue
        if rid in rows:
            duplicates.add(rid)
            continue
        rows[rid] = row
    return rows, {"rows": len(rows), "malformed_lines": malformed, "duplicate_ids": sorted(duplicates)}


def _footprint_text(doc: AdaptedDocument, fp: BridgeFootprint) -> str:
    return "\n[...]\n".join(doc.text[r.start:r.end].strip() for r in fp.ranges)


def _serialized_witness_bytes(doc: AdaptedDocument, footprints: Sequence[BridgeFootprint]) -> int:
    payload = [
        {
            "id": f"{doc.document_id}::{fp.footprint_id}",
            "version": 1,
            "ranges": [[r.start, r.end] for r in fp.ranges],
        }
        for fp in footprints
    ]
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _validate_once(
    doc: AdaptedDocument,
    footprints: Sequence[BridgeFootprint],
    *,
    critical: Segment | None,
) -> str:
    current: dict[str, Tile] = {}
    reads: list[TileVersion] = []
    for fp in footprints:
        fid = f"{doc.document_id}::{fp.footprint_id}"
        invalidated = critical is not None and range_is_covered(critical, (fp,))
        version = 2 if invalidated else 1
        current[fid] = Tile(fid, version, [(r.start, r.end) for r in fp.ranges])
        reads.append(TileVersion(fid, 1))
    update = Update(
        update_id=f"phase5c::{doc.document_id}",
        reads=reads,
        writes={f"target::{doc.document_id}"},
        physical_footprint=[],
        semantic_dependencies=set(),
        operation="replace",
    )
    return freshness_decision(update, current)


def _timed_validation(
    doc: AdaptedDocument,
    footprints: Sequence[BridgeFootprint],
    *,
    critical: Segment | None,
    repeats: int,
) -> tuple[str, float]:
    if repeats <= 0:
        raise ValueError("validation repeats must be positive")
    decision = ""
    start = time.perf_counter_ns()
    for _ in range(repeats):
        decision = _validate_once(doc, footprints, critical=critical)
    elapsed = time.perf_counter_ns() - start
    return decision, elapsed / repeats / 1e6


def evaluate_bridge_query(
    doc: AdaptedDocument,
    footprints: Sequence[BridgeFootprint],
    *,
    method: str,
    retrieval_latency_seconds: float | None,
    validation_repeats: int = PHASE5C_VALIDATION_REPEATS,
) -> dict[str, Any]:
    gold = canonical_gold_ranges(doc)
    captured = [range_is_covered(g, footprints) for g in gold]
    captured_n = sum(int(x) for x in captured)
    wr = captured_n / len(gold)
    complete = int(captured_n == len(gold))

    safe_decision, safe_ms = _timed_validation(doc, footprints, critical=None, repeats=validation_repeats)
    if safe_decision != "FRESH":
        raise RuntimeError(f"{doc.document_id}:{method}: safe control did not remain FRESH")

    mutation_rows: list[dict[str, Any]] = []
    stale_ms: list[float] = []
    for idx, g in enumerate(gold):
        decision, ms = _timed_validation(doc, footprints, critical=g, repeats=validation_repeats)
        expected = "REVALIDATE" if captured[idx] else "FRESH"
        if decision != expected:
            raise RuntimeError(
                f"{doc.document_id}:{method}: verifier mismatch support={idx} expected={expected} got={decision}"
            )
        stale_ms.append(ms)
        mutation_rows.append({
            "support_index": idx,
            "range": [g.start, g.end],
            "captured": bool(captured[idx]),
            "decision": decision,
            "unsafe_auto_commit": decision == "FRESH",
            "validation_latency_ms": ms,
        })

    context_bytes = sum(len(_footprint_text(doc, fp).encode("utf-8")) for fp in footprints)
    witness_bytes = _serialized_witness_bytes(doc, footprints)
    non_auto = sum(int(x["decision"] != "FRESH") for x in mutation_rows)
    return {
        "document_id": doc.document_id,
        "cluster_id": retrieval_cluster_id(doc),
        "dataset": str(doc.metadata.get("dataset") or "unknown"),
        "method": method,
        "query": query_text(doc),
        "canonical_gold_policy": "first_usable_source_order_alternative",
        "gold_supports": len(gold),
        "captured_supports": captured_n,
        "witness_recall": wr,
        "complete_witness": complete,
        "hidden_dependency": 1 - complete,
        "selected_footprints": [
            {
                "tile_id": fp.footprint_id,
                "source_rank": fp.source_rank,
                "ranges": [[r.start, r.end] for r in fp.ranges],
            }
            for fp in footprints
        ],
        "selected_footprint_count": len(footprints),
        "context_bytes": context_bytes,
        "witness_metadata_bytes": witness_bytes,
        "retrieval_latency_seconds": retrieval_latency_seconds,
        "safe_control": {
            "decision": safe_decision,
            "safe_auto_commit": safe_decision == "FRESH",
            "validation_latency_ms": safe_ms,
        },
        "critical_mutations": mutation_rows,
        "stale_dependent_direct_commits": sum(int(x["decision"] == "FRESH") for x in mutation_rows),
        "unsafe_auto_commits": sum(int(x["unsafe_auto_commit"]) for x in mutation_rows),
        "revalidate_interventions": non_auto,
        "review_or_blocked": 0,
        "mean_validation_latency_ms": mean([safe_ms] + stale_ms),
    }


def aggregate_bridge_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty bridge rows")
    n = len(rows)
    supports = sum(int(r["gold_supports"]) for r in rows)
    captured = sum(int(r["captured_supports"]) for r in rows)
    stale_direct = sum(int(r["stale_dependent_direct_commits"]) for r in rows)
    unsafe = sum(int(r["unsafe_auto_commits"]) for r in rows)
    revalidate = sum(int(r["revalidate_interventions"]) for r in rows)
    safe_controls = n
    safe_auto = sum(int(bool(r["safe_control"]["safe_auto_commit"])) for r in rows)
    total_updates = supports + safe_controls
    return {
        "queries": n,
        "gold_support_mutation_scenarios": supports,
        "safe_control_scenarios": safe_controls,
        "total_update_scenarios": total_updates,
        "witness_recall_macro": mean(float(r["witness_recall"]) for r in rows),
        "witness_recall_micro": captured / max(1, supports),
        "complete_witness_rate": mean(float(r["complete_witness"]) for r in rows),
        "hidden_dependency_rate": mean(float(r["hidden_dependency"]) for r in rows),
        "stale_dependent_direct_commit_rate": stale_direct / max(1, supports),
        "unsafe_auto_commit_rate_on_stale": unsafe / max(1, supports),
        "safe_auto_commit_rate": safe_auto / max(1, safe_controls),
        "revalidate_rate_on_stale": revalidate / max(1, supports),
        "review_or_blocked_rate": 0.0,
        "non_auto_intervention_rate_all_updates": revalidate / max(1, total_updates),
        "mean_context_bytes": mean(float(r["context_bytes"]) for r in rows),
        "mean_witness_metadata_bytes": mean(float(r["witness_metadata_bytes"]) for r in rows),
        "mean_validation_latency_ms": mean(float(r["mean_validation_latency_ms"]) for r in rows),
        "mean_retrieval_latency_seconds": (
            mean(float(r["retrieval_latency_seconds"]) for r in rows if r.get("retrieval_latency_seconds") is not None)
            if any(r.get("retrieval_latency_seconds") is not None for r in rows)
            else None
        ),
    }
