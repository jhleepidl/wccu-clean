from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Iterable, Sequence

from .benchmark_adapters import AdaptedDocument, GoldSemanticTile
from .segmentation import Segment, _cosine_bow
from .semantic_tiles import (
    PreparedLexicalGraph,
    SemanticTile,
    lexical_graph_tiles_from_prepared,
    prepare_lexical_graph,
)


@dataclass(frozen=True)
class AdaptiveBlockConfig:
    target_turns: int = 12
    min_turns: int = 4
    max_turns: int = 24
    boundary_window_turns: int = 3
    size_weight: float = 0.35
    cohesion_weight: float = 0.70
    cut_weight: float = 0.40
    metadata_weight: float = 0.08

    def validate(self) -> None:
        if not (0 < self.min_turns <= self.target_turns <= self.max_turns):
            raise ValueError("require 0 < min_turns <= target_turns <= max_turns")
        if self.boundary_window_turns <= 0:
            raise ValueError("boundary_window_turns must be > 0")
        for name in ("size_weight", "cohesion_weight", "cut_weight", "metadata_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class OverlapGraphConfig:
    similarity_threshold: float = 0.08
    top_k: int = 2
    min_shared_terms: int = 1

    def validate(self) -> None:
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0,1]")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.min_shared_terms <= 0:
            raise ValueError("min_shared_terms must be > 0")


def _turn_segments(doc: AdaptedDocument) -> tuple[Segment, ...]:
    raw = doc.metadata.get("turn_ranges", ())
    turns = tuple(Segment(int(a), int(b)) for a, b in raw)
    if not turns:
        raise ValueError(f"{doc.document_id}: missing QMSum turn ranges")
    return turns


def _turn_texts(doc: AdaptedDocument) -> list[str]:
    return [doc.text[s.start:s.end] for s in _turn_segments(doc)]


def _boundary_cohesions(turn_texts: Sequence[str], window_turns: int) -> list[float]:
    """Return lexical cohesion at each boundary k between turn k-1 and k.

    Index 0 and n are sentinels with zero cohesion; a low internal value marks a
    plausible topic boundary. Windowed contexts are deliberately provider-free.
    """
    n = len(turn_texts)
    out = [0.0] * (n + 1)
    for k in range(1, n):
        left = " ".join(turn_texts[max(0, k - window_turns):k])
        right = " ".join(turn_texts[k:min(n, k + window_turns)])
        out[k] = _cosine_bow(left, right)
    return out


def adaptive_turn_blocks(doc: AdaptedDocument, config: AdaptiveBlockConfig) -> tuple[Segment, ...]:
    """Globally optimize variable-length contiguous local blocks over QMSum turns.

    The objective is annotation-free. It balances block-count cost, target-size
    regularity, lexical incoherence *inside* a block, and the cost of cutting through
    a lexically cohesive boundary. All block boundaries remain aligned to official
    transcript turns so downstream offsets stay exact.
    """
    config.validate()
    turns = _turn_segments(doc)
    texts = _turn_texts(doc)
    n = len(turns)
    if n <= config.max_turns and n < config.min_turns:
        return (Segment(turns[0].start, turns[-1].end),)

    cohesion = _boundary_cohesions(texts, config.boundary_window_turns)
    # Prefix of internal incoherence: boundary k contributes 1-cohesion[k].
    prefix = [0.0] * (n + 1)
    for k in range(1, n):
        prefix[k + 1] = prefix[k] + (1.0 - cohesion[k])
    prefix[n] = prefix[n] if n == 0 else prefix[n]

    best = [math.inf] * (n + 1)
    prev = [-1] * (n + 1)
    best[0] = 0.0

    for end in range(1, n + 1):
        min_len = config.min_turns if end != n else 1
        for length in range(min_len, config.max_turns + 1):
            start = end - length
            if start < 0:
                break
            # Avoid a tiny leading fragment unless the whole document is tiny.
            if start > 0 and start < config.min_turns:
                continue
            if math.isinf(best[start]):
                continue
            internal_count = max(0, length - 1)
            if internal_count:
                # Internal boundaries are start+1 .. end-1.
                internal_sum = prefix[end] - prefix[start + 1]
                internal_incoherence = internal_sum / internal_count
            else:
                internal_incoherence = 0.0
            size_penalty = ((length - config.target_turns) / config.target_turns) ** 2
            cut_penalty = cohesion[end] if end < n else 0.0
            cost = (
                best[start]
                + config.metadata_weight
                + config.size_weight * size_penalty
                + config.cohesion_weight * internal_incoherence
                + config.cut_weight * cut_penalty
            )
            if cost < best[end]:
                best[end] = cost
                prev[end] = start

    if math.isinf(best[n]):
        # Deterministic fallback: target-sized groups aligned to turns.
        out = []
        for i in range(0, n, config.target_turns):
            block = turns[i:min(n, i + config.target_turns)]
            out.append(Segment(block[0].start, block[-1].end))
        return tuple(out)

    points = [n]
    cursor = n
    while cursor > 0:
        cursor = prev[cursor]
        if cursor < 0:
            raise RuntimeError("broken adaptive-turn DP backpointer")
        points.append(cursor)
    points.reverse()
    blocks = []
    for a, b in zip(points[:-1], points[1:]):
        blocks.append(Segment(turns[a].start, turns[b - 1].end))
    return tuple(blocks)


def overlapping_ego_tiles_from_prepared(
    prepared: PreparedLexicalGraph,
    *,
    similarity_threshold: float,
    top_k: int,
) -> tuple[SemanticTile, ...]:
    """Create overlapping lexical communities without exclusive union-find membership.

    Each atom is allowed to seed its own semantic ego tile using its strongest
    non-adjacent lexical links. Distinct ego sets are deduplicated, but membership is
    otherwise intentionally overlapping: an atom may participate in many tiles.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    atoms = prepared.atoms
    neighbors: list[list[tuple[float, int]]] = [[] for _ in atoms]
    for sim, i, j in prepared.candidates:
        if sim < similarity_threshold:
            break
        neighbors[i].append((sim, j))
        neighbors[j].append((sim, i))

    candidates: list[tuple[float, tuple[int, ...]]] = []
    for center, rows in enumerate(neighbors):
        if not rows:
            continue
        rows = sorted(rows, key=lambda x: (-x[0], x[1]))[:top_k]
        members = tuple(sorted({center, *(j for _, j in rows)}))
        if len(members) < 2:
            continue
        score = sum(sim for sim, _ in rows) / len(rows)
        candidates.append((score, members))

    # Keep the best score for an exact member set; overlapping but non-identical sets survive.
    best_by_members: dict[tuple[int, ...], float] = {}
    for score, members in candidates:
        best_by_members[members] = max(score, best_by_members.get(members, -math.inf))

    tiles: list[SemanticTile] = []
    for idx, (members, score) in enumerate(sorted(best_by_members.items(), key=lambda x: (x[0][0], x[0]))):
        ranges = tuple(atoms[i] for i in members)
        tiles.append(SemanticTile(
            tile_id=f"overlap_{idx:04d}",
            ranges=ranges,
            kind="overlap_ego_graph",
            score=score,
        ))
    return tuple(tiles)


def _range_intersection_length(a: Segment, b: Segment) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _union_length(ranges: Sequence[Segment]) -> int:
    if not ranges:
        return 0
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    total = 0
    start, end = ordered[0].start, ordered[0].end
    for r in ordered[1:]:
        if r.start <= end:
            end = max(end, r.end)
        else:
            total += end - start
            start, end = r.start, r.end
    return total + end - start


def _tile_scores(gold: GoldSemanticTile, pred: SemanticTile) -> tuple[float, float, float]:
    # Union pairwise intersection pieces before measuring overlap. This prevents
    # double-counting when either side contains overlapping ranges.
    overlap = _intersection_union_length(gold.ranges, pred.ranges)
    gold_len = max(1, _union_length(gold.ranges))
    pred_len = max(1, _union_length(pred.ranges))
    precision = overlap / pred_len
    recall = overlap / gold_len
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1



def _merged_union_length(ranges: Sequence[Segment]) -> int:
    if not ranges:
        return 0
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    total = 0
    a, b = ordered[0].start, ordered[0].end
    for r in ordered[1:]:
        if r.start <= b:
            b = max(b, r.end)
        else:
            total += b - a
            a, b = r.start, r.end
    return total + (b - a)


def _intersection_union_length(left: Sequence[Segment], right: Sequence[Segment]) -> int:
    # Both lists are tiny for our benchmark tiles; pairwise intersections followed by union is clear and safe.
    pieces: list[Segment] = []
    for a in left:
        for b in right:
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if lo < hi:
                pieces.append(Segment(lo, hi))
    return _merged_union_length(pieces)


def _greedy_k_tile_cover(gold, semantic: Sequence[SemanticTile], k: int) -> tuple[float, float, int]:
    """Greedily combine up to k predicted tiles, maximizing marginal gold coverage.

    Returns precision, recall and tiles used. This is an evaluation diagnostic rather than
    a prediction-time oracle: gold is used only to measure how compactly the predicted tile
    inventory can represent a read footprint.
    """
    if k <= 0 or not semantic:
        return 0.0, 0.0, 0
    gold_len = _merged_union_length(gold.ranges)
    chosen: list[SemanticTile] = []
    remaining = list(semantic)
    current_ranges: list[Segment] = []
    current_intersection = 0
    for _ in range(k):
        best = None
        for idx, tile in enumerate(remaining):
            combined = current_ranges + list(tile.ranges)
            inter = _intersection_union_length(gold.ranges, combined)
            pred_len = _merged_union_length(combined)
            precision = inter / pred_len if pred_len else 0.0
            recall = inter / gold_len if gold_len else 0.0
            marginal = inter - current_intersection
            candidate = (marginal, recall, precision, -pred_len, -idx, idx, inter)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[0] <= 0:
            break
        idx = best[-2]
        tile = remaining.pop(idx)
        chosen.append(tile)
        current_ranges.extend(tile.ranges)
        current_intersection = best[-1]
        if current_intersection >= gold_len:
            break
    pred_len = _merged_union_length(current_ranges)
    precision = current_intersection / pred_len if pred_len else 0.0
    recall = current_intersection / gold_len if gold_len else 0.0
    return precision, recall, len(chosen)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0,
            "recall_ge_0_8_rate": 0.0, "complete_coverage_rate": 0.0,
            "complete_coverage_with_2x_overhead_rate": 0.0,
            "complete_coverage_with_1_5x_overhead_rate": 0.0,
            "greedy_complete_coverage_at_2_rate": 0.0, "greedy_complete_coverage_at_4_rate": 0.0,
        }
    n = len(rows)
    eps = 1e-12
    return {
        "count": n,
        "mean_precision": sum(float(r["precision"]) for r in rows) / n,
        "mean_recall": sum(float(r["recall"]) for r in rows) / n,
        "mean_f1": sum(float(r["f1"]) for r in rows) / n,
        "recall_ge_0_8_rate": sum(float(r["recall"]) >= 0.8 - eps for r in rows) / n,
        "complete_coverage_rate": sum(float(r["recall"]) >= 1.0 - eps for r in rows) / n,
        "complete_coverage_with_2x_overhead_rate": sum(
            float(r["recall"]) >= 1.0 - eps and float(r["precision"]) >= 0.5 - eps
            for r in rows
        ) / n,
        "complete_coverage_with_1_5x_overhead_rate": sum(
            float(r["recall"]) >= 1.0 - eps and float(r["precision"]) >= (2.0 / 3.0) - eps
            for r in rows
        ) / n,
        "greedy_complete_coverage_at_2_rate": sum(float(r.get("greedy_recall_at_2", 0.0)) >= 1.0 - eps for r in rows) / n,
        "greedy_complete_coverage_at_4_rate": sum(float(r.get("greedy_recall_at_4", 0.0)) >= 1.0 - eps for r in rows) / n,
    }


def bootstrap_mean_ci(values: Sequence[float], *, seed: int = 20260902, samples: int = 2000) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "low_95": 0.0, "high_95": 0.0, "bootstrap_samples": samples}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return {
        "n": n,
        "mean": sum(values) / n,
        "low_95": lo,
        "high_95": hi,
        "bootstrap_samples": samples,
    }


def deterministic_content_hash_split(
    documents: Sequence[AdaptedDocument], *, holdout_fraction: float = 0.20
) -> tuple[list[AdaptedDocument], list[AdaptedDocument], dict[str, Any]]:
    if not (0.05 <= holdout_fraction <= 0.50):
        raise ValueError("holdout_fraction must be in [0.05, 0.50]")
    scored = []
    for doc in documents:
        digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        scored.append((digest, doc.document_id, doc))
    scored.sort(key=lambda x: (x[0], x[1]))
    holdout_n = max(1, round(len(scored) * holdout_fraction))
    holdout = [doc for _, _, doc in scored[:holdout_n]]
    development = [doc for _, _, doc in scored[holdout_n:]]
    manifest = {
        "method": "sha256(text)-sorted-first-fraction-holdout",
        "holdout_fraction": holdout_fraction,
        "documents": len(scored),
        "development_documents": len(development),
        "holdout_documents": len(holdout),
        "holdout": [
            {"document_id": doc.document_id, "sha256": digest}
            for digest, _, doc in scored[:holdout_n]
        ],
    }
    return development, holdout, manifest


def _evaluate_predictions(
    documents: Sequence[AdaptedDocument],
    predictions: Sequence[tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]],
    *,
    method: str,
    config: dict[str, Any],
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    per_gold: list[dict[str, Any]] = []
    block_counts: list[int] = []
    tile_counts: list[int] = []
    memberships: list[int] = []
    multi_membership_fractions: list[float] = []
    memberships_per_block: list[float] = []
    per_document: list[dict[str, Any]] = []
    for doc, (atoms, semantic) in zip(documents, predictions):
        block_counts.append(len(atoms))
        tile_counts.append(len(semantic))
        memberships.append(sum(len(t.ranges) for t in semantic))
        atom_keys = [(a.start, a.end) for a in atoms]
        member_counts = {key: 0 for key in atom_keys}
        for tile in semantic:
            for r in tile.ranges:
                key = (r.start, r.end)
                if key in member_counts:
                    member_counts[key] += 1
        if atom_keys:
            multi_fraction = sum(v > 1 for v in member_counts.values()) / len(atom_keys)
            memberships_per_atom = sum(member_counts.values()) / len(atom_keys)
            multi_membership_fractions.append(multi_fraction)
            memberships_per_block.append(memberships_per_atom)
        else:
            multi_fraction = 0.0
            memberships_per_atom = 0.0
        per_document.append({
            "document_id": doc.document_id,
            "physical_blocks": len(atoms),
            "semantic_tiles": len(semantic),
            "semantic_memberships": sum(len(t.ranges) for t in semantic),
            "multi_membership_physical_block_fraction": multi_fraction,
            "semantic_memberships_per_physical_block": memberships_per_atom,
        })
        for gold in doc.tiles:
            candidates = [(_tile_scores(gold, pred), pred) for pred in semantic]
            if candidates:
                (p, r, f1), best = max(candidates, key=lambda item: item[0][2])
                pred_ranges = len(best.ranges)
            else:
                p = r = f1 = 0.0
                pred_ranges = 0
            gp2, gr2, gu2 = _greedy_k_tile_cover(gold, semantic, 2)
            gp4, gr4, gu4 = _greedy_k_tile_cover(gold, semantic, 4)
            per_gold.append({
                "document_id": doc.document_id,
                "tile_id": gold.tile_id,
                "source": gold.source,
                "noncontiguous": gold.is_noncontiguous,
                "gold_ranges": len(gold.ranges),
                "best_pred_ranges": pred_ranges,
                "precision": p,
                "recall": r,
                "f1": f1,
                "greedy_precision_at_2": gp2,
                "greedy_recall_at_2": gr2,
                "greedy_tiles_used_at_2": gu2,
                "greedy_precision_at_4": gp4,
                "greedy_recall_at_4": gr4,
                "greedy_tiles_used_at_4": gu4,
            })
    noncontig = [r for r in per_gold if r["noncontiguous"]]
    contiguous = [r for r in per_gold if not r["noncontiguous"]]
    return {
        "method": method,
        "config": config,
        "documents": len(documents),
        "gold_tiles": len(per_gold),
        "physical_blocks": sum(block_counts),
        "semantic_tiles": sum(tile_counts),
        "semantic_memberships": sum(memberships),
        "mean_physical_blocks_per_document": sum(block_counts) / max(1, len(block_counts)),
        "mean_semantic_tiles_per_document": sum(tile_counts) / max(1, len(tile_counts)),
        "mean_semantic_memberships_per_document": sum(memberships) / max(1, len(memberships)),
        "mean_total_objects_per_document": (sum(block_counts) + sum(tile_counts)) / max(1, len(block_counts)),
        "mean_semantic_memberships_per_physical_block": sum(memberships_per_block) / max(1, len(memberships_per_block)),
        "mean_multi_membership_physical_block_fraction": sum(multi_membership_fractions) / max(1, len(multi_membership_fractions)),
        "all_gold": _aggregate(per_gold),
        "noncontiguous_gold": _aggregate(noncontig),
        "contiguous_gold": _aggregate(contiguous),
        "noncontiguous_f1_bootstrap_95": (bootstrap_mean_ci([float(r["f1"]) for r in noncontig]) if include_bootstrap else None),
        "noncontiguous_f1_cluster_bootstrap_95": (clustered_bootstrap_f1_ci(per_gold) if include_bootstrap else None),
        "per_gold": per_gold,
        "per_document": per_document,
    }


def evaluate_overlap_method(
    documents: Sequence[AdaptedDocument],
    block_config: AdaptiveBlockConfig,
    graph_config: OverlapGraphConfig,
) -> dict[str, Any]:
    block_config.validate(); graph_config.validate()
    predictions = []
    for doc in documents:
        atoms = adaptive_turn_blocks(doc, block_config)
        prepared = prepare_lexical_graph(
            doc.text,
            atoms,
            min_shared_terms=graph_config.min_shared_terms,
            link_adjacent=False,
        )
        semantic = overlapping_ego_tiles_from_prepared(
            prepared,
            similarity_threshold=graph_config.similarity_threshold,
            top_k=graph_config.top_k,
        )
        predictions.append((atoms, semantic))
    return _evaluate_predictions(
        documents,
        predictions,
        method="mosaic_overlap_v1",
        config={"block": block_config.__dict__, "graph": graph_config.__dict__},
    )


def evaluate_union_baseline(
    documents: Sequence[AdaptedDocument],
    *,
    turn_group_size: int,
    similarity_threshold: float,
    max_cluster_units: int = 8,
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        turns = _turn_segments(doc)
        atoms = []
        for i in range(0, len(turns), turn_group_size):
            block = turns[i:i + turn_group_size]
            atoms.append(Segment(block[0].start, block[-1].end))
        prepared = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        semantic = lexical_graph_tiles_from_prepared(
            prepared,
            similarity_threshold=similarity_threshold,
            include_atomic_tiles=False,
            max_cluster_units=max_cluster_units,
        )
        predictions.append((tuple(atoms), tuple(semantic)))
    return _evaluate_predictions(
        documents,
        predictions,
        method="phase3d_fixed_union_baseline",
        config={
            "turn_group_size": turn_group_size,
            "similarity_threshold": similarity_threshold,
            "max_cluster_units": max_cluster_units,
        },
    )


def _objective(metrics: dict[str, Any], *, tile_penalty: float, block_penalty: float) -> float:
    return (
        float(metrics["noncontiguous_gold"]["mean_f1"])
        - tile_penalty * float(metrics["mean_semantic_tiles_per_document"])
        - block_penalty * float(metrics["mean_physical_blocks_per_document"])
    )


def phase3e_sweep(
    development: Sequence[AdaptedDocument],
    holdout: Sequence[AdaptedDocument],
    *,
    target_turns_values: Sequence[int] = (8, 12, 16, 24),
    thresholds: Sequence[float] = (0.04, 0.08, 0.12, 0.16),
    top_k_values: Sequence[int] = (1, 2, 3),
    tile_penalty: float = 0.0015,
    block_penalty: float = 0.0003,
) -> dict[str, Any]:
    """Tune MosaicOverlap-v1 only on development, then freeze for fresh holdout."""
    runs: list[dict[str, Any]] = []
    for target in target_turns_values:
        block = AdaptiveBlockConfig(
            target_turns=int(target),
            min_turns=max(2, int(target) // 2),
            max_turns=max(int(target) + 2, int(target) * 2),
        )
        # Cache adaptive blocks + graph scores once for all threshold/top-k settings.
        prepared_docs: list[tuple[AdaptedDocument, tuple[Segment, ...], PreparedLexicalGraph]] = []
        for doc in development:
            atoms = adaptive_turn_blocks(doc, block)
            prepared = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
            prepared_docs.append((doc, atoms, prepared))
        for threshold in thresholds:
            for top_k in top_k_values:
                predictions = []
                docs = []
                for doc, atoms, prepared in prepared_docs:
                    semantic = overlapping_ego_tiles_from_prepared(
                        prepared, similarity_threshold=float(threshold), top_k=int(top_k)
                    )
                    docs.append(doc)
                    predictions.append((atoms, semantic))
                metrics = _evaluate_predictions(
                    docs,
                    predictions,
                    method="mosaic_overlap_v1",
                    config={
                        "block": block.__dict__,
                        "graph": {"similarity_threshold": float(threshold), "top_k": int(top_k), "min_shared_terms": 1},
                    },
                    include_bootstrap=False,
                )
                objective = _objective(metrics, tile_penalty=tile_penalty, block_penalty=block_penalty)
                runs.append({
                    "target_turns": int(target),
                    "threshold": float(threshold),
                    "top_k": int(top_k),
                    "objective": objective,
                    "metrics": {k: v for k, v in metrics.items() if k != "per_gold"},
                })
    best = max(runs, key=lambda r: (r["objective"], -r["target_turns"], -r["top_k"], -r["threshold"]))
    target = int(best["target_turns"])
    frozen_block = AdaptiveBlockConfig(
        target_turns=target,
        min_turns=max(2, target // 2),
        max_turns=max(target + 2, target * 2),
    )
    frozen_graph = OverlapGraphConfig(
        similarity_threshold=float(best["threshold"]),
        top_k=int(best["top_k"]),
    )
    holdout_metrics = evaluate_overlap_method(holdout, frozen_block, frozen_graph)
    return {
        "selection_split": "qmsum_train_content_hash_development",
        "evaluation_split": "qmsum_train_content_hash_fresh_holdout",
        "tile_penalty": tile_penalty,
        "block_penalty": block_penalty,
        "selected": {
            "target_turns": target,
            "min_turns": frozen_block.min_turns,
            "max_turns": frozen_block.max_turns,
            "threshold": frozen_graph.similarity_threshold,
            "top_k": frozen_graph.top_k,
            "development_objective": best["objective"],
        },
        "development_sweep": runs,
        "holdout_metrics": holdout_metrics,
    }



def clustered_bootstrap_f1_ci(rows: Sequence[dict[str, Any]], *, seed: int = 20260902, samples: int = 4000) -> dict[str, Any]:
    filtered = [r for r in rows if r.get("noncontiguous")]
    by_doc: dict[str, list[float]] = {}
    for r in filtered:
        by_doc.setdefault(str(r["document_id"]), []).append(float(r["f1"]))
    docs = sorted(by_doc)
    if not docs:
        return {"documents": 0, "tiles": 0, "mean": 0.0, "low_95": 0.0, "high_95": 0.0, "bootstrap_samples": samples}
    rng = random.Random(seed)
    boots = []
    for _ in range(samples):
        sampled = [docs[rng.randrange(len(docs))] for _ in docs]
        values = [v for doc in sampled for v in by_doc[doc]]
        boots.append(sum(values) / max(1, len(values)))
    boots.sort()
    values = [v for doc in docs for v in by_doc[doc]]
    return {
        "documents": len(docs),
        "tiles": len(values),
        "mean": sum(values) / len(values),
        "low_95": boots[int(0.025 * (samples - 1))],
        "high_95": boots[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
    }


def paired_cluster_bootstrap_f1_difference(
    left: dict[str, Any], right: dict[str, Any], *, seed: int = 20260902, samples: int = 4000
) -> dict[str, Any]:
    lrows = {(r["document_id"], r["tile_id"]): r for r in left["per_gold"] if r["noncontiguous"]}
    rrows = {(r["document_id"], r["tile_id"]): r for r in right["per_gold"] if r["noncontiguous"]}
    keys = sorted(set(lrows) & set(rrows))
    by_doc: dict[str, list[float]] = {}
    for key in keys:
        doc = str(key[0])
        by_doc.setdefault(doc, []).append(float(lrows[key]["f1"]) - float(rrows[key]["f1"]))
    docs = sorted(by_doc)
    if not docs:
        return {"documents": 0, "tiles": 0, "mean_difference": 0.0, "low_95": 0.0, "high_95": 0.0, "bootstrap_samples": samples}
    rng = random.Random(seed)
    boots = []
    for _ in range(samples):
        sampled = [docs[rng.randrange(len(docs))] for _ in docs]
        vals = [v for doc in sampled for v in by_doc[doc]]
        boots.append(sum(vals) / max(1, len(vals)))
    boots.sort()
    vals = [v for doc in docs for v in by_doc[doc]]
    return {
        "documents": len(docs),
        "tiles": len(vals),
        "mean_difference": sum(vals) / len(vals),
        "low_95": boots[int(0.025 * (samples - 1))],
        "high_95": boots[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
        "left_minus_right": f"{left['method']} - {right['method']}",
    }

def paired_bootstrap_f1_difference(
    left: dict[str, Any], right: dict[str, Any], *, noncontiguous_only: bool = True,
    seed: int = 20260902, samples: int = 4000,
) -> dict[str, Any]:
    lrows = {(r["document_id"], r["tile_id"]): r for r in left["per_gold"] if (r["noncontiguous"] or not noncontiguous_only)}
    rrows = {(r["document_id"], r["tile_id"]): r for r in right["per_gold"] if (r["noncontiguous"] or not noncontiguous_only)}
    keys = sorted(set(lrows) & set(rrows))
    diffs = [float(lrows[k]["f1"]) - float(rrows[k]["f1"]) for k in keys]
    if not diffs:
        return {"n": 0, "mean_difference": 0.0, "low_95": 0.0, "high_95": 0.0, "bootstrap_samples": samples}
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(samples):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {
        "n": n,
        "mean_difference": sum(diffs) / n,
        "low_95": boots[int(0.025 * (samples - 1))],
        "high_95": boots[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
        "left_minus_right": f"{left['method']} - {right['method']}",
    }
