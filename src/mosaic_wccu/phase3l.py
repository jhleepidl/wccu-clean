from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3e import _evaluate_predictions, overlapping_ego_tiles_from_prepared
from .phase3j import (
    LSAHashModel,
    paragraph_atoms,
    prepare_dense_for_atoms,
    prepare_title_bridge_for_atoms,
)
from .semantic_tiles import SemanticTile, prepare_lexical_graph


@dataclass(frozen=True)
class OverlayBudgetConfig:
    """Query-independent budget for materialized semantic overlays.

    Physical paragraph atoms are never removed.  ``max_overlay_ratio`` limits the
    number of semantic overlay objects to ceil(ratio * physical_atoms).
    """

    max_overlay_ratio: float = 0.5
    priority_mode: str = "score_distance"
    locality_exponent: float = 2.0
    lexical_threshold: float = 0.16
    dense_threshold: float = 0.15
    top_k: int = 1

    def validate(self) -> None:
        if self.max_overlay_ratio < 0:
            raise ValueError("max_overlay_ratio must be >= 0")
        if self.priority_mode not in {"score", "score_distance", "score_locality"}:
            raise ValueError("priority_mode must be score, score_distance, or score_locality")
        if self.locality_exponent < 0:
            raise ValueError("locality_exponent must be >= 0")
        if not (0 <= self.lexical_threshold <= 1):
            raise ValueError("lexical_threshold must be in [0,1]")
        if not (0 <= self.dense_threshold <= 1):
            raise ValueError("dense_threshold must be in [0,1]")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")


def _dedupe_keep_max_score(tiles: Sequence[SemanticTile]) -> tuple[SemanticTile, ...]:
    best: dict[tuple[tuple[int, int], ...], SemanticTile] = {}
    for tile in tiles:
        key = tuple(sorted((r.start, r.end) for r in tile.ranges))
        prev = best.get(key)
        if prev is None or float(tile.score) > float(prev.score):
            best[key] = tile
    out = []
    for i, key in enumerate(sorted(best)):
        tile = best[key]
        out.append(SemanticTile(
            tile_id=f"overlay_{i:04d}", ranges=tile.ranges,
            kind="budget_candidate", score=float(tile.score),
        ))
    return tuple(out)


def paragraph_overlay_candidates(
    doc: AdaptedDocument,
    *,
    model: LSAHashModel | None,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    top_k: int = 1,
) -> tuple[tuple, tuple[SemanticTile, ...]]:
    """Return physical atoms and all thresholded multiview+dense overlay candidates."""
    atoms, titles = paragraph_atoms(doc)
    lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
    title = prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=False)
    fused = prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=True)
    specs = [(lexical, lexical_threshold), (title, lexical_threshold), (fused, lexical_threshold)]
    if model is not None:
        dense = prepare_dense_for_atoms(doc, atoms, model)
        specs.append((dense, dense_threshold))
    tiles: list[SemanticTile] = []
    for graph, threshold in specs:
        tiles.extend(overlapping_ego_tiles_from_prepared(
            graph, similarity_threshold=threshold, top_k=top_k,
        ))
    return atoms, _dedupe_keep_max_score(tiles)


def _member_indices(atoms: Sequence, tile: SemanticTile) -> tuple[int, ...]:
    index = {(a.start, a.end): i for i, a in enumerate(atoms)}
    members = []
    for r in tile.ranges:
        i = index.get((r.start, r.end))
        if i is not None:
            members.append(i)
    return tuple(sorted(set(members)))


def _tile_pairs(atoms: Sequence, tile: SemanticTile) -> tuple[tuple[int, int], ...]:
    members = _member_indices(atoms, tile)
    return tuple((members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members)))


def select_overlay_budget(
    atoms: Sequence,
    candidates: Sequence[SemanticTile],
    *,
    max_overlay_ratio: float,
    priority_mode: str = "score_distance",
    locality_exponent: float = 2.0,
) -> tuple[SemanticTile, ...]:
    """Greedily select non-redundant semantic relationships under an object budget.

    Utility is annotation-free. A candidate receives credit only for *new* atom
    pairs not already represented by selected overlays. ``score_distance`` favors
    distant relationships; ``score_locality`` instead favors stronger local
    relationships; ``score`` ignores distance. The tile graph score scales utility.
    """
    if max_overlay_ratio < 0:
        raise ValueError("max_overlay_ratio must be >= 0")
    if priority_mode not in {"score", "score_distance", "score_locality"}:
        raise ValueError("priority_mode must be score, score_distance, or score_locality")
    if locality_exponent < 0:
        raise ValueError("locality_exponent must be >= 0")
    n = len(atoms)
    if n == 0 or not candidates or max_overlay_ratio == 0:
        return ()
    budget = min(len(candidates), int(math.ceil(max_overlay_ratio * n)))
    if budget <= 0:
        return ()

    rows = []
    denom = max(1, n - 1)
    for tile in candidates:
        pairs = _tile_pairs(atoms, tile)
        if not pairs:
            continue
        score = max(0.0, min(1.0, float(tile.score)))
        if priority_mode == "score":
            pair_weights = {pair: score for pair in pairs}
        elif priority_mode == "score_distance":
            pair_weights = {
                pair: score * (1.0 + math.log1p((pair[1] - pair[0]) / denom))
                for pair in pairs
            }
        else:
            # Phase 3T: use locality only as a materialization prior. The
            # semantic candidate set remains unchanged; this merely spends the
            # fixed overlay budget first on stronger, more local relationships.
            pair_weights = {
                pair: score * max(0.0, 1.0 - ((pair[1] - pair[0]) / denom)) ** locality_exponent
                for pair in pairs
            }
        rows.append((tile, pair_weights))

    selected: list[SemanticTile] = []
    represented: set[tuple[int, int]] = set()
    remaining = list(rows)
    while remaining and len(selected) < budget:
        best_idx = -1
        best_key = None
        for idx, (tile, pair_weights) in enumerate(remaining):
            marginal = sum(weight for pair, weight in pair_weights.items() if pair not in represented)
            # Stable tie-breakers prefer compact tiles, then stronger score, then ranges.
            key = (marginal, -len(tile.ranges), float(tile.score), tuple((r.start, r.end) for r in tile.ranges))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx < 0 or best_key is None or best_key[0] <= 0:
            break
        tile, pair_weights = remaining.pop(best_idx)
        selected.append(SemanticTile(
            tile_id=f"budget_{len(selected):04d}", ranges=tile.ranges,
            kind="budgeted_overlay", score=tile.score,
        ))
        represented.update(pair_weights)
    return tuple(selected)


def budgeted_paragraph_predictions(
    doc: AdaptedDocument,
    *,
    model: LSAHashModel | None,
    config: OverlayBudgetConfig,
) -> tuple[tuple, tuple[SemanticTile, ...]]:
    config.validate()
    atoms, candidates = paragraph_overlay_candidates(
        doc, model=model, lexical_threshold=config.lexical_threshold,
        dense_threshold=config.dense_threshold, top_k=config.top_k,
    )
    selected = select_overlay_budget(
        atoms, candidates, max_overlay_ratio=config.max_overlay_ratio,
        priority_mode=config.priority_mode, locality_exponent=config.locality_exponent,
    )
    return atoms, selected


def _atomic_tiles(atoms: Sequence) -> tuple[SemanticTile, ...]:
    return tuple(SemanticTile(
        tile_id=f"physical_{i:04d}", ranges=(a,), kind="physical_atom", score=1.0,
    ) for i, a in enumerate(atoms))


def evaluate_full_inventory(
    documents: Sequence[AdaptedDocument],
    predictions: Sequence[tuple[Sequence, Sequence[SemanticTile]]],
    *,
    method: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate physical atoms + semantic overlays while charging each object once."""
    augmented = [(tuple(atoms), _atomic_tiles(atoms) + tuple(overlays)) for atoms, overlays in predictions]
    out = _evaluate_predictions(
        documents, augmented, method=method, config=config, include_bootstrap=False,
    )
    per_doc = []
    physical = semantic = memberships = 0
    for doc, (atoms, overlays) in zip(documents, predictions):
        p = len(atoms); s = len(overlays); m = sum(len(t.ranges) for t in overlays)
        physical += p; semantic += s; memberships += m
        per_doc.append({
            "document_id": doc.document_id,
            "physical_blocks": p,
            "semantic_tiles": s,
            "semantic_memberships": m,
            "total_objects": p + s,
        })
    n = max(1, len(documents))
    out.update({
        "physical_blocks": physical,
        "semantic_tiles": semantic,
        "semantic_memberships": memberships,
        "mean_physical_blocks_per_document": physical / n,
        "mean_semantic_tiles_per_document": semantic / n,
        "mean_semantic_memberships_per_document": memberships / n,
        "mean_total_objects_per_document": (physical + semantic) / n,
        "per_document": per_doc,
        "inventory_semantics": "physical atoms are always candidates; semantic overlays are extra objects",
    })
    return out


def evaluate_budgeted_paragraph_mosaic(
    documents: Sequence[AdaptedDocument],
    *,
    model: LSAHashModel | None,
    config: OverlayBudgetConfig,
) -> dict[str, Any]:
    predictions = [budgeted_paragraph_predictions(doc, model=model, config=config) for doc in documents]
    return evaluate_full_inventory(
        documents, predictions,
        method="paragraph_budgeted_multiview_dense" if model is not None else "paragraph_budgeted_multiview",
        config={
            "max_overlay_ratio": config.max_overlay_ratio,
            "priority_mode": config.priority_mode,
            "locality_exponent": config.locality_exponent,
            "lexical_threshold": config.lexical_threshold,
            "dense_threshold": config.dense_threshold,
            "top_k": config.top_k,
            "selection": "greedy marginal nonlocal atom-pair utility; no gold/query evidence",
        },
    )


def pareto_frontier(rows: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    """Maximize F1 while minimizing object count."""
    out = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (other["mean_f1"] >= row["mean_f1"] and other["objects"] <= row["objects"] and
                (other["mean_f1"] > row["mean_f1"] or other["objects"] < row["objects"])):
                dominated = True
                break
        if not dominated:
            out.append(dict(row))
    return sorted(out, key=lambda r: (r["objects"], -r["mean_f1"]))


def choose_knee(frontier: Sequence[dict[str, float]]) -> dict[str, float]:
    """Choose closest normalized point to the utopia (max F1, min objects)."""
    if not frontier:
        raise ValueError("frontier is empty")
    f1s = [r["mean_f1"] for r in frontier]; objs = [r["objects"] for r in frontier]
    f_lo, f_hi = min(f1s), max(f1s); o_lo, o_hi = min(objs), max(objs)
    best = None
    for row in frontier:
        f_loss = 0.0 if f_hi == f_lo else (f_hi - row["mean_f1"]) / (f_hi - f_lo)
        o_cost = 0.0 if o_hi == o_lo else (row["objects"] - o_lo) / (o_hi - o_lo)
        dist = math.sqrt(f_loss * f_loss + o_cost * o_cost)
        cand = (dist, row["objects"], -row["mean_f1"], row)
        if best is None or cand[:3] < best[:3]:
            best = cand
    return dict(best[3])


def choose_gain_retention(
    rows: Sequence[dict[str, float]],
    *,
    baseline_f1: float,
    full_f1: float,
    baseline_objects: float,
    min_gain_fraction: float = 0.90,
    max_object_multiple: float = 2.0,
) -> dict[str, float]:
    """Smallest-object operating point meeting explicit gain/cost constraints."""
    gain = max(0.0, full_f1 - baseline_f1)
    eligible = []
    for row in rows:
        retained = 1.0 if gain == 0 else (row["mean_f1"] - baseline_f1) / gain
        multiple = row["objects"] / max(1e-12, baseline_objects)
        enriched = dict(row)
        enriched["retained_full_f1_gain_fraction"] = retained
        enriched["object_multiple_vs_physical"] = multiple
        if retained >= min_gain_fraction and multiple <= max_object_multiple:
            eligible.append(enriched)
    if not eligible:
        raise ValueError("no budget point satisfies requested gain/cost constraints")
    return min(eligible, key=lambda r: (r["objects"], -r["mean_f1"], r.get("ratio", math.inf)))
