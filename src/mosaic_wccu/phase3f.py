from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import re
from statistics import mean
from typing import Any, Iterable, Sequence

from .benchmark_adapters import AdaptedDocument
from .revision_tuning import prepare_revision_cases
from .segmentation import (
    Segment,
    Segmentation,
    _boundary_strength,
    _candidate_boundaries,
    mosaic_dp_chunks,
)
from .semantic_tiles import PreparedLexicalGraph, SemanticTile, prepare_lexical_graph
from .phase3e import _evaluate_predictions, overlapping_ego_tiles_from_prepared


@dataclass(frozen=True)
class HistoryDPConfig:
    """Non-uniform edit-prior variant of MosaicDP.

    With no prior history, normalized density is uniform and the collision term is
    algebraically identical to the Phase-3 MosaicDP squared-length proxy. Historical
    edit spans redistribute (rather than increase) total edit probability mass.
    """

    target_size: int = 48
    min_size: int = 8
    max_size: int = 120
    collision_weight: float = 1.0
    metadata_weight: float = 0.3
    semantic_cut_weight: float = 0.55
    history_weight: float = 4.0
    history_bandwidth_fraction: float = 0.02

    def validate(self) -> None:
        if not (0 < self.min_size <= self.target_size <= self.max_size):
            raise ValueError("require 0 < min_size <= target_size <= max_size")
        if self.collision_weight < 0 or self.metadata_weight < 0 or self.semantic_cut_weight < 0:
            raise ValueError("cost weights must be non-negative")
        if self.history_weight < 0:
            raise ValueError("history_weight must be non-negative")
        if not (0.0 < self.history_bandwidth_fraction <= 0.5):
            raise ValueError("history_bandwidth_fraction must be in (0, 0.5]")


@dataclass(frozen=True)
class HistoryRevision:
    doc_id: str
    revision_depth: int
    normalized_spans: tuple[tuple[float, float], ...]


def normalized_edit_spans(row: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    case = prepare_revision_cases([row])[0]
    n = max(1, len(case.before))
    return tuple((a / n, b / n) for a, b in case.spans)


def build_revision_history_index(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[HistoryRevision, ...]]:
    grouped: dict[str, list[HistoryRevision]] = defaultdict(list)
    for row in rows:
        doc_id = row.get("doc_id")
        depth = row.get("revision_depth")
        if doc_id is None or depth is None:
            continue
        grouped[str(doc_id)].append(
            HistoryRevision(str(doc_id), int(depth), normalized_edit_spans(row))
        )
    return {
        doc_id: tuple(sorted(items, key=lambda r: r.revision_depth))
        for doc_id, items in grouped.items()
    }


def prior_spans_for_row(
    row: dict[str, Any], history_index: dict[str, tuple[HistoryRevision, ...]]
) -> tuple[tuple[float, float], ...]:
    doc_id = row.get("doc_id")
    depth = row.get("revision_depth")
    if doc_id is None or depth is None:
        return ()
    spans: list[tuple[float, float]] = []
    for revision in history_index.get(str(doc_id), ()):
        if revision.revision_depth < int(depth):
            spans.extend(revision.normalized_spans)
    return tuple(spans)


def history_edit_density(
    text_length: int,
    normalized_prior_spans: Sequence[tuple[float, float]],
    *,
    weight: float,
    bandwidth_fraction: float,
) -> list[float]:
    """Build a mean-one triangular-kernel edit-density prior.

    Normalizing the density to mean one is important: history changes *where* the
    optimizer spends boundaries rather than silently increasing its global preference
    for more objects.
    """
    if text_length <= 0:
        return []
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if not (0.0 < bandwidth_fraction <= 0.5):
        raise ValueError("bandwidth_fraction must be in (0, 0.5]")
    density = [1.0] * text_length
    if weight > 0 and normalized_prior_spans:
        radius = max(6, int(round(bandwidth_fraction * text_length)))
        for a, b in normalized_prior_spans:
            if not (0.0 <= a <= b <= 1.0):
                raise ValueError(f"invalid normalized span {(a, b)}")
            center = int(round(((a + b) / 2.0) * max(0, text_length - 1)))
            lo = max(0, center - radius)
            hi = min(text_length, center + radius + 1)
            for pos in range(lo, hi):
                density[pos] += weight * (1.0 - abs(pos - center) / (radius + 1))
    total = sum(density)
    if total <= 0:
        return [1.0] * text_length
    scale = text_length / total
    return [value * scale for value in density]


def mosaic_history_dp_chunks(
    text: str,
    normalized_prior_spans: Sequence[tuple[float, float]],
    config: HistoryDPConfig = HistoryDPConfig(),
) -> Segmentation:
    config.validate()
    if not text:
        return Segmentation("mosaic_history_dp", (Segment(0, 0),))

    density = history_edit_density(
        len(text),
        normalized_prior_spans,
        weight=config.history_weight,
        bandwidth_fraction=config.history_bandwidth_fraction,
    )
    prefix = [0.0]
    for value in density:
        prefix.append(prefix[-1] + value)

    candidates = _candidate_boundaries(text, config.target_size)
    n = len(text)
    best = [math.inf] * len(candidates)
    prev = [-1] * len(candidates)
    best[0] = 0.0

    for j in range(1, len(candidates)):
        end = candidates[j]
        for i in range(j - 1, -1, -1):
            start = candidates[i]
            length = end - start
            if length > config.max_size:
                break
            if length < config.min_size and end != n:
                continue
            risk_mass = prefix[end] - prefix[start]
            # Uniform density => (length / target)^2, exactly matching MosaicDP.
            collision = (
                config.collision_weight
                * (length / max(1, config.target_size))
                * (risk_mass / max(1, config.target_size))
            )
            size_regularizer = 0.10 * (
                (length - config.target_size) / max(1, config.target_size)
            ) ** 2
            cut_penalty = 0.0 if end == n else (
                config.semantic_cut_weight * max(0.0, 1.0 - _boundary_strength(text, end))
            )
            cost = best[i] + collision + size_regularizer + config.metadata_weight + cut_penalty
            if cost < best[j]:
                best[j] = cost
                prev[j] = i

    if math.isinf(best[-1]):
        fallback = mosaic_dp_chunks(
            text,
            target_size=config.target_size,
            min_size=config.min_size,
            max_size=config.max_size,
            collision_weight=config.collision_weight,
            metadata_weight=config.metadata_weight,
            semantic_cut_weight=config.semantic_cut_weight,
        )
        return Segmentation("mosaic_history_dp_fallback", fallback.segments)

    indices = [len(candidates) - 1]
    cursor = indices[0]
    while cursor > 0:
        cursor = prev[cursor]
        if cursor < 0:
            raise RuntimeError("broken history-aware DP backpointer")
        indices.append(cursor)
    indices.reverse()
    points = [candidates[i] for i in indices]
    if points[0] != 0:
        points.insert(0, 0)
    segments = tuple(Segment(a, b) for a, b in zip(points[:-1], points[1:]) if a < b)
    return Segmentation("mosaic_history_dp", segments)


def evaluate_history_policy(
    rows: Sequence[dict[str, Any]],
    history_index: dict[str, tuple[HistoryRevision, ...]],
    config: HistoryDPConfig,
    *,
    history_only: bool = False,
) -> dict[str, Any]:
    amps: list[float] = []
    counts: list[int] = []
    raw: list[dict[str, Any]] = []
    for row in rows:
        priors = prior_spans_for_row(row, history_index)
        if history_only and not priors:
            continue
        case = prepare_revision_cases([row])[0]
        seg = mosaic_history_dp_chunks(case.before, priors, config)
        seg.validate(case.before)
        touched = [
            s for s in seg.segments
            if any(max(s.start, a) < min(s.end, b) for a, b in case.spans)
        ]
        amp = sum(s.length for s in touched) / max(1, case.changed_chars)
        amps.append(amp)
        counts.append(len(seg.segments))
        raw.append({
            "case_id": row["case_id"],
            "doc_id": row.get("doc_id"),
            "revision_depth": row.get("revision_depth"),
            "has_prior_history": bool(priors),
            "prior_span_count": len(priors),
            "write_amplification": amp,
            "segments": len(seg.segments),
        })
    return {
        "policy": "mosaic_history_dp",
        "rows": len(raw),
        "mean_write_amplification": mean(amps) if amps else 0.0,
        "mean_segments_per_document": mean(counts) if counts else 0.0,
        "raw": raw,
    }


def chronological_rows_with_history(
    rows: Sequence[dict[str, Any]],
    history_index: dict[str, tuple[HistoryRevision, ...]],
) -> list[dict[str, Any]]:
    return [row for row in rows if prior_spans_for_row(row, history_index)]


def paired_bootstrap_difference(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    seed: int = 20260902,
    samples: int = 4000,
) -> dict[str, Any]:
    l = {r["case_id"]: float(r["write_amplification"]) for r in left["raw"]}
    r = {r["case_id"]: float(r["write_amplification"]) for r in right["raw"]}
    keys = sorted(set(l) & set(r))
    diffs = [l[k] - r[k] for k in keys]
    if not diffs:
        return {"n": 0, "mean_difference": 0.0, "low_95": 0.0, "high_95": 0.0}
    rng = random.Random(seed)
    n = len(diffs)
    boots = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    boots.sort()
    return {
        "n": n,
        "mean_difference": sum(diffs) / n,
        "low_95": boots[int(0.025 * (samples - 1))],
        "high_95": boots[int(0.975 * (samples - 1))],
        "bootstrap_samples": samples,
        "left_minus_right": f"{left['policy']} - {right['policy']}",
    }


# --- HotpotQA provider-free semantic-edge benchmark helpers -----------------

_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")


def _normalized_phrase(value: str) -> str:
    return " ".join(tok.lower() for tok in _ENTITY_TOKEN_RE.findall(value) if tok)


def prepare_hotpot_title_bridge_graph(
    doc: AdaptedDocument, *, include_lexical_base: bool = True
) -> PreparedLexicalGraph:
    """Sentence graph augmented with query-independent paragraph-title mention edges.

    The graph does not use the Hotpot question, answer, or supporting-fact labels. A
    sentence gets a deterministic bonus when it mentions another paragraph title,
    approximating a Wikipedia entity bridge without an NER model or external API.
    """
    raw_ranges = doc.metadata.get("sentence_ranges", ())
    atoms = tuple(Segment(int(a), int(b)) for a, b in raw_ranges)
    base = (
        prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        if include_lexical_base
        else PreparedLexicalGraph(atoms, ())
    )
    titles = [str(x) for x in doc.metadata.get("sentence_titles", ())]
    paragraph_titles = [str(x) for x in doc.metadata.get("paragraph_titles", ())]
    if len(titles) != len(atoms):
        raise ValueError(f"{doc.document_id}: sentence title metadata mismatch")

    pair_scores: dict[tuple[int, int], float] = {
        (i, j): sim for sim, i, j in base.candidates
    }
    normalized_titles = [_normalized_phrase(t) for t in paragraph_titles]
    sentence_text = [_normalized_phrase(doc.text[a.start:a.end]) for a in atoms]
    title_to_sentence_indices: dict[str, list[int]] = defaultdict(list)
    for idx, title in enumerate(titles):
        title_to_sentence_indices[_normalized_phrase(title)].append(idx)

    for i, content in enumerate(sentence_text):
        own_title = _normalized_phrase(titles[i])
        for target_title in normalized_titles:
            if not target_title or target_title == own_title or len(target_title) < 3:
                continue
            if target_title not in content:
                continue
            for j in title_to_sentence_indices.get(target_title, ()):  # link into mentioned page
                if abs(i - j) <= 1:
                    continue
                key = (min(i, j), max(i, j))
                pair_scores[key] = min(1.0, max(pair_scores.get(key, 0.0), 0.34))

    candidates = tuple(sorted(
        ((score, i, j) for (i, j), score in pair_scores.items()), reverse=True
    ))
    return PreparedLexicalGraph(atoms, candidates)


def evaluate_hotpot_semantic_graph(
    documents: Sequence[AdaptedDocument],
    *,
    similarity_threshold: float = 0.16,
    top_k: int = 1,
    title_bridge: bool = False,
    title_only: bool = False,
    include_bootstrap: bool = False,
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = tuple(Segment(int(a), int(b)) for a, b in doc.metadata.get("sentence_ranges", ()))
        if not atoms:
            raise ValueError(f"{doc.document_id}: no sentence ranges")
        if title_only:
            prepared = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
        elif title_bridge:
            prepared = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
        else:
            prepared = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        semantic = overlapping_ego_tiles_from_prepared(
            prepared,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )
        predictions.append((atoms, semantic))
    return _evaluate_predictions(
        documents,
        predictions,
        method=(
            "hotpot_title_only_overlap" if title_only
            else "hotpot_title_bridge_overlap" if title_bridge
            else "hotpot_frozen_lexical_overlap"
        ),
        config={"similarity_threshold": similarity_threshold, "top_k": top_k},
        include_bootstrap=include_bootstrap,
    )


def evaluate_hotpot_representation_baseline(
    documents: Sequence[AdaptedDocument], *, mode: str
) -> dict[str, Any]:
    """Query-independent simple representation baselines for HotpotQA.

    These baselines deliberately do not inspect the question, answer, or supporting-fact
    labels. They test whether non-contiguous evidence recovery is already trivial under
    whole-context, paragraph, sentence, or contiguous two-sentence representations.
    """
    if mode not in {"whole", "paragraph", "sentence", "sentence_window2"}:
        raise ValueError(f"unknown Hotpot baseline mode: {mode}")
    predictions = []
    for doc in documents:
        sentence_atoms = tuple(Segment(int(a), int(b)) for a, b in doc.metadata.get("sentence_ranges", ()))
        titles = [str(x) for x in doc.metadata.get("sentence_titles", ())]
        if not sentence_atoms:
            raise ValueError(f"{doc.document_id}: no sentence ranges")
        if mode == "whole":
            atoms = (Segment(0, len(doc.text)),)
            semantic = (SemanticTile("whole_0000", atoms, kind="whole"),)
        elif mode == "sentence":
            atoms = sentence_atoms
            semantic = tuple(
                SemanticTile(f"sentence_{i:04d}", (span,), kind="sentence")
                for i, span in enumerate(sentence_atoms)
            )
        elif mode == "sentence_window2":
            atoms = sentence_atoms
            windows = []
            for i in range(len(sentence_atoms)):
                j = min(len(sentence_atoms) - 1, i + 1)
                span = Segment(sentence_atoms[i].start, sentence_atoms[j].end)
                windows.append(SemanticTile(f"window2_{i:04d}", (span,), kind="sentence_window2"))
            semantic = tuple(windows)
        else:
            if len(titles) != len(sentence_atoms):
                raise ValueError(f"{doc.document_id}: sentence title metadata mismatch")
            paragraphs: list[Segment] = []
            start = 0
            while start < len(sentence_atoms):
                end = start + 1
                while end < len(sentence_atoms) and titles[end] == titles[start]:
                    end += 1
                paragraphs.append(Segment(sentence_atoms[start].start, sentence_atoms[end - 1].end))
                start = end
            atoms = tuple(paragraphs)
            semantic = tuple(
                SemanticTile(f"paragraph_{i:04d}", (span,), kind="paragraph")
                for i, span in enumerate(paragraphs)
            )
        predictions.append((atoms, semantic))
    return _evaluate_predictions(
        documents, predictions, method=f"hotpot_{mode}_baseline",
        config={"mode": mode}, include_bootstrap=False,
    )


def _dedupe_semantic_tiles(tiles: Sequence[SemanticTile]) -> tuple[SemanticTile, ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[SemanticTile] = []
    for tile in tiles:
        key = tuple(sorted((r.start, r.end) for r in tile.ranges))
        if key in seen:
            continue
        seen.add(key)
        out.append(SemanticTile(
            f"multiview_{len(out):04d}", tile.ranges, kind="hotpot_multiview", score=tile.score
        ))
    return tuple(out)


def evaluate_hotpot_multiview_inventory(
    documents: Sequence[AdaptedDocument], *, similarity_threshold: float = 0.16, top_k: int = 1
) -> dict[str, Any]:
    """Exploratory multi-view inventory: keep lexical, title-only, and fused tiles.

    This is intentionally an inventory rather than a winner-take-all edge fusion: a
    physical sentence may participate in multiple semantic views simultaneously. HotpotQA
    was already inspected before this method was proposed, so results from this dataset are
    development evidence only and must not be treated as a fresh transfer claim.
    """
    predictions = []
    for doc in documents:
        atoms = tuple(Segment(int(a), int(b)) for a, b in doc.metadata.get("sentence_ranges", ()))
        lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        title_only = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
        fused = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
        tiles: list[SemanticTile] = []
        for prepared in (lexical, title_only, fused):
            tiles.extend(overlapping_ego_tiles_from_prepared(
                prepared, similarity_threshold=similarity_threshold, top_k=top_k
            ))
        predictions.append((atoms, _dedupe_semantic_tiles(tiles)))
    return _evaluate_predictions(
        documents, predictions, method="hotpot_multiview_inventory_exploratory",
        config={
            "views": ["lexical", "title_only", "lexical_plus_title_fused"],
            "similarity_threshold": similarity_threshold, "top_k": top_k,
            "fresh_evidence": False,
        }, include_bootstrap=False,
    )
