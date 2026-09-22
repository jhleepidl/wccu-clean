from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .segmentation import Segment, sentence_chunks, Segmentation, default_policies
from .semantic_tiles import (
    SemanticTile, lexical_graph_tiles_from_prepared, lexical_graph_tiles_from_segments, prepare_lexical_graph,
)


@dataclass(frozen=True)
class GoldSemanticTile:
    tile_id: str
    label: str
    ranges: tuple[Segment, ...]
    source: str

    @property
    def exact_chars(self) -> int:
        return sum(r.length for r in self.ranges)

    @property
    def envelope_chars(self) -> int:
        return max(r.end for r in self.ranges) - min(r.start for r in self.ranges)

    @property
    def is_noncontiguous(self) -> bool:
        ordered = sorted(self.ranges, key=lambda x: (x.start, x.end))
        return len(ordered) > 1 and any(a.end < b.start for a, b in zip(ordered, ordered[1:]))


@dataclass(frozen=True)
class AdaptedDocument:
    document_id: str
    text: str
    tiles: tuple[GoldSemanticTile, ...]
    metadata: dict[str, Any]


def _first(obj: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _is_intlike(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and value.strip().lstrip("-").isdigit()


def _span_pairs(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = _first(value, ("spans", "ranges", "relevant_text_span", "relevant_text_spans"), [])
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_intlike(x) for x in value):
        return [(int(value[0]), int(value[1]))]
    out = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(_is_intlike(x) for x in item):
                out.append((int(item[0]), int(item[1])))
    return out


def adapt_qmsum_record(record: dict[str, Any], *, document_id: str | None = None) -> AdaptedDocument:
    """Adapt one official QMSum meeting record.

    QMSum relevant_text_span indices are transcript-turn spans. We map them to exact
    character ranges in a deterministic canonical transcript. End indices are treated
    as inclusive, matching the public annotation examples.
    """
    turns = _first(record, ("meeting_transcripts", "transcript", "turns"), [])
    if not isinstance(turns, list) or not turns:
        raise ValueError("QMSum record has no transcript turns")
    pieces: list[str] = []
    turn_ranges: list[Segment] = []
    cursor = 0
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {idx} is not an object")
        speaker = str(_first(turn, ("speaker", "role", "name"), "speaker"))
        content = str(_first(turn, ("content", "text", "utterance"), ""))
        piece = f"{speaker}: {content}\n"
        pieces.append(piece)
        turn_ranges.append(Segment(cursor, cursor + len(piece)))
        cursor += len(piece)
    text = "".join(pieces)

    tiles: list[GoldSemanticTile] = []
    annotation_sources = (
        ("topic_list", "topic"),
        ("specific_query_list", "query"),
    )
    tile_idx = 0
    for field, label_key in annotation_sources:
        rows = record.get(field, []) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            span_value = _first(row, ("relevant_text_span", "relevant_text_spans", "relevant_span"), [])
            pairs = _span_pairs(span_value)
            ranges: list[Segment] = []
            for start_turn, end_turn in pairs:
                if start_turn < 0 or end_turn < start_turn or end_turn >= len(turn_ranges):
                    raise ValueError(f"invalid QMSum turn span {(start_turn, end_turn)} for {len(turn_ranges)} turns")
                ranges.append(Segment(turn_ranges[start_turn].start, turn_ranges[end_turn].end))
            if not ranges:
                continue
            label = str(_first(row, (label_key, "query", "topic"), f"tile-{tile_idx}"))
            tiles.append(GoldSemanticTile(
                tile_id=f"qmsum_{tile_idx:05d}", label=label,
                ranges=tuple(ranges), source=field,
            ))
            tile_idx += 1
    doc_id = document_id or str(_first(record, ("meeting_id", "id", "meeting_name"), "qmsum_unknown"))
    return AdaptedDocument(doc_id, text, tuple(tiles), {
        "turn_count": len(turns),
        "turn_ranges": tuple((r.start, r.end) for r in turn_ranges),
        "dataset": "QMSum",
    })


def load_qmsum(path: Path) -> list[AdaptedDocument]:
    content = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(content)
        records = raw if isinstance(raw, list) else [raw]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    return [adapt_qmsum_record(r, document_id=f"{path.stem}:{i}") for i, r in enumerate(records)]


def adapt_longmemeval_record(record: dict[str, Any], *, row_id: str) -> dict[str, Any]:
    """Provider-free structural adapter; it does not call a reader or LLM judge."""
    qtype = str(_first(record, ("question_type", "question_category", "type"), "unknown"))
    sessions = _first(record, ("haystack_sessions", "history", "sessions"), [])
    evidence = _first(record, ("answer_session_ids", "evidence_session_ids", "gold_session_ids"), [])
    return {
        "case_id": row_id,
        "question": str(record.get("question", "")),
        "question_type": qtype,
        "session_count": len(sessions) if isinstance(sessions, list) else None,
        "evidence_session_ids": list(evidence) if isinstance(evidence, list) else [],
        "has_update_signal": "update" in qtype.lower() or "temporal" in qtype.lower(),
    }


def load_longmemeval(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LongMemEval JSON must contain a list")
    return [adapt_longmemeval_record(r, row_id=f"longmemeval:{i}") for i, r in enumerate(raw)]


def adapt_iterater_row(row: dict[str, Any], *, row_id: str) -> dict[str, Any]:
    """Schema-tolerant adapter for IteraTeR distributions.

    The public releases have changed packaging over time, so this intentionally accepts
    common before/after field aliases and fails if a revision pair cannot be located.
    """
    before = _first(row, (
        "before_revision", "before_sent", "before", "source", "src", "original", "input", "sentence1"
    ))
    after = _first(row, (
        "after_revision", "after_sent", "after", "target", "tgt", "revised", "output", "sentence2"
    ))
    if before is None or after is None:
        raise ValueError(f"{row_id}: cannot locate before/after text fields")
    actions = row.get("edit_actions", [])
    if not isinstance(actions, list):
        actions = []
    intent = _first(row, ("major_intent", "intent", "edit_intent", "labels", "label", "revision_type"), None)
    if intent is None and actions:
        action_intents = [str(a.get("major_intent")) for a in actions if isinstance(a, dict) and a.get("major_intent")]
        intent = action_intents[0] if len(set(action_intents)) == 1 and action_intents else ("mixed" if action_intents else "unknown")
    if intent is None:
        intent = "unknown"
    return {
        "case_id": row_id, "before": str(before), "after": str(after), "intent": str(intent),
        "edit_actions": actions, "doc_id": row.get("doc_id"), "revision_depth": row.get("revision_depth"),
    }


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_iterater(path: Path) -> list[dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    return [adapt_iterater_row(r, row_id=f"iterater:{i}") for i, r in enumerate(rows)]


def _intersects(a: Segment, b: Segment) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)


def _union_length(ranges: list[Segment]) -> int:
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


def qmsum_footprint_metrics(
    documents: list[AdaptedDocument],
    policies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure how physical partitions represent QMSum gold semantic footprints."""
    policies = policies or default_policies()
    gold_tiles = [tile for doc in documents for tile in doc.tiles]
    if not gold_tiles:
        raise ValueError("no QMSum semantic annotations found")
    noncontiguous = [t for t in gold_tiles if t.is_noncontiguous]
    envelope_amps = [t.envelope_chars / max(1, t.exact_chars) for t in noncontiguous]
    summary: dict[str, Any] = {}
    for name, fn in policies.items():
        touched_counts: list[int] = []
        physical_amps: list[float] = []
        stored_amps: list[float] = []
        for doc in documents:
            seg = fn(doc.text)
            seg.validate(doc.text)
            for tile in doc.tiles:
                touched = [s for s in seg.segments if any(_intersects(s, r) for r in tile.ranges)]
                touched_counts.append(len(touched))
                physical_amps.append(_union_length(touched) / max(1, tile.exact_chars))
                stored_amps.append(sum(s.length for s in touched) / max(1, tile.exact_chars))
        summary[name] = {
            "mean_segments_touched_per_gold_tile": sum(touched_counts) / len(touched_counts),
            "mean_physical_cover_amplification": sum(physical_amps) / len(physical_amps),
            "mean_versioned_stored_char_amplification": sum(stored_amps) / len(stored_amps),
        }

    # QMSum has a natural public-data baseline unavailable to generic text-only policies:
    # one physical object per transcript turn. Reporting it avoids making sentence/fixed
    # chunk policies look artificially privileged and exposes the object-count tradeoff.
    turn_touched_counts: list[int] = []
    turn_physical_amps: list[float] = []
    turn_stored_amps: list[float] = []
    for doc in documents:
        raw_turn_ranges = doc.metadata.get("turn_ranges", ())
        turn_segments = [Segment(int(a), int(b)) for a, b in raw_turn_ranges]
        for tile in doc.tiles:
            touched = [s for s in turn_segments if any(_intersects(s, r) for r in tile.ranges)]
            turn_touched_counts.append(len(touched))
            turn_physical_amps.append(_union_length(touched) / max(1, tile.exact_chars))
            turn_stored_amps.append(sum(s.length for s in touched) / max(1, tile.exact_chars))
    if turn_touched_counts:
        summary["qmsum_turn"] = {
            "mean_segments_touched_per_gold_tile": sum(turn_touched_counts) / len(turn_touched_counts),
            "mean_physical_cover_amplification": sum(turn_physical_amps) / len(turn_physical_amps),
            "mean_versioned_stored_char_amplification": sum(turn_stored_amps) / len(turn_stored_amps),
        }

    return {
        "documents": len(documents),
        "gold_tiles": len(gold_tiles),
        "noncontiguous_gold_tiles": len(noncontiguous),
        "noncontiguous_rate": len(noncontiguous) / len(gold_tiles),
        "mean_contiguous_envelope_amplification_noncontiguous": (
            sum(envelope_amps) / len(envelope_amps) if envelope_amps else 1.0
        ),
        "policies": summary,
    }


def _range_intersection_length(a: Segment, b: Segment) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _ranges_intersection_length(left: tuple[Segment, ...], right: tuple[Segment, ...]) -> int:
    return sum(_range_intersection_length(a, b) for a in left for b in right)


def _range_set_length(ranges: tuple[Segment, ...]) -> int:
    return _union_length(list(ranges))


def _tile_match_scores(gold: GoldSemanticTile, pred: SemanticTile) -> tuple[float, float, float]:
    overlap = _ranges_intersection_length(gold.ranges, pred.ranges)
    gold_len = max(1, _range_set_length(gold.ranges))
    pred_len = max(1, _range_set_length(pred.ranges))
    precision = overlap / pred_len
    recall = overlap / gold_len
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def _qmsum_grouped_atoms(doc: AdaptedDocument, turn_group_size: int = 1) -> tuple[Segment, ...]:
    if turn_group_size <= 0:
        raise ValueError("turn_group_size must be > 0")
    raw_turn_ranges = doc.metadata.get("turn_ranges", ())
    turns = tuple(Segment(int(a), int(b)) for a, b in raw_turn_ranges)
    if not turns:
        raise ValueError(f"{doc.document_id}: missing QMSum turn ranges")
    if turn_group_size == 1:
        return turns
    grouped: list[Segment] = []
    for i in range(0, len(turns), turn_group_size):
        block = turns[i:i + turn_group_size]
        grouped.append(Segment(block[0].start, block[-1].end))
    return tuple(grouped)


def qmsum_semantic_graph_metrics(
    documents: list[AdaptedDocument],
    *,
    similarity_threshold: float = 0.20,
    min_shared_terms: int = 1,
    max_cluster_units: int = 8,
    turn_group_size: int = 1,
) -> dict[str, Any]:
    """Evaluate provider-free cross-cutting semantic tiles against QMSum gold ranges.

    Gold and predicted spans are compared on exact character coverage induced by official
    transcript-turn annotations. Atomic tiles are excluded from prediction matching so a
    system cannot score well merely by reproducing every transcript turn.
    """
    per_gold: list[dict[str, Any]] = []
    predicted_semantic_tiles = 0
    predicted_noncontiguous_tiles = 0
    total_turns = 0
    for doc in documents:
        raw_turn_ranges = doc.metadata.get("turn_ranges", ())
        total_turns += len(raw_turn_ranges)
        atoms = _qmsum_grouped_atoms(doc, turn_group_size)
        tiles = lexical_graph_tiles_from_segments(
            doc.text,
            atoms,
            similarity_threshold=similarity_threshold,
            min_shared_terms=min_shared_terms,
            include_atomic_tiles=False,
            max_cluster_units=max_cluster_units,
            link_adjacent=False,
        )
        semantic = [t for t in tiles if t.kind == "lexical_graph"]
        predicted_semantic_tiles += len(semantic)
        predicted_noncontiguous_tiles += sum(t.is_noncontiguous for t in semantic)
        for gold in doc.tiles:
            candidates = [(_tile_match_scores(gold, pred), pred) for pred in semantic]
            if candidates:
                (precision, recall, f1), best = max(candidates, key=lambda item: item[0][2])
                best_ranges = len(best.ranges)
            else:
                precision = recall = f1 = 0.0
                best_ranges = 0
            per_gold.append({
                "source": gold.source,
                "noncontiguous": gold.is_noncontiguous,
                "gold_ranges": len(gold.ranges),
                "best_pred_ranges": best_ranges,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        if not rows:
            return {"count": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0}
        return {
            "count": len(rows),
            "mean_precision": sum(float(r["precision"]) for r in rows) / len(rows),
            "mean_recall": sum(float(r["recall"]) for r in rows) / len(rows),
            "mean_f1": sum(float(r["f1"]) for r in rows) / len(rows),
        }

    noncontiguous = [r for r in per_gold if r["noncontiguous"]]
    contiguous = [r for r in per_gold if not r["noncontiguous"]]
    return {
        "documents": len(documents),
        "turns": total_turns,
        "gold_tiles": len(per_gold),
        "similarity_threshold": similarity_threshold,
        "min_shared_terms": min_shared_terms,
        "max_cluster_units": max_cluster_units,
        "turn_group_size": turn_group_size,
        "predicted_semantic_tiles": predicted_semantic_tiles,
        "predicted_noncontiguous_tiles": predicted_noncontiguous_tiles,
        "mean_predicted_semantic_tiles_per_document": predicted_semantic_tiles / max(1, len(documents)),
        "all_gold": aggregate(per_gold),
        "noncontiguous_gold": aggregate(noncontiguous),
        "contiguous_gold": aggregate(contiguous),
    }


def _qmsum_graph_metrics_for_thresholds(
    documents: list[AdaptedDocument],
    thresholds: tuple[float, ...],
    *,
    min_shared_terms: int = 1,
    max_cluster_units: int = 8,
    turn_group_size: int = 1,
) -> dict[float, dict[str, Any]]:
    """Evaluate several thresholds while preparing each document only once."""
    thresholds = tuple(dict.fromkeys(float(t) for t in thresholds))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    states = {
        t: {
            "per_gold": [],
            "predicted_semantic_tiles": 0,
            "predicted_noncontiguous_tiles": 0,
            "total_turns": 0,
        }
        for t in thresholds
    }
    for doc in documents:
        raw_turn_ranges = doc.metadata.get("turn_ranges", ())
        atoms = _qmsum_grouped_atoms(doc, turn_group_size)
        prepared = prepare_lexical_graph(
            doc.text, atoms, min_shared_terms=min_shared_terms, link_adjacent=False
        )
        for threshold in thresholds:
            state = states[threshold]
            state["total_turns"] += len(raw_turn_ranges)
            semantic = list(lexical_graph_tiles_from_prepared(
                prepared,
                similarity_threshold=threshold,
                include_atomic_tiles=False,
                max_cluster_units=max_cluster_units,
            ))
            state["predicted_semantic_tiles"] += len(semantic)
            state["predicted_noncontiguous_tiles"] += sum(t.is_noncontiguous for t in semantic)
            for gold in doc.tiles:
                candidates = [(_tile_match_scores(gold, pred), pred) for pred in semantic]
                if candidates:
                    (precision, recall, f1), best = max(candidates, key=lambda item: item[0][2])
                    best_ranges = len(best.ranges)
                else:
                    precision = recall = f1 = 0.0
                    best_ranges = 0
                state["per_gold"].append({
                    "source": gold.source,
                    "noncontiguous": gold.is_noncontiguous,
                    "gold_ranges": len(gold.ranges),
                    "best_pred_ranges": best_ranges,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        if not rows:
            return {"count": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0}
        return {
            "count": len(rows),
            "mean_precision": sum(float(r["precision"]) for r in rows) / len(rows),
            "mean_recall": sum(float(r["recall"]) for r in rows) / len(rows),
            "mean_f1": sum(float(r["f1"]) for r in rows) / len(rows),
        }

    out: dict[float, dict[str, Any]] = {}
    for threshold, state in states.items():
        rows = state["per_gold"]
        noncontiguous = [r for r in rows if r["noncontiguous"]]
        contiguous = [r for r in rows if not r["noncontiguous"]]
        predicted_semantic_tiles = int(state["predicted_semantic_tiles"])
        out[threshold] = {
            "documents": len(documents),
            "turns": int(state["total_turns"]),
            "gold_tiles": len(rows),
            "similarity_threshold": threshold,
            "min_shared_terms": min_shared_terms,
            "max_cluster_units": max_cluster_units,
            "turn_group_size": turn_group_size,
            "predicted_semantic_tiles": predicted_semantic_tiles,
            "predicted_noncontiguous_tiles": int(state["predicted_noncontiguous_tiles"]),
            "mean_predicted_semantic_tiles_per_document": predicted_semantic_tiles / max(1, len(documents)),
            "all_gold": aggregate(rows),
            "noncontiguous_gold": aggregate(noncontiguous),
            "contiguous_gold": aggregate(contiguous),
        }
    return out


def qmsum_hierarchical_graph_sweep(
    dev_documents: list[AdaptedDocument],
    test_documents: list[AdaptedDocument],
    *,
    turn_group_sizes: tuple[int, ...] = (4, 8, 16, 32),
    thresholds: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
    metadata_penalty: float = 0.002,
    max_cluster_units: int = 8,
) -> dict[str, Any]:
    """Tune a two-level provider-free Mosaic graph on dev and freeze for test.

    Level 1 forms contiguous local physical blocks from consecutive QMSum turns.
    Level 2 links non-adjacent blocks with the lexical graph. This removes the old
    structural ceiling where a semantic tile could contain at most eight individual
    turns even though QMSum gold ranges commonly span tens of turns.
    """
    runs: list[dict[str, Any]] = []
    for group_size in turn_group_sizes:
        metrics_by_threshold = _qmsum_graph_metrics_for_thresholds(
            dev_documents, thresholds, max_cluster_units=max_cluster_units, turn_group_size=group_size
        )
        for threshold in thresholds:
            metrics = metrics_by_threshold[float(threshold)]
            objective = (
                float(metrics["noncontiguous_gold"]["mean_f1"])
                - metadata_penalty * float(metrics["mean_predicted_semantic_tiles_per_document"])
            )
            runs.append({
                "turn_group_size": group_size,
                "threshold": threshold,
                "objective": objective,
                "metrics": metrics,
            })
    best = max(runs, key=lambda row: (row["objective"], -row["turn_group_size"], -row["threshold"]))
    group_size = int(best["turn_group_size"])
    threshold = float(best["threshold"])
    test_metrics = _qmsum_graph_metrics_for_thresholds(
        test_documents, (threshold,), max_cluster_units=max_cluster_units, turn_group_size=group_size
    )[threshold]
    return {
        "selection_split": "dev",
        "evaluation_split": "test",
        "metadata_penalty": metadata_penalty,
        "selected_turn_group_size": group_size,
        "selected_threshold": threshold,
        "max_cluster_units": max_cluster_units,
        "dev_sweep": runs,
        "test_metrics": test_metrics,
        "note": "Exploratory Phase 3D method; QMSum test had already been used for Phase 3C diagnostics and is not a pristine blind test for this method.",
    }


def qmsum_graph_threshold_sweep(
    dev_documents: list[AdaptedDocument],
    test_documents: list[AdaptedDocument],
    *,
    thresholds: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
    metadata_penalty: float = 0.002,
) -> dict[str, Any]:
    """Tune on dev and freeze for test, reusing threshold-independent pair scores.

    The dev objective rewards non-contiguous gold F1 while applying a small object-count
    penalty. The test split is never consulted during selection.
    """
    dev_metrics = _qmsum_graph_metrics_for_thresholds(dev_documents, thresholds)
    dev_runs = []
    for threshold in thresholds:
        metrics = dev_metrics[float(threshold)]
        objective = (
            float(metrics["noncontiguous_gold"]["mean_f1"])
            - metadata_penalty * float(metrics["mean_predicted_semantic_tiles_per_document"])
        )
        dev_runs.append({"threshold": threshold, "objective": objective, "metrics": metrics})
    best = max(dev_runs, key=lambda row: (row["objective"], -row["threshold"]))
    frozen_threshold = float(best["threshold"])
    test_metrics = _qmsum_graph_metrics_for_thresholds(test_documents, (frozen_threshold,))[frozen_threshold]
    return {
        "selection_split": "dev",
        "evaluation_split": "test",
        "metadata_penalty": metadata_penalty,
        "selected_threshold": frozen_threshold,
        "dev_sweep": dev_runs,
        "test_metrics": test_metrics,
    }


def _hotpot_context_rows(context: Any) -> list[tuple[str, list[str]]]:
    """Normalize original HotpotQA and Hugging Face datasets-server context schemas."""
    if isinstance(context, dict):
        titles = context.get("title", []) or []
        sentences = context.get("sentences", []) or []
        if len(titles) != len(sentences):
            raise ValueError("HotpotQA context title/sentences length mismatch")
        return [(str(t), [str(s) for s in ss]) for t, ss in zip(titles, sentences)]
    out: list[tuple[str, list[str]]] = []
    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), [str(s) for s in item[1]]))
            elif isinstance(item, dict):
                title = _first(item, ("title", "name"), "")
                sentences = _first(item, ("sentences", "text"), [])
                if isinstance(sentences, str):
                    sentences = [sentences]
                out.append((str(title), [str(s) for s in sentences]))
    return out


def _hotpot_support_pairs(value: Any) -> list[tuple[str, int]]:
    if isinstance(value, dict):
        titles = value.get("title", []) or []
        sent_ids = value.get("sent_id", []) or []
        if len(titles) != len(sent_ids):
            raise ValueError("HotpotQA supporting_facts title/sent_id length mismatch")
        return [(str(t), int(i)) for t, i in zip(titles, sent_ids)]
    out: list[tuple[str, int]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), int(item[1])))
            elif isinstance(item, dict):
                out.append((str(item.get("title", "")), int(item.get("sent_id", -1))))
    return out


def adapt_hotpotqa_record(record: dict[str, Any], *, document_id: str | None = None) -> AdaptedDocument:
    """Adapt one HotpotQA distractor record into a sentence-offset semantic footprint.

    The canonical text keeps paragraph titles outside sentence ranges. Each sentence
    range includes its trailing newline so consecutive supporting sentences in the same
    paragraph remain physically contiguous; crossing a paragraph title creates a real
    gap and therefore a non-contiguous gold footprint.
    """
    context = _hotpot_context_rows(record.get("context", []))
    if not context:
        raise ValueError("HotpotQA record has no context")
    support = _hotpot_support_pairs(record.get("supporting_facts", []))
    if not support:
        raise ValueError("HotpotQA record has no supporting_facts")

    pieces: list[str] = []
    sentence_ranges: list[Segment] = []
    sentence_titles: list[str] = []
    lookup: dict[tuple[str, int], Segment] = {}
    paragraph_titles: list[str] = []
    cursor = 0
    for title, sentences in context:
        paragraph_titles.append(title)
        header = f"[{title}]\n"
        pieces.append(header)
        cursor += len(header)
        for sent_id, sentence in enumerate(sentences):
            piece = f"{sentence}\n"
            span = Segment(cursor, cursor + len(piece))
            pieces.append(piece)
            sentence_ranges.append(span)
            sentence_titles.append(title)
            lookup[(title, sent_id)] = span
            cursor += len(piece)
    text = "".join(pieces)

    gold_ranges: list[Segment] = []
    missing: list[tuple[str, int]] = []
    for key in support:
        span = lookup.get(key)
        if span is None:
            missing.append(key)
        else:
            gold_ranges.append(span)
    if missing:
        raise ValueError(f"HotpotQA supporting facts missing from context: {missing[:3]}")
    # Deduplicate while preserving source order.
    seen = set()
    deduped = []
    for span in gold_ranges:
        key = (span.start, span.end)
        if key not in seen:
            seen.add(key); deduped.append(span)

    doc_id = document_id or str(record.get("_id") or record.get("id") or "hotpot_unknown")
    tile = GoldSemanticTile(
        tile_id="hotpot_support_00000",
        label=str(record.get("question", "supporting_facts")),
        ranges=tuple(deduped),
        source="supporting_facts",
    )
    return AdaptedDocument(doc_id, text, (tile,), {
        "dataset": "HotpotQA",
        "question": str(record.get("question", "")),
        "answer": str(record.get("answer", "")),
        "question_type": str(record.get("type", "unknown")),
        "level": str(record.get("level", "unknown")),
        "paragraph_titles": tuple(paragraph_titles),
        "sentence_ranges": tuple((r.start, r.end) for r in sentence_ranges),
        "sentence_titles": tuple(sentence_titles),
        "supporting_fact_count": len(deduped),
    })


def _hotpot_invalid_supporting_facts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return supporting-fact annotations that cannot resolve in the supplied context.

    HotpotQA's distractor validation release contains at least one publicly reported
    malformed annotation (record 5ae61bfd5542992663a4f261 has sentence id 902 for a
    five-sentence paragraph).  We do not guess a corrected sentence id.
    """
    context = _hotpot_context_rows(record.get("context", []))
    lengths = {title: len(sentences) for title, sentences in context}
    invalid: list[dict[str, Any]] = []
    for title, sent_id in _hotpot_support_pairs(record.get("supporting_facts", [])):
        if title not in lengths:
            invalid.append({
                "title": title, "sent_id": sent_id, "reason": "missing_title",
                "paragraph_sentence_count": None,
            })
        elif sent_id < 0 or sent_id >= lengths[title]:
            invalid.append({
                "title": title, "sent_id": sent_id, "reason": "sentence_id_out_of_range",
                "paragraph_sentence_count": lengths[title],
            })
    return invalid


def load_hotpotqa_with_audit(
    path: Path, *, invalid_support_policy: str = "skip_record"
) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    """Load HotpotQA while making annotation corruption explicit.

    Policies:
      - ``skip_record`` (default): exclude a record with any unresolved supporting fact.
        This preserves benchmark integrity without inventing a gold correction.
      - ``raise``: fail on the first malformed record.

    The audit payload is intended to be persisted with experiment results.
    """
    if invalid_support_policy not in {"skip_record", "raise"}:
        raise ValueError("invalid_support_policy must be 'skip_record' or 'raise'")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
        records = [item.get("row", item) for item in raw["rows"]]
    elif isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = [raw]
    else:
        raise ValueError("unsupported HotpotQA JSON structure")

    docs: list[AdaptedDocument] = []
    skipped: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        row = dict(record)
        invalid = _hotpot_invalid_supporting_facts(row)
        if invalid:
            record_id = str(row.get("_id") or row.get("id") or f"row:{i}")
            detail = {
                "row_index": i,
                "record_id": record_id,
                "question_type": str(row.get("type", "unknown")),
                "level": str(row.get("level", "unknown")),
                "invalid_supporting_facts": invalid,
            }
            if invalid_support_policy == "raise":
                raise ValueError(f"HotpotQA malformed supporting facts: {detail}")
            skipped.append(detail)
            continue
        docs.append(adapt_hotpotqa_record(row, document_id=f"{path.stem}:{i}"))

    audit = {
        "source_records": len(records),
        "loaded_records": len(docs),
        "skipped_records": len(skipped),
        "invalid_support_policy": invalid_support_policy,
        "skipped": skipped,
    }
    return docs, audit


def load_hotpotqa(path: Path) -> list[AdaptedDocument]:
    docs, _ = load_hotpotqa_with_audit(path, invalid_support_policy="skip_record")
    return docs


# --- 2WikiMultiHopQA ---------------------------------------------------------

def _2wiki_evidence_triples(value: Any) -> list[tuple[str, str, str]]:
    """Normalize 2Wiki ``evidences`` into (subject, relation, object) triples."""
    out: list[tuple[str, str, str]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append((str(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            s = _first(item, ("subject", "s", "head"), "")
            r = _first(item, ("relation", "property", "predicate", "r"), "")
            o = _first(item, ("object", "o", "tail"), "")
            if s or r or o:
                out.append((str(s), str(r), str(o)))
    return out


def adapt_2wiki_record(record: dict[str, Any], *, document_id: str | None = None) -> AdaptedDocument:
    """Adapt one 2WikiMultiHopQA record without using question/evidence labels as input.

    The text/gold supporting-fact footprint follows the exact same canonicalization as
    HotpotQA. Structured Wikidata evidence triples are retained *only* in metadata for
    downstream evaluation and subgroup analysis; tile constructors must not inspect them.
    """
    base = adapt_hotpotqa_record(record, document_id=document_id)
    metadata = dict(base.metadata)
    metadata.update({
        "dataset": "2WikiMultiHopQA",
        "evidence_triples": tuple(_2wiki_evidence_triples(record.get("evidences", []))),
        "evidence_ids": record.get("evidences_id", ()),
        "entity_ids": record.get("entity_ids", ()),
        "answer_id": record.get("answer_id"),
    })
    return AdaptedDocument(base.document_id, base.text, base.tiles, metadata)


def _load_json_or_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON document or newline-delimited JSON objects.

    A JSONL file normally starts with ``{`` too, so inspecting only the first
    non-whitespace character is not enough to distinguish it from a single
    JSON object. Prefer streaming line-by-line parsing for ``.jsonl``/``.ndjson``
    paths; for other suffixes, try ordinary JSON first and fall back to JSONL
    only when the whole file contains multiple top-level JSON values.
    """

    def load_jsonl() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object")
                records.append(dict(item))
        return records

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return load_jsonl()

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        # ``Extra data`` is the common signature of JSONL stored with a .json
        # suffix. Falling back also makes the loader robust to such mirrors.
        try:
            return load_jsonl()
        except ValueError:
            raise exc

    if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
        return [dict(item.get("row", item)) for item in raw["rows"]]
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise ValueError("unsupported JSON structure")


def _load_optional_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "reading 2Wiki parquet requires pyarrow; use the official dev.json or "
            "install pyarrow in a separate benchmark environment"
        ) from exc
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def load_2wiki_with_audit(
    path: Path, *, invalid_support_policy: str = "skip_record"
) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    """Load 2Wiki dev/train data and make unresolved supporting facts explicit."""
    if invalid_support_policy not in {"skip_record", "raise"}:
        raise ValueError("invalid_support_policy must be 'skip_record' or 'raise'")
    if path.suffix.lower() == ".parquet":
        records = _load_optional_parquet_records(path)
        source_format = "parquet"
    else:
        records = _load_json_or_jsonl_records(path)
        source_format = "json"

    docs: list[AdaptedDocument] = []
    skipped: list[dict[str, Any]] = []
    missing_evidence = 0
    for i, row in enumerate(records):
        invalid = _hotpot_invalid_supporting_facts(row)
        if invalid:
            record_id = str(row.get("_id") or row.get("id") or f"row:{i}")
            detail = {
                "row_index": i,
                "record_id": record_id,
                "question_type": str(row.get("type", "unknown")),
                "invalid_supporting_facts": invalid,
            }
            if invalid_support_policy == "raise":
                raise ValueError(f"2Wiki malformed supporting facts: {detail}")
            skipped.append(detail)
            continue
        if not _2wiki_evidence_triples(row.get("evidences", [])):
            missing_evidence += 1
        docs.append(adapt_2wiki_record(row, document_id=f"{path.stem}:{i}"))

    audit = {
        "source_records": len(records),
        "loaded_records": len(docs),
        "skipped_records": len(skipped),
        "records_without_evidence_triples": missing_evidence,
        "invalid_support_policy": invalid_support_policy,
        "source_format": source_format,
        "skipped": skipped,
    }
    return docs, audit


def load_2wiki(path: Path) -> list[AdaptedDocument]:
    docs, _ = load_2wiki_with_audit(path, invalid_support_policy="skip_record")
    return docs

# --- MuSiQue ----------------------------------------------------------------

def adapt_musique_record(record: dict[str, Any], *, document_id: str | None = None) -> AdaptedDocument:
    """Adapt one MuSiQue-Answerable record into a paragraph-support footprint.

    Only paragraph text/title and ``is_supporting`` are used to construct the benchmark
    footprint. Question decomposition and answers are retained only as metadata/audit and
    must not be inspected by tile constructors.
    """
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("MuSiQue record has no paragraphs")
    pieces: list[str] = []
    sentence_ranges: list[Segment] = []
    sentence_titles: list[str] = []
    paragraph_titles: list[str] = []
    supporting_ranges: list[Segment] = []
    supporting_indices: list[int] = []
    cursor = 0
    for pos, raw in enumerate(paragraphs):
        if not isinstance(raw, dict):
            raise ValueError(f"MuSiQue paragraph {pos} is not an object")
        idx = int(raw.get("idx", pos))
        title = str(raw.get("title", ""))
        body = str(raw.get("paragraph_text", ""))
        paragraph_titles.append(title)
        header = f"[{title}]\n"
        pieces.append(header); cursor += len(header)
        body_piece = body.rstrip() + "\n"
        body_start = cursor
        pieces.append(body_piece); cursor += len(body_piece)
        body_end = cursor
        if bool(raw.get("is_supporting", False)):
            supporting_ranges.append(Segment(body_start, body_end))
            supporting_indices.append(idx)
        # Sentence atoms are local to the paragraph body so titles create true gaps.
        local = sentence_chunks(body_piece).segments
        for span in local:
            if span.start == span.end:
                continue
            sentence_ranges.append(Segment(body_start + span.start, body_start + span.end))
            sentence_titles.append(title)
    if not supporting_ranges:
        raise ValueError("MuSiQue answerable record has no supporting paragraphs")
    text = "".join(pieces)
    doc_id = document_id or str(record.get("id") or "musique_unknown")
    tile = GoldSemanticTile(
        tile_id="musique_support_00000",
        label=str(record.get("question", "supporting_paragraphs")),
        ranges=tuple(supporting_ranges),
        source="supporting_paragraphs",
    )
    metadata = {
        "dataset": "MuSiQue-Answerable",
        "question": str(record.get("question", "")),
        "answerable": bool(record.get("answerable", True)),
        "paragraph_titles": tuple(paragraph_titles),
        "sentence_ranges": tuple((r.start, r.end) for r in sentence_ranges),
        "sentence_titles": tuple(sentence_titles),
        "supporting_paragraph_indices": tuple(supporting_indices),
        "supporting_paragraph_count": len(supporting_ranges),
        # Evaluation/audit metadata only. Constructors must not inspect these.
        "question_decomposition": record.get("question_decomposition", ()),
    }
    return AdaptedDocument(doc_id, text, (tile,), metadata)


def load_musique_with_audit(path: Path) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    records = _load_json_or_jsonl_records(path)
    docs: list[AdaptedDocument] = []
    skipped: list[dict[str, Any]] = []
    for i, row in enumerate(records):
        try:
            if not bool(row.get("answerable", True)):
                skipped.append({"row_index": i, "record_id": str(row.get("id", i)), "reason": "unanswerable"})
                continue
            docs.append(adapt_musique_record(row, document_id=f"{path.stem}:{i}"))
        except Exception as exc:
            skipped.append({"row_index": i, "record_id": str(row.get("id", i)), "reason": type(exc).__name__, "detail": str(exc)})
    return docs, {
        "source_records": len(records), "loaded_records": len(docs), "skipped_records": len(skipped),
        "skipped": skipped,
    }


def load_musique(path: Path) -> list[AdaptedDocument]:
    docs, _ = load_musique_with_audit(path)
    return docs

# --- QASPER -----------------------------------------------------------------

def _norm_ws(text: str) -> str:
    return " ".join(str(text).split())


def _qasper_answer_payload(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    answer = row.get("answer", row)
    return answer if isinstance(answer, dict) else {}


def adapt_qasper_question(
    paper_id: str,
    paper: dict[str, Any],
    qa: dict[str, Any],
    *,
    question_index: int,
    exclude_float_evidence: bool = True,
) -> tuple[AdaptedDocument | None, dict[str, Any]]:
    """Adapt one QASPER question while preserving annotator evidence alternatives.

    QASPER textual ``evidence`` entries are paragraph-level references.  Multiple
    answer annotations are *alternatives* and must not be unioned into one gold
    footprint.  This adapter therefore stores one GoldSemanticTile per clean
    annotation and marks them as alternative references in metadata.

    Figure/table evidence begins with ``FLOAT SELECTED``.  Phase 3K is a textual
    paragraph-evidence gate, so float-only/mixed alternatives are audited and
    excluded rather than heuristically mapped to captions.
    """
    if not isinstance(paper, dict) or not isinstance(qa, dict):
        raise ValueError("QASPER paper/qa must be objects")

    pieces: list[str] = []
    paragraph_ranges: list[Segment] = []
    paragraph_titles: list[str] = []
    norm_to_indices: dict[str, list[int]] = {}
    cursor = 0

    abstract = str(paper.get("abstract", "") or "").strip()
    if abstract:
        header = "[Abstract]\n"
        pieces.append(header); cursor += len(header)
        body = abstract.rstrip() + "\n"
        start = cursor; pieces.append(body); cursor += len(body)
        paragraph_ranges.append(Segment(start, cursor))
        paragraph_titles.append("Abstract")
        norm_to_indices.setdefault(_norm_ws(abstract), []).append(len(paragraph_ranges) - 1)

    full_text = paper.get("full_text", []) or []
    if not isinstance(full_text, list):
        raise ValueError("QASPER full_text must be a list")
    for section_pos, section in enumerate(full_text):
        if not isinstance(section, dict):
            raise ValueError(f"QASPER section {section_pos} is not an object")
        title = str(section.get("section_name", "") or f"section_{section_pos}")
        paragraphs = section.get("paragraphs", []) or []
        if not isinstance(paragraphs, list):
            raise ValueError(f"QASPER section {section_pos} paragraphs must be a list")
        for para in paragraphs:
            para = str(para or "").strip()
            if not para:
                continue
            header = f"[{title}]\n"
            pieces.append(header); cursor += len(header)
            body = para.rstrip() + "\n"
            start = cursor; pieces.append(body); cursor += len(body)
            paragraph_ranges.append(Segment(start, cursor))
            paragraph_titles.append(title)
            norm_to_indices.setdefault(_norm_ws(para), []).append(len(paragraph_ranges) - 1)

    if not paragraph_ranges:
        raise ValueError("QASPER paper has no textual paragraphs")

    alternatives: list[GoldSemanticTile] = []
    skipped_annotations: list[dict[str, Any]] = []
    answers = qa.get("answers", []) or []
    if not isinstance(answers, list):
        answers = []
    for ann_idx, ann in enumerate(answers):
        payload = _qasper_answer_payload(ann)
        evidence = payload.get("evidence", []) or []
        if not isinstance(evidence, list):
            evidence = []
        textual = []
        has_float = False
        for ev in evidence:
            ev = str(ev or "")
            if ev.lstrip().startswith("FLOAT SELECTED"):
                has_float = True
            elif ev.strip():
                textual.append(ev)
        if exclude_float_evidence and has_float:
            skipped_annotations.append({"annotation_index": ann_idx, "reason": "contains_float_evidence"})
            continue
        if not textual:
            skipped_annotations.append({"annotation_index": ann_idx, "reason": "no_textual_evidence"})
            continue

        matched_indices: list[int] = []
        bad = None
        for ev in textual:
            matches = norm_to_indices.get(_norm_ws(ev), [])
            if len(matches) == 1:
                matched_indices.append(matches[0])
            elif not matches:
                bad = {"annotation_index": ann_idx, "reason": "textual_evidence_not_found", "evidence_preview": ev[:120]}
                break
            else:
                bad = {"annotation_index": ann_idx, "reason": "ambiguous_duplicate_paragraph", "matches": len(matches), "evidence_preview": ev[:120]}
                break
        if bad is not None:
            skipped_annotations.append(bad)
            continue
        unique_indices = sorted(set(matched_indices))
        ranges = tuple(paragraph_ranges[i] for i in unique_indices)
        alternatives.append(GoldSemanticTile(
            tile_id=f"qasper_alt_{ann_idx:03d}",
            label=str(qa.get("question", "qasper_evidence")),
            ranges=ranges,
            source="qasper_answer_evidence_alternative",
        ))

    audit = {
        "question_id": str(qa.get("question_id", f"{paper_id}:{question_index}")),
        "answer_annotations": len(answers),
        "usable_textual_alternatives": len(alternatives),
        "skipped_annotations": skipped_annotations,
    }
    if not alternatives:
        return None, audit

    text = "".join(pieces)
    qid = audit["question_id"]
    doc = AdaptedDocument(
        document_id=f"qasper:{paper_id}:{qid}",
        text=text,
        tiles=tuple(alternatives),
        metadata={
            "dataset": "QASPER",
            "paper_id": paper_id,
            "question_id": qid,
            "question": str(qa.get("question", "")),
            "paragraph_ranges": tuple((r.start, r.end) for r in paragraph_ranges),
            "paragraph_titles": tuple(paragraph_titles),
            "gold_reference_semantics": "alternative_answer_annotations",
            "usable_alternatives": len(alternatives),
        },
    )
    return doc, audit


def load_qasper_with_audit(path: Path, *, exclude_float_evidence: bool = True) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("QASPER raw v0.3 file must be an object keyed by paper id")
    docs: list[AdaptedDocument] = []
    question_audits: list[dict[str, Any]] = []
    total_questions = 0
    for paper_id, paper in raw.items():
        if not isinstance(paper, dict):
            continue
        qas = paper.get("qas", []) or []
        if not isinstance(qas, list):
            continue
        for q_idx, qa in enumerate(qas):
            if not isinstance(qa, dict):
                continue
            total_questions += 1
            doc, audit = adapt_qasper_question(
                str(paper_id), paper, qa, question_index=q_idx,
                exclude_float_evidence=exclude_float_evidence,
            )
            question_audits.append(audit)
            if doc is not None:
                docs.append(doc)
    skipped = [x for x in question_audits if not x["usable_textual_alternatives"]]
    return docs, {
        "papers": len(raw),
        "questions": total_questions,
        "loaded_textual_questions": len(docs),
        "skipped_questions": len(skipped),
        "exclude_float_evidence": exclude_float_evidence,
        "questions_with_multiple_usable_references": sum(int(x["usable_textual_alternatives"] > 1) for x in question_audits),
        "annotation_skip_reason_counts": {
            reason: sum(1 for x in question_audits for a in x["skipped_annotations"] if a.get("reason") == reason)
            for reason in sorted({a.get("reason") for x in question_audits for a in x["skipped_annotations"] if a.get("reason")})
        },
        "question_audits": question_audits,
    }


def load_qasper(path: Path) -> list[AdaptedDocument]:
    docs, _ = load_qasper_with_audit(path)
    return docs


def load_contractnli_with_audit(path: Path) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    """Load ContractNLI evidence-bearing hypothesis instances.

    The official ContractNLI release stores a full contract, an ordered list of
    sentence/list-item spans, and one annotation set whose evidence refers to
    indices in that span list.  NotMentioned hypotheses have no evidence and are
    excluded from semantic-footprint evaluation.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ValueError("ContractNLI JSON must contain a documents list")
    labels = raw.get("labels", {}) if isinstance(raw.get("labels", {}), dict) else {}
    docs: list[AdaptedDocument] = []
    skipped: dict[str, int] = {}
    choices: dict[str, int] = {}
    source_documents = 0
    for d_idx, row in enumerate(raw["documents"]):
        if not isinstance(row, dict):
            continue
        source_documents += 1
        text = str(row.get("text", ""))
        raw_spans = row.get("spans", []) or []
        spans: list[Segment] = []
        valid = True
        for s_idx, pair in enumerate(raw_spans):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_is_intlike(x) for x in pair)):
                valid = False
                break
            a, b = int(pair[0]), int(pair[1])
            if a < 0 or b <= a or b > len(text):
                valid = False
                break
            spans.append(Segment(a, b))
        if not valid or not spans:
            skipped["invalid_span_inventory"] = skipped.get("invalid_span_inventory", 0) + 1
            continue
        sets = row.get("annotation_sets", []) or []
        if not isinstance(sets, list) or not sets or not isinstance(sets[0], dict):
            skipped["missing_annotation_set"] = skipped.get("missing_annotation_set", 0) + 1
            continue
        annotations = sets[0].get("annotations", {}) or {}
        if not isinstance(annotations, dict):
            skipped["invalid_annotations"] = skipped.get("invalid_annotations", 0) + 1
            continue
        contract_id = str(row.get("id", f"contract_{d_idx}"))
        for hyp_key, ann in annotations.items():
            if not isinstance(ann, dict):
                continue
            choice = str(ann.get("choice", ""))
            choices[choice] = choices.get(choice, 0) + 1
            evidence_indices = ann.get("spans", []) or []
            if choice == "NotMentioned":
                continue
            if choice not in {"Entailment", "Contradiction"}:
                skipped["unknown_choice"] = skipped.get("unknown_choice", 0) + 1
                continue
            if not isinstance(evidence_indices, list) or not evidence_indices:
                skipped["evidence_bearing_choice_without_spans"] = skipped.get("evidence_bearing_choice_without_spans", 0) + 1
                continue
            evidence: list[Segment] = []
            bad = False
            for value in evidence_indices:
                if not _is_intlike(value):
                    bad = True
                    break
                idx = int(value)
                if idx < 0 or idx >= len(spans):
                    bad = True
                    break
                evidence.append(spans[idx])
            if bad or not evidence:
                skipped["invalid_evidence_index"] = skipped.get("invalid_evidence_index", 0) + 1
                continue
            # Deduplicate while preserving the official span order.
            uniq = tuple(dict.fromkeys((r.start, r.end) for r in evidence))
            ranges = tuple(Segment(a, b) for a, b in uniq)
            hypothesis = str((labels.get(str(hyp_key), {}) or {}).get("hypothesis", hyp_key))
            gold = GoldSemanticTile(
                tile_id=f"contractnli_{contract_id}_{hyp_key}",
                label=hypothesis,
                ranges=ranges,
                source="ContractNLI.annotation_sets[0].annotations",
            )
            docs.append(AdaptedDocument(
                document_id=f"contractnli:{contract_id}:{hyp_key}",
                text=text,
                tiles=(gold,),
                metadata={
                    "dataset": "ContractNLI",
                    "contract_id": contract_id,
                    "hypothesis_key": str(hyp_key),
                    "hypothesis": hypothesis,
                    "choice": choice,
                    "official_span_ranges": tuple((r.start, r.end) for r in spans),
                },
            ))
    return docs, {
        "source_documents": source_documents,
        "loaded_evidence_instances": len(docs),
        "choice_counts": choices,
        "skipped_counts": skipped,
        "annotation_set_policy": "first annotation set; official release documents one annotation set per contract",
        "notmentioned_policy": "exclude from evidence-footprint evaluation",
    }


def load_contractnli(path: Path) -> list[AdaptedDocument]:
    docs, _ = load_contractnli_with_audit(path)
    return docs

# ---------------------------------------------------------------------------
# Phase 5A fresh confirmation: PeerQA paragraph evidence retrieval.
# Protocol note: this adapter was implemented from the public dataset schema and
# synthetic fixtures before any PeerQA benchmark bytes were opened in this project.
# ---------------------------------------------------------------------------

def adapt_peerqa_rows(
    qa_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    qrel_rows: list[dict[str, Any]],
) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    """Adapt the algorithmic intersection of PeerQA QA/papers/paragraph-qrels.

    Eligibility is deliberately mechanical: a question must have a paper in the
    supplied paper table and at least one paragraph qrel.  No answer text,
    answerability field, score, or model outcome is consulted for filtering.

    Paper rows are grouped by ``pidx`` in ascending first-``idx`` order.  Content
    rows belonging to the same paragraph are concatenated in ``idx`` order with
    spaces.  Each question becomes one AdaptedDocument over its full paper; gold
    is a single non-contiguous tile spanning all qrel paragraphs.
    """
    from collections import defaultdict

    qa_by_qid: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            raise ValueError("PeerQA QA row missing question_id")
        if qid in qa_by_qid:
            raise ValueError(f"duplicate PeerQA question_id: {qid}")
        qa_by_qid[qid] = row

    rows_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paper_rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError("PeerQA paper row missing paper_id")
        if not _is_intlike(row.get("idx")) or not _is_intlike(row.get("pidx")):
            raise ValueError(f"PeerQA paper row has non-integer idx/pidx for {paper_id}")
        rows_by_paper[paper_id].append(row)

    qrels_by_qid: dict[str, set[int]] = defaultdict(set)
    for row in qrel_rows:
        qid = str(row.get("question_id") or "")
        idx = row.get("idx")
        relevant = row.get("relevant", row.get("relevance", 1))
        if not qid or not _is_intlike(idx):
            raise ValueError("PeerQA qrel row missing question_id/integer paragraph idx")
        if int(relevant) <= 0:
            continue
        qrels_by_qid[qid].add(int(idx))

    # Canonicalize each paper exactly once; repeated QA over the same paper then
    # share byte-identical text and paragraph ranges.
    paper_cache: dict[str, tuple[str, tuple[tuple[int, int], ...], tuple[int, ...], tuple[str, ...]]] = {}
    for paper_id, rows in rows_by_paper.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        first_idx: dict[int, int] = {}
        for row in rows:
            pidx = int(row["pidx"])
            grouped[pidx].append(row)
            first_idx[pidx] = min(first_idx.get(pidx, int(row["idx"])), int(row["idx"]))
        ordered_pidx = sorted(grouped, key=lambda p: (first_idx[p], p))
        pieces: list[str] = []
        ranges: list[tuple[int, int]] = []
        titles: list[str] = []
        cursor = 0
        for pidx in ordered_pidx:
            part_rows = sorted(grouped[pidx], key=lambda r: (int(r["idx"]), int(r["sidx"]) if _is_intlike(r.get("sidx")) else -1))
            content = " ".join(str(r.get("content") or "").strip() for r in part_rows if str(r.get("content") or "").strip()).strip()
            if not content:
                # Keep the index map deterministic while omitting zero-length atoms.
                continue
            heading = next((str(r.get("last_heading") or "").strip() for r in part_rows if str(r.get("last_heading") or "").strip()), "")
            piece = content + "\n"
            pieces.append(piece)
            ranges.append((cursor, cursor + len(piece)))
            titles.append(heading)
            cursor += len(piece)
        if not ranges:
            continue
        text = "".join(pieces)
        # ordered_pidx may include empty paragraphs; map only emitted atoms.
        emitted_pidx = []
        for pidx in ordered_pidx:
            part_rows = sorted(grouped[pidx], key=lambda r: (int(r["idx"]), int(r["sidx"]) if _is_intlike(r.get("sidx")) else -1))
            if any(str(r.get("content") or "").strip() for r in part_rows):
                emitted_pidx.append(pidx)
        paper_cache[paper_id] = (text, tuple(ranges), tuple(emitted_pidx), tuple(titles))

    documents: list[AdaptedDocument] = []
    missing_question = missing_paper = missing_qrel_atom = 0
    for qid in sorted(qrels_by_qid):
        qa = qa_by_qid.get(qid)
        if qa is None:
            missing_question += 1
            continue
        paper_id = str(qa.get("paper_id") or "")
        cached = paper_cache.get(paper_id)
        if cached is None:
            missing_paper += 1
            continue
        text, paragraph_ranges, pidxs, paragraph_titles = cached
        pidx_to_range = {pidx: Segment(a, b) for pidx, (a, b) in zip(pidxs, paragraph_ranges)}
        gold_ranges: list[Segment] = []
        missing = []
        for pidx in sorted(qrels_by_qid[qid]):
            seg = pidx_to_range.get(pidx)
            if seg is None:
                missing.append(pidx)
            else:
                gold_ranges.append(seg)
        if missing:
            missing_qrel_atom += 1
            raise ValueError(f"PeerQA qrel paragraph(s) absent from canonical paper {paper_id} question {qid}: {missing}")
        if not gold_ranges:
            continue
        question = str(qa.get("question") or "").strip()
        if not question:
            raise ValueError(f"PeerQA empty question: {qid}")
        gold = GoldSemanticTile(
            tile_id=f"peerqa_gold_{qid}",
            label=question,
            ranges=tuple(gold_ranges),
            source="PeerQA:qrels-paragraphs:permissive",
        )
        documents.append(AdaptedDocument(
            document_id=f"peerqa:{qid}",
            text=text,
            tiles=(gold,),
            metadata={
                "dataset": "PeerQA",
                "paper_id": paper_id,
                "question_id": qid,
                "question": question,
                "paragraph_ranges": paragraph_ranges,
                "paragraph_titles": paragraph_titles,
                "peerqa_pidxs": pidxs,
                "gold_paragraph_pidxs": tuple(sorted(qrels_by_qid[qid])),
                "fresh_peerqa_primary": True,
            },
        ))

    audit = {
        "dataset": "PeerQA",
        "eligibility_rule": "question_id intersection of QA, permissive papers, and positive paragraph qrels; no manual filtering",
        "qa_rows": len(qa_rows),
        "paper_rows": len(paper_rows),
        "qrel_rows": len(qrel_rows),
        "qa_questions": len(qa_by_qid),
        "papers": len(rows_by_paper),
        "qrel_questions": len(qrels_by_qid),
        "eligible_questions": len(documents),
        "unique_eligible_papers": len({d.metadata["paper_id"] for d in documents}),
        "missing_question_for_qrel": missing_question,
        "missing_paper_for_qrel_question": missing_paper,
        "missing_qrel_atom_cases": missing_qrel_atom,
    }
    return documents, audit


def _peerqa_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - fresh-run optional dependency
        raise RuntimeError("Phase 5A PeerQA requires pyarrow in the base environment") from exc
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def load_peerqa_parquet_with_audit(
    qa_path: Path,
    papers_path: Path,
    qrels_path: Path,
) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    return adapt_peerqa_rows(
        _peerqa_parquet_rows(qa_path),
        _peerqa_parquet_rows(papers_path),
        _peerqa_parquet_rows(qrels_path),
    )

# ---------------------------------------------------------------------------
# Phase 5B: Evidence Inference 2.0 fresh biomedical evidence retrieval adapter
# ---------------------------------------------------------------------------

def _evidence_inference_norm_key(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _evidence_inference_field(row: dict[str, Any], *aliases: str, default: Any = None) -> Any:
    by_norm = {_evidence_inference_norm_key(k): v for k, v in row.items()}
    for alias in aliases:
        key = _evidence_inference_norm_key(alias)
        if key in by_norm:
            return by_norm[key]
    return default


def evidence_inference_query(intervention: str, comparator: str, outcome: str) -> str:
    """Public task wording frozen before Evidence Inference bytes are opened."""
    return (
        f"With respect to {outcome}, characterize the reported difference between "
        f"{intervention} and those receiving {comparator}."
    )


def evidence_inference_plaintext_paragraph_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return deterministic physical atoms for Evidence Inference plain text.

    Evidence Inference distributes character-offset annotations against a generated
    plain-text article.  The corpus exporter emits structural newlines at paragraph,
    section, title, and table-cell boundaries.  For retrieval we therefore use each
    non-empty physical line of the *unmodified* distributed text as one stable atom.

    This routine is deliberately outcome/gold independent: it looks only at raw text
    bytes and never at evidence offsets, labels, prompts, or model outputs.  Keeping
    offsets against the original string is essential because gold evidence spans are
    defined in that coordinate system.
    """
    if not isinstance(text, str):
        raise TypeError("Evidence Inference article text must be str")
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        if line.strip():
            ranges.append((cursor, end))
        cursor = end
    if cursor != len(text):
        raise AssertionError("Evidence Inference line segmentation lost text bytes")
    if not ranges and text.strip():
        # Defensive fallback for unusual strings not split by splitlines().
        ranges.append((0, len(text)))
    return tuple(ranges)


def adapt_evidence_inference_v2_rows(
    prompts: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    test_article_ids: Iterable[int | str],
    texts: dict[str, str],
) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    """Adapt Evidence Inference 2.0 test prompts into retrieval cases.

    This adapter is intentionally mechanical and outcome-blind:
      * use prompts whose PMCID belongs to the published test-article split;
      * require an available exact plain-text article;
      * require at least one annotation for the same PromptID/PMCID with an
        in-bounds non-empty evidence character span;
      * retain each distinct valid annotator span as a gold alternative;
      * never use the clinical outcome label to construct the query or rank tiles.

    The task query follows the public Intervention/Comparator/Outcome wording.
    Character offsets are interpreted against the unmodified txt_files article.
    """
    test_ids = {str(int(x)) if str(x).strip().isdigit() else str(x).strip().removeprefix("PMC") for x in test_article_ids}

    prompt_rows: list[tuple[int, str, dict[str, Any]]] = []
    malformed_prompt_rows = 0
    for row in prompts:
        try:
            pid = int(str(_evidence_inference_field(row, "PromptID", "prompt_id")).strip())
            pmcid_raw = str(_evidence_inference_field(row, "PMCID", "pmcid")).strip().removeprefix("PMC")
            pmcid = str(int(pmcid_raw))
        except Exception:
            malformed_prompt_rows += 1
            continue
        if pmcid in test_ids:
            prompt_rows.append((pid, pmcid, row))

    ann_by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    malformed_annotation_rows = 0
    for row in annotations:
        try:
            pid = int(str(_evidence_inference_field(row, "PromptID", "prompt_id")).strip())
            pmcid_raw = str(_evidence_inference_field(row, "PMCID", "pmcid")).strip().removeprefix("PMC")
            pmcid = str(int(pmcid_raw))
        except Exception:
            malformed_annotation_rows += 1
            continue
        ann_by_key.setdefault((pid, pmcid), []).append(row)

    documents: list[AdaptedDocument] = []
    missing_text = 0
    no_valid_span = 0
    invalid_span_rows = 0
    distinct_gold_spans = 0
    seen_prompt_ids: set[int] = set()
    duplicate_prompt_rows = 0

    for pid, pmcid, row in sorted(prompt_rows, key=lambda x: (x[1], x[0])):
        if pid in seen_prompt_ids:
            duplicate_prompt_rows += 1
            continue
        seen_prompt_ids.add(pid)
        text = texts.get(pmcid)
        if text is None:
            text = texts.get(f"PMC{pmcid}")
        if text is None:
            missing_text += 1
            continue

        spans: set[tuple[int, int]] = set()
        for ann in ann_by_key.get((pid, pmcid), []):
            start_raw = _evidence_inference_field(ann, "Evidence Start", "Start Evidence", "evidence_start", default=-1)
            end_raw = _evidence_inference_field(ann, "Evidence End", "End Evidence", "evidence_end", default=-1)
            try:
                start = int(float(str(start_raw).strip()))
                end = int(float(str(end_raw).strip()))
            except Exception:
                invalid_span_rows += 1
                continue
            if start < 0 or end <= start or end > len(text):
                invalid_span_rows += 1
                continue
            spans.add((start, end))
        if not spans:
            no_valid_span += 1
            continue

        intervention = str(_evidence_inference_field(row, "Intervention", "intervention", default="")).strip()
        comparator = str(_evidence_inference_field(row, "Comparator", "comparator", default="")).strip()
        outcome = str(_evidence_inference_field(row, "Outcome", "outcome", default="")).strip()
        if not (intervention and comparator and outcome):
            malformed_prompt_rows += 1
            continue
        query = evidence_inference_query(intervention, comparator, outcome)
        paragraph_ranges = evidence_inference_plaintext_paragraph_ranges(text)
        if not paragraph_ranges:
            # A non-empty evidence span cannot be retrieved without any physical atom.
            raise ValueError(f"Evidence Inference article PMC{pmcid} has no non-empty physical text atoms")
        gold_tiles = tuple(
            GoldSemanticTile(
                tile_id=f"ei2_{pid}_gold_{idx:02d}",
                label=query,
                ranges=(Segment(start, end),),
                source="annotations_merged.csv:evidence_span",
            )
            for idx, (start, end) in enumerate(sorted(spans))
        )
        distinct_gold_spans += len(gold_tiles)
        documents.append(AdaptedDocument(
            document_id=f"evidence_inference:{pid}",
            text=text,
            tiles=gold_tiles,
            metadata={
                "dataset": "EvidenceInference2",
                "paper_id": f"PMC{pmcid}",
                "pmcid": pmcid,
                "prompt_id": pid,
                "question": query,
                "intervention": intervention,
                "comparator": comparator,
                "outcome": outcome,
                "gold_span_count": len(gold_tiles),
                "paragraph_ranges": paragraph_ranges,
                "paragraph_titles": tuple("" for _ in paragraph_ranges),
                "paragraph_segmentation": "evidence_inference_plaintext_nonempty_line_v1",
                "phase5b_original_preopen_role": "secondary_fresh_domain_shift_protocol",
            },
        ))

    audit = {
        "dataset": "EvidenceInference2",
        "split": "test",
        "eligibility_rule": (
            "published test PMCID + prompt row + exact article text + >=1 same-PromptID/same-PMCID "
            "non-empty in-bounds evidence span; every eligible prompt; no manual or outcome-driven exclusions"
        ),
        "test_article_ids": len(test_ids),
        "prompt_rows_total": len(prompts),
        "prompt_rows_in_test_articles": len(prompt_rows),
        "annotation_rows_total": len(annotations),
        "eligible_prompts": len(documents),
        "unique_eligible_articles": len({d.metadata["pmcid"] for d in documents}),
        "distinct_gold_span_alternatives": distinct_gold_spans,
        "missing_text": missing_text,
        "no_valid_span": no_valid_span,
        "invalid_span_rows": invalid_span_rows,
        "malformed_prompt_rows": malformed_prompt_rows,
        "malformed_annotation_rows": malformed_annotation_rows,
        "duplicate_prompt_rows": duplicate_prompt_rows,
        "label_used_for_retrieval": False,
    }
    return documents, audit


def _read_evidence_inference_csv(path: Path) -> list[dict[str, Any]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def locate_evidence_inference_v2_files(data_dir: Path) -> dict[str, Path]:
    """Locate required v2.0 files without assuming whether the archive has a root dir."""
    def unique_named(name: str) -> Path:
        hits = sorted(p for p in data_dir.rglob(name) if p.is_file())
        if len(hits) != 1:
            raise RuntimeError(f"Evidence Inference expected exactly one {name}, found {len(hits)}")
        return hits[0]

    annotations = unique_named("annotations_merged.csv")
    prompts = unique_named("prompts_merged.csv")
    split = unique_named("test_article_ids.txt")
    txt_dirs = sorted(p for p in data_dir.rglob("txt_files") if p.is_dir())
    if len(txt_dirs) != 1:
        raise RuntimeError(f"Evidence Inference expected exactly one txt_files directory, found {len(txt_dirs)}")
    return {"annotations": annotations, "prompts": prompts, "test_ids": split, "txt_dir": txt_dirs[0]}


def load_evidence_inference_v2_with_audit(data_dir: Path) -> tuple[list[AdaptedDocument], dict[str, Any]]:
    files = locate_evidence_inference_v2_files(data_dir)
    prompts = _read_evidence_inference_csv(files["prompts"])
    annotations = _read_evidence_inference_csv(files["annotations"])
    test_ids = [line.strip() for line in files["test_ids"].read_text(encoding="utf-8").splitlines() if line.strip()]
    texts: dict[str, str] = {}
    for p in sorted(files["txt_dir"].glob("PMC*.txt")):
        pmcid = p.stem.removeprefix("PMC")
        texts[pmcid] = p.read_text(encoding="utf-8")
    docs, audit = adapt_evidence_inference_v2_rows(prompts, annotations, test_ids, texts)
    audit["located_files"] = {
        "annotations": str(files["annotations"]),
        "prompts": str(files["prompts"]),
        "test_ids": str(files["test_ids"]),
        "txt_dir": str(files["txt_dir"]),
        "available_txt_articles": len(texts),
    }
    return docs, audit
