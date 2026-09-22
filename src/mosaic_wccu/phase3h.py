from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3e import _evaluate_predictions, overlapping_ego_tiles_from_prepared
from .phase3f import (
    _dedupe_semantic_tiles,
    _normalized_phrase,
    evaluate_hotpot_representation_baseline,
    prepare_hotpot_title_bridge_graph,
)
from .segmentation import Segment
from .semantic_tiles import PreparedLexicalGraph, SemanticTile, prepare_lexical_graph


def _relabel(metrics: dict[str, Any], method: str, **config_updates: Any) -> dict[str, Any]:
    out = dict(metrics)
    out["method"] = method
    cfg = dict(out.get("config", {}))
    cfg.update(config_updates)
    out["config"] = cfg
    return out


def evaluate_2wiki_representation_baseline(
    documents: Sequence[AdaptedDocument], *, mode: str
) -> dict[str, Any]:
    metrics = evaluate_hotpot_representation_baseline(documents, mode=mode)
    return _relabel(metrics, f"2wiki_{mode}_baseline", transferred_from="phase3g_hotpot")


def _sentence_atoms(doc: AdaptedDocument) -> tuple[Segment, ...]:
    atoms = tuple(Segment(int(a), int(b)) for a, b in doc.metadata.get("sentence_ranges", ()))
    if not atoms:
        raise ValueError(f"{doc.document_id}: no sentence ranges")
    return atoms


def _frozen_views(doc: AdaptedDocument) -> tuple[PreparedLexicalGraph, PreparedLexicalGraph, PreparedLexicalGraph]:
    atoms = _sentence_atoms(doc)
    lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
    title_only = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
    fused = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
    return lexical, title_only, fused


def evaluate_2wiki_frozen_lexical(
    documents: Sequence[AdaptedDocument], *, similarity_threshold: float = 0.16, top_k: int = 1
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        prepared = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        semantic = overlapping_ego_tiles_from_prepared(
            prepared, similarity_threshold=similarity_threshold, top_k=top_k
        )
        predictions.append((atoms, semantic))
    return _evaluate_predictions(
        documents, predictions, method="2wiki_frozen_phase3e_lexical",
        config={"similarity_threshold": similarity_threshold, "top_k": top_k,
                "frozen_before_2wiki": True}, include_bootstrap=False,
    )


def evaluate_2wiki_title_bridge(
    documents: Sequence[AdaptedDocument], *, similarity_threshold: float = 0.16, top_k: int = 1
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        prepared = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
        semantic = overlapping_ego_tiles_from_prepared(
            prepared, similarity_threshold=similarity_threshold, top_k=top_k
        )
        predictions.append((atoms, semantic))
    return _evaluate_predictions(
        documents, predictions, method="2wiki_frozen_phase3f_title_bridge",
        config={"similarity_threshold": similarity_threshold, "top_k": top_k,
                "frozen_before_2wiki": True}, include_bootstrap=False,
    )


def build_2wiki_multiview_predictions(
    documents: Sequence[AdaptedDocument], *, similarity_threshold: float = 0.16, top_k: int = 1
) -> list[tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]]:
    """Frozen Phase-3G multi-view inventory transferred to 2Wiki without tuning."""
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        tiles: list[SemanticTile] = []
        for prepared in _frozen_views(doc):
            tiles.extend(overlapping_ego_tiles_from_prepared(
                prepared, similarity_threshold=similarity_threshold, top_k=top_k
            ))
        predictions.append((atoms, _dedupe_semantic_tiles(tiles)))
    return predictions


def evaluate_2wiki_frozen_multiview(
    documents: Sequence[AdaptedDocument], *, similarity_threshold: float = 0.16, top_k: int = 1
) -> dict[str, Any]:
    predictions = build_2wiki_multiview_predictions(
        documents, similarity_threshold=similarity_threshold, top_k=top_k
    )
    return _evaluate_predictions(
        documents, predictions, method="2wiki_frozen_phase3g_multiview",
        config={
            "views": ["lexical", "title_only", "lexical_plus_title_fused"],
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "frozen_before_2wiki": True,
        }, include_bootstrap=False,
    )


def _title_sentence_ranges(doc: AdaptedDocument) -> dict[str, tuple[Segment, ...]]:
    atoms = _sentence_atoms(doc)
    titles = [str(x) for x in doc.metadata.get("sentence_titles", ())]
    if len(atoms) != len(titles):
        raise ValueError(f"{doc.document_id}: sentence title metadata mismatch")
    out: dict[str, list[Segment]] = defaultdict(list)
    for title, span in zip(titles, atoms):
        out[_normalized_phrase(title)].append(span)
    return {key: tuple(value) for key, value in out.items()}


def _endpoint_title_candidates(endpoint: str, title_map: dict[str, tuple[Segment, ...]]) -> tuple[str, ...]:
    target = _normalized_phrase(endpoint)
    if not target:
        return ()
    exact = tuple(k for k in title_map if k == target)
    if exact:
        return exact
    # Conservative alias-free fallback: containment only for reasonably specific names.
    if len(target) < 5:
        return ()
    return tuple(k for k in title_map if target in k or k in target)


def _tile_touches_any(tile: SemanticTile, spans: Sequence[Segment]) -> bool:
    for a in tile.ranges:
        for b in spans:
            if max(a.start, b.start) < min(a.end, b.end):
                return True
    return False


def evaluate_2wiki_evidence_path_connectivity(
    documents: Sequence[AdaptedDocument],
    predictions: Sequence[tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]],
) -> dict[str, Any]:
    """Evaluate whether predicted tiles connect mappable Wikidata evidence endpoints.

    Evidence triples are *evaluation-only*. They never influence the prediction graph.
    A triple is mappable when its subject and object can each be conservatively aligned
    to a context paragraph title. Coverage means one predicted semantic tile touches at
    least one sentence from both mapped endpoint paragraphs.
    """
    total = 0
    mappable = 0
    covered = 0
    by_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    raw: list[dict[str, Any]] = []
    for doc, (_, semantic) in zip(documents, predictions):
        title_map = _title_sentence_ranges(doc)
        qtype = str(doc.metadata.get("question_type", "unknown"))
        for subject, relation, obj in doc.metadata.get("evidence_triples", ()):
            total += 1
            s_keys = _endpoint_title_candidates(str(subject), title_map)
            o_keys = _endpoint_title_candidates(str(obj), title_map)
            if not s_keys or not o_keys:
                raw.append({
                    "document_id": doc.document_id, "subject": subject, "relation": relation,
                    "object": obj, "mappable": False, "covered": False,
                })
                continue
            mappable += 1
            s_spans = tuple(span for key in s_keys for span in title_map[key])
            o_spans = tuple(span for key in o_keys for span in title_map[key])
            ok = any(_tile_touches_any(tile, s_spans) and _tile_touches_any(tile, o_spans) for tile in semantic)
            covered += int(ok)
            by_relation[str(relation)][0] += 1
            by_relation[str(relation)][1] += int(ok)
            by_type[qtype][0] += 1
            by_type[qtype][1] += int(ok)
            raw.append({
                "document_id": doc.document_id, "subject": subject, "relation": relation,
                "object": obj, "mappable": True, "covered": bool(ok),
            })
    return {
        "evidence_triples": total,
        "mappable_triples": mappable,
        "mappable_rate": mappable / max(1, total),
        "covered_mappable_triples": covered,
        "endpoint_pair_coverage": covered / max(1, mappable),
        "by_relation": {
            key: {"mappable": n, "covered": c, "coverage": c / max(1, n)}
            for key, (n, c) in sorted(by_relation.items())
        },
        "by_question_type": {
            key: {"mappable": n, "covered": c, "coverage": c / max(1, n)}
            for key, (n, c) in sorted(by_type.items())
        },
        "raw": raw,
        "oracle_guardrail": "evidences used only after predictions are constructed",
    }
