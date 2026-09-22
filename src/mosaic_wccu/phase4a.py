from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .benchmark_adapters import AdaptedDocument, GoldSemanticTile
from .phase3e import _tile_scores
from .phase3i import LSAHashModel
from .phase3j import paragraph_atoms
from .phase3l import _atomic_tiles, select_overlay_budget
from .phase3m import OverlayCandidate, paragraph_overlay_candidate_records
from .segmentation import Segment
from .semantic_tiles import SemanticTile


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "will",
    "with", "you", "your",
}


class EmbeddingBackend(Protocol):
    """Minimal backend contract used by retrieval evaluation.

    Implementations must return L2-normalized row vectors.  They may be local LSA,
    open embedding checkpoints, or an externally produced embedding service.  The
    evaluator never passes gold evidence to this interface.
    """

    name: str

    def encode_queries(self, texts: Sequence[str]): ...
    def encode_documents(self, texts: Sequence[str]): ...


@dataclass(frozen=True)
class RetrievalConfig:
    ks: tuple[int, ...] = (1, 2, 4, 8)
    relevance_threshold_for_mrr: float = 0.5
    trace_top_k: int = 8


@dataclass(frozen=True)
class RankedRetrievalResult:
    method: str
    n: int
    metrics: Mapping[str, float]
    per_query: tuple[Mapping[str, Any], ...]


def query_text(doc: AdaptedDocument) -> str:
    """Return only the runtime-visible query/hypothesis, never evidence labels/ranges."""
    for key in ("question", "hypothesis", "claim", "query"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    # QMSum topics and a few legacy adapters store the query in GoldSemanticTile.label.
    # The label is runtime-visible task text; ranges remain hidden.
    if doc.tiles and doc.tiles[0].label:
        return str(doc.tiles[0].label)
    raise ValueError(f"{doc.document_id}: no runtime-visible query text")


def tile_text(text: str, tile: SemanticTile, *, separator: str = "\n[...]\n") -> str:
    return separator.join(text[r.start:r.end].strip() for r in tile.ranges if r.end > r.start)


def inventory_tiles(atoms: Sequence[Segment], overlays: Sequence[SemanticTile]) -> tuple[SemanticTile, ...]:
    """Return physical atoms + overlays as one deduplicated retrieval inventory."""
    tiles = list(_atomic_tiles(atoms)) + list(overlays)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[SemanticTile] = []
    for tile in tiles:
        key = tuple(sorted((r.start, r.end) for r in tile.ranges))
        if key in seen:
            continue
        seen.add(key)
        out.append(SemanticTile(
            tile_id=f"retrieval_{len(out):05d}", ranges=tile.ranges,
            kind=tile.kind, score=float(tile.score),
        ))
    return tuple(out)


def _all_pair_tiles(atoms: Sequence[Segment]) -> list[tuple[int, int, SemanticTile]]:
    rows: list[tuple[int, int, SemanticTile]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            rows.append((i, j, SemanticTile(
                tile_id=f"pair_{i}_{j}", ranges=(atoms[i], atoms[j]),
                kind="pair_ablation", score=1.0,
            )))
    return rows


def random_pair_overlays(
    atoms: Sequence[Segment], *, max_overlay_ratio: float, seed_key: str, random_state: int = 20260903,
) -> tuple[SemanticTile, ...]:
    """Deterministic random pair baseline under the same object budget."""
    budget = min(int(math.ceil(max_overlay_ratio * len(atoms))), len(atoms) * max(0, len(atoms) - 1) // 2)
    if budget <= 0:
        return ()
    rows = _all_pair_tiles(atoms)
    seed = int(hashlib.sha256(f"{random_state}:{seed_key}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return tuple(SemanticTile(
        tile_id=f"random_{k:04d}", ranges=row[2].ranges, kind="random_pair_overlay", score=1.0,
    ) for k, row in enumerate(rows[:budget]))


def locality_pair_overlays(atoms: Sequence[Segment], *, max_overlay_ratio: float) -> tuple[SemanticTile, ...]:
    """Pure locality baseline: nearest physical pairs, independent of semantic scores."""
    budget = min(int(math.ceil(max_overlay_ratio * len(atoms))), len(atoms) * max(0, len(atoms) - 1) // 2)
    if budget <= 0:
        return ()
    rows = _all_pair_tiles(atoms)
    rows.sort(key=lambda row: (row[1] - row[0], row[0], row[1]))
    return tuple(SemanticTile(
        tile_id=f"local_{k:04d}", ranges=row[2].ranges, kind="locality_pair_overlay", score=1.0,
    ) for k, row in enumerate(rows[:budget]))


def _filter_candidates(candidates: Sequence[OverlayCandidate], views: set[str]) -> tuple[OverlayCandidate, ...]:
    if not views:
        return tuple(candidates)
    return tuple(c for c in candidates if any(v in views for v in c.views))


def build_inventory_variant(
    doc: AdaptedDocument,
    *,
    lsa: LSAHashModel | None,
    variant: str,
    max_overlay_ratio: float = 1.0,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    top_k: int = 1,
    locality_exponent: float = 2.0,
) -> tuple[SemanticTile, ...]:
    """Build query-independent inventory variants for Phase 4A ablations.

    `locality_pair` and `random_pair` do not use semantic candidate scores at all.
    `semantic_*` variants share the thresholded candidate pool from Phase 3T.
    """
    atoms, candidates = paragraph_overlay_candidate_records(
        doc, model=lsa, lexical_threshold=lexical_threshold,
        dense_threshold=dense_threshold, top_k=top_k,
    )
    if variant == "physical":
        overlays: tuple[SemanticTile, ...] = ()
    elif variant == "random_pair":
        overlays = random_pair_overlays(atoms, max_overlay_ratio=max_overlay_ratio, seed_key=hashlib.sha256(doc.text.encode("utf-8")).hexdigest())
    elif variant == "locality_pair":
        overlays = locality_pair_overlays(atoms, max_overlay_ratio=max_overlay_ratio)
    elif variant == "semantic_score":
        overlays = select_overlay_budget(
            atoms, [c.tile for c in candidates], max_overlay_ratio=max_overlay_ratio,
            priority_mode="score", locality_exponent=locality_exponent,
        )
    elif variant == "semantic_locality":
        overlays = select_overlay_budget(
            atoms, [c.tile for c in candidates], max_overlay_ratio=max_overlay_ratio,
            priority_mode="score_locality", locality_exponent=locality_exponent,
        )
    elif variant == "lexical_only":
        filtered = _filter_candidates(candidates, {"lexical"})
        overlays = select_overlay_budget(
            atoms, [c.tile for c in filtered], max_overlay_ratio=max_overlay_ratio,
            priority_mode="score_locality", locality_exponent=locality_exponent,
        )
    elif variant == "dense_only":
        if lsa is None:
            raise ValueError("dense_only requires an LSA/dense model")
        filtered = _filter_candidates(candidates, {"dense"})
        overlays = select_overlay_budget(
            atoms, [c.tile for c in filtered], max_overlay_ratio=max_overlay_ratio,
            priority_mode="score_locality", locality_exponent=locality_exponent,
        )
    elif variant == "semantic_full":
        overlays = tuple(c.tile for c in candidates)
    else:
        raise ValueError(f"unknown Phase4A inventory variant: {variant}")
    return inventory_tiles(atoms, overlays)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOP]


def lexical_query_scores(query: str, candidate_texts: Sequence[str]) -> list[float]:
    """Query-conditioned, annotation-free IDF-weighted token-overlap scorer."""
    q = set(_tokens(query))
    if not candidate_texts:
        return []
    token_sets = [set(_tokens(t)) for t in candidate_texts]
    df: dict[str, int] = {}
    for toks in token_sets:
        for token in toks:
            df[token] = df.get(token, 0) + 1
    n = len(token_sets)
    scores = []
    for toks in token_sets:
        shared = q & toks
        numerator = sum(math.log((n + 1) / (df.get(tok, 0) + 1)) + 1.0 for tok in shared)
        qnorm = sum(math.log((n + 1) / (df.get(tok, 0) + 1)) + 1.0 for tok in q) or 1.0
        length_penalty = math.sqrt(max(1.0, len(toks) / max(1, len(q))))
        scores.append(float(numerator / qnorm / length_penalty))
    return scores


def embedding_query_scores(
    query: str, candidate_texts: Sequence[str], backend: EmbeddingBackend,
) -> list[float]:
    if not candidate_texts:
        return []
    q = backend.encode_queries([query])
    d = backend.encode_documents(list(candidate_texts))
    # numpy-like arrays expected, kept duck-typed to avoid hard dependency here.
    return [float(x) for x in (d @ q[0])]


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def hybrid_query_scores(
    query: str, candidate_texts: Sequence[str], backend: EmbeddingBackend,
    *, lexical_weight: float = 0.5,
) -> list[float]:
    if not 0 <= lexical_weight <= 1:
        raise ValueError("lexical_weight must be in [0,1]")
    lex = _minmax(lexical_query_scores(query, candidate_texts))
    emb = _minmax(embedding_query_scores(query, candidate_texts, backend))
    return [lexical_weight * a + (1.0 - lexical_weight) * b for a, b in zip(lex, emb)]


def _union_segments(ranges: Iterable[Segment]) -> tuple[Segment, ...]:
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    if not ordered:
        return ()
    out: list[Segment] = []
    start, end = ordered[0].start, ordered[0].end
    for r in ordered[1:]:
        if r.start <= end:
            end = max(end, r.end)
        else:
            out.append(Segment(start, end)); start, end = r.start, r.end
    out.append(Segment(start, end))
    return tuple(out)


def _best_gold_tile_score(gold: Sequence[GoldSemanticTile], pred: SemanticTile) -> tuple[float, float, float]:
    if not gold:
        return (0.0, 0.0, 0.0)
    return max((_tile_scores(g, pred) for g in gold), key=lambda x: (x[2], x[1], x[0]))


def _dcg(relevances: Sequence[float]) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(i + 2.0) for i, rel in enumerate(relevances))


def _query_metrics(
    doc: AdaptedDocument,
    ranked_tiles: Sequence[SemanticTile],
    *,
    config: RetrievalConfig,
) -> dict[str, Any]:
    per_tile_rel = [_best_gold_tile_score(doc.tiles, tile)[2] for tile in ranked_tiles]
    ideal = sorted(per_tile_rel, reverse=True)
    out: dict[str, Any] = {}
    for k in config.ks:
        top = tuple(ranked_tiles[:k])
        if top:
            union = SemanticTile(
                tile_id=f"top{k}_union", ranges=_union_segments(r for t in top for r in t.ranges),
                kind="retrieved_union", score=1.0,
            )
            p, r, f1 = _best_gold_tile_score(doc.tiles, union)
            best_tile = max((_best_gold_tile_score(doc.tiles, tile)[2] for tile in top), default=0.0)
            complete = float(any(all(
                any(pr.start <= gr.start and pr.end >= gr.end for pr in union.ranges)
                for gr in gold.ranges
            ) for gold in doc.tiles))
        else:
            p = r = f1 = best_tile = complete = 0.0
        rel = per_tile_rel[:k]
        ideal_rel = ideal[:k]
        denom = _dcg(ideal_rel)
        out[f"union_precision_at_{k}"] = p
        out[f"union_recall_at_{k}"] = r
        out[f"union_f1_at_{k}"] = f1
        out[f"best_tile_f1_at_{k}"] = best_tile
        out[f"complete_coverage_at_{k}"] = complete
        out[f"ndcg_at_{k}"] = (_dcg(rel) / denom) if denom > 0 else 0.0
    rr = 0.0
    for rank, rel in enumerate(per_tile_rel, 1):
        if rel >= config.relevance_threshold_for_mrr:
            rr = 1.0 / rank
            break
    out["mrr_f1_ge_threshold"] = rr
    out["inventory_size"] = float(len(ranked_tiles))
    return out




def retrieval_cluster_id(doc: AdaptedDocument) -> str:
    """Return the document-level resampling unit for paired statistical audits.

    QASPER has multiple questions per paper, so its paper id is the cluster.  Other
    current Phase-4 adapters have one query per adapted document and fall back to the
    document id.  This is evaluation metadata only; it is never passed to a ranker.
    """
    for key in ("paper_id", "contract_id", "meeting_id", "source_document_id"):
        value = doc.metadata.get(key)
        if value:
            return f"{doc.metadata.get('dataset', 'unknown')}:{key}:{value}"
    return f"{doc.metadata.get('dataset', 'unknown')}:document:{doc.document_id}"


def sample_fingerprint(documents: Sequence[AdaptedDocument]) -> str:
    """Hash the exact ordered evaluation cases so independent runners can align them."""
    h = hashlib.sha256()
    for doc in documents:
        h.update(doc.document_id.encode("utf-8")); h.update(b"\0")
        h.update(hashlib.sha256(doc.text.encode("utf-8")).digest()); h.update(b"\n")
    return h.hexdigest()


def inventory_cost_summary(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
) -> dict[str, Any]:
    """Report both query-weighted and unique-document inventory materialization cost.

    QASPER repeats a paper across multiple questions.  System storage/build cost must not
    count that same inventory once per query, so unique-document totals are primary.
    """
    if len(documents) != len(inventories):
        raise ValueError("documents/inventories length mismatch")
    query_sizes = [len(inv) for inv in inventories]
    unique: dict[str, int] = {}
    for doc, inv in zip(documents, inventories):
        key = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        size = len(inv)
        if key in unique and unique[key] != size:
            raise ValueError(f"same document text produced inconsistent inventory sizes: {key}")
        unique[key] = size
    sizes = list(unique.values())
    return {
        "queries": len(documents),
        "query_weighted_mean_objects": (mean(query_sizes) if query_sizes else 0.0),
        "unique_documents": len(unique),
        "unique_document_total_objects": sum(sizes),
        "unique_document_mean_objects": (mean(sizes) if sizes else 0.0),
        "unique_document_min_objects": (min(sizes) if sizes else 0),
        "unique_document_max_objects": (max(sizes) if sizes else 0),
    }


def _rank_trace(
    doc: AdaptedDocument,
    ranked_tiles: Sequence[SemanticTile],
    ranked_scores: Sequence[float],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Store frozen ranking output plus post-ranking relevance for error analysis."""
    rows: list[dict[str, Any]] = []
    for rank, (tile, score) in enumerate(zip(ranked_tiles[:limit], ranked_scores[:limit]), 1):
        p, r, f1 = _best_gold_tile_score(doc.tiles, tile)
        rows.append({
            "rank": rank,
            "tile_id": tile.tile_id,
            "kind": tile.kind,
            "score": float(score),
            "ranges": [[seg.start, seg.end] for seg in tile.ranges],
            "posthoc_gold_precision": p,
            "posthoc_gold_recall": r,
            "posthoc_gold_f1": f1,
        })
    return rows


def evaluate_query_conditioned_retrieval(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
    *,
    method: str,
    scorer: str,
    embedding_backend: EmbeddingBackend | None = None,
    hybrid_lexical_weight: float = 0.5,
    config: RetrievalConfig = RetrievalConfig(),
) -> RankedRetrievalResult:
    if len(documents) != len(inventories):
        raise ValueError("documents/inventories length mismatch")
    per_query: list[dict[str, Any]] = []
    for doc, inventory in zip(documents, inventories):
        query = query_text(doc)
        texts = [tile_text(doc.text, t) for t in inventory]
        if scorer == "lexical":
            scores = lexical_query_scores(query, texts)
        elif scorer == "embedding":
            if embedding_backend is None:
                raise ValueError("embedding scorer requires embedding_backend")
            scores = embedding_query_scores(query, texts, embedding_backend)
        elif scorer == "hybrid":
            if embedding_backend is None:
                raise ValueError("hybrid scorer requires embedding_backend")
            scores = hybrid_query_scores(query, texts, embedding_backend, lexical_weight=hybrid_lexical_weight)
        else:
            raise ValueError(f"unknown retrieval scorer: {scorer}")
        ranked_rows = sorted(
            ((float(score), -idx, tile) for idx, (score, tile) in enumerate(zip(scores, inventory))),
            key=lambda x: (x[0], x[1]), reverse=True,
        )
        ranked = tuple(row[2] for row in ranked_rows)
        ranked_scores = tuple(row[0] for row in ranked_rows)
        row = _query_metrics(doc, ranked, config=config)
        row.update({
            "document_id": doc.document_id,
            "cluster_id": retrieval_cluster_id(doc),
            "dataset": str(doc.metadata.get("dataset", "unknown")),
            "query": query,
            "top_tile_id": ranked[0].tile_id if ranked else None,
            "top_score": ranked_scores[0] if ranked_scores else None,
            "gold_alternatives": len(doc.tiles),
            "ranked_top": _rank_trace(doc, ranked, ranked_scores, limit=config.trace_top_k),
        })
        per_query.append(row)
    metric_names = sorted({k for row in per_query for k, v in row.items() if isinstance(v, (int, float)) and k != "gold_alternatives"})
    metrics = {name: mean(float(row[name]) for row in per_query if name in row) for name in metric_names}
    return RankedRetrievalResult(method=method, n=len(per_query), metrics=metrics, per_query=tuple(per_query))


class LSABackend:
    def __init__(self, model: LSAHashModel, *, name: str = "lsa_hash") -> None:
        self.model = model
        self.name = name

    def encode_queries(self, texts: Sequence[str]):
        return self.model.encode(texts)

    def encode_documents(self, texts: Sequence[str]):
        return self.model.encode(texts)


def retrieval_training_pairs(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
    *,
    positive_threshold: float = 0.5,
) -> Iterable[dict[str, Any]]:
    """Yield supervised query/tile pairs for *training splits only*.

    The continuous target is best evidence-footprint F1.  `is_positive` is a convenience
    label for reranker training.  Callers are responsible for split discipline.
    """
    for doc, inventory in zip(documents, inventories):
        q = query_text(doc)
        for tile in inventory:
            rel = _best_gold_tile_score(doc.tiles, tile)[2]
            yield {
                "case_id": doc.document_id,
                "cluster_id": retrieval_cluster_id(doc),
                "dataset": str(doc.metadata.get("dataset", "unknown")),
                "query": q,
                "candidate_text": tile_text(doc.text, tile),
                "candidate_ranges": [[r.start, r.end] for r in tile.ranges],
                "target_f1": rel,
                "is_positive": int(rel >= positive_threshold),
            }

@dataclass
class LearnedQueryTileReranker:
    """Lightweight supervised query/tile baseline used before neural fine-tuning.

    It is deliberately simple: logistic regression over annotation-free runtime features.
    Gold is used only to create labels on consumed training splits.  The same evaluator
    and candidate inventories are later reused by CrossEncoder/LLM cells.
    """

    lsa: LSAHashModel
    random_state: int = 20260903
    positive_threshold: float = 0.5
    max_negative_ratio: int = 6

    def __post_init__(self) -> None:
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("LearnedQueryTileReranker requires numpy/scikit-learn") from exc
        self._np = np
        self.classifier = LogisticRegression(
            max_iter=600, class_weight="balanced", random_state=self.random_state,
            solver="liblinear",
        )
        self.fitted = False

    def _features(self, doc: AdaptedDocument, inventory: Sequence[SemanticTile]):
        import numpy as np
        q = query_text(doc)
        texts = [tile_text(doc.text, t) for t in inventory]
        lex = lexical_query_scores(q, texts)
        qz = self.lsa.encode([q])[0]
        dz = self.lsa.encode(texts) if texts else np.zeros((0, len(qz)), dtype="float32")
        dense = [float(x) for x in (dz @ qz)] if texts else []
        atoms, _ = paragraph_atoms(doc)
        atom_index = {(a.start, a.end): i for i, a in enumerate(atoms)}
        denom = max(1, len(atoms) - 1)
        rows = []
        for tile, l, d, txt in zip(inventory, lex, dense, texts):
            members = sorted(atom_index[(r.start, r.end)] for r in tile.ranges if (r.start, r.end) in atom_index)
            gaps = [abs(b - a) / denom for i, a in enumerate(members) for b in members[i + 1:]]
            physical_chars = sum(r.length for r in tile.ranges)
            envelope = tile.envelope_chars
            qlen = max(1, len(_tokens(q))); tlen = max(1, len(_tokens(txt)))
            rows.append([
                float(l), float(d), max(0.0, min(1.0, float(tile.score))),
                float(len(tile.ranges)), max(gaps) if gaps else 0.0,
                sum(gaps) / len(gaps) if gaps else 0.0,
                envelope / max(1, physical_chars),
                min(qlen, tlen) / max(qlen, tlen),
                1.0 if len(tile.ranges) == 1 else 0.0,
            ])
        return np.asarray(rows, dtype="float32")

    def fit(
        self,
        documents: Sequence[AdaptedDocument],
        inventories: Sequence[Sequence[SemanticTile]],
        *, max_queries: int = 2000,
    ) -> "LearnedQueryTileReranker":
        if len(documents) != len(inventories):
            raise ValueError("documents/inventories length mismatch")
        rng = random.Random(self.random_state)
        order = list(range(len(documents)))
        rng.shuffle(order)
        order = order[:max_queries]
        xs = []; ys: list[int] = []
        positive_total = negative_total = 0
        for idx in order:
            doc, inv = documents[idx], inventories[idx]
            if not inv:
                continue
            feats = self._features(doc, inv)
            rel = [_best_gold_tile_score(doc.tiles, tile)[2] for tile in inv]
            pos = [i for i, x in enumerate(rel) if x >= self.positive_threshold]
            neg = [i for i, x in enumerate(rel) if x < self.positive_threshold]
            if not pos:
                continue
            rng.shuffle(neg)
            neg = neg[: max(self.max_negative_ratio, self.max_negative_ratio * len(pos))]
            select = pos + neg
            xs.append(feats[select])
            ys.extend([1] * len(pos) + [0] * len(neg))
            positive_total += len(pos); negative_total += len(neg)
        if not xs or len(set(ys)) < 2:
            raise ValueError("insufficient supervised query/tile labels")
        x = self._np.vstack(xs)
        y = self._np.asarray(ys, dtype="int32")
        self.classifier.fit(x, y)
        self.fitted = True
        self.training_pairs = int(len(y))
        self.training_positives = int(positive_total)
        self.training_negatives = int(negative_total)
        self.training_queries_requested = int(len(order))
        return self

    def scores(self, doc: AdaptedDocument, inventory: Sequence[SemanticTile]) -> list[float]:
        if not self.fitted:
            raise RuntimeError("reranker not fitted")
        if not inventory:
            return []
        return [float(x) for x in self.classifier.predict_proba(self._features(doc, inventory))[:, 1]]


def evaluate_learned_query_reranker(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
    *, model: LearnedQueryTileReranker,
    method: str = "learned_query_tile_reranker",
    config: RetrievalConfig = RetrievalConfig(),
) -> RankedRetrievalResult:
    if len(documents) != len(inventories):
        raise ValueError("documents/inventories length mismatch")
    per_query: list[dict[str, Any]] = []
    for doc, inventory in zip(documents, inventories):
        scores = model.scores(doc, inventory)
        ranked = tuple(tile for _, _, tile in sorted(
            ((float(score), -idx, tile) for idx, (score, tile) in enumerate(zip(scores, inventory))),
            key=lambda x: (x[0], x[1]), reverse=True,
        ))
        row = _query_metrics(doc, ranked, config=config)
        row.update({
            "document_id": doc.document_id,
            "dataset": str(doc.metadata.get("dataset", "unknown")),
            "query": query_text(doc),
            "top_tile_id": ranked[0].tile_id if ranked else None,
            "top_score": max(scores) if scores else None,
            "gold_alternatives": len(doc.tiles),
        })
        per_query.append(row)
    metric_names = sorted({k for row in per_query for k, v in row.items() if isinstance(v, (int, float)) and k != "gold_alternatives"})
    metrics = {name: mean(float(row[name]) for row in per_query if name in row) for name in metric_names}
    return RankedRetrievalResult(method=method, n=len(per_query), metrics=metrics, per_query=tuple(per_query))


def build_embedding_topneighbor_inventory(
    doc: AdaptedDocument,
    *,
    backend: EmbeddingBackend,
    max_overlay_ratio: float = 1.0,
    construction_top_k: int = 1,
    priority_mode: str = "score_locality",
    locality_exponent: float = 2.0,
    exclude_adjacent: bool = False,
) -> tuple[SemanticTile, ...]:
    """Construct Mosaic overlays from a modern embedding backend without a tuned threshold.

    Every atom contributes its top-k cosine neighbors; the fixed object budget then
    materializes a subset.  Avoiding a model-specific similarity threshold makes BGE/Qwen
    constructor comparisons less calibration-sensitive.
    """
    if construction_top_k <= 0:
        raise ValueError("construction_top_k must be > 0")
    atoms, _ = paragraph_atoms(doc)
    if len(atoms) < 2:
        return inventory_tiles(atoms, ())
    texts = [doc.text[a.start:a.end] for a in atoms]
    z = backend.encode_documents(texts)
    sim = z @ z.T
    edge_best: dict[tuple[int, int], float] = {}
    for i in range(len(atoms)):
        rows = []
        for j in range(len(atoms)):
            if i == j or (exclude_adjacent and abs(i - j) <= 1):
                continue
            rows.append((float(sim[i, j]), j))
        rows.sort(key=lambda x: (-x[0], x[1]))
        for score, j in rows[:construction_top_k]:
            key = (min(i, j), max(i, j))
            edge_best[key] = max(score, edge_best.get(key, -1.0))
    candidates = [SemanticTile(
        tile_id=f"embed_pair_{idx:05d}", ranges=(atoms[i], atoms[j]),
        kind="embedding_topneighbor", score=max(0.0, min(1.0, (score + 1.0) / 2.0)),
    ) for idx, ((i, j), score) in enumerate(sorted(edge_best.items()))]
    overlays = select_overlay_budget(
        atoms, candidates, max_overlay_ratio=max_overlay_ratio,
        priority_mode=priority_mode, locality_exponent=locality_exponent,
    )
    return inventory_tiles(atoms, overlays)

class PairRerankerBackend(Protocol):
    name: str
    def score_pairs(self, queries: Sequence[str], documents: Sequence[str]) -> list[float]: ...


def evaluate_pair_reranker(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
    *, backend: PairRerankerBackend,
    method: str,
    config: RetrievalConfig = RetrievalConfig(),
) -> RankedRetrievalResult:
    if len(documents) != len(inventories):
        raise ValueError("documents/inventories length mismatch")
    per_query: list[dict[str, Any]] = []
    for doc, inventory in zip(documents, inventories):
        q = query_text(doc)
        texts = [tile_text(doc.text, t) for t in inventory]
        scores = backend.score_pairs([q] * len(texts), texts) if texts else []
        ranked_rows = sorted(
            ((float(score), -idx, tile) for idx, (score, tile) in enumerate(zip(scores, inventory))),
            key=lambda x: (x[0], x[1]), reverse=True,
        )
        ranked = tuple(row[2] for row in ranked_rows)
        ranked_scores = tuple(row[0] for row in ranked_rows)
        row = _query_metrics(doc, ranked, config=config)
        row.update({
            "document_id": doc.document_id,
            "cluster_id": retrieval_cluster_id(doc),
            "dataset": str(doc.metadata.get("dataset", "unknown")),
            "query": q,
            "top_tile_id": ranked[0].tile_id if ranked else None,
            "top_score": ranked_scores[0] if ranked_scores else None,
            "gold_alternatives": len(doc.tiles),
            "ranked_top": _rank_trace(doc, ranked, ranked_scores, limit=config.trace_top_k),
        })
        per_query.append(row)
    metric_names = sorted({k for row in per_query for k, v in row.items() if isinstance(v, (int, float)) and k != "gold_alternatives"})
    metrics = {name: mean(float(row[name]) for row in per_query if name in row) for name in metric_names}
    return RankedRetrievalResult(method=method, n=len(per_query), metrics=metrics, per_query=tuple(per_query))


def _inventory_cache_key(doc: AdaptedDocument, inventory: Sequence[SemanticTile]) -> str:
    h = hashlib.sha256(doc.text.encode("utf-8"))
    for tile in inventory:
        for r in tile.ranges:
            h.update(f"|{r.start}:{r.end}".encode())
        h.update(b";")
    return h.hexdigest()


def evaluate_embedding_retrieval_cached(
    documents: Sequence[AdaptedDocument],
    inventories: Sequence[Sequence[SemanticTile]],
    *,
    backend: EmbeddingBackend,
    method: str,
    hybrid_lexical_weight: float | None = None,
    config: RetrievalConfig = RetrievalConfig(),
) -> tuple[RankedRetrievalResult, dict[str, Any]]:
    """Evaluate a bi-encoder with a reusable per-inventory embedding index.

    This mirrors deployment more faithfully than re-encoding every candidate tile for
    every query. QASPER in particular has many questions per paper.  Query encoding is
    still performed per query; candidate embeddings are built once per unique inventory.
    """
    if len(documents) != len(inventories):
        raise ValueError("documents/inventories length mismatch")
    import numpy as np
    index_cache: dict[str, tuple[list[str], Any]] = {}
    per_query: list[dict[str, Any]] = []
    encoded_candidates = 0
    for doc, inventory in zip(documents, inventories):
        key = _inventory_cache_key(doc, inventory)
        if key not in index_cache:
            texts = [tile_text(doc.text, t) for t in inventory]
            z = backend.encode_documents(texts) if texts else np.zeros((0, 1), dtype="float32")
            index_cache[key] = (texts, z)
            encoded_candidates += len(texts)
        texts, dz = index_cache[key]
        q = query_text(doc)
        qz = backend.encode_queries([q])[0]
        dense = [float(x) for x in (dz @ qz)] if len(texts) else []
        if hybrid_lexical_weight is None:
            scores = dense
        else:
            lex = _minmax(lexical_query_scores(q, texts))
            emb = _minmax(dense)
            scores = [hybrid_lexical_weight * a + (1.0 - hybrid_lexical_weight) * b for a, b in zip(lex, emb)]
        ranked_rows = sorted(
            ((float(score), -idx, tile) for idx, (score, tile) in enumerate(zip(scores, inventory))),
            key=lambda x: (x[0], x[1]), reverse=True,
        )
        ranked = tuple(row[2] for row in ranked_rows)
        ranked_scores = tuple(row[0] for row in ranked_rows)
        row = _query_metrics(doc, ranked, config=config)
        row.update({
            "document_id": doc.document_id,
            "cluster_id": retrieval_cluster_id(doc),
            "dataset": str(doc.metadata.get("dataset", "unknown")),
            "query": q,
            "top_tile_id": ranked[0].tile_id if ranked else None,
            "top_score": ranked_scores[0] if ranked_scores else None,
            "gold_alternatives": len(doc.tiles),
            "ranked_top": _rank_trace(doc, ranked, ranked_scores, limit=config.trace_top_k),
        })
        per_query.append(row)
    metric_names = sorted({k for row in per_query for k, v in row.items() if isinstance(v, (int, float)) and k != "gold_alternatives"})
    metrics = {name: mean(float(row[name]) for row in per_query if name in row) for name in metric_names}
    return RankedRetrievalResult(method=method, n=len(per_query), metrics=metrics, per_query=tuple(per_query)), {
        "unique_embedding_indexes": len(index_cache),
        "encoded_candidate_tiles": encoded_candidates,
        "queries": len(documents),
        "cache_semantics": "candidate tile embeddings once per unique text+inventory; query embeddings per query",
    }
