from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3e import _evaluate_predictions, overlapping_ego_tiles_from_prepared
from .phase3f import _dedupe_semantic_tiles, prepare_hotpot_title_bridge_graph
from .phase3h import _sentence_atoms, build_2wiki_multiview_predictions
from .segmentation import Segment
from .semantic_tiles import PreparedLexicalGraph, SemanticTile, prepare_lexical_graph

try:
    import numpy as np
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # pragma: no cover - exercised by dependency guard
    np = None
    sparse = None
    TruncatedSVD = None
    HashingVectorizer = None
    TfidfTransformer = None
    LogisticRegression = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


def require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Phase 3I requires numpy/scipy/scikit-learn; install requirements_phase3i_free.txt"
        ) from _SKLEARN_IMPORT_ERROR


def deterministic_split(
    documents: Sequence[AdaptedDocument], *, train_fraction: float = 0.70
) -> tuple[list[AdaptedDocument], list[AdaptedDocument], dict[str, Any]]:
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be in [0.5, 0.9]")
    rows = []
    for doc in documents:
        digest = hashlib.sha256((doc.document_id + "\0" + doc.text).encode("utf-8")).hexdigest()
        rows.append((digest, doc.document_id, doc))
    rows.sort(key=lambda x: (x[0], x[1]))
    n_train = max(1, min(len(rows) - 1, round(len(rows) * train_fraction)))
    train = [x[2] for x in rows[:n_train]]
    holdout = [x[2] for x in rows[n_train:]]
    return train, holdout, {
        "method": "sha256(document_id\\0text)-sorted-prefix-train",
        "train_fraction": train_fraction,
        "train_documents": len(train),
        "holdout_documents": len(holdout),
    }


def deterministic_subsample(documents: Sequence[AdaptedDocument], limit: int) -> list[AdaptedDocument]:
    if limit <= 0 or len(documents) <= limit:
        return list(documents)
    rows = sorted(
        (
            hashlib.sha256((d.document_id + "\0" + d.text).encode("utf-8")).hexdigest(),
            d.document_id,
            d,
        )
        for d in documents
    )
    return [x[2] for x in rows[:limit]]


def _sentence_texts(doc: AdaptedDocument) -> list[str]:
    return [doc.text[a.start:a.end] for a in _sentence_atoms(doc)]


@dataclass
class LSAHashModel:
    n_features: int = 4096
    n_components: int = 48
    random_state: int = 20260902
    max_fit_sentences: int = 80000

    def __post_init__(self) -> None:
        require_sklearn()
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        self.tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.fitted = False

    def fit(self, documents: Sequence[AdaptedDocument]) -> "LSAHashModel":
        sentences: list[str] = []
        for doc in documents:
            sentences.extend(_sentence_texts(doc))
            if len(sentences) >= self.max_fit_sentences:
                break
        sentences = sentences[: self.max_fit_sentences]
        if len(sentences) < 4:
            raise ValueError("not enough sentences to fit LSA")
        x = self.vectorizer.transform(sentences)
        x = self.tfidf.fit_transform(x)
        max_components = max(2, min(self.n_components, x.shape[0] - 1, x.shape[1] - 1))
        if max_components != self.n_components:
            self.n_components = max_components
            self.svd = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        self.svd.fit(x)
        self.fitted = True
        return self

    def encode(self, sentences: Sequence[str]):
        if not self.fitted:
            raise RuntimeError("LSA model is not fitted")
        x = self.vectorizer.transform(list(sentences))
        x = self.tfidf.transform(x)
        z = self.svd.transform(x).astype("float32", copy=False)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return z / norms


def prepare_dense_graph(doc: AdaptedDocument, model: LSAHashModel) -> PreparedLexicalGraph:
    atoms = _sentence_atoms(doc)
    z = model.encode([doc.text[a.start:a.end] for a in atoms])
    sim = z @ z.T
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 2, len(atoms)):
            score = float(sim[i, j])
            if score > 0.0:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)
    return PreparedLexicalGraph(atoms, tuple(candidates))


def evaluate_dense_overlap(
    documents: Sequence[AdaptedDocument], model: LSAHashModel, *, threshold: float, top_k: int = 1,
    method: str = "lsa_dense_overlap"
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        prepared = prepare_dense_graph(doc, model)
        tiles = overlapping_ego_tiles_from_prepared(prepared, similarity_threshold=threshold, top_k=top_k)
        predictions.append((atoms, tiles))
    return _evaluate_predictions(
        documents, predictions, method=method,
        config={"threshold": threshold, "top_k": top_k, "n_features": model.n_features,
                "n_components": model.n_components}, include_bootstrap=False,
    )


def evaluate_multiview_plus_dense(
    documents: Sequence[AdaptedDocument], model: LSAHashModel, *, dense_threshold: float,
    lexical_threshold: float = 0.16, top_k: int = 1, method: str = "multiview_plus_lsa"
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        base_tiles: list[SemanticTile] = []
        lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        title_only = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
        fused = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
        dense = prepare_dense_graph(doc, model)
        for prepared, threshold in (
            (lexical, lexical_threshold), (title_only, lexical_threshold),
            (fused, lexical_threshold), (dense, dense_threshold),
        ):
            base_tiles.extend(overlapping_ego_tiles_from_prepared(
                prepared, similarity_threshold=threshold, top_k=top_k
            ))
        predictions.append((atoms, _dedupe_semantic_tiles(base_tiles)))
    return _evaluate_predictions(
        documents, predictions, method=method,
        config={"views": ["lexical", "title_only", "fused", "lsa_dense"],
                "dense_threshold": dense_threshold, "lexical_threshold": lexical_threshold,
                "top_k": top_k}, include_bootstrap=False,
    )


def _gold_atom_indices(doc: AdaptedDocument) -> set[int]:
    atoms = _sentence_atoms(doc)
    gold_ranges = {(r.start, r.end) for tile in doc.tiles for r in tile.ranges}
    return {i for i, a in enumerate(atoms) if (a.start, a.end) in gold_ranges}


def _token_set(text: str) -> set[str]:
    import re
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text) if len(t) > 2}


PAIR_FEATURE_NAMES = ("lexical_cosine", "lsa_cosine", "title_bridge", "normalized_distance", "same_paragraph", "token_jaccard", "length_ratio")


def pair_features(doc: AdaptedDocument, model: LSAHashModel):
    atoms = _sentence_atoms(doc)
    dense_z = model.encode([doc.text[a.start:a.end] for a in atoms])
    dense_sim = dense_z @ dense_z.T
    lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
    lexical_scores = {(i, j): float(score) for score, i, j in lexical.candidates}
    title_only = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
    title_scores = {(i, j): float(score) for score, i, j in title_only.candidates}
    titles = [str(x) for x in doc.metadata.get("sentence_titles", ())]
    texts = [doc.text[a.start:a.end] for a in atoms]
    toks = [_token_set(t) for t in texts]
    n = max(1, len(atoms) - 1)
    rows = []
    pairs = []
    for i in range(len(atoms)):
        for j in range(i + 2, len(atoms)):
            a, b = toks[i], toks[j]
            union = len(a | b)
            jacc = len(a & b) / union if union else 0.0
            length_ratio = min(len(texts[i]), len(texts[j])) / max(1, max(len(texts[i]), len(texts[j])))
            rows.append([
                lexical_scores.get((i, j), 0.0),
                float(dense_sim[i, j]),
                1.0 if (i, j) in title_scores else 0.0,
                abs(i - j) / n,
                1.0 if i < len(titles) and j < len(titles) and titles[i] == titles[j] else 0.0,
                jacc,
                length_ratio,
            ])
            pairs.append((i, j))
    return np.asarray(rows, dtype="float32"), pairs


@dataclass
class SupervisedEdgeModel:
    lsa: LSAHashModel
    random_state: int = 20260902

    def __post_init__(self) -> None:
        require_sklearn()
        self.classifier = LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=self.random_state,
            solver="liblinear",
        )
        self.fitted = False

    def fit(self, documents: Sequence[AdaptedDocument], *, max_docs: int = 1200, negative_ratio: int = 6) -> "SupervisedEdgeModel":
        docs = deterministic_subsample(documents, max_docs)
        xs = []
        ys = []
        rng = random.Random(self.random_state)
        for doc in docs:
            features, pairs = pair_features(doc, self.lsa)
            gold = _gold_atom_indices(doc)
            pos_idx = [k for k, (i, j) in enumerate(pairs) if i in gold and j in gold]
            neg_idx = [k for k, (i, j) in enumerate(pairs) if not (i in gold and j in gold)]
            if not pos_idx:
                continue
            rng.shuffle(neg_idx)
            keep_neg = neg_idx[: min(len(neg_idx), max(negative_ratio * len(pos_idx), negative_ratio))]
            select = pos_idx + keep_neg
            xs.append(features[select])
            ys.extend([1] * len(pos_idx) + [0] * len(keep_neg))
        if not xs or len(set(ys)) < 2:
            raise ValueError("insufficient pair labels for supervised edge training")
        x = np.vstack(xs)
        y = np.asarray(ys, dtype="int32")
        self.classifier.fit(x, y)
        self.fitted = True
        self.training_pairs = int(len(y))
        self.positive_pairs = int(y.sum())
        return self

    def prepare_graph(self, doc: AdaptedDocument) -> PreparedLexicalGraph:
        if not self.fitted:
            raise RuntimeError("supervised edge model is not fitted")
        atoms = _sentence_atoms(doc)
        features, pairs = pair_features(doc, self.lsa)
        if not pairs:
            return PreparedLexicalGraph(atoms, ())
        prob = self.classifier.predict_proba(features)[:, 1]
        candidates = tuple(sorted(
            ((float(p), i, j) for p, (i, j) in zip(prob, pairs)), reverse=True
        ))
        return PreparedLexicalGraph(atoms, candidates)


def evaluate_supervised_overlap(
    documents: Sequence[AdaptedDocument], model: SupervisedEdgeModel, *, threshold: float,
    top_k: int = 1, method: str = "supervised_edge_overlap"
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        prepared = model.prepare_graph(doc)
        tiles = overlapping_ego_tiles_from_prepared(prepared, similarity_threshold=threshold, top_k=top_k)
        predictions.append((atoms, tiles))
    return _evaluate_predictions(
        documents, predictions, method=method,
        config={"threshold": threshold, "top_k": top_k,
                "training_pairs": getattr(model, "training_pairs", None),
                "positive_pairs": getattr(model, "positive_pairs", None)}, include_bootstrap=False,
    )


def evaluate_multiview_plus_supervised(
    documents: Sequence[AdaptedDocument], model: SupervisedEdgeModel, *, supervised_threshold: float,
    lexical_threshold: float = 0.16, top_k: int = 1, method: str = "multiview_plus_supervised"
) -> dict[str, Any]:
    predictions = []
    for doc in documents:
        atoms = _sentence_atoms(doc)
        tiles: list[SemanticTile] = []
        lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        title_only = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=False)
        fused = prepare_hotpot_title_bridge_graph(doc, include_lexical_base=True)
        supervised = model.prepare_graph(doc)
        for prepared, threshold in (
            (lexical, lexical_threshold), (title_only, lexical_threshold),
            (fused, lexical_threshold), (supervised, supervised_threshold),
        ):
            tiles.extend(overlapping_ego_tiles_from_prepared(
                prepared, similarity_threshold=threshold, top_k=top_k
            ))
        predictions.append((atoms, _dedupe_semantic_tiles(tiles)))
    return _evaluate_predictions(
        documents, predictions, method=method,
        config={"views": ["lexical", "title_only", "fused", "supervised_edge"],
                "supervised_threshold": supervised_threshold,
                "lexical_threshold": lexical_threshold, "top_k": top_k}, include_bootstrap=False,
    )


def choose_threshold(
    documents: Sequence[AdaptedDocument], evaluator, thresholds: Sequence[float], *, objective: str = "f1"
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best = None
    for threshold in thresholds:
        metrics = evaluator(documents, threshold)
        agg = metrics["all_gold"]
        # Encourage complete multi-range recovery and penalize object explosion lightly.
        score = float(agg["mean_f1"]) + 0.20 * float(agg["complete_coverage_rate"]) - 0.0005 * float(metrics["mean_total_objects_per_document"])
        row = {
            "threshold": threshold,
            "score": score,
            "mean_f1": agg["mean_f1"],
            "mean_recall": agg["mean_recall"],
            "complete_coverage_rate": agg["complete_coverage_rate"],
            "objects_per_document": metrics["mean_total_objects_per_document"],
        }
        rows.append(row)
        candidate = (score, -float(metrics["mean_total_objects_per_document"]), -threshold, threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return float(best[-1]), rows
