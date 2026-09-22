from __future__ import annotations

from collections import defaultdict
import re
from typing import Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3e import _evaluate_predictions, overlapping_ego_tiles_from_prepared
from .phase3f import _dedupe_semantic_tiles
from .phase3i import LSAHashModel
from .segmentation import Segment
from .semantic_tiles import PreparedLexicalGraph, SemanticTile, prepare_lexical_graph

_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")


def _normalized_phrase(value: str) -> str:
    return " ".join(tok.lower() for tok in _ENTITY_TOKEN_RE.findall(value) if tok)


def paragraph_atoms(doc: AdaptedDocument) -> tuple[tuple[Segment, ...], tuple[str, ...]]:
    # Prefer exact paragraph boundaries when an adapter provides them.  Grouping
    # sentence atoms by repeated section titles can accidentally merge several
    # distinct paragraphs (notably in QASPER), so title grouping is only a
    # compatibility fallback for older adapters.
    raw_paragraphs = tuple(doc.metadata.get("paragraph_ranges", ()))
    raw_titles = tuple(str(x) for x in doc.metadata.get("paragraph_titles", ()))
    if raw_paragraphs:
        atoms = tuple(Segment(int(a), int(b)) for a, b in raw_paragraphs)
        if raw_titles and len(raw_titles) != len(atoms):
            raise ValueError(f"{doc.document_id}: paragraph range/title metadata mismatch")
        titles = raw_titles if raw_titles else tuple("" for _ in atoms)
        return atoms, titles

    sentence_atoms = tuple(Segment(int(a), int(b)) for a, b in doc.metadata.get("sentence_ranges", ()))
    titles = tuple(str(x) for x in doc.metadata.get("sentence_titles", ()))
    if not sentence_atoms or len(sentence_atoms) != len(titles):
        raise ValueError(f"{doc.document_id}: paragraph atoms require paragraph ranges or sentence ranges/titles")
    atoms: list[Segment] = []
    atom_titles: list[str] = []
    start = 0
    while start < len(sentence_atoms):
        end = start + 1
        while end < len(sentence_atoms) and titles[end] == titles[start]:
            end += 1
        atoms.append(Segment(sentence_atoms[start].start, sentence_atoms[end - 1].end))
        atom_titles.append(titles[start])
        start = end
    return tuple(atoms), tuple(atom_titles)


def prepare_title_bridge_for_atoms(
    doc: AdaptedDocument,
    atoms: Sequence[Segment],
    atom_titles: Sequence[str],
    *,
    include_lexical_base: bool,
) -> PreparedLexicalGraph:
    atoms = tuple(atoms)
    if len(atoms) != len(atom_titles):
        raise ValueError("atom title metadata mismatch")
    base = (
        prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
        if include_lexical_base
        else PreparedLexicalGraph(atoms, ())
    )
    pair_scores = {(i, j): float(score) for score, i, j in base.candidates}
    normalized_titles = [_normalized_phrase(t) for t in atom_titles]
    title_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, title in enumerate(normalized_titles):
        if title:
            title_to_indices[title].append(i)
    contents = [_normalized_phrase(doc.text[a.start:a.end]) for a in atoms]
    for i, content in enumerate(contents):
        own = normalized_titles[i]
        for target in normalized_titles:
            if not target or target == own or len(target) < 3 or target not in content:
                continue
            for j in title_to_indices.get(target, ()):
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                pair_scores[key] = max(pair_scores.get(key, 0.0), 0.34)
    candidates = tuple(sorted(((score, i, j) for (i, j), score in pair_scores.items()), reverse=True))
    return PreparedLexicalGraph(atoms, candidates)


def prepare_dense_for_atoms(doc: AdaptedDocument, atoms: Sequence[Segment], model: LSAHashModel) -> PreparedLexicalGraph:
    atoms = tuple(atoms)
    z = model.encode([doc.text[a.start:a.end] for a in atoms])
    sim = z @ z.T
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            score = float(sim[i, j])
            if score > 0:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)
    return PreparedLexicalGraph(atoms, tuple(candidates))


def paragraph_predictions(
    doc: AdaptedDocument,
    *,
    model: LSAHashModel | None = None,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    mode: str = "multiview",
    top_k: int = 1,
) -> tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]:
    atoms, titles = paragraph_atoms(doc)
    lexical = prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False)
    title = prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=False)
    fused = prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=True)
    specs: list[tuple[PreparedLexicalGraph, float]]
    if mode == "lexical":
        specs = [(lexical, lexical_threshold)]
    elif mode == "title":
        specs = [(title, lexical_threshold)]
    elif mode == "fused":
        specs = [(fused, lexical_threshold)]
    elif mode == "multiview":
        specs = [(lexical, lexical_threshold), (title, lexical_threshold), (fused, lexical_threshold)]
    elif mode == "multiview_dense":
        if model is None:
            raise ValueError("multiview_dense requires an LSA model")
        dense = prepare_dense_for_atoms(doc, atoms, model)
        specs = [(lexical, lexical_threshold), (title, lexical_threshold), (fused, lexical_threshold), (dense, dense_threshold)]
    else:
        raise ValueError(f"unknown paragraph Mosaic mode: {mode}")
    tiles: list[SemanticTile] = []
    for graph, threshold in specs:
        tiles.extend(overlapping_ego_tiles_from_prepared(graph, similarity_threshold=threshold, top_k=top_k))
    semantic = _dedupe_semantic_tiles(tiles) if len(specs) > 1 else tuple(tiles)
    return atoms, semantic


def evaluate_paragraph_mosaic(
    documents: Sequence[AdaptedDocument],
    *,
    model: LSAHashModel | None = None,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    mode: str = "multiview",
    top_k: int = 1,
) -> dict:
    predictions = [paragraph_predictions(
        doc, model=model, lexical_threshold=lexical_threshold,
        dense_threshold=dense_threshold, mode=mode, top_k=top_k,
    ) for doc in documents]
    return _evaluate_predictions(
        documents, predictions, method=f"paragraph_mosaic_{mode}",
        config={"mode": mode, "lexical_threshold": lexical_threshold,
                "dense_threshold": dense_threshold, "top_k": top_k,
                "development_only_after_musique_inspection": True},
        include_bootstrap=False,
    )
