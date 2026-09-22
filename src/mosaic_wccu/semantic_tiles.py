from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

from .segmentation import Segment, sentence_chunks


@dataclass(frozen=True)
class SemanticTile:
    """A semantic tile may contain multiple non-contiguous physical ranges."""

    tile_id: str
    ranges: tuple[Segment, ...]
    kind: str = "semantic"
    score: float = 1.0

    @property
    def physical_chars(self) -> int:
        return sum(r.length for r in self.ranges)

    @property
    def envelope_chars(self) -> int:
        if not self.ranges:
            return 0
        return max(r.end for r in self.ranges) - min(r.start for r in self.ranges)

    @property
    def is_noncontiguous(self) -> bool:
        if len(self.ranges) < 2:
            return False
        ordered = sorted(self.ranges, key=lambda r: (r.start, r.end))
        return any(a.end < b.start for a, b in zip(ordered, ordered[1:]))


@dataclass(frozen=True)
class PreparedLexicalGraph:
    """Threshold-independent lexical graph state reusable across sweeps."""

    atoms: tuple[Segment, ...]
    # Sorted descending by similarity. Only pairs satisfying min_shared_terms are stored.
    candidates: tuple[tuple[float, int, int], ...]


_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "will",
    "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 2 and t not in _STOP]


def _tfidf_vectors(parts: list[str]) -> list[dict[str, float]]:
    doc_tokens = [_tokens(p) for p in parts]
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for token in set(toks):
            df[token] = df.get(token, 0) + 1
    n = max(1, len(parts))
    out: list[dict[str, float]] = []
    for toks in doc_tokens:
        counts: dict[str, int] = {}
        for token in toks:
            counts[token] = counts.get(token, 0) + 1
        out.append({
            token: count * (math.log((1 + n) / (1 + df[token])) + 1.0)
            for token, count in counts.items()
        })
    return out


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def prepare_lexical_graph(
    text: str,
    atoms: Sequence[Segment],
    *,
    min_shared_terms: int = 1,
    link_adjacent: bool = False,
) -> PreparedLexicalGraph:
    """Precompute TF-IDF pair similarities once for repeated threshold evaluation."""
    if not text:
        return PreparedLexicalGraph((Segment(0, 0),), ())
    atoms = tuple(atoms)
    if not atoms:
        raise ValueError("atoms must not be empty")
    previous_end = -1
    for idx, span in enumerate(atoms):
        if not (0 <= span.start < span.end <= len(text)):
            raise ValueError(f"invalid atom {idx}: {span}")
        if span.start < previous_end:
            raise ValueError("atoms must be ordered and non-overlapping")
        previous_end = span.end

    parts = [text[s.start:s.end] for s in atoms]
    vectors = _tfidf_vectors(parts)
    term_sets = [set(_tokens(p)) for p in parts]
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(atoms)):
        first_j = i + 1 if link_adjacent else i + 2
        for j in range(first_j, len(atoms)):
            if len(term_sets[i] & term_sets[j]) < min_shared_terms:
                continue
            sim = _cos(vectors[i], vectors[j])
            candidates.append((sim, i, j))
    candidates.sort(reverse=True)
    return PreparedLexicalGraph(atoms, tuple(candidates))


def lexical_graph_tiles_from_prepared(
    prepared: PreparedLexicalGraph,
    *,
    similarity_threshold: float = 0.20,
    include_atomic_tiles: bool = True,
    max_cluster_units: int = 8,
) -> tuple[SemanticTile, ...]:
    """Build semantic tiles from threshold-independent prepared graph state."""
    atoms = prepared.atoms
    # Empty-document sentinel.
    if len(atoms) == 1 and atoms[0].start == atoms[0].end == 0:
        return (SemanticTile("atomic_0000", atoms, kind="atomic"),) if include_atomic_tiles else ()

    parent = list(range(len(atoms)))
    sizes = [1] * len(atoms)
    edge_score: dict[tuple[int, int], float] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb or sizes[ra] + sizes[rb] > max_cluster_units:
            return False
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        return True

    # Candidates are descending, so once similarity drops below the threshold we stop.
    for sim, i, j in prepared.candidates:
        if sim < similarity_threshold:
            break
        if union(i, j):
            edge_score[(i, j)] = sim

    clusters: dict[int, list[int]] = {}
    for idx in range(len(atoms)):
        clusters.setdefault(find(idx), []).append(idx)

    tiles: list[SemanticTile] = []
    if include_atomic_tiles:
        tiles.extend(
            SemanticTile(f"atomic_{i:04d}", (span,), kind="atomic", score=1.0)
            for i, span in enumerate(atoms)
        )

    semantic_idx = 0
    for members in sorted(clusters.values(), key=lambda xs: xs[0]):
        if len(members) < 2:
            continue
        ranges = tuple(atoms[i] for i in members)
        scores = [
            score for (i, j), score in edge_score.items()
            if i in members and j in members
        ]
        tiles.append(SemanticTile(
            f"semantic_{semantic_idx:04d}",
            ranges,
            kind="lexical_graph",
            score=(sum(scores) / len(scores)) if scores else 0.0,
        ))
        semantic_idx += 1
    return tuple(tiles)


def lexical_graph_tiles_from_segments(
    text: str,
    atoms: Sequence[Segment],
    *,
    similarity_threshold: float = 0.20,
    min_shared_terms: int = 1,
    include_atomic_tiles: bool = True,
    max_cluster_units: int = 8,
    link_adjacent: bool = False,
) -> tuple[SemanticTile, ...]:
    """Build overlapping semantic tiles over caller-supplied atomic spans.

    Threshold-independent pair scores are prepared separately so benchmark sweeps can
    reuse them. This wrapper preserves the original one-shot API.
    """
    prepared = prepare_lexical_graph(
        text,
        atoms,
        min_shared_terms=min_shared_terms,
        link_adjacent=link_adjacent,
    )
    return lexical_graph_tiles_from_prepared(
        prepared,
        similarity_threshold=similarity_threshold,
        include_atomic_tiles=include_atomic_tiles,
        max_cluster_units=max_cluster_units,
    )


def lexical_graph_tiles(
    text: str,
    *,
    similarity_threshold: float = 0.20,
    min_shared_terms: int = 1,
    include_atomic_tiles: bool = True,
    max_cluster_units: int = 8,
) -> tuple[SemanticTile, ...]:
    """Sentence-atomic convenience wrapper around lexical_graph_tiles_from_segments."""
    if not text:
        return (SemanticTile("atomic_0000", (Segment(0, 0),), kind="atomic"),)
    atoms = sentence_chunks(text).segments
    return lexical_graph_tiles_from_segments(
        text,
        atoms,
        similarity_threshold=similarity_threshold,
        min_shared_terms=min_shared_terms,
        include_atomic_tiles=include_atomic_tiles,
        max_cluster_units=max_cluster_units,
        link_adjacent=False,
    )


def validate_tiles(text: str, tiles: Iterable[SemanticTile]) -> None:
    n = len(text)
    seen_ids: set[str] = set()
    for tile in tiles:
        if tile.tile_id in seen_ids:
            raise ValueError(f"duplicate tile_id: {tile.tile_id}")
        seen_ids.add(tile.tile_id)
        if not tile.ranges:
            raise ValueError(f"{tile.tile_id}: no ranges")
        for r in tile.ranges:
            # Empty range is only tolerated for the empty-document sentinel.
            if n == 0 and r.start == r.end == 0:
                continue
            if not (0 <= r.start < r.end <= n):
                raise ValueError(f"{tile.tile_id}: invalid range {r}")
