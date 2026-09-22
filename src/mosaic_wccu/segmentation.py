from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Segmentation:
    policy: str
    segments: tuple[Segment, ...]

    def validate(self, text: str, *, require_cover: bool = True) -> None:
        n = len(text)
        if not self.segments:
            raise ValueError(f"{self.policy}: empty segmentation")
        for seg in self.segments:
            if not (0 <= seg.start < seg.end <= n):
                raise ValueError(f"{self.policy}: invalid segment {seg}")
        if require_cover:
            covered = [False] * n
            for seg in self.segments:
                for i in range(seg.start, seg.end):
                    covered[i] = True
            if n and not all(covered):
                missing = [i for i, value in enumerate(covered) if not value]
                raise ValueError(f"{self.policy}: uncovered offsets beginning at {missing[:5]}")

    @property
    def stored_chars(self) -> int:
        return sum(s.length for s in self.segments)


def _nonempty_spans(spans: Iterable[tuple[int, int]]) -> tuple[Segment, ...]:
    out: list[Segment] = []
    for start, end in spans:
        if start < end:
            out.append(Segment(start, end))
    return tuple(out)


def whole_text(text: str) -> Segmentation:
    return Segmentation("whole", (Segment(0, len(text)),))


def fixed_chunks(text: str, size: int = 64) -> Segmentation:
    if size <= 0:
        raise ValueError("size must be > 0")
    return Segmentation(
        f"fixed_{size}",
        _nonempty_spans((i, min(i + size, len(text))) for i in range(0, len(text), size)),
    )


def overlapping_windows(text: str, size: int = 80, stride: int = 40) -> Segmentation:
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be > 0")
    if len(text) <= size:
        return Segmentation(f"overlap_{size}_{stride}", (Segment(0, len(text)),))
    starts = list(range(0, max(1, len(text) - size + 1), stride))
    final_start = max(0, len(text) - size)
    if starts[-1] != final_start:
        starts.append(final_start)
    return Segmentation(
        f"overlap_{size}_{stride}",
        tuple(Segment(s, min(s + size, len(text))) for s in starts),
    )


def paragraph_chunks(text: str) -> Segmentation:
    # Keep paragraph separators attached to the preceding paragraph so coverage is exact.
    boundaries = [0]
    for match in re.finditer(r"\n\s*\n", text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))
    return Segmentation(
        "paragraph",
        _nonempty_spans(zip(boundaries[:-1], boundaries[1:])),
    )


def sentence_chunks(text: str) -> Segmentation:
    # Lightweight deterministic sentence-ish splitting; no model/provider dependency.
    cuts = {0, len(text)}
    for match in re.finditer(r"(?:[.!?](?:[\"')\]]*)\s+|\n\s*\n+)", text):
        cuts.add(match.end())
    ordered = sorted(cuts)
    return Segmentation("sentence", _nonempty_spans(zip(ordered[:-1], ordered[1:])))


_GEAR = tuple(
    int.from_bytes(__import__("hashlib").sha256(f"mosaic-gear-{i}".encode()).digest()[:8], "little")
    for i in range(256)
)


def cdc_gear_chunks(text: str, min_size: int = 32, avg_size: int = 64, max_size: int = 128) -> Segmentation:
    """Small deterministic Gear-hash CDC baseline with Unicode-safe offsets.

    This is a study baseline, not a claim of FastCDC equivalence. Gear state is
    updated from the UTF-8 byte stream, but a cut is considered only after a full
    Unicode code point has been consumed. Size limits and returned offsets are in
    Python character offsets. Therefore ASCII behavior is unchanged while arbitrary
    Unicode text can be segmented without splitting a UTF-8 sequence or confusing
    byte offsets with character offsets.
    """
    if not (0 < min_size <= avg_size <= max_size):
        raise ValueError("require 0 < min_size <= avg_size <= max_size")
    if not text:
        return Segmentation("cdc_gear", (Segment(0, 0),))
    bits = max(1, round(math.log2(avg_size)))
    mask = (1 << bits) - 1
    chunks: list[Segment] = []
    start_char = 0
    h = 0
    for char_index, char in enumerate(text):
        for byte in char.encode("utf-8"):
            h = ((h << 1) + _GEAR[byte]) & ((1 << 64) - 1)
        end_char = char_index + 1
        length = end_char - start_char
        if length < min_size:
            continue
        if (h & mask) == 0 or length >= max_size:
            chunks.append(Segment(start_char, end_char))
            start_char = end_char
            h = 0
    if start_char < len(text):
        chunks.append(Segment(start_char, len(text)))
    return Segmentation(f"cdc_gear_{min_size}_{avg_size}_{max_size}", tuple(chunks))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _cosine_bow(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    dot = sum(v * cb.get(t, 0) for t, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _sentence_boundaries(text: str) -> list[int]:
    seg = sentence_chunks(text)
    return [s.end for s in seg.segments[:-1]]


def lexical_tiling(text: str, threshold: float = 0.22, window_chars: int = 100) -> Segmentation:
    """TextTiling-inspired lexical-cohesion baseline over candidate sentence boundaries."""
    candidates = _sentence_boundaries(text)
    if not candidates:
        return whole_text(text).__class__("lexical_tiling", whole_text(text).segments)
    scored: list[tuple[int, float]] = []
    for pos in candidates:
        left = text[max(0, pos - window_chars):pos]
        right = text[pos:min(len(text), pos + window_chars)]
        scored.append((pos, _cosine_bow(left, right)))
    cuts = {0, len(text)}
    for idx, (pos, sim) in enumerate(scored):
        left_sim = scored[idx - 1][1] if idx else 1.0
        right_sim = scored[idx + 1][1] if idx + 1 < len(scored) else 1.0
        if sim <= threshold or (sim < left_sim and sim < right_sim and sim < 0.45):
            cuts.add(pos)
    ordered = sorted(cuts)
    return Segmentation("lexical_tiling", _nonempty_spans(zip(ordered[:-1], ordered[1:])))


def _candidate_boundaries(text: str, target_size: int) -> list[int]:
    cuts = {0, len(text)}
    # Strong structural/syntactic candidates.
    for match in re.finditer(r"\n\s*\n|[.!?;](?:[\"')\]]*)\s+", text):
        cuts.add(match.end())
    # Add sparse word-boundary candidates so the optimizer is not forced to use fixed offsets.
    last = 0
    for match in re.finditer(r"\s+", text):
        pos = match.end()
        if pos - last >= max(16, target_size // 3):
            cuts.add(pos)
            last = pos
    return sorted(cuts)


def _boundary_strength(text: str, pos: int, window_chars: int = 90) -> float:
    if pos <= 0 or pos >= len(text):
        return 1.0
    left = text[max(0, pos - window_chars):pos]
    right = text[pos:min(len(text), pos + window_chars)]
    semantic_shift = 1.0 - _cosine_bow(left, right)
    pre = text[max(0, pos - 4):pos]
    around = text[max(0, pos - 2):min(len(text), pos + 2)]
    structural = 0.0
    if "\n\n" in around or pre.endswith("\n\n"):
        structural = 1.0
    elif pre.rstrip().endswith((".", "!", "?")):
        structural = 0.85
    elif pre.rstrip().endswith(";"):
        structural = 0.65
    elif pre.rstrip().endswith(":"):
        structural = 0.20
    elif pre.rstrip().endswith(","):
        structural = 0.10
    return min(1.5, 0.72 * semantic_shift + 0.55 * structural)


def mosaic_dp_chunks(
    text: str,
    *,
    target_size: int = 64,
    min_size: int = 20,
    max_size: int = 140,
    collision_weight: float = 1.0,
    metadata_weight: float = 0.18,
    semantic_cut_weight: float = 0.55,
) -> Segmentation:
    """Versioning-aware global partition candidate.

    Objective (lower is better):
      - expected random-edit collision/write-amplification proxy ~ segment_length^2;
      - metadata cost per segment;
      - penalty for cutting through a weak semantic/structural boundary.

    This is deliberately provider-free. It is a proposed research candidate, not
    a novelty claim; later history-aware variants can replace the uniform edit prior.
    """
    if not text:
        return Segmentation("mosaic_dp", (Segment(0, 0),))
    candidates = _candidate_boundaries(text, target_size)
    n = len(text)
    best = [math.inf] * len(candidates)
    prev = [-1] * len(candidates)
    best[0] = 0.0

    for j in range(1, len(candidates)):
        end = candidates[j]
        for i in range(j - 1, -1, -1):
            start = candidates[i]
            length = end - start
            if length > max_size:
                break
            if length < min_size and end != n:
                continue
            # The squared term is proportional to expected same-chunk collision under a uniform edit prior.
            collision = collision_weight * (length / max(1, target_size)) ** 2
            size_regularizer = 0.10 * ((length - target_size) / max(1, target_size)) ** 2
            metadata = metadata_weight
            if end == n:
                cut_penalty = 0.0
            else:
                strength = _boundary_strength(text, end)
                cut_penalty = semantic_cut_weight * max(0.0, 1.0 - strength)
            cost = best[i] + collision + size_regularizer + metadata + cut_penalty
            if cost < best[j]:
                best[j] = cost
                prev[j] = i

    if math.isinf(best[-1]):
        # Fallback keeps the benchmark total rather than silently dropping a case.
        return fixed_chunks(text, max(min_size, target_size)).__class__(
            "mosaic_dp_fallback", fixed_chunks(text, max(min_size, target_size)).segments
        )

    boundaries = [len(candidates) - 1]
    cursor = len(candidates) - 1
    while cursor > 0:
        cursor = prev[cursor]
        if cursor < 0:
            raise RuntimeError("broken DP backpointer")
        boundaries.append(cursor)
    boundaries.reverse()
    points = [candidates[i] for i in boundaries]
    if points[0] != 0:
        points.insert(0, 0)
    return Segmentation("mosaic_dp", _nonempty_spans(zip(points[:-1], points[1:])))



def lexical_change_point(text: str, threshold: float = 0.48, window_chars: int = 72) -> Segmentation:
    """Traditional lexical change-point baseline over structural candidate boundaries."""
    candidates = _candidate_boundaries(text, 64)[1:-1]
    cuts = {0, len(text)}
    for pos in candidates:
        left = text[max(0, pos - window_chars):pos]
        right = text[pos:min(len(text), pos + window_chars)]
        sim = _cosine_bow(left, right)
        if sim <= threshold:
            cuts.add(pos)
    ordered = sorted(cuts)
    return Segmentation("lexical_change_point", _nonempty_spans(zip(ordered[:-1], ordered[1:])))



def tfidf_change_point(text: str, threshold: float = 0.32) -> Segmentation:
    """Provider-free TF-IDF adjacent-unit change-point baseline."""
    # Candidate units at sentence/semicolon/paragraph-like structural boundaries.
    cuts = [0]
    for match in re.finditer(r"\n\s*\n|[.!?;](?:[\"')\]]*)\s+|\n+", text):
        cuts.append(match.end())
    cuts.append(len(text))
    cuts = sorted(set(cuts))
    units = [(a, b, text[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if a < b]
    if len(units) <= 1:
        return Segmentation("tfidf_change_point", (Segment(0, len(text)),))
    token_sets = [set(_tokens(u[2])) for u in units]
    df: dict[str, int] = {}
    for ts in token_sets:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n_units = len(units)
    vectors = []
    for _, _, content in units:
        counts: dict[str, int] = {}
        for t in _tokens(content):
            counts[t] = counts.get(t, 0) + 1
        vec = {t: c * (math.log((1 + n_units) / (1 + df[t])) + 1.0) for t, c in counts.items()}
        vectors.append(vec)
    def cos(v1, v2):
        dot = sum(v * v2.get(k, 0.0) for k, v in v1.items())
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0
    sims = [cos(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    out_cuts = {0, len(text)}
    for i, sim in enumerate(sims):
        left = sims[i - 1] if i else 1.0
        right = sims[i + 1] if i + 1 < len(sims) else 1.0
        if sim <= threshold or (sim < left and sim < right and sim < 0.5):
            out_cuts.add(units[i][1])
    ordered = sorted(out_cuts)
    return Segmentation("tfidf_change_point", _nonempty_spans(zip(ordered[:-1], ordered[1:])))

def mosaic_dp_stress_tuned_v1(text: str) -> Segmentation:
    """Parameters chosen on partition_stress_v1 dev only; keep separate from the untuned candidate."""
    result = mosaic_dp_chunks(
        text,
        target_size=48,
        min_size=8,
        max_size=120,
        collision_weight=1.5,
        metadata_weight=0.6,
        semantic_cut_weight=0.2,
    )
    return Segmentation("mosaic_dp_stress_tuned_v1", result.segments)

def default_policies() -> dict[str, Callable[[str], Segmentation]]:
    return {
        "whole": whole_text,
        "fixed_48": lambda text: fixed_chunks(text, 48),
        "overlap_72_36": lambda text: overlapping_windows(text, 72, 36),
        "sentence": sentence_chunks,
        "paragraph": paragraph_chunks,
        "cdc_gear": lambda text: cdc_gear_chunks(text, 24, 48, 96),
        "lexical_tiling": lexical_tiling,
        "lexical_change_point": lexical_change_point,
        "tfidf_change_point": tfidf_change_point,
        "mosaic_dp": mosaic_dp_chunks,
        "mosaic_dp_tuned": mosaic_dp_stress_tuned_v1,
    }
