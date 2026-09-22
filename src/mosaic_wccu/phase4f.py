from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from .benchmark_adapters import AdaptedDocument
from .segmentation import Segment
from .semantic_tiles import SemanticTile
from .phase4d2 import physical_paragraph_id_map


@dataclass(frozen=True)
class TileRerankPolicy:
    trace_depth: int = 8
    output_k: int = 2
    rrf_k: int = 60
    fusion_k: int = 60
    label_pool: tuple[str, ...] = tuple(str(i) for i in range(8))


def _deterministic_labels(case_id: str, n: int, pool: Sequence[str]) -> tuple[str, ...]:
    if n > len(pool):
        raise ValueError('not enough labels for candidates')
    keyed = []
    for label in pool[:n]:
        h = hashlib.sha256(f'{case_id}\0{label}'.encode('utf-8')).hexdigest()
        keyed.append((h, label))
    return tuple(label for _, label in sorted(keyed))


def rrf_tile_candidates(
    case: AdaptedDocument,
    rrf_per_query: Mapping[str, Any],
    *,
    policy: TileRerankPolicy = TileRerankPolicy(),
) -> tuple[dict[str, Any], ...]:
    """Recover the frozen RRF top tiles without consuming score or post-hoc gold fields."""
    if str(rrf_per_query.get('document_id') or '') != case.document_id:
        raise ValueError('RRF trace document_id mismatch')
    ranked = rrf_per_query.get('ranked_top')
    if not isinstance(ranked, list) or not ranked:
        raise ValueError('RRF trace missing ranked_top')
    rows = ranked[: policy.trace_depth]
    # A valid finite inventory may contain fewer candidates than nominal output_k.
    # Preserve the full available trace and let downstream top-k selection use
    # min(output_k, |C|); missing/empty traces remain fail-closed above.

    exact = physical_paragraph_id_map(case)
    labels = _deterministic_labels(case.document_id, len(rows), policy.label_pool)
    out: list[dict[str, Any]] = []
    for pos, (raw, label) in enumerate(zip(rows, labels), start=1):
        if not isinstance(raw, Mapping):
            raise ValueError('RRF ranked tile must be an object')
        raw_rank = int(raw.get('rank', pos))
        if raw_rank != pos:
            raise ValueError('RRF ranked_top rank order mismatch')
        ranges = raw.get('ranges')
        if not isinstance(ranges, list) or not ranges or len(ranges) > 2:
            raise ValueError('RRF tile must contain one or two ranges')
        mapped: list[tuple[int, int, str, str]] = []
        for pair in ranges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('RRF range must be [start,end]')
            key = (int(pair[0]), int(pair[1]))
            if key not in exact:
                raise ValueError(f'RRF range is not an exact physical paragraph atom: {key}')
            pid, visible = exact[key]
            mapped.append((key[0], key[1], pid, visible))
        out.append({
            'label': label,
            'rrf_rank': pos,
            'tile_id': str(raw.get('tile_id') or f'rrf_{pos:02d}'),
            'ranges': [[a, b] for a, b, _, _ in mapped],
            'paragraph_ids': [pid for _, _, pid, _ in mapped],
            'paragraphs': [text for _, _, _, text in mapped],
        })
    return tuple(out)


def validate_complete_label_ranking(
    candidates: Sequence[Mapping[str, Any]],
    ranked_labels: Sequence[str],
) -> tuple[str, ...]:
    expected = [str(c['label']) for c in candidates]
    got = [str(x) for x in ranked_labels]
    if len(got) != len(expected):
        raise ValueError('ranking length mismatch')
    if len(set(got)) != len(got):
        raise ValueError('ranking contains duplicate labels')
    if set(got) != set(expected):
        raise ValueError('ranking label set mismatch')
    return tuple(got)


def ranking_to_tiles(
    case: AdaptedDocument,
    candidates: Sequence[Mapping[str, Any]],
    ranked_labels: Sequence[str],
    *,
    output_k: int = 2,
    kind: str = 'external_model_rrf_tile_rerank',
) -> tuple[SemanticTile, ...]:
    ranking = validate_complete_label_ranking(candidates, ranked_labels)
    by_label = {str(c['label']): c for c in candidates}
    out = []
    for pos, label in enumerate(ranking[:output_k], start=1):
        c = by_label[label]
        ranges = tuple(Segment(int(a), int(b)) for a, b in c['ranges'])
        for r in ranges:
            if not (0 <= r.start < r.end <= len(case.text)):
                raise ValueError('candidate range outside document')
        out.append(SemanticTile(
            tile_id=f'{kind}_{pos:02d}_{c["tile_id"]}',
            ranges=ranges,
            kind=kind,
            score=float(output_k - pos + 1),
        ))
    return tuple(out)


def reciprocal_rank_fuse_tile_ranks(
    candidates: Sequence[Mapping[str, Any]],
    llm_ranked_labels: Sequence[str],
    *,
    k: int = 60,
) -> tuple[str, ...]:
    llm = validate_complete_label_ranking(candidates, llm_ranked_labels)
    llm_rank = {label: i for i, label in enumerate(llm, start=1)}
    scored = []
    for c in candidates:
        label = str(c['label'])
        rr = int(c['rrf_rank'])
        score = 1.0 / (k + rr) + 1.0 / (k + llm_rank[label])
        scored.append((score, -rr, label))
    scored.sort(reverse=True)
    return tuple(label for _, _, label in scored)


def normalize_allowed_logits(label_to_logit: Mapping[str, float]) -> dict[str, float]:
    if not label_to_logit:
        raise ValueError('empty label logits')
    values = {str(k): float(v) for k, v in label_to_logit.items()}
    mx = max(values.values())
    exp = {k: math.exp(v - mx) for k, v in values.items()}
    z = sum(exp.values())
    return {k: exp[k] / z for k in values}
