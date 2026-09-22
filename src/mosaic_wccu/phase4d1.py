from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase3j import paragraph_atoms
from .segmentation import Segment
from .semantic_tiles import SemanticTile


@dataclass(frozen=True)
class IndexedPredictionPolicy:
    max_tiles: int = 8
    max_paragraphs_per_tile: int = 2


def phase4d1_instruction() -> str:
    return (
        'Return exactly one JSON object with keys case_id and tiles. Do not answer the query. '
        'tiles must be ordered from most useful to least useful and contain at most 8 items. '
        'Each tile is a JSON array containing 1 or 2 paragraph IDs copied exactly from the IDs shown. '
        'A tile may combine two non-contiguous paragraphs. Use an empty tiles array if no paragraph is useful. '
        'Do not output paragraph text, summaries, reasoning, confidence values, dependencies, markdown, or IDs that are not listed.'
    )


def phase4d1_schema_demonstrations() -> tuple[dict[str, Any], ...]:
    return (
        {
            'case_id': 'synthetic_single',
            'query': 'Which city is the observatory in?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'The telescope was upgraded in 2024.'},
                {'id': 'p0002', 'text': 'The observatory is in La Serena, Chile.'},
                {'id': 'p0003', 'text': 'The project publishes open data.'},
            ],
            'answer': {'case_id': 'synthetic_single', 'tiles': [['p0002']]},
        },
        {
            'case_id': 'synthetic_pair',
            'query': 'Which scientist led the project and where was she born?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'Dr. Mira Chen led the Aurora project.'},
                {'id': 'p0002', 'text': 'The instrument uses a cryogenic detector.'},
                {'id': 'p0003', 'text': 'Mira Chen was born in Taipei.'},
            ],
            'answer': {'case_id': 'synthetic_pair', 'tiles': [['p0001', 'p0003']]},
        },
        {
            'case_id': 'synthetic_abstain',
            'query': 'What is the launch date?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'The proposal describes a new retrieval system.'},
                {'id': 'p0002', 'text': 'The appendix lists evaluation metrics.'},
            ],
            'answer': {'case_id': 'synthetic_abstain', 'tiles': []},
        },
    )


def indexed_paragraphs(case: AdaptedDocument) -> tuple[tuple[str, Segment, str], ...]:
    atoms, _ = paragraph_atoms(case)
    out: list[tuple[str, Segment, str]] = []
    for i, seg in enumerate(atoms, start=1):
        visible = case.text[seg.start:seg.end].rstrip('\r\n')
        out.append((f'p{i:04d}', seg, visible))
    return tuple(out)


def export_indexed_cases(documents: Sequence[AdaptedDocument], path: Path, *, instruction: str | None = None) -> dict[str, Any]:
    effective = instruction or phase4d1_instruction()
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraph_counts: list[int] = []
    with path.open('w', encoding='utf-8') as f:
        for doc in documents:
            indexed = indexed_paragraphs(doc)
            paragraph_counts.append(len(indexed))
            row = {
                'case_id': doc.document_id,
                'query': str(doc.metadata.get('query') or doc.metadata.get('question') or ''),
                'paragraphs': [{'id': pid, 'text': text} for pid, _, text in indexed],
                'instruction': effective,
            }
            # query_text is stored by adapters under different keys; fall back to imported helper lazily.
            if not row['query']:
                from .phase4a import query_text
                row['query'] = query_text(doc)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {
        'cases': len(documents),
        'path': str(path),
        'gold_exposed': False,
        'instruction': effective,
        'mean_paragraphs': sum(paragraph_counts) / max(1, len(paragraph_counts)),
        'max_paragraphs': max(paragraph_counts, default=0),
    }


def validate_indexed_prediction(
    case: AdaptedDocument,
    prediction: Mapping[str, Any],
    *,
    policy: IndexedPredictionPolicy = IndexedPredictionPolicy(),
) -> tuple[SemanticTile, ...]:
    if str(prediction.get('case_id')) != case.document_id:
        raise ValueError('prediction case_id mismatch')
    raw_tiles = prediction.get('tiles')
    if not isinstance(raw_tiles, list):
        raise ValueError('prediction tiles must be a list')
    if len(raw_tiles) > policy.max_tiles:
        raise ValueError(f'prediction exceeds max_tiles={policy.max_tiles}')

    mapping = {pid: seg for pid, seg, _ in indexed_paragraphs(case)}
    out: list[SemanticTile] = []
    seen_keys: set[tuple[str, ...]] = set()
    for pos, raw in enumerate(raw_tiles):
        if not isinstance(raw, list) or not raw:
            raise ValueError(f'tile {pos} must be a non-empty paragraph-id array')
        if len(raw) > policy.max_paragraphs_per_tile:
            raise ValueError(f'tile {pos} exceeds max_paragraphs_per_tile={policy.max_paragraphs_per_tile}')
        ids = [str(x) for x in raw]
        if len(set(ids)) != len(ids):
            raise ValueError(f'tile {pos} repeats a paragraph id')
        unknown = [x for x in ids if x not in mapping]
        if unknown:
            raise ValueError(f'tile {pos} contains unknown paragraph ids: {unknown[:3]}')
        key = tuple(ids)
        if key in seen_keys:
            raise ValueError(f'tile {pos} duplicates an earlier tile')
        seen_keys.add(key)
        out.append(SemanticTile(
            tile_id=f't{pos+1}',
            ranges=tuple(mapping[x] for x in ids),
            kind='external_model_indexed_tile',
            score=float(len(raw_tiles) - pos),
        ))
    return tuple(out)


def load_indexed_predictions_fail_closed(
    documents: Sequence[AdaptedDocument],
    path: Path,
    *,
    policy: IndexedPredictionPolicy = IndexedPredictionPolicy(),
) -> tuple[tuple[tuple[SemanticTile, ...], ...], dict[str, Any]]:
    parsed: dict[str, Mapping[str, Any]] = {}
    malformed_lines = 0
    duplicate_ids: set[str] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            malformed_lines += 1
            continue
        if not isinstance(row, dict):
            malformed_lines += 1
            continue
        cid = str(row.get('case_id') or '')
        if not cid:
            malformed_lines += 1
            continue
        if cid in parsed:
            duplicate_ids.add(cid)
            continue
        parsed[cid] = row

    expected = {d.document_id for d in documents}
    extra_ids = sorted(set(parsed) - expected)
    outputs: list[tuple[SemanticTile, ...]] = []
    failures: list[dict[str, str]] = []
    valid = abstained = 0
    for doc in documents:
        if doc.document_id in duplicate_ids:
            outputs.append(())
            failures.append({'case_id': doc.document_id, 'reason': 'duplicate_case_id'})
            continue
        row = parsed.get(doc.document_id)
        if row is None:
            outputs.append(())
            failures.append({'case_id': doc.document_id, 'reason': 'missing_prediction'})
            continue
        try:
            tiles = validate_indexed_prediction(doc, row, policy=policy)
        except Exception as exc:
            outputs.append(())
            failures.append({'case_id': doc.document_id, 'reason': str(exc)})
            continue
        outputs.append(tiles)
        valid += 1
        if not tiles:
            abstained += 1

    audit = {
        'cases': len(documents),
        'valid_cases': valid,
        'invalid_or_missing_cases': len(documents) - valid,
        'invalid_or_missing_rate': (len(documents) - valid) / max(1, len(documents)),
        'valid_abstentions': abstained,
        'malformed_unassigned_lines': malformed_lines,
        'duplicate_case_ids': sorted(duplicate_ids),
        'extra_case_ids': extra_ids,
        'failures': failures,
        'semantics': 'unknown/malformed/missing paragraph-ID predictions are scored as empty tiles; no repair',
    }
    return tuple(outputs), audit
