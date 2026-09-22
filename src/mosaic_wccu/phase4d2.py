from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .benchmark_adapters import AdaptedDocument
from .phase4d1 import IndexedPredictionPolicy, indexed_paragraphs, validate_indexed_prediction
from .semantic_tiles import SemanticTile


@dataclass(frozen=True)
class MinimalIdPolicy:
    max_tiles: int = 8
    max_paragraphs_per_tile: int = 2
    terminator: str = 'END'
    abstain_token: str = 'NONE'


_LINE_RE = re.compile(r'^p\d{4}(?:\+p\d{4})?$')


def phase4d2_instruction() -> str:
    return (
        'Select evidence paragraph IDs only. Output 1 to 8 tile lines ordered from most useful to least useful, '
        'then output END on its own final line. Each tile line must be exactly p#### or p####+p####, using only IDs shown below. '
        'A + line may combine two non-contiguous paragraphs. If no paragraph is useful, output exactly NONE on one line and END on the next line. '
        'Do not output JSON, case_id, paragraph text, answers, reasoning, confidence, markdown, bullets, numbering, spaces around +, or any text after END.'
    )


def phase4d2_schema_demonstrations() -> tuple[dict[str, Any], ...]:
    return (
        {
            'case_id': 'synthetic_single',
            'query': 'Which city is the observatory in?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'The telescope was upgraded in 2024.'},
                {'id': 'p0002', 'text': 'The observatory is in La Serena, Chile.'},
                {'id': 'p0003', 'text': 'The project publishes open data.'},
            ],
            'answer_text': 'p0002\nEND',
        },
        {
            'case_id': 'synthetic_pair',
            'query': 'Which scientist led the project and where was she born?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'Dr. Mira Chen led the Aurora project.'},
                {'id': 'p0002', 'text': 'The instrument uses a cryogenic detector.'},
                {'id': 'p0003', 'text': 'Mira Chen was born in Taipei.'},
            ],
            'answer_text': 'p0001+p0003\nEND',
        },
        {
            'case_id': 'synthetic_abstain',
            'query': 'What is the launch date?',
            'paragraphs': [
                {'id': 'p0001', 'text': 'The proposal describes a new retrieval system.'},
                {'id': 'p0002', 'text': 'The appendix lists evaluation metrics.'},
            ],
            'answer_text': 'NONE\nEND',
        },
    )


def parse_minimal_id_output(text: str, *, policy: MinimalIdPolicy = MinimalIdPolicy()) -> tuple[list[list[str]] | None, str | None]:
    """Strictly parse the frozen Phase 4D.2 grammar.

    Only terminal CR/LF characters are treated as transport framing. No punctuation,
    quote, whitespace, or prose repair is performed.
    """
    if not isinstance(text, str):
        return None, 'response_not_text'
    framed = text.rstrip('\r\n')
    if not framed:
        return None, 'empty_response'
    lines = framed.splitlines()
    if not lines or lines[-1] != policy.terminator:
        return None, 'missing_exact_END_terminator'
    body = lines[:-1]
    if not body:
        return None, 'missing_tile_or_NONE_before_END'
    if body == [policy.abstain_token]:
        return [], None
    if policy.abstain_token in body:
        return None, 'NONE_must_be_the_only_body_line'
    if len(body) > policy.max_tiles:
        return None, f'exceeds_max_tiles={policy.max_tiles}'
    tiles: list[list[str]] = []
    for idx, line in enumerate(body):
        if not _LINE_RE.fullmatch(line):
            return None, f'invalid_tile_line_{idx+1}'
        ids = line.split('+')
        if len(ids) > policy.max_paragraphs_per_tile:
            return None, f'tile_{idx+1}_exceeds_max_paragraphs_per_tile={policy.max_paragraphs_per_tile}'
        tiles.append(ids)
    return tiles, None


def canonical_prediction(case_id: str, response_text: str, *, policy: MinimalIdPolicy = MinimalIdPolicy()) -> tuple[dict[str, Any] | None, str | None]:
    tiles, reason = parse_minimal_id_output(response_text, policy=policy)
    if tiles is None:
        return None, reason
    return {'case_id': case_id, 'tiles': tiles}, None


def validate_minimal_prediction(
    case: AdaptedDocument,
    prediction: Mapping[str, Any],
    *,
    policy: MinimalIdPolicy = MinimalIdPolicy(),
) -> tuple[SemanticTile, ...]:
    return validate_indexed_prediction(
        case,
        prediction,
        policy=IndexedPredictionPolicy(
            max_tiles=policy.max_tiles,
            max_paragraphs_per_tile=policy.max_paragraphs_per_tile,
        ),
    )


def physical_paragraph_id_map(case: AdaptedDocument) -> dict[tuple[int, int], tuple[str, str]]:
    return {(seg.start, seg.end): (pid, visible) for pid, seg, visible in indexed_paragraphs(case)}
