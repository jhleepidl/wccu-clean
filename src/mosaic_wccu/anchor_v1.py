from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    exact_quote: str
    prefix: str = ""
    suffix: str = ""
    position_hint: tuple[int, int] | None = None


@dataclass(frozen=True)
class AnchorResultV1:
    status: str  # resolved | ambiguous | missing
    start: int | None = None
    end: int | None = None
    confidence: float = 0.0
    candidate_count: int = 0


def _occurrences(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    if not needle:
        return starts
    i = text.find(needle)
    while i >= 0:
        starts.append(i)
        i = text.find(needle, i + 1)
    return starts


def resolve_anchor(text: str, anchor: Anchor) -> AnchorResultV1:
    """Resolve an exact-quote anchor, using context only to disambiguate.

    This v1 resolver intentionally fails closed. Prefix/suffix and a position
    hint may reduce an exact-quote candidate set, but they never manufacture a
    match when the exact quote is absent. A fuzzy resolver can be studied as a
    later ablation without changing the frozen starter implementation.
    """

    starts = _occurrences(text, anchor.exact_quote)
    if not starts:
        return AnchorResultV1("missing", candidate_count=0)

    candidates = starts
    if anchor.prefix:
        candidates = [
            s for s in candidates if text[max(0, s - len(anchor.prefix)) : s] == anchor.prefix
        ]
    if anchor.suffix:
        qlen = len(anchor.exact_quote)
        candidates = [
            s
            for s in candidates
            if text[s + qlen : s + qlen + len(anchor.suffix)] == anchor.suffix
        ]

    if len(candidates) == 1:
        s = candidates[0]
        contextual = bool(anchor.prefix or anchor.suffix)
        return AnchorResultV1(
            "resolved",
            s,
            s + len(anchor.exact_quote),
            confidence=1.0 if contextual or len(starts) == 1 else 0.9,
            candidate_count=len(starts),
        )

    if not candidates:
        return AnchorResultV1("missing", candidate_count=len(starts))

    if anchor.position_hint is not None:
        hint_start, _ = anchor.position_hint
        distances = sorted((abs(s - hint_start), s) for s in candidates)
        if len(distances) == 1 or distances[0][0] < distances[1][0]:
            s = distances[0][1]
            return AnchorResultV1(
                "resolved",
                s,
                s + len(anchor.exact_quote),
                confidence=0.75,
                candidate_count=len(starts),
            )

    return AnchorResultV1("ambiguous", candidate_count=len(starts))
