from dataclasses import dataclass

@dataclass
class AnchorResult:
    status: str  # resolved | ambiguous | missing
    start: int | None = None
    end: int | None = None


def resolve_quote(text: str, exact_quote: str) -> AnchorResult:
    if not exact_quote:
        return AnchorResult("missing")
    starts=[]; i=text.find(exact_quote)
    while i >= 0:
        starts.append(i); i=text.find(exact_quote, i+1)
    if not starts: return AnchorResult("missing")
    if len(starts)>1: return AnchorResult("ambiguous")
    s=starts[0]; return AnchorResult("resolved", s, s+len(exact_quote))
