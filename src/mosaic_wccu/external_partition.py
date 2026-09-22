from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalTile:
    tile_id: str
    quotes: tuple[str, ...]
    semantic_key: str | None = None
    dependencies: tuple[str, ...] = ()
    confidence: float | None = None


class ExternalPartitionError(ValueError):
    pass


def validate_external_tiles(text: str, rows: list[dict[str, Any]]) -> list[ExternalTile]:
    """Validate LLM/learned-model tile outputs without trusting raw model offsets.

    External policies return exact quotes.  We resolve only uniquely occurring quotes
    in this first protocol; ambiguous outputs fail closed and can later use contextual
    selectors compatible with anchor_v1.
    """
    seen: set[str] = set()
    out: list[ExternalTile] = []
    for i, row in enumerate(rows):
        tile_id = row.get("tile_id")
        quotes = row.get("quotes")
        if not isinstance(tile_id, str) or not tile_id or tile_id in seen:
            raise ExternalPartitionError(f"invalid/duplicate tile_id at row {i}")
        if not isinstance(quotes, list) or not quotes or not all(isinstance(q, str) and q for q in quotes):
            raise ExternalPartitionError(f"row {i}: quotes must be a non-empty string list")
        for q in quotes:
            count = text.count(q)
            if count != 1:
                raise ExternalPartitionError(f"row {i}: quote {q!r} occurs {count} times; fail closed")
        deps = row.get("dependencies", [])
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ExternalPartitionError(f"row {i}: dependencies must be a string list")
        conf = row.get("confidence")
        if conf is not None and not isinstance(conf, (int, float)):
            raise ExternalPartitionError(f"row {i}: confidence must be numeric")
        out.append(ExternalTile(
            tile_id=tile_id,
            quotes=tuple(quotes),
            semantic_key=row.get("semantic_key") if isinstance(row.get("semantic_key"), str) else None,
            dependencies=tuple(deps),
            confidence=float(conf) if conf is not None else None,
        ))
        seen.add(tile_id)
    unknown = {d for tile in out for d in tile.dependencies if d not in seen}
    if unknown:
        raise ExternalPartitionError(f"dependencies reference unknown tiles: {sorted(unknown)}")
    return out
