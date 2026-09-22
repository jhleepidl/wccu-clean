from .models import Update, Tile


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def physical_overlap(u1: Update, u2: Update) -> bool:
    return any(overlaps(a,b) for a in u1.physical_footprint for b in u2.physical_footprint)


def classify_pair(u1: Update, u2: Update) -> str:
    # Prototype only: same semantic write is a rebase conflict.
    if u1.writes & u2.writes:
        return "REBASE"
    # A write to a semantic dependency of the other update requires revalidation,
    # even if physical ranges are disjoint.
    if (u1.writes & u2.semantic_dependencies) or (u2.writes & u1.semantic_dependencies):
        return "REVALIDATE"
    # Physical overlap is conservative unless semantic write sets are proven independent.
    if physical_overlap(u1,u2):
        return "REBASE"
    return "COMMUTE"


def freshness_decision(update: Update, current_tiles: dict[str, Tile]) -> str:
    for r in update.reads:
        cur=current_tiles.get(r.tile_id)
        if cur is None or cur.version != r.version:
            return "REVALIDATE"
    return "FRESH"
