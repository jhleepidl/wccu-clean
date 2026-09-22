from dataclasses import dataclass, field
from typing import List, Tuple, Set

Range = Tuple[int, int]

@dataclass(frozen=True)
class TileVersion:
    tile_id: str
    version: int

@dataclass
class Tile:
    tile_id: str
    version: int
    coverage: List[Range]
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class Update:
    update_id: str
    reads: List[TileVersion]
    writes: Set[str]
    physical_footprint: List[Range]
    semantic_dependencies: Set[str] = field(default_factory=set)
    operation: str = "replace"
