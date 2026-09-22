from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .anchor_v1 import Anchor, resolve_anchor
from .edit_trace import apply_trace

VALID_ORACLES = {"COMMUTE", "REBASE", "REVALIDATE", "REVIEW", "REJECT"}
VALID_CASE_TYPES = {"anchor_stability", "concurrency"}


class WorkloadValidationError(ValueError):
    pass


def stable_json(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_span(text: str, quote: str) -> list[int]:
    starts: list[int] = []
    i = text.find(quote)
    while i >= 0:
        starts.append(i)
        i = text.find(quote, i + 1)
    if len(starts) != 1:
        raise WorkloadValidationError(
            f"expected exactly one occurrence of {quote!r}; found {len(starts)}"
        )
    return [starts[0], starts[0] + len(quote)]


def validate_range(text: str, value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(x, int) for x in value):
        raise WorkloadValidationError(f"{field} must be [start, end] integers")
    start, end = value
    if not (0 <= start <= end <= len(text)):
        raise WorkloadValidationError(
            f"{field} range [{start}, {end}) invalid for text length {len(text)}"
        )
    return start, end


def validate_case(case: dict[str, Any]) -> None:
    required = {"schema_version", "case_id", "case_type", "base_text"}
    missing = required - case.keys()
    if missing:
        raise WorkloadValidationError(f"missing fields {sorted(missing)}")
    if case["schema_version"] != "controlled-v1":
        raise WorkloadValidationError("unsupported schema_version")
    if case["case_type"] not in VALID_CASE_TYPES:
        raise WorkloadValidationError(f"invalid case_type {case['case_type']!r}")
    if not isinstance(case["base_text"], str) or not case["base_text"]:
        raise WorkloadValidationError("base_text must be non-empty")

    text = case["base_text"]
    if case["case_type"] == "anchor_stability":
        anchor = case.get("anchor")
        trace = case.get("trace")
        if not isinstance(anchor, dict) or not isinstance(trace, list) or not trace:
            raise WorkloadValidationError("anchor_stability requires anchor object and non-empty trace")
        if not isinstance(anchor.get("exact_quote"), str) or not anchor["exact_quote"]:
            raise WorkloadValidationError("anchor.exact_quote must be non-empty")
        # Strict trace application is itself an integrity check.
        apply_trace(text, trace)
        return

    tiles = case.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise WorkloadValidationError("concurrency case requires non-empty tiles")
    tile_ids: set[str] = set()
    for ti, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            raise WorkloadValidationError(f"tiles[{ti}] must be an object")
        tile_id = tile.get("tile_id")
        if not isinstance(tile_id, str) or not tile_id or tile_id in tile_ids:
            raise WorkloadValidationError(f"invalid/duplicate tile_id at tiles[{ti}]")
        tile_ids.add(tile_id)
        coverage = tile.get("coverage")
        if not isinstance(coverage, list) or not coverage:
            raise WorkloadValidationError(f"tiles[{ti}].coverage must be non-empty")
        for ri, r in enumerate(coverage):
            validate_range(text, r, f"tiles[{ti}].coverage[{ri}]")
        if not isinstance(tile.get("version"), int) or tile["version"] < 1:
            raise WorkloadValidationError(f"tiles[{ti}].version must be >= 1")
        dependencies = tile.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
            raise WorkloadValidationError(f"tiles[{ti}].dependencies must be string list")

    for ti, tile in enumerate(tiles):
        unknown = set(tile.get("dependencies", [])) - tile_ids
        if unknown:
            raise WorkloadValidationError(f"tiles[{ti}].dependencies reference unknown tiles {sorted(unknown)}")

    for edit_name in ("edit_a", "edit_b"):
        edit = case.get(edit_name)
        if not isinstance(edit, dict):
            raise WorkloadValidationError(f"{edit_name} must be an object")
        start, end = validate_range(text, edit.get("physical_footprint"), f"{edit_name}.physical_footprint")
        target_quote = edit.get("target_quote")
        if not isinstance(target_quote, str) or not target_quote:
            raise WorkloadValidationError(f"{edit_name}.target_quote must be non-empty")
        if text[start:end] != target_quote:
            raise WorkloadValidationError(
                f"{edit_name}.physical_footprint extracts {text[start:end]!r}, not target_quote {target_quote!r}"
            )
        if "anchor" in edit:
            anchor = edit["anchor"]
            if not isinstance(anchor, dict) or not isinstance(anchor.get("exact_quote"), str) or not anchor["exact_quote"]:
                raise WorkloadValidationError(f"{edit_name}.anchor must contain non-empty exact_quote")
        writes = edit.get("writes")
        if not isinstance(writes, list) or not writes or not all(w in tile_ids for w in writes):
            raise WorkloadValidationError(f"{edit_name}.writes must reference known tiles")
        reads = edit.get("reads", [])
        if not isinstance(reads, list):
            raise WorkloadValidationError(f"{edit_name}.reads must be a list")
        for ri, read in enumerate(reads):
            if not isinstance(read, dict) or read.get("tile_id") not in tile_ids or not isinstance(read.get("version"), int):
                raise WorkloadValidationError(f"invalid {edit_name}.reads[{ri}]")
        deps = edit.get("semantic_dependencies", [])
        if not isinstance(deps, list) or not all(d in tile_ids for d in deps):
            raise WorkloadValidationError(f"{edit_name}.semantic_dependencies must reference known tiles")


def validate_oracle(oracle: dict[str, Any], case_ids: set[str]) -> None:
    if oracle.get("schema_version") != "controlled-v1":
        raise WorkloadValidationError("oracle has unsupported schema_version")
    if oracle.get("case_id") not in case_ids:
        raise WorkloadValidationError(f"oracle references unknown case {oracle.get('case_id')!r}")
    case_type = oracle.get("case_type")
    if case_type not in VALID_CASE_TYPES:
        raise WorkloadValidationError("oracle has invalid case_type")
    if case_type == "anchor_stability":
        expected = oracle.get("expected")
        if not isinstance(expected, dict) or expected.get("anchor_status") not in {"resolved", "ambiguous", "missing"}:
            raise WorkloadValidationError("anchor oracle requires expected.anchor_status")
    else:
        if oracle.get("decision") not in VALID_ORACLES:
            raise WorkloadValidationError("concurrency oracle has invalid decision")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkloadValidationError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise WorkloadValidationError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def validate_frozen_workload(cases_path: Path, oracle_path: Path) -> dict[str, int]:
    cases = read_jsonl(cases_path)
    oracles = read_jsonl(oracle_path)
    seen: set[str] = set()
    type_counts = {"anchor_stability": 0, "concurrency": 0}
    for case in cases:
        validate_case(case)
        case_id = case["case_id"]
        if case_id in seen:
            raise WorkloadValidationError(f"duplicate case_id {case_id}")
        seen.add(case_id)
        type_counts[case["case_type"]] += 1
    oracle_seen: set[str] = set()
    for oracle in oracles:
        validate_oracle(oracle, seen)
        case_id = oracle["case_id"]
        if case_id in oracle_seen:
            raise WorkloadValidationError(f"duplicate oracle case_id {case_id}")
        oracle_seen.add(case_id)
    if seen != oracle_seen:
        raise WorkloadValidationError(
            f"case/oracle mismatch: missing_oracle={sorted(seen-oracle_seen)}, extra_oracle={sorted(oracle_seen-seen)}"
        )
    return {"cases": len(cases), **type_counts}


def evaluate_anchor_case(case: dict[str, Any]) -> dict[str, Any]:
    final_text, applied = apply_trace(case["base_text"], case["trace"])
    a = case["anchor"]
    hint = tuple(a["position_hint"]) if a.get("position_hint") is not None else None
    result = resolve_anchor(
        final_text,
        Anchor(
            exact_quote=a["exact_quote"],
            prefix=a.get("prefix", ""),
            suffix=a.get("suffix", ""),
            position_hint=hint,
        ),
    )
    return {
        "case_id": case["case_id"],
        "status": result.status,
        "start": result.start,
        "end": result.end,
        "candidate_count": result.candidate_count,
        "confidence": result.confidence,
        "final_text_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
        "operations": [record.operation for record in applied],
    }
