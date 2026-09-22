from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AppliedEdit:
    operation: str
    before_length: int
    after_length: int
    changed_before: tuple[int, int]
    changed_after: tuple[int, int]


class EditTraceError(ValueError):
    pass


def _require_range(text: str, start: int, end: int) -> None:
    if not (0 <= start <= end <= len(text)):
        raise EditTraceError(
            f"invalid range [{start}, {end}) for text length {len(text)}"
        )


def apply_edit(text: str, edit: Mapping[str, Any]) -> tuple[str, AppliedEdit]:
    """Apply one deterministic character-offset edit.

    Offsets are interpreted against the input text for this operation.  Move
    destinations are also expressed in pre-edit coordinates.  The function is
    deliberately strict so malformed frozen traces fail during construction,
    rather than being silently normalized by an experiment implementation.
    """

    op = str(edit.get("op", ""))
    before_len = len(text)

    if op in {"insert", "split"}:
        at = int(edit["at"])
        _require_range(text, at, at)
        inserted = str(edit.get("text", "\n\n" if op == "split" else ""))
        out = text[:at] + inserted + text[at:]
        return out, AppliedEdit(op, before_len, len(out), (at, at), (at, at + len(inserted)))

    if op in {"delete", "rewrite", "merge"}:
        start = int(edit["start"])
        end = int(edit["end"])
        _require_range(text, start, end)
        if start == end and op == "delete":
            raise EditTraceError("delete requires a non-empty range")
        replacement = "" if op == "delete" else str(edit.get("text", " " if op == "merge" else ""))
        if "expected_text" in edit and text[start:end] != edit["expected_text"]:
            raise EditTraceError(
                f"{op} expected {edit['expected_text']!r} at [{start}, {end}), "
                f"found {text[start:end]!r}"
            )
        out = text[:start] + replacement + text[end:]
        return out, AppliedEdit(
            op,
            before_len,
            len(out),
            (start, end),
            (start, start + len(replacement)),
        )

    if op == "move":
        start = int(edit["start"])
        end = int(edit["end"])
        destination = int(edit["destination"])
        _require_range(text, start, end)
        _require_range(text, destination, destination)
        if start == end:
            raise EditTraceError("move requires a non-empty source range")
        if start <= destination <= end:
            raise EditTraceError("move destination cannot lie inside the source range")
        segment = text[start:end]
        if "expected_text" in edit and segment != edit["expected_text"]:
            raise EditTraceError(
                f"move expected {edit['expected_text']!r} at [{start}, {end}), found {segment!r}"
            )
        without = text[:start] + text[end:]
        adjusted_destination = destination
        if destination > end:
            adjusted_destination -= end - start
        out = without[:adjusted_destination] + segment + without[adjusted_destination:]
        return out, AppliedEdit(
            op,
            before_len,
            len(out),
            (start, end),
            (adjusted_destination, adjusted_destination + len(segment)),
        )

    raise EditTraceError(f"unsupported edit operation: {op!r}")


def apply_trace(text: str, edits: list[Mapping[str, Any]]) -> tuple[str, list[AppliedEdit]]:
    applied: list[AppliedEdit] = []
    current = text
    for edit in edits:
        current, record = apply_edit(current, edit)
        applied.append(record)
    return current, applied
