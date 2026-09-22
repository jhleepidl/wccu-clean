from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import random
from statistics import mean
from typing import Any, Callable, Sequence

from .segmentation import Segmentation, lexical_change_point, mosaic_dp_chunks, mosaic_dp_stress_tuned_v1


@dataclass(frozen=True)
class RevisionCase:
    case_id: str
    before: str
    changed_chars: int
    spans: tuple[tuple[int, int], ...]


def _changed_chars(before: str, after: str) -> int:
    sm = SequenceMatcher(a=before, b=after, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def prepare_revision_cases(rows: Sequence[dict[str, Any]]) -> list[RevisionCase]:
    out = []
    for row in rows:
        before, after = row["before"], row["after"]
        action_spans = []
        changed_from_actions = 0
        for action in row.get("edit_actions", []):
            if not isinstance(action, dict):
                continue
            try:
                start = int(action.get("start_char_pos")); end = int(action.get("end_char_pos"))
            except (TypeError, ValueError):
                continue
            before_piece = str(action.get("before") or "")
            after_piece = str(action.get("after") or "")
            if start == end and before:
                span = (max(0, min(start, len(before)-1)), min(len(before), max(1, start+1)))
            elif 0 <= start < end <= len(before):
                span = (start, end)
            else:
                continue
            action_spans.append(span)
            changed_from_actions += max(1, len(before_piece), len(after_piece))
        if action_spans:
            spans = tuple(action_spans)
            changed = max(1, changed_from_actions)
        else:
            sm = SequenceMatcher(a=before, b=after, autojunk=False)
            spans = tuple((i1, i2) for tag, i1, i2, _, _ in sm.get_opcodes() if tag != "equal" and i1 < i2)
            if not spans and before != after and before:
                spans = ((len(before)-1, len(before)),)
            changed = max(1, _changed_chars(before, after))
        out.append(RevisionCase(row["case_id"], before, changed, spans))
    return out


def deterministic_case_sample(cases: Sequence[RevisionCase], limit: int) -> tuple[list[RevisionCase], dict[str, Any]]:
    scored = []
    for case in cases:
        digest = hashlib.sha256((case.before + "\0" + case.case_id).encode("utf-8")).hexdigest()
        scored.append((digest, case.case_id, case))
    scored.sort(key=lambda x: (x[0], x[1]))
    chosen = [c for _, _, c in scored[:min(limit, len(scored))]]
    return chosen, {
        "method": "sha256(before+NUL+case_id)-sorted-prefix",
        "requested": limit,
        "selected": len(chosen),
        "case_ids": [c.case_id for c in chosen],
    }


def evaluate_revision_policy(cases: Sequence[RevisionCase], fn: Callable[[str], Segmentation], *, name: str) -> dict[str, Any]:
    amps = []
    counts = []
    stored = []
    raw = []
    for case in cases:
        seg = fn(case.before)
        seg.validate(case.before)
        touched = [s for s in seg.segments if any(max(s.start,a) < min(s.end,b) for a,b in case.spans)]
        amp = sum(s.length for s in touched) / max(1, case.changed_chars)
        amps.append(amp); counts.append(len(seg.segments)); stored.append(seg.stored_chars/max(1,len(case.before)))
        raw.append({"case_id":case.case_id,"write_amplification":amp,"segments":len(seg.segments)})
    return {
        "policy": name,
        "rows": len(cases),
        "mean_write_amplification": mean(amps) if amps else 0.0,
        "mean_segments_per_document": mean(counts) if counts else 0.0,
        "mean_stored_chars_ratio": mean(stored) if stored else 0.0,
        "raw": raw,
    }


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        a = row["train_metrics"]["mean_write_amplification"]
        s = row["train_metrics"]["mean_segments_per_document"]
        dominated = False
        for other in rows:
            if other is row:
                continue
            oa = other["train_metrics"]["mean_write_amplification"]
            os = other["train_metrics"]["mean_segments_per_document"]
            if oa <= a and os <= s and (oa < a or os < s):
                dominated = True; break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r:(r["train_metrics"]["mean_segments_per_document"],r["train_metrics"]["mean_write_amplification"]))


def select_frontier_representatives(frontier: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not frontier:
        raise ValueError("empty frontier")
    locality = min(frontier, key=lambda r:(r["train_metrics"]["mean_write_amplification"], r["train_metrics"]["mean_segments_per_document"]))
    compact = min(frontier, key=lambda r:(r["train_metrics"]["mean_segments_per_document"], r["train_metrics"]["mean_write_amplification"]))
    amps=[r["train_metrics"]["mean_write_amplification"] for r in frontier]
    segs=[r["train_metrics"]["mean_segments_per_document"] for r in frontier]
    amin,amax=min(amps),max(amps); smin,smax=min(segs),max(segs)
    def dist(r):
        a=r["train_metrics"]["mean_write_amplification"]; s=r["train_metrics"]["mean_segments_per_document"]
        an=(a-amin)/max(1e-12,amax-amin); sn=(s-smin)/max(1e-12,smax-smin)
        return an*an+sn*sn
    balanced=min(frontier,key=dist)
    return {"locality":locality,"balanced":balanced,"compact":compact}


def paired_bootstrap_amp_difference(left: dict[str, Any], right: dict[str, Any], *, seed: int=20260902, samples: int=4000) -> dict[str, Any]:
    l={r["case_id"]:r["write_amplification"] for r in left["raw"]}
    rr={r["case_id"]:r["write_amplification"] for r in right["raw"]}
    keys=sorted(set(l)&set(rr)); diffs=[l[k]-rr[k] for k in keys]
    if not diffs: return {"n":0,"mean_difference":0.0,"low_95":0.0,"high_95":0.0}
    rng=random.Random(seed); n=len(diffs); boots=[]
    for _ in range(samples): boots.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    boots.sort()
    return {"n":n,"mean_difference":sum(diffs)/n,"low_95":boots[int(.025*(samples-1))],"high_95":boots[int(.975*(samples-1))],"bootstrap_samples":samples,"left_minus_right":f"{left['policy']} - {right['policy']}"}


def mosaic_candidate_fn(params: dict[str, Any]):
    return lambda text: mosaic_dp_chunks(text, **params)


def baseline_functions() -> dict[str, Callable[[str], Segmentation]]:
    return {
        "lexical_change_point": lexical_change_point,
        "mosaic_dp_tuned_phase3a": mosaic_dp_stress_tuned_v1,
    }
