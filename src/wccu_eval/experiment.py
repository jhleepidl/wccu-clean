"""Run matched initial selectors and a shared, stateless fallback reference."""
from __future__ import annotations
from .common import normalize, score_answer, validate_job
from .reader import initial_answer, dynamic_reference, compose_selective
from .selection import POLICIES


def evaluate(jobs: list[dict], labels: dict, call, demos: str, *,
             policies: tuple[str, ...] = ("rank", "singleton", "bundle", "full"),
             budgets: tuple[int, ...] = (512, 1024, 2048), fallback_budget: int = 1024,
             include_reference: bool = True) -> list[dict]:
    """Labels score finished answers only; no label is passed to a selector or reader.

    The cache/transport must implement __call__(messages, limit) and usage(ids).
    Both one-call and selective outputs are retained. Full context has no source cap.
    """
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job IDs; split cohorts before evaluation")
    if not jobs or any(p not in POLICIES for p in policies):
        raise ValueError("Nonempty jobs and supported policies are required")
    if len(policies) != len(set(policies)) or len(budgets) != len(set(budgets)):
        raise ValueError("Duplicate policies/caps would duplicate observations")
    if any(type(b) is not int or b < 0 for b in budgets) or type(fallback_budget) is not int or fallback_budget < 0:
        raise ValueError("Source caps must be nonnegative integers")
    # Input validation happens before any potentially paid request. Labels stay
    # outside all reader/selector arguments and only score completed outputs.
    for job in jobs:
        validate_job(job)
        label = labels[job["id"]]
        if not isinstance(label.get("answers"), list) or not label["answers"] or not all(isinstance(a, str) for a in label["answers"]):
            raise ValueError("Nonempty answer-alias list required")
        support = label.get("support_indices")
        if not isinstance(support, list) or any(type(i) is not int or i < 0 or i >= len(job["docs"]) for i in support):
            raise ValueError("Invalid support indices")
    rows = []
    for job in jobs:
        validate_job(job)
        ref = None
        initials = []
        for policy in policies:
            for budget in ((0,) if policy == "full" else budgets):
                first = initial_answer(job, call, demos, policy, budget)
                initials.append((policy, budget, first))
        if include_reference or any(normalize(first["answer"]) == "unknown" for _, _, first in initials):
            ref = dynamic_reference(job, call, demos, fallback_budget)
        label = labels[job["id"]]
        if not label.get("answers") or not all(isinstance(a, str) for a in label["answers"]):
            raise ValueError("Nonempty answer-alias list required")
        supports = set(label["support_indices"])
        if any(type(i) is not int or i < 0 or i >= len(job["docs"]) for i in supports):
            raise ValueError("Invalid support indices")
        common = {"id": job["id"], "stratum": str(job.get("stratum", "all"))}
        for policy, budget, first in initials:
            for selective in (False, True):
                result = compose_selective(first, ref, selective)
                rows.append({**common, "policy": policy, "budget": None if policy == "full" else budget,
                             "selective": selective, **result, **score_answer(result["answer"], label["answers"]),
                             "complete_support": float(supports <= set(first["selected"])),
                             "first_em": score_answer(first["answer"], label["answers"])["em"],
                             "source_tokens": first["source_tokens"], **call.usage(result["request_ids"])})
        if include_reference:
            rows.append({**common, "policy": "always_dynamic", "budget": fallback_budget,
                         "selective": False, "trigger": True, "answer": ref["answer"],
                         "parser_ok": ref["parser_ok"], "request_ids": ref["request_ids"],
                         **score_answer(ref["answer"], label["answers"]),
                         "complete_support": float(supports <= set(ref["selected"])),
                         **call.usage(ref["request_ids"])})
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row.get("budget"), row["selective"])].append(row)
    result = []
    for (policy, budget, selective), rr in sorted(groups.items(), key=lambda kv: str(kv[0])):
        result.append({"policy": policy, "budget": budget, "selective": selective, "n": len(rr),
                       "fallback_count": sum(r.get("trigger", False) for r in rr),
                       **{k: sum(r[k] for r in rr) / len(rr) for k in ("em", "f1", "complete_support", "calls", "prompt_tokens", "output_tokens")}})
    return result
