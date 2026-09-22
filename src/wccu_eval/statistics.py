"""Paired stratified percentile bootstrap used in the close-control extension."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def paired_bootstrap(a: dict, b: dict, metric: str, *, ratio: bool = False,
                     seed: int = 20260921, replicates: int = 20000) -> dict:
    if not a or set(a) != set(b):
        raise ValueError("Nonempty, identical question IDs are required")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be positive")
    groups = defaultdict(list)
    for q, row in a.items():
        if str(row["stratum"]) != str(b[q]["stratum"]):
            raise ValueError("Stratum mismatch in a paired observation")
        groups[str(row["stratum"])].append(q)
        if not np.isfinite(row[metric]) or not np.isfinite(b[q][metric]):
            raise ValueError("Finite numeric outcomes are required")
    rng = np.random.default_rng(seed)
    ar, br = np.zeros(replicates), np.zeros(replicates)
    for _, qq in sorted(groups.items()):
        qq = sorted(qq)
        ix = rng.integers(0, len(qq), (replicates, len(qq)))
        weight = len(qq) / len(a)
        ar += weight * np.array([a[q][metric] for q in qq])[ix].mean(axis=1)
        br += weight * np.array([b[q][metric] for q in qq])[ix].mean(axis=1)
    av = sum(row[metric] for row in a.values()) / len(a)
    bv = sum(row[metric] for row in b.values()) / len(b)
    if ratio and (bv <= 0 or np.any(br <= 0)):
        raise ValueError("Ratio requires positive baseline means in every resample")
    stat = ar / br if ratio else ar - br
    return {"estimate": float(av / bv if ratio else av - bv),
            "ci95": np.quantile(stat, [.025, .975]).tolist(),
            "n": len(a), "replicates": replicates, "seed": seed,
            "marginal_exploratory": True}
