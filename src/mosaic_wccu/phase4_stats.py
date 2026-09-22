from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("document_id", ""))
        if not key:
            raise ValueError("per-query row missing document_id")
        if key in out:
            raise ValueError(f"duplicate document_id in per-query trace: {key}")
        out[key] = row
    return out


def align_paired_rows(
    baseline: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Align two retrieval traces by case id and verify cluster identities."""
    a = _row_map(baseline)
    b = _row_map(challenger)
    if set(a) != set(b):
        missing_b = sorted(set(a) - set(b))[:5]
        missing_a = sorted(set(b) - set(a))[:5]
        raise ValueError(f"paired case mismatch: missing challenger={missing_b}, missing baseline={missing_a}")
    pairs = []
    for key in sorted(a):
        ra, rb = a[key], b[key]
        if str(ra.get("cluster_id")) != str(rb.get("cluster_id")):
            raise ValueError(f"cluster mismatch for {key}")
        pairs.append((ra, rb))
    return pairs


def paired_cluster_bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    samples: int = 20000,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Cluster bootstrap of challenger-minus-baseline mean retrieval metric.

    The resampling unit comes from ``cluster_id``.  For QASPER that is the paper,
    avoiding a false iid assumption across multiple questions from the same paper.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    pairs = align_paired_rows(baseline, challenger)
    grouped: dict[str, list[float]] = defaultdict(list)
    for ra, rb in pairs:
        if metric not in ra or metric not in rb:
            raise ValueError(f"metric missing from paired trace: {metric}")
        grouped[str(ra["cluster_id"])].append(float(rb[metric]) - float(ra[metric]))
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        raise ValueError("no clusters")

    import numpy as np

    sums = np.asarray([sum(grouped[c]) for c in cluster_ids], dtype="float64")
    counts = np.asarray([len(grouped[c]) for c in cluster_ids], dtype="float64")
    observed = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    reps = np.empty(samples, dtype="float64")
    c = len(cluster_ids)
    batch = 1000
    offset = 0
    while offset < samples:
        n = min(batch, samples - offset)
        idx = rng.integers(0, c, size=(n, c), endpoint=False)
        reps[offset:offset+n] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        offset += n
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {
        "metric": metric,
        "direction": "challenger_minus_baseline",
        "n_queries": len(pairs),
        "n_clusters": c,
        "bootstrap_samples": samples,
        "seed": seed,
        "mean_difference": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_probability_le_zero": float((reps <= 0.0).mean()),
    }
