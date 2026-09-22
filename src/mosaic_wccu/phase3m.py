from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import math
import random
import subprocess
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .benchmark_adapters import AdaptedDocument, GoldSemanticTile
from .phase3e import _tile_scores, overlapping_ego_tiles_from_prepared
from .phase3i import LSAHashModel, deterministic_subsample, require_sklearn
from .phase3j import paragraph_atoms, prepare_dense_for_atoms, prepare_title_bridge_for_atoms
from .phase3l import _atomic_tiles, evaluate_full_inventory, select_overlay_budget
from .segmentation import Segment, Segmentation, mosaic_dp_stress_tuned_v1, paragraph_chunks, sentence_chunks, whole_text
from .semantic_tiles import PreparedLexicalGraph, SemanticTile, prepare_lexical_graph
from .workload import read_jsonl
from .partition_benchmark import NONCOMMUTE, _semantic_guard

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # pragma: no cover
    np = None
    LogisticRegression = None
    _SKLEARN_ERROR = exc
else:
    _SKLEARN_ERROR = None


@dataclass(frozen=True)
class OverlayCandidate:
    tile: SemanticTile
    view_scores: tuple[tuple[str, float], ...]

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.view_scores)

    def score_for(self, view: str) -> float:
        return dict(self.view_scores).get(view, 0.0)


VIEWS = ("lexical", "title", "fused", "dense")
OVERLAY_FEATURE_NAMES = (
    "max_score", "mean_score", "n_views",
    "lexical_score", "title_score", "fused_score", "dense_score",
    "member_count", "max_gap", "mean_gap", "coverage_ratio", "envelope_ratio",
)


def _range_key(tile: SemanticTile) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r.start, r.end) for r in tile.ranges))


def _aggregate_candidates(view_tiles: Iterable[tuple[str, SemanticTile]]) -> tuple[OverlayCandidate, ...]:
    rows: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    for view, tile in view_tiles:
        key = _range_key(tile)
        row = rows.setdefault(key, {"ranges": tile.ranges, "scores": {}})
        row["scores"][view] = max(float(tile.score), float(row["scores"].get(view, 0.0)))
    out = []
    for idx, key in enumerate(sorted(rows)):
        row = rows[key]
        scores = tuple(sorted((k, float(v)) for k, v in row["scores"].items()))
        max_score = max(v for _, v in scores)
        out.append(OverlayCandidate(
            SemanticTile(f"candidate_{idx:04d}", tuple(row["ranges"]), kind="phase3m_candidate", score=max_score),
            scores,
        ))
    return tuple(out)


def paragraph_overlay_candidate_records(
    doc: AdaptedDocument,
    *,
    model: LSAHashModel | None,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    top_k: int = 1,
) -> tuple[tuple[Segment, ...], tuple[OverlayCandidate, ...]]:
    atoms, titles = paragraph_atoms(doc)
    specs: list[tuple[str, PreparedLexicalGraph, float]] = [
        ("lexical", prepare_lexical_graph(doc.text, atoms, min_shared_terms=1, link_adjacent=False), lexical_threshold),
        ("title", prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=False), lexical_threshold),
        ("fused", prepare_title_bridge_for_atoms(doc, atoms, titles, include_lexical_base=True), lexical_threshold),
    ]
    if model is not None:
        specs.append(("dense", prepare_dense_for_atoms(doc, atoms, model), dense_threshold))
    view_tiles: list[tuple[str, SemanticTile]] = []
    for view, graph, threshold in specs:
        for tile in overlapping_ego_tiles_from_prepared(graph, similarity_threshold=threshold, top_k=top_k):
            view_tiles.append((view, tile))
    return atoms, _aggregate_candidates(view_tiles)


def generic_overlay_candidate_records(
    text: str,
    atoms: Sequence[Segment],
    *,
    model: LSAHashModel | None,
    lexical_threshold: float = 0.16,
    dense_threshold: float = 0.15,
    top_k: int = 1,
) -> tuple[OverlayCandidate, ...]:
    atoms = tuple(atoms)
    doc = AdaptedDocument(
        document_id="generic",
        text=text,
        tiles=(),
        metadata={"paragraph_ranges": tuple((a.start, a.end) for a in atoms), "paragraph_titles": tuple("" for _ in atoms)},
    )
    specs: list[tuple[str, PreparedLexicalGraph, float]] = [
        ("lexical", prepare_lexical_graph(text, atoms, min_shared_terms=1, link_adjacent=False), lexical_threshold),
    ]
    if model is not None and len(atoms) >= 2:
        specs.append(("dense", prepare_dense_for_atoms(doc, atoms, model), dense_threshold))
    view_tiles: list[tuple[str, SemanticTile]] = []
    for view, graph, threshold in specs:
        for tile in overlapping_ego_tiles_from_prepared(graph, similarity_threshold=threshold, top_k=top_k):
            view_tiles.append((view, tile))
    return _aggregate_candidates(view_tiles)


def candidate_features(atoms: Sequence[Segment], candidate: OverlayCandidate, text_len: int) -> list[float]:
    atoms = tuple(atoms)
    index = {(a.start, a.end): i for i, a in enumerate(atoms)}
    members = sorted(index[(r.start, r.end)] for r in candidate.tile.ranges if (r.start, r.end) in index)
    n = max(1, len(atoms) - 1)
    gaps = [abs(b - a) / n for i, a in enumerate(members) for b in members[i + 1:]]
    scores = dict(candidate.view_scores)
    vals = list(scores.values()) or [float(candidate.tile.score)]
    physical_chars = sum(r.length for r in candidate.tile.ranges)
    envelope = candidate.tile.envelope_chars
    return [
        max(vals),
        sum(vals) / len(vals),
        float(len(scores)),
        scores.get("lexical", 0.0),
        scores.get("title", 0.0),
        scores.get("fused", 0.0),
        scores.get("dense", 0.0),
        float(len(members)),
        max(gaps) if gaps else 0.0,
        sum(gaps) / len(gaps) if gaps else 0.0,
        physical_chars / max(1, text_len),
        envelope / max(1, physical_chars),
    ]


def _candidate_gain_over_physical(gold_tiles: Sequence[GoldSemanticTile], atoms: Sequence[Segment], cand: OverlayCandidate) -> float:
    if not gold_tiles:
        return 0.0
    physical = _atomic_tiles(atoms)
    best_gain = 0.0
    for gold in gold_tiles:
        base = max((_tile_scores(gold, p)[2] for p in physical), default=0.0)
        c = _tile_scores(gold, cand.tile)[2]
        best_gain = max(best_gain, c - base)
    return best_gain


@dataclass
class LearnedOverlaySelector:
    random_state: int = 20260902
    positive_gain: float = 0.05

    def __post_init__(self) -> None:
        require_sklearn()
        if _SKLEARN_ERROR is not None:
            raise RuntimeError("scikit-learn unavailable") from _SKLEARN_ERROR
        self.classifier = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=self.random_state,
            solver="liblinear",
        )
        self.fitted = False

    def fit(
        self,
        documents: Sequence[AdaptedDocument],
        *,
        lsa: LSAHashModel,
        max_docs: int = 1000,
        max_negative_ratio: int = 5,
    ) -> "LearnedOverlaySelector":
        docs = deterministic_subsample(documents, max_docs)
        xs: list[list[float]] = []
        ys: list[int] = []
        rng = random.Random(self.random_state)
        for doc in docs:
            atoms, candidates = paragraph_overlay_candidate_records(doc, model=lsa)
            positives: list[list[float]] = []
            negatives: list[list[float]] = []
            for cand in candidates:
                feat = candidate_features(atoms, cand, len(doc.text))
                gain = _candidate_gain_over_physical(doc.tiles, atoms, cand)
                (positives if gain >= self.positive_gain else negatives).append(feat)
            if not positives:
                continue
            rng.shuffle(negatives)
            negatives = negatives[: max(len(positives) * max_negative_ratio, max_negative_ratio)]
            xs.extend(positives); ys.extend([1] * len(positives))
            xs.extend(negatives); ys.extend([0] * len(negatives))
        if len(set(ys)) < 2:
            raise ValueError("insufficient positive/negative overlay labels")
        x = np.asarray(xs, dtype="float32")
        y = np.asarray(ys, dtype="int32")
        self.classifier.fit(x, y)
        self.fitted = True
        self.training_candidates = int(len(y))
        self.positive_candidates = int(y.sum())
        self.training_documents = len(docs)
        return self

    def probabilities(self, atoms: Sequence[Segment], candidates: Sequence[OverlayCandidate], text_len: int) -> list[float]:
        if not self.fitted:
            raise RuntimeError("learned overlay selector is not fitted")
        if not candidates:
            return []
        x = np.asarray([candidate_features(atoms, c, text_len) for c in candidates], dtype="float32")
        return [float(x) for x in self.classifier.predict_proba(x)[:, 1]]

    def select(
        self,
        atoms: Sequence[Segment],
        candidates: Sequence[OverlayCandidate],
        *,
        text_len: int,
        max_overlay_ratio: float = 1.0,
    ) -> tuple[SemanticTile, ...]:
        if max_overlay_ratio < 0:
            raise ValueError("max_overlay_ratio must be >= 0")
        budget = min(len(candidates), int(math.ceil(len(atoms) * max_overlay_ratio)))
        if budget <= 0:
            return ()
        probs = self.probabilities(atoms, candidates, text_len)
        rows = sorted(
            zip(probs, candidates),
            key=lambda x: (x[0], float(x[1].tile.score), -len(x[1].tile.ranges), _range_key(x[1].tile)),
            reverse=True,
        )
        out = []
        represented: set[tuple[tuple[int, int], ...]] = set()
        for prob, cand in rows:
            key = _range_key(cand.tile)
            if key in represented:
                continue
            represented.add(key)
            out.append(SemanticTile(
                f"learned_{len(out):04d}", cand.tile.ranges, kind="learned_budget_overlay", score=prob,
            ))
            if len(out) >= budget:
                break
        return tuple(out)


def static_select(atoms: Sequence[Segment], candidates: Sequence[OverlayCandidate], *, ratio: float = 1.0) -> tuple[SemanticTile, ...]:
    return select_overlay_budget(atoms, [c.tile for c in candidates], max_overlay_ratio=ratio, priority_mode="score")


def evaluate_selector_on_documents(
    documents: Sequence[AdaptedDocument],
    *,
    lsa: LSAHashModel,
    selector: LearnedOverlaySelector,
    ratio: float = 1.0,
) -> dict[str, Any]:
    predictions_physical = []
    predictions_static = []
    predictions_learned = []
    predictions_full = []
    for doc in documents:
        atoms, candidates = paragraph_overlay_candidate_records(doc, model=lsa)
        predictions_physical.append((atoms, ()))
        predictions_static.append((atoms, static_select(atoms, candidates, ratio=ratio)))
        predictions_learned.append((atoms, selector.select(atoms, candidates, text_len=len(doc.text), max_overlay_ratio=ratio)))
        predictions_full.append((atoms, tuple(c.tile for c in candidates)))
    return {
        "physical": evaluate_full_inventory(documents, predictions_physical, method="phase3m_physical", config={}),
        "static_budget": evaluate_full_inventory(documents, predictions_static, method="phase3m_static_budget", config={"ratio": ratio}),
        "learned_budget": evaluate_full_inventory(documents, predictions_learned, method="phase3m_learned_budget", config={"ratio": ratio}),
        "full": evaluate_full_inventory(documents, predictions_full, method="phase3m_full", config={}),
    }


def _intersects(seg: Segment, span: tuple[int, int]) -> bool:
    return max(seg.start, span[0]) < min(seg.end, span[1])


def _touching_physical(atoms: Sequence[Segment], span: tuple[int, int]) -> set[int]:
    return {i for i, a in enumerate(atoms) if _intersects(a, span)}


def _touching_overlay(overlays: Sequence[SemanticTile], span: tuple[int, int]) -> set[int]:
    return {i for i, t in enumerate(overlays) if any(_intersects(r, span) for r in t.ranges)}


def _write_amp(atoms: Sequence[Segment], span: tuple[int, int]) -> float:
    touched = _touching_physical(atoms, span)
    return sum(atoms[i].length for i in touched) / max(1, span[1] - span[0])


def generic_policy_inventory(
    text: str,
    *,
    physical: str,
    overlay: str,
    lsa: LSAHashModel | None,
    selector: LearnedOverlaySelector | None = None,
    ratio: float = 1.0,
) -> tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]:
    if physical == "whole":
        seg = whole_text(text)
    elif physical == "sentence":
        seg = sentence_chunks(text)
    elif physical == "mosaic_dp_tuned":
        seg = mosaic_dp_stress_tuned_v1(text)
    elif physical == "paragraph":
        seg = paragraph_chunks(text)
    else:
        raise ValueError(f"unknown physical policy {physical}")
    atoms = tuple(seg.segments)
    if overlay == "none" or len(atoms) < 2:
        return atoms, ()
    candidates = generic_overlay_candidate_records(text, atoms, model=lsa)
    if overlay == "full":
        return atoms, tuple(c.tile for c in candidates)
    if overlay == "static":
        return atoms, static_select(atoms, candidates, ratio=ratio)
    if overlay == "learned":
        if selector is None:
            raise ValueError("learned overlay requires selector")
        return atoms, selector.select(atoms, candidates, text_len=len(text), max_overlay_ratio=ratio)
    raise ValueError(f"unknown overlay policy {overlay}")


def run_wccu_tile_benchmark(
    cases_path: Path,
    oracle_path: Path,
    *,
    lsa: LSAHashModel,
    selector: LearnedOverlaySelector,
) -> dict[str, Any]:
    policies = {
        "whole": ("whole", "none"),
        "sentence_physical": ("sentence", "none"),
        "mosaic_dp_physical": ("mosaic_dp_tuned", "none"),
        "sentence_full_overlay": ("sentence", "full"),
        "sentence_static_budget": ("sentence", "static"),
        "sentence_learned_budget": ("sentence", "learned"),
        "mosaic_dp_learned_budget": ("mosaic_dp_tuned", "learned"),
    }
    cases = [c for c in read_jsonl(cases_path) if c.get("case_type") == "concurrency"]
    oracle = {o["case_id"]: o for o in read_jsonl(oracle_path) if o.get("case_type") == "concurrency"}
    rows: list[dict[str, Any]] = []
    for name, (physical, overlay) in policies.items():
        for case in cases:
            atoms, overlays = generic_policy_inventory(
                case["base_text"], physical=physical, overlay=overlay,
                lsa=lsa, selector=selector, ratio=1.0,
            )
            a = tuple(case["edit_a"]["physical_footprint"])
            b = tuple(case["edit_b"]["physical_footprint"])
            pa, pb = _touching_physical(atoms, a), _touching_physical(atoms, b)
            oa, ob = _touching_overlay(overlays, a), _touching_overlay(overlays, b)
            physical_share = bool(pa & pb)
            overlay_share = bool(oa & ob)
            shares = physical_share or overlay_share
            decision = oracle[case["case_id"]]["decision"]
            guard = _semantic_guard(case)
            contextual = any(bool(e.get("reads") or e.get("semantic_dependencies")) for e in (case["edit_a"], case["edit_b"]))
            overlay_revalidation_hint = bool(overlay_share and contextual and not physical_share)
            rows.append({
                "policy": name,
                "case_id": case["case_id"],
                "oracle": decision,
                "physical_share": physical_share,
                "overlay_share": overlay_share,
                "shares_any_tile": shares,
                "false_conflict": decision == "COMMUTE" and shares,
                "false_conflict_physical": decision == "COMMUTE" and physical_share,
                "unsafe_accept_tiles": decision in NONCOMMUTE and not shares,
                "unsafe_accept_guarded": decision in NONCOMMUTE and not (shares or guard),
                "automatic_overlay_rescue": decision in NONCOMMUTE and overlay_revalidation_hint,
                "overlay_revalidation_hint": overlay_revalidation_hint,
                "false_revalidation_hint": decision == "COMMUTE" and overlay_revalidation_hint,
                "unsafe_accept_read_conditioned": decision in NONCOMMUTE and not (physical_share or overlay_revalidation_hint),
                "unsafe_accept_read_conditioned_guarded": decision in NONCOMMUTE and not (physical_share or overlay_revalidation_hint or guard),
                "write_amp_a": _write_amp(atoms, a),
                "write_amp_b": _write_amp(atoms, b),
                "invalidated_objects_a": len(pa) + len(oa),
                "invalidated_objects_b": len(pb) + len(ob),
                "physical_objects": len(atoms),
                "semantic_objects": len(overlays),
                "total_objects": len(atoms) + len(overlays),
            })
    summary = {}
    for name in policies:
        sub = [r for r in rows if r["policy"] == name]
        safe = [r for r in sub if r["oracle"] == "COMMUTE"]
        unsafe = [r for r in sub if r["oracle"] in NONCOMMUTE]
        summary[name] = {
            "cases": len(sub),
            "false_conflict_rate_naive_overlay_mutex": sum(r["false_conflict"] for r in safe) / max(1, len(safe)),
            "false_conflict_rate_physical_mutex": sum(r["false_conflict_physical"] for r in safe) / max(1, len(safe)),
            "unsafe_accept_rate_tiles": sum(r["unsafe_accept_tiles"] for r in unsafe) / max(1, len(unsafe)),
            "unsafe_accept_rate_guarded": sum(r["unsafe_accept_guarded"] for r in unsafe) / max(1, len(unsafe)),
            "automatic_overlay_rescues": sum(r["automatic_overlay_rescue"] for r in unsafe),
            "false_revalidation_hint_rate": sum(r["false_revalidation_hint"] for r in safe) / max(1, len(safe)),
            "unsafe_accept_rate_read_conditioned": sum(r["unsafe_accept_read_conditioned"] for r in unsafe) / max(1, len(unsafe)),
            "unsafe_accept_rate_read_conditioned_guarded": sum(r["unsafe_accept_read_conditioned_guarded"] for r in unsafe) / max(1, len(unsafe)),
            "mean_write_amplification": mean([r["write_amp_a"] for r in sub] + [r["write_amp_b"] for r in sub]),
            "mean_invalidated_objects_per_edit": mean([r["invalidated_objects_a"] for r in sub] + [r["invalidated_objects_b"] for r in sub]),
            "mean_physical_objects": mean(r["physical_objects"] for r in sub),
            "mean_semantic_objects": mean(r["semantic_objects"] for r in sub),
            "mean_total_objects": mean(r["total_objects"] for r in sub),
        }
    return {"summary": summary, "rows": rows, "claim_boundary": "semantic guard remains authoritative; overlay-only safety is diagnostic"}


def _object_signature(text: str, ranges: Sequence[Segment]) -> tuple[str, ...]:
    return tuple(sorted(hashlib.sha256(text[r.start:r.end].encode("utf-8")).hexdigest() for r in ranges))


def _reuse_rate(old: Sequence[tuple[str, ...]], new: Sequence[tuple[str, ...]]) -> float:
    rem = Counter(new)
    matched = 0
    for sig in old:
        if rem[sig] > 0:
            rem[sig] -= 1
            matched += 1
    return matched / max(1, len(old))


def _changed_spans(opcodes) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    old_spans, new_spans, changed = [], [], 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        changed += (i2 - i1) + (j2 - j1)
        if i2 > i1: old_spans.append((i1, i2))
        if j2 > j1: new_spans.append((j1, j2))
    return old_spans, new_spans, changed


def _touched_count(atoms: Sequence[Segment], overlays: Sequence[SemanticTile], spans: Sequence[tuple[int, int]]) -> tuple[int, int]:
    p = {i for span in spans for i, a in enumerate(atoms) if _intersects(a, span)}
    s = {i for span in spans for i, t in enumerate(overlays) if any(_intersects(r, span) for r in t.ranges)}
    return len(p), len(s)


def run_revision_overlay_benchmark(
    bundle: Path,
    *,
    lsa: LSAHashModel,
    selector: LearnedOverlaySelector,
    file_path: str = "README.md",
) -> dict[str, Any]:
    policies = ("physical", "static_budget", "learned_budget", "full")
    with tempfile.TemporaryDirectory(prefix="mosaic_phase3m_revision_") as td:
        repo = Path(td) / "repo"
        subprocess.check_call(["git", "clone", "-q", str(bundle), str(repo)])
        commits = subprocess.check_output(["git", "log", "--follow", "--reverse", "--format=%H", "--", file_path], cwd=repo, text=True).splitlines()
        snapshots = []
        for commit in commits:
            try:
                text = subprocess.check_output(["git", "show", f"{commit}:{file_path}"], cwd=repo, text=True, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue
            snapshots.append((commit, text))
        inventory_cache: dict[tuple[str, str], tuple[tuple[Segment, ...], tuple[SemanticTile, ...]]] = {}
        def inv(commit: str, text: str, policy: str):
            key = (commit, policy)
            if key in inventory_cache:
                return inventory_cache[key]
            atoms = tuple(paragraph_chunks(text).segments)
            candidates = generic_overlay_candidate_records(text, atoms, model=lsa) if len(atoms) >= 2 else ()
            if policy == "physical": overlays = ()
            elif policy == "static_budget": overlays = static_select(atoms, candidates, ratio=1.0)
            elif policy == "learned_budget": overlays = selector.select(atoms, candidates, text_len=len(text), max_overlay_ratio=1.0)
            elif policy == "full": overlays = tuple(c.tile for c in candidates)
            else: raise ValueError(policy)
            inventory_cache[key] = (atoms, overlays)
            return atoms, overlays

        rows = []
        for (old_c, old), (new_c, new) in zip(snapshots, snapshots[1:]):
            if old == new: continue
            opcodes = SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
            old_spans, new_spans, changed = _changed_spans(opcodes)
            for policy in policies:
                oa, oo = inv(old_c, old, policy)
                na, no = inv(new_c, new, policy)
                old_phys_sig = [_object_signature(old, (a,)) for a in oa]
                new_phys_sig = [_object_signature(new, (a,)) for a in na]
                old_over_sig = [_object_signature(old, t.ranges) for t in oo]
                new_over_sig = [_object_signature(new, t.ranges) for t in no]
                old_p_touch, old_s_touch = _touched_count(oa, oo, old_spans)
                new_p_touch, new_s_touch = _touched_count(na, no, new_spans)
                rows.append({
                    "policy": policy,
                    "old_commit": old_c,
                    "new_commit": new_c,
                    "changed_chars": changed,
                    "old_physical_objects": len(oa),
                    "old_semantic_objects": len(oo),
                    "physical_reuse_rate": _reuse_rate(old_phys_sig, new_phys_sig),
                    "semantic_reuse_rate": _reuse_rate(old_over_sig, new_over_sig) if oo else 1.0,
                    "invalidated_physical_objects": old_p_touch + new_p_touch,
                    "invalidated_semantic_objects": old_s_touch + new_s_touch,
                    "invalidated_total_objects": old_p_touch + new_p_touch + old_s_touch + new_s_touch,
                    "invalidated_objects_per_1k_changed_chars": 1000.0 * (old_p_touch + new_p_touch + old_s_touch + new_s_touch) / max(1, changed),
                })
    summary = {}
    for policy in policies:
        sub = [r for r in rows if r["policy"] == policy]
        summary[policy] = {
            "revision_pairs": len(sub),
            "mean_physical_objects": mean(r["old_physical_objects"] for r in sub),
            "mean_semantic_objects": mean(r["old_semantic_objects"] for r in sub),
            "mean_physical_reuse_rate": mean(r["physical_reuse_rate"] for r in sub),
            "mean_semantic_reuse_rate": mean(r["semantic_reuse_rate"] for r in sub),
            "mean_invalidated_total_objects": mean(r["invalidated_total_objects"] for r in sub),
            "mean_invalidated_semantic_objects": mean(r["invalidated_semantic_objects"] for r in sub),
            "mean_invalidated_objects_per_1k_changed_chars": mean(r["invalidated_objects_per_1k_changed_chars"] for r in sub),
        }
    return {"file_path": file_path, "summary": summary, "rows": rows}
