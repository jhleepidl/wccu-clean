from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PHASE6A_DATASET_REPO = "CooperBench/cooperbench-dataset"
PHASE6A_DATASET_REVISION = "b612b1a35af722751454813d9e5a7888f065fc9e"
PHASE6A_SEED = 20260908
PHASE6A_SCENARIOS = 150
PHASE6A_DIAGNOSTIC_PER_ISSUE_FAMILY = 60
PHASE6A_DIAGNOSTIC_SAFE = 300
PHASE6A_ISSUE_FAMILIES = (
    "freshness",
    "commitment",
    "authority",
    "operation",
    "materialized_view",
    "witness_gap",
)
PHASE6A_POLICIES = (
    "execution_witness_wccu",
    "projection_trace_wccu",
    "model_certificate_wccu",
    "read_set_occ",
    "wccu_without_read_validation",
    "adaptive_no_wccu",
    "snapshot_occ",
    "uniform_review",
    "append_only",
)


class Phase6AError(RuntimeError):
    pass


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))


def _feature_number(name: str) -> int | None:
    m = re.fullmatch(r"feature(\d+)", name)
    return int(m.group(1)) if m else None


def _task_number(name: str) -> str | None:
    m = re.fullmatch(r"task(.+)", name)
    return m.group(1) if m else None


def touched_files_from_patch(text: str) -> list[str]:
    files: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                files.add(parts[2][2:])
    return sorted(files)


def collect_pair_records(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all unordered feature pairs from a pinned CooperBench folder snapshot.

    The output intentionally carries only IDs, file names, and hashes; feature prose and patch
    contents remain in the external benchmark tree and are not copied into research artifacts.
    """
    if not dataset_root.is_dir():
        raise Phase6AError(f"missing CooperBench dataset root: {dataset_root}")
    records: list[dict[str, Any]] = []
    input_files: list[dict[str, str]] = []
    task_count = 0
    feature_count = 0
    for repo_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.endswith("_task")):
        for task_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir() and _task_number(p.name) is not None):
            features: list[dict[str, Any]] = []
            for feature_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and _feature_number(p.name) is not None):
                n = _feature_number(feature_dir.name)
                assert n is not None
                md = feature_dir / "feature.md"
                patch = feature_dir / "feature.patch"
                if not md.is_file() or not patch.is_file():
                    continue
                md_sha = sha256_file(md)
                patch_sha = sha256_file(patch)
                patch_text = patch.read_text(encoding="utf-8", errors="replace")
                touched = touched_files_from_patch(patch_text)
                features.append({
                    "feature": n,
                    "feature_md_sha256": md_sha,
                    "patch_sha256": patch_sha,
                    "touched_files": touched,
                })
                for p, kind in ((md, "feature.md"), (patch, "feature.patch")):
                    input_files.append({
                        "path": str(p.relative_to(dataset_root)),
                        "kind": kind,
                        "sha256": sha256_file(p),
                    })
            if len(features) < 2:
                continue
            task_count += 1
            feature_count += len(features)
            task_id = _task_number(task_dir.name)
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    a, b = features[i], features[j]
                    shared = sorted(set(a["touched_files"]) & set(b["touched_files"]))
                    record_id = f"{repo_dir.name}/task{task_id}/feature{a['feature']}+feature{b['feature']}"
                    records.append({
                        "record_id": record_id,
                        "repo_dir": repo_dir.name,
                        "task_id": str(task_id),
                        "feature_a": a,
                        "feature_b": b,
                        "shared_touched_files": shared,
                    })
    if len(records) < PHASE6A_SCENARIOS:
        raise Phase6AError(
            f"CooperBench snapshot yielded only {len(records)} feature pairs; "
            f"Phase6A requires at least {PHASE6A_SCENARIOS}"
        )
    input_files.sort(key=lambda r: r["path"])
    relevant_tree_sha = sha256_json(input_files)
    manifest = {
        "dataset_repo": PHASE6A_DATASET_REPO,
        "dataset_revision": PHASE6A_DATASET_REVISION,
        "task_directories_with_2plus_features": task_count,
        "feature_records": feature_count,
        "pair_records": len(records),
        "relevant_input_files": len(input_files),
        "relevant_tree_sha256": relevant_tree_sha,
        "input_files": input_files,
    }
    return records, manifest


def selection_key(record_id: str, seed: int = PHASE6A_SEED) -> str:
    return sha256_bytes(f"{seed}|{record_id}".encode("utf-8"))


def select_records(
    records: Iterable[dict[str, Any]],
    *,
    sample_size: int = PHASE6A_SCENARIOS,
    seed: int = PHASE6A_SEED,
) -> list[dict[str, Any]]:
    rows = list(records)
    if len(rows) < sample_size:
        raise Phase6AError(f"requested {sample_size} records from universe of {len(rows)}")
    return sorted(rows, key=lambda r: (selection_key(str(r["record_id"]), seed), str(r["record_id"])))[:sample_size]


def make_stale_scenario(record: dict[str, Any], index: int) -> dict[str, Any]:
    rid = str(record["record_id"])
    workspace = f"workspace:{record['repo_dir']}:task{record['task_id']}"
    patch_target = f"patch:{rid}:A"
    commitment = f"commitment:{rid}:B"
    return {
        "schema_version": "phase6a-write-side-v1",
        "scenario_id": f"phase6a-{index:03d}-{sha256_bytes(rid.encode())[:10]}",
        "source_record_id": rid,
        "repo_dir": record["repo_dir"],
        "task_id": record["task_id"],
        "feature_a": int(record["feature_a"]["feature"]),
        "feature_b": int(record["feature_b"]["feature"]),
        "feature_a_patch_sha256": record["feature_a"]["patch_sha256"],
        "feature_b_patch_sha256": record["feature_b"]["patch_sha256"],
        "feature_a_touched_files": list(record["feature_a"]["touched_files"]),
        "feature_b_touched_files": list(record["feature_b"]["touched_files"]),
        "shared_touched_files": list(record["shared_touched_files"]),
        "workspace_scope": workspace,
        "patch_target": patch_target,
        "peer_commitment": commitment,
        "base_versions": {patch_target: 1, commitment: 1},
        "peer_revision": {"object_id": commitment, "from_version": 1, "to_version": 2},
        "patch_update": {
            "update_id": f"{rid}:patch-A",
            "actor": "agent_a",
            "target_id": patch_target,
            "target_version_seen": 1,
            "reads_execution": [{"object_id": commitment, "version": 1}],
            "reads_projection": [{"object_id": commitment, "version": 1}],
            "reads_model_certificate": [{"object_id": commitment, "version": 1}],
            "declared_operation": "replace",
            "actual_operation": "replace",
            "actor_authority": 1,
            "required_authority": 1,
            "target_unique": True,
            "view_dependencies_execution": [],
            "view_dependencies_projection": [],
            "view_dependencies_model_certificate": [],
            "workspace_scope": workspace,
            "issue_label": "stale_peer_commitment",
        },
        "peer_update": {
            "update_id": f"{rid}:commitment-B",
            "actor": "agent_b",
            "target_id": commitment,
            "target_version_seen": 1,
            "reads_execution": [],
            "reads_projection": [],
            "reads_model_certificate": [],
            "declared_operation": "replace",
            "actual_operation": "replace",
            "actor_authority": 2,
            "required_authority": 1,
            "target_unique": True,
            "view_dependencies_execution": [],
            "view_dependencies_projection": [],
            "view_dependencies_model_certificate": [],
            "workspace_scope": workspace,
            "issue_label": None,
        },
    }


def build_stale_scenarios(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [make_stale_scenario(r, i + 1) for i, r in enumerate(records)]


def _current(read: dict[str, Any], versions: dict[str, int]) -> bool:
    return versions.get(str(read["object_id"])) == int(read["version"])


def _all_current(reads: list[dict[str, Any]], versions: dict[str, int]) -> bool:
    return all(_current(r, versions) for r in reads)


def _views_current(views: list[dict[str, Any]], versions: dict[str, int]) -> bool:
    for view in views:
        for source in view.get("sources", []):
            if not _current(source, versions):
                return False
    return True


def _wccu_decision(update: dict[str, Any], versions: dict[str, int], evidence: str, *, validate_reads: bool = True) -> dict[str, Any]:
    reads = list(update.get(f"reads_{evidence}", []))
    views = list(update.get(f"view_dependencies_{evidence}", []))
    failures: list[str] = []
    if not bool(update.get("target_unique", False)):
        failures.append("O_TARGET")
    if str(update.get("declared_operation")) != str(update.get("actual_operation")):
        failures.append("O_OP")
    if int(update.get("actor_authority", 0)) < int(update.get("required_authority", 0)):
        failures.append("O_AUTH")
    if validate_reads:
        if evidence != "model_certificate" and bool(update.get("requires_runtime_witness", False)) and not reads:
            failures.append("O_READ")
        if not _all_current(reads, versions):
            failures.append("O_FRESH")
    if not _views_current(views, versions):
        failures.append("O_VIEW")
    if failures:
        revalidate = {"O_READ", "O_FRESH", "O_VIEW"}
        decision = "REVALIDATE" if any(x in revalidate for x in failures) else "REVIEW"
    else:
        decision = "AUTO_COMMIT"
    return {"decision": decision, "failed_obligations": failures}


def evaluate_policy(policy: str, update: dict[str, Any], versions: dict[str, int]) -> dict[str, Any]:
    if policy == "execution_witness_wccu":
        return _wccu_decision(update, versions, "execution")
    if policy == "projection_trace_wccu":
        return _wccu_decision(update, versions, "projection")
    if policy == "model_certificate_wccu":
        return _wccu_decision(update, versions, "model_certificate")
    if policy == "wccu_without_read_validation":
        return _wccu_decision(update, versions, "execution", validate_reads=False)
    if policy == "read_set_occ":
        reads = list(update.get("reads_execution", []))
        ok = _all_current(reads, versions)
        return {"decision": "AUTO_COMMIT" if ok else "REVALIDATE", "failed_obligations": [] if ok else ["READ_SET_STALE"]}
    if policy == "adaptive_no_wccu":
        failures: list[str] = []
        if not bool(update.get("target_unique", False)):
            failures.append("TARGET_SELECTOR")
        if str(update.get("declared_operation")) != str(update.get("actual_operation")):
            failures.append("TYPE_SELECTOR")
        if int(update.get("actor_authority", 0)) < int(update.get("required_authority", 0)):
            failures.append("AUTHORITY_SELECTOR")
        return {"decision": "REVIEW" if failures else "AUTO_COMMIT", "failed_obligations": failures}
    if policy == "snapshot_occ":
        target = str(update["target_id"])
        seen = int(update.get("target_version_seen", -1))
        ok = versions.get(target) == seen
        return {"decision": "AUTO_COMMIT" if ok else "REVALIDATE", "failed_obligations": [] if ok else ["TARGET_VERSION_STALE"]}
    if policy == "uniform_review":
        return {"decision": "REVIEW", "failed_obligations": ["UNIFORM_REVIEW"]}
    if policy == "append_only":
        return {"decision": "AUTO_COMMIT", "failed_obligations": []}
    raise Phase6AError(f"unknown policy {policy!r}")


def evaluate_stale_replay(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    per_policy: dict[str, Any] = {}
    for policy in PHASE6A_POLICIES:
        rows: list[dict[str, Any]] = []
        stale_accepts = 0
        unsafe_auto = 0
        non_auto = 0
        safe_control_auto = 0
        peer_auto = 0
        for scenario in scenarios:
            base = {str(k): int(v) for k, v in scenario["base_versions"].items()}
            peer = scenario["peer_update"]
            patch = scenario["patch_update"]

            peer_eval = evaluate_policy(policy, peer, base)
            peer_is_auto = peer_eval["decision"] == "AUTO_COMMIT"
            peer_auto += int(peer_is_auto)
            non_auto += int(not peer_is_auto)

            mutated = dict(base)
            rev = scenario["peer_revision"]
            mutated[str(rev["object_id"])] = int(rev["to_version"])
            stale_eval = evaluate_policy(policy, patch, mutated)
            stale_is_auto = stale_eval["decision"] == "AUTO_COMMIT"
            stale_accepts += int(stale_is_auto)
            unsafe_auto += int(stale_is_auto)
            non_auto += int(not stale_is_auto)

            safe_eval = evaluate_policy(policy, patch, base)
            safe_control_auto += int(safe_eval["decision"] == "AUTO_COMMIT")
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "source_record_id": scenario["source_record_id"],
                "peer_update": peer_eval,
                "stale_patch_update": stale_eval,
                "safe_control_patch_update": safe_eval,
            })
        n = len(scenarios)
        per_policy[policy] = {
            "scenarios": n,
            "stale_dependent_patch_updates": n,
            "generated_updates_in_stale_trial": 2 * n,
            "stale_dependent_direct_accepts": stale_accepts,
            "unsafe_auto_commits": unsafe_auto,
            "non_auto_handling": non_auto,
            "peer_update_auto_commits": peer_auto,
            "safe_control_auto_commits": safe_control_auto,
            "safe_control_auto_commit_rate": safe_control_auto / max(1, n),
            "rows": rows,
        }
    return {
        "phase": "6A",
        "status": "PHASE6A_STALE_REPLAY_COMPLETE",
        "fresh": False,
        "evidence_class": "POST_OPEN_FIXED_WRITE_SIDE_REPLICATION_WITH_PINNED_PUBLIC_COOPERBENCH_AND_DETERMINISTIC_PROPOSALS",
        "policies": per_policy,
        "denominators": {
            "stale_scenarios": len(scenarios),
            "stale_dependent_patch_updates": len(scenarios),
            "generated_updates_in_stale_trial_per_policy": 2 * len(scenarios),
            "safe_control_patch_updates_per_policy": len(scenarios),
        },
    }


def _base_diagnostic_update(case_id: str) -> tuple[dict[str, Any], dict[str, int]]:
    target = f"diag-target:{case_id}"
    dep = f"diag-dep:{case_id}"
    view_source = f"diag-view-source:{case_id}"
    versions = {target: 1, dep: 1, view_source: 1}
    update = {
        "update_id": case_id,
        "actor": "agent_a",
        "target_id": target,
        "target_version_seen": 1,
        "reads_execution": [{"object_id": dep, "version": 1}],
        "reads_projection": [{"object_id": dep, "version": 1}],
        "reads_model_certificate": [{"object_id": dep, "version": 1}],
        "declared_operation": "replace",
        "actual_operation": "replace",
        "actor_authority": 1,
        "required_authority": 1,
        "target_unique": True,
        "view_dependencies_execution": [],
        "view_dependencies_projection": [],
        "view_dependencies_model_certificate": [],
        "requires_runtime_witness": False,
        "issue_label": None,
    }
    return update, versions


def make_diagnostic_case(family: str, index: int) -> dict[str, Any]:
    case_id = f"phase6a-diag-{family}-{index:03d}"
    update, versions = _base_diagnostic_update(case_id)
    issue = family != "safe"
    if family == "freshness":
        versions[f"diag-dep:{case_id}"] = 2
        update["issue_label"] = "freshness"
    elif family == "commitment":
        dep = f"peer-commitment:{case_id}"
        versions = {update["target_id"]: 1, dep: 2, f"diag-view-source:{case_id}": 1}
        read = {"object_id": dep, "version": 1}
        update["reads_execution"] = [read]
        update["reads_projection"] = [read]
        update["reads_model_certificate"] = [read]
        update["issue_label"] = "commitment"
    elif family == "authority":
        update["actor_authority"] = 0
        update["required_authority"] = 2
        update["issue_label"] = "authority"
    elif family == "operation":
        update["declared_operation"] = "append"
        update["actual_operation"] = "replace"
        update["issue_label"] = "operation"
    elif family == "materialized_view":
        src = f"diag-view-source:{case_id}"
        versions[src] = 2
        view = {"view_id": f"view:{case_id}", "sources": [{"object_id": src, "version": 1}]}
        update["view_dependencies_execution"] = [view]
        update["view_dependencies_projection"] = [view]
        # The model certificate intentionally lacks runtime view provenance; this is the mechanism under test.
        update["view_dependencies_model_certificate"] = []
        update["issue_label"] = "materialized_view"
    elif family == "witness_gap":
        hidden = f"hidden-dependency:{case_id}"
        versions[hidden] = 2
        hidden_read = {"object_id": hidden, "version": 1}
        update["reads_execution"] = [hidden_read]
        update["reads_projection"] = [hidden_read]
        update["reads_model_certificate"] = []
        update["requires_runtime_witness"] = True
        update["issue_label"] = "witness_gap"
    elif family == "safe":
        update["issue_label"] = None
    else:
        raise Phase6AError(f"unknown diagnostic family {family}")
    return {
        "case_id": case_id,
        "family": family,
        "is_issue": issue,
        "versions": versions,
        "update": update,
    }


def build_diagnostic_cases(
    *,
    per_issue_family: int = PHASE6A_DIAGNOSTIC_PER_ISSUE_FAMILY,
    safe_cases: int = PHASE6A_DIAGNOSTIC_SAFE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in PHASE6A_ISSUE_FAMILIES:
        rows.extend(make_diagnostic_case(family, i + 1) for i in range(per_issue_family))
    rows.extend(make_diagnostic_case("safe", i + 1) for i in range(safe_cases))
    return rows


def evaluate_obligation_diagnostic(cases: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for c in cases:
        family_counts[c["family"]] = family_counts.get(c["family"], 0) + 1
    policies: dict[str, Any] = {}
    for policy in PHASE6A_POLICIES:
        issue_accepts = 0
        safe_auto = 0
        per_family: dict[str, dict[str, int]] = {}
        rows: list[dict[str, Any]] = []
        for case in cases:
            ev = evaluate_policy(policy, case["update"], case["versions"])
            auto = ev["decision"] == "AUTO_COMMIT"
            fam = str(case["family"])
            bucket = per_family.setdefault(fam, {"cases": 0, "auto_commits": 0, "issue_accepts": 0, "non_auto": 0})
            bucket["cases"] += 1
            bucket["auto_commits"] += int(auto)
            bucket["non_auto"] += int(not auto)
            if case["is_issue"] and auto:
                issue_accepts += 1
                bucket["issue_accepts"] += 1
            if not case["is_issue"] and auto:
                safe_auto += 1
            rows.append({"case_id": case["case_id"], "family": fam, "is_issue": case["is_issue"], **ev})
        issue_total = sum(1 for c in cases if c["is_issue"])
        safe_total = sum(1 for c in cases if not c["is_issue"])
        policies[policy] = {
            "issue_cases": issue_total,
            "safe_cases": safe_total,
            "issue_accepts": issue_accepts,
            "unsafe_rate": issue_accepts / max(1, issue_total),
            "safe_auto_commits": safe_auto,
            "safe_auto_commit_rate": safe_auto / max(1, safe_total),
            "per_family": per_family,
            "rows": rows,
        }
    return {
        "phase": "6A",
        "status": "PHASE6A_OBLIGATION_DIAGNOSTIC_COMPLETE",
        "fresh": False,
        "evidence_class": "POST_OPEN_AUTHOR_CONSTRUCTED_DETERMINISTIC_OBLIGATION_MECHANISM_DIAGNOSTIC",
        "family_counts": family_counts,
        "policies": policies,
        "interpretation_boundary": "Mechanism diagnostic only; family frequencies are constructed and are not prevalence estimates.",
    }
