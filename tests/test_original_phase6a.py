from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mosaic_wccu.phase6a import (
    PHASE6A_ISSUE_FAMILIES,
    PHASE6A_POLICIES,
    build_diagnostic_cases,
    build_stale_scenarios,
    collect_pair_records,
    evaluate_obligation_diagnostic,
    evaluate_stale_replay,
    select_records,
    touched_files_from_patch,
)


class Phase6ATests(unittest.TestCase):
    def test_patch_parser(self):
        text = "diff --git a/a.py b/a.py\n@@\n+1\ndiff --git a/b/c.ts b/b/c.ts\n"
        self.assertEqual(touched_files_from_patch(text), ["a.py", "b/c.ts"])

    def test_collect_and_select_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / "repo_task" / "task1"
            for i in range(1, 19):
                d = task / f"feature{i}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "feature.md").write_text(f"feature {i}\n")
                (d / "feature.patch").write_text(f"diff --git a/f{i}.py b/f{i}.py\n")
            records, manifest = collect_pair_records(root)
            self.assertEqual(len(records), 153)
            self.assertEqual(manifest["pair_records"], 153)
            a = select_records(records, sample_size=150)
            b = select_records(list(reversed(records)), sample_size=150)
            self.assertEqual([x["record_id"] for x in a], [x["record_id"] for x in b])

    def test_stale_replay_has_explicit_denominators_and_no_tuned_outcomes(self):
        records = []
        for i in range(3):
            records.append({
                "record_id": f"r/task1/feature{i+1}+feature{i+2}",
                "repo_dir": "r_task",
                "task_id": "1",
                "feature_a": {"feature": i+1, "patch_sha256": "a"*64, "touched_files": ["a.py"]},
                "feature_b": {"feature": i+2, "patch_sha256": "b"*64, "touched_files": ["b.py"]},
                "shared_touched_files": [],
            })
        scenarios = build_stale_scenarios(records)
        out = evaluate_stale_replay(scenarios)
        self.assertEqual(out["denominators"]["generated_updates_in_stale_trial_per_policy"], 6)
        self.assertEqual(out["policies"]["execution_witness_wccu"]["stale_dependent_direct_accepts"], 0)
        self.assertEqual(out["policies"]["read_set_occ"]["stale_dependent_direct_accepts"], 0)
        self.assertEqual(out["policies"]["snapshot_occ"]["stale_dependent_direct_accepts"], 3)
        self.assertEqual(out["policies"]["uniform_review"]["non_auto_handling"], 6)
        self.assertEqual(out["policies"]["execution_witness_wccu"]["safe_control_auto_commit_rate"], 1.0)

    def test_obligation_mechanism_matrix(self):
        cases = build_diagnostic_cases(per_issue_family=2, safe_cases=4)
        out = evaluate_obligation_diagnostic(cases)
        self.assertEqual(set(out["family_counts"]), set(PHASE6A_ISSUE_FAMILIES) | {"safe"})
        self.assertEqual(set(out["policies"]), set(PHASE6A_POLICIES))
        exec_row = out["policies"]["execution_witness_wccu"]
        read_row = out["policies"]["read_set_occ"]
        cert_row = out["policies"]["model_certificate_wccu"]
        self.assertEqual(exec_row["issue_accepts"], 0)
        self.assertGreater(read_row["issue_accepts"], 0)
        self.assertGreater(cert_row["per_family"]["witness_gap"]["issue_accepts"], 0)
        self.assertEqual(exec_row["safe_auto_commits"], 4)


if __name__ == "__main__":
    unittest.main()
