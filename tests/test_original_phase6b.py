import unittest

from mosaic_wccu.benchmark_adapters import AdaptedDocument, GoldSemanticTile
from mosaic_wccu.phase6b import (
    adjacency_budget_control,
    attach_budget_metadata,
    footprint_context_bytes,
    ranked_budget_control,
)
from mosaic_wccu.segmentation import Segment


class Phase6BBudgetControlTests(unittest.TestCase):
    def doc(self):
        text = "aaaa\nbbbbbbbb\ncc\nddddd\neeeeeee\nffff\n"
        ranges = ((0,5),(5,14),(14,17),(17,23),(23,31),(31,36))
        return AdaptedDocument(
            "toy:q1", text,
            (GoldSemanticTile("g0","q",(Segment(0,5),Segment(23,31)),"alt0"),),
            {"dataset":"Toy","question":"q","paragraph_ranges":ranges,"paragraph_titles":tuple("" for _ in ranges)},
        )

    def rrf(self):
        # Frozen ranking intentionally jumps around source order.
        order = [0,4,1,3,2,5]
        ranges = [(0,5),(5,14),(14,17),(17,23),(23,31),(31,36)]
        return {"ranked_top":[
            {"kind":"physical_atom","tile_id":f"retrieval_{idx:05d}","ranges":[list(ranges[idx])]}
            for idx in order
        ]}

    def test_ranked_control_never_exceeds_feasible_budget_and_skips_oversize(self):
        doc=self.doc(); base=ranked_budget_control(doc,self.rrf(),target_context_bytes=20)
        self.assertTrue(base.budget_feasible)
        self.assertLessEqual(base.achieved_context_bytes,20)
        self.assertEqual(base.footprints[0].footprint_id,"retrieval_00000")
        self.assertEqual(base.footprints[1].footprint_id,"retrieval_00004")
        # At least one later candidate can be added under this toy cap.
        self.assertGreaterEqual(base.added_count,1)

    def test_infeasible_baseline_is_kept_intact_and_marked(self):
        doc=self.doc(); result=ranked_budget_control(doc,self.rrf(),target_context_bytes=3)
        self.assertFalse(result.budget_feasible)
        self.assertEqual(len(result.footprints),2)
        self.assertGreater(result.achieved_context_bytes,3)

    def test_adjacency_order_uses_distance_then_source_order(self):
        doc=self.doc(); result=adjacency_budget_control(doc,self.rrf(),target_context_bytes=1000)
        ids=[fp.footprint_id for fp in result.footprints]
        # Base atoms are source 0 and 4; nearest candidates are 1,3,5 before 2.
        self.assertEqual(ids[:2],["retrieval_00000","retrieval_00004"])
        self.assertEqual(ids[2:],["adjacent_physical_00001","adjacent_physical_00003","adjacent_physical_00005","adjacent_physical_00002"])
        self.assertEqual(result.achieved_context_bytes,footprint_context_bytes(doc,result.footprints))

    def test_budget_metadata_is_explicit(self):
        doc=self.doc(); control=ranked_budget_control(doc,self.rrf(),target_context_bytes=20)
        row=attach_budget_metadata({"method":"x"},control)
        meta=row["phase6b_budget_control"]
        self.assertEqual(meta["target_context_bytes"],20)
        self.assertFalse(meta["outcome_dependent_selection"])
        self.assertTrue(meta["whole_paragraphs_only"])


if __name__ == "__main__":
    unittest.main()
