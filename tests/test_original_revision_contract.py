import unittest
from dataclasses import replace
from wccu_eval.revisions import build_case,cpu,oracle
from wccu_eval.revisions import decode
from native_signed_receipt.engine import Snapshot,Plan,Transition,issue,advance
class Contract(unittest.TestCase):
 def test_omitted_dependency_changes_truth_without_selection(self):
  c=build_case('invoice',0,'dev');_,rows=cpu([c]);r=next(r for r in rows if r['change']=='hidden_policy')
  self.assertTrue(r['oracle_changed']);self.assertFalse(any(r['refresh'][a] for a in ['content_only','query_exact','query_postings','query_receipt']));self.assertTrue(r['refresh']['epoch'])
 def test_insertion_is_not_exposed_content_mutation(self):
  _,rows=cpu([build_case('allocation',1,'dev')]);r=next(r for r in rows if r['change']=='insertion')
  self.assertFalse(r['exposed_content_stale']);self.assertTrue(r['refresh']['query_receipt'])
 def test_query_change_does_not_imply_changed_truth(self):
  _,rows=cpu([build_case('refund',0,'dev')]);r=next(r for r in rows if r['change']=='query_config')
  self.assertTrue(r['selection_changed']);self.assertFalse(r['oracle_changed'])
 def test_incomplete_update_history_falls_back(self):
  c=build_case('routing',0,'dev');old=Snapshot(c['documents']);new=Snapshot(c['revisions']['insertion']['documents']);p=Plan(c['id'],3)
  d=advance(new,p,issue(old,p).receipt,Transition.from_snapshots(old,new,complete=False));self.assertEqual(d.reason,'provenance_or_plan_fallback');self.assertEqual(d.receipt.selected,issue(new,p).receipt.selected)
 def test_strict_proposal_preservation(self):
  for s in ['{"decision":"commit","value":true}','{"decision":"commit","value":1,"value":2}','```json\n{"decision":"commit","value":1}\n```','{"decision":"deny","value":3}','{"decision":"commit","value":1,"extra":2}']:self.assertIsNone(decode(s))
  self.assertEqual(decode('{"decision":"commit","value":13}'),{'decision':'commit','value':13})
 def test_structured_labels_match_public_policy_json(self):
  import json
  for kind in ['invoice','allocation','refund','routing']:
   c=build_case(kind,0,'dev')
   for cfg in c['revisions'].values():
    for d in cfg['documents']:
     if 'policy' in d:
      value=json.loads(d['text'].splitlines()[1]);value.pop('type');self.assertEqual(value,d['policy'])
if __name__=='__main__':unittest.main()
