import unittest
from mosaic_wccu.models import Tile, TileVersion, Update
from mosaic_wccu.conflict import classify_pair, freshness_decision
from mosaic_wccu.anchor import resolve_quote

class MosaicTests(unittest.TestCase):
    def test_same_block_independent_tiles_commute_when_ranges_do_not_overlap(self):
        a=Update('a',[],{'deadline'},[(0,20)])
        b=Update('b',[],{'owner'},[(25,40)])
        self.assertEqual(classify_pair(a,b),'COMMUTE')

    def test_disjoint_physical_ranges_can_have_semantic_conflict(self):
        a=Update('a',[],{'deadline'},[(0,20)])
        b=Update('b',[],{'reminder'},[(80,100)],semantic_dependencies={'deadline'})
        self.assertEqual(classify_pair(a,b),'REVALIDATE')

    def test_same_semantic_write_rebases(self):
        a=Update('a',[],{'deadline'},[(0,20)])
        b=Update('b',[],{'deadline'},[(0,20)])
        self.assertEqual(classify_pair(a,b),'REBASE')

    def test_stale_read_requires_revalidation(self):
        tiles={'deadline':Tile('deadline',2,[(0,20)])}
        u=Update('u',[TileVersion('deadline',1)],{'reminder'},[(40,50)])
        self.assertEqual(freshness_decision(u,tiles),'REVALIDATE')

    def test_ambiguous_anchor_fails_closed(self):
        self.assertEqual(resolve_quote('x deadline y deadline','deadline').status,'ambiguous')

    def test_unique_anchor_resolves(self):
        r=resolve_quote('alpha deadline friday','deadline')
        self.assertEqual(r.status,'resolved')
        self.assertEqual((r.start,r.end),(6,14))

if __name__=='__main__': unittest.main()
