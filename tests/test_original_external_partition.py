import unittest
from mosaic_wccu.external_partition import ExternalPartitionError, validate_external_tiles

class ExternalPartitionTests(unittest.TestCase):
    def test_unique_quotes_validate(self):
        text='Launch is Friday. Owner is Alice.'
        rows=[
            {'tile_id':'launch','quotes':['Launch is Friday.'],'dependencies':[]},
            {'tile_id':'owner','quotes':['Owner is Alice.'],'dependencies':['launch']},
        ]
        tiles=validate_external_tiles(text,rows)
        self.assertEqual(len(tiles),2)

    def test_ambiguous_quote_fails_closed(self):
        with self.assertRaises(ExternalPartitionError):
            validate_external_tiles('Alice. Reviewer Alice.', [{'tile_id':'x','quotes':['Alice']}])

if __name__=='__main__': unittest.main()
