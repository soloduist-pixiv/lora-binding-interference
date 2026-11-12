"""CPU checks distinguish mass loss from changed old-answer rankings."""
import unittest
import numpy as np
from source_oracle import diagnose

class SourceOracle(unittest.TestCase):
    def fixture(self,shift):
        active=np.arange(1,33,2);keys=np.repeat(active,2)
        old=np.arange(100,196,2);new=np.roll(old+1,-shift)
        p=np.zeros((len(keys),96));ids=np.arange(len(keys))
        p[ids,(keys-1+shift)%48]=.1;p[ids,48+keys-1]=.9
        return p,keys,dict(old=old.tolist(),new=new.tolist(),active_original=active.tolist())

    def test_lower_old_mass_alone_preserves_the_old_ranking(self):
        p,keys,protocol=self.fixture(0);m=diagnose(p,keys,protocol,123)
        self.assertAlmostEqual(m["raw_correct"],.1)
        self.assertEqual(m["readouts"]["all48"]["oracle_top1"]["observed"],1)
        self.assertEqual(m["readouts"]["active16"]["oracle_top1"]["observed"],1)

    def test_new_prefix_association_can_survive_without_correct_old_binding(self):
        p,keys,protocol=self.fixture(24);m=diagnose(p,keys,protocol,123)
        self.assertEqual(m["raw_correct"],0)
        self.assertEqual(m["readouts"]["all48"]["oracle_top1"]["observed"],0)
        self.assertEqual(m["maple_prefix_match"]["observed"],1)
        self.assertLess(m["maple_prefix_match"]["exact_null_mean"],1)

if __name__=="__main__":unittest.main()
