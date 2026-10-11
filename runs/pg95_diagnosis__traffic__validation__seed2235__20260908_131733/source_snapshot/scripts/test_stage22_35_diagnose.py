import json
import unittest
import numpy as np
import torch
from stage22_35_diagnose import ROOT,parsed_namespace,geometry,summarize_group


class Tests(unittest.TestCase):
    def test_namespace_is_literal_not_execution(self):
        args=parsed_namespace("Namespace(seed=2027, use_mipp=True, device=device(type='cuda', index=0))")
        self.assertEqual(args.seed,2027); self.assertEqual(str(args.device),'cpu')
        with self.assertRaises(ValueError): parsed_namespace("Namespace(seed=arbitrary_code())")

    def test_weighted_error_sign(self):
        a={'complete_count':np.array([1.,3.]),'complete_sse_on':np.array([1.,3.]),'complete_sae_on':np.array([1.,3.])}
        b={**a,'complete_sse_on':np.array([2.,6.])}
        out=summarize_group(a,b,'on',np.array([True,True]))
        self.assertEqual(out['mse_change_percent'],100.)
        self.assertEqual(out['mae_change_percent'],0.)
        self.assertEqual(summarize_group(a,b,'on',np.array([False,False]))['windows'],0)

    def test_geometry_constant_not_diverse(self):
        out=geometry(torch.ones(5,8))
        self.assertEqual(out['centered_effective_rank'],0.)
        self.assertAlmostEqual(out['pairwise_cosine_mean'],1.)
        self.assertIsNone(geometry(torch.ones(1,8))['pairwise_cosine_mean'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (ROOT/'reports/stage22_35_test.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
