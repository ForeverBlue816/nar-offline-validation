import unittest
import numpy as np
import mpmath as mp
from nar.e34_report import ratio_summary,paired_ppl_combo,T90_DF63

class E34Statistics(unittest.TestCase):
    def test_chunk_t_quantile(self):
        df=63;x=T90_DF63
        cdf=1-mp.betainc(df/2,.5,0,df/(df+x*x),regularized=True)/2
        self.assertAlmostEqual(float(cdf),.95,places=11)
    def test_zero_count_chunks_and_constant_class_effect(self):
        counts=np.arange(64)%4
        result=ratio_summary(counts*2.,counts)
        for name in ['estimate','ci90_low','ci90_high']:self.assertAlmostEqual(result[name],2.)
        empty=ratio_summary(np.zeros(64),np.zeros(64));self.assertIsNone(empty['estimate'])
    def test_paired_ppl_cancellation_preserves_covariance(self):
        base=1+np.arange(192).reshape(3,64)/200
        candidate=base+np.sin(base)/100
        result=paired_ppl_combo([(1,candidate,base),(-1,candidate,base)])
        for value in result.values():self.assertEqual(value,0)
    def test_signed_share_is_not_clipped(self):
        result=ratio_summary(np.ones(64)*2,np.ones(64)*-1)
        self.assertEqual(result['estimate'],-2)
if __name__=='__main__':unittest.main()
