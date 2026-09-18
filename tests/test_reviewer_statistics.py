import unittest
import numpy as np
import mpmath as mp
from nar.run_reviewer_experiments import stats,T90_DF191,T90_DF2,plan
class PairingTests(unittest.TestCase):
    def test_null_paired_effect(self):
        x=np.arange(192).reshape(3,64)/100+1
        s=stats(x,x)
        for field in ['delta','ci90_low','ci90_high','seed_ci90_low','seed_ci90_high']:self.assertEqual(s[field],0)
    def test_t_constants(self):
        for df,x in [(191,T90_DF191),(2,T90_DF2)]:
            cdf=1-mp.betainc(df/2,.5,0,df/(df+x*x),regularized=True)/2
            self.assertAlmostEqual(float(cdf),.95,places=12)
    def test_center_is_mean_corpus_ppl_difference(self):
        x=np.arange(192).reshape(3,64)/1000+2
        y=x+np.array([.01,.02,.03])[:,None]
        s=stats(y,x)
        self.assertAlmostEqual(s['delta'],float((np.exp(y.mean(1))-np.exp(x.mean(1))).mean()),places=12)
        self.assertLess(s['ci90_low'],s['delta']);self.assertGreater(s['ci90_high'],s['delta'])
    def test_fixed_rows_and_paired_quantizers(self):
        p=plan(29,'qwen3_4b_base');self.assertEqual(len(p),8)
        for aa,bb in [(p[1],p[3]),(p[5],p[7])]:
            for k in ['label','solver','rank','variant']:self.assertEqual(aa[k],bb[k])
        self.assertEqual(len(plan(31,'qwen3_4b_base')),11)
if __name__=='__main__':unittest.main()
