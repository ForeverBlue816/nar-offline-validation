import unittest
import numpy as np
import torch
from nar.e34_anchor_attribution import column_pattern,swap_targets,flags_from_scores,Hooks,plan

class E34Contract(unittest.TestCase):
    def test_requested_walsh_patterns(self):
        low=column_pattern(64);high=column_pattern(1)
        self.assertTrue(torch.equal(low[:64],torch.full((64,),128**-.5)))
        self.assertTrue(torch.equal(low[64:],torch.full((64,),-128**-.5)))
        self.assertTrue(torch.equal(high[::2],torch.full((64,),128**-.5)))
        self.assertTrue(torch.equal(high[1::2],torch.full((64,),-128**-.5)))
        self.assertAlmostEqual(float(low@high),0,places=7)
    def test_minimal_swap_remains_in_same_group(self):
        target=torch.randperm(512,generator=torch.Generator().manual_seed(9))
        for col in [1,64]:
            new=swap_targets(target,col)
            self.assertTrue(torch.equal(new.sort().values,torch.arange(512)))
            self.assertTrue(torch.equal(new//128,target//128))
            self.assertEqual(int((new!=target).sum()),8)
            self.assertTrue(torch.equal(swap_targets(new,col),target))
            unaffected=(target%128!=0)&(target%128!=col)
            self.assertTrue(torch.equal(new[unaffected],target[unaffected]))
    def test_top_fraction_and_tie_break_are_fixed(self):
        x=np.ones((64,2048),np.float32);massive,bos=flags_from_scores(x)
        self.assertEqual(int(massive.sum()),132);self.assertEqual(int(bos.sum()),64)
        self.assertFalse(massive[:,0].any());self.assertTrue(massive[0,1:133].all())
        self.assertLess(float((massive|bos).mean()),.005)
    def test_unmasked_bf16_values_bypass_roundtrip(self):
        class PerturbedIdentity:
            def apply(self,s,l,v,transpose=False):return v.float()+(.01 if not transpose else -.01)
        original=torch.tensor([[[1.03125]*128,[2.015625]*128]],dtype=torch.bfloat16)
        h=Hooks(None,PerturbedIdentity(),True,torch.tensor([True,False]),False)
        result=h.transform('qkv',0,original)
        self.assertTrue(torch.equal(result[:,1],original[:,1]));self.assertTrue(h.bypass_exact)
    def test_baselines_replay_first_and_masked_symmetric_pairs_exist(self):
        rows=plan();self.assertEqual(len(rows),16)
        self.assertTrue(all(r['baseline'] for r in rows[:4]))
        for mask in ['M1','M2']:
            self.assertEqual(sum(r['mask']==mask for r in rows),4)
if __name__=='__main__':unittest.main()
