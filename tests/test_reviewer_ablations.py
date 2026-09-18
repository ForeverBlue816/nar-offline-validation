import unittest
import numpy as np
import torch
from nar import reviewer_ablations as a

class NumericalContract(unittest.TestCase):
    def test_wy_maps_selected_space_and_is_orthogonal(self):
        gen=torch.Generator().manual_seed(7)
        vectors=torch.randn((512,4),generator=gen,dtype=torch.float64)
        v,w,y=a.wy64(vectors)
        mapped=v.T-(v.T@w)@y.T
        target=torch.zeros_like(mapped);target[torch.arange(4),torch.arange(4)*128]=1
        self.assertLess(float((mapped-target).norm()),1e-12)
        x=torch.randn((13,512),generator=gen)
        wf,yf=w.float(),y.float();z=x-(x@wf)@yf.T
        back=z-(z@yf)@wf.T
        self.assertLess(float((back-x).norm()/x.norm()),1e-6)
    def test_permutations_keep_anchors_and_are_bijections(self):
        e=torch.arange(512,dtype=torch.float64).flip(0)
        for k in (1,4):
            for variant in ('P1','P2','P3','P4','P5'):
                src,dst=a.orders(e,512,k,variant,11)
                self.assertEqual(sorted(src.tolist()),list(range(512)))
                self.assertEqual(sorted(dst.tolist()),list(range(512)))
                mapping=dict(zip(src.tolist(),dst.tolist()))
                for i in range(k):self.assertEqual(mapping[i*128],i*128)
        for pair in zip(a.orders(e,512,4,'P1',0),a.orders(e,512,4,'P4',0)):
            self.assertTrue(torch.equal(*pair))
    def test_full_width_paley_dc_and_actual_transpose(self):
        for n in [2560,9728,3072,8192]:
            unit=torch.zeros((1,n));unit[0,0]=1
            mapped=a.dc_hadamard_rows(unit)
            self.assertLess(float((mapped-torch.ones_like(mapped)/n**.5).norm()),1e-6)
            x=torch.randn((3,n),generator=torch.Generator().manual_seed(113))
            back=a.dc_hadamard_rows(a.dc_hadamard_rows(x),True)
            self.assertLess(float((back-x).norm()/x.norm()),1e-6)

    def test_quantizer_offsets_and_degenerate_groups(self):
        x=torch.tensor([[2.,3.,4.,5.],[0.,0.,0.,0.]])
        sym=a.quantize(x,4,True);asym=a.quantize(x,4,False)
        self.assertTrue(torch.isfinite(sym).all() and torch.isfinite(asym).all())
        self.assertTrue(torch.equal(sym[1],x[1]))
        self.assertLess(float((asym[0]-x[0]).abs().max()),.01)
        self.assertGreater(float((sym[0]-x[0]).abs().max()),.05)

if __name__=='__main__':unittest.main()
