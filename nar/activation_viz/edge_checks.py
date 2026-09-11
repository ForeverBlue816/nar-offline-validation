"""Small numerical integrity checks; these tensors are never used for figures."""
import argparse,json
from pathlib import Path
import torch
from .metrics import measure,pool

def run():
    cases={}
    for name,y in [('zero',torch.zeros(3,256)),('constant',torch.full((3,256),1.25)),
                   ('signed',torch.linspace(-2,3,3*256).reshape(3,256)),
                   ('underflow',torch.linspace(0,1e-8,3*256).reshape(3,256))]:
        r,_,_,_=measure(y);cases[name]=r
    assert cases['zero']['nmse'] is None
    assert cases['constant']['nmse']==0 and cases['constant']['f']==0
    assert cases['underflow']['nmse']==1 and cases['underflow']['nonfinite']==0
    x=torch.zeros(2048,4096);x[0,4000]=9;x[-1,-1]=-17
    z,t,c=pool(x)
    assert z[0].max()==9 and z[-1,-1]==17
    assert t[0]==0 and t[1]==1 and t[-1]==2048 and c[-1]==4096
    return {'passed':True,'data_role':'synthetic edge checks only; no figure data','cases':cases,
            'max_pooling_preserves_first_and_last_peaks':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
