"""Small numerical integrity checks; these tensors are never used for figures."""
import argparse,json
from pathlib import Path
import torch
from .metrics import measure

def run():
    cases={}
    for name,y in [('zero',torch.zeros(3,256)),('constant',torch.full((3,256),1.25)),
                   ('signed',torch.linspace(-2,3,3*256).reshape(3,256)),
                   ('underflow',torch.linspace(0,1e-8,3*256).reshape(3,256))]:
        r,_,_,_=measure(y);cases[name]=r
    assert cases['zero']['nmse'] is None
    assert cases['constant']['nmse']==0 and cases['constant']['f']==0
    assert cases['underflow']['nmse']==1 and cases['underflow']['nonfinite']==0
    return {'passed':True,'data_role':'synthetic edge checks only; no figure data','cases':cases}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
