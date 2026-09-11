"""Inspect unchanged factor precision without fitting or renormalization."""
import argparse,json
from pathlib import Path
import torch
from .capture import digest,write

def run(work,output):
    work=Path(work);records=[];torch.set_num_threads(2)
    root=work/'activations/qwen3_8b_base'
    paths=[root/'e14_rotations'/f'r1_{k}.pt' for k in ('k8','kmax')]
    paths += [root/'e18v2_factors'/m/f'down_layer_{l:02d}.pt' for m in ('nar_k8','nar_kmax') for l in (0,12,23,35)]
    for p in paths:
        f=torch.load(p,map_location='cpu',weights_only=True)
        norm=f['reflectors'][f['active'].bool()].double().square().sum(-1)
        n=f['n'];identity=torch.arange(n)
        valid=all(torch.equal(f[k].long().sort().values,identity) for k in ('source_order','target_order'))
        assert valid
        records.append({'path':str(p.relative_to(work)),'sha256':digest(p),'active_reflectors':int(f['active'].sum()),
                        'squared_norms_fp64':norm.tolist(),'max_squared_norm_deviation':float((norm-1).abs().max()),'permutation_bijective':valid})
    write(Path(output),{'records':records,'factors_modified':False,'interpretation':'Finite-precision stored reflectors are not exactly unit-norm; report observed deviations, do not renormalize frozen checkpoints.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workdir',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.workdir,a.output)
