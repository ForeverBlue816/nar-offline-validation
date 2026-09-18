"""Reconstruct reviewer-facing diagnostics from immutable new factor assets."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a


def assignment_compare(left,right,targets):
    targets=set(targets)
    aa={(int(s),int(t)) for s,t in zip(left['source'],left['target']) if int(t) in targets}
    bb={(int(s),int(t)) for s,t in zip(right['source'],right['target']) if int(t) in targets}
    if not aa and not bb:return '', ''
    return len(aa&bb)/len(aa|bb),1-len(aa&bb)/len(aa)


def calibration_shift():
    m='qwen3_4b_base';out=[]
    for seed in a.SEEDS:
        for s,l,n in a.keys(m):
            aa=a.loadt(a.factor_path(m,f'wt2_n128_s{seed}','S3','max','P1',0,s,l))
            bb=a.loadt(a.factor_path(m,f'c4_n128_s{seed}','S3','max','P1',0,s,l))
            k=aa['k'];anchors=set(range(0,k*128,128));fillers=set(range(k*128,n,128));residual=set(range(n))-anchors-fillers
            r=dict(model=m,seed=seed,site=s,layer=l,k=k,f_wt2=aa['calibration_fraction'],f_c4=bb['calibration_fraction'],f_delta=bb['calibration_fraction']-aa['calibration_fraction'])
            for name,slots in [('all',range(n)),('anchor',anchors),('filler',fillers),('residual',residual)]:
                j,c=assignment_compare(aa,bb,slots);r[name+'_jaccard']=j;r[name+'_changed_fraction']=c;r[name+'_slots']=len(slots)
            out.append(r)
    a.write(a.REPO/'results'/m/'e30_slot_assignments.csv',out)
    means=[]
    for s in ['qkv','down']:
        rr=[r for r in out if r['site']==s]
        r=dict(model=m,site=s,layer_seed_pairs=len(rr),f_wt2_mean=np.mean([r['f_wt2'] for r in rr]),f_c4_mean=np.mean([r['f_c4'] for r in rr]),f_absolute_difference_mean=np.mean([abs(r['f_delta']) for r in rr]),f_absolute_difference_max=max(abs(r['f_delta']) for r in rr),f_difference_over_003=sum(abs(r['f_delta'])>=.03 for r in rr))
        for name in ['all','anchor','filler','residual']:
            for suffix in ['jaccard','changed_fraction']:
                vals=[x[name+'_'+suffix] for x in rr if x[name+'_'+suffix]!=''];r[name+'_'+suffix+'_mean']=np.mean(vals) if vals else ''
        means.append(r)
    a.write(a.REPO/'results'/m/'e30_slot_summary.csv',means)


def spectral_diagnostics(m):
    rows=[];summaries=[]
    for s,l,n in a.keys(m):
        p=a.loadt(a.ASSETS/m/'spectral/wt2_n128_s0'/f'{s}_{l:02d}.pt')
        for solver in (['S1','S2','S3','S4'] if m=='qwen3_4b_base' else ['S3','S4']):
            d=p['solvers'][solver];k=n//128
            assert len(d['ritz_residuals'])==len(d['principal_angles_deg'])==k
            summaries.append(dict(model=m,solver=solver,site=s,layer=l,k=k,captured_fraction=d['captured_fraction'],ritz_residual_median=float(d['ritz_residuals'].median()),ritz_residual_max=float(d['ritz_residuals'].max()),principal_angle_max_deg=float(d['principal_angles_deg'].max())))
            for i in range(k):
                rows.append(dict(model=m,solver=solver,site=s,layer=l,k=k,index=i+1,ritz_residual=float(d['ritz_residuals'][i]),eigenvalue=float(d['eigenvalues'][i]),principal_angle_deg=float(d['principal_angles_deg'][i]),angle_order='ascending principal-angle spectrum, not paired with individual Ritz vector'))
    a.write(a.REPO/'results'/m/'e32_ritz_and_angles.csv',rows)
    # Use ordinary statistical median, including the average middle pair.
    for r in summaries:
        vals=[z['ritz_residual'] for z in rows if all(z[f]==r[f] for f in ['solver','site','layer'])];r['ritz_residual_median']=float(np.median(vals))
    a.write(a.REPO/'results'/m/'e32_spectral_sites.csv',summaries)
    grouped=[]
    for solver in sorted({r['solver'] for r in summaries}):
        rr=[r for r in summaries if r['solver']==solver]
        grouped.append(dict(model=m,solver=solver,ritz_residual_median_over_sites=float(np.median([r['ritz_residual_median'] for r in rr])),ritz_residual_max=max(r['ritz_residual_max'] for r in rr),principal_angle_max_deg=max(r['principal_angle_max_deg'] for r in rr),f_down_mean=float(np.mean([r['captured_fraction'] for r in rr if r['site']=='down'])),f_qkv_mean=float(np.mean([r['captured_fraction'] for r in rr if r['site']=='qkv']))))
    a.write(a.REPO/'results'/m/'e32_spectral_summary.csv',grouped)


def factor_contract(m):
    """Assert E29 quantizer pairs and all E31 permutations use precisely one G."""
    out=[]
    for s,l,n in a.keys(m):
        ranks=['max','8'] if m=='qwen3_4b_base' else ['max']
        for rank in ranks:
            default=a.loadt(a.factor_path(m,'wt2_n128_s0','S3',rank,'P1',0,s,l))
            for variant in (['P1','P2','P3','P4','P5'] if m=='qwen3_4b_base' else ['P1']):
                for seed in (a.SEEDS if variant=='P3' else [0]):
                    other=a.loadt(a.factor_path(m,'wt2_n128_s0','S3',rank,variant,seed,s,l))
                    same=all(torch.equal(default[f],other[f]) for f in ['w','y','vectors']);assert same
                    same_order=all(torch.equal(default[f],other[f]) for f in ['source','target'])
                    if variant=='P4' and rank=='max':assert same_order
                    out.append(dict(model=m,site=s,layer=l,rank=rank,variant=variant,permutation_seed=seed,shared_G_bit_identical=same,permutation_identical_to_P1=same_order))
    a.write(a.REPO/'results'/m/'e31_shared_G_audit.csv',out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=int,required=True,choices=[30,31,32]);p.add_argument('--model',default='qwen3_4b_base');arg=p.parse_args()
    if arg.experiment==30:calibration_shift()
    elif arg.experiment==31:factor_contract(arg.model)
    else:spectral_diagnostics(arg.model)
if __name__=='__main__':main()
