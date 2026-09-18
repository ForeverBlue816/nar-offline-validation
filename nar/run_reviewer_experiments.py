"""Fixed E29--E32 row matrix; no result-driven row selection or tuning."""
from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
import numpy as np
import torch
T90_DF191 = 1.6528705472303895
T90_DF2 = 2.919985580353725
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a


def plan(exp,m):
    result=[]
    def add(name,method='pq',**kw):
        r=dict(row=name,method=method,label='wt2_n128_s0',solver='S3',rank='max',variant='P1',group=128,symmetric=False,evalsets=['wt2'])
        r.update(kw);result.append(r)
    if exp==29:
        for q in range(1,5):
            for method in ['hadamard','pq']:
                add(f'Q{q}_{method}',method,rank='max' if q<3 else '1',group=128 if q<3 else 'token',symmetric=q in [2,4])
    elif exp==30:
        assert m=='qwen3_4b_base'
        add('hadamard','hadamard',evalsets=['wt2','c4'])
        for corpus,n in [('wt2',64),('wt2',128),('wt2',256),('c4',128)]:
            add(f'{corpus}_n{n}',label=f'{corpus}_n{n}_s{{seed}}',evalsets=['wt2','c4'])
    elif exp==31:
        assert m=='qwen3_4b_base';add('hadamard','hadamard')
        for k in ['8','max']:
            for p in ['P1','P2','P3','P4','P5']:add(f'{p}_k{k}',rank=k,variant=p)
    elif exp==32:
        add('hadamard','hadamard')
        for s in (['S1','S2','S3','S4'] if m=='qwen3_4b_base' else ['S3','S4']):add(s,solver=s)
    return result


def reference(exp,row):
    if exp==29:return row.split('_')[0]+'_hadamard'
    if exp==30:return 'hadamard'
    if exp==31:return 'hadamard' if row=='hadamard' else 'P1_'+row.split('_')[1]
    return 'S4'


def copy_measurement(m,exp,destrow,seed,sourceexp,sourcerow,evalset='wt2'):
    src=a.REPO/'results'/m/f'e{sourceexp}_per_sequence.csv'
    rr=[r for r in a.rows(src) if r['row']==sourcerow and int(r['seed'])==seed and r['eval_set']==evalset]
    if len(rr)!=64:return False
    assert {int(r['chunk']) for r in rr}==set(range(64))
    for r in rr:r.update(experiment=exp,row=destrow)
    dest=a.REPO/'results'/m/f'e{exp}_per_sequence.csv'
    a.append(dest,rr,['model','row','seed','eval_set','chunk'])
    audits=[r for r in a.rows(src.with_name(f'e{sourceexp}_gates.csv')) if r['row']==sourcerow and int(r['seed'])==seed]
    for r in audits:r.update(experiment=exp,row=destrow)
    a.append(dest.with_name(f'e{exp}_gates.csv'),audits,['model','row','seed','site','layer'])
    a.append(dest.with_name(f'e{exp}_reuse_audit.csv'),[dict(model=m,row=destrow,seed=seed,eval_set=evalset,source=f'results/{m}/e{sourceexp}_per_sequence.csv',source_row=sourcerow,source_sha256=a.sha(src),reason='identical new-experiment rotation, quantizer and frozen evaluation chunks')],['row','seed','eval_set'])
    return True


def stats(candidate,reference):
    """Corpus-PPL delta plus the requested paired-chunk delta-method interval."""
    c=np.asarray(candidate,dtype=float);r=np.asarray(reference,dtype=float)
    assert c.shape==r.shape==(3,64)
    cp=np.exp(c.mean(1));rp=np.exp(r.mean(1));diff=cp-rp
    influence=diff[:,None]+cp[:,None]*(c-c.mean(1)[:,None])-rp[:,None]*(r-r.mean(1)[:,None])
    half=float(T90_DF191*influence.std(ddof=1)/np.sqrt(192))
    seedhalf=float(T90_DF2*diff.std(ddof=1)/np.sqrt(3))
    nd=(c-r).reshape(-1);nh=float(T90_DF191*nd.std(ddof=1)/np.sqrt(192))
    return dict(mean_ppl=float(cp.mean()),seed_std=float(cp.std(ddof=1)),seed_ppls=json.dumps(cp.tolist()),
                delta=float(diff.mean()),ci90_low=float(diff.mean()-half),ci90_high=float(diff.mean()+half),
                seed_ci90_low=float(diff.mean()-seedhalf),seed_ci90_high=float(diff.mean()+seedhalf),
                nll_delta=float(nd.mean()),nll_ci90_low=float(nd.mean()-nh),nll_ci90_high=float(nd.mean()+nh),
                ci_method='paired-chunk delta-method Student-t df191; repeated texts across seeds',seeds=3,chunks_per_seed=64)


def summarize(exp,m):
    data=a.rows(a.REPO/'results'/m/f'e{exp}_per_sequence.csv');output=[]
    table={(r['row'],int(r['seed']),r['eval_set'],int(r['chunk'])):float(r['nll']) for r in data}
    for spec in plan(exp,m):
        row=spec['row'];ref=reference(exp,row)
        for ev in spec['evalsets']:
            required=[(name,s,ev,i) for name in [row,ref] for s in a.SEEDS for i in range(64)]
            if not all(k in table for k in required):continue
            c=[[table[row,s,ev,i] for i in range(64)] for s in a.SEEDS];r=[[table[ref,s,ev,i] for i in range(64)] for s in a.SEEDS]
            output.append(dict(experiment=exp,model=m,row=row,eval_set=ev,reference=ref,**stats(c,r)))
    a.write(a.REPO/'results'/m/f'e{exp}_summary.csv',output)
    return output


def record_metadata(exp,m):
    c,ds=a.config(m)
    content={'experiment':exp,'model':m,'created_utc':a.utc(),'preregistration_sha256':a.sha(a.REPO/'experiments/e29_e33_preregistration.md'),
        'rows':plan(exp,m),'seeds':list(a.SEEDS),'dimensions':ds,'layers':c.num_hidden_layers,
        'weights':'bf16 unchanged','KV':'bf16 unchanged','sites':['post-RMSNorm qkv input','down_proj input'],'fold':'R^T Q(R x), rounded to original bf16 only after transpose',
        'tokens':a.js(a.ASSETS/m/'tokens/DONE.json'),'row_bits':{},'source_sha256':{str(p.relative_to(a.REPO)):a.sha(p) for p in [Path(__file__),a.REPO/'nar/reviewer_ablations.py']}}
    for r in content['rows']:
        overhead=16 if r['symmetric'] else 32
        content['row_bits'][r['row']]={s:4+overhead/(n if r['group']=='token' else r['group']) for s,n in ds.items()}
    a.savej(a.REPO/'results'/m/f'e{exp}_metadata.json',content)


def row_diagnostics(exp,m,spec,seed,rot):
    if exp not in [31,32] or spec['method']!='pq':return
    out=[]
    for s,l,n in a.keys(m):
        d=rot.data[s,l];per=torch.empty(n,dtype=torch.float64,device='cuda');per[d['target']]=d['energy_after_g'][d['source']]
        grouped=per.reshape(-1,128)[:,1:].sum(1);spread=float(grouped.max()/grouped.median().clamp_min(1e-30))
        item=dict(model=m,row=spec['row'],seed=seed,site=s,layer=l,rank=d['k'],residual_group_energy_max_median=spread,captured_fraction=d['calibration_fraction'])
        if exp==32:
            p=a.loadt(a.ASSETS/m/'spectral'/spec['label']/f'{s}_{l:02d}.pt')['solvers'][spec['solver']]
            item.update(ritz_residual_median=float(p['ritz_residuals'].median()),ritz_residual_max=float(p['ritz_residuals'].max()),principal_angle_max_deg=float(p['principal_angles_deg'].max()))
        out.append(item)
    a.append(a.REPO/'results'/m/f'e{exp}_diagnostics.csv',out,['row','seed','site','layer'])


def run(exp,m):
    a.prepare_tokens(m);record_metadata(exp,m)
    model=None
    for spec in plan(exp,m):
        for seed in a.SEEDS:
            ss=dict(spec);ss['label']=spec['label'].format(seed=seed)
            if ss['method']=='pq':a.build_factors(m,ss['label'],ss['solver'],ss['rank'],ss['variant'],seed)
            for ev in spec['evalsets']:
                rr=a.rows(a.REPO/'results'/m/f'e{exp}_per_sequence.csv')
                if sum(r['row']==ss['row'] and int(r['seed'])==seed and r['eval_set']==ev for r in rr)==64:continue
                alias=None
                if exp!=29 and ss['method']=='hadamard':alias=(29,'Q1_hadamard')
                if (exp==31 and ss['row'] in ['P1_kmax','P4_kmax']) or (exp==32 and ss['row']=='S3'):alias=(29,'Q1_pq')
                if exp==30 and ss['row']=='wt2_n128' and seed==0:alias=(29,'Q1_pq')
                if alias and copy_measurement(m,exp,ss['row'],seed,*alias,ev):continue
                rot=a.Rotation(m,seed,ss['method'],ss['label'],ss['solver'],ss['rank'],ss['variant'])
                if model is None:model=a.base.load_model(a.act.MODEL_IDS[m],a.WORK)
                row_diagnostics(exp,m,ss,seed,rot)
                a.evaluate_row(model,m,exp,ss['row'],seed,rot,ss['group'],ss['symmetric'],ev)
                del rot;gc.collect();torch.cuda.empty_cache()
        summarize(exp,m)
    del model;gc.collect();torch.cuda.empty_cache()
    for spec in plan(exp,m):
        if exp in [31,32] and spec['method']=='pq':
            for seed in a.SEEDS:
                rot=a.Rotation(m,seed,spec['method'],spec['label'],spec['solver'],spec['rank'],spec['variant']);row_diagnostics(exp,m,spec,seed,rot);del rot
    summary=summarize(exp,m)
    assert len(summary)==sum(len(r['evalsets']) for r in plan(exp,m)),(exp,m,'incomplete summary')
    a.savej(a.REPO/'results'/m/f'e{exp}_DONE.json',{'status':'COMPLETE','completed_utc':a.utc(),'model':m,'experiment':exp,'summary_rows':len(summary),'hardware':a.base.hardware_info()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=int,required=True,choices=[29,30,31,32]);p.add_argument('--model',required=True);p.add_argument('--summarize',action='store_true');args=p.parse_args();a.setup()
    if args.summarize:summarize(args.experiment,args.model)
    else:run(args.experiment,args.model)
if __name__=='__main__':main()
