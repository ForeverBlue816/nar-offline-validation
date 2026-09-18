"""E33: unfitted range-law prediction, preserving historical and new paired data."""
from __future__ import annotations
import argparse,gc,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a, e20_null_space as e20
from nar.e12_wy import compact_wy


def archive():
    out=[]
    for p in sorted((a.REPO/'results').glob('*/e20_range_vs_config.csv')):
        for r in a.rows(p):
            measured=float(r['mean_group_range'])/15;had=float(r['hadamard_reference_range'])/15;f=float(r['absorbed_energy_fraction'])
            pred=had*math.sqrt(max(0,1-f))
            out.append(dict(model=r['model'],source=str(p.relative_to(a.REPO)),source_sha256=a.sha(p),row=r['row'],site=r['site'],layer=r['layer'],group=r['group'],m=r['m'],k=r['slots'],s_meas=measured,s_hadamard=had,f=f,s_pred=pred,relative_error=(pred-measured)/measured,seed_count=1,seed_ci90_low='',seed_ci90_high='',observations=r['rows_used'],scope='historical logged diagnostic; no invented seed replication'))
    a.write(a.REPO/'results/e33_historical_predictions.csv',out)


def grid(model):
    if model=='qwen3_4b_base':return [('E22_k8',128,1),('E22_kmax',128,1)]
    result=[(f'E11_{s}',g,1) for s,g in [('g64_kmax',64),('g128_k8',128),('g128_k16',128),('g128_k32',128),('g128_kmax',128),('g256_kmax',256)]]
    result += [(f'E20_{r.name}',r.group,r.m) for r in e20.ROWS if r.method!='bf16']
    return result


@lru_cache(maxsize=4)
def site_inputs(model,site,layer):
    v,en=e20.load_site_data(a.WORK,model,site,layer,torch.device('cuda'))
    weights=torch.tensor(e20.eigen_fractions(a.WORK,model)[site,layer],device='cuda',dtype=torch.float32)
    return v,en,weights

@lru_cache(maxsize=32)
def e20_factor(model,site,layer,group,m):
    v,en,_=site_inputs(model,site,layer)
    return e20.build_nar_factor(v,en,group,m)

@lru_cache(maxsize=16)
def frozen_factor(path):
    return a.loadt(path),a.act.RotationFactor.load(path,torch.device('cuda'))

_WY_CACHE={}
def cached_wy(fac):
    key=id(fac)
    if key not in _WY_CACHE:_WY_CACHE[key]=compact_wy(fac.reflectors,fac.active)
    return _WY_CACHE[key]

def with_transpose(fn,fac,signs):
    if fac is None:
        fn.inverse=lambda x:a.full_hadamard_rows_transpose(x,signs)
    else:
        w,y=cached_wy(fac)
        def inverse(x):
            n=fac.n;z=a.act.ext._fast_walsh_hadamard(x.reshape(-1,n//fac.b,fac.b)).reshape(-1,n)*signs
            un=torch.empty_like(z);un[:,fac.source_order]=z[:,fac.target_order]
            return un-(un@y)@w.T
        fn.inverse=inverse
    return fn


def legacy(model,name,site,layer,n,seed):
    gen=torch.Generator().manual_seed(a.base.DEFAULT_SEED+seed+1000*layer+(100000 if site=='down' else 0)+(0 if name.startswith('E20') else 128))
    signs=(torch.randint(0,2,(n,),generator=gen).float()*2-1).cuda()
    if name.startswith('E20'):
        row=next(r for r in e20.ROWS if 'E20_'+r.name==name)
        v,en,weights=site_inputs(model,site,layer)
        fac=e20_factor(model,site,layer,row.group,row.m) if row.method=='nar' else None
        projector=e20.null_space_basis(n,row.group,row.m,torch.device('cuda'))
        if fac is None:apply=lambda x:a.act.full_hadamard_rows(x,signs)
        else:
            w,y=cached_wy(fac)
            def apply(x):
                z=x-(x@w)@y.T;p=torch.empty_like(z);p[:,fac.target_order]=z[:,fac.source_order]
                return a.act.ext._fast_walsh_hadamard((p*signs).reshape(-1,n//fac.b,fac.b)).reshape(-1,n)
        take=min(v.shape[1],len(weights));mapped=apply(v[:,:take].T)
        share=(mapped@projector.T).square().sum(1)/mapped.square().sum(1).clamp_min(1e-30)
        f=float((share*weights[:take]).sum());k=0 if fac is None else len(fac.reflectors)
        return with_transpose(apply,fac,signs),signs,f,k,row.group,row.m,'E11 energy-weighted P_N capture; frozen E20 construction'
    if name.startswith('E22'):
        rank=name.split('_')[1];path=a.WORK/'activations'/model/'e18v2_factors'/f'nar_{rank}'/f'{site}_layer_{layer:02d}.pt'
    else:
        label=name[4:]
        path=(a.act.factor_dir(a.WORK,model) if label=='g128_kmax' else a.WORK/'activations'/model/'e11_calibration/factors'/('nar_b'+label[1:]))/f'{site}_layer_{layer:02d}.pt'
    payload,fac=frozen_factor(path);k=len(fac.reflectors)
    w,y=cached_wy(fac)
    def apply(x):
        z=x-(x@w)@y.T;p=torch.empty_like(z);p[:,fac.target_order]=z[:,fac.source_order]
        return a.act.ext._fast_walsh_hadamard((p*signs).reshape(-1,n//fac.b,fac.b)).reshape(-1,n)
    if 'eigenvalues' in payload:
        f=float(payload['eigenvalues'][:k].sum()/payload['trace']);definition='saved factor eigenvalue sum / calibration trace'
    elif model=='llama32_3b' and name=='E11_g128_kmax':
        src='q_input' if site=='qkv' else 'down_input'
        eig=a.loadt(a.WORK/'activations'/model/'wide_cal_a/analysis/eigenspaces'/f'{src}_layer_{layer:02d}.pt')
        f=float(eig['eigenvalues'][:k].sum()/eig['trace']);definition='frozen E1c eigenspace paired with activation_factors'
    else:
        rr=[r for r in a.rows(a.REPO/'results'/model/'e11_calibration_eigenspace.csv') if r['site']==site and int(r['layer'])==layer and int(r['rank'])<=k]
        assert len(rr)==k,(model,name,site,layer,k,len(rr))
        f=sum(float(r['fraction_total_energy']) for r in rr);definition='frozen E11 selected eigenvalue fractions'
    return with_transpose(apply,fac,signs),signs,f,k,fac.b,1,definition


@torch.inference_mode()
def measure_site(model,site,layer,data):
    """Process all 64×2048 unquantized activations; never subsample rows."""
    n=data.shape[-1];dest=a.REPO/'results'/model/'e33_per_seed.csv'
    done={(r['row'],int(r['seed']),r['site'],int(r['layer'])) for r in a.rows(dest)}
    for name,group,mm in grid(model):
        for seed in a.SEEDS:
            if (name,seed,site,layer) in done:continue
            apply,signs,f,k,g,m,definition=legacy(model,name,site,layer,n,seed)
            assert g==group and m==mm
            gen=torch.Generator().manual_seed(8800+layer)
            probe=torch.randn((8,n),generator=gen).cuda()
            error=float((apply.inverse(apply(probe))-probe).norm()/probe.norm())
            a.append(dest.with_name('e33_gates.csv'),[dict(model=model,row=name,site=site,layer=layer,seed=seed,round_trip=error,tolerance=1e-6)],['row','site','layer','seed'])
            assert error<=1e-6,(model,name,site,layer,seed,error)
            total=0.;measured=0.;hadamard=0.;captured=0.;count=0
            basis=e20.aq.walsh_basis(g,m,torch.device('cuda'))
            for chunk in range(64):
                x=data[chunk].cuda().float();z=apply(x);h=a.act.full_hadamard_rows(x,signs)
                grouped=z.reshape(-1,n//g,g)
                coeff=grouped@basis.T;captured+=float(coeff.square().sum(dtype=torch.float64));total+=float(x.square().sum(dtype=torch.float64))
                if m>1:grouped=grouped-(grouped@basis[1:].T)@basis[1:]
                measured+=float(((grouped.amax(-1)-grouped.amin(-1))/15).sum(dtype=torch.float64))
                hg=h.reshape(-1,n//g,g);hadamard+=float(((hg.amax(-1)-hg.amin(-1))/15).sum(dtype=torch.float64));count+=grouped.shape[0]*grouped.shape[1]
            s_meas=measured/count;s_h=hadamard/count;pred=s_h*math.sqrt(max(0,1-f))
            a.append(dest,[dict(model=model,row=name,site=site,layer=layer,seed=seed,group=g,m=m,k=k,s_meas=s_meas,s_hadamard=s_h,f_calibration=f,f_test_capture=captured/total,s_pred=pred,relative_error=(pred-s_meas)/s_meas,observations=64*a.LENGTH,groups_measured=count,f_definition=definition)],['row','site','layer','seed'])
            del apply;gc.collect();torch.cuda.empty_cache()
    _WY_CACHE.clear();site_inputs.cache_clear();e20_factor.cache_clear();frozen_factor.cache_clear()
    a.LOG.info('E33 %s %s layer=%d all rows/seeds complete',model,site,layer)


@torch.inference_mode()
def run(model_key):
    a.prepare_tokens(model_key)
    a.savej(a.REPO/'results'/model_key/'e33_metadata.json',{'model':model_key,'grid':grid(model_key),'created_utc':a.utc(),'source_sha256':a.sha(Path(__file__)),'tokens':a.js(a.ASSETS/model_key/'tokens/DONE.json'),'protocol':'64 full frozen chunks; unquantized bf16 activations; fp32 rotations; range/15; calibration fraction; 3 paired sign seeds; no fitted coefficient','frozen_factor_seed_note':'E11/E22 signs use E27 mapping; E20 uses E20 mapping without +128'})
    tokens=a.loadt(a.tokenpath(model_key,'wt2_eval'));assert len(tokens)==64
    model=a.base.load_model(a.act.MODEL_IDS[model_key],a.WORK);first={}
    assert not getattr(model.model,'has_sliding_layers',False)
    class Captured(Exception):pass
    def catch(mod,args,kw):first.update(kw);raise Captured()
    handle=model.model.layers[0].register_forward_pre_hook(catch,with_kwargs=True)
    try:
        try:model.model(input_ids=tokens[:1].cuda(),use_cache=False)
        except Captured:pass
    finally:handle.remove()
    hidden=torch.cat([model.model.embed_tokens(t[None].cuda()).cpu() for t in tokens])
    for layer,block in enumerate(model.model.layers):
        buffers={};output=torch.empty_like(hidden);at=0
        def consume(site,value):
            if site not in buffers:buffers[site]=torch.empty((64,a.LENGTH,value.shape[-1]),dtype=torch.bfloat16)
            buffers[site][at:at+1]=value.cpu()
        hs=[block.input_layernorm.register_forward_hook(lambda mod,inp,out:consume('qkv',out)),block.mlp.down_proj.register_forward_pre_hook(lambda mod,inp:consume('down',inp[0]))]
        try:
            for at in range(64):output[at:at+1]=block(hidden[at:at+1].cuda(),**first).cpu()
        finally:
            for h in hs:h.remove()
        hidden=output
        for site,data in buffers.items():measure_site(model_key,site,layer,data)
        del buffers;gc.collect();torch.cuda.empty_cache()
    expected=len(grid(model_key))*3*2*len(model.model.layers)
    rr=a.rows(a.REPO/'results'/model_key/'e33_per_seed.csv');assert len(rr)==expected,(len(rr),expected)
    a.savej(a.REPO/'results'/model_key/'e33_DONE.json',{'status':'COMPLETE','rows':len(rr),'completed_utc':a.utc(),'hardware':a.base.hardware_info()})


def summarize():
    group=defaultdict(list)
    for p in (a.REPO/'results').glob('*/e33_per_seed.csv'):
        for r in a.rows(p):group[r['model'],r['row'],r['site'],r['layer']].append(r)
    out=[]
    for key,rr in sorted(group.items()):
        if {int(r['seed']) for r in rr}!={0,1,2}:continue
        d={k:rr[0][k] for k in ['model','row','site','layer','group','m','k']}
        for field in ['s_meas','s_hadamard','s_pred','relative_error','f_calibration','f_test_capture']:
            vals=np.array([float(r[field]) for r in rr]);half=2.919985580353725*vals.std(ddof=1)/math.sqrt(3)
            d[field]=float(vals.mean());d[field+'_ci90_low']=float(vals.mean()-half);d[field+'_ci90_high']=float(vals.mean()+half)
        d.update(seeds=3,observations_per_seed=64*a.LENGTH);out.append(d)
    a.write(a.REPO/'results/e33_rangelaw_persite.csv',out)
    grouped=defaultdict(list)
    for r in out:grouped[r['model'],r['row'],r['site'],r['k']].append(abs(r['relative_error']))
    a.write(a.REPO/'results/e33_error_summary.csv',[dict(model=k[0],row=k[1],site=k[2],k=k[3],layers=len(v),median_absolute_relative_error=float(np.median(v)),p90_absolute_relative_error=float(np.quantile(v,.9))) for k,v in sorted(grouped.items())])


def main():
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--archive',action='store_true');p.add_argument('--summarize',action='store_true');arg=p.parse_args();a.setup()
    if arg.archive:archive()
    elif arg.summarize:summarize()
    else:run(arg.model)
if __name__=='__main__':main()
