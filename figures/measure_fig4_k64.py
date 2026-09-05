#!/usr/bin/env python3
"""Add only the missing E11 k=64 measurement using its frozen eigenbasis."""
from pathlib import Path
import argparse, gc, hashlib, json, math, sys
import numpy as np
import pandas as pd
import torch

METHOD='nar_b128_k64'
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--seed',type=int,default=20260902)
    a=p.parse_args();sys.path.insert(0,str(a.code_root))
    from nar import e11_fair_baselines as e11
    from nar import activation_experiments as act
    from nar import experiment as base
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');base.seed_everything(a.seed)
    output=a.repo/'results'/a.model;output.mkdir(parents=True,exist_ok=True)
    result=output/'e11_k64_summary.csv';partial=output/'e11_k64_per_sequence.partial.csv'
    if result.exists():
        print('Completed measurement already exists',flush=True);return
    a.scratch.mkdir(parents=True,exist_ok=True);base.setup_logging(a.scratch,f'fig4-k64-{a.model}')
    cal=a.assets/'activations'/a.model/'e11_calibration';meta=json.loads((cal/'DONE.json').read_text())
    layers=int(meta['layers']);dims=meta['dimensions'];model_id=meta['model_id']
    e11.NAR_SPECS={METHOD:(128,64)}
    factor_root=e11.calibration_dir(a.scratch,a.model)
    factor_done=factor_root/'K64_FACTORS.json'
    model=base.load_model(model_id,a.assets)
    if not factor_done.exists():
        vectors={};audits=[]
        for site,n in dims.items():
            rank=min(64,n//128)
            for layer in range(layers):
                path=cal/'factors'/'nar_b64_kmax'/f'{site}_layer_{layer:02d}.pt'
                saved=act.RotationFactor.load(path,torch.device('cuda'))
                anchors=torch.zeros(rank,n,device='cuda'); anchors[torch.arange(rank),torch.arange(rank)*64]=1
                # G maps frozen v_i to e_(64*i); G^-1 recovers them without an eigensolver.
                recovered=act.apply_reflectors(anchors,saved.reflectors[:rank].flip(0),saved.active[:rank].flip(0))
                err=float((recovered@recovered.T-torch.eye(rank,device='cuda')).abs().max())
                refs,active,_=act.reflectors_from_vectors(recovered[:min(rank,32)].T,128)
                control=act.RotationFactor.load(cal/'factors'/'nar_b128_k32'/f'{site}_layer_{layer:02d}.pt',torch.device('cuda'))
                prefix_error=float((refs-control.reflectors).abs().max())
                assert err<1e-4 and prefix_error<1e-4,(site,layer,err,prefix_error)
                vectors[(site,layer)]=recovered.T.cpu()
                audits.append(dict(model=a.model,site=site,layer=layer,requested_k=64,realized_k=rank,
                    orthogonality_error=err,reconstructed_k32_reflector_error=prefix_error,
                    source_file=str(path.relative_to(a.assets)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                del saved,anchors,recovered,control,refs
        train=base.prepare_token_chunks(model_id,'train',0,128,2048,a.assets)
        collector=e11.VariantEnergyCollector(model,vectors,32,layers);collector.install()
        try: e11._model_pass(model,train,1,'Figure 4 k64 permutation calibration',collector)
        finally: collector.close()
        factor_rows,anchor_error=e11._save_factor_variants(factor_root,dims,layers,collector)
        pd.DataFrame(factor_rows).to_csv(output/'e11_k64_factor_audit.csv',index=False)
        pd.DataFrame(audits).to_csv(output/'e11_k64_basis_audit.csv',index=False)
        base.atomic_json(factor_done,{'source':'frozen E11 b64 factor inverse; new b128 k64 permutation-energy pass',
            'calibration_sequences':128,'permutation_stride':32,'max_anchor_error':anchor_error})
        del collector,train,vectors;gc.collect();torch.cuda.empty_cache()
    test=base.prepare_token_chunks(model_id,'test',0,64,2048,a.assets)
    stats=torch.load(cal/'channel_stats.pt',map_location='cpu',weights_only=True)
    manager=e11.WeightManager(model,layers)
    rows=pd.read_csv(partial).to_dict('records') if partial.exists() else []
    for seed_index in range(3):
        seed=a.seed+seed_index
        if sum(int(r['seed'])==seed for r in rows)==64: continue
        manager.restore()
        transform=e11.Transform(METHOD,a.model,a.scratch,seed_index,a.seed,layers,dims,torch.device('cuda'),stats)
        fold_error=manager.apply(transform,512)
        hooks=e11.Hooks(model,transform);hooks.install()
        try: losses=act.evaluate_nlls(model,test,f'Figure 4 {a.model} k64 seed={seed}')
        finally: hooks.close()
        assert len(losses)==64 and np.isfinite(losses).all()
        rows.extend(dict(model=a.model,method=METHOD,seed=seed,sequence=i,nll=loss,tokens_scored=2047,
            source='supplementary E11 k64 measurement for Figure 4',weight_fold_max_relative_error=fold_error) for i,loss in enumerate(losses))
        pd.DataFrame(rows).to_csv(partial,index=False)
        print(json.dumps({'model':a.model,'seed':seed,'ppl':math.exp(float(np.mean(losses))),'fold_error':fold_error}),flush=True)
        del transform;gc.collect();torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);assert len(frame)==192
    ppls=frame.groupby('seed').nll.mean().map(math.exp)
    summary=dict(model=a.model,method=METHOD,seeds=3,mean_ppl=float(ppls.mean()),seed_ppl_std=float(ppls.std(ddof=1)),
        seed_ppls=json.dumps(ppls.to_dict()),chunks=64,site_kmax=json.dumps({s:n//128 for s,n in dims.items()}),
        site_k_actual=json.dumps({s:min(64,n//128) for s,n in dims.items()}),
        source='e11_k64_per_sequence.csv',protocol='frozen E11 activation-only, original 64 chunks, three original rotation seeds')
    frame.to_csv(output/'e11_k64_per_sequence.csv',index=False)
    base.atomic_json(output/'E11_K64_DONE.json',{'model':a.model,'summary':summary,'code_commit':'f424b82',
        'evaluation_token_sha256':hashlib.sha256(test.numpy().tobytes()).hexdigest(),
        'device':torch.cuda.get_device_name(),'factor_cache':str(factor_root),
        'source_calibration':str(cal),'no_eigensolver_rerun':True})
    pd.DataFrame([summary]).to_csv(result,index=False);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
