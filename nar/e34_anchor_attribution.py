"""E34 frozen-rotation column controls, token losses, and masked interventions."""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a

MODELS=('qwen3_4b_base','llama32_3b')
COLUMNS={'Q1':0,'Q5a':64,'Q5b':1}
PREREG=a.REPO/'experiments/e34_preregistration.md'

def root(m):return a.REPO/'results'/m
def assets(m):return a.ASSETS/m/'e34'
def sha_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def plan():
    out=[]
    for sym in (False,True):
        fmt='sym' if sym else 'asym'
        for name,method in [('Q1','pq'),('Had','hadamard'),('Q5a','pq'),('Q5b','pq')]:
            out.append(dict(row=f'{name}_{fmt}',column=COLUMNS.get(name,0),method=method,symmetric=sym,mask='all',baseline=(('Q2' if sym else 'Q1')+'_'+method) if name in ['Q1','Had'] else None))
    # Baseline replays must all pass before any new column/mask PPL is run.
    out.sort(key=lambda r:r['baseline'] is None)
    for mask in ['M1','M2']:
        for sym in (False,True):
            for method in ['pq','hadamard']:
                out.append(dict(row=f'{mask}_{"pq" if method=="pq" else "had"}_{"sym" if sym else "asym"}',column=0,method=method,symmetric=sym,mask=mask,baseline=None))
    return out

def swap_targets(target,column,group=128):
    result=target.clone()
    if column:
        local=target%group;result[local==0]+=column;result[local==column]-=column
    return result

def column_pattern(column,device='cpu'):
    unit=torch.zeros((1,128),device=device);unit[0,column]=1
    return a.act.ext._fast_walsh_hadamard(unit)[0]

def flags_from_scores(scores):
    ids=np.arange(scores.size);candidates=ids[ids%scores.shape[1]!=0]
    keep=math.ceil(.001*len(candidates));order=np.lexsort((candidates,-scores.reshape(-1)[candidates]))
    massive=np.zeros(scores.size,dtype=bool);massive[candidates[order[:keep]]]=True
    massive=massive.reshape(scores.shape);bos=np.zeros_like(massive);bos[:,0]=True
    return massive,bos

class ColumnRotation(a.Rotation):
    def __init__(self,m,seed,method='pq',column=0):
        super().__init__(m,seed,method)
        self.column=column
        if method=='pq':
            for d in self.data.values():d['target']=swap_targets(d['target'],column)
    def gate(self):
        out=[]
        for site,layer,n in a.keys(self.m):
            x=torch.randn((8,n),generator=torch.Generator().manual_seed(7000+layer)).cuda()
            back=self.apply(site,layer,self.apply(site,layer,x),True)
            rt=float((back-x).norm()/x.norm());anchor=0.
            if self.method=='pq':
                d=self.data[site,layer];mapped=self.apply(site,layer,d['vectors'].T.float())
                target=torch.zeros_like(mapped);pattern=column_pattern(self.column,'cuda')
                for j in range(d['k']):target[j,j*128:(j+1)*128]=pattern
                anchor=float(torch.minimum((mapped-target).norm(dim=1),(mapped+target).norm(dim=1)).max())
            out.append(dict(site=site,layer=layer,round_trip=rt,anchor_residual=anchor))
        return out

def freeze():
    dest=a.REPO/'experiments/e34_execution_manifest.json'
    if dest.exists():return a.js(dest)
    files=[PREREG,Path(__file__),a.REPO/'nar/reviewer_ablations.py',a.REPO/'nar/run_reviewer_experiments.py',a.REPO/'slurm_e34.sh']
    inputs={};factor_inventory=[]
    for m in MODELS:
        for p in (root(m)/'e29_per_sequence.csv',root(m)/'e29_gates.csv',root(m)/'e29_metadata.json',root(m)/'e29_summary.csv',a.tokenpath(m,'wt2_eval')):inputs[str(p)]=a.sha(p)
        for s,l,n in a.keys(m):
            p=a.factor_path(m,'wt2_n128_s0','S3','max','P1',0,s,l)
            factor_inventory.append(dict(model=m,site=s,layer=l,path=str(p),sha256=a.sha(p)))
    import subprocess
    baseline=subprocess.check_output(['git','ls-tree','-r','--name-only','6251b7f','results/'],text=True,cwd=a.REPO).splitlines()
    frozen=dict(utc=a.utc(),status='FROZEN_BEFORE_FIRST_FORWARD',parent_commit='6251b7f',source_sha256={str(p.relative_to(a.REPO)):a.sha(p) for p in files},input_sha256=inputs,rows=plan(),factors=factor_inventory,old_results_sha256={p:a.sha(a.REPO/p) for p in baseline},report_prefix_sha256=hashlib.sha256(subprocess.check_output(['git','show','6251b7f:report.md'],cwd=a.REPO)).hexdigest())
    a.savej(dest,frozen);return frozen

def verify_frozen(m):
    manifest=a.js(a.REPO/'experiments/e34_execution_manifest.json')
    for name,sha in manifest['source_sha256'].items():assert a.sha(a.REPO/name)==sha,('changed frozen source',name)
    for name,sha in manifest['input_sha256'].items():
        if m in name:assert a.sha(name)==sha,('changed input',name)
    for r in manifest['factors']:
        if r['model']==m:assert a.sha(r['path'])==r['sha256']
    return manifest

def gate_all(m):
    dest=root(m)/'e34_gates.csv';out=[];contracts=[]
    for method,col in [('hadamard',0),('pq',0),('pq',64),('pq',1)]:
        for seed in a.SEEDS:
            rot=ColumnRotation(m,seed,method,col);checks=rot.gate()
            assert all(r['round_trip']<=1e-6 and r['anchor_residual']<=1e-6 for r in checks),checks
            if method=='pq':
                for s,l,n in a.keys(m):
                    old=a.loadt(a.factor_path(m,'wt2_n128_s0','S3','max','P1',0,s,l));d=rot.data[s,l]
                    assert torch.equal(old['w'],d['w'].cpu().float()) and torch.equal(old['y'],d['y'].cpu().float())
                    assert torch.equal(old['vectors'],d['vectors'].cpu()) and torch.equal(old['source'],d['source'].cpu())
                    target=d['target'].cpu();changed=int((target!=old['target']).sum());assert changed==(2*d['k'] if col else 0)
                    assert torch.equal(target.sort().values,torch.arange(n))
                    pattern=column_pattern(col);transitions=int((pattern[1:]!=pattern[:-1]).sum());assert transitions=={0:0,64:1,1:127}[col]
                    contracts.append(dict(model=m,seed=seed,site=s,layer=l,column=col,selected_directions=d['k'],group_count=n//128,other_groups=0,changed_target_coordinates=changed,unchanged_G=True,unchanged_source_order=True,unchanged_D=True,original_target_sha256=sha_tensor(old['target']),new_target_sha256=sha_tensor(target),D_sha256=sha_tensor(rot.signs[s,l]),column_sign_transitions=transitions,column_crest_factor=float(pattern.abs().max()/pattern.square().mean().sqrt())))
            for spec in plan():
                if spec['method']==method and spec['column']==col:
                    out += [dict(model=m,row=spec['row'],seed=seed,**r) for r in checks]
            del rot;gc.collect();torch.cuda.empty_cache()
    a.write(dest,out);a.write(root(m)/'e34_rotation_contract.csv',contracts)

@torch.inference_mode()
def capture_flags_ranges(model,m):
    done=root(m)/'e34_flags.json'
    if done.exists():return a.loadt(assets(m)/'flags.pt')
    tokens=a.loadt(a.tokenpath(m,'wt2_eval'));assert tuple(tokens.shape)==(64,2048)
    c,ds=a.config(m);nl=c.num_hidden_layers;scores=np.empty((nl,64,2048),dtype=np.float32)
    rots=[ColumnRotation(m,seed) for seed in a.SEEDS]
    ranges={(name,seed,l):torch.zeros(ds['down']//128,dtype=torch.float64,device='cuda') for name in ['Q1','Q5a','Q5b','Had'] for seed in a.SEEDS for l in range(nl)}
    energy={(site,l):torch.zeros(2,dtype=torch.float64,device='cuda') for site in ['qkv','down'] for l in range(nl)}
    handles=[];chunk=0
    def capture_residual(l,v):scores[l,chunk]=v[0].abs().amax(-1).float().cpu().numpy()
    def consume(site,l,value):
        n=value.shape[-1];x=value.float().reshape(-1,n);d=rots[0].data[site,l]
        xx=x.double();gx=(xx-(xx@d['w'])@d['y'].T).float()
        target0=d['target'];source=d['source']
        for seed in (a.SEEDS if site=='down' else [0]):
            sign=rots[seed].signs[site,l]
            for name,col in (COLUMNS.items() if site=='down' else [('Q1',0)]):
                z=torch.empty_like(gx);z[:,swap_targets(target0,col)]=gx[:,source]
                y=a.act.ext._fast_walsh_hadamard((z*sign).reshape(-1,n//128,128))
                if name=='Q1' and seed==0:
                    energy[site,l][0]+=(y.mean(-1).double().square()*128).sum()
                    energy[site,l][1]+=x.double().square().sum()
                if site=='down':ranges[name,seed,l]+=(y.amax(-1)-y.amin(-1)).sum(0,dtype=torch.float64)
            if site=='down':
                h=a.act.full_hadamard_rows(x,sign).reshape(-1,n//128,128)
                ranges['Had',seed,l]+=(h.amax(-1)-h.amin(-1)).sum(0,dtype=torch.float64)
    for l,block in enumerate(model.model.layers):
        handles += [block.register_forward_pre_hook(lambda mod,inp,l=l:capture_residual(l,inp[0])),block.input_layernorm.register_forward_hook(lambda mod,inp,out,l=l:consume('qkv',l,out)),block.mlp.down_proj.register_forward_pre_hook(lambda mod,inp,l=l:consume('down',l,inp[0]))]
    try:
        for chunk in range(64):
            model.model(input_ids=tokens[chunk:chunk+1].cuda(),use_cache=False)
            if chunk%8==0:a.LOG.info('E34 flags/common-input ranges %s chunk%d/64',m,chunk+1)
    finally:
        for h in handles:h.remove()
    shares=[dict(model=m,site=site,layer=l,dc_energy=float(v[0]),total_energy=float(v[1]),dc_share=float(v[0]/v[1]),rotation_seed=0,samples=64*2048) for (site,l),v in energy.items()]
    chosen=max([r for r in shares if r['site']=='qkv'],key=lambda r:(r['dc_share'],-r['layer']))
    massive,bos=flags_from_scores(scores[chosen['layer']]);flagged=massive|bos
    assert .0005<=massive.mean()<=.005 and .0005<=flagged.mean()<=.005
    payload=dict(massive=torch.from_numpy(massive),bos=torch.from_numpy(bos),flagged=torch.from_numpy(flagged),scores=torch.from_numpy(scores[chosen['layer']]),layer=chosen['layer'])
    a.savet(assets(m)/'flags.pt',payload)
    a.write(root(m)/'e34_dc_share_by_layer.csv',shares)
    a.write(root(m)/'e34_flagged_tokens.csv',[dict(model=m,chunk=int(ch),input_position=int(pos),token_id=int(tokens[ch,pos]),token_class='BOS' if pos==0 else 'massive',residual_linf=float(scores[chosen['layer'],ch,pos]),selected_layer=chosen['layer'],has_scored_next_token=bool(pos<2047)) for ch,pos in np.argwhere(flagged)])
    out=[]
    for (name,seed,l),values in ranges.items():
        for group,v in enumerate(values.cpu().tolist()):out.append(dict(model=m,row=name,quantizer='asym_and_sym_same_prequant_input',seed=seed,site='down',layer=l,group=group,group_class='other' if name=='Had' else 'anchor',observations=64*2048,mean_range=v/(64*2048),scope='common_unquantized_bf16_forward'))
    a.write(root(m)/'e34_range_common_input.csv',out)
    a.savej(done,dict(status='COMPLETE',utc=a.utc(),model=m,selected_layer=chosen['layer'],selection_site='post-RMSNorm qkv DC share; pre-layer residual norm',selected_dc_share=chosen['dc_share'],input_positions=flagged.size,massive_count=int(massive.sum()),bos_count=int(bos.sum()),flagged_count=int(flagged.sum()),massive_fraction=float(massive.mean()),flagged_fraction=float(flagged.mean()),scored_positions=64*2047,scored_massive_count=int(massive[:,:-1].sum()),unscored_flagged_final_positions=int(flagged[:,-1].sum()),flag_tensor_sha256=sha_tensor(payload['flagged']),flag_file_sha256=a.sha(assets(m)/'flags.pt'),token_sha256=a.sha(a.tokenpath(m,'wt2_eval'))))
    del rots;gc.collect();torch.cuda.empty_cache();return payload

class Hooks(a.Hooks):
    def __init__(self,model,rot,symmetric,mask,collect):
        super().__init__(model,rot,128,symmetric);self.mask=mask;self.collect=collect;self.ranges={};self.bypass_exact=True
    def transform(self,s,l,v):
        z=self.rot.apply(s,l,v)
        if self.collect and s=='down':
            g=z.reshape(-1,z.shape[-1]//128,128);self.ranges[l]=(g.amax(-1)-g.amin(-1)).sum(0,dtype=torch.float64).cpu()
        q=self.rot.apply(s,l,a.quantize(z,128,self.symmetric),True).to(v.dtype)
        if self.mask is not None:
            q=torch.where(self.mask[None,:,None],q,v)
            self.bypass_exact=self.bypass_exact and torch.equal(q[:,~self.mask],v[:,~self.mask])
        return q

def baseline_rows(m,spec,seed):
    if not spec['baseline']:return {}
    rr=[r for r in a.rows(root(m)/'e29_per_sequence.csv') if r['row']==spec['baseline'] and int(r['seed'])==seed and r['eval_set']=='wt2']
    assert len(rr)==64;return {int(r['chunk']):r for r in rr}

def checkpoint(m,row,seed,chunk):return assets(m)/'replays'/row/f'seed{seed}'/f'chunk{chunk:02d}.pt'

@torch.inference_mode()
def evaluate(model,m,spec,seed,flags):
    rot=ColumnRotation(m,seed,spec['method'],spec['column']);old=baseline_rows(m,spec,seed)
    tok=a.loadt(a.tokenpath(m,'wt2_eval'));dest=root(m)/'e34_per_sequence.csv';token_hash=a.sha(a.tokenpath(m,'wt2_eval'))
    keys=['row','seed','chunk'];collect=spec['mask']=='all'
    for chunk in range(64):
        cp=checkpoint(m,spec['row'],seed,chunk)
        if cp.exists():saved=a.loadt(cp)
        else:
            mask=None
            if spec['mask']!='all':
                mask=flags['flagged'][chunk].cuda()
                if spec['mask']=='M2':mask=~mask
            hooks=Hooks(model,rot,spec['symmetric'],mask,collect);hooks.install()
            try:
                batch=tok[chunk:chunk+1].cuda();logits=model(input_ids=batch,use_cache=False).logits
                flat=logits[:,:-1].float().reshape(-1,logits.shape[-1]);targets=batch[:,1:].reshape(-1)
                loss=F.cross_entropy(flat,targets);assert loss.dtype==torch.float32 and bool(loss.isfinite())
                per_token=F.cross_entropy(flat,targets,reduction='none').cpu() if old else torch.empty(0)
                assert hooks.bypass_exact
                saved=dict(nll=float(loss),token_nll=per_token,range_sum=torch.stack([hooks.ranges[l] for l in sorted(hooks.ranges)]) if collect else torch.empty(0),unmasked_bf16_bit_identical=hooks.bypass_exact)
                if old:
                    actual=np.float32(saved['nll']).tobytes();expected=np.float32(float(old[chunk]['nll'])).tobytes()
                    if actual!=expected:
                        a.savej(root(m)/f"e34_{spec['row']}_seed{seed}_REPLAY_FAILURE.json",dict(chunk=chunk,old_nll=old[chunk]['nll'],new_nll=saved['nll'],absolute_difference=abs(saved['nll']-float(old[chunk]['nll'])),source_sha256=a.sha(root(m)/'e29_per_sequence.csv')))
                        raise RuntimeError(f'E29 replay not bit identical: {m} {spec["row"]} seed{seed} chunk{chunk}')
                a.savet(cp,saved)
                del flat,logits,loss,per_token
            finally:hooks.close()
        if old:
            assert np.float32(saved['nll']).tobytes()==np.float32(float(old[chunk]['nll'])).tobytes()
            a.append(root(m)/'e34_replay_audit.csv',[dict(model=m,row=spec['row'],seed=seed,chunk=chunk,source_row=spec['baseline'],old_nll=old[chunk]['nll'],replay_nll=saved['nll'],float32_bit_identical=True,per_token_mean_difference=float(saved['token_nll'].mean())-saved['nll'],source_sha256=a.sha(root(m)/'e29_per_sequence.csv'))],keys)
        quantized=2048 if spec['mask']=='all' else int(flags['flagged'][chunk].sum())
        if spec['mask']=='M2':quantized=2048-quantized
        a.append(dest,[dict(model=m,row=spec['row'],seed=seed,chunk=chunk,nll=old[chunk]['nll'] if old else saved['nll'],tokens=2047,quantized_input_positions=quantized,token_sha256=token_hash,unmasked_bf16_bit_identical=saved['unmasked_bf16_bit_identical'])],keys)
        if chunk%16==0:a.LOG.info('E34 %s %s seed%d chunk%d/64',m,spec['row'],seed,chunk+1)
    if old:
        oldvec=np.array([float(old[i]['nll']) for i in range(64)],dtype=np.float32)
        newvec=np.array([a.loadt(checkpoint(m,spec['row'],seed,i))['nll'] for i in range(64)],dtype=np.float32)
        assert oldvec.tobytes()==newvec.tobytes()
        a.append(root(m)/'e34_baseline_reuse.csv',[dict(model=m,row=spec['row'],seed=seed,source_row=spec['baseline'],source_sha256=a.sha(root(m)/'e29_per_sequence.csv'),source_fp32_nll_sha256=hashlib.sha256(oldvec.tobytes()).hexdigest(),replay_fp32_nll_sha256=hashlib.sha256(newvec.tobytes()).hexdigest(),chunks=64,all_bit_identical=True)],['row','seed'])
    if collect:
        sums=sum((a.loadt(checkpoint(m,spec['row'],seed,i))['range_sum'] for i in range(64)))
        rows=[dict(model=m,row=spec['row'],seed=seed,site='down',layer=l,group=g,group_class='other' if spec['method']=='hadamard' else 'anchor',observations=64*2048,mean_range=float(v)/(64*2048),scope='actual_quantized_forward') for l,layer in enumerate(sums) for g,v in enumerate(layer)]
        a.append(root(m)/'e34_range_runtime.csv',rows,['row','seed','layer','group'])
    del rot;gc.collect();torch.cuda.empty_cache()

def export_tokens(m,flags):
    tok=a.loadt(a.tokenpath(m,'wt2_eval'));baseline=['Q1_asym','Had_asym','Q1_sym','Had_sym']
    for seed in a.SEEDS:
        dest=root(m)/f'e34_per_token_seed{seed}.csv';tmp=dest.with_suffix('.tmp')
        fields=['model','seed','chunk','predictor_position','target_position','target_token_id','predictor_class','target_class','nll_pq_asym','nll_had_asym','nll_pq_sym','nll_had_sym','gain','offset_spec']
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for chunk in range(64):
                losses={r:a.loadt(checkpoint(m,r,seed,chunk))['token_nll'].double().numpy() for r in baseline}
                def category(pos):return 'BOS' if pos==0 else ('massive' if flags['massive'][chunk,pos] else 'other')
                for p in range(2047):
                    pa,ha,ps,hs=[float(losses[r][p]) for r in baseline]
                    writer.writerow(dict(model=m,seed=seed,chunk=chunk,predictor_position=p,target_position=p+1,target_token_id=int(tok[chunk,p+1]),predictor_class=category(p),target_class=category(p+1),nll_pq_asym=pa,nll_had_asym=ha,nll_pq_sym=ps,nll_had_sym=hs,gain=ha-pa,offset_spec=(ps-pa)-(hs-ha)))
            f.flush();import os;os.fsync(f.fileno())
        tmp.replace(dest)

def run(m):
    manifest=verify_frozen(m);a.LOG.info('E34 prereg SHA256 %s',a.sha(PREREG));a.LOG.info('E34 execution manifest SHA256 %s',a.sha(a.REPO/'experiments/e34_execution_manifest.json'))
    a.savej(root(m)/'e34_metadata.json',dict(model=m,created_utc=a.utc(),preregistration_sha256=a.sha(PREREG),execution_manifest_sha256=a.sha(a.REPO/'experiments/e34_execution_manifest.json'),source_sha256=a.sha(Path(__file__)),rows=plan(),hardware=a.base.hardware_info(),indexing='predictor p=0..2046 -> target p+1=1..2047; BOS-associated first-text prediction, no BOS-target NLL',DC_layer_selection='largest qkv DC share under frozen seed-0 Q1 on all unquantized evaluation tokens'))
    gate_all(m)
    model=a.base.load_model(a.act.MODEL_IDS[m],a.WORK)
    flags=capture_flags_ranges(model,m)
    for spec in plan():
        for seed in a.SEEDS:evaluate(model,m,spec,seed,flags)
    del model;gc.collect();torch.cuda.empty_cache()
    export_tokens(m,flags);verify_frozen(m)
    rr=a.rows(root(m)/'e34_per_sequence.csv');assert len(rr)==len(plan())*192
    a.savej(root(m)/'e34_DONE.json',dict(status='COMPLETE',utc=a.utc(),rows=len(plan()),chunk_records=len(rr),baseline_token_records=3*64*2047,preregistration_sha256=a.sha(PREREG),hardware=a.base.hardware_info()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS);p.add_argument('--freeze',action='store_true');arg=p.parse_args();a.setup()
    if arg.freeze:freeze()
    else:run(arg.model)
if __name__=='__main__':main()
