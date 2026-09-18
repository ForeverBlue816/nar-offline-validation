"""Shared, resumable E29--E33 execution. Existing experiment assets are read-only."""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, logging, math, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nar import activation_experiments as act, experiment as base
from nar.e18_v2 import full_hadamard_rows_transpose

LOG = logging.getLogger('nar')
REPO = Path(__file__).resolve().parent.parent
WORK = Path('/projects/nar/nar-validation')
ASSETS = Path(os.environ.get('ABLATION_ASSETS', '/home/yanlongc/e29-e33-assets'))
SEEDS = (0,1,2)
LENGTH, GROUP = 2048,128

def utc(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text())
def savej(p,d): base.atomic_json(Path(p),d)
def savet(p,d): base.atomic_torch_save(Path(p),d)
def loadt(p): return torch.load(p,map_location='cpu',weights_only=True)
def rows(p): return base.read_csv(p) if Path(p).exists() else []
def write(p,rs):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rs:return
    fields=list(dict.fromkeys(k for r in rs for k in r))
    temp=p.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
    os.replace(temp,p)
def append(p,new,key):
    old=rows(p); known={tuple(str(r[k]) for k in key):r for r in old}
    for r in new:
        kk=tuple(str(r[k]) for k in key)
        if kk in known:
            for k,v in r.items():
                if str(known[kk].get(k,''))!=str(v):raise RuntimeError(f'Conflicting append {p}: {kk} {k}')
        else:old.append(r);known[kk]=r
    write(p,old)
def config(m):
    from transformers import AutoConfig
    c=AutoConfig.from_pretrained(act.MODEL_IDS[m],cache_dir=str(WORK/'cache/huggingface'),local_files_only=True)
    return c,{'qkv':c.hidden_size,'down':c.intermediate_size}
def keys(m):
    c,ds=config(m)
    return [(s,l,n) for s,n in ds.items() for l in range(c.num_hidden_layers)]
def tokenpath(m,label):return ASSETS/m/'tokens'/f'{label}.pt'
def prepare_tokens(m):
    """Only new asset paths are written; E20/E27 evaluation tensors are reused exactly."""
    done=ASSETS/m/'tokens/DONE.json'
    if done.exists():return
    from transformers import AutoTokenizer
    from datasets import Dataset
    mid=act.MODEL_IDS[m]; key=base.model_key_from_id(mid)
    frozen={}
    for split,n,label in [('test',64,'wt2_eval'),('train',128,'wt2_n128_s0')]:
        p=WORK/'cache/tokenized'/f'{key}-{split}-o0-n{n}-l2048.pt'
        if not p.exists() and split=='test':
            candidates=sorted((WORK/'cache/tokenized').glob(f'{key}-test-o0-n*-l2048.pt'))
            candidates=[z for z in candidates if len(loadt(z))>=n]
            if candidates:p=candidates[0]
        if not p.exists(): raise RuntimeError(f'Missing frozen token cache: {p}')
        x=loadt(p)[:n];assert tuple(x.shape)==(n,LENGTH)
        savet(tokenpath(m,label),x); frozen[label]={'file':str(p.relative_to(WORK)),'sha256':sha(p),'tensor_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
    if m=='qwen3_4b_base':
        tok=AutoTokenizer.from_pretrained(mid,cache_dir=str(WORK/'cache/huggingface'),local_files_only=True,use_fast=True)
        arrow=list((WORK/'cache/datasets/Salesforce___wikitext').glob('**/wikitext-train.arrow'))
        if not arrow: raise RuntimeError('No cached WikiText-2 training Arrow file')
        # Require the raw-v1 cache, never a different text preprocessing configuration.
        arrow=[p for p in arrow if 'wikitext-2-raw-v1' in str(p)]
        ds=Dataset.from_file(str(sorted(arrow)[0])); text='\n\n'.join(r['text'] for r in ds if r['text'].strip())
        ids=tok(text,add_special_tokens=False,return_attention_mask=False)['input_ids']
        c,_=config(m); bos=tok.bos_token_id if tok.bos_token_id is not None else c.bos_token_id
        total=len(ids)//(LENGTH-1)
        chunks=torch.tensor(ids[:total*(LENGTH-1)],dtype=torch.long).reshape(total,LENGTH-1)
        pool=torch.cat([torch.full((total,1),bos),chunks],1)
        assert torch.equal(pool[:128],loadt(tokenpath(m,'wt2_n128_s0'))), 'Frozen WT2 subset mismatch'
        assignment={}
        for seed in SEEDS:
            ix=np.arange(total) if seed==0 else np.random.default_rng(seed).permutation(total)
            assignment[str(seed)]=ix[:256].tolist()
            for n in [64,128,256]:
                p=tokenpath(m,f'wt2_n{n}_s{seed}')
                v=pool[torch.tensor(ix[:n])]
                if p.exists():assert torch.equal(loadt(p),v)
                else:savet(p,v)
        frozen['wt2_nested_indices']=assignment
        # Whole documents split evaluation from calibration; no shared boundary document.
        from nar import e22_qwen3_family as e22
        e22.WORKDIR=WORK
        iterator=enumerate(e22.c4_validation_rows()); specs={}
        for label,count in [('c4_eval',64),('c4_pool',384)]:
            ids=[];doc_ids=[]
            for doc_id,item in iterator:
                ids.extend(tok(item['text']+'\n\n',add_special_tokens=False)['input_ids']);doc_ids.append(doc_id)
                if len(ids)>=count*(LENGTH-1):break
            assert len(ids)>=count*(LENGTH-1)
            x=torch.tensor(ids[:count*(LENGTH-1)],dtype=torch.long).reshape(count,LENGTH-1)
            x=torch.cat([torch.full((count,1),bos),x],1)
            savet(tokenpath(m,label),x); specs[label]={'document_ids':doc_ids,'discarded_tail_tokens':len(ids)-count*(LENGTH-1)}
        assert not set(specs['c4_eval']['document_ids']) & set(specs['c4_pool']['document_ids'])
        cp=loadt(tokenpath(m,'c4_pool'))
        for seed in SEEDS:
            ix=np.random.default_rng(seed).permutation(len(cp))[:128]
            savet(tokenpath(m,f'c4_n128_s{seed}'),cp[torch.tensor(ix)])
            specs[f'c4_s{seed}']={'pool_indices':ix.tolist()}
        ev={hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in loadt(tokenpath(m,'c4_eval'))}
        assert not ev & {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in cp}
        frozen['c4_disjoint_documents']=specs
    savej(done,{'created_utc':utc(),'model':m,'sources':frozen,'token_files':{p.name:sha(p) for p in done.parent.glob('*.pt')}})

class Capture:
    def __init__(self,m,folder,ntokens,ns):
        self.folder=folder;self.at=0;self.handles=[];self.maps={}
        c,ds=config(m)
        for s,n in ds.items():
            for l in range(c.num_hidden_layers):
                self.maps[s,l]=np.memmap(folder/f'{s}_{l:02d}.bf16',mode='w+',dtype=np.uint16,shape=(ns,ntokens,n))
    def consume(self,s,l,v):
        assert v.dtype==torch.bfloat16
        self.maps[s,l][self.at:self.at+len(v)]=v.detach().cpu().contiguous().view(torch.uint16).numpy()
    def install(self,model):
        for l,b in enumerate(model.model.layers):
            self.handles.append(b.input_layernorm.register_forward_hook(lambda mod,inp,out,l=l:self.consume('qkv',l,out)))
            self.handles.append(b.mlp.down_proj.register_forward_pre_hook(lambda mod,inp,l=l:self.consume('down',l,inp[0])))
    def close(self):
        for h in self.handles:h.remove()
        for x in self.maps.values():x.flush()

def capture(m,label):
    folder=ASSETS/m/'capture'/label;done=folder/'DONE.json'
    if done.exists():
        assert js(done)['token_sha256']==sha(tokenpath(m,label));return
    folder.mkdir(parents=True,exist_ok=True)
    tokens=loadt(tokenpath(m,label));model=base.load_model(act.MODEL_IDS[m],WORK)
    cap=Capture(m,folder,LENGTH,len(tokens));cap.install(model)
    try:
        with torch.inference_mode():
            for i in range(len(tokens)):
                cap.at=i;model.model(input_ids=tokens[i:i+1].cuda(),use_cache=False)
                if i%8==0:LOG.info('capture %s %s %d/%d',m,label,i+1,len(tokens))
    finally:cap.close()
    savej(done,{'created_utc':utc(),'sequences':len(tokens),'length':LENGTH,'token_sha256':sha(tokenpath(m,label)),'hardware':base.hardware_info()})
    del model,cap;gc.collect();torch.cuda.empty_cache()

def blocks(m,label,s,l,n,stride=1,batch=4096,device='cuda'):
    folder=ASSETS/m/'capture'/label;meta=js(folder/'DONE.json')
    mm=np.memmap(folder/f'{s}_{l:02d}.bf16',mode='r',dtype=np.uint16,shape=(meta['sequences'],LENGTH,n))
    if stride==1:
        a=mm.reshape(-1,n)
        for i in range(0,len(a),batch):yield torch.from_numpy(np.array(a[i:i+batch],copy=True)).view(torch.bfloat16).to(device).float()
    else:
        # Preserve original stride-32, separately within every 2048-token sequence.
        for i in range(meta['sequences']):yield torch.from_numpy(np.array(mm[i,::stride],copy=True)).view(torch.bfloat16).to(device).float()

def eigenspaces(m,label,exact=False):
    folder=ASSETS/m/'spectral'/label;folder.mkdir(parents=True,exist_ok=True)
    for s,l,n in keys(m):
        target=folder/f'{s}_{l:02d}.pt'
        if target.exists():continue
        rank=n//GROUP;width=rank+16
        gen=torch.Generator().manual_seed(base.DEFAULT_SEED+100000*(s=='down')+l)
        q=torch.linalg.qr(torch.randn((n,width),generator=gen,dtype=torch.float64),mode='reduced').Q.cuda()
        pack={}; count=0;trace=0.;sigma=None
        if exact:
            sigma=torch.zeros((n,n),dtype=torch.float64,device='cuda')
            for x in blocks(m,label,s,l,n):
                xd=x.double();sigma.addmm_(xd.T,xd);count+=len(x)
            sigma/=count;trace=float(sigma.trace());sigma=(sigma+sigma.T)/2
        for passes in range(1,4):
            if exact:cq=sigma@q
            else:
                cq=torch.zeros_like(q);trace=0.;count=0
                for x in blocks(m,label,s,l,n):
                    # fp32 covariance action + fp64 accumulation matches existing calibration arithmetic.
                    cq+=(x.T@(x@q.float())).double();trace+=float(x.square().sum(dtype=torch.float64));count+=len(x)
                cq/=count;trace/=count
            small=q.T@cq;value,u=torch.linalg.eigh((small+small.T)/2)
            value=value[-rank:].flip(0);u=u[:,-rank:].flip(1);v=q@u
            v=torch.linalg.qr(v,mode='reduced').Q
            pack[f'S{passes}']={'vectors':v.cpu(),'eigenvalues':value.cpu(),'trace':trace}
            if passes<3:q=torch.linalg.qr(cq,mode='reduced').Q
        if exact:
            value,v=torch.linalg.eigh(sigma);value=value[-rank:].flip(0);v=v[:,-rank:].flip(1)
            pack['S4']={'vectors':v.cpu(),'eigenvalues':value.cpu(),'trace':trace}
            for name,item in pack.items():
                vv=item['vectors'].cuda();sv=sigma@vv;lam=(vv*sv).sum(0)
                resid=(sv-vv*lam).norm(dim=0)/lam.abs().clamp_min(1e-30)
                angles=torch.rad2deg(torch.acos(torch.linalg.svdvals(v.T@vv).clamp(0,1)))
                item.update(ritz_residuals=resid.cpu(),principal_angles_deg=angles.cpu(),captured_fraction=float(lam.sum()/trace),eigenvalues=lam.cpu())
                if name=='S4':assert float(resid.max())<=1e-10,(m,s,l,float(resid.max()))
            # Save explicit fp64 moment so all row diagnostics use the same matrix.
            savet(folder/f'{s}_{l:02d}_sigma.pt',sigma.cpu())
        savet(target,{'model':m,'site':s,'layer':l,'n':n,'rows':count,'exact_moment':exact,'solvers':pack})
        LOG.info('spectral %s %s %s layer=%d/%d exact=%s',m,label,s,l+1,config(m)[0].num_hidden_layers,exact)
        del pack,q,cq,sigma;gc.collect();torch.cuda.empty_cache()

def wy64(v):
    """Stable Householders onto fixed spaced anchors, and their compact product."""
    v=torch.linalg.qr(v.double(),mode='reduced').Q
    n,k=v.shape;w=[];y=[];work=v.clone()
    for i in range(k):
        delta=work[:,i].clone();delta[i*GROUP]-=1
        norm=delta.norm()
        if float(norm)<1e-14:continue
        ref=delta/norm;work[:,i:]-=2*ref[:,None]*(ref@work[:,i:])[None,:]
        wi=2*ref if not w else 2*(ref-torch.stack(w,1)@(torch.stack(y,1).T@ref))
        w.append(wi);y.append(ref)
    w=torch.stack(w,1);y=torch.stack(y,1)
    return v,w,y

def orders(energy,n,k,variant,seed):
    groups=n//GROUP;anchors=[i*GROUP for i in range(k)];aset=set(anchors)
    if variant=='P2':return torch.arange(n),torch.arange(n)
    rest=[i for i in range(n) if i not in aset]
    if variant=='P3':
        perm=np.random.default_rng(seed).permutation(rest).tolist()
        return torch.tensor(anchors+perm),torch.tensor(anchors+rest)
    filler=sorted(rest)[:groups-k] if variant=='P4' else sorted(rest,key=lambda i:(float(energy[i]),i))[:groups-k]
    remaining=[i for i in rest if i not in set(filler)]
    remaining.sort(key=lambda i:(-float(energy[i]),i))
    target=[g*GROUP for g in range(groups)]+base._balanced_target_slots([float(energy[i]) for i in remaining],groups,GROUP)
    return torch.tensor(anchors+filler+remaining),torch.tensor(target)

def factor_path(m,label,solver,rank,variant,seed,s,l):
    # Sign-only seeds share P1/P2/P4/P5; P3 has its own permutation seed.
    return ASSETS/m/'factors'/label/solver/f'k{rank}'/f'{variant}_s{seed if variant=="P3" else 0}'/f'{s}_{l:02d}.pt'

def build_factors(m,label,solver='S3',rank='max',variant='P1',seed=0):
    for s,l,n in keys(m):
        dest=factor_path(m,label,solver,rank,variant,seed,s,l)
        if dest.exists():continue
        spectral=loadt(ASSETS/m/'spectral'/label/f'{s}_{l:02d}.pt')
        spec=spectral['solvers'][solver];k=n//GROUP if rank=='max' else min(int(rank),n//GROUP)
        v,w,y=wy64(spec['vectors'][:,:k].cuda())
        energy=torch.zeros(n,dtype=torch.float64,device='cuda');count=0
        for x in blocks(m,label,s,l,n,stride=32):
            x=x.double();gx=x-(x@w)@y.T;energy+=gx.square().sum(0);count+=len(x)
        energy=(energy/count).cpu()
        source,target=orders(energy,n,k,variant,base.DEFAULT_SEED+seed+1000*l+(100000 if s=='down' else 0))
        if rank=='1':source=target=torch.arange(n) # one full-width DC target at coordinate zero
        saved={'n':n,'k':k,'block':n if rank=='1' else GROUP,'w':w.float().cpu(),'y':y.float().cpu(),'vectors':v.cpu(),
               'source':source,'target':target,'energy_after_g':energy,'solver':solver,'label':label,'variant':variant,
               'calibration_fraction':float(spec.get('captured_fraction',spec['eigenvalues'][:k].sum()/spec['trace'])) if k==n//GROUP else float(spec['eigenvalues'][:k].sum()/spec['trace'])}
        savet(dest,saved)
    LOG.info('factors complete %s %s %s k%s %s seed%d',m,label,solver,rank,variant,seed)

_DC_SIGNS = {}

def dc_hadamard_rows(x,transpose=False):
    """Normalize the Paley product's first column to DC for the rank-one PQ row.

    Paley-II order 76 is orthogonal but its first column is signed. A fixed
    output sign normalization gives H_DC e0 = ones/sqrt(n); the actual
    transpose reverses this normalization. The frozen Hadamard baseline
    retains its original matrix. No learned quantity enters this correction.
    """
    n=x.shape[-1];key=(n,str(x.device),x.dtype)
    if key not in _DC_SIGNS:
        unit=torch.zeros((1,n),device=x.device,dtype=x.dtype);unit[0,0]=1
        _DC_SIGNS[key]=act.full_hadamard_rows(unit,torch.ones(n,device=x.device,dtype=x.dtype)).sign()[0]
    ones=torch.ones(n,device=x.device,dtype=x.dtype);normalization=_DC_SIGNS[key]
    if transpose:return full_hadamard_rows_transpose(x*normalization,ones)
    return act.full_hadamard_rows(x,ones)*normalization

class Rotation:
    def __init__(self,m,seed,method='pq',label='wt2_n128_s0',solver='S3',rank='max',variant='P1'):
        self.m=m;self.method=method;self.data={};self.signs={};self.variant=variant
        for s,l,n in keys(m):
            gen=torch.Generator().manual_seed(act._seed(base.DEFAULT_SEED,seed,l,s))
            self.signs[s,l]=(torch.ones(n) if variant=='P5' else torch.randint(0,2,(n,),generator=gen).float()*2-1).cuda()
            if method=='pq':
                d=loadt(factor_path(m,label,solver,rank,variant,seed,s,l))
                self.data[s,l]={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in d.items()}
    def apply(self,s,l,x,transpose=False):
        shape=x.shape;x=x.float().reshape(-1,shape[-1]);sign=self.signs[s,l]
        if self.method=='hadamard':
            f=full_hadamard_rows_transpose if transpose else act.full_hadamard_rows
            return f(x,sign).reshape(shape)
        d=self.data[s,l];n=d['n'];b=d['block']
        if transpose:
            x=dc_hadamard_rows(x,True) if b==n else act.ext._fast_walsh_hadamard(x.reshape(-1,n//b,b)).reshape(-1,n)
            x=x*sign;z=torch.empty_like(x);z[:,d['source']]=x[:,d['target']]
            out=z-(z@d['y'])@d['w'].T
        else:
            x=x-(x@d['w'])@d['y'].T;z=torch.empty_like(x);z[:,d['target']]=x[:,d['source']]
            z=z*sign
            out=dc_hadamard_rows(z) if b==n else act.ext._fast_walsh_hadamard(z.reshape(-1,n//b,b)).reshape(-1,n)
        return out.reshape(shape)
    def gate(self):
        out=[]
        for s,l,n in keys(self.m):
            gen=torch.Generator().manual_seed(7000+l)
            x=torch.randn((8,n),generator=gen).cuda();r=self.apply(s,l,x);inv=self.apply(s,l,r,True)
            rt=float((inv-x).norm()/x.norm());anchor=0.
            if self.method=='pq':
                d=self.data[s,l];v=d['vectors'].T.float();r=self.apply(s,l,v)
                target=torch.zeros_like(r)
                if d['block']==n:target[:]=1/math.sqrt(n)
                else:
                    for i in range(d['k']):target[i,i*GROUP:(i+1)*GROUP]=1/math.sqrt(GROUP)
                anchor=float(torch.minimum((r-target).norm(dim=1),(r+target).norm(dim=1)).max())
            out.append({'site':s,'layer':l,'round_trip':rt,'anchor_residual':anchor})
            assert rt<=1e-6 and anchor<=1e-6, out[-1]
        return out

def quantize(x,group,symmetric):
    if not symmetric:return base.dynamic_asym_int4(x,group)[0]
    g=x.float().reshape(*x.shape[:-1],-1,group)
    scale=(g.abs().amax(-1,keepdim=True)/7).half();scale=torch.where(scale>0,scale,torch.ones_like(scale)).float()
    return (torch.round(g/scale).clamp(-8,7)*scale).reshape_as(x)

class Hooks:
    def __init__(self,model,rotation,group,symmetric):self.model=model;self.rot=rotation;self.group=group;self.symmetric=symmetric;self.handles=[]
    def transform(self,s,l,v):
        z=self.rot.apply(s,l,v);g=z.shape[-1] if self.group=='token' else int(self.group)
        return self.rot.apply(s,l,quantize(z,g,self.symmetric),True).to(v.dtype)
    def install(self):
        for l,b in enumerate(self.model.model.layers):
            self.handles.append(b.input_layernorm.register_forward_hook(lambda mod,inp,out,l=l:self.transform('qkv',l,out)))
            self.handles.append(b.mlp.down_proj.register_forward_pre_hook(lambda mod,inp,l=l:(self.transform('down',l,inp[0]),)))
    def close(self):
        for h in self.handles:h.remove()

@torch.inference_mode()
def evaluate_row(model,m,exp,row,seed,rotation,group=128,symmetric=False,evalset='wt2'):
    dest=REPO/'results'/m/f'e{exp}_per_sequence.csv'
    known={int(r['chunk']) for r in rows(dest) if r['row']==row and int(r['seed'])==seed and r['eval_set']==evalset}
    if known==set(range(64)):return
    audit=rotation.gate()
    append(REPO/'results'/m/f'e{exp}_gates.csv',[dict(experiment=exp,model=m,row=row,seed=seed,**r) for r in audit],['model','row','seed','site','layer'])
    tok=loadt(tokenpath(m,f'{evalset}_eval'));assert tuple(tok.shape)==(64,LENGTH)
    hooks=Hooks(model,rotation,group,symmetric);hooks.install()
    try:
        for i in range(64):
            if i in known:continue
            batch=tok[i:i+1].cuda();logits=model(input_ids=batch,use_cache=False).logits
            loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1))
            assert loss.dtype==torch.float32 and bool(loss.isfinite())
            append(dest,[dict(experiment=exp,model=m,row=row,seed=seed,eval_set=evalset,chunk=i,nll=float(loss),tokens=2047,token_sha256=sha(tokenpath(m,f'{evalset}_eval')))],['model','row','seed','eval_set','chunk'])
            if i%16==0:LOG.info('E%d %s %s s%d %s chunk%d/64',exp,m,row,seed,evalset,i+1)
            del logits,loss
    finally:hooks.close()

def setup():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert (REPO/'experiments/e29_e33_preregistration.md').exists()
    ASSETS.mkdir(parents=True,exist_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tokens','capture','spectral','factors']);p.add_argument('--model',required=True);p.add_argument('--label',default='wt2_n128_s0');p.add_argument('--exact',action='store_true');p.add_argument('--solver',default='S3');p.add_argument('--rank',default='max');p.add_argument('--variant',default='P1');p.add_argument('--seed',type=int,default=0);a=p.parse_args();setup()
    if a.stage=='tokens':prepare_tokens(a.model)
    elif a.stage=='capture':capture(a.model,a.label)
    elif a.stage=='spectral':eigenspaces(a.model,a.label,a.exact)
    elif a.stage=='factors':build_factors(a.model,a.label,a.solver,a.rank,a.variant,a.seed)
if __name__=='__main__':main()
