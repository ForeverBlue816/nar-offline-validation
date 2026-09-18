"""Fixed calibration-size/corpus preparation; all observations retained."""
import argparse,os,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
import torch


def prefix_view(m,parent,child):
    """Hard links to immutable raw captures; the reader uses the declared prefix length."""
    folder=a.ASSETS/m/'capture'/child;done=folder/'DONE.json'
    if done.exists():return
    p=a.ASSETS/m/'capture'/parent;meta=a.js(p/'DONE.json')
    child_tokens=a.loadt(a.tokenpath(m,child));parent_tokens=a.loadt(a.tokenpath(m,parent))
    assert torch.equal(child_tokens,parent_tokens[:len(child_tokens)])
    folder.mkdir(parents=True,exist_ok=True)
    for f in p.glob('*.bf16'):
        dst=folder/f.name
        if not dst.exists():os.link(f,dst)
    a.savej(done,dict(meta,sequences=len(child_tokens),token_sha256=a.sha(a.tokenpath(m,child)),capture_source=parent,scope='exact prefix of same unquantized forward activations; no new capture'))


def streamed_spectral(m,label):
    # Same fp64 covariance action as the E32 explicit-matrix default, without
    # storing a dense moment for every calibration-corpus/size variation.
    folder=a.ASSETS/m/'spectral'/label;folder.mkdir(parents=True,exist_ok=True)
    for site,layer,n in a.keys(m):
        target=folder/f'{site}_{layer:02d}.pt'
        if target.exists():continue
        rank=n//128
        gen=torch.Generator().manual_seed(a.base.DEFAULT_SEED+100000*(site=='down')+layer)
        q=torch.linalg.qr(torch.randn((n,rank+16),generator=gen,dtype=torch.float64),mode='reduced').Q.cuda()
        pack={}
        for passes in range(1,4):
            cq=torch.zeros_like(q);trace=0.;count=0
            for x in a.blocks(m,label,site,layer,n):
                x=x.double();cq.addmm_(x.T,x@q);trace+=float(x.square().sum());count+=len(x)
            cq/=count;trace/=count
            small=q.T@cq;val,u=torch.linalg.eigh((small+small.T)/2)
            val=val[-rank:].flip(0);v=q@u[:,-rank:].flip(1);v=torch.linalg.qr(v,mode='reduced').Q
            pack[f'S{passes}']={'vectors':v.cpu(),'eigenvalues':val.cpu(),'trace':trace,'captured_fraction':float(val.sum()/trace)}
            if passes<3:q=torch.linalg.qr(cq,mode='reduced').Q
        a.savet(target,{'model':m,'site':site,'layer':layer,'n':n,'rows':count,'exact_moment':False,'covariance_action_dtype':'float64','solvers':pack})
        a.LOG.info('E30 spectral %s %s layer=%d',label,site,layer)
        del q,cq,pack;gc.collect();torch.cuda.empty_cache()


def prepared(m,label):
    return all(a.factor_path(m,label,'S3','max','P1',0,s,l).exists() for s,l,n in a.keys(m))


def evict_capture(m,label):
    # Only new, fully consumed, replayable intermediate captures are removed.
    assert prepared(m,label)
    folder=a.ASSETS/m/'capture'/label
    meta=a.js(folder/'DONE.json')
    files=list(folder.glob('*.bf16'))
    meta['cache_eviction']={'utc':a.utc(),'files':len(files),'logical_bytes':sum(p.stat().st_size for p in files),'reason':'all planned S3 factors and solver diagnostics persisted; deterministic token replay retained'}
    a.savej(folder/'DONE.json',meta)
    for p in files:p.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1,2]);args=p.parse_args()
    a.setup();m='qwen3_4b_base';a.prepare_tokens(m)
    from nar.reviewer_layerwise import run as layerwise
    for seed in (a.SEEDS if args.seed is None else [args.seed]):
        for label in [f'wt2_n{n}_s{seed}' for n in [64,128,256]]+[f'c4_n128_s{seed}']:
            if not prepared(m,label):layerwise(m,label,False)
            checkpoint=a.ASSETS/m/'layerwise'/label/'hidden_checkpoint.pt'
            if label!='wt2_n128_s0' and checkpoint.exists() and prepared(m,label):
                checkpoint.unlink()
    name='e30_preparation.json' if args.seed is None else f'e30_preparation_seed{args.seed}.json'
    a.savej(a.REPO/'results'/m/name,{'status':'COMPLETE','created_utc':a.utc(),'corpora':['wt2','c4'],'optional_nonweb':'unavailable at preregistration','seed0_default_source':'shared exact-moment S3 calibration prepared before E29'})
if __name__=='__main__':main()
