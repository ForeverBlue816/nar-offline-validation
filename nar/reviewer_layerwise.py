"""Lossless layer-at-a-time calibration; same model operations, bounded activation storage."""
from __future__ import annotations
import argparse,gc,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar.prepare_e30 import streamed_spectral


def digest_tensor(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def specs(model,exact):
    if not exact:return [('S3','max','P1',0)]
    result=[(s,'max','P1',0) for s in ['S1','S2','S3','S4']]+[('S3','1','P1',0)]
    if model=='qwen3_4b_base':
        for rank in ['8','max']:
            for variant in ['P1','P2','P3','P4','P5']:
                for seed in (range(3) if variant=='P3' else [0]):result.append(('S3',rank,variant,seed))
    return list(dict.fromkeys(result))


@torch.inference_mode()
def run(m,label,exact,smoke=False):
    a.prepare_tokens(m)
    tokens=a.loadt(a.tokenpath(m,label))[:2] if smoke else a.loadt(a.tokenpath(m,label))
    state_dir=a.ASSETS/m/'layerwise'/label
    if not smoke and (state_dir/'DONE.json').exists():return
    model=a.base.load_model(a.act.MODEL_IDS[m],a.WORK);layers=model.model.layers
    assert not getattr(model.model,'has_sliding_layers',False),'model needs per-layer mask dispatch'
    expected={};sample=0;handles=[];first={}
    def capture_kwargs(mod,args,kw):
        if not first:first.update(kw)
    handles.append(layers[0].register_forward_pre_hook(capture_kwargs,with_kwargs=True))
    for l,b in enumerate(layers):
        handles.append(b.input_layernorm.register_forward_hook(lambda mod,inp,out,l=l:expected.__setitem__((l,'qkv',sample),digest_tensor(out))))
        handles.append(b.mlp.down_proj.register_forward_pre_hook(lambda mod,inp,l=l:expected.__setitem__((l,'down',sample),digest_tensor(inp[0]))))
    reference=[]
    for sample in range(2):reference.append(model.model(input_ids=tokens[sample:sample+1].cuda(),use_cache=False).last_hidden_state.cpu())
    for h in handles:h.remove()
    assert first.get('past_key_values') is None and first.get('use_cache') is False
    hidden=torch.cat([model.model.embed_tokens(t[None].cuda()).cpu() for t in tokens],0)
    begin=0;state_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=state_dir/'hidden_checkpoint.pt';meta=state_dir/'checkpoint.json'
    if not smoke and checkpoint.exists() and meta.exists():
        cc=a.loadt(checkpoint);begin=cc['next_layer'];hidden=cc['hidden']
        assert cc['token_sha256']==a.sha(a.tokenpath(m,label))
    audit=[]
    old_keys,old_blocks=a.keys,a.blocks
    try:
        for l in range(begin,len(layers)):
            block=layers[l];buffers={};out=torch.empty_like(hidden);at=0
            need=not smoke and any(not a.factor_path(m,label,solver,rank,variant,seed,s,l).exists() for s in ['qkv','down'] for solver,rank,variant,seed in specs(m,exact))
            def consume(site,value):
                if at<2:
                    got=digest_tensor(value);assert got==expected[l,site,at],(m,l,site,at,'layerwise mismatch')
                    audit.append(dict(layer=l,site=site,sequence=at,sha256=got,bit_identical=True))
                if need:
                    if site not in buffers:buffers[site]=torch.empty((len(tokens),a.LENGTH,value.shape[-1]),dtype=torch.bfloat16)
                    buffers[site][at:at+1]=value.detach().cpu()
            hs=[block.input_layernorm.register_forward_hook(lambda mod,inp,val:consume('qkv',val)),
                block.mlp.down_proj.register_forward_pre_hook(lambda mod,inp:consume('down',inp[0]))]
            try:
                for at in range(len(tokens)):out[at:at+1]=block(hidden[at:at+1].cuda(),**first).cpu()
            finally:
                for h in hs:h.remove()
            hidden=out
            if need:
                for site,data in buffers.items():
                    n=data.shape[-1]
                    def ram_blocks(model,label_arg,s,layer,width,stride=1,batch=4096,device='cuda'):
                        assert (s,layer,width)==(site,l,n)
                        if stride==1:
                            flat=data.reshape(-1,n)
                            for i in range(0,len(flat),batch):yield flat[i:i+batch].to(device).float()
                        else:
                            for i in range(len(data)):yield data[i,::stride].to(device).float()
                    a.keys=lambda model:[(site,l,n)];a.blocks=ram_blocks
                    if exact:a.eigenspaces(m,label,True)
                    else:streamed_spectral(m,label)
                    for solver,rank,variant,seed in specs(m,exact):a.build_factors(m,label,solver,rank,variant,seed)
                    a.keys=old_keys;a.blocks=old_blocks
            del buffers;gc.collect();torch.cuda.empty_cache()
            a.LOG.info('layerwise %s %s layer=%d/%d bit-identical=True',m,label,l+1,len(layers))
            if not smoke and ((l+1)%4==0 or l+1==len(layers)):
                a.savet(checkpoint,{'next_layer':l+1,'hidden':hidden,'token_sha256':a.sha(a.tokenpath(m,label))})
                a.savej(meta,{'next_layer':l+1,'utc':a.utc()})
        for i in range(2):
            final=model.model.norm(hidden[i:i+1].cuda()).cpu()
            assert torch.equal(final,reference[i]),(m,i,'final hidden mismatch')
        target=(a.REPO/'experiments'/f'e29_e33_{m}_layerwise_smoke.json') if smoke else state_dir/'DONE.json'
        a.savej(target,{'status':'PASS','model':m,'label':label,'sequences':len(tokens),'whole_model_reference_sequences':2,'bit_identical_sites':audit,'final_hidden_bit_identical':True,'created_utc':a.utc(),'hardware':a.base.hardware_info()})
    finally:
        a.keys=old_keys;a.blocks=old_blocks
        del model;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--label',default='wt2_n128_s0');p.add_argument('--exact',action='store_true');p.add_argument('--smoke',action='store_true');arg=p.parse_args();a.setup();run(arg.model,arg.label,arg.exact,arg.smoke)
if __name__=='__main__':main()
