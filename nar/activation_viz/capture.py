"""Capture real floating inputs; run on one GPU, no calibration or training."""
from __future__ import annotations
import argparse, gc, hashlib, inspect, json, os, subprocess, sys
from pathlib import Path
import torch
from nar import e19_qwen3_e2e as e19, e14_w4a4kv4 as e14, experiment as base

METHODS = ('unrotated', 'hadamard', 'nar_k8', 'nar_kmax')
SITES = {'down_proj': 'mlp.down_proj', 'q_proj': 'self_attn.q_proj'}
LAYERS = [0, 12, 23, 35]

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()

def tensor_hash(x): return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def norm_fuse(model):
    # Exact norm-affine part of E14, without any rotation. Q/K head norms untouched.
    for b in model.model.layers:
        for norm, consumers in ((b.input_layernorm,[b.self_attn.q_proj,b.self_attn.k_proj,b.self_attn.v_proj]),
                                (b.post_attention_layernorm,[b.mlp.gate_proj,b.mlp.up_proj])):
            for module in consumers: module.weight.mul_(norm.weight.unsqueeze(0))
            norm.weight.fill_(1)
    model.lm_head.weight.mul_(model.model.norm.weight.unsqueeze(0))
    model.model.norm.weight.fill_(1)

def forward(model, ids):
    hidden=[]
    handle=model.model.norm.register_forward_hook(lambda module,inputs,output:hidden.append(output.detach().cpu().clone()))
    try:
        logits=model(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids,device='cuda'),
                     use_cache=False, logits_to_keep=1).logits.detach().cpu()
    finally:handle.remove()
    # All final hidden positions determine all logits; materialize the last-token head only.
    return torch.cat((hidden[0].flatten(),logits.flatten()))

class Observer(e14.RuntimeHooks):
    """Capture at the entry of the real quantizer, after registered rotation hooks."""
    def __init__(self, *args, targets, sink, **kwargs):
        super().__init__(*args,**kwargs)
        self.targets=targets; self.sink=sink; self.enabled=False
        self.kv_counts={'attention':0,'key_qdq':0,'value_qdq':0}
    def quantize_input(self,module,inputs):
        if self.enabled and id(module) in self.targets:
            self.sink(self.targets[id(module)],inputs[0])
        return super().quantize_input(module,inputs)
    def close(self):
        super().close()
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self.attention_key, None)
        ALL_ATTENTION_FUNCTIONS._local_mapping.pop(self.attention_key, None)
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(self.attention_key, None)
        ALL_MASK_ATTENTION_FUNCTIONS._local_mapping.pop(self.attention_key, None)

    def attention(self,*args,**kwargs):
        # Count actual QDQ invocations made within the original attention body.
        original_key=e14._kivi_key_qdq; original_quant=base.dynamic_asym_int4
        def key(*a,**kw):
            self.kv_counts['key_qdq']+=1
            return original_key(*a,**kw)
        def quant(x,g):
            # Only the V call here has last axis=head_dim; K is transposed to 32.
            if g==self.rotations.head_dim: self.kv_counts['value_qdq']+=1
            return original_quant(x,g)
        e14._kivi_key_qdq=key; base.dynamic_asym_int4=quant
        self.kv_counts['attention']+=1
        try: return super().attention(*args,**kwargs)
        finally: e14._kivi_key_qdq=original_key; base.dynamic_asym_int4=original_quant

@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    work=Path(args.workdir); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    e19.install_extension_hooks()
    import transformers, numpy as np
    from transformers import AutoTokenizer
    all_ids=e19.eval_tokens(work,2048)
    indices=torch.randperm(len(all_ids),generator=torch.Generator().manual_seed(42))[:8]
    ids=all_ids[indices].clone()
    tokenizer=AutoTokenizer.from_pretrained(e19.MODEL_ID,cache_dir=str(work/'cache/huggingface'),local_files_only=True)
    torch.save({'input_ids':ids,'attention_mask':torch.ones_like(ids),'sample_indices':indices},out/'inputs.pt')
    write(out/'inputs.json',{'sample_indices':indices.tolist(),'input_ids':ids.tolist(),
        'attention_mask':'all ones; no padding','input_ids_sha256':tensor_hash(ids),
        'decoded_text_sha256':[hashlib.sha256(tokenizer.decode(x).encode()).hexdigest() for x in ids],
        'token0_id':int(ids[0,0]),'token0_string':tokenizer.convert_ids_to_tokens(int(ids[0,0])),
        'tokenizer_bos_token_id':tokenizer.bos_token_id,'preprocessing':'nar.experiment.prepare_token_chunks; config BOS fallback prepended to each 2047-token content window'})
    model=e19.load_model_fp32(work)
    audit=e19.architecture_audit(model)
    snapshot=Path(work/'cache/huggingface/models--Qwen--Qwen3-8B-Base/snapshots')
    manifest={'model_id':e19.MODEL_ID,'model_key':e19.MODEL_KEY,'base_revision':getattr(model.config,'_commit_hash',None),'cached_snapshot_revisions':sorted(p.name for p in snapshot.iterdir()),
        'base_checkpoint':str(snapshot),'architecture':audit,'layers_zero_based':LAYERS,'sites':SITES,
        'methods':METHODS,'sample_seed':42,'rotation_seed':0,'samples':8,'seq_len':2048,
        'reference':'BF16 checkpoint values loaded and norm-fused in FP32; q_norm/k_norm remain affine',
        'git_commit':e19.git_commit(),'command':' '.join(sys.argv),'python':sys.version,
        'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
        'gpu':torch.cuda.get_device_name(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'quantizer':{'implementation':'nar.experiment.dynamic_asym_int4','bits':4,'group_size':128,'group_axis':-1,
          'scale':'fp16((max-min)/15), nonpositive or fp16-underflow scale replaced by 1',
          'offset':'fp16(group minimum), real-valued offset; not group mean and not integer zero point',
          'codes':'uint8 container, round-to-nearest-even, clamp [0,15]', 'clipping':'observed min/max; no activation MSE search',
          'weight_protocol':'g128_asym GPTQ','KV':'E14 KIVI: K token-axis g32; V channel-axis g128; residual length 32',
          'compute_dtype':'float32','source_sha256':digest(Path(base.__file__))},
        'rotation_factors':{},'checkpoints':{},'raw_activation_root':str(out/'activations'),
        'display':{'overview_token_bins':128,'overview_channel_bins':256,'token0_separate':True,
          'detail_token_slice':[0,128],'detail_channel_slice':[0,512],'pooling':'max absolute; residual before pooling',
          'elev':25,'azim':-60},'source_files':{}}
    for module in (e14,e19,base,sys.modules[__name__]): manifest['source_files'][Path(module.__file__).name]=digest(module.__file__)
    manifest['qwen_forward_source']=inspect.getsource(type(model.model.layers[0]).forward)
    manifest['mlp_forward_source']=inspect.getsource(type(model.model.layers[0].mlp).forward)
    manifest['attention_forward_source']=inspect.getsource(type(model.model.layers[0].self_attn).forward)
    checks=[]; inventory=[]
    def check(name,value,limit,**kw):
        checks.append(dict(check=name,value=float(value),limit=limit,passed=bool(value<=limit),**kw))
        write(out/'validation_report.json',{'checks':checks,'passed':all(c['passed'] for c in checks)})
        if value>limit: raise AssertionError(checks[-1])
    write(out/'run_manifest.json',manifest)
    reference=forward(model,ids[:1,:128]); norm_fuse(model)
    fused=forward(model,ids[:1,:128])
    check('norm_affine_fusion_relative_hidden_and_last_logit_error',(fused-reference).norm()/reference.norm(),1e-5)
    rotations={m:e19.Qwen3RotationSet(work,e19.MODEL_KEY,m,0,model.config,torch.device('cuda')) for m in METHODS[1:]}
    for m,r in rotations.items():
        files=[]
        if r.r1 is not None:
            root=e14.rotation_dir(work,e19.MODEL_KEY,0)
            files=[root/f'r1_{r.R1_LABEL[m]}.pt']+list(root.glob('r2_v_layer_*.pt'))+list((work/'activations'/e19.MODEL_KEY/'e18v2_factors'/r.R4_LABEL[m]).glob('down_layer_*.pt'))
        manifest['rotation_factors'][m]={'ranks':r.ranks(),'files':[{'path':str(p.relative_to(work)),'sha256':digest(p)} for p in sorted(files)],
            'r4_actual_k':{str(l):int(r.r4[l].active.sum()) if r.r4 else None for l in LAYERS},
            'sign_hashes':{f'{site}:{l}':tensor_hash((torch.ones(12288) if m=='hadamard' and site=='down_proj' else r.signs('r1' if site=='q_proj' else 'r4',0 if site=='q_proj' else l,4096 if site=='q_proj' else 12288).cpu())) for site in SITES for l in LAYERS}}
    for p in (work/'activations'/e19.MODEL_KEY/'e14_rotations/DONE.json',work/'activations'/e19.MODEL_KEY/'e18v2_factors/DONE.json'):
        manifest.setdefault('calibration',{})[str(p.relative_to(work))]=json.loads(p.read_text())
    write(out/'run_manifest.json',manifest)
    current={'sample':0,'mode':'paired_local','method':'unrotated'}
    def save(key,value,method=None,canonical_hash=None):
        layer,site=key; method=method or current['method']
        x=value.detach().float().reshape(-1,value.shape[-1]).cpu().clone()
        assert x.shape[0]==2048 and torch.isfinite(x).all()
        p=out/'activations'/current['mode']/method/f's{current["sample"]:02d}_l{layer:02d}_{site}.pt'
        p.parent.mkdir(parents=True,exist_ok=True); torch.save(x,p)
        inventory.append({'file':str(p.relative_to(out)),'mode':current['mode'],'method':method,'sample':current['sample'],
            'layer':layer,'site':site,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':digest(p),
            'canonical_tensor_sha256':canonical_hash,'tensor_sha256':tensor_hash(x)})
    def paired(key,value):
        layer,site=key; x=value.detach().float(); h=tensor_hash(x.cpu())
        save(key,x,'unrotated',h)
        label='r1' if site=='q_proj' else 'r4'; ri=0 if site=='q_proj' else layer
        for m,r in rotations.items():
            y=r.apply(label,ri,x); save(key,y,m,h)
            a=x.double().square().sum(-1); b=y.double().square().sum(-1)
            check('per_token_energy_relative',((a-b).abs()/a.clamp_min(1e-30)).max(),2e-5,method=m,layer=layer,site=site,sample=current['sample'])
            if current['sample']==0:
                small=x.reshape(-1,x.shape[-1])[:8]
                recovered=e19.apply_transpose(r,label,ri,y.reshape(-1,y.shape[-1])[:8])
                check('rotation_inverse_relative',(recovered-small).norm()/small.norm().clamp_min(1e-30),1e-5,method=m,layer=layer,site=site)
                w=model.model.layers[layer].get_submodule(SITES[site]).weight[:8].float()
                left=small@w.T; right=r.apply(label,ri,small)@r.apply(label,ri,w).T
                check('compensated_linear_relative',(left-right).norm()/left.norm().clamp_min(1e-30),2e-5,method=m,layer=layer,site=site)
                if m.startswith('nar_'):
                    factor=r.r1 if label=='r1' else r.r4[layer]
                    direct=factor.apply(small,r.signs(label,ri,small.shape[-1]))
                    check('WY_vs_unfused_factor_relative',(direct-r.apply(label,ri,small)).norm()/direct.norm().clamp_min(1e-30),1e-5,method=m,layer=layer,site=site)
    # Smoke first: one complete 2048-token sample, one selected layer, no expansion before success.
    no_hooks=forward(model,ids[:1]); handles=[]
    for site,path in SITES.items():
        handles.append(model.model.layers[0].get_submodule(path).register_forward_pre_hook(lambda module,inputs,site=site:paired((0,site),inputs[0])))
    captured=forward(model,ids[:1])
    for h in handles:h.remove()
    check('smoke_capture_bitwise_logit_difference',(captured-no_hooks).abs().max(),0)
    write(out/'smoke_passed.json',{'passed':True,'sample':0,'layer':0,'tokens':2048})
    print('SMOKE PASSED',flush=True)
    # Reuse smoke shards; sample zero remaining layers, then all layers for 7 further samples.
    for sample in range(8):
        current['sample']=sample; handles=[]
        for layer in (LAYERS[1:] if sample==0 else LAYERS):
            for site,path in SITES.items():
                handles.append(model.model.layers[layer].get_submodule(path).register_forward_pre_hook(lambda module,inputs,layer=layer,site=site:paired((layer,site),inputs[0])))
        forward(model,ids[sample:sample+1])
        for h in handles:h.remove()
        print('PAIRED',sample,flush=True)
    write(out/'activation_inventory.json',inventory)
    del model,rotations; gc.collect(); torch.cuda.empty_cache()
    current['mode']='end_to_end'
    for method in METHODS[1:]:
        current['method']=method
        model,r=e14.load_quantized_model(work,work/'artifacts/e22',e19.MODEL_KEY,method,0,128,protocol='g128_asym')
        ckpt=e14.checkpoint_dir(work/'artifacts/e22',e19.MODEL_KEY,method,0,'g128_asym')
        done=json.loads((ckpt/'DONE.json').read_text())
        state=torch.load(ckpt/'layer_00.pt',map_location='cpu',weights_only=True)
        actual_dtype=str(next(iter(state.values())).dtype);del state
        manifest['checkpoints'][method]={'path':str(ckpt),'done':done,'actual_saved_weight_dtype':actual_dtype,
          'layer_files':[{'name':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(ckpt.glob('layer_*.pt'))]}
        targets={id(model.model.layers[l].get_submodule(path)):(l,s) for l in LAYERS for s,path in SITES.items()}
        hooks=Observer(model,r,'asymmetric_g128',True,targets=targets,sink=save); hooks.install()
        off=forward(model,ids[:1]); hooks.kv_counts={k:0 for k in hooks.kv_counts};hooks.enabled=True
        for sample in range(8):
            current['sample']=sample; observed=forward(model,ids[sample:sample+1])
            if sample==0:check('end_to_end_capture_bitwise_logit_difference',(off-observed).abs().max(),0,method=method)
            print('END_TO_END',method,sample,flush=True)
        check('kv_key_calls',abs(hooks.kv_counts['key_qdq']-8*36),0,method=method)
        check('kv_value_calls',abs(hooks.kv_counts['value_qdq']-8*36),0,method=method)
        manifest['checkpoints'][method]['observed_prefill_kv_calls']=hooks.kv_counts
        hooks.close();del hooks,targets,model,r;gc.collect();torch.cuda.empty_cache()
        write(out/'run_manifest.json',manifest);write(out/'activation_inventory.json',inventory)
    manifest['status']='capture_complete';manifest['input_ids_sha256']=tensor_hash(ids)
    write(out/'run_manifest.json',manifest);write(out/'activation_inventory.json',inventory)
    write(out/'capture_complete.json',{'shards':len(inventory),'passed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workdir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if not (Path(args.output)/'capture_complete.json').exists():
        run(args)
    from .metrics import run as metrics_run
    metrics_run(args.output, 'cuda')
