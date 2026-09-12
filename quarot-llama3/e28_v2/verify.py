"""Frozen operator, all-layer R4, and two-window implementation checks."""
from __future__ import annotations
import math, traceback
from .common import *

def compare(out,ref):
    import torch, quarot
    out=out.float();ref=ref.float()
    finite=bool(torch.isfinite(out).all() and torch.isfinite(ref).all())
    if not finite:return {'status':'FAIL','reason':'nonfinite transform','relative_l2':None,'code_match_fraction':None}
    rel=float((out-ref).norm()/ref.norm().clamp_min(1e-30))
    # Both operands cross the same FP16 I/O contract and actual compiled quantizer.
    q=quarot.nn.Quantizer(); a=q(out.half());b=q(ref.half())
    from quarot.functional.quantization import unpack_i4
    frac=float((unpack_i4(a.quantized_x)==unpack_i4(b.quantized_x)).float().mean())
    return {'status':'PASS' if rel<=.002 and frac>=.999 else 'FAIL','relative_l2':rel,'code_match_fraction':frac,
      'scale_finite':bool(torch.isfinite(a.scales_x).all() and torch.isfinite(b.scales_x).all()),'max_abs':float((out-ref).abs().max())}

def r4(a):
    import torch,nar_r4_kernel as nk
    from nar import activation_experiments as act,e17_v2 as v2
    from nar.fold_signed_permutation import FoldedR4
    payload=nk.load_factors(factors_path(a.model),torch.device('cuda'));n=payload['n'];rows=[]
    for i,layer in enumerate(payload['layers']):
        factorfile=v2.factor_path(WORK,MODELS[a.model],8,i)
        factor=act.RotationFactor.load(factorfile,torch.device('cpu'))
        signs=v2.signs_for(n,i,0,torch.device('cpu'));fold=FoldedR4.from_factor(factor,signs)
        factor_exact=bool(torch.equal(fold.y_prime_fp32,layer['y_prime_fp32'].cpu()) and torch.equal(fold.w_h_t_fp32,layer['w_h_t_fp32'].cpu()))
        datafile=WORK/'activations'/MODELS[a.model]/'e27'/f'down_layer_{i:02d}.bf16'
        raw=torch.from_file(str(datafile),shared=False,size=datafile.stat().st_size//2,dtype=torch.bfloat16).reshape(-1,n)
        real=fold.q_unfolded(raw[:8192]).half()
        direction_old=factor.apply(raw[:8].float(),signs)
        direction_new=fold.apply(fold.q_unfolded(raw[:8].float()))
        direction_rel=float((direction_old-direction_new).norm()/direction_old.norm().clamp_min(1e-30))
        exported_direction=nk.reference_transform(fold.q_unfolded(raw[:8].float()).cuda(),layer['y_prime_fp32'],layer['w_h_t_fp32']).cpu()
        export_rel=float((direction_old-exported_direction).norm()/direction_old.norm().clamp_min(1e-30))
        export_factor_max_abs=max(float((fold.y_prime_fp32-layer['y_prime_fp32'].cpu()).abs().max()),float((fold.w_h_t_fp32-layer['w_h_t_fp32'].cpu()).abs().max()))
        mod=nk.NARDownTransform(layer['y_prime_fp32'],layer['w_h_t_fp32'],8,selection(a.model)).cuda()
        source={'path':str(datafile),'file_sha256':sha(datafile),'rows_available':len(raw),'consumed_rows':8192,'factor_sha256':sha(factorfile),'fold_export_exact':factor_exact,'unfolded_vs_folded_relative_l2':direction_rel,'unfolded_vs_exported_relative_l2':export_rel,'export_factor_max_abs':export_factor_max_abs}
        for T in ([8,8192,16384] if getattr(a,'extended',False) else [1,2048,32768]):
            g=torch.Generator(device='cpu').manual_seed(17+i)
            base=torch.randn((1,n),generator=g).half()
            cases={'real':real.repeat((math.ceil(T/len(real)),1))[:T],
               'zero':torch.zeros((1,n),dtype=torch.float16).expand(T,-1),
               'constant':torch.full((1,n),.5,dtype=torch.float16).expand(T,-1),
               'rounding_boundary':((torch.arange(n).float()%15-7+.5)/7).half()[None,:].expand(T,-1),
               'common_offset_small_residual':(64+base*.03125).expand(T,-1),
               'outlier':base.clone().expand(T,-1).clone()}
            cases['outlier'][:,0]=128
            for name,cpu in cases.items():
                x=cpu.contiguous().cuda();out=mod(x)
                # Chunk only the independent reference, never the tested launch.
                ref=torch.empty_like(out,dtype=torch.float32)
                for lo in range(0,T,256):ref[lo:lo+256]=nk.reference_transform(x[lo:lo+256],layer['y_prime_fp32'],layer['w_h_t_fp32'])
                row={'layer':i,'tokens':T,'case':name,'source':source,'config':{'proj':mod.plan(T)[0].__dict__,'tile':mod.plan(T)[1].__dict__},'repeated_real_rows':name=='real' and T>len(real),**compare(out,ref)}
                if export_rel>2e-5 or direction_rel>2e-5:row.update(status='FAIL',reason='factor export/fold direction mismatch')
                rows.append(row);del x,out,ref
            del cases
        write(a.run/'correctness'/f'r4_{"extended_" if getattr(a,"extended",False) else ""}{a.model}.json',{'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','complete':i+1==len(payload['layers']),'completed_layers':i+1,'rows':rows})
        print('R4',a.model,i,'failures',sum(r['status']=='FAIL' for r in rows),flush=True)
        del mod
    return {'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','complete':True,'completed_layers':len(payload['layers']),'rows':rows}

def operators(a):
    import torch,quarot
    from quarot.functional.quantization import pack_i4,unpack_i4
    from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit,unpack_i4_and_asym_dequantize,matmul_had_cuda
    rows=[]
    vals=torch.arange(-8,8,device='cuda',dtype=torch.int8).repeat(8).reshape(1,128)
    packed=pack_i4(vals);unpacked=unpack_i4(packed)
    rows.append({'test':'signed_int4_nibble_order','status':'PASS' if torch.equal(vals.int(),unpacked) else 'FAIL','first_bytes':packed[0,:8].cpu().tolist()})
    x=torch.tensor([-9,-8,-7.5,-2.5,-1.5,-.5,0,.5,1.5,2.5,7,8],dtype=torch.float16,device='cuda').repeat(32)[:128].reshape(1,-1)
    scale=torch.ones((1,1),dtype=torch.float16,device='cuda')
    q=unpack_i4(quarot.sym_quant(x,scale));expected=x.round().clamp(-8,7).int()
    rows.append({'test':'half_division_nearest_even_saturation','status':'PASS' if torch.equal(q,expected) else 'FAIL','observed':q[0,:12].cpu().tolist()})
    zero=quarot.nn.Quantizer()(torch.zeros((1,128),dtype=torch.float16,device='cuda'))
    rows.append({'test':'zero_row_contract','status':'OBSERVED','scale':zero.scales_x.cpu().tolist(),'codes':torch.unique(unpack_i4(zero.quantized_x)).cpu().tolist(),'note':'Backend has no zero-scale guard; this is an observation, not a safe-zero proof.'})
    for n in [128,8192,14336]:
        for kind in ['random_signed','large_accumulator']:
            gen=torch.Generator(device='cpu').manual_seed(23)
            qa=(torch.randint(-8,8,(16,n),generator=gen,dtype=torch.int8) if kind=='random_signed' else torch.full((16,n),7,dtype=torch.int8)).cuda()
            qw=(torch.randint(-8,8,(32,n),generator=gen,dtype=torch.int8) if kind=='random_signed' else torch.full((32,n),7,dtype=torch.int8)).cuda()
            sx=torch.full((16,1),.01,device='cuda',dtype=torch.float16);sw=torch.full((32,1),.02,device='cuda',dtype=torch.float16)
            integer=quarot.matmul(pack_i4(qa),pack_i4(qw));ref_int=(qa.float()@qw.float().T).int()
            out=quarot.sym_dequant(integer,sx,sw)
            # Exact backend semantic reference includes narrowing the accumulator.
            semantic=(sx*sw.T)*ref_int.half()
            float_ref=(qa.float()*sx.float())@(qw.float()*sw.float()).T
            finite=bool(torch.isfinite(out).all())
            rel=float((out.float()-float_ref).norm()/float_ref.norm()) if finite else None
            rows.append({'test':'int4_gemm_dequant','n':n,'case':kind,'integer_exact':bool(torch.equal(integer,ref_int)),
              'matches_backend_narrowing_semantics':bool(torch.equal(out,semantic)),'finite':finite,'relative_l2_vs_matched_float':rel,
              'max_integer_accumulator':int(integer.abs().max()),'status':'PASS' if finite and rel<=.002 and torch.equal(integer,ref_int) else 'FAIL'})
    # Genuine page boundary, GQA, and same cached (dequantized) K/V reference.
    for fp16 in [True,False]:
      for heads in [24,32]:
        cache=MultiLayerPagedKVCache4Bit(1,16,48,torch.device('cuda'),1,8,128,disable_quant=fp16,hadamard_dtype=None if fp16 else torch.float16)
        cache.pages.zero_();cache.scales.zero_()
        torch.manual_seed(29)
        k=torch.randn((1,15,8,128),device='cuda',dtype=torch.float16)*.2;v=torch.randn_like(k)*.2
        cache.update(k,v,0,{})
        for step in range(1,4):
            k=torch.randn((1,1,8,128),device='cuda',dtype=torch.float16)*.2;v=torch.randn_like(k)*.2
            stub=cache.update(k,v,0,{});q=torch.randn((1,1,heads,128),device='cuda',dtype=torch.float16)*.2
            observed=stub(q=q)
            keys=[];values=[]
            for pos in range(cache.length):
                page,offset=divmod(pos,16)
                for kv,target in [(0,keys),(1,values)]:
                    codes=cache.pages[page,0,kv,:,offset,:]
                    params=cache.scales[page,0,kv,:,offset,:]
                    z=codes if fp16 else unpack_i4_and_asym_dequantize(codes,params[:,0:1],params[:,1:2])
                    target.append(z.float())
            K=torch.stack(keys).repeat_interleave(heads//8,dim=1).transpose(0,1)
            V=torch.stack(values).repeat_interleave(heads//8,dim=1).transpose(0,1)
            Q=(q if fp16 else matmul_had_cuda(q,torch.float16)).reshape(heads,1,128).float()
            ref=(torch.softmax((Q@K.transpose(1,2))/math.sqrt(128),dim=-1)@V).reshape_as(observed)
            rel=float((observed.float()-ref).norm()/ref.norm())
            rows.append({'test':'gqa_paged_decode','cache':'fp16' if fp16 else 'int4','query_heads':heads,'kv_heads':8,'length':cache.length,'crossed_page':step>=2,'relative_l2':rel,'status':'PASS' if rel<=.002 else 'FAIL'})
    return {'status':'PASS' if all(r['status']!='FAIL' for r in rows) else 'FAIL','rows':rows}

def model_check(a):
    import torch
    from .cache_adapter import make_cache,clear,snapshot
    from .benchmark import finite_check
    m=build(a.model,a.method)
    loaded_checkpoint=None;rope_reference=None
    if getattr(a,'real',False):
        from .rope_check import check as check_rope
        rope_reference=check_rope(m)
        write(a.run/'correctness'/f'rope_reference_{a.model}.json',{'started':now(),'environment':environment(),**rope_reference,'ended':now()})
        if a.method!='fp16':raise RuntimeError('No compatible real INT4 checkpoint loader')
        from .checkpoint import load_base
        loaded_checkpoint=load_base(a.model,m)
    state=state_hashes(m)
    latest_hidden={}
    hidden_hook=m.lm_head.register_forward_pre_hook(lambda mod,args:latest_hidden.update(value=args[0].detach().clone()))
    tokenpath=WORK/'cache/tokenized'/f'{MODELS[a.model]}-wikitext2-test-full-l2048.pt'
    if not tokenpath.exists():tokenpath=WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt'
    windows=torch.load(tokenpath,map_location='cpu',weights_only=True)
    if isinstance(windows,dict):windows=windows.get('input_ids',windows.get('tokens'))
    windows=windows.reshape(-1,2048)[:2,:256]
    # Prime both prefill and decode TorchScript specializations before comparing
    # paths. Cold-vs-warm KV packing differs even on the legacy path itself.
    warm=make_cache(m,1,256,legacy=True,page_size=64)
    for _ in range(2):
        clear(warm);m(windows[0,:128].cuda().int().view(1,-1),past_key_values=warm)
        for token in windows[0,128:130]:m(token.cuda().int().view(1,1),past_key_values=warm)
    del warm
    rows=[];prefill_finite_checks=[]
    for wi,tokens in enumerate(windows):
        inp=tokens[:128].view(1,-1).cuda().int();states=[]
        for legacy in [True,False]:
            cache=make_cache(m,1,256,legacy=legacy,page_size=64)
            finite=finite_check(m,inp,cache)
            prefill_finite_checks.append({'window':wi,'legacy_cache':legacy,**finite})
            clear(cache);out=m(inp,past_key_values=cache)
            saved={'prefill':out.logits.detach().cpu(),'prefill_hidden':latest_hidden['value'].cpu(),'finite':finite,'steps':{}}
            for step in range(1,129):
                token=tokens[127+step].view(1,1).cuda().int()
                out=m(token,past_key_values=cache,output_hidden_states=True)
                if step in [1,2,9,64,65,128]:
                    saved['steps'][step]={'logits':out.logits.detach().cpu(),'hidden':latest_hidden['value'].cpu(),'cache':snapshot(cache)}
            states.append(saved);del cache
        old,new=states
        for step in [0,1,2,9,64,65,128]:
            u,v=(old['prefill'],new['prefill']) if step==0 else (old['steps'][step]['logits'],new['steps'][step]['logits'])
            finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all())
            rel=float((u-v).norm()/u.norm().clamp_min(1e-30)) if finite else None
            cache_equal=step==0 or old['steps'][step]['cache']==new['steps'][step]['cache']
            hu,hv=(old['prefill_hidden'],new['prefill_hidden']) if step==0 else (old['steps'][step]['hidden'],new['steps'][step]['hidden'])
            hidden_rel=float((hu.float()-hv.float()).norm()/hu.float().norm().clamp_min(1e-30)) if bool(torch.isfinite(hu).all() and torch.isfinite(hv).all()) else None
            rows.append({'window':wi,'step':step,'finite':finite,'relative_l2':rel,'hidden_relative_l2':hidden_rel,'cache_exact':cache_equal,'cache_before':None if step==0 else old['steps'][step]['cache'],'cache_after':None if step==0 else new['steps'][step]['cache'],'status':'PASS' if finite and rel<=.002 and hidden_rel is not None and hidden_rel<=.002 and cache_equal else 'FAIL'})
    return {'status':'PASS' if all(r['status']=='PASS' for r in rows) and all(r['status']=='PASS' for r in prefill_finite_checks) and (rope_reference is None or rope_reference['status']=='PASS') else 'FAIL','rows':rows,'prefill_finite_checks':prefill_finite_checks,'rope_reference':rope_reference,'shared_state':state,'rope_cache_fix':m._e28_rope_fix_report,
      'weights':'real base checkpoint, FP16 implementation check only' if loaded_checkpoint else 'random-weight implementation check; no model quality claim','loaded_checkpoint':loaded_checkpoint,'text_source':str(tokenpath),'text_sha256':sha(tokenpath),'token_window_hashes':[tensor_hash(t) for t in windows],
      'real_checkpoint':{'status':'BLOCKED','reason':'No complete matching k8 integer checkpoint/loader; paper fake-quantized per-group weights are incompatible with this row/column-scale GEMM.'}}

def main():
    p=arguments(__doc__);p.add_argument('--level',choices=['r4','operators','model'],required=True);p.add_argument('--real',action='store_true');p.add_argument('--extended',action='store_true');a=p.parse_args()
    dest=a.run/'correctness'/f'{a.level}_{"real_" if a.real else "extended_" if a.extended else ""}{a.model or "all"}{"_"+a.method if a.method else ""}.json'
    if dest.exists() and read(dest).get('complete',True):print('EXISTS',dest,flush=True);return
    import torch
    initialize();payload={'started':now(),'environment':environment()}
    try:
        with torch.inference_mode():payload.update({'r4':r4,'operators':operators,'model':model_check}[a.level](a))
    except Exception as exc:payload.update(status='BLOCKED',reason=repr(exc),traceback=traceback.format_exc());print(payload['traceback'],flush=True)
    payload['ended']=now();write(dest,payload)

if __name__=='__main__':main()
