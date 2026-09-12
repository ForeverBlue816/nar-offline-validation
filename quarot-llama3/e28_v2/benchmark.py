"""One independent model/method/mode/phase process; no profiler in measurements."""
from __future__ import annotations
import gc, time
from .common import *

def finite_check(model,ids,cache):
    import torch
    from .cache_adapter import clear
    checks=[]; handles=[]
    def hook(name):
        def check(mod,inputs,out):
            values=[out] if torch.is_tensor(out) else [getattr(out,'scales_x',None)]
            for v in values:
                if torch.is_tensor(v) and v.is_floating_point():
                    checks.append({'module':name,'finite':bool(torch.isfinite(v).all()),'max_abs':float(v.abs().max()) if bool(torch.isfinite(v).all()) else None})
        return check
    for name,mod in model.named_modules():
        if mod.__class__.__name__ in ('Linear4bit','Quantizer','RMSNorm','QuarotLlamaMLP','QuarotNARLlamaMLP','QuarotLlamaAttention','NARDownTransform') or name=='lm_head':
            handles.append(mod.register_forward_hook(hook(name)))
    clear(cache)
    try:
        out=model(ids,past_key_values=cache)
        finite=bool(torch.isfinite(out.logits).all()) and all(c['finite'] for c in checks)
        return {'status':'PASS' if finite else 'FAIL','checks':checks,'output_finite':bool(torch.isfinite(out.logits).all())}
    finally:
        for h in handles:h.remove()

def main():
    p=arguments(__doc__)
    p.add_argument('--phase',choices=['prefill1','prefill16','decode','decode8','decode8192'],required=True)
    p.add_argument('--mode',choices=['eager_sequence','eager_step_sync','legacy_diagnostic'],default='eager_sequence')
    p.add_argument('--session',type=int,required=True)
    a=p.parse_args()
    key=f'{a.model}_{a.method}_{a.mode}_{a.phase}_s{a.session}'
    dest=a.run/'raw_runs'/f'{key}.json'
    if dest.exists():print('EXISTS',dest,flush=True);return
    import torch
    from .cache_adapter import make_cache,clear,snapshot
    initialize(); protocol=read(a.run/'protocol.json')
    env=environment(); result={'key':key,'model':a.model,'method':a.method,'mode':a.mode,'phase':a.phase,'session':a.session,'started':now(),'environment':env,'protocol_sha256':sha(a.run/'protocol.json'),'source_manifest_sha256':sha(a.run/'source_manifest.json'),'status':'RUNNING'}
    write(dest,result)
    try:
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats();m=build(a.model,a.method)
            result['loaded']=mempoint();result['shared_state']=state_hashes(m)
            batch=16 if a.phase=='prefill16' else 8 if a.phase=='decode8' else 1
            prefix=8192 if a.phase=='decode8192' else 2048
            decoding=a.phase.startswith('decode'); capacity=prefix+128 if decoding else prefix
            g=torch.Generator(device='cpu').manual_seed(0)
            ids=torch.randint(100,200,(batch,prefix),generator=g,dtype=torch.int32).cuda()
            token=torch.full((batch,1),100,dtype=torch.int32,device='cuda')
            cache=make_cache(m,batch,capacity,legacy=a.mode=='legacy_diagnostic')
            result.update(batch=batch,prefix=prefix,capacity=capacity,input_sha256=tensor_hash(ids),decode_token=100)
            result['finite']=finite_check(m,ids,cache)
            if result['finite']['status']!='PASS':
                result.update(status='INVALID',reason='Nonfinite model outputs/intermediates before timing');return
            # Full content and all flags are reset; rebuild the prefix outside timing.
            def prepare():
                clear(cache)
                if decoding:m(ids,past_key_values=cache)
            # Stabilize both scripted KV shapes before the exact reset audit.
            if decoding:
                for _ in range(2):
                    prepare()
                    for _ in range(2):m(token,past_key_values=cache)
            prepare(); first=snapshot(cache)
            if decoding:
                for _ in range(2):m(token,past_key_values=cache)
            prepare(); second=snapshot(cache)
            result['reset_validation']={'status':'PASS' if first==second else 'FAIL','first':first,'second':second}
            if first!=second:result.update(status='INVALID',reason='Cache reset is not reproducible');return
            def run(events=False):
                prepare()
                if decoding:
                    for _ in range(8):m(token,past_key_values=cache)
                torch.cuda.synchronize()
                start_event=torch.cuda.Event(enable_timing=True) if events else None
                end_event=torch.cuda.Event(enable_timing=True) if events else None
                if events:start_event.record()
                start=time.perf_counter()
                if not decoding:m(ids,past_key_values=cache)
                elif a.mode in ('eager_step_sync','legacy_diagnostic'):
                    steps=[]
                    for _ in range(120):
                        torch.cuda.synchronize();t=time.perf_counter()
                        m(token,past_key_values=cache)
                        torch.cuda.synchronize();steps.append(time.perf_counter()-t)
                else:
                    # Exactly 120 causal one-token calls; no host observations in loop.
                    for _ in range(120):m(token,past_key_values=cache)
                if events:end_event.record()
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                if decoding and a.mode in ('eager_step_sync','legacy_diagnostic'):elapsed=sum(steps)
                return elapsed, (start_event.elapsed_time(end_event) if events else None)
            warmups=10;runs=50
            if a.mode=='legacy_diagnostic':warmups=1;runs=1
            for _ in range(warmups):run()
            gc.collect();torch.cuda.synchronize()
            result['warmed']=mempoint()
            result['storage']=storage_breakdown(m,cache,{'input_ids':ids,'next_token':token})
            torch.cuda.reset_peak_memory_stats()
            result['samples']=[]
            for i in range(runs):
                stamp=now();elapsed,_=run()
                row={'run':i,'timestamp':stamp,'elapsed_s':elapsed}
                row['ms_per_token']=elapsed*1000/120 if decoding else None
                row['prefill_tokens_per_second']=batch*prefix/elapsed if not decoding else None
                result['samples'].append(row)
                if i%10==0: print(key,'run',i,round(elapsed,5),flush=True)
            result['inference_peak']=mempoint()
            # Events in a separate extra run, outside primary samples/peak region.
            _,event_ms=run(events=True)
            result['cuda_event_elapsed_ms']=event_ms
            result['summary']=stats([r['ms_per_token'] if decoding else r['prefill_tokens_per_second'] for r in result['samples']])
            result['nar_dispatch']=[{'layer':i,'compiled':{str(t):v is not False for t,v in l.mlp.down_proj[0]._compiled.items()}} for i,l in enumerate(m.model.layers)] if a.method=='nar' else []
            result.update(status='PASS',ended=now(),environment_after=environment())
    except Exception as exc:
        import traceback
        result.update(status='FAIL',reason=repr(exc),traceback=traceback.format_exc(),ended=now())
        print(result['traceback'],flush=True)
    finally:write(dest,result)

if __name__=='__main__':main()
