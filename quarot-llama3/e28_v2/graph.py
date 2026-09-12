"""Bounded one-step graph adapter; all replay metadata updates are timed."""
from .common import *
import time, math, types

def main():
    p=arguments(__doc__);a=p.parse_args();dest=a.run/'correctness'/f'graph_{a.model}_{a.method}.json'
    if dest.exists():return
    import torch
    from .cache_adapter import make_cache,clear,snapshot
    initialize();result={'started':now(),'environment':environment(),'scope':'full one-step model forward, all three methods identical capture boundary','status':'RUNNING','checks':[]}
    try:
      with torch.inference_mode():
        m=build(a.model,a.method)
        latest_hidden={}
        hidden_hook=m.lm_head.register_forward_pre_hook(lambda mod,args:latest_hidden.update(value=args[0].detach().clone()))
        # Indexed RoPE values remain identical, but the view must include future
        # positions; the Python seq_len during capture never advances on replay.
        for layer in m.model.layers:
            def full_cache(self,x,seq_len=None):return self.cos_cached,self.sin_cached
            layer.self_attn.rotary_emb.forward=types.MethodType(full_cache,layer.self_attn.rotary_emb)
        textfile=WORK/'cache/tokenized'/f'{MODELS[a.model]}-wikitext2-test-full-l2048.pt'
        if not textfile.exists():textfile=WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt'
        tokens=torch.load(textfile,weights_only=True).reshape(-1)
        for prefix,page in [(128,64),(2048,2176)]:
            capacity=prefix+128;cache=make_cache(m,1,capacity,page_size=page)
            ids=[tokens[j*4096:j*4096+capacity].cuda().int() for j in range(2)]
            static_token=torch.empty((1,1),device='cuda',dtype=torch.int32)
            pos=torch.empty((1,1),device='cuda',dtype=torch.int64)
            max_pages=math.ceil(capacity/page)
            spec={'kv_data':cache.pages,'kv_param':cache.scales,
              'kv_indices':torch.arange(max_pages,device='cuda',dtype=torch.int32),
              'kv_indptr':torch.zeros(2,device='cuda',dtype=torch.int32),
              'last_page_offset':torch.empty(1,device='cuda',dtype=torch.int32)}
            def metadata(length):
                spec['kv_indptr'][1:].fill_(math.ceil(length/page))
                spec['last_page_offset'].fill_((length-1)%page+1)
                pos.fill_(length-1)
            def reset(window):
                cache.graph_mode=False;clear(cache)
                m(ids[window][:prefix].reshape(1,-1),past_key_values=cache)
                cache._spec=spec;cache.graph_mode=True;cache.length=prefix
                static_token.copy_(ids[window][prefix:prefix+1].reshape(1,1));metadata(prefix+1)
            reset(0)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            prep=time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(3):m(static_token,past_key_values=cache,position_ids=pos)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();reset(0)
            g=torch.cuda.CUDAGraph()
            before=torch.cuda.memory_allocated()
            with torch.cuda.graph(g):out=m(static_token,past_key_values=cache,position_ids=pos)
            captured_hidden=latest_hidden['value']
            torch.cuda.synchronize();result.setdefault('graphs',[]).append({'prefix':prefix,'page_size':page,'prepare_capture_s':time.perf_counter()-prep,'allocated_delta_bytes':torch.cuda.memory_allocated()-before})
            for sequence,window in enumerate([0,1,0]):
                reset(window);reference={}
                cache.graph_mode=False
                for step in range(1,129):
                    o=m(ids[window][prefix+step-1:prefix+step].reshape(1,1),past_key_values=cache)
                    if step in [1,2,9,64,65,128]:reference[step]=(o.logits.detach().clone(),latest_hidden['value'].clone(),snapshot(cache))
                reset(window)
                for step in range(1,129):
                    static_token.copy_(ids[window][prefix+step-1:prefix+step].reshape(1,1));metadata(prefix+step)
                    g.replay();cache.length=prefix+step
                    if step in reference:
                        expected,expected_hidden,cs=reference[step];finite=bool(torch.isfinite(out.logits).all() and torch.isfinite(captured_hidden).all())
                        rel=float((out.logits.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-30)) if finite else None
                        hidden_rel=float((captured_hidden.float()-expected_hidden.float()).norm()/expected_hidden.float().norm().clamp_min(1e-30)) if finite else None
                        # Ignore inactive extra index capacity; compare active metadata.
                        gs=snapshot(cache);gs['metadata']['kv_indices']=gs['metadata']['kv_indices'][:math.ceil(cache.length/page)]
                        same=gs==cs
                        result['checks'].append({'prefix':prefix,'page_size':page,'sequence':sequence,'window':window,'step':step,'finite':finite,'relative_l2':rel,'hidden_relative_l2':hidden_rel,'cache_exact':same,'status':'PASS' if finite and rel<=.002 and hidden_rel<=.002 and same else 'FAIL'})
            if any(x['status']!='PASS' for x in result['checks']):
                result.update(status='INVALID',reason='Growing-cache/A-B-A replay check failed; graph timing excluded');break
            # Only a validated adapter is eligible for an independent graph panel.
            # Main three-session graph timing requires an additional protocol run;
            # this bounded feasibility process records no fake fixed-step speedup.
            del g,out,cache
        else:result['status']='PASS'
    except Exception as exc:
        import traceback
        result.update(status='BLOCKED',reason=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    result['ended']=now();result['timing']={'value':None,'reason':'Feasibility/correctness process only; no unvalidated or unmatched graph speedup.'};write(dest,result)
if __name__=='__main__':main()
