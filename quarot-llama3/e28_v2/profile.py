"""Separate CPU/CUDA timeline; profiled times never enter deployment samples."""
from .common import *

def main():
    p=arguments(__doc__);a=p.parse_args();dest=a.run/'profiles'/f'{a.model}_{a.method}'
    if dest.with_suffix('.json').exists():return
    import torch,time
    from .cache_adapter import make_cache,clear
    initialize();out={'started':now(),'environment':environment()}
    try:
      with torch.inference_mode():
        m=build(a.model,a.method);cache=make_cache(m,1,2176)
        inp=torch.full((1,2048),100,device='cuda',dtype=torch.int32);token=inp[:,:1].contiguous()
        clear(cache);m(inp,past_key_values=cache)
        for _ in range(8):m(token,past_key_values=cache)
        torch.cuda.synchronize()
        dest.parent.mkdir(parents=True,exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            for _ in range(4):
                with torch.profiler.record_function('e28_v2_causal_decode_step'):m(token,past_key_values=cache)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(dest.with_suffix('.trace.json')))
        trace=read(dest.with_suffix('.trace.json'))
        kernels=[e for e in trace.get('traceEvents',[]) if e.get('cat')=='kernel' and 'dur' in e]
        spans=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
        merged=[]
        for left,right in spans:
            if merged and left<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],right)
            else:merged.append([left,right])
        busy=sum(right-left for left,right in merged)
        out['timeline_summary']={'kernel_count':len(kernels),'kernel_sum_ms_per_step':sum(e['dur'] for e in kernels)/4000 if kernels else None,'kernel_union_busy_ms':busy/1000 if kernels else None,'device_kernel_span_ms':(max(b for _,b in spans)-spans[0][0])/1000 if spans else None,'inter_kernel_gap_ms':((max(b for _,b in spans)-spans[0][0])-busy)/1000 if spans else None,'gap_interpretation':'Includes host submission, copies/synchronization/profiler effects; not CPU arithmetic time.','hardware_counters':{'value':None,'reason':'No Nsight hardware counters collected; no measured DRAM or compute utilization claim.'}}
        by_kernel={}
        for event in kernels:
            item=by_kernel.setdefault(event['name'],{'name':event['name'],'calls':0,'total_us':0.})
            item['calls']+=1;item['total_us']+=event['dur']
        out['kernels_by_total_duration']=sorted(by_kernel.values(),key=lambda e:e['total_us'],reverse=True)
        runtime={}
        for event in trace.get('traceEvents',[]):
            if event.get('cat') in ('cuda_runtime','cuda_driver') and 'dur' in event:
                item=runtime.setdefault(event['name'],{'calls':0,'total_host_us':0.})
                item['calls']+=1;item['total_host_us']+=event['dur']
        out['cuda_api_activity']=runtime
        events=[]
        for e in prof.key_averages():
            events.append({'name':e.key,'device_type':e.device_type.name,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total,'cpu_memory_bytes':e.cpu_memory_usage,'device_memory_bytes':e.device_memory_usage})
        out['allocation_conversion_diagnostics']=[e for e in events if any(word in e['name'] for word in ('aten::empty','aten::_to_copy','aten::copy_','aten::contiguous'))]
        out.update(status='PASS',steps=4,events=events,kernel_sum_ms_per_step=out['timeline_summary']['kernel_sum_ms_per_step'],
           warning='Sum of profiler kernel durations is distinct from wall time and CUDA event elapsed; overlap/idle/launch/profiler overhead prevent treating the difference as CPU compute.')
    except Exception as exc:
        import traceback
        out.update(status='BLOCKED',reason=repr(exc),traceback=traceback.format_exc())
    out['ended']=now();write(dest.with_suffix('.json'),out)
if __name__=='__main__':main()
