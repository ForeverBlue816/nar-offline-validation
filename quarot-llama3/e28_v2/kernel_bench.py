"""Fixed layer-0 ablations, actual quantizer, direct chain timings; no autotune."""
from .common import *
import time

def timed(fn):
    import torch
    for _ in range(10):fn()
    torch.cuda.synchronize();wall=[];events=[]
    for _ in range(50):
        # One invocation per sample; warm L2 caveat applies, no bandwidth counter.
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); t=time.perf_counter();fn();torch.cuda.synchronize()
        wall.append((time.perf_counter()-t)*1e6)
        start.record();fn();end.record();end.synchronize();events.append(start.elapsed_time(end)*1000)
    return {'wall_us':stats(wall),'cuda_event_elapsed_us':stats(events),'samples_wall_us':wall,'samples_event_elapsed_us':events}

def main():
    p=arguments(__doc__);p.add_argument('--session',type=int,required=True);a=p.parse_args()
    dest=a.run/'raw_runs'/f'kernels_{a.model}_s{a.session}.json'
    if dest.exists():return
    import torch,quarot,nar_r4_kernel as nk
    from nar.kernels import r4_fused_v3 as k3
    from .verify import compare
    initialize();result={'started':now(),'environment':environment(),'model':a.model,'session':a.session,'layer':0,'rows':[]}
    try:
      with torch.inference_mode():
        layer=nk.load_factors(factors_path(a.model),torch.device('cuda'))['layers'][0]
        n=layer['y_prime_fp32'].shape[0]
        for T in [1,2048,32768]:
            torch.manual_seed(31);x=(torch.randn((T,n),device='cuda')*.5).half()
            mod=nk.NARDownTransform(layer['y_prime_fp32'],layer['w_h_t_fp32'],8,selection(a.model)).cuda()
            mod(x);proj,tile=mod.plan(T);partial=torch.empty((T,proj.splits*8),device='cuda',dtype=torch.float32);out=torch.empty_like(x)
            had=quarot.nn.OnlineHadamard(n).cuda();q=quarot.nn.Quantizer()
            def A(): k3.launch_projection_dot(x,mod.y_terms,partial,8,proj,nk.TERMS)
            def B(): nk.launch_kernel_b(x,partial,None,out,8,tile,mod.h128,mod.w_hi,mod.w_lo)
            def generic():A();B();return out
            fast=mod._compiled.get(T)
            def prebound():
                ra,rb,pbuf=fast
                ra(x,mod.y_terms,pbuf);rb(x,pbuf,mod.w_hi,mod.w_lo,mod.h128,out)
                return out
            shuffle=nk.TileConfig(8,4,2,False)
            def Bs():nk.launch_kernel_b(x,partial,layer['w_h_t_fp32'],out,8,shuffle)
            def generic_shuffle():A();Bs();return out
            ref=torch.empty_like(x,dtype=torch.float32)
            for lo in range(0,T,256):ref[lo:lo+256]=nk.reference_transform(x[lo:lo+256],layer['y_prime_fp32'],layer['w_h_t_fp32'])
            valid=compare(mod(x),ref);validshuffle=compare(generic_shuffle(),ref)
            from quarot.functional.hadamard import matmul_hadU
            had_ref=torch.empty_like(ref)
            for lo in range(0,T,256):had_ref[lo:lo+256]=matmul_hadU(x[lo:lo+256].float())
            valid_had=compare(had(x),had_ref)
            del had_ref
            bitwise=bool(fast is not False and torch.equal(prebound().clone(),generic().clone()))
            transformed=mod(x).clone()
            fns=[('E28 slot','hadamard_fp16',lambda:had(x),valid_had),
                 ('E28 slot','nar_module',lambda:mod(x),valid),
                 ('E28 dispatch (preallocated)','nar_prebound',prebound if fast is not False else generic,dict(valid,status=valid['status'] if fast is not False and bitwise else 'FAIL',prebound_bitwise=bitwise)),
                 ('E28 dispatch (preallocated)','nar_generic',generic,dict(valid,prebound_bitwise=bitwise)),
                 ('E28 dispatch (preallocated)','nar_generic_shuffle',generic_shuffle,validshuffle),
                 ('E28 stage','A_only',A,valid),('E28 stage','B_tc_only',B,valid),('E28 stage','B_shuffle_only',Bs,validshuffle),
                 ('E28 stage','shared_quantizer',lambda:q(transformed),valid),
                 ('E28 frontend','hadamard_plus_quantizer',lambda:q(had(x)),valid_had),
                 ('E28 frontend','nar_plus_quantizer',lambda:q(mod(x)),valid)]
            if a.session%2==0:fns.reverse()
            for scope,name,fn,validation in fns:
                status='VALID' if validation['status']=='PASS' and (name!='nar_generic' or bitwise) else 'INVALID'
                result['rows'].append({'tokens':T,'d':n,'rank':8,'scope':scope,'implementation':name,'status':status,'validation':validation,
                  'proj':proj.__dict__,'tile':(shuffle if 'shuffle' in name else tile).__dict__,**timed(fn)})
            # E17-native group128 asymmetric output, same environment and fixed tiles.
            # Separate from the FP16 I/O ablations above; no cuBLAS out_dtype dependency.
            bx=x.to(torch.bfloat16);y=layer['y_prime_fp32'];hi=y.bfloat16();lo=(y-hi.float()).bfloat16();third=(y-hi.float()-lo.float()).bfloat16()
            terms=torch.zeros((3,n,16),device='cuda',dtype=torch.bfloat16)
            terms[0,:,:8]=hi;terms[1,:,:8]=lo;terms[2,:,:8]=third;terms=terms.reshape(3*n,16)
            outputs=k3.allocate_outputs(T,n,torch.device('cuda'));hadout=k3.allocate_outputs(T,n,torch.device('cuda'))
            native_tile=k3.TileConfig(8,4,2)
            def native():
                k3.launch_projection_dot(bx,terms,partial,8,proj,3)
                k3.launch_nar(bx,layer['w_h_t_fp32'],outputs,partial,8,native_tile)
            def native_had():k3.launch_hadamard(bx,hadout,native_tile)
            native();native_had()
            # Native reference uses FP16 rounded scales/real offsets, floor(q+.5).
            def native_valid(packed,reference):
                r=reference.reshape(T,n//128,128);z=r.amin(-1).half();s=((r.amax(-1)-r.amin(-1))/15);s=torch.where(s>0,s,torch.ones_like(s)).half()
                codes=((r-z.float()[...,None])/s.float()[...,None]+.5).floor().clamp(0,15).to(torch.uint8).reshape(T,n)
                unpacked=k3.unpack(packed.codes,n)
                match=float((unpacked==codes).float().mean())
                actual=unpacked.float().reshape(T,n//128,128)*packed.scales.float()[...,None]+packed.zeros.float()[...,None]
                expected=codes.float().reshape(T,n//128,128)*s.float()[...,None]+z.float()[...,None]
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                rel=float((actual-expected).norm()/expected.norm().clamp_min(1e-30)) if finite else None
                scale_rel=float((packed.scales.float()-s.float()).norm()/s.float().norm().clamp_min(1e-30)) if finite else None
                offset_rel=float((packed.zeros.float()-z.float()).norm()/z.float().norm().clamp_min(1e-30)) if finite else None
                return {'status':'PASS' if finite and match>=.999 and rel<=.002 else 'FAIL','code_match_fraction':match,'finite':finite,'matched_dequant_relative_l2':rel,'scale_relative_l2':scale_rel,'offset_relative_l2':offset_rel,'format':'E17 native group128 asymmetric, not QuaRot quantizer'}
            native_ref=torch.empty_like(ref)
            for low in range(0,T,256):native_ref[low:low+256]=nk.reference_transform(bx[low:low+256],layer['y_prime_fp32'],layer['w_h_t_fp32'])
            nv=native_valid(outputs,native_ref)
            zero_y=torch.zeros_like(layer['y_prime_fp32']);zero_w=torch.zeros_like(layer['w_h_t_fp32'])
            for low in range(0,T,256):native_ref[low:low+256]=nk.reference_transform(bx[low:low+256],zero_y,zero_w)
            hv=native_valid(hadout,native_ref)
            for name,fn,validation in [('nar_native',native,nv),('block_hadamard_native',native_had,hv)]:
                result['rows'].append({'tokens':T,'d':n,'rank':8,'scope':'E17 native','input_dtype':'bfloat16','factor_terms':3,'projection':proj.__dict__,'implementation':name,'status':'VALID' if validation['status']=='PASS' else 'INVALID','validation':validation,'tile':native_tile.__dict__,**timed(fn)})
            write(dest,result);print('KERNEL',a.model,T,'session',a.session,flush=True)
        result['status']='COMPLETE'
    except Exception as exc:
        import traceback
        result.update(status='FAIL',reason=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    result['ended']=now();write(dest,result)
if __name__=='__main__':main()
