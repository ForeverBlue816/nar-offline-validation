"""Locate first nonfinite intermediate without changing weights or thresholds."""
import torch
from .common import *
from .cache_adapter import make_cache,clear

def main():
 a=arguments(__doc__).parse_args();initialize()
 import quarot,nar_r4_kernel as nk
 with torch.inference_mode():
  m=build(a.model,'nar');layers=nk.load_factors(factors_path(a.model),torch.device('cuda'))['layers'];rows=[]
  def transform_pre(index):
   def f(mod,inputs):
    x=inputs[0].reshape(-1,mod.n);u=x.float()@layers[index]['y_prime_fp32']
    rows.append({'event':'R4_input','layer':index,'x_finite':bool(torch.isfinite(x).all()),'x_max_abs':float(x.abs().max()) if bool(torch.isfinite(x).all()) else None,'u_max_abs':float(u.abs().max()) if bool(torch.isfinite(u).all()) else None,'u_exceeds_fp16':bool((u.abs()>65504).any())})
   return f
  def output_hook(name):
   def f(mod,inputs,out):
    if isinstance(out,tuple):out=out[0]
    if not torch.is_tensor(out):return
    ok=bool(torch.isfinite(out).all())
    if name.startswith('residual_layer_') or name=='final_norm':rows.append({'event':'residual_or_norm','module':name,'finite':ok,'max_abs':float(out.abs().max()) if ok else None,'nonfinite_count':int((~torch.isfinite(out)).sum())})
    if not ok:
     row={'event':'nonfinite_output','module':name,'type':type(mod).__name__}
     if isinstance(mod,quarot.nn.Linear4bit):
      integer=quarot.matmul(inputs[0].quantized_x,mod.weight)
      row.update(integer_max_abs=int(integer.abs().max()),input_scale_finite=bool(torch.isfinite(inputs[0].scales_x).all()),input_scale_max=float(inputs[0].scales_x.max()) if bool(torch.isfinite(inputs[0].scales_x).all()) else None)
     rows.append(row)
   return f
  for i,l in enumerate(m.model.layers):
   l.mlp.down_proj[0].register_forward_pre_hook(transform_pre(i))
   l.register_forward_hook(output_hook('residual_layer_'+str(i)))
  m.model.norm.register_forward_hook(output_hook('final_norm'))
  for name,mod in m.named_modules():
   if isinstance(mod,(nk.NARDownTransform,quarot.nn.Linear4bit)) or name=='lm_head':mod.register_forward_hook(output_hook(name))
  text=WORK/'cache/tokenized'/f'{MODELS[a.model]}-wikitext2-test-full-l2048.pt'
  if not text.exists():text=WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt'
  ids=torch.load(text,weights_only=True).reshape(-1)[:128].cuda().int().view(1,-1)
  cache=make_cache(m,1,256,page_size=64);clear(cache);out=m(ids,past_key_values=cache)
  result={'model':a.model,'method':'nar','environment':environment(),'logits_finite':bool(torch.isfinite(out.logits).all()),'events':rows,'first_nonfinite':next((r for r in rows if r['event']=='nonfinite_output'),None)}
  write(a.run/'correctness'/f'activation_diagnose_{a.model}.json',result)
  print(result['first_nonfinite'],flush=True)
if __name__=='__main__':main()
