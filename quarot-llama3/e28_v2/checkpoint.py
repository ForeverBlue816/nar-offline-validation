"""Read-only real base-FP16 checkpoint loading; never a paper INT4 adapter."""
from .common import *

def base_files(model):
    repo='models--unsloth--'+('Llama-3.2-3B' if model=='3b' else 'Meta-Llama-3.1-8B')
    root=WORK/'cache/huggingface'/repo
    snapshots=sorted((root/'snapshots').glob('*'))
    if len(snapshots)!=1:raise RuntimeError(f'Expected one frozen base snapshot for {model}, got {snapshots}')
    files=sorted(snapshots[0].glob('*.safetensors'))
    if not files:raise FileNotFoundError(str(snapshots[0]))
    return snapshots[0],files

def load_base(model,instance):
    from safetensors.torch import load_file
    snapshot,files=base_files(model);loaded=[];unexpected=[]
    for f in files:
        state=load_file(str(f),device='cpu')
        result=instance.load_state_dict(state,strict=False);unexpected.extend(result.unexpected_keys);loaded.extend(state)
        del state
    required=set(instance.state_dict())
    missing=required-set(loaded)
    if instance.config.tie_word_embeddings and 'model.embed_tokens.weight' in loaded:missing.discard('lm_head.weight')
    # Nonpersistent RoPE buffers are recomputed by the common corrected setup.
    if missing or unexpected:raise RuntimeError({'missing':sorted(missing),'unexpected':unexpected})
    return {'snapshot':str(snapshot),'files':[str(f) for f in files],'loaded_tensor_count':len(loaded),'dtype':'original BF16 checkpoint converted to FP16 for this backend','native_integer_support':False}

def main():
    a=arguments(__doc__).parse_args();rows=[]
    for model in MODELS:
      snapshot,files=base_files(model)
      rows.append({'model':model,'snapshot':str(snapshot),'files':[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in files],
        'status':'AVAILABLE_FP16_BASE_ONLY','integer_k8_status':'BLOCKED','reason':'A base floating checkpoint is not a compensated/quantized k8 checkpoint or a groupwise integer loader.'})
    write(a.run/'base_checkpoint_manifest.json',{'rows':rows,'timestamp':now()})
if __name__=='__main__':main()
