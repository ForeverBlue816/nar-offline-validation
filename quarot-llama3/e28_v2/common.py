"""Provenance and shared state, deliberately independent of historical collectors."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, pathlib, statistics, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path('/projects/nar/nar-validation')
sys.path[:0] = [str(ROOT / 'quarot-llama3'), str(ROOT)]
METHODS = ['fp16', 'hadamard', 'nar']
MODELS = {'3b': 'llama32_3b', '8b': 'llama31_8b'}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(pathlib.Path(path).read_text())
def write(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'); tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def command(cmd):
    try: return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=60).strip()
    except Exception as exc: return 'UNAVAILABLE: '+str(exc)
def stats(x):
    if not x: return {'n':0,'median':None,'mean':None,'std':None,'reason':'no valid measurements'}
    return {'n':len(x),'median':statistics.median(x),'mean':statistics.fmean(x),'std':statistics.pstdev(x),'min':min(x),'max':max(x)}
def arguments(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--run',type=pathlib.Path,required=True)
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--method',choices=METHODS)
    return p

def initialize():
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0)
    name=torch.cuda.get_device_name(); cap=torch.cuda.get_device_capability()
    torch.empty(1,device='cuda')  # establish this process's CUDA context for UUID accounting
    if 'A40' not in name or cap!=(8,6): raise RuntimeError(f'A40 sm86 required, got {name}, {cap}')

def environment():
    import importlib.metadata as md, torch
    versions={}
    for pkg in ['torch','triton','transformers','flash-attn','fast-hadamard-transform','flashinfer','quarot']:
        try: versions[pkg]=md.version(pkg)
        except md.PackageNotFoundError: versions[pkg]=None
    return {'timestamp':now(),'python':sys.version,'versions':versions,'cuda_runtime':torch.version.cuda,
      'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
      'gpu':command(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate,memory.used','--format=csv']),
      'processes':command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
      'throttle':command(['nvidia-smi','-q','-d','PERFORMANCE']),
      'cpu':command(['lscpu']), 'affinity': sorted(os.sched_getaffinity(0)),
      'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),
      'cpu_max':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text() if pathlib.Path('/sys/fs/cgroup/cpu.max').exists() else None,
      'environment':{k:v for k,v in os.environ.items() if k.startswith(('SLURM_','OMP_','MKL_')) or k=='CUDA_VISIBLE_DEVICES'},
      'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
      'nvcc':command(['nvcc','--version']), 'host':command(['hostname']),
      'command':sys.argv,'pid':os.getpid(),'weight_initialization':'legacy_bytes_1_to_6' if os.environ.get('E28_LEGACY_PACKED_BYTES')=='1' else 'uniform_full_signed_int4_bytes_0_to_255',
      'execution_python_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'quarot-llama3/e28_v2').glob('*.py')}}

def tensor_hash(t):
    import torch
    v=t.detach().contiguous().cpu()
    return hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()

def selection(model): return read(ROOT/'results/e28'/model/'nar_kernel_selection.json')['selection']
def factors_path(model): return WORK/'tmp/e28'/f'nar_factors_{MODELS[model]}_k8.pt'

def build(model,method):
    import gc, torch, e28_bench as e, modeling_llama3 as ml, nar_r4_kernel as nk
    config=ml.load_llama3_config(e.config_path(WORK,e.MODELS[model]['hf']))
    factors=nk.load_factors(factors_path(model),torch.device('cuda')) if method=='nar' else None
    m=e.build_model(method,config,factors,selection(model) if factors else None,0)
    # The old constructors consume RNG in different orders when NAR replaces MLPs.
    # Initialize every shared INT4 tensor by its stable fully-qualified name.
    for name,mod in m.named_modules():
        if mod.__class__.__name__=='Linear4bit':
            g=torch.Generator(device='cpu').manual_seed(int(hashlib.sha256(('e28-v2:'+name).encode()).hexdigest()[:15],16))
            lo,hi=(1,7) if os.environ.get('E28_LEGACY_PACKED_BYTES')=='1' else (0,256)
            mod.weight.copy_(torch.randint(lo,hi,mod.weight.shape,generator=g,dtype=torch.uint8))
            mod.weight_scales.copy_((torch.rand(mod.weight_scales.shape,generator=g)*.002+.001).half())
    # E28 changed inv_freq after construction but left the old cos/sin cache.
    # Rebuild at the existing capacity for every method; no footprint shortcut.
    m._e28_rope_fix_report=[]
    for i,layer in enumerate(m.model.layers):
        r=layer.self_attn.rotary_emb
        old=r.cos_cached[2047].clone()
        r._set_cos_sin_cache(r.max_seq_len_cached,r.inv_freq.device,torch.float16)
        m._e28_rope_fix_report.append({'layer':i,'old_vs_corrected_cos_max_abs_at_2047':float((old-r.cos_cached[2047]).abs().max())})
    del factors; gc.collect(); torch.cuda.empty_cache()
    return m

def state_hashes(m):
    shared={}; allowed={}
    for n,t in m.state_dict().items():
        row={'shape':list(t.shape),'dtype':str(t.dtype),'sha256':tensor_hash(t)}
        (allowed if '.mlp.down_proj.0.' in n else shared)[n]=row
    return {'shared':shared,'allowed_r4':allowed,'rule':'Only .mlp.down_proj.0.* may differ between Hadamard and NAR.'}

def mempoint():
    import torch
    return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
      'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
      'nvidia_smi':command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'])}

def storage_breakdown(model,cache=None,extra=None):
    import torch, nar_r4_kernel as nk
    groups={}; seen={}; aliases=[]
    def add(name,t,kind):
        if not torch.is_tensor(t) or t.device.type!='cuda':return
        s=t.untyped_storage(); key=(str(t.device),s.data_ptr(),s.nbytes())
        if key in seen: aliases.append({'name':name,'alias_of':seen[key]}); return
        seen[key]=name
        groups.setdefault(kind,[]).append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'storage_bytes':s.nbytes()})
    for n,t in model.state_dict().items():
        kind=('embedding_head' if 'embed_tokens' in n or 'lm_head' in n else 'h128' if n.endswith('.h128') else 'nar_factors' if '.mlp.down_proj.0.' in n and n.rsplit('.',1)[-1] in ('y_terms','w_hi','w_lo','w_h_t') else 'scales' if 'weight_scales' in n else 'rope' if 'rotary_emb' in n else 'weights_and_fixed_buffers')
        add(n,t,kind)
    # RoPE cos/sin are nonpersistent buffers and absent from state_dict.
    for n,t in model.named_buffers(): add(n,t,'rope' if 'rotary_emb' in n else 'other_buffers')
    for k,t in nk._SHARED_PARTIAL.items():add(str(k),t,'partial_workspace')
    if cache is not None:
        add('cache.pages',cache.pages,'kv_codes_or_fp16');add('cache.scales',cache.scales,'kv_scales_offsets')
        for n,t in vars(cache).items():
            if torch.is_tensor(t):add('cache.'+n,t,'cache_metadata')
        for n,t in getattr(cache,'_spec',{}).items():add('cache.spec.'+n,t,'cache_metadata')
    for n,t in (extra or {}).items():add(n,t,'harness_static_io')
    totals={k:sum(v['storage_bytes'] for v in vals) for k,vals in groups.items()}
    return {'groups':groups,'bytes_by_category':totals,'unique_storage_bytes':sum(totals.values()),'aliases':aliases,'unattributed_allocator_bytes':torch.cuda.memory_allocated()-sum(totals.values()),'units':'bytes; GB=1e9, MB=1e6; tensor views deduplicated by underlying storage'}
