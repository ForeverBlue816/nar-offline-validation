"""Freeze protocol and immutable source provenance before measurement."""
from .common import *

def main():
    p=arguments(__doc__);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    import shutil
    for source in (ROOT/'quarot-llama3/e28_v2/docs').glob('*.md'):
        if not (a.run/source.name).exists():shutil.copyfile(source,a.run/source.name)
    protocol={
      'schema':2,'run_id':a.run.name,'frozen_at':now(),'priority':['P0','P1','P2'],
      'models':MODELS,'methods':METHODS,'rank':8,'weights':'random-weight performance benchmark',
      'shared_random_state':'Old FP16 parameter initialization; INT4 packed bytes uniform 1..6 (upper nibble zero), scales uniform .001..003; name-hashed CPU generator per Linear4bit; byte hashes checked for every common state tensor.',
      'seed':0,'hardware':{'name':'NVIDIA A40','capability':[8,6],'same_physical_gpu':'one Slurm allocation for every formal phase and all three sessions','cpus':4,'torch_threads':4,'interop_threads':1},
      'environment':'existing e28-env torch2.4.1+cu124, triton3.0.0; no dependency upgrade',
      'core':{'prefill':{'batch':[1,16],'length':2048},'decode':{'batch':1,'prefix':2048,'steps':128,'discard':8},'warmup':10,'runs':50,'sessions':3},
      'method_order':[['fp16','hadamard','nar'],['hadamard','nar','fp16'],['nar','fp16','hadamard']],
      'modes':['eager_sequence','eager_step_sync'],'prefill_mode':'eager_sequence',
      'shared_implementation_changes':['Name-stable INT4 initialization','Rebuild stale RoPE cos/sin after installing Llama3 inverse frequencies, at original capacity','Equal-length unmasked KV metadata shared across layers; attention/GEMM/quantization unchanged'],
      'legacy_diagnostic':{'description':'Original per-layer KV wrapper and per-step synchronization; v2 corrected RoPE/state; one diagnostic sample, not a core speedup denominator','warmup':1,'runs':1},
      'decode_reset':'zero all pages/scales; length=0; all needs_init=True; invalidate metadata; recompute complete prefix outside timer; compare full storage hashes across reset',
      'timing':'sequence: sync immediately before/after 120 causal calls; step_sync: sync before/after each of 120 calls; first8 excluded; fixed token100, no sampling',
      'events':'one separate complete run after primary timing and memory capture; report CUDA event elapsed, not pure arithmetic',
      'thresholds':{'r4_relative_l2_max':.002,'r4_actual_quantizer_code_match_min':.999,'fold_relative_l2_max':2e-5,'model_logits_relative_l2_max':.002,'cache_bytes_metadata_exact':True,'gemm_relative_l2_max':.002,'gqa_relative_l2_max':.002,'near_parity_fraction':.01,'retention_min_hadamard_speedup':1.05},
      'r4_shapes':[1,2048,32768],'r4_cases':['real','zero','constant','rounding_boundary','common_offset_small_residual','outlier'],
      'real_activation_policy':'E27 matching model/down/layer BF16 replay, all8192 recorded rows; T32768 repeats these unchanged four times to exercise actual launch shape; not32768 independent samples',
      'selected_configs':{m:selection(m) for m in MODELS},
      'microbenchmark':{'layer':0,'warmup':10,'runs':50,'sessions':3,'fixed_shuffle_b':{'block_t':8,'num_warps':4,'num_stages':2},'fixed_e17_b':{'block_t':8,'num_warps':4,'num_stages':2},'ablation':['generic_vs_prebound_identical_kernel','shuffle_vs_tc_same_fp16_io'],'scope':['E28 slot','E28 frontend','E17 native']},
      'graph':'bounded full-one-step capture using existing backend only; dynamic GPU position/page metadata; validate A-B-A at steps1,2,9,64,128 and page boundary; no timing if invalid',
      'text_windows':'First256 tokens of the first two cached WikiText2 test2048 windows; prefix128 plus128 teacher-forced causal tokens; random weights checks implementation only',
      'extensions':[{'batch':8,'prefix':2048,'steps':128,'warmup':10,'runs':50,'sessions':1},{'batch':1,'prefix':8192,'steps':128,'warmup':10,'runs':50,'sessions':1}],
      'statistics':{'dispersion':'population std of individual samples; throughput std computed after per-sample inversion; session median/std separately; pooled150 runs are3 hardware sessions','prefill_speedup':'method throughput / FP16 throughput','decode_speedup':'FP16 latency / method latency','nar_latency_overhead_pct':'100*(nar/had-1)','nar_throughput_ratio_pct':'100*nar/had','memory_saving_pct':'100*(1-nar/fp16)','retention':'100*(speedup_nar-1)/(speedup_had-1), only when every session had>1.05, else null+reason'},
      'memory':'independent phase processes; original RoPE capacity; load/warmed/inference allocated+reserved, process nvidia-smi separate; unique storage bytes; factors FP32 source released; shared H128/partial once; allocator outside tensors explicitly unattributed; GB=1e9',
      'exclusions':['No deleting slow runs','Nonfinite forwards/reset mismatches invalid, raw diagnostics retained','Unverified kernels excluded from valid rankings','Unavailable data null+reason','No graph results before growing-cache validation','OOM not rescued by changing one row workload'],
      'scope_limits':['No new GEMM/backend/training/calibration','No complete PPL/task sweep','No wide autotuning','No edits to E17/E22/E26 histories or results/e28/report_e28.md']}
    if (a.run/'protocol.json').exists():
        old=read(a.run/'protocol.json');protocol=old
    else:write(a.run/'protocol.json',protocol)
    paths=list((ROOT/'quarot-llama3').glob('*.py'))+list((ROOT/'quarot-llama3/e28_v2').glob('*.py'))+list((ROOT/'nar').glob('*.py'))+list((ROOT/'nar/kernels').glob('*.py'))
    ext=WORK/'external/quarot'
    paths+=list((ext/'quarot').rglob('*.py'))+list((ext/'quarot/kernels').rglob('*.cu'))+list((ext/'quarot/kernels').rglob('*.cpp'))+list((ext/'quarot/kernels/include').rglob('*.h'))+list((ext/'quarot/kernels/include').rglob('*.cuh'))+[ext/'setup.py',ROOT/'quarot-llama3/build_env.sh',ROOT/'quarot-llama3/quarot_gqa_decode.patch']
    paths+=list((ext/'quarot').glob('*.so'))+list((WORK/'e28-env/lib/python3.11/site-packages').glob('*hadamard*.so'))
    import e28_bench as e
    for m in MODELS:paths += [factors_path(m),e.config_path(WORK,e.MODELS[m]['hf']),ROOT/'results/e28'/m/'nar_kernel_selection.json']
    manifest={'captured_at':now(),'head':command(['git','-C',str(ROOT),'rev-parse','HEAD']),'status':command(['git','-C',str(ROOT),'status','--short']),
      'files':{str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(set(paths)) if f.is_file()},
      'quarot_commit':command(['git','-C',str(ext),'rev-parse','HEAD']),'quarot_submodules':command(['git','-C',str(ext),'submodule','status']),
      'compiled_parameters_source':str(ext/'setup.py'),'extension_flags_note':'Existing binary SHA256 recorded; setup/build scripts and submodule commits recorded. Historical shell compiler environment cannot be reconstructed beyond old env.txt.'}
    # Never replace a manifest that already identified executed code.
    dest=a.run/'source_manifest.json'
    if not dest.exists():write(dest,manifest)
    (a.run/'source_dirty.diff').write_text(command(['git','-C',str(ROOT),'diff','HEAD']))
    (a.run/'quarot_dirty.diff').write_text(command(['git','-C',str(ext),'diff','HEAD']))
    (a.run/'scope.md').write_text('''# E28-v2 frozen scope\n\nP0 first: audit, numerical checks, three-session eager timing, independent memory attribution. P1: bounded graph feasibility and two fixed kernel ablations. P2: only decode b8/p2048 and b1/p8192, plus real-checkpoint interface audit.\n\nAll rows are random-weight performance experiments. A cached real-text input does not turn random weights into a quality evaluation. No full PPL/task evaluation is authorized by this plan.\n\nThe old reports/results are immutable. The v2 comparison uses common corrected RoPE caches, common shared KV metadata, and explicitly identical shared INT4 tensors. Therefore old and new times are not interchangeable. A separate legacy-wrapper diagnostic retains old synchronization behavior.\n\nThresholds, selected configurations, counts, exclusions and formulas were saved before formal timing in protocol.json. Pressure failures are retained without relaxing thresholds. No throughput target controls selection.\n''')
    audits=[
      ('INT4 prefill faster than FP16','原始记录支持','Old50-run raw prefill records; one session only, random weights; new measurement pending.'),
      ('INT4 decode acceleration','当前不支持','Old speedup Had3B0.61,Had8B0.65, both below1.'),
      ('93%/101% acceleration retained','当前不支持','Negative/negative ratios are not acceleration retention; new collector emits null.'),
      ('NAR vs Had stable1% lead','仍是推测','Only one original session, no balanced order.'),
      ('CPU launch bound / shuffle compute bound','仍是推测','Kernel sums vs wall do not isolate CPU compute or provide utilization counters.'),
      ('Original standalone decode peak memory','当前不支持','Same process previously warmed b16; shared workspace persists.'),
      ('96d bytes of factors per layer','原始记录支持','FP16Y high/low with rank padded16 =64d; FP16W high/low rank8=32d; H12832768 bytes shared separately.'),
      ('All actual R4 launch shapes validated','当前不支持','Old all-layer check used T256; actual measuredT1/2048/32768 require checks.'),
      ('Original code-match uses actual quantizer','当前不支持','Old proxy uses FP32 division and epsilon; actual CUDA uses FP16 division and has no zero guard.'),
      ('Correct Llama3 RoPE after inv_freq install','当前不支持','Transformers4.36 builds cos/sin in constructor; old install only replaces inv_freq. v2 rebuilds cache uniformly.'),
      ('Shared seed proves equal INT4 weights','当前不支持','NAR reconstructs MLPs, consuming RNG again; v2 resets by name and hashes tensors.'),
      ('Paper native group128 asymmetric deployment','当前不支持','Current GEMM has one scale per row/column and current KV quantizes channels per token with no residual window.')]
    write(a.run/'old_claim_audit.json',{'claims':[{'claim':c,'status':s,'evidence':e} for c,s,e in audits], 'original_files':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'report_e28.md',ROOT/'results/e28/env.txt',ROOT/'results/e28/table.tex',ROOT/'results/e28/e17v3_a40_summary.json']+list((ROOT/'results/e28').glob('*/metrics*.json'))}})
    cp=[]
    for m,key in MODELS.items():
      root=WORK/'artifacts/e14'/key
      cp.append({'model':m,'path':str(root),'available_files':[{'path':str(f),'bytes':f.stat().st_size} for f in root.glob('*/*') if f.is_file()],
        'complete_manifests':[str(f) for f in root.glob('*/DONE.json')], 'status':'BLOCKED','reason':'No complete local k8 checkpoint; existing E14 loader is fake quantization and incompatible with current integer GEMM/KV contract.'})
    write(a.run/'checkpoint_audit.json',{'rows':cp})
    print('FROZEN',a.run,sha(a.run/'protocol.json'),flush=True)

if __name__=='__main__':main()
