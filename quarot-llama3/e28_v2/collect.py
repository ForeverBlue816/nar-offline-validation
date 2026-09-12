"""Regenerate JSON, four tables and paper text without repeating GPU work."""
from .common import *
import collections

def fmt(v):
    if v is None:return 'N/A'
    if 0<abs(v)<.01:return f'{v:.2e}'
    return f'{v:.2f}'
def escaped(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def table(path,headers,rows,caption):
    path.parent.mkdir(parents=True,exist_ok=True)
    head=' & '.join(escaped(h) for h in headers)+r' \\'
    if len(rows)>28:
        lines=['{\\small','\\begin{longtable}{'+'l'*len(headers)+'}','\\caption{'+escaped(caption)+r'} \\','\\toprule',head,'\\midrule','\\endfirsthead','\\toprule',head,'\\midrule','\\endhead']
        lines += [' & '.join(escaped(x) for x in r)+r' \\' for r in rows]
        lines += ['\\bottomrule','\\end{longtable}','}']
    else:
        lines=['\\begin{table}[t]','\\centering','\\small','\\caption{'+escaped(caption)+'}','\\begin{tabular}{'+'l'*len(headers)+'}','\\toprule',head,'\\midrule']
        lines += [' & '.join(escaped(x) for x in r)+r' \\' for r in rows]
        lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    path.with_suffix('.tex').write_text('\n'.join(lines)+'\n')
    path.with_suffix('.md').write_text(caption+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n')

def main():
    p=arguments(__doc__);a=p.parse_args();out=a.run.resolve()
    raw=[read(f) for f in (out/'raw_runs').glob('*.json')]
    checks=[];uuids=set();environments=set();input_hashes=collections.defaultdict(set)
    core_sources={name:set() for name in ['common.py','cache_adapter.py','benchmark.py']}
    for record in raw:
        if record.get('samples') and record.get('status')=='PASS':
            env=record['environment'];pid=str(env['pid'])
            for name,hashes in core_sources.items():hashes.add(env.get('execution_python_sha256',{}).get('quarot-llama3/e28_v2/'+name))
            for line in env.get('processes','').splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)>1 and fields[1]==pid:uuids.add(fields[0])
            environments.add(json.dumps({k:env.get(k) for k in ['versions','capability','cuda_runtime','affinity','torch_threads','torch_interop_threads']},sort_keys=True))
            input_hashes[(record['model'],record['phase'])].add(record['input_sha256'])
            modelcheck=out/'correctness'/f'model_{record["model"]}_{record["method"]}.json'
            reference=read(modelcheck).get('shared_state') if modelcheck.exists() else None
            checks.append({'key':record['key'],'shared_state_matches_verified_model':bool(reference and record.get('shared_state')==reference)})
    same_core_sources=all(len(v)==1 and None not in v for v in core_sources.values())
    write(out/'protocol_conformance.json',{'core_source_hashes':{k:sorted(v,key=str) for k,v in core_sources.items()},'same_core_measurement_sources':same_core_sources,'gpu_uuids':sorted(uuids),'same_gpu':len(uuids)==1 if uuids else None,'environment_variants':len(environments),'same_inputs':all(len(v)==1 for v in input_hashes.values()),'state_checks':checks,'status':'PASS' if same_core_sources and checks and len(uuids)==1 and len(environments)==1 and all(c['shared_state_matches_verified_model'] for c in checks) and all(len(v)==1 for v in input_hashes.values()) else 'PENDING_OR_REVIEW_REQUIRED'})
    groups=collections.defaultdict(list)
    for r in raw:
        if r.get('status')=='PASS' and r.get('samples') and r.get('mode')!='legacy_diagnostic':groups[(r['model'],r['method'],r['mode'],r['phase'])].append(r)
    aggregate=[];lookup={}
    for key,records in sorted(groups.items()):
        records.sort(key=lambda r:r['session']);model,method,mode,phase=key;decode=phase.startswith('decode')
        values=[s['ms_per_token'] if decode else s['prefill_tokens_per_second'] for r in records for s in r['samples']]
        per_session=[dict(session=r['session'],**r['summary']) for r in records]
        entry={'model':model,'method':method,'mode':mode,'phase':phase,'summary':stats(values),'session_summaries':per_session,'session_median_dispersion':stats([s['median'] for s in per_session]),'independent_sessions':len(records),'required_sessions':1 if phase in ['decode8','decode8192'] else 3,
          'peak_allocated_bytes':max(r['inference_peak']['peak_allocated'] for r in records),'peak_reserved_bytes':max(r['inference_peak']['peak_reserved'] for r in records),'source_keys':[r['key'] for r in records]}
        entry['status']='COMPLETE' if len(records)==entry['required_sessions'] and all(len(r['samples'])==50 for r in records) else 'PROVISIONAL'
        aggregate.append(entry);lookup[key]=entry
    def get(model,method,mode,phase):return lookup.get((model,method,mode,phase))
    def med(r):return r['summary']['median'] if r else None
    comparisons=[]
    for model in MODELS:
      for mode in ['eager_sequence','eager_step_sync','cuda_graph_sequence']:
       for phase in ['prefill1','prefill16','decode','decode8','decode8192']:
        rows={method:get(model,method,mode,phase) for method in METHODS}
        if not any(rows.values()):continue
        f,h,n=[med(rows[m]) for m in METHODS];decode=phase.startswith('decode')
        ratio=lambda v:f/v if decode else v/f
        sh=ratio(h) if f and h else None;sn=ratio(n) if f and n else None
        supported=False
        if rows['fp16'] and rows['hadamard'] and rows['nar']:
            sessions={method:{r['session']:r['median'] for r in rows[method]['session_summaries']} for method in METHODS}
            common=set(sessions['fp16'])&set(sessions['hadamard'])&set(sessions['nar'])
            supported=len(common)==3 and all((sessions['fp16'][s]/sessions['hadamard'][s] if decode else sessions['hadamard'][s]/sessions['fp16'][s])>1.05 for s in common)
        c={'model':model,'mode':mode,'phase':phase,'hadamard_speedup':sh,'nar_speedup':sn,
          'nar_latency_overhead_pct':100*(n/h-1) if decode and n and h else None,
          'nar_throughput_ratio_pct':100*n/h if not decode and n and h else None,
          'acceleration_retention_pct':100*(sn-1)/(sh-1) if supported else None,
          'retention_reason':None if supported else 'Hadamard must exceed1.05x in each of three comparable sessions; negative/near-zero/unstable denominators are not acceleration retention.'}
        if h and n and rows['hadamard'] and rows['nar']:
            hs={r['session']:r['median'] for r in rows['hadamard']['session_summaries']};ns={r['session']:r['median'] for r in rows['nar']['session_summaries']}
            diffs=[ns[s]/hs[s]-1 for s in hs.keys()&ns.keys()];c['session_nar_over_had_minus_one']=diffs
            c['direction_consistent']=len(diffs)==3 and (all(d>0 for d in diffs) or all(d<0 for d in diffs))
            c['interpretation']=('fewer than three comparable sessions; no stable advantage established' if len(diffs)<3 else 'near parity / within observed variability' if abs(n/h-1)<=.01 or not c['direction_consistent'] else 'consistent observed direction across three sessions, specific to this implementation/workload')
        comparisons.append(c)
    correct={f.stem:read(f) for f in (out/'correctness').glob('*.json')}
    write(out/'env.json',{'generated_at':now(),'per_process_sources':'All raw_runs/*.json and correctness/*.json preserve start environment; deployment records also preserve end environment.','session_examples':{str(session):next((r['environment'] for r in raw if r.get('session')==session and r.get('samples')),None) for session in [1,2,3]},'correctness_example':next((v['environment'] for v in correct.values() if 'environment' in v),None)})
    summary={'generated_at':now(),'protocol_sha256':sha(out/'protocol.json'),'deployment':aggregate,'comparisons':comparisons,'correctness_status':{k:{t:v.get(t) for t in ['status','reason','complete','completed_layers']} for k,v in correct.items()},
      'real_checkpoint':read(out/'checkpoint_audit.json') if (out/'checkpoint_audit.json').exists() else None,
      'graph_performance':{'value':None,'reason':'No validated three-session full graph timing records; feasibility status is separate.'},'missing_policy':'null+reason; never fill unmeasured performance with zero'}
    from .graph_collect import generate as graph_generate
    summary['graph_performance']=graph_generate(out)
    write(out/'metrics_summary.json',summary)
    memory=[]
    for r in raw:
      if r.get('status')=='PASS' and r.get('storage'):
        memory.append({k:r[k] for k in ['key','model','method','phase','mode','session','loaded','warmed','inference_peak','storage']})
    write(out/'memory_breakdown.json',{'records':memory,'theory':{'3b_factor_bytes':28*8192*96,'8b_factor_bytes':32*14336*96,'shared_h128_bytes':32768,'formula':'per layer 2*d*16*2+2*k*d*2=96d bytes at k8; shared scratch adds4*T*splits*k per distinct shape','GB_divisor':1e9,'MB_divisor':1e6}})
    indexed={(r['model'],r['method'],r['mode'],r['phase'],r['session']):r for r in raw if r.get('status')=='PASS' and 'storage' in r}
    deltas=[]
    for key,nar in indexed.items():
        model,method,mode,phase,session=key
        if method!='nar':continue
        had=indexed.get((model,'hadamard',mode,phase,session))
        if had is None:continue
        cat=nar['storage']['bytes_by_category']
        removed=sum(t['storage_bytes'] for group in had['storage']['groups'].values() for t in group if '.mlp.down_proj.0.' in t['name'])
        expected=cat.get('nar_factors',0)+cat.get('h128',0)-removed
        loaded=nar['loaded']['allocated']-had['loaded']['allocated']
        partial_delta=cat.get('partial_workspace',0)-had['storage']['bytes_by_category'].get('partial_workspace',0)
        warmed_delta=nar['warmed']['allocated']-had['warmed']['allocated']
        storage_delta=nar['storage']['unique_storage_bytes']-had['storage']['unique_storage_bytes']
        deltas.append({'model':model,'mode':mode,'phase':phase,'session':session,'factor_bytes':cat.get('nar_factors',0),'shared_h128_bytes':cat.get('h128',0),'removed_hadamard_buffer_bytes':removed,'expected_model_storage_delta_bytes':expected,'observed_model_loaded_allocated_delta_bytes':loaded,'unattributed_loaded_delta_bytes':loaded-expected,'partial_workspace_bytes':cat.get('partial_workspace',0),'partial_workspace_delta_bytes':partial_delta,'warmed_allocated_delta_bytes':warmed_delta,'warmed_unique_storage_delta_bytes':storage_delta,'unattributed_warmed_delta_bytes':warmed_delta-storage_delta,'factor_and_partial_unexplained_bytes':warmed_delta-expected-partial_delta,'inference_peak_allocated_delta_bytes':nar['inference_peak']['peak_allocated']-had['inference_peak']['peak_allocated'],'note':'Unattributed remainder is retained explicitly; allocator granularity and extension allocations are not conflated with tensor storage.'})
    write(out/'memory_deltas.json',{'rows':deltas,'units':'exact bytes; MB=1e6; same model/mode/phase/session only'})
    names={'fp16':'FP16','hadamard':'QuaRot Hadamard','nar':'PrismQuant k=8'}
    pre=[];dec=[];mem=[]
    for model in MODELS:
      for method in METHODS:
        a1=get(model,method,'eager_sequence','prefill1');a16=get(model,method,'eager_sequence','prefill16');ad=get(model,method,'eager_sequence','decode')
        f1=med(get(model,'fp16','eager_sequence','prefill1'));f16=med(get(model,'fp16','eager_sequence','prefill16'));fd=med(get(model,'fp16','eager_sequence','decode'))
        pre.append([model,names[method],fmt(med(a1)),fmt(med(a1)/f1 if a1 and f1 else None),fmt(med(a16)),fmt(med(a16)/f16 if a16 and f16 else None)])
        for mode in ['eager_sequence','eager_step_sync']:
            r=get(model,method,mode,'decode');f=med(get(model,'fp16',mode,'decode'))
            dec.append([model,names[method],mode,fmt(med(r)),fmt(r['summary']['std'] if r else None),fmt(f/med(r) if f and r else None),str(r['independent_sessions']) if r else '0'])
        hd=get(model,'hadamard','eager_sequence','decode');fp=get(model,'fp16','eager_sequence','decode')
        mem.append([model,names[method],fmt(a16['peak_allocated_bytes']/1e9 if a16 else None),fmt(ad['peak_allocated_bytes']/1e9 if ad else None),fmt((ad['peak_allocated_bytes']-hd['peak_allocated_bytes'])/1e6 if ad and hd and method=='nar' else None),fmt(100*(1-ad['peak_allocated_bytes']/fp['peak_allocated_bytes']) if ad and fp else None)])
    table(out/'tables/deployment_prefill',['Model','Method','B1 tok/s','Speedup','B16 tok/s','Speedup'],pre,'Random weights; total input throughput. Core values require three sessions; incomplete rows are provisional. Run/session dispersion is in deployment_sessions.md.')
    table(out/'tables/deployment_decode',['Model','Method','Mode','ms/token','Run std','Speedup','Sessions'],dec,'Random weights;128 causal steps, first8 discarded. Pooled run dispersion is not session dispersion.')
    table(out/'tables/memory',['Model','Method','Prefill16 GB','Decode GB','Extra decode vs Had MB','Saved vs FP16 %'],mem,'Independent process peak allocated bytes, GB=1e9. Reserved and NVML values are reported separately in JSON.')
    kernels=[]
    for r in raw:
        if 'layer' in r and 'rows' in r:
            for k in r['rows']:
                kernels.append([r['model'],str(k['tokens']),str(r['session']),k['scope'],k['implementation'],fmt(k['wall_us']['median']),fmt(k['cuda_event_elapsed_us']['median']),k['status']])
    table(out/'tables/kernel_sessions',['Model','T','Session','Scope','Implementation','Wall us','Event us','Status'],kernels,'Fixed layer0 configurations, directly timed calls/chains; stage times must not be summed into chain measurements. Invalid rows excluded from rankings.')
    session_rows=[]
    extension_rows=[]
    for r in aggregate:
        for session in r['session_summaries']:
            session_rows.append([r['model'],names[r['method']],r['mode']+' / '+r['phase'],str(session['session']),fmt(session['median']),fmt(session['std']),str(session['n'])])
        if r['phase'] in ('decode8','decode8192'):
            extension_rows.append([r['model'],names[r['method']],r['phase'],fmt(r['summary']['median']),fmt(r['summary']['std']),r['status']])
    table(out/'tables/deployment_sessions',['Model','Method','Mode / phase','Session','Median','Run std','Runs'],session_rows,'Core session medians and run population std. Prefill units are total input tok/s; decode units are ms/causal step (batch8 advances eight sequences per step). Session-median dispersion is separate in metrics_summary.json.')
    table(out/'tables/extension_decode',['Model','Method','Workload','ms/step','Run std','Status'],extension_rows,'Appendix only: fixed batch8/prefix2048 and batch1/prefix8192 workloads, one session of50 runs after10 complete warmups; not independent-session evidence of a stable advantage.')
    comp=[['E28 kernel swap','Signed INT4 W/A; token-channel affine KV4','Random weights','See correctness JSON','Paper format not connected'],['E17 native R4','Group128 asymmetric packed A4','Real calibrated factors','Local only','No model GEMM/KV claim'],['Paper accuracy path','GPTQ W4; group128 asymmetric A4; K/V residual32','Local k8 checkpoint absent','BLOCKED','Current row/column-scale GEMM incompatible']]
    table(out/'tables/compatibility',['Path','Format','Weights','Numerical status','Deployment status'],comp,'Performance, numerical consistency and native-paper-format deployment are separate claims.')
    (out/'paper').mkdir(exist_ok=True)
    (out/'paper/experiment.tex').write_text('We benchmark random-weight Llama-3.2-3B and Llama-3.1-8B on one NVIDIA A40 with the existing QuaRot integer backend. Core measurements use ten complete warm-up runs and fifty repetitions in each of three balanced sessions. Decode is a sequence of causal model-forward calls, excludes the first eight of128 steps, and is not serving or sampling latency. We report independent process memory and separate wall, CUDA-event and profiler durations. Per-session and pooled statistics are generated from raw JSON; missing or invalid results are not imputed.\n')
    (out/'paper/kernel.tex').write_text(r'For row-major activations, $U=XA$ and $Z=XH_{128}-UB$, where $A\in\mathbb{R}^{d\times8}$ and $B\in\mathbb{R}^{8\times d}$. Kernel A forms split FP32 partials from two FP16 factor terms, with rank padded to16. Kernel B uses a normalized128-channel Hadamard and three high/low correction products, writes FP16, and feeds the unchanged QuaRot quantizer. The E17 native groupwise packing epilogue is a distinct contract. A shared scratch buffer assumes sequential execution.'+'\n')
    from .insights import generate
    progress=generate(out)
    statuslines=[f'| {k} | {v.get("status","MISSING")} | {v.get("reason", "")} |' for k,v in sorted(correct.items())]
    operator_failures=[r for r in correct.get('operators_all',{}).get('rows',[]) if r.get('status')=='FAIL']
    boundary_text='\n'.join(f"- `{r.get('test')}` / {r.get('case')}, reduction width {r.get('n')}: integer accumulation {r.get('max_integer_accumulator')} is exact, but conversion to FP16 before scale multiplication produces nonfinite output. This remains **FAIL** in the existing backend." for r in operator_failures)
    real_text='; '.join(f"{model}: {correct.get('model_real_'+model+'_fp16',{}).get('status','PENDING')}" for model in MODELS)
    answer=[]
    paper_results=[]
    for c in comparisons:
        if c['mode']=='eager_sequence' and c['phase'] in ['prefill1','prefill16','decode']:
            answer.append(f"- {c['model']} {c['phase']}: Hadamard/FP16 speedup {fmt(c['hadamard_speedup'])}; PrismQuant/FP16 {fmt(c['nar_speedup'])}. {c.get('interpretation','')}.")
            complete=all(get(c['model'],method,c['mode'],c['phase']) and get(c['model'],method,c['mode'],c['phase'])['status']=='COMPLETE' for method in METHODS)
            if complete:
                paper_results.append(f"For {dict([('3b','Llama-3.2-3B'),('8b','Llama-3.1-8B')])[c['model']]} {c['phase']}, Hadamard and PrismQuant achieve {fmt(c['hadamard_speedup'])}x and {fmt(c['nar_speedup'])}x relative to FP16 under the same eager-sequence timing protocol. {c.get('interpretation','')}. ")
    (out/'paper/results.tex').write_text(''.join(paper_results).replace('_',r'\_') if paper_results else 'Core three-session collection is incomplete; no final speedup conclusion is emitted.\n')

    report=f'''# E28-v2: auditable deployment experiments\n\nGenerated {now()}. This report contains completed records only where the linked JSON says COMPLETE/PASS; other stages remain explicitly incomplete, failed or blocked. All model performance rows use **random weights**, even when validation inputs are real text. No full model-quality evaluation is claimed.\n\n## One-page status\n\nCore timing: **{progress['core_complete']}/72** independent processes complete. Full timing completion does not override any numerical failure below.\n\n- Preserved original E28/E17/E22/E26 artifacts.\n- Froze counts, shapes, selected configs, thresholds and exclusions in [protocol.json](protocol.json); execution provenance in [source_manifest.json](source_manifest.json) and [execution_source_manifest.json](execution_source_manifest.json).\n- Corrected shared INT4 initialization, stale RoPE caches, timing boundaries, complete cache reset, throughput statistics, independent memory and invalid acceleration-retention ratios. The common metadata optimization is checked against the original wrapper.\n- The exact original environment and A40 are reused; no new GEMM, calibration or serving framework.\n- Native paper-format integer deployment is BLOCKED by missing matching complete checkpoint and incompatible group-scale/KV interfaces.\n\n| Check | Status | Reason |\n| --- | --- | --- |\n{chr(10).join(statuslines)}\n\n## Numerical boundaries and real weights\n\n{boundary_text or 'No failing operator boundary row has been recorded yet.'}\n\nReal base-checkpoint FP16 implementation checks: {real_text}. This is separate from compensated INT4 checkpoint support and from model quality. Prefill hidden/logits and multi-step decode are checked; exact cache comparisons cover steps1,2,9,64,65,128.\n\n## Six questions\n\n1. **Which stages accelerate relative to FP16?** Same-mode ratios only; [raw metrics and session status](metrics_summary.json), [deployment prefill](tables/deployment_prefill.md), [decode](tables/deployment_decode.md).\n{chr(10).join(answer) or '- No complete measurements yet; no speedup claim.'}\n\n2. **What does PrismQuant add versus Hadamard?** See directly measured A/B/quantizer/chain times in [kernel table](tables/kernel.md), same-mode wall ratios in [metrics](metrics_summary.json), and exact unique-storage categories/deltas in [memory_breakdown.json](memory_breakdown.json) and [memory_deltas.json](memory_deltas.json). Factor storage is96d bytes per layer plus one shared32768-byte H128; scratch is recorded separately. No rounded-to-zero overhead claim.\n\n3. **Does dispatch optimization help reproducibly?** The generic/prebound ablation uses identical kernels and byte-equal outputs, with three separately recorded sessions. See [kernel records](tables/kernel.md). A one-percent difference without consistent session direction is near parity, not a stable advantage.\n\n4. **Does graph replay grow the KV context?** See [graphability audit](graphability_audit.md) and the graph correctness rows above. Only A-B-A replay with steps1,2,9,64,65,128, full cache hashes and actual page crossing can pass. Original-binary capture and the bounded private current-stream adapter have separate records. Any matched adapter timing is in the [separate graph panel](tables/graph_deployment.md), with [preparation and memory](graph_memory_breakdown.json). No graph performance panel is populated from a fixed-context microbenchmark; absent timings remain null.\n\n5. **Which results execute integer kernels?** E28 Hadamard/NAR use the actual packed INT4 GEMM/quantizer and INT4 decode cache; FP16 uses FP16 arithmetic/cache. E17-native rows are local packed-activation microbenchmarks, not an end-to-end model. Random states are verified by [shared_state_audit.json](shared_state_audit.json). Real model quality remains untested.\n\n6. **What is missing for the paper's group128 asymmetric path?** See [quantizer contract](quantizer_contract.md) and [checkpoint audit](checkpoint_audit.json). Per-group reduction scales/offsets need partial sums/corrections absent from this GEMM interface; token/channel KV grouping and residual windows also differ. No discarded offsets or FP16 substitute is presented as native integer deployment.\n\n## Technical appendix\n\n[Kernel design](kernel_design.md), [old claims audit](old_claim_audit.json), [compatibility table](tables/compatibility.md), [memory table](tables/memory.md), and [measured evidence appendix](evidence_appendix.md), [profile summary](profile_summary.json) and [profile traces](profiles/) provide the detailed evidence. `correctness/attempt_1/` retains initial harness failures; [harness corrections](verification_harness_corrections.json) explain the repair without changing numerical thresholds.\n\nAll tables are regenerated by `collect`, which never launches a GPU experiment. Memory is bytes/decimal GB; pooled150 runs represent3 sessions. Profiler kernel sums, CUDA-event elapsed and wall time are distinct. Missing counters mean bandwidth/compute bottleneck labels remain hypotheses.\n'''
    if (out/'initialization_revision.json').exists():
        revision=read(out/'initialization_revision.json')
        prior=pathlib.Path(revision['predecessor']).name
        note=f'## Initialization revision and reused evidence\n\nThe first E28-v2 cohort used the legacy1..6 packed-byte range with name-stable shared weights: half the channels are zero and all other weights positive. Its8B NAR model produced nonfinite outputs: the last decoder residual overflowed FP16 although the R4 operators stayed finite. [Original failure and partial measurements](../{prior}/report_e28_v2.md) are retained. This run uses one predefined uniform0..255 packed-byte initialization for both integer methods, with unchanged scales/seeds/kernels/thresholds. It is a separate frozen random-weight protocol, not a claim of improved model quality.\n\n[Allocator-preflight correction](allocator_preflight_revision.json) identifies the first two affected processes and their fully retained/repeated measurements. The reruns apply one consistent memory-preparation protocol; no run was selected by speed.\n\n[Reused correctness records](reused_correctness.json) identify unchanged local operators/factors and FP16 states verified in the same A40 allocation. All four integer model paths are rechecked for the new state; all formal performance sessions are newly measured.\n\n'
        report=report.replace('## One-page status',note+'## One-page status')
        report=report.replace('`correctness/attempt_1/` retains initial harness failures;',f'The [predecessor attempt directory](../{prior}/correctness/attempt_1/) retains initial harness failures;')
    if (out/'cohort_status.json').exists():
        stopped=read(out/'cohort_status.json')
        report=report.replace('## One-page status',f"## Cohort stopped\n\n**{stopped['status']}**: {stopped['reason']} See the [replacement run](../{stopped['replacement_run']}/report_e28_v2.md). The records below are historical partial evidence, not an active pending run.\n\n## One-page status")
    (out/'report_e28_v2.md').write_text(report)
    rel=out.relative_to(ROOT)
    (ROOT/'report_e28_v2.md').write_text(f'# E28-v2\n\nThe current experiment record is [{rel.name}]({rel}/report_e28_v2.md).\n\nThis is a random-weight integer kernel-swap study with explicit numerical gates. Native paper-format model deployment and real model quality are not established; the report also records the backend accumulator-overflow failures. Consult the linked report for completed, failed, blocked and pending stages.\n')
    print('COLLECT',len(aggregate),'groups',len(kernels),'kernel rows',flush=True)
if __name__=='__main__':main()
