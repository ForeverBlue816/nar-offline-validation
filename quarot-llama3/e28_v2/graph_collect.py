"""Keep private-backend graph panels separate from original-binary core timing."""
from .common import *
from collections import defaultdict


def generate(out):
    from .collect import table, fmt, escaped
    grouped = defaultdict(list)
    raw = [read(p) for p in sorted((out / 'raw_graph_runs').glob('*.json'))]
    uuids=set();environments=set();binaries=set();inputs=set();state_checks=[]
    sources={name:set() for name in ['common.py','cache_adapter.py','benchmark.py','graph_benchmark.py']}
    for r in raw:
        if r.get('status')!='PASS':continue
        env=r['environment']
        for name,hashes in sources.items():hashes.add(env.get('execution_python_sha256',{}).get('quarot-llama3/e28_v2/'+name))
        for line in env.get('processes','').splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)>1 and fields[1]==str(env['pid']):uuids.add(fields[0])
        environments.add(json.dumps({k:env.get(k) for k in ['versions','capability','cuda_runtime','affinity','torch_threads','torch_interop_threads']},sort_keys=True))
        binaries.add(r['backend_manifest']['binary_sha256']);inputs.add(r['input_sha256'])
        expected=read(out/'correctness'/f'model_{r["model"]}_{r["method"]}.json').get('shared_state')
        state_checks.append({'key':r['key'],'state_matches_original_verified_model':bool(expected and expected==r['shared_state'])})
    same_sources=all(len(v)==1 and None not in v for v in sources.values())
    conformance={'same_measurement_sources':same_sources,'source_hashes':{k:sorted(v,key=str) for k,v in sources.items()},'same_gpu':len(uuids)==1 if uuids else None,'gpu_uuids':sorted(uuids),'same_environment':len(environments)==1 if environments else None,'binary_sha256':sorted(binaries),'same_input':len(inputs)==1 if inputs else None,'state_checks':state_checks}
    conformance['status']='PASS' if same_sources and state_checks and len(uuids)==len(environments)==len(binaries)==len(inputs)==1 and all(r['state_matches_original_verified_model'] for r in state_checks) else 'PENDING_OR_REVIEW_REQUIRED'
    write(out/'graph_protocol_conformance.json',conformance)
    for row in raw:
        if row.get('status') == 'PASS' and len(row.get('samples', [])) == 50:
            grouped[(row['model'], row['method'], row['mode'])].append(row)
    rows = []
    lookup = {}
    for key, records in sorted(grouped.items()):
        records.sort(key=lambda r: r['session'])
        entry = {'model': key[0], 'method': key[1], 'mode': key[2],
                 'summary': stats([s['ms_per_token'] for r in records for s in r['samples']]),
                 'session_summaries': [dict(session=r['session'], **r['summary']) for r in records],
                 'session_median_dispersion': stats([r['summary']['median'] for r in records]),
                 'independent_sessions': len(records),
                 'status': 'COMPLETE' if len(records) == 3 else 'PROVISIONAL',
                 'inference_peak_allocated_bytes': max(r['inference_peak']['peak_allocated'] for r in records),
                 'source_keys': [r['key'] for r in records]}
        rows.append(entry)
        lookup[key] = entry
    comparisons = []
    display = []
    for model in MODELS:
        for mode in ['eager_sequence', 'cuda_graph_sequence']:
            f = lookup.get((model, 'fp16', mode))
            for method in METHODS:
                r = lookup.get((model, method, mode))
                if not r:
                    continue
                value = r['summary']['median']
                speedup = f['summary']['median'] / value if f else None
                eager = lookup.get((model, method, 'eager_sequence'))
                gain = eager['summary']['median'] / value if eager and mode == 'cuda_graph_sequence' else None
                comparisons.append({'model': model, 'method': method, 'mode': mode,
                                    'same_mode_fp16_speedup': speedup, 'same_method_graph_vs_eager_speedup': gain,
                                    'status': 'COMPLETE' if r['status']=='COMPLETE' and f and f['status']=='COMPLETE' else 'PROVISIONAL'})
                display.append([model, method, mode, fmt(value), fmt(r['summary']['std']),
                                fmt(speedup), str(r['independent_sessions'])])
    def paired_ratio(numerator,denominator):
        if not numerator or not denominator:return {'samples':[],'summary':stats([]),'interpretation':'Missing matching method/mode records'}
        ns={r['session']:r['median'] for r in numerator['session_summaries']}
        ds={r['session']:r['median'] for r in denominator['session_summaries']}
        samples=[{'session':s,'ratio':ns[s]/ds[s]} for s in sorted(ns.keys()&ds.keys())]
        values=[r['ratio'] for r in samples]
        consistent=len(values)==3 and (all(v<1 for v in values) or all(v>1 for v in values))
        summary=stats(values)
        interpretation=('fewer than three matched sessions; no stable advantage established' if len(values)<3 else 'near parity / within observed variability' if not consistent or abs(summary['median']-1)<=.01 else 'consistent observed direction in these three sessions')
        return {'samples':samples,'summary':summary,'direction_consistent':consistent,'interpretation':interpretation}
    paired=[]
    for model in MODELS:
        for method in METHODS:
            eager=lookup.get((model,method,'eager_sequence'));graph=lookup.get((model,method,'cuda_graph_sequence'))
            paired.append({'model':model,'method':method,'comparison':'same-method eager / graph latency; greater than one means graph is faster',**paired_ratio(eager,graph)})
        for mode in ['eager_sequence','cuda_graph_sequence']:
            nar=lookup.get((model,'nar',mode));had=lookup.get((model,'hadamard',mode))
            paired.append({'model':model,'mode':mode,'comparison':'NAR / Hadamard latency; greater than one means NAR is slower',**paired_ratio(nar,had)})
    for model in MODELS:
        for mode in ['eager_sequence','cuda_graph_sequence']:
            fp=lookup.get((model,'fp16',mode))
            for method in ['hadamard','nar']:
                candidate=lookup.get((model,method,mode))
                paired.append({'model':model,'method':method,'mode':mode,'comparison':'FP16 / method latency; greater than one means faster than FP16',**paired_ratio(fp,candidate)})
    method_comparisons=[]
    for model in MODELS:
        for mode in ['eager_sequence','cuda_graph_sequence']:
            panel={method:lookup.get((model,method,mode)) for method in METHODS}
            if not all(panel.values()):continue
            f,h,n=[panel[method]['summary']['median'] for method in METHODS]
            sessions={method:{r['session']:r['median'] for r in panel[method]['session_summaries']} for method in METHODS}
            common=set(sessions['fp16'])&set(sessions['hadamard'])&set(sessions['nar'])
            supported=len(common)==3 and all(sessions['fp16'][i]/sessions['hadamard'][i]>1.05 for i in common)
            method_comparisons.append({'model':model,'mode':mode,'hadamard_speedup':f/h,'nar_speedup':f/n,'nar_latency_overhead_pct':100*(n/h-1),
              'acceleration_retention_pct':100*(f/n-1)/(f/h-1) if supported else None,
              'retention_reason':None if supported else 'Hadamard must exceed1.05x in every one of three matched sessions; negative, small or unstable denominators are not acceleration retention.'})
    finished=out/'stream_pipeline_finished.json'
    stage=read(finished) if finished.exists() else None
    complete=len(rows)==12 and all(r['status']=='COMPLETE' for r in rows)
    status=('COMPLETE' if conformance['status']=='PASS' else 'REVIEW_REQUIRED') if complete else ('PARTIAL_OR_BLOCKED' if stage else 'INCOMPLETE')
    summary = {'timestamp': now(), 'scope': 'Private current-stream patch, same binary for every row in this panel; not original-binary core timing',
               'deployment': rows, 'comparisons': comparisons, 'method_comparisons': method_comparisons, 'paired_session_comparisons':paired, 'protocol_conformance':conformance, 'stage_completion':stage,
               'raw_status': [{'key': r.get('key'), 'status': r.get('status'), 'reason': r.get('reason')} for r in raw],
               'status': status,
               'missing_reason': None if rows else 'No validated matched graph timing records yet'}
    write(out / 'graph_metrics_summary.json', summary)
    table(out / 'tables/graph_deployment', ['Model', 'Method', 'Mode', 'ms/step', 'Run std', 'FP16 speedup', 'Sessions'], display,
          'Private current-stream backend only. Full one-step growing-context graph and matched eager use prefix2048,128 steps, discard8. Metadata updates are included. Incomplete rows are provisional.')
    memory = []
    for r in raw:
        if r.get('status') == 'PASS':
            memory.append({'key': r['key'], 'loaded': r['loaded'], 'warmed': r['warmed'],
                           'inference_peak': r['inference_peak'], 'graph_preparation': r['graph_preparation'], 'storage': r['storage']})
    capture_scope={'capture_deltas':'Global torch.cuda allocated/reserved after capture minus before capture; these are net changes, not an isolated per-pool measurement.',
                   'negative_reserved_delta':'A negative net reserved difference is an allocator-state change across the capture boundary, not negative Graph memory usage.',
                   'standalone_graph_pool_bytes':{'value':None,'reason':'No per-pool allocator snapshot or counter was captured. Pool identifiers and global net deltas are recorded; independent-process inference peaks include the retained graph.'}}
    write(out / 'graph_memory_breakdown.json', {'rows': memory, 'units': 'bytes; allocated/reserved and NVML are distinct','capture_scope':capture_scope})
    memory_summary=[]
    for key,records in sorted(grouped.items()):
        preparations=[r['graph_preparation'] for r in records if r['graph_preparation'].get('graphs')]
        item={'model':key[0],'method':key[1],'mode':key[2],'sessions':len(records),
              'inference_peak_allocated_bytes':max(r['inference_peak']['peak_allocated'] for r in records),
              'inference_peak_reserved_bytes':max(r['inference_peak']['peak_reserved'] for r in records),
              'capture_preparation_seconds':stats([p['seconds'] for p in preparations]),
              'capture_net_allocated_delta_bytes':stats([p['allocated_delta_bytes'] for p in preparations]),
              'capture_net_reserved_delta_bytes':stats([p['reserved_delta_bytes'] for p in preparations])}
        memory_summary.append(item)
    write(out/'graph_memory_summary.json',{'rows':memory_summary,'capture_scope':capture_scope})
    table(out/'tables/graph_memory',['Model','Method','Mode','Peak alloc GB','Peak reserv GB','Capture s','Net alloc MB'],
          [[r['model'],r['method'],r['mode'],fmt(r['inference_peak_allocated_bytes']/1e9),fmt(r['inference_peak_reserved_bytes']/1e9),fmt(r['capture_preparation_seconds']['median']),fmt(r['capture_net_allocated_delta_bytes']['median']/1e6 if r['capture_net_allocated_delta_bytes']['median'] is not None else None)] for r in memory_summary],
          'Private panel; independent-process inference peaks. Capture preparation is separate from steady-state timing. Net capture allocated/reserved changes do not isolate private-pool size; negative reserved changes are retained in JSON. Eager capture fields are N/A.')
    (out/'paper').mkdir(exist_ok=True)
    paper=[]
    if status=='COMPLETE':
        paper.append('The following decode results use the isolated current-stream adapter and a separate matched eager/Graph panel on the same physical A40. All six full-model Graph paths passed growing-cache A-B-A checks, including a page boundary. ')
        for model in MODELS:
            fp=lookup[(model,'fp16','cuda_graph_sequence')]['summary']['median']
            had=lookup[(model,'hadamard','cuda_graph_sequence')]['summary']['median']
            nar=lookup[(model,'nar','cuda_graph_sequence')]['summary']['median']
            variability=next(x['interpretation'] for x in paired if x['model']==model and x.get('mode')=='cuda_graph_sequence' and x['comparison'].startswith('NAR /'))
            paper.append(f'For {model}, Graph decode medians are {fmt(fp)}, {fmt(had)}, and {fmt(nar)} ms per step for FP16, Hadamard, and PrismQuant. PrismQuant achieves {fmt(fp/nar)}x relative to FP16 and has {fmt(100*(nar/had-1))}% latency difference relative to Hadamard; {variability}. ')
    (out/'paper/graph_results.tex').write_text(escaped(''.join(paper).strip())+'\n' if paper else 'The matched three-session private Graph panel is incomplete; no final Graph speedup conclusion is emitted.\n')
    return summary
