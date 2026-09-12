"""Keep private-backend graph panels separate from original-binary core timing."""
from .common import *
from collections import defaultdict


def generate(out):
    from .collect import table, fmt
    grouped = defaultdict(list)
    raw = [read(p) for p in sorted((out / 'raw_graph_runs').glob('*.json'))]
    uuids=set();environments=set();binaries=set();inputs=set();state_checks=[]
    for r in raw:
        if r.get('status')!='PASS':continue
        env=r['environment']
        for line in env.get('processes','').splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)>1 and fields[1]==str(env['pid']):uuids.add(fields[0])
        environments.add(json.dumps({k:env.get(k) for k in ['versions','capability','cuda_runtime','affinity','torch_threads','torch_interop_threads']},sort_keys=True))
        binaries.add(r['backend_manifest']['binary_sha256']);inputs.add(r['input_sha256'])
        expected=read(out/'correctness'/f'model_{r["model"]}_{r["method"]}.json').get('shared_state')
        state_checks.append({'key':r['key'],'state_matches_original_verified_model':bool(expected and expected==r['shared_state'])})
    conformance={'same_gpu':len(uuids)==1 if uuids else None,'gpu_uuids':sorted(uuids),'same_environment':len(environments)==1 if environments else None,'binary_sha256':sorted(binaries),'same_input':len(inputs)==1 if inputs else None,'state_checks':state_checks}
    conformance['status']='PASS' if state_checks and len(uuids)==len(environments)==len(binaries)==len(inputs)==1 and all(r['state_matches_original_verified_model'] for r in state_checks) else 'PENDING_OR_REVIEW_REQUIRED'
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
                                    'status': r['status']})
                display.append([model, method, mode, fmt(value), fmt(r['summary']['std']),
                                fmt(speedup), str(r['independent_sessions'])])
    summary = {'timestamp': now(), 'scope': 'Private current-stream patch, same binary for every row in this panel; not original-binary core timing',
               'deployment': rows, 'comparisons': comparisons, 'protocol_conformance':conformance,
               'raw_status': [{'key': r.get('key'), 'status': r.get('status'), 'reason': r.get('reason')} for r in raw],
               'status': 'COMPLETE' if len(rows) == 12 and all(r['status'] == 'COMPLETE' for r in rows) else 'INCOMPLETE',
               'missing_reason': None if rows else 'No validated matched graph timing records yet'}
    write(out / 'graph_metrics_summary.json', summary)
    table(out / 'tables/graph_deployment', ['Model', 'Method', 'Mode', 'ms/step', 'Run std', 'FP16 speedup', 'Sessions'], display,
          'Private current-stream backend only. Full one-step growing-context graph and matched eager use prefix2048,128 steps, discard8. Metadata updates are included. Incomplete rows are provisional.')
    memory = []
    for r in raw:
        if r.get('status') == 'PASS':
            memory.append({'key': r['key'], 'loaded': r['loaded'], 'warmed': r['warmed'],
                           'inference_peak': r['inference_peak'], 'graph_preparation': r['graph_preparation'], 'storage': r['storage']})
    write(out / 'graph_memory_breakdown.json', {'rows': memory, 'units': 'bytes; allocated/reserved and graph-pool deltas are distinct'})
    return summary
