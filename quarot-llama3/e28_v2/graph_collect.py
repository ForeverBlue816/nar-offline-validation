"""Keep private-backend graph panels separate from original-binary core timing."""
from .common import *
from collections import defaultdict


def generate(out):
    from .collect import table, fmt
    grouped = defaultdict(list)
    raw = [read(p) for p in sorted((out / 'raw_graph_runs').glob('*.json'))]
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
               'deployment': rows, 'comparisons': comparisons,
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
