#!/usr/bin/env python3
"""CPU-only, frozen-source builder; never imports experiment or GPU code."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMMIT = '6747960b96b0b03e626e98eaeeccf0d1abc34e79'
RUN = 'results/e28_v2/20260912_a40_v2_full_int4'
HASHES = {}
ROWS = []


def read(path):
    path = str(path)
    data = (ROOT / path).read_bytes()
    frozen = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT)
    assert data == frozen, f'Source differs from frozen commit: {path}'
    HASHES[path] = hashlib.sha256(data).hexdigest()
    return data.decode()


def raw(name):
    return json.loads(read(f'{RUN}/{name}'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-12), (a, b)


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def add(metric, model, method, values, center, *, backend, mode, phase,
        unit, formula, sources, keys, statistic='pooled median of 150 samples',
        baseline='', batch=1, tokens=1, numerator=None, denominator=None):
    common = dict(metric=metric, model=model, method=method, baseline=baseline,
        backend=backend, mode=mode, phase=phase, batch=batch, tokens=tokens, k=8,
        unit=unit, formula=formula, source_commit=COMMIT, source_run=RUN,
        source_file=json.dumps(sources), source_key=json.dumps(keys),
        numerator=json.dumps(numerator), denominator=json.dumps(denominator))
    ROWS.append(dict(**common, session=0, statistic=statistic, value=center))
    for i, value in enumerate(values, 1):
        ROWS.append(dict(**common, session=i, statistic='paired session ratio' if baseline else
                        ('session inference peak' if metric=='peak_memory' else 'session median'), value=value))


def series(d, graph=False):
    samples, values, sources = [], [], []
    field = 'prefill_tokens_per_second' if d.get('phase', '').startswith('prefill') else 'ms_per_token'
    for key, summary in zip(d['source_keys'], d['session_summaries']):
        file = f"{'raw_graph_runs' if graph else 'raw_runs'}/{key}.json"
        obj = raw(file)
        assert obj['status'] == 'PASS' and obj['model'] == d['model'] and obj['method'] == d['method']
        assert obj['mode'] == d['mode'] and len(obj['samples']) == 50
        sample = [x[field] for x in obj['samples']]
        assert all(math.isfinite(x) and x > 0 for x in sample)
        val = median(sample); close(val, summary['median'])
        values.append(val); samples.extend(sample); sources.append(f'{RUN}/{file}')
    center = median(samples); close(center, d['summary']['median'])
    assert len(values)==3 and len(samples)==150
    return dict(center=center, sessions=values, files=sources,
                keys=[f'/samples/*/{field}', '/summary/median'])


def ratio(metric, a, b, model, method, *, scale=1, shift=0, **kw):
    vals = [scale*(x/y-shift) for x,y in zip(a['sessions'], b['sessions'])]
    center = scale*(a['center']/b['center']-shift)
    add(metric,model,method,vals,center,sources=a['files']+b['files'],
        keys=a['keys']+b['keys'],numerator=a['center'],denominator=b['center'],
        statistic='ratio of pooled medians (150 samples per method)',**kw)
    return center


def build():
    HASHES.clear(); ROWS.clear()
    report = read('report_e28_v2.md'); assert f'{RUN}/report_e28_v2.md' in report
    p = raw('protocol.json'); s = raw('metrics_summary.json'); gp = s['graph_performance']
    assert p['rank']==8 and p['core']['sessions']==3 and p['core']['runs']==50
    assert p['core']['decode']==dict(batch=1,discard=8,prefix=2048,steps=128)
    assert gp['status']=='COMPLETE' and gp['protocol_conformance']['status']=='PASS'
    assert len(gp['raw_status'])==36 and all(x['status']=='PASS' for x in gp['raw_status'])
    for m in ['3b','8b']:
        for method in ['fp16','hadamard','nar']:
            assert raw(f'correctness/graph_stream_{m}_{method}.json')['status']=='PASS'
    assert raw('shared_state_audit.json')['status']=='PASS'
    for name in ['kernel_design.md','quantizer_contract.md','tables/completion.md',
                 'tables/deployment_prefill.md','tables/deployment_sessions.md',
                 'tables/graph_deployment.md','tables/graph_sessions.md','tables/graph_memory.md',
                 'tables/kernel_ratios.md','tables/kernel.md','tables/kernel_sessions.md']:
        read(f'{RUN}/{name}')
    for name in ['correctness/backend_fingerprint_current_stream.json',
                 'correctness/backend_fingerprint_original.json','stream_allocation.json',
                 'stream_backend_manifest.json','graph_protocol_conformance.json',
                 'protocol_graph_allocation_addendum.json','recovery_allocation_audit.json']:
        raw(name)
    backend='original binary; uniform0..255 cohort'
    for model in ['3b','8b']:
        for batch in [1,16]:
            phase=f'prefill{batch}'
            group={d['method']:series(d) for d in s['deployment'] if d['model']==model and d['phase']==phase and d['mode']=='eager_sequence'}
            assert set(group)=={'fp16','hadamard','nar'}
            for method,a in group.items():
                add('prefill_throughput',model,method,a['sessions'],a['center'],
                    backend=backend,mode='eager_sequence',phase=phase,batch=batch,tokens=2048,
                    unit='input tokens/s',formula='median(batch * 2048 / elapsed_s)',sources=a['files'],keys=a['keys'])
                if method!='fp16':
                    ratio('prefill_speedup',a,group['fp16'],model,method,backend=backend,
                          mode='eager_sequence',phase=phase,batch=batch,tokens=2048,baseline='fp16',
                          unit='ratio',formula='throughput_method / throughput_FP16')
    memory=raw('graph_memory_breakdown.json')['rows']
    private='private current-stream; uniform0..255 cohort'
    graph_series={}
    for model in ['3b','8b']:
        for mode in ['eager_sequence','cuda_graph_sequence']:
            group={d['method']:series(d,graph=True) for d in gp['deployment'] if d['model']==model and d['mode']==mode}
            graph_series[model,mode]=group
            for method,a in group.items():
                add('decode_latency',model,method,a['sessions'],a['center'],backend=private,
                    mode=mode,phase='decode',unit='ms/token',formula='median(elapsed_s * 1000 / 120)',sources=a['files'],keys=a['keys'])
                ratio('decode_speedup',group['fp16'],a,model,method,backend=private,mode=mode,
                    phase='decode',unit='ratio',baseline='fp16',formula='latency_FP16 / latency_method')
            ratio('decode_overhead',group['nar'],group['hadamard'],model,'nar',scale=100,shift=1,
                backend=private,mode=mode,phase='decode',unit='%',baseline='hadamard',formula='100 * (latency_PrismQuant / latency_Hadamard - 1)')
            mm={}
            for method in ['fp16','hadamard','nar']:
                keys=[f'{model}_{method}_{mode}_decode_s{i}' for i in [1,2,3]]
                matches=[next((j,x) for j,x in enumerate(memory) if x['key']==k) for k in keys]
                values=[x['inference_peak']['peak_allocated'] for j,x in matches]
                official=next(x for x in gp['deployment'] if (x['model'],x['method'],x['mode'])==(model,method,mode))
                assert max(values)==official['inference_peak_allocated_bytes']
                mm[method]=values
                add('peak_memory',model,method,[v/1e9 for v in values],max(values)/1e9,backend=private,
                    mode=mode,phase='decode',unit='GB',formula='inference_peak.peak_allocated / 1e9',
                    sources=[f'{RUN}/graph_memory_breakdown.json'],keys=[f'/rows/{j}/inference_peak/peak_allocated' for j,x in matches],
                    statistic='maximum of 3 independent-session inference peaks',numerator=values,denominator=1e9)
            for method in ['hadamard','nar']:
                add('memory_saving',model,method,[100*(1-x/y) for x,y in zip(mm[method],mm['fp16'])],
                    100*(1-max(mm[method])/max(mm['fp16'])),backend=private,mode=mode,phase='decode',unit='%',
                    baseline='fp16',formula='100 * (1 - peak_bytes_method / peak_bytes_FP16)',
                    sources=[f'{RUN}/graph_memory_breakdown.json'],keys=[f'/rows/[key={model}_{method}_{mode}_decode_s*]/inference_peak/peak_allocated',f'/rows/[key={model}_fp16_{mode}_decode_s*]/inference_peak/peak_allocated'],
                    statistic='ratio of maximum independent-session inference peaks',numerator=mm[method],denominator=mm['fp16'])
        for method in ['fp16','hadamard','nar']:
            ratio('graph_vs_eager',graph_series[model,'eager_sequence'][method],graph_series[model,'cuda_graph_sequence'][method],model,method,
                  backend=private,mode='matched eager / graph',phase='decode',unit='ratio',baseline=method,
                  formula='same-private-backend eager latency / graph latency')
    comparisons=raw('kernel_comparisons.json')['comparisons']
    for index,d in enumerate(comparisons):
        vals=[]; files=[]; keys=[]
        for v in d['sessions']:
            f=f"raw_runs/kernels_{d['model']}_s{v['session']}.json"; obj=raw(f)
            found=[]
            for impl in [d['candidate'],d['baseline']]:
                i, row=next((i,x) for i,x in enumerate(obj['rows']) if x['implementation']==impl and x['tokens']==d['tokens'])
                assert row['status']=='VALID'
                close(median(row['samples_wall_us']),row['wall_us']['median'])
                found.append(row['wall_us']['median']);keys.append(f'/rows/{i}/samples_wall_us')
            val=found[0]/found[1];close(val,v['wall_ratio']); vals.append(val);files.append(f'{RUN}/{f}')
        close(median(vals),d['wall_ratio']['median'])
        add('kernel_ratio',d['model'],d['candidate'],vals,median(vals),backend=backend,mode='wall time',phase=d['scope'],
            unit='ratio',baseline=d['baseline'],tokens=d['tokens'],formula='median_session(candidate session median wall_us / baseline session median wall_us)',
            sources=[f'{RUN}/kernel_comparisons.json']+files,keys=[f'/comparisons/{index}/wall_ratio/median']+keys,
            statistic='median of 3 paired-session wall-time ratios')
    accuracy=[]
    for filename in ['fig4a.csv','fig4a_8b.csv','fig4b.csv']:
        rows=list(csv.DictReader(read(f'figures/{filename}').splitlines()))
        for i,row in enumerate(rows,2):
            source=list(csv.DictReader(read(row['source_file']).splitlines()))[int(row['source_csv_line'])-2]
            if row['kind'] in ['point','bf16_reference']:close(float(row['ppl']),float(source['mean_ppl']))
            elif row['kind']=='recovery':
                close(float(row['ppl_k']),float(source.get('mean_ppl',source.get('ppl'))))
                close(float(row['recovery']),(float(row['ppl_hadamard'])-float(row['ppl_k']))/(float(row['ppl_hadamard'])-float(row['ppl_bf16'])))
                row['recovery_percent']=100*float(row['recovery'])
            row.update(source_commit=COMMIT,source_run='original E20 / E11 / E18 / E17 protocols',
                retained_csv=f'figures/{filename}',retained_csv_line=i)
            accuracy.append(row)
    assert sum(x['kind']=='point' for x in accuracy)==22
    assert sum(x['kind']=='recovery' for x in accuracy)==15
    assert sum(x['kind']=='kernel_share' for x in accuracy)==4
    write_csv('deployment_efficiency_data.csv',ROWS)
    write_csv('fig4_revised_data.csv',accuracy+[x for x in ROWS if x['metric']=='decode_overhead' and x['mode']=='cuda_graph_sequence'])
    meta=dict(source_commit=COMMIT,source_run=RUN,graph_correctness='6/6 PASS',matched_private_timing='36/36 PASS',
        backend='Python/matplotlib; CPU-only artifact generation',source_hashes=HASHES,
        figure_width_inches=5.5,figure_number='fig_deployment_efficiency; numbering deferred pending complete manuscript',
        statistics=dict(deployment_center='pooled median of 150 formal runs; three separately recorded sessions',
            kernel_center='median of three paired-session wall-time ratios',memory='max of independent-session inference peaks',
            whiskers='min–max of three session medians (paired ratios where applicable); memory uses session peaks; not 95% CI',
            session_points='all three, deterministic orthogonal offset for visibility; measured coordinate unchanged'),
        accuracy_protocols=json.loads(read('figures/fig4_metadata.json'))['panels']['fig4b']['protocols'],
        known_limitations=['Random-weight performance, not a compensated checkpoint quality benchmark',
            'E28 per-token symmetric INT4 differs from paper group128 asymmetric format',
            'Large-accumulator backend FP16 conversion stress failures remain disclosed in report_e28_v2.md',
            'Graph timing uses preallocated page2176; separate page64 correctness crosses pages'])
    (HERE/'deployment_efficiency_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Verified {len(HASHES)} frozen sources; {len(ROWS)} deployment rows; all 22 budget and 15 rank points retained.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-data',action='store_true');args=ap.parse_args()
    if args.reuse_data:
        for name in ['deployment_efficiency_data.csv','fig4_revised_data.csv','deployment_efficiency_metadata.json']:assert (HERE/name).exists()
        print('Reusing frozen figure data; no experiments run.')
    else:build()
