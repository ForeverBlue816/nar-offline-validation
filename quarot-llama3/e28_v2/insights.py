"""Evidence-derived appendix; runs on CPU and never changes experiment records."""
from .common import *
from collections import defaultdict


def pair_summary(records, baseline, candidate):
    """Compare valid, same-session measurements; invalid rows stay unranked."""
    paired = []
    for session in sorted(records):
        rows = records[session]
        left, right = rows.get(baseline), rows.get(candidate)
        if not left or not right or any(r['status'] != 'VALID' for r in (left, right)):
            continue
        paired.append({'session': session,
                       'wall_ratio': right['wall_us']['median'] / left['wall_us']['median'],
                       'event_ratio': right['cuda_event_elapsed_us']['median'] / left['cuda_event_elapsed_us']['median']})
    ratios = [p['wall_ratio'] for p in paired]
    consistent = len(paired) == 3 and (all(v < 1 for v in ratios) or all(v > 1 for v in ratios))
    median = statistics.median(ratios) if ratios else None
    interpretation = ('not enough valid sessions' if len(paired) < 3 else
                      'near parity / within observed variability' if not consistent or abs(median - 1) <= .01 else
                      'consistent direction in these three sessions; no cross-hardware claim')
    return {'baseline': baseline, 'candidate': candidate, 'sessions': paired,
            'wall_ratio': stats(ratios), 'event_ratio': stats([p['event_ratio'] for p in paired]),
            'direction_consistent': consistent, 'interpretation': interpretation,
            'ratio_definition': 'candidate elapsed / baseline elapsed; greater than one means slower'}


def generate(out):
    from .collect import table, fmt
    raw = [read(f) for f in sorted((out / 'raw_runs').glob('*.json'))]
    grouped = defaultdict(dict)
    traffic = []
    for record in raw:
        if record.get('layer') != 0:
            continue
        for row in record.get('rows', []):
            grouped[(record['model'], row['tokens'])].setdefault(record['session'], {})[row['implementation']] = row
            if row['implementation'] == 'nar_module' and row['status'] == 'VALID':
                t, d, k, splits = row['tokens'], row['d'], row['rank'], row['proj']['splits']
                parts = {'input_A_and_B': 4*t*d, 'output': 2*t*d, 'padded_factors': 96*d,
                         'partial_write_read': 8*t*splits*k, 'shared_h128': 32768}
                byte_count = sum(parts.values())
                traffic.append({'model': record['model'], 'tokens': t, 'session': record['session'],
                                'compulsory_bytes': byte_count, 'components_bytes': parts,
                                'estimated_GB_per_second_event': byte_count / row['cuda_event_elapsed_us']['median'] / 1000,
                                'interpretation': 'Compulsory-data estimate, not measured DRAM bandwidth; factors, scratch and H may be reread by programs and served from cache.'})
    summaries=[]
    short_names={'hadamard_fp16':'Hadamard (FP16)','nar_module':'NAR module','nar_prebound':'NAR prebound','nar_generic':'NAR generic','nar_generic_shuffle':'NAR shuffle','A_only':'NAR A','B_tc_only':'NAR B (TC)','B_shuffle_only':'NAR B (shuffle)','shared_quantizer':'Shared quantizer','hadamard_plus_quantizer':'Hadamard + Q','nar_plus_quantizer':'NAR + Q','nar_native':'NAR native','block_hadamard_native':'Block-Had native'}
    for (model,tokens),sessions in sorted(grouped.items()):
        names=sorted({name for records in sessions.values() for name in records})
        for name in names:
            records=[rows[name] for rows in sessions.values() if name in rows]
            valid=sum(r['status']=='VALID' for r in records)
            summaries.append({'model':model,'tokens':tokens,'implementation':name,'label':short_names.get(name,name),'scope':records[0]['scope'],
                              'wall_us':stats([v for r in records for v in r['samples_wall_us']]),
                              'event_us':stats([v for r in records for v in r['samples_event_elapsed_us']]),
                              'sessions':len(records),'valid_sessions':valid,
                              'status':'INVALID' if valid<len(records) else 'COMPLETE' if len(records)==3 else 'PROVISIONAL'})
    write(out/'kernel_summary.json',{'rows':summaries,'dispersion':'Run population std;150 repeats represent3 sessions, not150 independent hardware sessions. Invalid timings are retained but not ranked.'})
    table(out/'tables/kernel',['Model / T','Implementation','Wall us','Event us','Valid sessions','Status'],
          [[r['model']+' / '+str(r['tokens']),r['label'],fmt(r['wall_us']['median']),fmt(r['event_us']['median']),str(r['valid_sessions']),r['status']] for r in summaries],
          'Fixed layer0. FP16 module/stage/dispatch and E17 native packed-output contracts are distinct; Q is the shared QuaRot quantizer. Chains are directly timed. Complete records have three50-run sessions; invalid rows are not ranked.')
    pairs = [('E28 slot', 'hadamard_fp16', 'nar_module'),
             ('E28 frontend', 'hadamard_plus_quantizer', 'nar_plus_quantizer'),
             ('Dispatch', 'nar_generic', 'nar_prebound'),
             ('B implementation', 'nar_generic_shuffle', 'nar_generic'),
             ('E17 native', 'block_hadamard_native', 'nar_native')]
    comparisons = []
    for (model, tokens), records in sorted(grouped.items()):
        for scope, baseline, candidate in pairs:
            comparisons.append({'model': model, 'tokens': tokens, 'scope': scope,
                                **pair_summary(records, baseline, candidate)})
    write(out / 'kernel_comparisons.json', {'comparisons': comparisons, 'traffic_estimates': traffic})
    table(out / 'tables/kernel_ratios', ['Model', 'T', 'Comparison', 'Wall ratio', 'Event ratio', 'Valid sessions'],
          [[r['model'], str(r['tokens']), r['scope'], fmt(r['wall_ratio']['median']),
            fmt(r['event_ratio']['median']), str(len(r['sessions']))] for r in comparisons],
          'Candidate/baseline elapsed ratios; greater than one is slower. Exact implementation names, per-session ratios and exclusions are in kernel_comparisons.json.')

    profiles = []
    for model in MODELS:
        for method in METHODS:
            path = out / 'profiles' / f'{model}_{method}.json'
            if not path.exists():
                profiles.append({'model': model, 'method': method, 'status': 'PENDING', 'reason': 'No profile record yet'})
                continue
            record = read(path)
            timeline = record.get('timeline_summary', {})
            profiles.append({'model': model, 'method': method, 'status': record.get('status'),
                             'reason': record.get('reason'), 'source': str(path.relative_to(out)), **timeline})
    write(out / 'profile_summary.json', {'rows': profiles,
          'interpretation': 'Profiler durations are separate diagnostics, not main timing samples. Kernel sum, wall and event elapsed are distinct. Gaps do not measure CPU arithmetic.'})
    profile_rows = []
    for r in profiles:
        profile_rows.append([r['model'], r['method'], r['status'],
                             fmt(r.get('kernel_sum_ms_per_step')),
                             str(r.get('kernel_count', 'N/A')),
                             fmt(r.get('inter_kernel_gap_ms'))])
    table(out / 'tables/profile', ['Model', 'Method', 'Status', 'Kernel ms/step', 'Kernels/4 steps', 'Gap ms/4 steps'],
          profile_rows, 'Separate four-step profiler trace; profiler perturbation prevents replacing steady-state wall measurements with these times.')

    core = [r for r in raw if r.get('phase') in ('prefill1', 'prefill16', 'decode') and r.get('mode') in ('eager_sequence', 'eager_step_sync')]
    complete = [r for r in core if r.get('status') == 'PASS' and len(r.get('samples', [])) == 50]
    failures = [{'key': r.get('key'), 'status': r.get('status'), 'reason': r.get('reason')} for r in raw if r.get('status') in ('FAIL', 'INVALID', 'BLOCKED')]
    correctness = []
    for path in sorted((out / 'correctness').glob('*.json')):
        result = read(path)
        bad = [r for r in result.get('rows', []) if r.get('status') in ('FAIL', 'INVALID', 'BLOCKED')]
        correctness.append({'source': str(path.relative_to(out)), 'status': result.get('status'),
                            'reason': result.get('reason'), 'failed_rows': bad})
    write(out / 'completion_audit.json', {'timestamp': now(), 'core_complete_processes': len(complete),
          'core_expected_processes': 72, 'core_status': 'COMPLETE' if len(complete) == 72 else 'INCOMPLETE',
          'raw_stage_failures': failures, 'correctness': correctness,
          'pipeline_finished': (out / 'pipeline_finished.json').exists()})

    summary = read(out / 'metrics_summary.json')
    details = ['# Measured evidence appendix', '',
               f'Core timing: {len(complete)}/72 independent processes complete (50 runs per process).', '',
               'Three balanced sessions are required for core conclusions. The expanded workloads have one session only.', '',
               '## Kernel comparisons', '',
               'Each ratio is candidate elapsed / baseline elapsed. A value above one means a longer elapsed time.', '']
    for r in comparisons:
        details.append(f"- {r['model']} T={r['tokens']} {r['scope']}: {r['candidate']} / {r['baseline']}, wall {fmt(r['wall_ratio']['median'])}, CUDA-event {fmt(r['event_ratio']['median'])}; {r['interpretation']}.")
    details += ['', '## Decode wall time and memory', '']
    deltas = read(out / 'memory_deltas.json')['rows']
    for model in MODELS:
        same = [r for r in deltas if r['model'] == model and r['mode'] == 'eager_sequence' and r['phase'] == 'decode']
        if same:
            r = same[0]
            details.append(f"- {model}: resident NAR factors {r['factor_bytes']} bytes, shared H128 {r['shared_h128_bytes']} bytes, removed Hadamard buffer {r['removed_hadamard_buffer_bytes']} bytes. Predicted loaded delta {r['expected_model_storage_delta_bytes']} bytes; observed {r['observed_model_loaded_allocated_delta_bytes']} bytes; unassigned remainder {r['unattributed_loaded_delta_bytes']} bytes. Decode partial workspace {r['partial_workspace_bytes']} bytes. Every session/phase is retained in [memory deltas](memory_deltas.json).")
        comparison = next((r for r in summary['comparisons'] if r['model'] == model and r['mode'] == 'eager_sequence' and r['phase'] == 'decode'), None)
        if comparison:
            details.append(f"- {model}: NAR vs Hadamard decode wall overhead {fmt(comparison['nar_latency_overhead_pct'])}%; {comparison.get('interpretation', 'not enough comparable results')}.")
        had = next((r for r in profiles if r['model'] == model and r['method'] == 'hadamard'), {})
        nar = next((r for r in profiles if r['model'] == model and r['method'] == 'nar'), {})
        if had.get('kernel_sum_ms_per_step') is not None and nar.get('kernel_sum_ms_per_step') is not None:
            details.append(f"- {model}: separate profiled NAR minus Hadamard kernel sum {fmt(nar['kernel_sum_ms_per_step'] - had['kernel_sum_ms_per_step'])} ms/step. This difference is not a wall-time estimate.")
    details += ['', '## Explicit failures and limits', '',
                'The operator boundary test is a deployment limitation even if random-weight forwards remain finite. Failed graph capture/replay cannot establish a graph speedup. Real FP16 base-checkpoint consistency cannot establish real INT4 model quality.', '',
                'Full details: [completion and failures](completion_audit.json), [profiles](profile_summary.json), [kernel ratios and byte estimates](kernel_comparisons.json), [session metrics](metrics_summary.json).', '']
    (out / 'evidence_appendix.md').write_text('\n'.join(details))
    return {'core_complete': len(complete), 'kernel_comparisons': len(comparisons)}
