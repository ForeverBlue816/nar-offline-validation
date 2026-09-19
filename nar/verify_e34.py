"""Read-only final E34 preservation, numerical, pairing and sampling gates."""
from pathlib import Path
import json,sys,hashlib,subprocess
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar import e34_anchor_attribution as e

def verify():
    manifest=a.js(a.REPO/'experiments/e34_execution_manifest.json');records=[]
    for path,sha in manifest['old_results_sha256'].items():assert a.sha(a.REPO/path)==sha,('old result changed',path)
    old_report=subprocess.check_output(['git','show','6251b7f:report.md'],cwd=a.REPO)
    assert (a.REPO/'report.md').read_bytes().startswith(old_report)
    for m in e.MODELS:
        e.verify_frozen(m);done=a.js(e.root(m)/'e34_DONE.json');assert done['status']=='COMPLETE'
        specs=e.plan();rr=a.rows(e.root(m)/'e34_per_sequence.csv')
        expected={(p['row'],str(seed),str(chunk)) for p in specs for seed in a.SEEDS for chunk in range(64)}
        actual={(r['row'],r['seed'],r['chunk']) for r in rr};assert actual==expected and len(actual)==len(rr)
        token_sha=a.sha(a.tokenpath(m,'wt2_eval'));assert all(r['token_sha256']==token_sha and int(r['tokens'])==2047 and np.isfinite(float(r['nll'])) and r['unmasked_bf16_bit_identical']=='True' for r in rr)
        gates=a.rows(e.root(m)/'e34_gates.csv');expected_g={(p['row'],str(seed),s,str(l)) for p in specs for seed in a.SEEDS for s,l,n in a.keys(m)}
        actual_g={(r['row'],r['seed'],r['site'],r['layer']) for r in gates};assert len(gates)==len(actual_g) and actual_g==expected_g
        assert all(float(r['round_trip'])<=1e-6 and float(r['anchor_residual'])<=1e-6 for r in gates)
        contracts=a.rows(e.root(m)/'e34_rotation_contract.csv');ds=defaultdict(set)
        for r in contracts:
            assert r['unchanged_G']==r['unchanged_D']==r['unchanged_source_order']=='True'
            assert int(r['other_groups'])==0 and int(r['selected_directions'])==int(r['group_count'])
            assert int(r['changed_target_coordinates'])==(0 if int(r['column'])==0 else 2*int(r['group_count']))
            ds[r['seed'],r['site'],r['layer']].add(r['D_sha256'])
        assert all(len(hashes)==1 for hashes in ds.values())
        replay=a.rows(e.root(m)/'e34_replay_audit.csv');assert len(replay)==4*192 and all(r['float32_bit_identical']=='True' for r in replay)
        old={(r['row'],r['seed'],r['chunk']):r for r in a.rows(e.root(m)/'e29_per_sequence.csv')}
        fresh={(r['row'],r['seed'],r['chunk']):r for r in rr}
        for spec in specs:
            if spec['baseline']:
                for seed in a.SEEDS:
                    for c in range(64):
                        x=old[spec['baseline'],str(seed),str(c)];y=fresh[spec['row'],str(seed),str(c)]
                        assert x['nll']==y['nll']
        reuse=a.rows(e.root(m)/'e34_baseline_reuse.csv');assert len(reuse)==12
        assert all(r['source_fp32_nll_sha256']==r['replay_fp32_nll_sha256'] and r['all_bit_identical']=='True' for r in reuse)
        flags=a.loadt(e.assets(m)/'flags.pt');meta=a.js(e.root(m)/'e34_flags.json')
        assert a.sha(e.assets(m)/'flags.pt')==meta['flag_file_sha256'] and e.sha_tensor(flags['flagged'])==meta['flag_tensor_sha256']
        massive,bos=e.flags_from_scores(flags['scores'].numpy());assert np.array_equal(massive,flags['massive'].numpy()) and np.array_equal(bos,flags['bos'].numpy())
        assert int(massive.sum())==132 and int(bos.sum())==64
        assert .0005<=meta['massive_fraction']<=.005 and .0005<=meta['flagged_fraction']<=.005
        dc=[r for r in a.rows(e.root(m)/'e34_dc_share_by_layer.csv') if r['site']=='qkv']
        chosen=max(dc,key=lambda r:(float(r['dc_share']),-int(r['layer'])))
        assert int(chosen['layer'])==meta['selected_layer']==flags['layer']
        counts=flags['flagged'].sum(1).numpy()
        for r in rr:
            c=int(r['chunk']);wanted=counts[c] if r['row'].startswith('M1_') else (2048-counts[c] if r['row'].startswith('M2_') else 2048)
            assert int(r['quantized_input_positions'])==wanted
        cfg,dimensions=a.config(m);groups=dimensions['down']//128
        for filename,nrows in [('e34_range_common_input.csv',4*3*cfg.num_hidden_layers*groups),('e34_range_runtime.csv',8*3*cfg.num_hidden_layers*groups)]:
            ranges=a.rows(e.root(m)/filename);assert len(ranges)==nrows
            unique={(r['row'],r['seed'],r['layer'],r['group']) for r in ranges};assert len(unique)==nrows
            assert all(int(r['observations'])==64*2048 and np.isfinite(float(r['mean_range'])) and float(r['mean_range'])>=0 for r in ranges)
        split=a.rows(e.root(m)/'e34_range_split.csv')
        assert all(r['mean_range']=='' for r in split if int(r['layer_group_count'])==0)
        assert all(int(r['layer_group_count'])==0 for r in split if r['group_class']=='other' and not r['row'].startswith('Had'))
        assert len(a.rows(e.root(m)/'e34_summary.csv'))==16
        attrs=a.rows(e.root(m)/'e34_attribution.csv');disjoint=[r for r in attrs if r['token_class']!='BOS+massive'];assert sum(int(r['token_count']) for r in disjoint)==64*2047
        target=a.rows(e.root(m)/'e34_attribution_target_position.csv');bosrow=next(r for r in target if r['token_class']=='BOS');assert bosrow['token_count']=='0' and bosrow['gain_mean']=='' and bosrow['offset_spec_mean']==''
        reduction=a.rows(e.root(m)/'e34_token_reduction_audit.csv');assert len(reduction)==768
        files={p.name:a.sha(p) for p in sorted(e.root(m).glob('e34_*')) if p.is_file()}
        largest=max((e.root(m)/name).stat().st_size for name in files);assert largest<90*1024*1024
        records.append(dict(model=m,status='PASS',ppl_chunk_records=len(rr),per_token_wide_rows=3*64*2047,numerical_gates=len(gates),max_round_trip=max(float(r['round_trip']) for r in gates),max_anchor_residual=max(float(r['anchor_residual']) for r in gates),replayed_E29_chunks=len(replay),all_replayed_chunk_scalars_bit_identical=True,max_token_reduction_order_difference=max(abs(float(r['difference'])) for r in reduction),largest_result_file_bytes=largest,flagged_fraction=meta['flagged_fraction'],sha256=files))
    result=dict(status='PASS',utc=a.utc(),models=records,old_result_files_preserved=len(manifest['old_results_sha256']),old_report_prefix_preserved=True,analysis_source_sha256={str(p.relative_to(a.REPO)):a.sha(p) for p in [a.REPO/'nar/e34_report.py',Path(__file__)]})
    a.savej(a.REPO/'experiments/e34_final_verification.json',result)
    print(json.dumps({**result,'models':[{k:v for k,v in r.items() if k!='sha256'} for r in records]},indent=2))
if __name__=='__main__':verify()
