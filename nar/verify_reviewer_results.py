"""Final completeness, numerical-gate and preservation audit for E29–E33."""
from pathlib import Path
import sys,json,subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar.run_reviewer_experiments import plan
from nar.e33_rangelaw import grid


def main():
    audit=[]
    for m,experiments in [('qwen3_4b_base',[29,30,31,32]),('llama32_3b',[29,32])]:
        sites=len(a.keys(m))
        for exp in experiments:
            root=a.REPO/'results'/m;meta=a.js(root/f'e{exp}_DONE.json');assert meta['status']=='COMPLETE'
            rows=a.rows(root/f'e{exp}_per_sequence.csv');expected={(p['row'],str(s),ev,str(c)) for p in plan(exp,m) for s in a.SEEDS for ev in p['evalsets'] for c in range(64)}
            got={(r['row'],r['seed'],r['eval_set'],r['chunk']) for r in rows};assert len(rows)==len(got) and got==expected
            assert all(int(r['tokens'])==2047 and np.isfinite(float(r['nll'])) for r in rows)
            gates=a.rows(root/f'e{exp}_gates.csv');assert len(gates)==len(plan(exp,m))*3*sites
            assert all(float(r['round_trip'])<=1e-6 and float(r['anchor_residual'])<=1e-6 for r in gates)
            audit.append(dict(model=m,experiment=exp,chunks=len(rows),numerical_gates=len(gates),max_round_trip=max(float(r['round_trip']) for r in gates),max_anchor_residual=max(float(r['anchor_residual']) for r in gates),status='PASS'))
    for m in ['qwen3_4b_base','llama32_3b','llama31_8b']:
        root=a.REPO/'results'/m;meta=a.js(root/'e33_DONE.json');assert meta['status']=='COMPLETE'
        rows=a.rows(root/'e33_per_seed.csv');expected={(name,str(seed),site,str(layer)) for name,g,mm in grid(m) for seed in a.SEEDS for site,layer,n in a.keys(m)}
        got={(r['row'],r['seed'],r['site'],r['layer']) for r in rows};assert got==expected and len(rows)==len(got)
        assert all(int(r['observations'])==64*2048 for r in rows)
        for r in rows:
            predicted=float(r['s_hadamard'])*np.sqrt(max(0,1-float(r['f_calibration'])))
            assert np.isclose(predicted,float(r['s_pred']),rtol=1e-13,atol=0)
            assert np.isclose((predicted-float(r['s_meas']))/float(r['s_meas']),float(r['relative_error']),rtol=1e-12,atol=1e-14)
        gates=a.rows(root/'e33_gates.csv');assert len(gates)==len(rows) and all(float(r['round_trip'])<=1e-6 for r in gates)
        audit.append(dict(model=m,experiment=33,rows=len(rows),numerical_gates=len(gates),max_round_trip=max(float(r['round_trip']) for r in gates),status='PASS'))
    for m in ['qwen3_4b_base','llama32_3b']:
        rr=a.rows(a.REPO/'results'/m/'e32_ritz_and_angles.csv');exact=[float(r['ritz_residual']) for r in rr if r['solver']=='S4'];assert max(exact)<=1e-10
    # Compare only files that existed in the baseline commit. New files are permitted.
    baseline=subprocess.check_output(['git','ls-tree','-r','--name-only','bd1dfc2','results/'],cwd=a.REPO,text=True).splitlines()
    all_changes=subprocess.check_output(['git','diff','--name-only','bd1dfc2','--','results/'],cwd=a.REPO,text=True).splitlines()
    changed=sorted(set(baseline)&set(all_changes))
    assert not changed,changed
    a.savej(a.REPO/'experiments/e29_e33_final_verification.json',{'status':'PASS','rows':audit,'original_result_files_checked':len(baseline),'original_result_files_modified':changed,'utc':a.utc()})
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
