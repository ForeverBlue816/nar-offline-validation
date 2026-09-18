"""Assemble completed, auditable reviewer-ablation tables without editing old results."""
from __future__ import annotations
import argparse,json,math,sys,re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar.run_reviewer_experiments import plan,summarize

MODELS=['qwen3_4b_base','llama32_3b']
NAMES={'qwen3_4b_base':'Qwen3-4B-Base','llama32_3b':'Llama-3.2-3B','llama31_8b':'Llama-3.1-8B'}
def num(x):return f'{float(x):.6f}'
def interval(r):return f"{num(r['delta'])} [{num(r['ci90_low'])}, {num(r['ci90_high'])}]"
def contains_zero(r):return float(r['ci90_low'])<=0<=float(r['ci90_high'])
def mdtable(headers,rows):
    def cell(v):return str(v).replace('|',r'\|')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in rows])


def complete(exp,m):
    root=a.REPO/'results'/m;done=a.js(root/f'e{exp}_DONE.json');assert done['status']=='COMPLETE'
    rr=a.rows(root/f'e{exp}_per_sequence.csv');n=sum(len(s['evalsets']) for s in plan(exp,m))*3*64
    assert len(rr)==n,(m,exp,len(rr),n)
    summary=summarize(exp,m);return {(r['row'],r['eval_set']):r for r in summary}


def baselines(table):
    m='llama32_3b';old=a.rows(a.REPO/'results'/m/'e27_summary.csv');e20=a.rows(a.REPO/'results'/m/'e20_summary.csv');out=[]
    for method,oldmethod,old20 in [('hadamard','hadamard','hadamard_g128_m1'),('pq','A_full','nar_g128_m1')]:
        fresh=table['Q1_'+method,'wt2']
        r=next(r for r in old if r['site']=='both' and r['method']==oldmethod)
        d=float(fresh['mean_ppl'])-float(r['mean_ppl']);sigma=float(r['seed_ppl_std'])
        out.append(dict(model=m,method=method,new_ppl=fresh['mean_ppl'],reference='E27 both '+oldmethod,reference_ppl=r['mean_ppl'],reference_seed_std=sigma,delta=d,delta_over_reference_std=d/sigma,within_one_reference_seed_std=abs(d)<=sigma,source='results/llama32_3b/e27_summary.csv'))
        r=next(r for r in e20 if r['row']==old20);sigma=float(r['seed_ppl_std']);d=float(fresh['mean_ppl'])-float(r['mean_ppl'])
        out.append(dict(model=m,method=method,new_ppl=fresh['mean_ppl'],reference='E20 '+old20,reference_ppl=r['mean_ppl'],reference_seed_std=sigma,delta=d,delta_over_reference_std=d/sigma,within_one_reference_seed_std=abs(d)<=sigma,source='results/llama32_3b/e20_summary.csv'))
    a.write(a.REPO/'results'/m/'e29_baseline_comparison.csv',out);return out


def e29():
    lines=[];hyp=[];tables={m:complete(29,m) for m in MODELS}
    for m,t in tables.items():
        lines += [f'**{NAMES[m]}**',mdtable(['Quantizer','Hadamard PPL ± seed SD','PQ PPL ± seed SD','PQ − Hadamard [90% CI]'],[[f'Q{q}',f"{num(t[f'Q{q}_hadamard','wt2']['mean_ppl'])} ± {num(t[f'Q{q}_hadamard','wt2']['seed_std'])}",f"{num(t[f'Q{q}_pq','wt2']['mean_ppl'])} ± {num(t[f'Q{q}_pq','wt2']['seed_std'])}",interval(t[f'Q{q}_pq','wt2'])] for q in range(1,5)])]
        gain1=-float(t['Q1_pq','wt2']['delta']);gain2=-float(t['Q2_pq','wt2']['delta']);gain3=-float(t['Q3_pq','wt2']['delta'])
        hyp += [dict(model=m,hypothesis='H29a',supported=gain1>0 and gain2<=.25*gain1,gain_Q1=gain1,gain_Q2=gain2,ratio=gain2/gain1 if gain1 else None),dict(model=m,hypothesis='H29b',status='historical k=1 reference unavailable; attenuation reported',gain_Q3=gain3,Q3_gain_smaller_than_Q1=0<=gain3<gain1),dict(model=m,hypothesis='H29c',supported=contains_zero(t['Q4_pq','wt2']))]
    old=baselines(tables['llama32_3b']);lines += ['**Frozen baseline comparison (Llama-3.2-3B)**',mdtable(['Reference','New − old PPL','Old seed SD','Within one old SD'],[[r['reference'],num(r['delta']),num(r['reference_seed_std']),str(r['within_one_reference_seed_std'])] for r in old])]
    lines += ['Q1/Q2 use precisely the same saved rotations; Q3/Q4 use the same full-width rank-one rotations. Effective bits and the rank cap are recorded by site in `e29_metadata.json`. The Paley-II full-width PQ matrix requires a fixed DC sign normalization, recorded before PPL in `experiments/e29_paley_dc_preflight.json`. E27 shares the sign mapping, but uses saved calibration/fold arithmetic; E20 additionally uses a different sign offset and factor construction, so these differences are exposed rather than attributed solely to hardware.']
    return lines,hyp


def e30():
    m='qwen3_4b_base';t=complete(30,m);lines=['**A–B. Calibration size and corpus (both evaluation sets)**'];hyp=[]
    lines.append(mdtable(['Calibration','Eval set','PQ/Hadamard PPL','Seed SD','Δ vs Hadamard [90% CI]'],[[row,ev,num(r['mean_ppl']),num(r['seed_std']),interval(r)] for (row,ev),r in t.items()]))
    g={key:-float(r['delta']) for key,r in t.items()};base=g['wt2_n128','wt2']
    change=abs(g['wt2_n64','wt2']-g['wt2_n256','wt2'])
    hyp.append(dict(hypothesis='H30a',supported=base>0 and change<.2*base,gain_N64=g['wt2_n64','wt2'],gain_N128=base,gain_N256=g['wt2_n256','wt2'],change_over_N128=change/abs(base) if base else None))
    hyp.append(dict(hypothesis='H30b',supported=base>0 and g['c4_n128','wt2']>=.75*base and g['c4_n128','c4']>g['wt2_n128','c4'],retained_gain_fraction=g['c4_n128','wt2']/base if base else None,C4_eval_gain_C4cal=g['c4_n128','c4'],C4_eval_gain_WT2cal=g['wt2_n128','c4']))
    shift=a.rows(a.REPO/'results'/m/'e30_slot_summary.csv');assert len(shift)==2
    lines.append('**C. Captured fraction and slot reassignment**')
    lines.append(mdtable(['Site','Mean f, WT2','Mean f, C4','All-slot Jaccard','Anchor Jaccard','Changed fraction','Max |Δf|'],[[r['site'],num(r['f_wt2_mean']),num(r['f_c4_mean']),num(r['all_jaccard_mean']),num(r['anchor_jaccard_mean']),num(r['all_changed_fraction_mean']),num(r['f_absolute_difference_max'])] for r in shift]))
    hyp.append(dict(hypothesis='H30c',supported=all(float(r['all_changed_fraction_mean'])>0 and float(r['f_absolute_difference_max'])<.03 for r in shift),sites=shift))
    # Calibration seed zero must reuse the exact new E29 baseline measurement.
    x={(r['seed'],r['chunk']):r['nll'] for r in a.rows(a.REPO/'results'/m/'e29_per_sequence.csv') if r['row']=='Q1_pq' and r['seed']=='0'}
    y={(r['seed'],r['chunk']):r['nll'] for r in a.rows(a.REPO/'results'/m/'e30_per_sequence.csv') if r['row']=='wt2_n128' and r['seed']=='0' and r['eval_set']=='wt2'}
    assert x==y and len(x)==64
    reference=complete(29,m)['Q1_pq','wt2'];current=t['wt2_n128','wt2']
    difference=float(current['mean_ppl'])-float(reference['mean_ppl']);reference_std=float(reference['seed_std'])
    comparison=dict(reference='new E29 Q1 activation-only default; E22 W4A4KV4 is not comparable',E29_mean_ppl=reference['mean_ppl'],E30_mean_ppl=current['mean_ppl'],delta=difference,E29_seed_std=reference_std,within_one_reference_seed_std=abs(difference)<=reference_std,seed0_per_chunk_bit_identical=True,other_seeds='calibration subsets vary as pre-registered')
    a.savej(a.REPO/'results'/m/'e30_default_comparison.json',comparison)
    lines.append(f"The three-seed N=128 WT2 mean differs from the new E29 default by {num(difference)} PPL (E29 seed SD {num(reference_std)}; within one SD: {abs(difference)<=reference_std}).")
    lines.append('At k=max there are no unused filler slots: selected anchors remain fixed, and the residual-coordinate assignments provide the permutation comparison. The N=128 seed-0 WT2 tokens and PPL chunks match the new E29 activation-only reference exactly; seeds 1/2 use the pre-registered independently drawn nested calibration subsets. Historical E22 W4A4KV4 PPL is not an activation-only replication target. Calibration/evaluation C4 documents are disjoint; the optional local non-web corpus was unavailable at pre-registration.')
    return lines,hyp


def e31():
    m='qwen3_4b_base';t=complete(31,m);diag=a.rows(a.REPO/'results'/m/'e31_diagnostics.csv');lines=[];hyp=[]
    rows=[]
    for (row,ev),r in t.items():
        spread=[float(x['residual_group_energy_max_median']) for x in diag if x['row']==row and x['site']=='down']
        rows.append([row,num(r['mean_ppl']),num(r['seed_std']),interval(r) if row!='hadamard' else '—',num(np.median(spread)) if spread else '—'])
    lines.append(mdtable(['Row','PPL','Seed SD','Δ vs same-rank P1 [90% CI]','Down spread, median across layers/seeds'],rows))
    hyp.append(dict(hypothesis='H31a',supported=all(float(t[p+'_kmax','wt2']['delta'])>=0 for p in ['P2','P3']),P1_minus_P2=-float(t['P2_kmax','wt2']['delta']),P1_minus_P3=-float(t['P3_kmax','wt2']['delta'])))
    hyp.append(dict(hypothesis='H31b',supported=float(t['P4_k8','wt2']['delta'])>0 and float(t['P4_kmax','wt2']['delta'])==0,P4_minus_P1_k8=float(t['P4_k8','wt2']['delta']),P4_minus_P1_kmax=float(t['P4_kmax','wt2']['delta'])))
    for rank in ['8','max']:
        delta=float(t['P5_k'+rank,'wt2']['delta']);std=float(t['P1_k'+rank,'wt2']['seed_std'])
        hyp.append(dict(hypothesis='H31c',rank=rank,supported=abs(delta)<std,P5_minus_P1=delta,P1_seed_std=std))
    balancing=t['P2_kmax','wt2']
    lines.append(f"The explicitly requested balancing contrast is P1 − P2 at k=max = {num(-float(balancing['delta']))} [{num(-float(balancing['ci90_high']))}, {num(-float(balancing['ci90_low']))}] PPL (90% paired CI).")
    lines.append('Residual spread is max/median over residual-coordinate group energies after G, excluding each DC anchor; the displayed value is the median across down layers and seeds, and every underlying value is retained in the diagnostics CSV. The energy estimates use the frozen stride-32 calibration sample for the greedy step, while eigenspaces and perplexity use their complete specified samples. P4 at k=max is algebraically identical to P1, and its exact new measurement is reused with a hash audit. All variants share bit-identical G at each rank.')
    return lines,hyp


def e32():
    lines=[];hyp=[]
    for m in MODELS:
        t=complete(32,m);ds=a.rows(a.REPO/'results'/m/'e32_spectral_summary.csv');d={r['solver']:r for r in ds};rows=[]
        for (row,ev),r in t.items():
            x=d.get(row);rows.append([row, f"{float(x['ritz_residual_median_over_sites']):.3e}" if x else '—',num(x['principal_angle_max_deg']) if x else '—',num(x['f_down_mean']) if x else '—',num(r['mean_ppl']),num(r['seed_std']),interval(r)])
        lines += [f'**{NAMES[m]}**',mdtable(['Row','Ritz residual','Max angle (°)','Down f','PPL','Seed SD','Δ vs S4 [90% CI]'],rows)]
        delta=float(t['S3','wt2']['delta']);std=float(t['S4','wt2']['seed_std'])
        hyp.append(dict(hypothesis='H32a',model=m,supported=abs(delta)<std,S3_minus_S4=delta,S4_seed_std=std))
        if m=='qwen3_4b_base':
            excluded=[row for row in ['S1','S2','S3'] if not contains_zero(t[row,'wt2'])]
            largest=float(d['S1']['principal_angle_max_deg'])>=max(float(d[s]['principal_angle_max_deg']) for s in ['S2','S3'])
            hyp.append(dict(hypothesis='H32b',supported=largest and (not excluded or excluded==['S1']),largest_angle_is_S1=largest,CIs_excluding_zero=excluded))
            rr=a.rows(a.REPO/'results'/m/'e32_spectral_sites.csv');pairs=defaultdict(dict)
            for r in rr:pairs[r['site'],r['layer']][r['solver']]=float(r['captured_fraction'])
            violations=[dict(site=k[0],layer=k[1],f=v) for k,v in pairs.items() if any(v[s1]>v[s2]+1e-12 for s1,s2 in [('S1','S2'),('S2','S3'),('S3','S4')])]
            hyp.append(dict(hypothesis='H32c',supported=not violations,layer_sites_checked=len(pairs),violations=violations))
    lines.append('Ritz residual is the median of each site/layer’s top-k median residual; maximum residuals and every top-k residual/angle are also exported. The exact solver uses the same explicit fp64 uncentered second moment as the randomized solvers. S3 shares the newly measured E29 default; Llama adds only S4 beyond this shared comparison. Pass counts include the final Rayleigh–Ritz product, and all rows use oversampling 16 and the same starting sketch.')
    return lines,hyp


def e33():
    from nar.e33_rangelaw import summarize as sum33,grid
    for m in [*MODELS,'llama31_8b']:
        d=a.js(a.REPO/'results'/m/'e33_DONE.json');assert d['status']=='COMPLETE'
        assert d['rows']==len(grid(m))*3*len(a.keys(m))
    sum33();rr=a.rows(a.REPO/'results/e33_rangelaw_persite.csv');group=defaultdict(list)
    for r in rr:group[r['model'],r['site'],int(r['k'])].append(float(r['relative_error']))
    out=[]
    for (m,s,k),v in sorted(group.items()):out.append(dict(model=m,site=s,k=k,layer_configuration_points=len(v),median_absolute_relative_error=float(np.median(np.abs(v))),p90_absolute_relative_error=float(np.quantile(np.abs(v),.9)),median_signed_relative_error=float(np.median(v)),positive_error_fraction=float(np.mean(np.array(v)>0))))
    a.write(a.REPO/'results/e33_by_rank_summary.csv',out)
    lines=[mdtable(['Model','Site','k','Layer/config points','Median |e|','P90 |e|','Median signed e'],[[NAMES[r['model']],r['site'],r['k'],r['layer_configuration_points'],f"{100*r['median_absolute_relative_error']:.2f}%",f"{100*r['p90_absolute_relative_error']:.2f}%",f"{100*r['median_signed_relative_error']:+.2f}%"] for r in out])]
    hyp=[]
    for m in [*MODELS,'llama31_8b']:
        row='E22_kmax' if m.startswith('qwen') else 'E11_g128_kmax'
        vals=[float(r['relative_error']) for r in rr if r['model']==m and r['site']=='down' and r['row']==row]
        med=float(np.median(np.abs(vals)));hyp.append(dict(hypothesis='H33',model=m,row=row,site='down',supported=med<.05,median_absolute_error=med,median_signed_error=float(np.median(vals)),p90_absolute_error=float(np.quantile(np.abs(vals),.9)),layers=len(vals)))
    lines.append('The rank-only table pools all retained configurations with the same absolute k; `e33_error_summary.csv` additionally separates group size, multiple-Walsh configuration and source row. H33 uses each model’s default g=128, k=max configuration, and evaluates the median of absolute layer errors after paired seed averaging. The figure includes all new three-seed points and their 90% Student-t intervals; historical one-seed points remain separately archived without invented confidence intervals. The quantization step is the unrounded group range/15, consistent with the saved range diagnostics; no coefficient is fitted.')
    return lines,hyp


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=int,required=True,choices=range(29,34));arg=p.parse_args();exp=arg.experiment
    lines,hyp=globals()[f'e{exp}']();a.savej(a.REPO/'results'/f'e{exp}_hypotheses.json',hyp)
    lines += ['**Pre-registered hypothesis checks**',mdtable(['Hypothesis','Model / rank','Outcome'],[[r['hypothesis'],r.get('model',r.get('rank','all')),('supported' if r['supported'] else 'not supported') if 'supported' in r else r['status']] for r in hyp])]
    # Never alter the verbatim preregistration. Results are confined to a named block.
    start=f'<!-- E{exp} RESULTS START -->';end=f'<!-- E{exp} RESULTS END -->';body='\n\n'.join(lines)
    report=a.REPO/'report.md';text=report.read_text();block=f'{start}\n\n{body}\n\n{end}'
    if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda m:block,text,flags=re.S)
    else:
        marker=f'## E{exp} —';begin=text.index(marker);following=text.find('\n## ',begin+1)
        if following<0:following=len(text)
        section=text[begin:following].replace('Status: PRE-REGISTERED; no new measurements have run.','Status: measurements complete; all original hypotheses retained.')
        text=text[:begin]+section.rstrip()+'\n\n'+block+'\n\n'+text[following:]
    report.write_text(text)
if __name__=='__main__':main()
