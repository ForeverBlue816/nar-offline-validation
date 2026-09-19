"""E34 fixed statistical decisions, loss attribution and report assembly."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar import e34_anchor_attribution as e
from nar.run_reviewer_experiments import stats,T90_DF191,T90_DF2
from nar.reviewer_report import mdtable,num
T90_DF63=1.669402221706
NAMES={'qwen3_4b_base':'Qwen3-4B-Base','llama32_3b':'Llama-3.2-3B'}

def ratio_summary(numerator,denominator):
    """Pooled ratio with 64 paired chunk clusters, including zero-count chunks."""
    u=np.asarray(numerator,dtype=float);d=np.asarray(denominator,dtype=float)
    assert u.shape==d.shape==(64,)
    if d.sum()==0:return dict(estimate=None,ci90_low=None,ci90_high=None,influence=None)
    value=u.sum()/d.sum();influence=(u-value*d)/d.mean()
    half=T90_DF63*influence.std(ddof=1)/8
    return dict(estimate=float(value),ci90_low=float(value-half),ci90_high=float(value+half),influence=influence)

def clean_ratio(r):return {k:v for k,v in r.items() if k!='influence'}
def matrix(rows,name):
    chosen=[r for r in rows if r['row']==name];assert len(chosen)==192
    out=np.empty((3,64));seen=set()
    for r in chosen:
        key=int(r['seed']),int(r['chunk']);assert key not in seen;seen.add(key);out[key]=float(r['nll'])
    return out

def ppl_influence(candidate,reference):
    cp=np.exp(candidate.mean(1));rp=np.exp(reference.mean(1));diff=cp-rp
    values=diff[:,None]+cp[:,None]*(candidate-candidate.mean(1)[:,None])-rp[:,None]*(reference-reference.mean(1)[:,None])
    return diff,values

def paired_ppl_combo(components):
    diffs=sum(sign*ppl_influence(c,r)[0] for sign,c,r in components)
    values=sum(sign*ppl_influence(c,r)[1] for sign,c,r in components)
    delta=float(diffs.mean());half=T90_DF191*values.std(ddof=1)/math.sqrt(192)
    sh=T90_DF2*diffs.std(ddof=1)/math.sqrt(3)
    return dict(delta=delta,ci90_low=float(delta-half),ci90_high=float(delta+half),seed_ci90_low=float(delta-sh),seed_ci90_high=float(delta+sh))

def attribution(m,flags):
    gains=np.empty((3,64,2047));offsets=np.empty_like(gains)
    token_ids=a.loadt(a.tokenpath(m,'wt2_eval')).numpy();massive=flags['massive'].numpy()
    old={(r['row'],int(r['seed']),int(r['chunk'])):float(r['nll']) for r in a.rows(e.root(m)/'e29_per_sequence.csv')}
    reductions=[]
    for seed in a.SEEDS:
        rows=a.rows(e.root(m)/f'e34_per_token_seed{seed}.csv');assert len(rows)==64*2047
        token_sums=np.zeros((4,64))
        for index,r in enumerate(rows):
            c,p=divmod(index,2047);assert int(r['chunk'])==c and int(r['predictor_position'])==p and int(r['target_position'])==p+1
            assert int(r['target_token_id'])==int(token_ids[c,p+1])
            assert r['predictor_class']==('BOS' if p==0 else ('massive' if massive[c,p] else 'other'))
            assert r['target_class']==('massive' if massive[c,p+1] else 'other')
            for j,field in enumerate(['nll_pq_asym','nll_had_asym','nll_pq_sym','nll_had_sym']):token_sums[j,c]+=float(r[field])
            gain=float(r['nll_had_asym'])-float(r['nll_pq_asym'])
            offset=(float(r['nll_pq_sym'])-float(r['nll_pq_asym']))-(float(r['nll_had_sym'])-float(r['nll_had_asym']))
            assert gain==float(r['gain']) and offset==float(r['offset_spec'])
            gains[seed,c,p]=gain;offsets[seed,c,p]=offset
        for j,old_row in enumerate(['Q1_pq','Q1_hadamard','Q2_pq','Q2_hadamard']):
            for c in range(64):
                mean=token_sums[j,c]/2047;reference=old[old_row,seed,c]
                assert abs(mean-reference)<=8*np.finfo(np.float32).eps*max(1.,abs(reference))
                reductions.append(dict(model=m,source_row=old_row,seed=seed,chunk=c,fp64_mean_of_token_losses=mean,frozen_fp32_chunk_nll=reference,difference=mean-reference,interpretation='reduction-order diagnostic; frozen scalar replay is separately bit-identical'))
    a.write(e.root(m)/'e34_token_reduction_audit.csv',reductions)
    gains=gains.mean(0);offsets=offsets.mean(0);out=[];chunk_rows=[];combined=None
    for basis in ['predictor','target']:
        sl=slice(None,-1) if basis=='predictor' else slice(1,None)
        mm=flags['massive'].numpy()[:,sl];bb=flags['bos'].numpy()[:,sl]
        masks={'BOS':bb,'massive':mm,'other':~(mm|bb),'BOS+massive':mm|bb}
        for name,mask in masks.items():
            counts=mask.sum(1);row=dict(model=m,index_basis=basis,token_class=name,token_count=int(counts.sum()),fraction_of_scored_positions=float(counts.sum()/gains.size),ci_unit='64 chunks; paired seeds averaged; ratio delta method; t63',BOS_note='BOS-associated first-text-token prediction' if basis=='predictor' else 'BOS target unscored: count zero, no invented loss')
            for field,values in [('gain',gains),('offset_spec',offsets)]:
                chunk=(values*mask).sum(1);total=values.sum(1);mean=ratio_summary(chunk,counts);share=ratio_summary(chunk,total)
                row[field+'_sum']=float(chunk.sum())
                for k,v in clean_ratio(mean).items():row[field+'_mean' if k=='estimate' else field+'_'+k]=v
                for k,v in clean_ratio(share).items():row[field+'_share' if k=='estimate' else field+'_share_'+k]=v if counts.sum() else None
                for c in range(64):chunk_rows.append(dict(model=m,index_basis=basis,token_class=name,chunk=c,metric=field,token_count=int(counts[c]),sum_seed_averaged_effect=float(chunk[c])))
            out.append(row)
            if basis=='predictor' and name=='BOS+massive':combined=row
    a.write(e.root(m)/'e34_attribution.csv',[r for r in out if r['index_basis']=='predictor'])
    a.write(e.root(m)/'e34_attribution_target_position.csv',[r for r in out if r['index_basis']=='target'])
    a.write(e.root(m)/'e34_attribution_per_chunk.csv',chunk_rows)
    a.write(e.root(m)/'e34_token_flags.csv',[dict(model=m,chunk=c,input_position=p,residual_linf=float(flags['scores'][c,p]),token_class='BOS' if p==0 else ('massive' if flags['massive'][c,p] else 'other'),selected_layer=flags['layer'],has_scored_next_token=p<2047) for c in range(64) for p in range(2048)])
    return out,combined,dict(gain=float(gains.sum()),offset_spec=float(offsets.sum()))

def range_summaries(m):
    out=[];ratios=[];means={}
    for scope,file in [('common_input','e34_range_common_input.csv'),('runtime','e34_range_runtime.csv')]:
        rows=a.rows(e.root(m)/file);grouped=defaultdict(list)
        for r in rows:grouped[r['row'],r['seed']].append(r)
        for (row,seed),rr in grouped.items():
            for cls in ['anchor','other']:
                chosen=[r for r in rr if r['group_class']==cls];weight=sum(int(r['observations']) for r in chosen)
                value=sum(float(r['mean_range'])*int(r['observations']) for r in chosen)/weight if weight else None
                out.append(dict(model=m,scope=scope,row=row,seed=seed,group_class=cls,layer_group_count=len(chosen),group_token_observations=weight,mean_range=value,status='MEASURED' if chosen else 'N/A: no groups in this stratum at k=max'))
        if scope=='common_input':
            for row in ['Q1','Q5a','Q5b','Had']:
                rr=[r for r in rows if r['row']==row];means[row]=float(np.mean([float(r['mean_range']) for r in rr]))
            for row in ['Q5a','Q5b']:
                ratios.append(dict(model=m,row=row,anchor_mean_range=means[row],Q1_anchor_mean_range=means['Q1'],anchor_range_ratio=means[row]/means['Q1'],anchor_exceeds_50_percent=means[row]/means['Q1']>1.5,other_group_count=0,other_group_ratio=None,other_clause='not identifiable at k=max'))
    a.write(e.root(m)/'e34_range_split.csv',out);a.write(e.root(m)/'e34_range_ratios.csv',ratios)
    return ratios,means

def analyze(m):
    assert a.js(e.root(m)/'e34_DONE.json')['status']=='COMPLETE'
    rows=a.rows(e.root(m)/'e34_per_sequence.csv');data={p['row']:matrix(rows,p['row']) for p in e.plan()};summary=[]
    for spec in e.plan():
        name=spec['row'];fmt='sym' if spec['symmetric'] else 'asym'
        reference=f'Q1_{fmt}' if spec['mask']=='all' else f"{spec['mask']}_had_{fmt}"
        summary.append(dict(model=m,row=name,reference=reference,**stats(data[name],data[reference])))
    a.write(e.root(m)/'e34_summary.csv',summary)
    flags=a.loadt(e.assets(m)/'flags.pt');attrs,combined,totals=attribution(m,flags)
    range_ratios,range_means=range_summaries(m)
    masks=[];rates={};mask_components=[]
    for mask in ['M1','M2']:
        pa,ha,ps,hs=[data[f'{mask}_{name}'] for name in ['pq_asym','had_asym','pq_sym','had_sym']]
        counts=flags['flagged'].numpy().sum(1).astype(float)
        if mask=='M2':counts=2048-counts
        effect=((ps-hs)-(pa-ha)).mean(0)*2047
        rates[mask]=ratio_summary(effect,counts)
        masks.append(dict(model=m,row=mask,quantized_input_positions=int(counts.sum()),**stats(pa,ha),offset_specific_nll_sum=float(effect.sum()),**{'offset_specific_per_quantized_token_'+k:v for k,v in clean_ratio(rates[mask]).items()}))
        mask_components.append((1,pa,ha))
    all_component=(1,data['Q1_asym'],data['Had_asym'])
    masks.append(dict(model=m,row='M1+M2',**paired_ppl_combo(mask_components)))
    masks.append(dict(model=m,row='all-token Q1',**stats(data['Q1_asym'],data['Had_asym'])))
    masks.append(dict(model=m,row='M1+M2 minus all-token Q1',**paired_ppl_combo(mask_components+[(-1,*all_component[1:])])))
    a.write(e.root(m)/'e34_masked_summary.csv',masks)
    rate_diff=rates['M1']['estimate']-rates['M2']['estimate'];influence=rates['M1']['influence']-rates['M2']['influence'];half=T90_DF63*influence.std(ddof=1)/8
    rate_contrast=dict(model=m,M1_minus_M2_per_quantized_token=rate_diff,ci90_low=float(rate_diff-half),ci90_high=float(rate_diff+half),ci_unit='64 paired chunk clusters, ratio delta method, seeds averaged')
    a.savej(e.root(m)/'e34_masked_offset_contrast.json',rate_contrast)
    ppl={k:float(np.exp(v.mean(1)).mean()) for k,v in data.items()};std=float(np.exp(data['Q1_sym'].mean(1)).std(ddof=1))
    h34a=[dict(row=row,difference=ppl[row+'_sym']-ppl['Q1_sym'],Q1_seed_std=std,supported=abs(ppl[row+'_sym']-ppl['Q1_sym'])<std) for row in ['Q5a','Q5b']]
    worst=max(['Q5a','Q5b'],key=lambda row:ppl[row+'_asym']);den=ppl['Had_asym']-ppl['Q1_asym'];assert den>0
    share=(ppl[worst+'_asym']-ppl['Q1_asym'])/den;decision='minor' if share<.25 else ('major' if share>.5 else 'two-part')
    h34d=dict(positive_total_offset_specific=totals['offset_spec']>0,total_offset_spec=totals['offset_spec'],flagged_signed_share=combined['offset_spec_share'],share_exceeds_half=combined['offset_spec_share']>.5 if combined['offset_spec_share'] is not None else False,class_fraction_below_one_percent=combined['fraction_of_scored_positions']<.01,M1_per_quantized_token_larger=rate_diff>0,**rate_contrast)
    h34d['supported']=all(h34d[k] for k in ['positive_total_offset_specific','share_exceeds_half','class_fraction_below_one_percent','M1_per_quantized_token_larger'])
    hypotheses=dict(model=m,H34a=h34a,H34b=dict(worse_variant=worst,share=share,decision=decision,denominator_hadamard_minus_Q1=den,Q5_minus_Q1=ppl[worst+'_asym']-ppl['Q1_asym']),H34c=dict(variants=range_ratios,other_clause='not identifiable; all groups anchored at k=max',full_hypothesis='not supported on anchor clause' if any(not r['anchor_exceeds_50_percent'] for r in range_ratios) else 'anchor clause supported; other clause not identifiable'),H34d=h34d)
    a.savej(e.root(m)/'e34_hypotheses.json',hypotheses)
    return dict(summary=summary,attribution=attrs,masks=masks,hypotheses=hypotheses,range_ratios=range_ratios,range_means=range_means,totals=totals,flags=a.js(e.root(m)/'e34_flags.json'))

def show(x,digits=6):return 'N/A' if x is None or x=='' else f'{float(x):.{digits}f}'
def ci(row):return f"{show(row['delta'])} [{show(row['ci90_low'])}, {show(row['ci90_high'])}]"
def mean_ci(row,field):return f"{show(row[field+'_mean'])} [{show(row[field+'_ci90_low'])}, {show(row[field+'_ci90_high'])}]"
def percentage(x):return 'N/A' if x is None else f'{100*float(x):.2f}%'

def report(results):
    lines=['All planned rows are complete. The original E29 chunk records are reused with source hashes; their independent E34 token-loss replays pass the bit-for-bit fp32 scalar gate. All old result files remain read only. The H34b decision bins apply to unrounded point estimates; the reported paired PPL intervals describe uncertainty and do not establish a sharp statistical separation at the 0.25 or 0.50 boundary.']
    lines.append('Both models use three paired seeds and the same frozen 64 chunks, with 16 fixed rows per model (6,144 chunk evaluations including exact baseline replays). PPL contrasts retain E29’s paired 3×64 delta-method intervals (t191); the same texts recur across seeds, and additional three-seed intervals are exported in the CSVs. Token attribution instead clusters over the 64 unique chunks after averaging seeds. Numerical gates, result hashes and preservation of all previous results are recorded in the [final audit](experiments/e34_final_verification.json); scheduler completion and pre-run hashes are in the [run manifest](experiments/e34_run_manifest.json) and [log excerpt](experiments/e34_job_log_excerpt.txt).')
    for m,result in results.items():
        h=result['hypotheses'];lookup={r['row']:r for r in result['summary']};f=result['flags']
        lines += [f'### {NAMES[m]}', '**A. Constant versus nonconstant target column**']
        table=[]
        for fmt in ['asym','sym']:
            for row in ['Q1','Q5a','Q5b','Had']:
                x=lookup[row+'_'+fmt];table.append([fmt,row,show(x['mean_ppl']),show(x['seed_std']),ci(x)])
        lines.append(mdtable(['Format','Row','PPL','Seed SD','Δ vs Q1 [90% paired CI]'],table))
        lines.append(mdtable(['Column','Common-input down range','Q1 range','Anchor range / Q1','Other groups'],[[r['row'],show(r['anchor_mean_range']),show(r['Q1_anchor_mean_range']),show(r['anchor_range_ratio']),'N/A (0 groups at k=max)'] for r in result['range_ratios']]))
        lines.append('Q5a has one sign transition within a group; Q5b alternates signs. Each moves the anchor by swapping exactly one residual coordinate in the same group; all other residual assignments, G and D are unchanged. The table uses matched unquantized inputs to avoid upstream-quantization confounding. An isolated nonconstant aligned level has range 2|c|, but extrema of its sum with residual activations do not add linearly, so a 50% increase of the total group range is an empirical hypothesis. Both the per-group common-input values and the actual-runtime values for every Part A row/seed/layer are exported, with empty strata explicitly marked N/A.')
        lines += ['**B. Where the next-token loss changes**',f"The fixed massive-token layer is **{f['selected_layer']} (zero based)**, chosen by the largest qkv DC share ({f['selected_dc_share']:.6f}) on the unquantized frozen inputs. There are {f['massive_count']} massive and {f['bos_count']} BOS input positions, totaling {percentage(f['flagged_fraction'])}; the massive fraction alone is {percentage(f['massive_fraction'])}. Of flagged positions, {f['unscored_flagged_final_positions']} occur at the unscored final predictor position."]
        ar=[r for r in result['attribution'] if r['index_basis']=='predictor']
        lines.append(mdtable(['Predictor class','Token count','Share of summed gain','Mean gain [90% CI]','Share of summed offset_spec','Mean offset_spec [90% CI]'],[[r['token_class'],r['token_count'],percentage(r['gain_share']),mean_ci(r,'gain'),percentage(r['offset_spec_share']),mean_ci(r,'offset_spec')] for r in ar]))
        lines.append(f"Summed seed-averaged gain = {result['totals']['gain']:.6f} NLL; summed offset_spec = {result['totals']['offset_spec']:.6f} NLL. Shares are signed ratios and are not percentages of an assumed positive benefit. BOS here means the loss of the first text token predicted from BOS; the frozen protocol has no BOS-target loss. The supplemental target-position table reports that missing BOS-target stratum explicitly. Counts are unique positions, and intervals cluster by 64 chunks after averaging paired seeds, rather than treating tokens/seeds as independent observations. The BOS+massive row is the union of the first two rows, not an additional disjoint class.")
        lines += ['**C. Whose input activation is quantized**',mdtable(['Mask/contrast','PQ − Had PPL [90% paired CI]'],[[r['row'],ci(r)] for r in result['masks']])]
        rates=[r for r in result['masks'] if r['row'] in ['M1','M2']]
        lines.append(mdtable(['Mask','Quantized input positions','Offset-specific NLL / quantized input [90% chunk CI]'],[[r['row'],r['quantized_input_positions'],f"{show(r['offset_specific_per_quantized_token_estimate'])} [{show(r['offset_specific_per_quantized_token_ci90_low'])}, {show(r['offset_specific_per_quantized_token_ci90_high'])}]"] for r in rates]))
        lines.append('M1 quantizes BOS+massive inputs; M2 quantizes their complement, with exactly the same fixed mask in every layer, at both sites, and across methods/seeds. Unmasked outputs retain the original bf16 values bit for bit. Matched symmetric controls make the offset-specific contrast identifiable. M1+M2 and its difference from all-token Q1 use joint paired influence values, preserving covariance; PPL effects need not be additive.')
        h_a='supported' if all(r['supported'] for r in h['H34a']) else 'not supported'
        lines += ['**Pre-registered decisions**',mdtable(['Hypothesis','Outcome'],[['H34a',h_a+'; Q5a/Q5b checks: '+', '.join(f"{r['row']} Δ={r['difference']:.6f}, Q1 SD={r['Q1_seed_std']:.6f}" for r in h['H34a'])],['H34b',f"{h['H34b']['decision']} contributor under the fixed rule; s={h['H34b']['share']:.6f}, worse variant={h['H34b']['worse_variant']}"],['H34c',h['H34c']['full_hypothesis']],['H34d','supported' if h['H34d']['supported'] else 'not supported']])]
        for_csv=f'results/{m}/'
        lines.append('Source tables: '+', '.join(f'[{label}]({for_csv}{file})' for label,file in [('all-row PPL','e34_summary.csv'),('group ranges','e34_range_common_input.csv'),('runtime ranges','e34_range_runtime.csv'),('range splits','e34_range_split.csv'),('token attribution','e34_attribution.csv'),('target-position attribution','e34_attribution_target_position.csv'),('masked contrasts','e34_masked_summary.csv'),('hypothesis details','e34_hypotheses.json')])+'. Per-token losses are in `e34_per_token_seed0/1/2.csv`; each row gives both predictor and target positions.')
        b=h['H34b'];d=h['H34d'];ratios=', '.join(f"{r['row']}={r['anchor_range_ratio']:.3f}×" for r in result['range_ratios'])
        lines.append(f"**Reading.** The worse nonconstant target ({b['worse_variant']}) changes asymmetric PPL by {b['Q5_minus_Q1']:.6f}, corresponding to s={b['share']:.3f} and the pre-registered **{b['decision']}** decision. The symmetric target-column check is {h_a}, while the paired common-input anchor ranges are {ratios}; the requested other-group comparison is unavailable because every group is anchored at k=max. The flagged classes account for {percentage(d['flagged_signed_share'])} of the signed offset-specific sum ({d['total_offset_spec']:.6f} NLL), and the total is {'positive' if d['positive_total_offset_specific'] else 'nonpositive'}. Matched masked controls give an M1−M2 offset-specific rate of {d['M1_minus_M2_per_quantized_token']:.6f} [{d['ci90_low']:.6f}, {d['ci90_high']:.6f}] NLL per quantized input, so H34d is {'supported' if d['supported'] else 'not supported'} under its full fixed rule. NLL at predictor position t reflects quantization at all positions up to t, so Part B locates changed losses, whereas Part C intervenes on quantized input positions and does not assume additive downstream effects.")
    lines.append('### Mechanism wording after the pre-registered decision')
    for m,r in results.items():
        b=r['hypotheses']['H34b'];kind=b['decision']
        wording={'minor':'At g=128, the observed advantage is primarily described in terms of within-group flatness and localization; the column intervention assigns only a minor share to constant-column offset absorption.','major':"At g=128, constant-column alignment makes a major contribution under the asymmetric format by the pre-registered intervention criterion. E29's comparable symmetric gain therefore cannot, by itself, rule out an offset contribution inside the asymmetric pipeline; its parity needs a separate explanation.",'two-part':'At g=128, the mechanism is described in two parts: localization/low crest factor and an additional constant-column benefit under the affine format.'}[kind]
        lines.append(f"**{NAMES[m]}.** {wording} The measured intervention ratio is {b['share']:.6f}. The minimal residual-coordinate swap, distributional changes and nonlinear propagation mean this ratio is not a pure additive causal fraction; the token and masking diagnostics above test additional, distinct parts of the mechanism.")
    lines.append('No manuscript main text is present in this repository; this section supplies the decision-conditioned replacement mechanism wording. Pre-registration, execution hashes and preservation checks accompany the results. No row was tuned or dropped after observing PPL.')
    path=a.REPO/'report.md';text=path.read_text();start='<!-- E34 RESULTS START -->';end='<!-- E34 RESULTS END -->';block=start+'\n\n'+'\n\n'.join(lines)+'\n\n'+end
    if start in text:text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
    else:text=text.rstrip()+'\n\n'+block+'\n'
    text=text.replace('Status: PRE-REGISTERED; no E34 measurements have run.','Status: measurements complete; fixed hypotheses and operational limitations retained.')
    path.write_text(text.rstrip()+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=e.MODELS);p.add_argument('--report',action='store_true');arg=p.parse_args()
    models=[arg.model] if arg.model else e.MODELS
    results={m:analyze(m) for m in models}
    if arg.report:
        assert set(results)==set(e.MODELS);report(results)
    for m,r in results.items():print(json.dumps(r['hypotheses'],indent=2))
if __name__=='__main__':main()
