#!/usr/bin/env python3
"""Trace frozen Figure 2/3 points and derive descriptive MoE summaries; CPU only."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE_COMMIT='519ddad56370051e59afbc83d18aa35db0456d55'
LAYERS=[1,5,9,13,18,22,27]
ORIGINAL_LAYERS=[1,13,27]
FAMILIES={'E1c activations':'Activations','E7 V cache':'V-cache','E20 multi-slot':'Multi-slot'}
FIT={'intercept':0.05980223276470587,'slope':0.8665147718600141,'r_squared':0.8613894973078933}


def read(path):return pd.read_csv(ROOT/path)
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
def r2(d):
    x=d.sqrt_one_minus_f.to_numpy();y=d.range_ratio_vs_hadamard.to_numpy()
    return float(np.corrcoef(x,y)[0,1]**2)


def build():
    files=['figures/fig2_capture.csv','figures/fig2b.csv','figures/fig2c.csv',
        'figures/fig3_token_projections.csv','figures/fig3_geometry_metadata.json',
        'figures/fig3_eigenspace_r256.csv','figures/fig3_range_law.csv','figures/fig3_range_law_moe.csv',
        'results/llama32_3b/e1c_per_layer.csv','results/llama32_3b/e1c_eigenspace.csv',
        'results/llama32_3b/e1c_range_vs_k.csv','results/llama32_3b/e7_range_vs_k.csv',
        'results/llama32_3b/e20_range_vs_config.csv',
        'results/qwen3_30b_a3b_base/e26_expert_range_law.csv',
        'results/qwen3_30b_a3b_base/e26_expert_f.csv','nar/e26_moe.py','nar/extended_experiment.py',
        'figures/prepare_fig3.py']
    for path in files:
        frozen=subprocess.check_output(['git','show',f'{SOURCE_COMMIT}:{path}'],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==digest(path),f'Frozen source changed: {path}'
    capture=read('figures/fig2_capture.csv')
    subset=capture[capture.model.eq('llama32_3b')&capture.site.eq('down')]
    assert len(subset)==84
    raw2=read('results/llama32_3b/e1c_per_layer.csv')
    names={'hadamard':'hadamard_full','duquant_style':'duquant','nar':'nar_kmax'}
    records=[]
    for letter,field in zip('abc',['f','mean_group_range','nmse']):
        for index,row in subset.iterrows():
            source='figures/fig2_capture.csv';line=int(index)+2;sourcefield=field
            if letter!='a':
                source='results/llama32_3b/e1c_per_layer.csv'
                match=raw2[raw2.layer.eq(row.layer)&raw2.site.eq('down_input')&raw2.method.eq(names[row.method])]
                assert len(match)==1
                sourcefield='mean_group_range' if letter=='b' else 'relative_quantization_error_nmse'
                close(row[field],match.iloc[0][sourcefield]);line=int(match.index[0])+2
            records.append(dict(panel=letter,model=row.model,layer=int(row.layer),method=row.method,
                metric=field,x=int(row.layer),y=float(row[field]),source_commit=SOURCE_COMMIT,
                source_file=source,source_csv_line=line,source_field=sourcefield,formula='identity',
                x_unit='layer index',y_unit='dimensionless' if letter!='b' else 'activation amplitude',
                round_for_display_only=True))
    pd.DataFrame(records).to_csv(HERE/'fig2_revised_data.csv',index=False)
    records=[]
    def append(frame,letter,family,source,xfield,yfield,**extra):
        for i,row in frame.iterrows():
            records.append(dict(panel=letter,family=family,source_file=source,source_csv_line=int(i)+2,
                source_commit=SOURCE_COMMIT,x=float(row[xfield]),y=float(row[yfield]),
                source_fields=f'{xfield};{yfield}',formula='identity',x_unit='dimensionless',y_unit='dimensionless',
                **{k:row[k] for k in ['model','site','layer','rank','expert'] if k in frame},**extra))
    projection=read('figures/fig3_token_projections.csv')
    for j,name in enumerate(['projection_v1','projection_v2']):
        projection[f'z{j+1}']=(projection[name]-projection[name].mean())/projection[name].std(ddof=0)
    append(projection,'a','Token projections','figures/fig3_token_projections.csv','z1','z2')
    for r in records:r.update(source_fields='projection_v1;projection_v2',formula='(coordinate-mean)/std(ddof=0), independently per axis',x_unit='standard deviations',y_unit='standard deviations')
    geometry=json.loads((HERE/'fig3_geometry_metadata.json').read_text())
    for name,v in geometry['arrows'].items():
        records.append(dict(panel='a',family=f'{name} direction',model='llama32_3b',layer=27,
            x=v['projection_v1'],y=v['projection_v2'],source_file='figures/fig3_geometry_metadata.json',
            source_commit=SOURCE_COMMIT,source_fields=f'arrows.{name}.projection_v1;arrows.{name}.projection_v2',
            formula='unit direction projected onto frozen v1/v2; shared display multiplier 1.0',
            x_unit='direction cosine',y_unit='direction cosine'))
    original=read('figures/fig3_eigenspace_r256.csv')
    append(original,'b','BOS excluded','figures/fig3_eigenspace_r256.csv','rank','cumulative_fraction_total_energy',protocol='BOS excluded; rank-256 randomized spectrum')
    energy=read('results/llama32_3b/e1c_eigenspace.csv')
    extra=energy[energy.site.eq('down_input')&energy.layer.isin([5,9,18,22])]
    assert len(extra)==256
    append(extra,'b','All tokens','results/llama32_3b/e1c_eigenspace.csv','rank','cumulative_fraction_total_energy',protocol='All tokens including BOS; rank-64 randomized spectrum')
    energy[energy.site.eq('down_input')&energy.layer.isin(LAYERS)].assign(source_commit=SOURCE_COMMIT,source_file='results/llama32_3b/e1c_eigenspace.csv').to_csv(HERE/'fig3_energy_all_token_context.csv',index=False)
    law=read('figures/fig3_range_law.csv')
    for family,part in law.groupby('source_family',sort=False):
        source=part.source_artifact.iloc[0];raw=read(source)
        for i,row in part.iterrows():
            if family=='E20 multi-slot':
                match=raw[raw.layer.eq(row.layer)&raw.site.eq(row.site)&raw.row.eq(row.configuration)]
                yf='range_ratio_vs_hadamard';formula='sqrt(1-absorbed_energy_fraction); range_ratio_vs_hadamard'
            else:
                k=int(row.configuration.removeprefix('nar_k'))
                match=raw[raw.layer.eq(row.layer)&raw.b.eq(row.group_size)&raw.k.eq(k)]
                if family=='E1c activations':match=match[match.site.eq(row.site)]
                yf='range_reduction_vs_hadamard';formula='sqrt(1-absorbed_energy_fraction); 1-range_reduction_vs_hadamard'
            assert len(match)==1,(family,i)
            v=match.iloc[0];close(row.absorbed_energy_fraction,v.absorbed_energy_fraction)
            close(row.sqrt_one_minus_f,np.sqrt(1-v.absorbed_energy_fraction))
            close(row.range_ratio_vs_hadamard,v[yf] if family=='E20 multi-slot' else 1-v[yf])
            records.append(dict(panel='c',family=FAMILIES[family],experiment_family=family,
                source_file=source,source_csv_line=int(match.index[0])+2,source_commit=SOURCE_COMMIT,
                source_fields=f'absorbed_energy_fraction;{yf}',formula=formula,
                x=row.sqrt_one_minus_f,y=row.range_ratio_vs_hadamard,f=row.absorbed_energy_fraction,
                model=row.model,site=row.site,layer=row.layer,configuration=row.configuration,
                x_unit='dimensionless',y_unit='paired range ratio'))
    fitted=np.linalg.lstsq(np.column_stack([np.ones(len(law)),law.sqrt_one_minus_f]),law.range_ratio_vs_hadamard,rcond=None)[0]
    close(fitted,[FIT['intercept'],FIT['slope']]);close(r2(law),FIT['r_squared'])
    source='results/qwen3_30b_a3b_base/e26_expert_range_law.csv'
    raw=read(source);old=read('figures/fig3_range_law_moe.csv')
    assert len(raw)==len(old)==5342
    close(raw[['layer','expert','rows']].to_numpy(),old[['layer','expert','routed_rows']].to_numpy())
    close(raw.f_at_k6,old.absorbed_energy_fraction);close(raw.range_nar_over_hadamard,old.range_ratio_vs_hadamard)
    close(np.sqrt(1-raw.f_at_k6),old.sqrt_one_minus_f)
    fsource='results/qwen3_30b_a3b_base/e26_expert_f.csv';cal=read(fsource)
    enriched=raw.merge(cal[['layer','expert','source']],on=['layer','expert'],validate='one_to_one')
    assert (enriched.source.eq('shrinkage')==enriched.routed_tokens.lt(2048)).all()
    moe=old.copy();moe['routed_tokens']=raw.routed_tokens;moe['calibration_source']=enriched.source
    moe['point_id']=[f'L{l:02d}-X{e:03d}' for l,e in zip(raw.layer,raw.expert)]
    # Ten equal-count bins, stable tie order by f, layer and expert. Every point belongs to one bin.
    order=moe.sort_values(['absorbed_energy_fraction','layer','expert'],kind='stable').index.to_numpy()
    moe['f_bin']=0
    for i,indices in enumerate(np.array_split(order,10),1):moe.loc[indices,'f_bin']=i
    for i,row in moe.iterrows():
        records.append(dict(panel='d',family='MoE experts',model=row.model,site=row.site,layer=row.layer,
            expert=row.expert,point_id=row.point_id,x=row.sqrt_one_minus_f,y=row.range_ratio_vs_hadamard,
            f=row.absorbed_energy_fraction,routed_rows=row.routed_rows,routed_tokens=row.routed_tokens,
            full_row=bool(row.routed_rows==256),cold=bool(row.routed_tokens<2048),calibration_source=row.calibration_source,
            f_bin=int(row.f_bin),source_file=source,source_csv_line=int(i)+2,source_commit=SOURCE_COMMIT,
            source_fields='f_at_k6;range_nar_over_hadamard;routed_tokens;rows',
            formula='sqrt(1-f_at_k6); range_nar_over_hadamard',x_unit='dimensionless',y_unit='paired range ratio'))
    summary=[]
    for bin_id,part in moe.groupby('f_bin',sort=True):
        summary.append(dict(f_bin=int(bin_id),n=len(part),f_min=part.absorbed_energy_fraction.min(),
            f_max=part.absorbed_energy_fraction.max(),f_median=part.absorbed_energy_fraction.median(),
            x_median=part.sqrt_one_minus_f.median(),y_median=part.range_ratio_vs_hadamard.median(),
            y_q1=part.range_ratio_vs_hadamard.quantile(.25),y_q3=part.range_ratio_vs_hadamard.quantile(.75),
            reference_y=FIT['intercept']+FIT['slope']*part.sqrt_one_minus_f.median(),
            median_y_over_sqrt= (part.range_ratio_vs_hadamard/part.sqrt_one_minus_f).median(),
            point_ids=';'.join(part.point_id),source_commit=SOURCE_COMMIT,
            rule='stable equal-count deciles of f; medians and linear-interpolated sample quartiles; no fitted/smoothed curve'))
    pd.DataFrame(summary).to_csv(HERE/'fig3_moe_binned_summary.csv',index=False)
    pd.DataFrame(records).assign(round_for_display_only=True).to_csv(HERE/'fig3_revised_data.csv',index=False)
    hot=moe.routed_tokens.ge(2048);full=moe.routed_rows.eq(256);high=moe.absorbed_energy_fraction.gt(.5)
    residual=moe.range_ratio_vs_hadamard-(FIT['intercept']+FIT['slope']*moe.sqrt_one_minus_f)
    diagnostics={}
    for name,mask in [('all',np.ones(len(moe),dtype=bool)),('full_rows_256',full),('fewer_than_256_rows',~full),('hot_tokens_ge_2048',hot),('cold_tokens_lt_2048',~hot),('f_le_0_5',~high)]:
        part=moe[mask];y=part.range_ratio_vs_hadamard;prediction=FIT['intercept']+FIT['slope']*part.sqrt_one_minus_f
        diagnostics[name]={'n':len(part),'within_subset_ols_r_squared':r2(part),
            'transferred_reference_predictive_r_squared':float(1-np.sum((y-prediction)**2)/np.sum((y-y.mean())**2)),
            'median_y_over_sqrt':float((y/part.sqrt_one_minus_f).median()),
            'x_variance_ddof0':float(part.sqrt_one_minus_f.var(ddof=0)),
            'reference_residual_sd_ddof0':float(residual[mask].std(ddof=0))}
    diagnostics['f_gt_0_5_count']=int(high.sum())
    diagnostics['cold_over_hot_reference_residual_sd']=float(residual[~hot].std(ddof=0)/residual[hot].std(ddof=0))
    diagnostics['dense_x_variance_ddof0']=float(law.sqrt_one_minus_f.var(ddof=0))
    metadata=dict(source_commit=SOURCE_COMMIT,source_hashes={f:digest(f) for f in files},
        count_fig2=252,count_fig3_by_panel=pd.DataFrame(records).groupby('panel').size().to_dict(),
        reference_fit=FIT,energy_layers=LAYERS,energy_original_layers=ORIGINAL_LAYERS,
        energy_protocols={'solid':'Original BOS-excluded rank-256, layers 1/13/27; 768 unchanged points',
                         'dashed':'Existing all-token rank-64, layers 5/9/18/22; 256 additional points; no continuation beyond measured rank'},
        moe_diagnostics=diagnostics,
        full_row_definition='rows == HELD_OUT_ROWS == 256 (original collector cap), predefined by source code, not selected by fit',
        cold_definition='routed_tokens < N0 == 2048, original shrinkage threshold',
        shrinkage='Sigma_e = (n_e Sigma_own + 2048 Sigma_layer_pool)/(n_e+2048) for cold experts',
        heldout_scope='MoE experts are excluded from the dense reference fit. Kept evaluation rows are part of the expert covariance collection; no independent calibration/test split is claimed.',
        original_expert_eligibility='Original range table includes n_e>=64: all 5342 available expert observations retained; remaining 802 of 6144 experts have no range observation, not a plotting exclusion.',
        r_squared_definition='Panel d reports within-subset OLS-with-intercept R² (equivalently squared Pearson correlation), descriptive only. Transferred-reference predictive R² is separately recorded; no MoE fitted line is drawn.',
        binning='Ten equal-count bins ordered by f/layer/expert; every point assigned; median x/y connected as descriptive polyline; Q1-Q3 shown as intervals, not confidence intervals.',
        display_precision='Only labels are rounded; computations and coordinates use frozen full precision.')
    (HERE/'fig2_fig3_source_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'fig2':252,'fig3':metadata['count_fig3_by_panel'],'moe_diagnostics':diagnostics},indent=2))
    return metadata

if __name__=='__main__':build()
