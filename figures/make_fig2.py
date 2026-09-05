#!/usr/bin/env python3
"""Figure 2: matched offline DuQuant diagnostics, guarded by the range check."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from figure_style import PALETTE, clean_2d_axis, configure_style, resolved_serif_family, save_panel
MODEL, SITE, LAYERS, GROUP = 'llama32_3b', 'down', 28, 128

def add_duquant_diagnostics(data, here):
    root=here.parent;completed=[];missing=[]
    for model in ['llama32_3b','llama31_8b']:
        source=root/'results'/model/'e1c_per_layer.csv'
        done=root/'results'/model/'E1C_DONE.json'
        if not source.exists() or not done.exists():
            missing.append(model);continue
        metadata=json.loads(done.read_text()).get('duquant_addendum')
        if metadata is None:
            missing.append(model);continue
        if metadata['plotting_gate']!='PASS':
            raise RuntimeError(f'{model}: DuQuant lies outside the required range bracket; do not plot')
        raw=pd.read_csv(source);du=raw[raw.method.eq('duquant')]
        if len(du)!=2*int(metadata['layers']): raise AssertionError('Incomplete DuQuant addendum')
        for index,row in du.iterrows():
            site={'q_input':'qkv','down_input':'down'}[row.site]
            mask=data.model.eq(model)&data.site.eq(site)&data.layer.eq(row.layer)&data.method.eq('duquant_style')
            assert int(mask.sum())==1
            data.loc[mask,'mean_group_range']=row.mean_group_range
            data.loc[mask,'nmse']=row.relative_quantization_error_nmse
            data.loc[mask,'frozen_range']=row.mean_group_range
            data.loc[mask,'frozen_nmse']=row.relative_quantization_error_nmse
            data.loc[mask,'frozen_rows']=row.evaluation_tokens
            data.loc[mask,'evaluation_tokens']=row.evaluation_tokens
            data.loc[mask,'quantitative_source']=f'results/{model}/e1c_per_layer.csv:{index+2}'
        completed.append(model)
    if MODEL not in completed: raise RuntimeError('Main-panel DuQuant measurements have not passed the plotting gate')
    return data,{'completed_models':completed,'missing_frozen_e1c_models':missing}

def write_metric_csv(data,here):
    source=here.parent/'results'/MODEL/'e1c_per_layer.csv';raw=pd.read_csv(source)
    site={'qkv':'q_input','down':'down_input'}[SITE]
    names={'hadamard':'hadamard_full','duquant_style':'duquant','nar':'nar_kmax'}
    for letter,column,source_column in [('b','mean_group_range','mean_group_range'),('c','nmse','relative_quantization_error_nmse')]:
        rows=[]
        for point in data.itertuples():
            match=raw[raw.site.eq(site)&raw.layer.eq(point.layer)&raw.method.eq(names[point.method])]
            assert len(match)==1
            source_row=match.iloc[0];value=float(getattr(point,column))
            np.testing.assert_allclose(value,source_row[source_column],rtol=1e-12,atol=1e-12)
            rows.append({'model':MODEL,'site':SITE,'layer':point.layer,'method':names[point.method],column:value,
                'evaluation_tokens':int(source_row.evaluation_tokens),'source_file':f'results/{MODEL}/e1c_per_layer.csv',
                'source_csv_line':int(match.index[0])+2,'source_row':f'{site};layer={point.layer};method={names[point.method]}',
                'source_column':source_column})
        pd.DataFrame(rows).to_csv(here/f'fig2{letter}.csv',index=False)

def validate_data(data):
    subset = data[data.model.eq(MODEL) & data.site.eq(SITE)].copy()
    for method in ('hadamard', 'duquant_style', 'nar'):
        part = subset[subset.method.eq(method)]
        if set(part.layer.astype(int)) != set(range(LAYERS)) or len(part) != LAYERS:
            raise AssertionError(f'{method}: missing or duplicated layers')
    paired = subset[subset.method.isin(['hadamard', 'duquant_style', 'nar'])]
    if not np.isfinite(paired[['mean_group_range','nmse']].to_numpy()).all():
        raise AssertionError('Nonfinite measurement')
    return subset

def paired_metric(data, column):
    p = data[data.method.isin(['hadamard','nar'])].pivot(index='layer',columns='method',values=column).sort_index()
    if (p.hadamard <= 0).any(): raise AssertionError('Nonpositive ratio denominator')
    return p.index.to_numpy(),p.hadamard.to_numpy(),p.nar.to_numpy(),float((100*(p.hadamard-p.nar)/p.hadamard).mean())

def draw(ax, data, column):
    clean_2d_axis(ax)
    if column == 'f':
        ax.axhline(1/GROUP,color=PALETTE['reference'],lw=.65,ls=(0,(3,2)),zorder=0)
        methods = [('hadamard','Hadamard'),('duquant_style','DuQuant'),('nar','PrismQuant')]
        for method,label in methods:
            p=data[data.method.eq(method)].sort_values('layer')
            color=PALETTE[{'hadamard':'hadamard','duquant_style':'duquant','nar':'prismquant'}[method]]
            ax.plot(p.layer,p.f,color=color,lw=1.8 if method=='nar' else 1.05,
                    marker='o',ms=2.7 if method=='nar' else 2.2,mec='white',mew=.2,label=label)
        ax.set_ylabel('null-space energy fraction, f')
        reduction=None
    else:
        x,h,p,reduction=paired_metric(data,column)
        ax.plot(x,h,color=PALETTE['hadamard'],lw=1,marker='o',ms=2.2)
        d=data[data.method.eq('duquant_style')].sort_values('layer')
        ax.plot(d.layer,d[column],color=PALETTE['duquant'],lw=1,marker='o',ms=2.2)
        ax.plot(x,p,color=PALETTE['prismquant'],lw=1.8,marker='o',ms=2.7,mec='white',mew=.2)
        ax.set_ylabel('mean group range' if column=='mean_group_range' else 'activation NMSE')
        ax.text(.98,1.06,f'mean reduction {reduction:.1f}%',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=7)
    ax.set_xlim(-.8,LAYERS-.2); ax.set_xticks([0,27]); ax.set_xlabel('layer index')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if column=='nmse' else '%.2f'))
    return reduction

def shared_legend(fig):
    handles=[Line2D([],[],color=PALETTE[k],lw=1.8 if k=='prismquant' else 1.05,marker='o',ms=2.5,label=label)
             for k,label in [('prismquant','PrismQuant'),('hadamard','Hadamard'),('duquant','DuQuant')]]
    legend=fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.985,.995),ncol=3,
                      fontsize=6.5,frameon=True,edgecolor=PALETTE['pane_edge'],framealpha=1,
                      handlelength=1.3,handletextpad=.4,columnspacing=1,borderpad=.4)
    legend.get_frame().set_linewidth(.6)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--csv',type=Path,default=Path(__file__).with_name('fig2_capture.csv'))
    args=parser.parse_args(); here=Path(__file__).resolve().parent
    configure_style(); complete,addendum=add_duquant_diagnostics(pd.read_csv(args.csv),here)
    data=validate_data(complete);write_metric_csv(data,here);reductions={}
    complete.to_csv(args.csv,index=False)
    for letter,column in zip('abc',['f','mean_group_range','nmse']):
        fig,ax=plt.subplots(figsize=(1.85,1.72)); fig.subplots_adjust(left=.28,right=.96,bottom=.25,top=.80)
        reductions[column]=draw(ax,data,column); save_panel(fig,here/f'fig2{letter}')
    fig,axes=plt.subplots(1,3,figsize=(6.4,2.08)); fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.71,wspace=.56)
    for ax,letter,column in zip(axes,'abc',['f','mean_group_range','nmse']):
        draw(ax,data,column)
        fig.text(ax.get_position().x0,.835,f'({letter})',fontsize=7,fontweight='bold')
    shared_legend(fig)
    save_panel(fig,here/'fig2')
    # Same physical size and geometry as the manuscript assembly.
    (here/'fig2_preview.png').write_bytes((here/'fig2.png').read_bytes())
    metadata={'duquant_offline_addendum':addendum,'model':MODEL,'site':SITE,'layers':LAYERS,'group_size':GROUP,
              'mean_range_reduction_percent':reductions['mean_group_range'],
              'mean_nmse_reduction_percent':reductions['nmse'],
              'font_family_resolved':resolved_serif_family(), 'palette':PALETTE,
              'legend':'figure upper-right, dedicated top strip, 6.5 pt, boxed; PrismQuant / Hadamard / DuQuant',
              'duquant_display_label':'DuQuant','duquant_implementation':'DuQuant-style diagnostic, not official full DuQuant; source method duquant_style',
              'statistics':'same 28 measured layers; arithmetic mean of paired per-layer percentage decreases; no seed CI inferred',
              'reference_line':'1 / group size, retained without label'}
    (here/'fig2_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
