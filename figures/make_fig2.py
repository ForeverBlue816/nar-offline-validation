#!/usr/bin/env python3
"""Figure 2: unchanged measurements with a shared, unobstructed legend."""
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

def validate_data(data):
    subset = data[data.model.eq(MODEL) & data.site.eq(SITE)].copy()
    for method in ('hadamard', 'duquant_style', 'nar'):
        part = subset[subset.method.eq(method)]
        if set(part.layer.astype(int)) != set(range(LAYERS)) or len(part) != LAYERS:
            raise AssertionError(f'{method}: missing or duplicated layers')
    paired = subset[subset.method.isin(['hadamard', 'nar'])]
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
    configure_style(); data=validate_data(pd.read_csv(args.csv)); reductions={}
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
    metadata={'model':MODEL,'site':SITE,'layers':LAYERS,'group_size':GROUP,
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
