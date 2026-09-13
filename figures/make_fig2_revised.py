#!/usr/bin/env python3
"""Final-size paired layer evidence; all frozen Figure 2 values unchanged."""
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from deployment_figure_style import *

PINK='#F5CBCB'
METHODS=[('hadamard','Hadamard',TEAL,TEAL,'o'),
         ('duquant_style','DuQuant',PINK,PINK,'D'),('nar','PrismQuant',BLUE,BLUE,'s')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draft',action='store_true');ap.add_argument('--reuse-data',action='store_true');args=ap.parse_args()
    typography=style();plt.rcParams.update({'axes.linewidth':.9,'xtick.major.width':.8,'ytick.major.width':.8});data=pd.read_csv(HERE/'fig2_revised_data.csv');source=json.loads((HERE/'fig2_fig3_source_metadata.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.55));fig.subplots_adjust(left=.095,right=.985,bottom=.23,top=.69,wspace=.62)
    titles=['Null-space energy','Activation range','INT4 error'];ylabs=['Energy fraction, f','Mean group range','Activation NMSE']
    reductions={}
    for ax,letter,title,ylabel in zip(axes,'abc',titles,ylabs):
        axis(ax,letter,title);ax.grid(False);p=data[data.panel.eq(letter)]
        for method,label,color,fill,marker in METHODS:
            part=p[p.method.eq(method)].sort_values('layer');assert len(part)==28
            ax.plot(part.x,part.y,color=color,lw=1.5 if method=='nar' else 1.05,
                marker=marker,ms=2.65 if method=='nar' else 2.45,mfc=fill,mec=color,mew=.45,label=label,zorder=5 if method=='nar' else 3)
        if letter=='a':
            ax.axhline(1/128,color=BLUE,lw=.85,ls=(0,(3,2)),zorder=0)
            ax.set_ylim(-.035,1.055);ax.set_yticks([0,.5,1]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            pairs=p.pivot(index='layer',columns='method',values='y')
            reduction=float((100*(pairs.hadamard-pairs.nar)/pairs.hadamard).mean());reductions[letter]=reduction
            ax.text(.98,1.015,f'↓ {reduction:.2f}% mean',ha='right',va='bottom',transform=ax.transAxes,color=BLUE,fontsize=7.5)
            ax.set_ylim(0,max(p.y)*1.13);ax.locator_params(axis='y',nbins=4)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if letter=='c' else '%.2f'))
        ax.set_xlim(-1,28);ax.set_xticks([0,9,18,27]);ax.set_xlabel('Layer index');ax.set_ylabel(ylabel)
        # Titles and letters use the same physical anchor across all three axes.
    handles=[Line2D([],[],color=color,marker=marker,mfc=fill,mec=color,mew=.45,lw=1.5 if method=='nar' else 1.05,ms=3.2,label=label)
        for method,label,color,fill,marker in [METHODS[2],METHODS[0],METHODS[1]]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.535,.995),ncol=3,frameon=False,handlelength=2,columnspacing=2.1)
    export(fig,HERE/'fig2_revised',draft=args.draft,panel_axes=dict(zip('abc',axes)),qa_directory=HERE/'qa/fig2_fig3',panel_directory=HERE/'panels/fig2_fig3')
    meta=dict(source_commit=source['source_commit'],size_inches=[5.5,2.55],panels=list('abc'),typography=typography,
        plotted_points=252,point_counts_per_panel=84,layer_indices=list(range(28)),mean_reduction_percent=reductions,
        line_width_pt={'PrismQuant':1.5,'comparators':1.05,'reference':.85,'axes':.9,'ticks':.8},
        marker_edge_width_pt=.45,palette_source='archive/fig2_fig3_before_oral_revision/fig2_metadata.json',
        minimum_font_pt=7.5,palette={'PrismQuant':BLUE,'Hadamard':TEAL,'DuQuant':PINK},
        source_table='fig2_revised_data.csv',statistics='Unchanged arithmetic mean of the 28 paired per-layer percentage reductions; no seed interval inferred.',
        diagnostic_scope='DuQuant is the existing DuQuant-style diagnostic, not the complete official implementation.')
    (HERE/'fig2_revised_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
