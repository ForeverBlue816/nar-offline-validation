#!/usr/bin/env python3
"""Geometry, depth coverage, pooled reference and complete cross-model expert evidence."""
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from deployment_figure_style import *
from build_fig3_moe_summary import FIT,LAYERS,ORIGINAL_LAYERS

WARM='#A77B58';LILAC='#A99DBB'
LAYER_COLORS=['#A8DADC','#85BFC2','#6DA6B3','#568B9F','#457B9D','#355B7A','#1D3557']
QA=HERE/'qa/fig2_fig3';PANELS=HERE/'panels/fig2_fig3'


def sqrt_xlabel(ax):
    """Complete radical with vector overbar and editable 8.5-pt Times radicand."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.text import Text
    from matplotlib.textpath import TextPath
    prop=FontProperties(family='Times New Roman',weight='bold',size=8.5)
    width=7+TextPath((0,0),'1 − f',prop=prop).get_extents().width+1
    drawing=DrawingArea(width,12,0,0)
    drawing.add_artist(Text(7,1.7,'1 − f',fontproperties=prop,color=BLUE))
    drawing.add_artist(Line2D([0,1.6,3.2,5.7,width],[4.4,5.5,1.4,10.5,10.5],color=BLUE,lw=.85,solid_joinstyle='miter'))
    label=AnnotationBbox(drawing,(.5,0),xycoords=ax.transAxes,xybox=(0,-24),boxcoords='offset points',box_alignment=(.5,.5),frameon=False,annotation_clip=False,pad=0)
    ax.set_xlabel('');ax.add_artist(label)


def geometry(ax,d):
    axis(ax,'a','Quantizer-aware geometry','Llama-3.2-3B · down-input, layer 27');ax.grid(False)
    cloud=d[d.family.eq('Token projections')]
    ax.scatter(cloud.x,cloud.y,s=3.3,color=STEEL,alpha=.24,linewidths=0,rasterized=False)
    for family,color,width,head in [('hadamard direction',TEAL_EDGE,1.7,5),('nar direction',BLUE,2.3,8.5)]:
        row=d[d.family.eq(family)].iloc[0]
        ax.annotate('',xy=(row.x,row.y),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=color,lw=width,mutation_scale=head,shrinkA=0,shrinkB=0))
    xy=cloud[['x','y']].to_numpy();lo=xy.min(axis=0);hi=xy.max(axis=0);pad=.08*(hi-lo)
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    ax.axhline(0,color=GRAY,lw=1.1,zorder=0);ax.axvline(0,ymax=.65,color=GRAY,lw=1.1,zorder=0)
    ax.set_xticks([-3,0,3,6]);ax.set_yticks([0,6,12,18])
    ax.set_xlabel('Projection on v1 (s.d.)');ax.set_ylabel('Projection on v2 (s.d.)')
    ax.text(.045,.96,'In-plane direction length',ha='left',va='top',transform=ax.transAxes,fontsize=7.5)
    ax.text(.045,.875,'Hadamard  0.015',ha='left',va='top',transform=ax.transAxes,color=TEAL_EDGE,fontsize=7.5)
    ax.text(.045,.79,'PrismQuant  1.000',ha='left',va='top',transform=ax.transAxes,color=BLUE,fontsize=7.5)


def energy(ax,d,appendix=False):
    axis(ax,'a' if appendix else 'b','Energy across depth','Llama-3.2-3B · seven measured layers')
    ax.grid(False)
    for i,(layer,color) in enumerate(zip(LAYERS,LAYER_COLORS)):
        p=d[d.layer.eq(layer)].sort_values('x');assert len(p)==(64 if appendix or layer not in ORIGINAL_LAYERS else 256)
        ls='-' if appendix or layer in ORIGINAL_LAYERS else (0,(3.5,2))
        ax.plot(p.x,p.y,color=color,lw=2.3 if layer==27 else 1.85,ls=ls,
            marker=['o','s','^','D','v','P','X'][i],ms=3.8,markevery=[0,3,15,63]+([255] if len(p)==256 else []),mec=color,mew=.65,label=str(layer))
    ax.set_xscale('log');ax.set_xlim(.85,300 if not appendix else 75);ax.set_ylim(0,1.035);ax.minorticks_off()
    ax.set_xticks([1,4,16,64]+([] if appendix else [256]));ax.set_xticklabels(['1','4','16','64']+([] if appendix else ['256']))
    ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel('Directions retained, k');ax.set_ylabel('Cumulative energy')
    ax.legend(title='Layer',loc='upper left',bbox_to_anchor=(.02,.86 if appendix else 1.0),ncol=4,handlelength=1.25,handletextpad=.35,columnspacing=.8,labelspacing=.3,borderaxespad=0,frameon=False,fontsize=7.5,title_fontsize=7.5)
    if not appendix:
        ax.text(.04,.76,'Solid: non-BOS\nDashed: all tokens',transform=ax.transAxes,fontsize=7.5,color=INK,va='top',linespacing=1.35)


def law_axes(ax,letter,title,subtitle,ymax):
    axis(ax,letter,title,subtitle);ax.grid(False)
    ax.set_xlim(-.025,1.04);ax.set_ylim(-.025,ymax)
    ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1] if ymax<1.3 else [0,.5,1,1.5])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'));ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_ylabel('Range / Hadamard range');sqrt_xlabel(ax)
    ax.plot([0,1.02],[0,1.02],color=GRAY_EDGE,lw=1.5,ls=(0,(3,2)),zorder=1,label='Identity')
    grid=np.array([0,1.02]);ax.plot(grid,FIT['intercept']+FIT['slope']*grid,color=BLUE,lw=2.3,zorder=5,label='Reference fit')


def dense(ax,d):
    law_axes(ax,'c','Shared range law','Llama-3.2-3B · 2,912 observations',1.12)
    for family,color,marker,size,alpha in [('Activations',STEEL,'o',6,.23),('V-cache',TEAL_EDGE,'s',10,.5),('Multi-slot',LILAC,'D',14,.65)]:
        p=d[d.family.eq(family)]
        ax.scatter(p.x,p.y,color=color,marker=marker,s=size,alpha=alpha,edgecolors='none',rasterized=False,zorder=3,label=family)
    ax.text(.04,.965,'y = 0.060 + 0.867x\nPooled R² = 0.86',transform=ax.transAxes,ha='left',va='top',fontsize=8,color=BLUE,linespacing=1.4)
    h=[Line2D([],[],marker=m,color=c,ls='',ms=4.6,label=n) for n,c,m in [('Activations',STEEL,'o'),('V-cache',TEAL_EDGE,'s'),('Multi-slot',LILAC,'D')]]
    ax.legend(handles=h,loc='lower right',bbox_to_anchor=(1,.01),frameon=False,fontsize=7.5,handlelength=1.1,labelspacing=.4,borderpad=.2)


def experts(ax,d,summary,stats):
    ymax=max(1.12,float(d.y.max())+.045)
    law_axes(ax,'d','Across MoE experts','Qwen3-30B-A3B-Base · 5,342 experts',ymax)
    cold=d.routed_tokens.lt(2048)
    # Constant area avoids conflating calibration routed tokens with retained evaluation rows.
    ax.scatter(d.loc[cold,'x'],d.loc[cold,'y'],s=6,color=TEAL_EDGE,alpha=.22,edgecolors='none',rasterized=False,zorder=2)
    ax.scatter(d.loc[~cold,'x'],d.loc[~cold,'y'],s=6,color=STEEL,alpha=.23,edgecolors='none',rasterized=False,zorder=3)
    b=summary.sort_values('x_median')
    # The intervals are empirical quartiles, not uncertainty on the median.
    ax.vlines(b.x_median,b.y_q1,b.y_q3,color=WARM,alpha=.58,lw=2.5,zorder=6)
    ax.plot(b.x_median,b.y_median,color=WARM,lw=2.1,marker='o',ms=3.9,mfc='#F5E9DE',mec=WARM,mew=.8,zorder=7)
    ax.text(.04,.97,'Within-MoE R²',transform=ax.transAxes,va='top',fontsize=8)
    ax.text(.04,.885,f'256 rows: {stats["full_rows_256"]["within_subset_ols_r_squared"]:.2f}',transform=ax.transAxes,va='top',fontsize=8,color=BLUE)
    ax.text(.04,.80,f'All experts: {stats["all"]["within_subset_ols_r_squared"]:.2f}',transform=ax.transAxes,va='top',fontsize=7.5,color=INK)
    h=[Line2D([],[],color=BLUE,lw=2.3,label='Dense reference'),
       Line2D([],[],color=WARM,marker='o',ms=3.9,mfc='#F5E9DE',lw=2.1,label='Bin median + IQR')]
    ax.legend(handles=h,loc='upper right',bbox_to_anchor=(1,-.30),frameon=False,fontsize=7.5,handlelength=1.7,labelspacing=.5,borderpad=.2,borderaxespad=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draft',action='store_true');ap.add_argument('--reuse-data',action='store_true');args=ap.parse_args()
    typography=style();d=pd.read_csv(HERE/'fig3_revised_data.csv');summary=pd.read_csv(HERE/'fig3_moe_binned_summary.csv')
    source=json.loads((HERE/'fig2_fig3_source_metadata.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(5.5,6.05));fig.subplots_adjust(left=.12,right=.978,bottom=.16,top=.907,wspace=.39,hspace=.57)
    a,b,c,e=axes.flat
    geometry(a,d[d.panel.eq('a')]);energy(b,d[d.panel.eq('b')]);dense(c,d[d.panel.eq('c')]);experts(e,d[d.panel.eq('d')],summary,source['moe_diagnostics'])
    export(fig,HERE/'fig3_revised',draft=args.draft,panel_axes=dict(zip('abcd',axes.flat)),qa_directory=QA,panel_directory=PANELS)
    energy_all=pd.read_csv(HERE/'fig3_energy_all_token_context.csv').rename(columns={'rank':'x','cumulative_fraction_total_energy':'y'})
    fig,ax=plt.subplots(figsize=(5.5,2.9));fig.subplots_adjust(left=.115,right=.96,bottom=.235,top=.79)
    energy(ax,energy_all,appendix=True)
    ax.text(.62,.87,'All-token spectra (BOS included)\nAll curves share the same rank-64 protocol',transform=ax.transAxes,fontsize=7.5,ha='center',va='top',linespacing=1.4)
    export(fig,HERE/'appendix/fig3_energy_all_token_context',draft=args.draft,qa_directory=QA)
    meta=dict(source_commit=source['source_commit'],size_inches=[5.5,6.05],panels=list('abcd'),typography=typography,
        panel_point_counts=source['count_fig3_by_panel'],energy_layers=LAYERS,energy_protocols=source['energy_protocols'],
        reference_fit=FIT,moe_diagnostics=source['moe_diagnostics'],moe_binning=source['binning'],
        marker_area_pt2=6,marker_encoding='fixed area; cold teal (alpha .22), hot blue (alpha .23); every expert retained',
        moe_calibration_scope=source['heldout_scope'],minimum_font_pt=7.5,
        experiment_ids='Provenance tables only; excluded from figure text, legend and scientific caption.',
        source_tables=['fig3_revised_data.csv','fig3_moe_binned_summary.csv'])
    (HERE/'fig3_revised_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
