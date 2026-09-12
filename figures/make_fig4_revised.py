#!/usr/bin/env python3
"""Retain all accuracy observations; replace main cost panel with matched Graph data."""
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from deployment_figure_style import *
from make_deployment_efficiency import select
from build_e28_figure_data import build


def budget(ax,data,letter='a',model='llama32_3b'):
    axis(ax,letter,'Activation metadata budget','Llama-3.2-3B' if model=='llama32_3b' else 'Llama-3.1-8B')
    pts=data[data.kind.eq('point')&data.model.eq(model)]
    bf=float(data[data.kind.eq('bf16_reference')&data.model.eq(model)].ppl.iloc[0])
    for method,color,edge,marker in [('hadamard',TEAL,TEAL_EDGE,'o'),('nar',BLUE,BLUE,'s')]:
        part=pts[pts.method.eq(method)&pts.m.eq(1)].sort_values('effective_bits')
        ax.plot(part.effective_bits,part.ppl,color=edge if method=='hadamard' else color,lw=2.1,
                marker=marker,ms=5.4,mfc=color,mec=edge,mew=1,zorder=4)
        extra=pts[pts.method.eq(method)&pts.m.gt(1)]
        if method=='nar':
            for row in extra.itertuples():
                origin=pts[pts.method.eq(method)&pts.g.eq(row.g)&pts.m.eq(1)].iloc[0]
                ax.plot([origin.effective_bits,row.effective_bits],[origin.ppl,row.ppl],ls=(0,(1.5,2)),color=BLUE,lw=1.7,zorder=2)
        ax.plot(extra.effective_bits,extra.ppl,ls='none',marker='o' if method=='hadamard' else '^',ms=5.6,
                mfc='white' if method=='hadamard' else BLUE,mec=edge,mew=1.1,zorder=5)
    lower=7.60 if model=='llama32_3b' else 6.19
    upper=7.845 if model=='llama32_3b' else 6.435
    ax.set_ylim(lower,upper);ax.set_xlim(4.087,4.57)
    ax.set_xticks([4.125,4.25,4.375,4.5],['4.13','4.25','4.38','4.50']);ax.grid(False)
    ax.set_yticks(np.arange(7.60,7.801,.05) if model=='llama32_3b' else [6.20,6.25,6.30,6.35,6.40]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Activation bits / value');ax.set_ylabel('WikiText-2 perplexity')
    ax.axhline(bf,lw=1.7,color=GRAY_EDGE,ls=(0,(3,2)))
    ax.annotate('bf16 reference',(4.55,bf),xytext=(0,4),textcoords='offset points',ha='right',fontsize=7.5,color=GRAY_EDGE)
    def pick(method,g,m):return pts[pts.method.eq(method)&pts.g.eq(g)&pts.m.eq(m)].iloc[0]
    h=pick('hadamard',128,1);n=pick('nar',128,1)
    y=max(pts.ppl)+.028
    ax.plot([4.125,4.125,4.25,4.25],[y-.004,y,y,y-.004],lw=1.1,color=STEEL)
    ax.annotate('Scale resolution',((4.125+4.25)/2,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7.5,color=STEEL)
    x=4.295
    ax.plot([x-.009,x,x,x-.009],[n.ppl,n.ppl,h.ppl,h.ppl],color=STEEL,lw=1.1)
    offsets={('hadamard',256,1):(0,-12),('hadamard',256,2):(-7,9),('hadamard',256,3):(26,7),
        ('hadamard',128,1):(-4,-11),('hadamard',64,1):(0,10),('nar',256,1):(3,-19),
        ('nar',256,2):(-1,16),('nar',256,3):(36,2),('nar',128,1):(-6,-13),
        ('nar',128,2):(0,-13),('nar',64,1):(-5,-13)}
    for row in pts.itertuples():
        dx,dy=offsets[(row.method,int(row.g),int(row.m))]
        if model=='llama31_8b' and row.method=='nar' and int(row.g)==128 and int(row.m)==2:dy=-22
        ax.annotate(f'({int(row.g)}, {int(row.m)})',(row.effective_bits,row.ppl),xytext=(dx,dy),
                    textcoords='offset points',ha='center',va='center',fontsize=7.5)


def budget_legend(fig):
    handles=[Line2D([],[],color=TEAL_EDGE,marker='o',mfc=TEAL,lw=2.1,label='Hadamard'),
             Line2D([],[],color=TEAL_EDGE,marker='o',mfc='white',lw=0,label='Hadamard + directions'),
             Line2D([],[],color=BLUE,marker='s',lw=2.1,label='PrismQuant'),
             Line2D([],[],color=BLUE,marker='^',ls=':',lw=1.7,label='PrismQuant + directions')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.087,.035),fontsize=7.5,
                handlelength=1.6,labelspacing=.48,handletextpad=.6)


def recovery(ax,data):
    axis(ax,'b','Recovery versus rank')
    pts=data[data.kind.eq('recovery')]
    ax.axvspan(-.17,.17,color=GREEN,alpha=.14,lw=0,zorder=0)
    for model,(color,marker,ls,label) in MODEL_STYLE.items():
        part=pts[pts.model.eq(model)].copy();part['order']=part.k_category.map({x:i for i,x in enumerate(['8','16','32','64','max'])});part=part.sort_values('order')
        assert len(part)==5
        ax.plot(part.order,part.recovery_percent,color=color,marker=marker,ls=ls,lw=2.1,ms=5.2,mew=.8,label=label,zorder=4)
    ax.set_xlim(-.45,4.35);ax.set_ylim(0,109);ax.set_xticks(range(5),['8','16','32','64','max'])
    ax.set_yticks([0,25,50,75,100]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Alignment rank (categories)');ax.set_ylabel('Activation-only PPL\nrecovery (%)',labelpad=5)
    ax.text(.16,.075,'k=8 operating point',transform=ax.transAxes,fontsize=7.5,color=INK)


def overhead(ax,d):
    axis(ax,'c','Measured cost at k=8')
    ext=[]
    for model,y in [('3b',1),('8b',0)]:
        val,ss=select(d,'decode_overhead',model,'nar',mode='cuda_graph_sequence');ext.extend(ss)
        ax.plot([0,val],[y,y],color=BLUE,lw=1.1,zorder=2)
        span(ax,val,ss,y,horizontal=True,dots=True,color=STEEL)
        ax.scatter([val],[y],s=31,marker='o' if model=='3b' else 's',color=BLUE,zorder=6)
        ax.annotate(f'{val:.2f}%',(val,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=8)
        ax.text(.06,y+.23,MODEL_NAMES[model],fontsize=7.5,ha='left')
    ax.set_xlim(0,max(3.2,max(ext)+.55));ax.set_ylim(-.5,1.55);ax.set_yticks([]);ax.set_xticks([0,1,2,3]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Extra decode latency\nvs Hadamard (%)')
    ax.spines[['left','top','right']].set_visible(False);ax.grid(False)
    ax.annotate('A40 · CUDA Graph · k=8',(1,0),xycoords='axes fraction',xytext=(0,-40),textcoords='offset points',ha='right',fontsize=7.5,color=INK)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-data',action='store_true');ap.add_argument('--draft',action='store_true');args=ap.parse_args()
    if not args.reuse_data:build()
    typography=style();data=pd.read_csv(HERE/'fig4_revised_data.csv');d=pd.read_csv(HERE/'deployment_efficiency_data.csv')
    fig=plt.figure(figsize=(5.5,4.65))
    gs=fig.add_gridspec(2,2,left=.105,right=.975,bottom=.28,top=.89,width_ratios=[1.14,1],height_ratios=[1.65,1],wspace=.41,hspace=.65)
    a=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1])
    budget(a,data);recovery(b,data);overhead(c,d);budget_legend(fig)
    handles=[Line2D([],[],color=color,marker=marker,ls=ls,lw=2.1,label=name) for color,marker,ls,name in MODEL_STYLE.values()]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.605,.035),fontsize=7.5,handlelength=1.6,labelspacing=.48)
    export(fig,HERE/'fig4_revised',draft=args.draft,panel_axes={'a':a,'b':b,'c':c})
    meta=dict(source_commit=json.loads((HERE/'deployment_efficiency_metadata.json').read_text())['source_commit'],
        size_inches=[5.5,4.65],typography=typography,sharex=False,metadata_point_count=11,
        rank_point_count=15,k8_overhead={m:select(d,'decode_overhead',m,'nar',mode='cuda_graph_sequence')[0] for m in ['3b','8b']})
    (HERE/'fig4_revised_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
