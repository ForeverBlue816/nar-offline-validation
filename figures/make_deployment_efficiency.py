#!/usr/bin/env python3
"""Final-width deployment evidence from the frozen E28 data; CPU only."""
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
from deployment_figure_style import *
from build_e28_figure_data import build


def select(data, metric, model=None, method=None, **kw):
    part=data[data.metric.eq(metric)]
    for key,val in dict(model=model,method=method,**kw).items():
        if val is not None:part=part[part[key].eq(val)]
    assert len(part)==4,(metric,model,method,kw,len(part))
    return float(part[part.session.eq(0)].value.iloc[0]),part[part.session.gt(0)].sort_values('session').value.to_numpy()


def implementation(fig):
    ax=fig.add_axes([.035,.867,.945,.125]);ax.set_axis_off()
    # All labels use real Times New Roman text, avoiding mathtext font substitution.
    boxes=[(.09,.36,.20,.42,'Kernel A','U = X A',True),
           (.35,.36,.27,.42,'Kernel B','Z = X H_block − U B',True),
           (.675,.36,.18,.42,'Shared INT4','token quantizer',False),
           (.90,.36,.10,.42,'INT4','GEMM',False)]
    for x,y,w,h,a,b,custom in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#E8F0F6' if custom else '#F0F2F4',edgecolor='#91A7B9' if custom else '#C4CDD6',lw=.8))
        ax.text(x+w/2,y+.28,a,ha='center',va='center',fontsize=7.5)
        ax.text(x+w/2,y+.11,b,ha='center',va='center',fontsize=7.5)
    ax.text(.016,.57,'X',ha='center',va='center',fontsize=9)
    for a,b in [((.033,.57),(.086,.57)),((.292,.57),(.346,.57)),((.623,.57),(.671,.57)),((.857,.57),(.896,.57))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=1,color=INK))
    ax.plot([.045,.045,.48],[.57,.95,.95],color=INK,lw=1)
    ax.add_patch(FancyArrowPatch((.48,.95),(.48,.795),arrowstyle='-|>',mutation_scale=7,lw=1,color=INK))
    ax.text(.646,.83,'FP16',ha='center',fontsize=7.5,color=INK)
    ax.text(.095,.08,'A, B: offline factors',fontsize=7.5,color=INK)
    ax.text(.675,.08,'symmetric',fontsize=7.5,color=INK)
    ax.text(.995,.08,'CUTLASS',fontsize=7.5,ha='right',color=INK)
    ax.set_xlim(0,1.005);ax.set_ylim(0,1.08)
    return ax


def prefill(ax,d):
    axis(ax,'a','Prefill throughput','2048 input tokens · eager')
    for i,(model,batch) in enumerate([('3b',1),('3b',16),('8b',1),('8b',16)]):
        for offset,method in [(-.215,'hadamard'),(.215,'nar')]:
            val,ss=select(d,'prefill_speedup',model,method,batch=batch)
            x=i+offset
            ax.bar(x,val,.35,color=COLORS[method],edgecolor=EDGES[method],linewidth=1,zorder=3)
            span(ax,val,ss,x)
            ax.text(x,max(val,max(ss))+.065,f'{val:.2f}',ha='center',va='bottom',fontsize=7.5)
    ax.axhline(1,ls=(0,(3,2)),lw=1,color=GRAY_EDGE,zorder=1)
    ax.set_xticks(range(4),['3B\nB1','3B\nB16','8B\nB1','8B\nB16'])
    ax.set_ylim(0,1.78);ax.set_yticks([0,.5,1,1.5]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Prefill speedup over FP16 (×)')
    ax.set_xlim(-.6,3.6)


def decode(ax,d):
    axis(ax,'b','Decode latency','Matched CUDA Graph · batch 1')
    centers={}
    for i,model in enumerate(['3b','8b']):
        for off,method in zip([-.30,0,.30],['fp16','hadamard','nar']):
            val,ss=select(d,'decode_latency',model,method,mode='cuda_graph_sequence');x=i+off
            centers[model,method]=val
            label_group=['fp16','hadamard','nar'] if model=='3b' else (['fp16'] if method=='fp16' else ['hadamard','nar'])
            group_top=max(select(d,'decode_latency',model,m,mode='cuda_graph_sequence')[0] for m in label_group)
            ax.bar(x,val,.26,color=COLORS[method],edgecolor=EDGES[method],linewidth=1,zorder=3)
            span(ax,val,ss,x)
            ax.text(x,group_top+.7,f'{val:.2f}',ha='center',va='bottom',fontsize=7.5)
    ratio=centers['8b','fp16']/centers['8b','nar']
    y=35.2
    ax.plot([.74,.74,1.26,1.26],[y-.4,y,y,y-.4],color=BLUE,lw=1.1)
    ax.text(1,y+.45,f'{ratio:.2f}× vs FP16',ha='center',va='bottom',fontsize=7.5)
    ax.set_xticks([0,1],['Llama 3B','Llama 8B']);ax.set_xlim(-.62,1.62)
    ax.set_ylim(0,40);ax.set_yticks([0,10,20,30]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Decode latency (ms/token)');ax.grid(False)
    ax.axhline(10,color='#DCE4EA',lw=.45,zorder=0)


def memory(ax,d):
    axis(ax,'c','Decode peak memory','Matched CUDA Graph · allocated')
    for i,model in enumerate(['3b','8b']):
        for off,method in zip([-.30,0,.30],['fp16','hadamard','nar']):
            val,ss=select(d,'peak_memory',model,method,mode='cuda_graph_sequence');x=i+off
            ax.bar(x,val,.26,color=COLORS[method],edgecolor=EDGES[method],linewidth=1,zorder=3)
            span(ax,val,ss,x)
            ax.text(x,max(val,max(ss))+.38,f'{val:.2f}',ha='center',va='bottom',fontsize=7.5)
        saving,_=select(d,'memory_saving',model,'nar',mode='cuda_graph_sequence')
        val,_=select(d,'peak_memory',model,'nar',mode='cuda_graph_sequence')
        ax.text(i+.25,val+2.6,f'−{saving:.2f}%',ha='center',va='bottom',fontsize=7.5,color=BLUE)
    ax.set_xticks([0,1],['Llama 3B','Llama 8B']);ax.set_xlim(-.62,1.62)
    ax.set_ylim(0,21.5);ax.set_yticks([0,5,10,15,20]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Peak GPU memory (GB)');ax.grid(False)


def ablation(ax,d):
    axis(ax,'d','Kernel implementation','Own baseline = 1.00 · wall time')
    # Axes y-space holds the two explicit baseline groups and model rows.
    specs=[('3b','nar_prebound','nar_generic',1,3.1),('8b','nar_prebound','nar_generic',1,2.2),
           ('3b','nar_generic','nar_generic_shuffle',2048,.7),('8b','nar_generic','nar_generic_shuffle',2048,-.2)]
    for model,method,base,t,y in specs:
        val,ss=select(d,'kernel_ratio',model,method,baseline=base,tokens=t)
        ax.plot([val,1],[y,y],color='#BDCBD6',lw=1.2,zorder=1)
        ax.scatter([1],[y],s=21,facecolor=GRAY,edgecolor=GRAY_EDGE,linewidth=.8,zorder=2)
        span(ax,val,ss,y,horizontal=True,dots=True,color=STEEL)
        ax.scatter([val],[y],s=28,marker='o' if model=='3b' else 's',color=BLUE,zorder=7)
        ax.annotate(f'{val:.2f}',(val,y),xytext=(0,3.5),textcoords='offset points',ha='center',va='bottom',fontsize=7.5)
    ax.text(.03,4.3,'Prebound / generic launch · T=1',fontsize=7.5,ha='left')
    ax.text(.03,1.6,'R4: TC-B / shuffle-B · T=2048',fontsize=7.5,ha='left')
    ax.set_yticks([3.1,2.2,.7,-.2],['3B','8B','3B','8B']);ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,1.1);ax.set_ylim(-.7,4.9);ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Elapsed-time ratio (lower is better)');ax.grid(False)
    ax.spines[['left','top','right']].set_visible(False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-data',action='store_true');ap.add_argument('--draft',action='store_true');args=ap.parse_args()
    if not args.reuse_data:build()
    typography=style();d=pd.read_csv(HERE/'deployment_efficiency_data.csv')
    fig=plt.figure(figsize=(5.5,5.35));gs=fig.add_gridspec(2,2,left=.105,right=.975,bottom=.14,top=.79,wspace=.44,hspace=.76)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    strip=implementation(fig)
    for fn,ax in zip([prefill,decode,memory,ablation],axes):fn(ax,d)
    method_legend(fig,.028)
    export(fig,HERE/'fig_deployment_efficiency',draft=args.draft,exclude_axes=[strip],panel_axes={**dict(zip('abcd',axes)),'implementation':strip})
    meta=json.loads((HERE/'deployment_efficiency_metadata.json').read_text());meta['typography']=typography
    meta['size_inches']=[5.5,5.35]
    (HERE/'deployment_efficiency_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
