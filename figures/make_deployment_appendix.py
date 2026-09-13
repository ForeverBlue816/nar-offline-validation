#!/usr/bin/env python3
"""Retain E17 rank cost, 8B metadata, matched private modes, and every ablation."""
import argparse
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from deployment_figure_style import *
from make_deployment_efficiency import select
from make_fig4_revised import budget, budget_legend


def e17(data,draft):
    fig,ax=plt.subplots(figsize=(5.5,2.9));fig.subplots_adjust(left=.12,right=.96,bottom=.26,top=.78)
    axis(ax,'a','Original E17 rank cost','RTX PRO 6000 Blackwell Server · E17 v3')
    for model,marker,color,name in [('llama32_3b','o',BLUE,'Llama 3B'),('llama31_8b','s',STEEL,'Llama 8B')]:
        rows=data[data.model.eq(model)&data.kind.eq('kernel_share')].sort_values('k')
        ax.plot([0,1],rows.share_percent,color=color,marker=marker,lw=2.1,ms=5.5,label=f'{name}: PrismQuant')
        for x,y in zip([0,1],rows.share_percent):
            offset=(-14,-2) if model=='llama31_8b' and x==0 else (0,7 if model=='llama32_3b' else -13)
            ax.annotate(f'{y:.2f}%',(x,y),xytext=offset,textcoords='offset points',ha='right' if offset[0] else 'center',fontsize=7.5)
        h=float(data[data.model.eq(model)&data.kind.eq('hadamard_kernel_share')].share_percent.iloc[0])
        ax.axhline(h,color=color,ls='--',lw=1.7)
        ax.annotate(f'{name}: Hadamard {h:.2f}%',(1.38,h),xytext=(0,5 if model=='llama32_3b' else -12),textcoords='offset points',ha='right',fontsize=7.5,color=color)
    ax.set_xlim(-.35,1.45);ax.set_ylim(0,10);ax.set_xticks([0,1],['8','32']);ax.set_xlabel('Alignment rank k')
    ax.set_ylabel('Transform share of layer time (%)');ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.grid(False)
    fig.legend(loc='lower center',bbox_to_anchor=(.54,.02),ncol=2,fontsize=7.5)
    export(fig,HERE/'appendix/fig_e17_rank_cost',draft=draft)


def budget8(data,draft):
    fig,ax=plt.subplots(figsize=(3.4,4.65));fig.subplots_adjust(left=.18,right=.96,bottom=.28,top=.89)
    budget(ax,data,model='llama31_8b');budget_legend(fig)
    export(fig,HERE/'appendix/fig4_metadata_8b',draft=draft)


def matched(d,draft):
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.95));fig.subplots_adjust(left=.145,right=.975,bottom=.27,top=.78,wspace=.65)
    for letter,model,ax in zip('ab',['3b','8b'],axes):
        axis(ax,letter,f'{MODEL_NAMES[model]} decode','Private current-stream backend')
        for i,method in enumerate(['fp16','hadamard','nar']):
            values=[]
            for j,mode in enumerate(['eager_sequence','cuda_graph_sequence']):
                val,ss=select(d,'decode_latency',model,method,mode=mode);values.append(val)
                y=2-i
                span(ax,val,ss,y,horizontal=True,color=EDGES[method])
                ax.scatter([val],[y],s=29,marker='o' if j==0 else 's',facecolor='white' if j==0 else COLORS[method],edgecolor=EDGES[method],linewidth=1.1,zorder=5)
                ax.annotate(f'{val:.2f}',(val,y),xytext=(0,6 if j==0 else -14),textcoords='offset points',ha='center',fontsize=7.5)
            ax.plot(values,[2-i,2-i],color=EDGES[method],lw=1.7)
        ax.set_yticks([2,1,0],['FP16','Hadamard','PrismQuant']);ax.tick_params(axis='y',length=0)
        ax.set_xlim(0,62);ax.set_ylim(-.6,2.6);ax.set_xticks([0,20,40,60]);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel('Decode latency (ms/token)');ax.grid(False);ax.spines['left'].set_visible(False)
    handles=[Line2D([],[],marker='o',mfc='white',mec=INK,color=INK,ls='',label='Eager'),Line2D([],[],marker='s',color=INK,ls='',label='CUDA Graph')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.035),ncol=2)
    export(fig,HERE/'appendix/fig_matched_eager_graph',draft=draft,panel_axes=dict(zip('ab',axes)))


def all_kernels(d,draft):
    specs=[('E28 slot','R4 / full-width Hadamard','nar_module','hadamard_fp16'),
           ('E28 frontend','R4+Q / Hadamard+Q','nar_plus_quantizer','hadamard_plus_quantizer'),
           ('Dispatch','Prebound / generic launch','nar_prebound','nar_generic'),
           ('B implementation','R4: TC-B / shuffle-B','nar_generic','nar_generic_shuffle'),
           ('E17 native','Native R4 / block-Hadamard','nar_native','block_hadamard_native')]
    fig=plt.figure(figsize=(5.5,7.8));gs=fig.add_gridspec(3,2,left=.105,right=.975,bottom=.11,top=.92,hspace=.68,wspace=.42)
    axes=[]
    for idx,(scope,title,method,base) in enumerate(specs):
        ax=fig.add_subplot(gs[idx//2,idx%2]);axes.append(ax)
        axis(ax,chr(97+idx),scope,title)
        vals=[]
        for j,t in enumerate([1,2048,32768]):
            for i,model in enumerate(['3b','8b']):
                val,ss=select(d,'kernel_ratio',model,method,baseline=base,tokens=t);vals.extend(ss)
                y=5-(2*j+i)
                ax.plot([val,1],[y,y],color='#BDCBD6',lw=1.1)
                span(ax,val,ss,y,horizontal=True,color=STEEL)
                ax.scatter([val],[y],s=26,color=BLUE,marker='o' if model=='3b' else 's',zorder=6)
                ax.annotate(f'{val:.2f}',(val,y),xytext=(0,3.5),textcoords='offset points',ha='center',va='bottom',fontsize=7.5)
        ax.scatter([1]*6,list(range(6)),s=16,color=GRAY,edgecolor=GRAY_EDGE,linewidth=.8,zorder=1)
        ax.set_yticks(range(6),['32768 · 8B','32768 · 3B','2048 · 8B','2048 · 3B','1 · 8B','1 · 3B'],fontsize=7.5)
        ax.tick_params(axis='y',length=0);# Token count and model are explicit in each row label.
        ax.set_ylim(-.55,6);ax.set_xlim(0,max(1.18,max(vals)*1.15));ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.locator_params(axis='x',nbins=4)
        ax.set_xlabel('Elapsed-time ratio');ax.grid(False);ax.spines['left'].set_visible(False)
    # Deliberately leave the sixth grid cell blank: all five comparison families are retained.
    handles=[Line2D([],[],color=BLUE,marker='o',ls='',label='Llama 3B'),Line2D([],[],color=BLUE,marker='s',ls='',label='Llama 8B'),Line2D([],[],color=GRAY_EDGE,marker='o',mfc=GRAY,ls='',label='Own baseline = 1')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.025),ncol=3,fontsize=7.5)
    export(fig,HERE/'appendix/fig_kernel_all_shapes',draft=draft,panel_axes={chr(97+i):ax for i,ax in enumerate(axes)})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-data',action='store_true');ap.add_argument('--draft',action='store_true');args=ap.parse_args()
    style();d=pd.read_csv(HERE/'deployment_efficiency_data.csv');data=pd.read_csv(HERE/'fig4_revised_data.csv')
    e17(data,args.draft);budget8(data,args.draft);matched(d,args.draft);all_kernels(d,args.draft)

if __name__=='__main__':main()
