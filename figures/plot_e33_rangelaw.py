"""Unfitted range-law validation; reads only complete measured three-seed data."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter
from deployment_figure_style import style,export

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
MODELS=[('llama32_3b','Llama-3.2-3B'),('llama31_8b','Llama-3.1-8B'),('qwen3_4b_base','Qwen3-4B-Base')]
CMAPS={'qkv':LinearSegmentedColormap.from_list('qkv_rank',['#B9CEDB','#1D3557']),
       'down':LinearSegmentedColormap.from_list('down_rank',['#E4C6D6','#601D49'])}
MARKERS={'qkv':'o','down':'^'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');arg=p.parse_args()
    src=ROOT/'results/e33_rangelaw_persite.csv'
    rows=list(csv.DictReader(src.open()));assert rows,'No measurements'
    for model,_ in MODELS:
        done=json.loads((ROOT/'results'/model/'e33_DONE.json').read_text());assert done['status']=='COMPLETE'
        assert sum(r['model']==model for r in rows)*3==done['rows'],'Missing aggregate rows'
    assert all(int(r['seeds'])==3 for r in rows)
    style();plt.rcParams.update({'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'lines.linewidth':.7})
    # Shade is linear in log2(k+1); the labeled color scales carry absolute rank.
    max_k=max(int(r['k']) for r in rows);norm=Normalize(0,np.log2(max_k+1))
    fig,axes=plt.subplots(1,3,figsize=(6.6,3.15))
    fig.subplots_adjust(left=.09,right=.98,bottom=.34,top=.84,wspace=.36)
    counts={};bounds={};primary_errors={}
    for ax,(model,title),letter in zip(axes,MODELS,'abc'):
        rr=[r for r in rows if r['model']==model];counts[model]=len(rr)
        fields=['s_meas','s_pred','s_meas_ci90_low','s_meas_ci90_high','s_pred_ci90_low','s_pred_ci90_high']
        all_values=np.array([float(r[f]) for r in rr for f in fields]);assert np.all(all_values>0),'Nonpositive seed CI requires a linear-scale design'
        lo=10**(np.log10(all_values.min())-.13);hi=10**(np.log10(all_values.max())+.13)
        bounds[model]=[lo,hi];ax.set_box_aspect(1);ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi))
        ax.plot([lo,hi],[lo,hi],color='#77828C',ls=(0,(3,3)),lw=.8,zorder=1)
        for r in sorted(rr,key=lambda r:(int(r['k']),r['site'])):
            site=r['site'];k=int(r['k']);color=CMAPS[site](norm(np.log2(k+1)))
            x,y=float(r['s_meas']),float(r['s_pred'])
            xe=np.array([[x-float(r['s_meas_ci90_low'])],[float(r['s_meas_ci90_high'])-x]])
            ye=np.array([[y-float(r['s_pred_ci90_low'])],[float(r['s_pred_ci90_high'])-y]])
            ax.errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[site],markersize=2.9,markeredgewidth=.15,
                        color=color,ecolor=color,elinewidth=.35,capsize=0,alpha=.68,zorder=2)
        primary='E22_kmax' if model.startswith('qwen') else 'E11_g128_kmax'
        primary_errors[model]=float(np.median([abs(float(r['relative_error'])) for r in rr if r['row']==primary and r['site']=='down']))
        ax.annotate(f"Down median |e|: {100*primary_errors[model]:.2f}%",(0,1),xytext=(0,4),xycoords='axes fraction',textcoords='offset points',ha='left',va='bottom',fontsize=7.5,color=CMAPS['down'](1.0),annotation_clip=False)
        ax.set_title(title,loc='left',pad=22,fontsize=9)
        ax.annotate(letter,(0,1),xytext=(-24,22),xycoords='axes fraction',textcoords='offset points',fontsize=10.5,ha='left',va='bottom',annotation_clip=False)
        ax.set_xlabel('Measured step',fontsize=8.5)
        if ax is axes[0]:ax.set_ylabel('Predicted step',fontsize=8.5)
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        # Ordinary numerals keep every glyph at the 7.5 pt text floor.
        formatter=FuncFormatter(lambda x,pos:f'{x:g}')
        ax.xaxis.set_major_formatter(formatter);ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='minor',length=1.7,width=.45)
        ax.grid(which='major',lw=.35,color='#E3E7EB')
    caxes=[]
    for s,left,label in [('qkv',.15,'Qkv'),('down',.59,'Down')]:
        cax=fig.add_axes([left,.125,.29,.032]);caxes.append(cax)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=CMAPS[s]),cax=cax,orientation='horizontal')
        ticks=sorted(set([0,8,32,128,max_k]));ticks=[k for k in ticks if k<=max_k]
        cb.set_ticks(np.log2(np.array(ticks)+1),labels=[str(k) for k in ticks]);cb.outline.set_linewidth(.5)
        cb.ax.tick_params(length=2,width=.5,labelsize=7.5,pad=2)
        fig.text(left-.025,.141,label,ha='right',va='center',fontsize=8.5,color=CMAPS[s](1.0))
    fig.text(.52,.028,'Selected rank k  ·  darker = higher rank  ·  whiskers: 90% seed CI',ha='center',fontsize=7.5)
    q=HERE/'qa/e33';q.mkdir(parents=True,exist_ok=True)
    (q/'source_data.json').write_text(json.dumps({'source':str(src.relative_to(ROOT)),'aggregate_counts':counts,'input_rows':len(rows),'plotted_rows':sum(counts.values()),'excluded_rows':0,'historical_records_retained_separately':len(list(csv.DictReader((ROOT/'results/e33_historical_predictions.csv').open()))),'historical_scatter_omission_reason':'different diagnostic sampling and no three-seed CI; matched saved-rotation configurations are measured again on all 64 chunks','axis_bounds':bounds,'uncertainty':'90% Student-t over 3 rotation seeds, df=2','rank_color_transform':'log2(k+1)','primary_down_median_absolute_error':primary_errors,'fitted_coefficient':None},indent=2)+'\n')
    export(fig,HERE/'fig_rangelaw_persite',draft=arg.draft,exclude_axes=caxes,qa_directory=q)
if __name__=='__main__':main()
