"""Render only measured display caches. Matplotlib; no model or tensor synthesis."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager,ticker
from .qa.panel_alignment import require_matplotlib_panel_alignment

METHODS=['unrotated','hadamard','nar_k8','nar_kmax']
LABELS={'unrotated':'Unrotated','hadamard':'Hadamard','nar_k8':'PrismQuant (k=8)','nar_kmax':'PrismQuant (k=max)'}
LAYERS=[0,12,23,35]
CMAP=colors.LinearSegmentedColormap.from_list('blue_warm_orange',['#225b8d','#4f8eb6','#96b8c6','#dac9a8','#eaa15d','#c46731'])
METHOD_COLORS=['#727c86','#5687a7','#d6a65e','#b86c45']

def style():
    fontdir=Path.home()/'.local/share/figure-fonts/times-new-roman'
    for p in fontdir.glob('*'):
        if p.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(p))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':6,
        'axes.titlesize':7,'axes.labelsize':5.5,'xtick.labelsize':5,'ytick.labelsize':5,
        'axes.linewidth':.45,'grid.linewidth':.3,'grid.color':'#d7dce0',
        'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','axes.unicode_minus':False})

def number(v,pos=None):
    if v==0:return '0.00'
    return f'{v:.2e}' if abs(v)>=1000 or abs(v)<.01 else f'{v:.2f}'

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(str(path)+'.pdf',dpi=600)
    fig.savefig(str(path)+'.svg',dpi=600)
    fig.savefig(str(path)+'.png',dpi=600)
    plt.close(fig)

def surface(ax,z,limit,view):
    # Transparent axes patches cannot erase neighboring 3-D tick labels.
    ax.patch.set_alpha(0)
    yy,xx=np.indices(z.shape)
    ax.plot_surface(xx,yy,z,cmap=CMAP,norm=colors.Normalize(0,limit),
        rcount=z.shape[0],ccount=z.shape[1],linewidth=0,antialiased=False,shade=False,rasterized=True)
    ax.view_init(elev=25,azim=-60);ax.set_box_aspect((1.28,1,.72),zoom=.88)
    ax.set_xlim(0,z.shape[1]-1);ax.set_ylim(0,z.shape[0]-1);ax.set_zlim(0,limit)
    ax.set_xticks([0,z.shape[1]-1]);ax.set_yticks([0,z.shape[0]-1]);ax.set_zticks([0,limit/2,limit])
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(number))
    ax.set_xlabel('Channel bin' if view=='overview' else 'Channel',labelpad=-10)
    ax.set_ylabel('Token bin' if view=='overview' else 'Token',labelpad=0)
    ax.tick_params(axis='both',pad=0,length=1.5)
    ax.tick_params(axis='z',pad=2,labelsize=5)
    for label in ax.get_zticklabels():label.set_horizontalalignment('left')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_pane_color((.975,.979,.982,1))
        axis.line.set_color('#869099');axis.line.set_linewidth(.4)
        axis._axinfo['grid']['linewidth']=.25
        axis._axinfo['grid']['color']=(.81,.84,.86,1)

def matrix(cache,out,mode,site,quantity,view,methods,layers=LAYERS,name='matrix',scale_methods=None):
    nrow=len(methods);ncol=len(layers)
    # Fixed physical subplot cells preserve readable panel exports and equal comparison geometry.
    width=7.2 if ncol==4 else 2.35
    height=(1.63*nrow+.48) if ncol==4 else 2.12
    fig=plt.figure(figsize=(width,height),dpi=600)
    grid=fig.add_gridspec(nrow,ncol,left=.085 if ncol==4 else .10,right=.93 if ncol==4 else .88,bottom=12/(height*72),top=1-33/(height*72),
                         wspace=.17,hspace=.16)
    scales={}
    for col,layer in enumerate(layers):
        candidates=scale_methods or (METHODS if mode=='paired_local' else METHODS[1:])
        limits=[float(cache[f'{mode}__{m}__{layer}__{site}__{quantity}__{view}'].max()) for m in candidates]
        limit=max(limits)*1.03 or 1.;scales[str(layer)]=limit
        for row,method in enumerate(methods):
            ax=fig.add_subplot(grid[row,col],projection='3d')
            z=cache[f'{mode}__{method}__{layer}__{site}__{quantity}__{view}']
            surface(ax,z,limit,view)
            if row==0:ax.set_title(f'Block {layer+1}',pad=1)
            if col==0:
                fig.text(.018 if ncol==4 else .045,1-(row+.5)/nrow*.895-.05,LABELS[method],
                         rotation=90,rotation_mode='anchor',va='center',ha='center',fontsize=6.5)
    magnitude='Max magnitude within bin' if view=='overview' else 'Magnitude'
    if name=='rotated_only_zoom':magnitude+='; shared rotated scale'
    title='Group-centered residual magnitude' if quantity=='residual' else 'Pre-quantization activation magnitude'
    fig.text(.52,1-3/(height*72),title,ha='center',va='top',fontsize=7)
    fig.text(.52,1-14/(height*72),magnitude,ha='center',va='top',fontsize=5.5)
    path=out/mode/site/quantity/view/name
    save(fig,path)
    path.with_suffix('.scales.json').write_text(json.dumps({'z_limits_by_layer':scales,'shared_across_methods':scale_methods or 'all methods in mode','linear':True},indent=2)+'\n')
    print('RENDERED',path,flush=True)

def distributions(root,out):
    ecdf=np.load(root/'range_ecdf.npz');p=np.linspace(0,1,1001)
    for mode in ('paired_local','end_to_end'):
        methods=METHODS if mode=='paired_local' else METHODS[1:]
        for site in ('down_proj','q_proj'):
            fig,axes=plt.subplots(1,4,figsize=(7.2,2.05),sharey=True)
            fig.subplots_adjust(left=.07,right=.98,bottom=.23,top=.77,wspace=.25)
            for ax,layer in zip(axes,LAYERS):
                for m in methods:
                    ax.plot(ecdf[f'{mode}__{m}__{layer}__{site}'],p,color=METHOD_COLORS[METHODS.index(m)],lw=1,label=LABELS[m])
                ax.set_title(f'Block {layer+1}');ax.set_xlabel('Signed group range');ax.set_ylim(0,1)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                ax.set_xscale('symlog',linthresh=.01)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(number))
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.grid(axis='y',alpha=.5)
                ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2)
            axes[0].set_ylabel('Cumulative probability')
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=len(methods),frameon=False,fontsize=6)
            save(fig,out/mode/site/'group_range_ecdf')

def run(root,selected='all',parts='all'):
    root=Path(root);out=root/'figures';style();cache=np.load(root/'display_cache.npz')
    for mode in ('paired_local','end_to_end'):
        methods=METHODS if mode=='paired_local' else METHODS[1:]
        for site in ('down_proj','q_proj'):
            for quantity in ('raw','residual'):
                for view in ('overview','detail'):
                    if selected!='all' and selected!='matrices' and selected!=f'{mode}/{site}/{quantity}/{view}':continue
                    matrix(cache,out,mode,site,quantity,view,methods)
                    if mode=='paired_local':
                        matrix(cache,out,mode,site,quantity,view,METHODS[1:],name='rotated_only_zoom',scale_methods=METHODS[1:])
                    if selected=='matrices' or parts=='matrix':continue
                    for m in methods:
                        matrix(cache,out,mode,site,quantity,view,[m],name=f'row_{m}')
                        for l in LAYERS:matrix(cache,out,mode,site,quantity,view,[m],[l],name=f'panel_{m}_block{l+1}')
    if selected in ('all','matrices'):distributions(root,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--select',default='all');a=p.parse_args();run(a.root,a.select)
