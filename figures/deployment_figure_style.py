"""Shared final-size styling and export for the two frozen-data figures."""
from pathlib import Path
import json
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from figure_typography import configure_times_bold
from qa_tools.audit_panel_alignment import require_matplotlib_panel_alignment

HERE=Path(__file__).resolve().parent
BLUE='#1D3557'; STEEL='#457B9D'; TEAL='#A8DADC'; TEAL_EDGE='#56858B'
GRAY='#D4DBE2'; GRAY_EDGE='#64748B'; GREEN='#73CC80'; INK='#385A75'
COLORS={'fp16':GRAY,'hadamard':TEAL,'nar':BLUE}
EDGES={'fp16':GRAY_EDGE,'hadamard':TEAL_EDGE,'nar':BLUE}
NAMES={'fp16':'FP16','hadamard':'Hadamard','nar':'PrismQuant'}
MODEL_NAMES={'3b':'Llama-3.2-3B','8b':'Llama-3.1-8B'}
MODEL_STYLE={'llama32_3b':(TEAL_EDGE,'o','-','Llama-3.2-3B'),
             'llama31_8b':(STEEL,'s','--','Llama-3.1-8B'),
             'qwen3_8b_base':(BLUE,'D','-','Qwen3-8B-Base')}


def style():
    typography=configure_times_bold(4)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
        'axes.linewidth':1.05,'xtick.major.width':.95,'ytick.major.width':.95,
        'xtick.major.size':3.3,'ytick.major.size':3.3,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelpad':4,'pdf.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'prismquant-e28-v2-6747960','figure.facecolor':'white','savefig.facecolor':'white',
        'lines.linewidth':2.1,'lines.markersize':5.2,'axes.axisbelow':True})
    return typography


def axis(ax, letter, title, subtitle=None):
    ax.annotate(letter,xy=(0,1),xytext=(-29,12),xycoords='axes fraction',textcoords='offset points',
                fontsize=10.5,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
    ax.set_title(title,loc='left',pad=12,fontsize=9)
    if subtitle:
        ax.annotate(subtitle,(0,1),xycoords='axes fraction',xytext=(0,2),textcoords='offset points',
                    ha='left',va='bottom',fontsize=7.5,color=INK,annotation_clip=False)
    ax.grid(axis='y',color='#DCE4EA',lw=.45)


def method_legend(fig, y=.035):
    handles=[Patch(facecolor=COLORS[m],edgecolor=EDGES[m],linewidth=1,label=NAMES[m]) for m in ['fp16','hadamard','nar']]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,y),ncol=3,
               handlelength=1.4,columnspacing=1.6,handletextpad=.5,frameon=False)


def span(ax, center, values, pos, horizontal=False, color=GRAY_EDGE, dots=False):
    lo,hi=min(values),max(values)
    if horizontal:
        ax.plot([lo,hi],[pos,pos],lw=1.1,color=color,zorder=5)
        for x in [lo,hi]:ax.plot([x,x],[pos-.045,pos+.045],lw=1.1,color=color,zorder=5)
        if dots:ax.scatter(values,[pos-.11,pos-.11,pos-.11],s=7,color=STEEL,zorder=6)
    else:
        ax.plot([pos,pos],[lo,hi],lw=1.1,color=color,zorder=5)
        for y in [lo,hi]:ax.plot([pos-.025,pos+.025],[y,y],lw=1.1,color=color,zorder=5)


def export(fig, outbase, *, draft=False, exclude_axes=(), panel_axes=None):
    outbase=Path(outbase);outbase.parent.mkdir(parents=True,exist_ok=True)
    q=HERE/'qa'/'deployment';q.mkdir(exist_ok=True,parents=True)
    fig.canvas.draw()
    extents=[]
    for ax in fig.axes:
        if ax in exclude_axes:continue
        xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim()
        for line in ax.lines:
            if line.get_transform()!=ax.transData:continue
            x,y=line.get_data(orig=False)
            if not len(x):continue
            assert min(x)>=xmin-1e-9 and max(x)<=xmax+1e-9, 'Clipped line or interval in x'
            assert min(y)>=ymin-1e-9 and max(y)<=ymax+1e-9, 'Clipped line or interval in y'
            extents.append(dict(x_min=float(min(x)),x_max=float(max(x)),y_min=float(min(y)),y_max=float(max(y))))
        for collection in ax.collections:
            if collection.get_offset_transform()!=ax.transData:continue
            points=collection.get_offsets()
            if not len(points):continue
            assert points[:,0].min()>=xmin and points[:,0].max()<=xmax
            assert points[:,1].min()>=ymin and points[:,1].max()<=ymax
    (q/f'{outbase.name}.extents.json').write_text(json.dumps({'all_data_and_intervals_within_axes':True,'data_lines':extents},indent=2)+'\n')
    require_matplotlib_panel_alignment(fig,json_out=q/f'{outbase.name}.alignment.json',
        strict=True,require_panel_labels=len(fig.axes)>1,exclude_axes=exclude_axes)
    fig.savefig(outbase.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(outbase.with_suffix('.svg'),metadata={'Date':None})
    resolution = 150 if draft else 600
    fig.savefig(outbase.with_suffix('.png'),dpi=resolution)
    result=subprocess.run([sys.executable,str(HERE/'qa_tools/audit_figure_collisions.py'),str(outbase.with_suffix('.pdf')),
        '--json-out',str(q/f'{outbase.name}.collision.json')],capture_output=True,text=True)
    print(outbase.name,result.stdout.strip().splitlines()[0] if result.stdout else result.stderr)
    if not draft and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    # Drafts remain inspectable while final exports require collision clearance.
    subprocess.run([sys.executable,str(HERE/'qa_tools/audit_pdf_text.py'),str(outbase.with_suffix('.pdf')),
        '--min-pt','7.5','--json'],stdout=(q/f'{outbase.name}.text.json').open('w'),check=True)
    if panel_axes and not draft:
        for legend in fig.legends:legend.set_visible(False)
        renderer=fig.canvas.get_renderer()
        for letter, ax in panel_axes.items():
            bbox=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(.04)
            other=[a for a in fig.axes if a is not ax]
            prior=[a.get_visible() for a in other]
            for a in other:a.set_visible(False)
            for ext in ['pdf','svg','png']:
                metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
                fig.savefig(HERE/'panels'/f'{outbase.name}_{letter}.{ext}',bbox_inches=bbox,dpi=600,metadata=metadata)
            for a,v in zip(other,prior):a.set_visible(v)
            path=HERE/'panels'/f'{outbase.name}_{letter}.pdf'
            subprocess.run([sys.executable,str(HERE/'qa_tools/audit_figure_collisions.py'),str(path),
                '--json-out',str(q/f'{outbase.name}_{letter}.collision.json')],capture_output=True,check=True)
    plt.close(fig)
