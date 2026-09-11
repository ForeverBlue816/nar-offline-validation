"""Shared figure style and full-resolution rendering entry point."""
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


def run(root, selected='all', parts='all'):
    """Render complete raw shards; the former pooled display cache is unused."""
    from .full_plot import run as full_run
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    source = root if (root/'activations').exists() else Path(manifest['raw_activation_root']).parent
    if selected == 'distributions':
        from .full_ecdf import run as ecdf_run
        ecdf_run(source, root)
        return
    selectors = [f'{mode}/{site}/{quantity}' for mode in ('paired_local','end_to_end')
                 for site in ('down_proj','q_proj') for quantity in ('raw','residual')]
    if selected not in ('all','matrices'):
        selectors = ['/'.join(selected.split('/')[:3])]
    for selector in selectors:
        full_run(source, root, selector, parts='matrix' if selected == 'matrices' else parts)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--select',default='all')
    a=p.parse_args(); run(a.root,a.select)
