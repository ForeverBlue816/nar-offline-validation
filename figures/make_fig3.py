#!/usr/bin/env python3
"""Data-calibrated 2D rotation illustration plus measured spectral capacity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from figure_style import PALETTE, clean_2d_axis, configure_style, resolved_serif_family, save_panel

def rotation(angle):
    theta=np.deg2rad(angle)
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def prepare(here):
    meta=json.loads((here/'fig1_metadata.json').read_text())
    z=np.load(here/'fig1_source_arrays.npz'); eigenvalues=z['geometry_covariance_eigenvalues']
    scores=z['geometry_scores']; radii=2*np.sqrt(eigenvalues)
    # Orientation of the leading centered PC relative to the selected raw
    # group's all-ones direction fixes the before angle in the canonical plane.
    cosine=meta['geometry_covariance']['raw_pc1_cosine_with_group_dc']
    before_angle=45+float(np.rad2deg(np.arccos(np.clip(cosine,0,1))))
    angles=[before_angle,45.0]
    transformed=[scores@rotation(a).T for a in angles]
    pooled=np.vstack(transformed)
    percentile=np.quantile(pooled,[.005,.995],axis=0)
    # Percentiles initialize bounds; expand to include ALL observed scores.
    low=np.minimum(percentile[0],pooled.min(0)); high=np.maximum(percentile[1],pooled.max(0))
    low=np.minimum(low,-radii[0]); high=np.maximum(high,radii[0])
    lo=float(low.min()); hi=float(high.max()); pad=.06*(hi-lo); limits=[lo-pad,hi+pad]
    inside=[int(np.all((x>=limits[0])&(x<=limits[1]),axis=1).sum()) for x in transformed]
    if inside != [len(scores),len(scores)]: raise AssertionError('Cloud clipping')
    ranges=[meta['trace_ranges']['raw'],meta['trace_ranges']['nar_kmax']]
    return {'model':meta['model'],'site':meta['site'],'layer':meta['layer'],
        'token_window':meta['token_window'],'covariance':meta['geometry_covariance'],
        'ellipse_semiaxes_two_sd':radii.tolist(),'angles_degrees':angles,
        'rotation_degrees':angles[1]-angles[0],'axis_limits_shared':limits,
        'initial_percentile_limits_0_5_to_99_5':percentile.tolist(),
        'cloud_counts':{'before':len(scores),'after':len(scores),'inside_frame':inside,'excluded':0},
        'ranges':{'before_raw':ranges[0],'after_prismquant':ranges[1],
                  'raw_group':meta['raw_trace_group'],'prismquant_group':meta['prismquant_receiving_group'],
                  'sequence':meta['hero']['sequence_index'],'token':meta['hero']['token_position'],
                  'statistic':'max-minus-min of the 128 actual channel values in Figure 1e/g',
                  'bracket_scale_shared':[0,12.0]},
        'interpretation':'Calibrated 2D rigid-rotation illustration: true centered-covariance eigenvalues from the same layer/window, leading ellipse axis aligned exactly with (1,1) after rotation. This is not an exact projection of the full k=max transform; its mapped centered PC1 DC cosine is recorded separately. Brackets are independently scaled measured group ranges, not ellipse projection widths.',
        'covariance_vs_energy':'Panel a uses centered sample covariance on the 512-token Figure 1 window. Panel b uses the existing uncentered second moment over 262016 non-BOS calibration rows; these have different estimands and populations.',
        'panels':['a_before','a_after','b'],'removed_panel':'c (range-law data retained; no c exports)',
        'palette':PALETTE,'font_family_resolved':resolved_serif_family()}

def draw_geometry(ax, meta, after):
    i=int(after); angle=meta['angles_degrees'][i]; a,b=meta['ellipse_semiaxes_two_sd']
    color=PALETTE['prismquant'] if after else PALETTE['identity']
    lo,hi=meta['axis_limits_shared']; ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_aspect('equal',adjustable='box'); clean_2d_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True,color=PALETTE['grid'],lw=.45,zorder=0)
    ax.add_patch(Ellipse((0,0),2*a,2*b,angle=angle,facecolor=PALETTE['identity'],alpha=.25,edgecolor='none',zorder=2))
    ax.add_patch(Ellipse((0,0),2*a,2*b,angle=angle,fill=False,edgecolor=color,lw=1.1,zorder=3))
    major=rotation(angle)@np.array([a,0])
    ax.plot([-major[0],major[0]],[-major[1],major[1]],color=color,lw=.7,ls=(0,(3,2)),zorder=3)
    end=min(hi*.70,a*1.65)/np.sqrt(2)
    ax.annotate('',xy=(end,end),xytext=(0,0),arrowprops={'arrowstyle':'-|>','lw':1.05,'color':PALETTE['text'],'mutation_scale':6},zorder=4)
    ax.text(.05,1.035,'free direction (1, 1)',transform=ax.transAxes,fontsize=7,va='bottom')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')

def draw_range(ax, meta, after):
    val=meta['ranges']['after_prismquant' if after else 'before_raw']
    color=PALETTE['prismquant'] if after else PALETTE['identity']
    ax.set_xlim(0,12); ax.set_ylim(0,1)
    ax.hlines(.65,0,val,color=color,lw=1.1); ax.vlines([0,val],.53,.77,color=color,lw=1)
    ax.text(.5,1.23,f"{'PrismQuant' if after else 'Raw'} range {val:.3f}",transform=ax.transAxes,ha='center',fontsize=7)
    ax.set_xticks([0,6,12]); ax.set_yticks([]); ax.set_xlabel('group range',fontsize=7,labelpad=2)
    for side in ['left','top','right']: ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(PALETTE['pane_edge']); ax.tick_params(labelsize=6)

def draw_energy(ax, data):
    clean_2d_axis(ax)
    specs=[(1,'-',1.2),(13,(0,(3,2)),1.0),(27,(0,(1,2)),1.6)]
    for layer,ls,lw in specs:
        part=data[data.layer.eq(layer)].sort_values('rank')
        if len(part)!=256 or not np.array_equal(part['rank'],np.arange(1,257)):
            raise AssertionError('Incomplete spectrum')
        y=part.cumulative_fraction_total_energy.to_numpy()
        if np.any(np.diff(y)<-1e-10) or np.any((y<0)|(y>1)): raise AssertionError('Invalid energy CDF')
        ax.plot(part['rank'],y,color=PALETTE['identity'],ls=ls,lw=lw,label=f'layer {layer}')
    ax.set_xlim(1,256); ax.set_ylim(0,1); ax.set_xticks([1,64,128,192,256])
    ax.set_xlabel('directions retained, k'); ax.set_ylabel('cumulative energy')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='lower right',fontsize=6.5,frameon=False,handlelength=1.8)

def geometry_figure(meta):
    fig=plt.figure(figsize=(4.15,2.95))
    geo=[fig.add_axes([.13,.37,.34,.48]),fig.add_axes([.62,.37,.34,.48])]
    bars=[fig.add_axes([.13,.115,.34,.065]),fig.add_axes([.62,.115,.34,.065])]
    for i,(ax,bar) in enumerate(zip(geo,bars)):
        draw_geometry(ax,meta,bool(i)); draw_range(bar,meta,bool(i))
    return fig,geo,bars

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figures-dir',type=Path,default=Path(__file__).resolve().parent)
    here=parser.parse_args().figures_dir; configure_style(); meta=prepare(here)
    data=pd.read_csv(here/'fig3_eigenspace_r256.csv')
    fig,geo,bars=geometry_figure(meta)
    save_panel(fig,here/'fig3a',axes=geo+bars,panel_ids=["left", "right", "range_left", "range_right"],row_groups=[["left","right"],["range_left","range_right"]],column_groups=[["left","range_left"],["right","range_right"]])
    fig,ax=plt.subplots(figsize=(2.25,2.2)); fig.subplots_adjust(left=.25,right=.95,bottom=.23,top=.92)
    draw_energy(ax,data);save_panel(fig,here/'fig3b')
    fig=plt.figure(figsize=(6.4,2.95))
    geo=[fig.add_axes([.065,.37,.225,.488136]),fig.add_axes([.365,.37,.225,.488136])]
    bars=[fig.add_axes([.065,.115,.225,.065]),fig.add_axes([.365,.115,.225,.065])]
    energy=fig.add_axes([.755,.30,.225,.55])
    for i,(ax,bar) in enumerate(zip(geo,bars)):
        draw_geometry(ax,meta,bool(i));draw_range(bar,meta,bool(i))
    draw_energy(energy,data)
    for x,label in [(.065,'(a)  2D alignment illustration'),(.755,'(b)  Energy captured')]:
        fig.text(x,.97,label,fontsize=7,va='top')
    save_panel(fig,here/'fig3',axes=geo+bars,panel_ids=["left", "right", "range_left", "range_right"],row_groups=[["left","right"],["range_left","range_right"]],column_groups=[["left","range_left"],["right","range_right"]])
    (here/'fig3_preview.png').write_bytes((here/'fig3.png').read_bytes())
    (here/'fig3_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
