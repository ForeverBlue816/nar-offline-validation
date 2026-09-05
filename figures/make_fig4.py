#!/usr/bin/env python3
"""Measured metadata-budget and activation-recovery/cost panels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
from matplotlib.path import Path as MplPath
from figure_style import PALETTE, configure_style, save_panel

MODELS = ['llama32_3b', 'llama31_8b', 'qwen3_8b_base']
WIDTH, HEIGHT = 2.7, 4.2
BLUE, LIGHT, STEEL, RED = '#1D3557', '#A8DADC', '#457B9D', '#E63946'
CATEGORIES = ['8','16','32','64','max']
CONFIGS = [('hadamard',64,1),('hadamard',128,1),('hadamard',256,1),
           ('hadamard',256,2),('hadamard',256,3),
           ('nar',64,1),('nar',128,1),('nar',256,1),
           ('nar',256,2),('nar',256,3),('nar',128,2)]

def read_source(root, relative):
    frame = pd.read_csv(root / relative)
    frame['source_csv_line'] = np.arange(len(frame)) + 2
    frame['source_file'] = str(relative)
    return frame

def one(frame, column, value):
    part = frame[frame[column].eq(value)]
    if len(part) != 1:
        raise ValueError(f'Expected one source row for {column}={value}, got {len(part)}')
    return part.iloc[0]

def budget_data(root, model):
    relative = Path('results') / model / 'e20_summary.csv'
    frame = read_source(root, relative)
    per = pd.read_csv(root/'results'/model/'e20_per_sequence.csv')
    rows = []
    missing = []
    for method,g,m in CONFIGS:
        name = f'{method}_g{g}_m{m}'
        if name not in set(frame.row):
            missing.append(name); continue
        source = one(frame,'row',name)
        bits = 4 + 16*(m+1)/g
        np.testing.assert_allclose(bits, source.effective_bits, atol=1e-12)
        samples = per[per.row.eq(name)]
        assert samples.groupby('seed').chunk.nunique().eq(64).all()
        assert samples.seed.nunique() == int(source.seeds) == 3
        seed_ppls = samples.groupby('seed').apply(lambda d: np.exp(np.average(d.nll,weights=d.tokens_scored)),include_groups=False)
        np.testing.assert_allclose(seed_ppls.mean(), source.mean_ppl, rtol=1e-12)
        rows.append(dict(kind='point',model=model,method=method,g=g,m=m,effective_bits=bits,
            ppl=float(source.mean_ppl),seeds=int(source.seeds),seed_ids=json.dumps(sorted(samples.seed.unique().tolist())),
            chunks=64,source_file=str(relative),source_row=name,source_csv_line=int(source.source_csv_line),
            formula='4 + 16*(m+1)/g'))
    base = one(frame,'row','bf16')
    rows.append(dict(kind='bf16_reference',model=model,method='bf16',ppl=float(base.mean_ppl),seeds=int(base.seeds),
        chunks=64,source_file=str(relative),source_row='bf16',source_csv_line=int(base.source_csv_line)))
    return pd.DataFrame(rows), missing

def knobs_data(root):
    rows=[]; missing={}; protocol={}
    for model in MODELS:
        qwen=model=='qwen3_8b_base'
        relative=Path('results')/model/('e18v2_summary.csv' if qwen else 'e11_summary.csv')
        frame=read_source(root,relative); key='method'; ppl='ppl' if qwen else 'mean_ppl'
        baseline=one(frame,key,'bf16'); had=one(frame,key,'hadamard' if qwen else 'hadamard_g128_asym')
        if qwen:
            arch=json.loads((root/'results'/model/'e18v2_fold_audit.json').read_text())['architecture_audit']
            kmax=arch['slot_counts']; ksource=f'results/{model}/e18v2_fold_audit.json:architecture_audit.slot_counts'
            audit=None
        else:
            audit=read_source(root,Path('results')/model/'e11_factor_audit.csv')
            kmax={site:int(part.dc_slots.iloc[0]) for site,part in audit[audit.b.eq(128)].groupby('site')}
            assert all(part.dc_slots.nunique()==1 for _,part in audit[audit.b.eq(128)].groupby('site'))
            ksource=f'results/{model}/e11_factor_audit.csv:b=128,dc_slots'
        if not qwen:
            supplemental = Path('results')/model/'e11_k64_summary.csv'
            if (root/supplemental).exists():
                frame = pd.concat([frame, read_source(root,supplemental)],ignore_index=True)
            per = pd.read_csv(root/'results'/model/'e11_per_sequence.csv')
            seed_ids = json.dumps(sorted(per.loc[per.seed.ge(0),"seed"].unique().tolist()))
        missing[model]=[]
        for category in CATEGORIES:
            method=('nar_k'+category) if qwen else ('nar_b128_k'+category)
            part=frame[frame[key].eq(method)]
            if part.empty:
                missing[model].append(category); continue
            source=one(frame,key,method)
            maximum=max(kmax.values())
            actual={site:(cap if category=='max' else min(int(category),cap)) for site,cap in kmax.items()}
            recovery=(float(had[ppl])-float(source[ppl]))/(float(had[ppl])-float(baseline[ppl]))
            assert 0 <= recovery <= 1.05
            rows.append(dict(kind='recovery',model=model,k=maximum if category=='max' else int(category),
                k_category=category,site_kmax=json.dumps(kmax,sort_keys=True),site_k_actual=json.dumps(actual,sort_keys=True),
                site_kmax_source=ksource,ppl_bf16=float(baseline[ppl]),ppl_hadamard=float(had[ppl]),ppl_k=float(source[ppl]),
                recovery=recovery,seeds=1 if qwen else int(source.seeds),
                seed_ids=str(int(source.seed)) if qwen else seed_ids, chunks=int(source.chunks) if qwen else 64,
                source_file=str(source.source_file),source_row=method,source_csv_line=int(source.source_csv_line),
                bf16_source_file=str(baseline.source_file),hadamard_source_file=str(had.source_file),
                bf16_source_csv_line=int(baseline.source_csv_line),hadamard_source_csv_line=int(had.source_csv_line),
                protocol='activation-only qkv + down, group 128, bf16 weights and KV',
                formula='(mean_PPL_Hadamard - mean_PPL_k)/(mean_PPL_Hadamard - PPL_bf16)'))
        protocol[model]={'site_kmax':kmax,'seeds':1 if qwen else 3,'chunks':146 if qwen else 64}
    for model in MODELS[:2]:
        relative=Path('results')/model/'e17v3_layer_overhead.csv'
        timing=read_source(root,relative)
        for k in [8,32]:
            source=one(timing,'k',k)
            share=100*float(source.nar_share_of_layer)
            np.testing.assert_allclose(share,100*source.nar_fused_ms/(source.decoder_layer_ms+source.nar_fused_ms),atol=1e-10)
            rows.append(dict(kind='kernel_share',model=model,k=k,k_category=str(k),method='PrismQuant',share_percent=share,
                decoder_layer_ms=float(source.decoder_layer_ms),kernel_ms=float(source.nar_fused_ms),
                source_file=str(relative),source_row=f'k={k}',source_csv_line=int(source.source_csv_line),
                formula='100*nar_fused_ms/(decoder_layer_ms+nar_fused_ms)'))
        # Use the deployed-k timing record as the one horizontal reference.
        source=one(timing,'k',8)
        rows.append(dict(kind='hadamard_kernel_share',model=model,k=8,k_category='all',method='Hadamard',
            share_percent=100*float(source.hadamard_share_of_layer),decoder_layer_ms=float(source.decoder_layer_ms),
            kernel_ms=float(source.hadamard_fused_ms),source_file=str(relative),source_row='k=8',
            source_csv_line=int(source.source_csv_line),formula='100*hadamard_fused_ms/(decoder_layer_ms+hadamard_fused_ms)',
            selection='deployed-k=8 record, constant horizontal reference; no averaging across timing rows'))
    return pd.DataFrame(rows),missing,protocol

def point_labels(fig, ax, data):
    """Place real point labels in display space without colliding with other text."""
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    occupied=[t.get_window_extent(renderer).expanded(1.04,1.08) for t in fig.findobj(plt.Text)
              if t.get_visible() and t.get_text().strip()]
    paths=[line.get_path().transformed(line.get_transform()) for line in ax.lines]
    points=ax.transData.transform(data[['effective_bits','ppl']].to_numpy())
    point_boxes=[Bbox.from_bounds(x-3,y-3,6,6) for x,y in points]
    plot=ax.get_window_extent(renderer)
    allowed=Bbox.from_extents(plot.x0-6,plot.y0+2,plot.x1+2,plot.y1+15)
    hints={('hadamard',256,1):(-1,-9),('hadamard',256,2):(-2,9),('hadamard',256,3):(15,10),
           ('hadamard',128,1):(-10,-11),('hadamard',64,1):(-10,9),
           ('nar',256,1):(0,-10),('nar',256,2):(-10,11),('nar',256,3):(22,1),
           ('nar',128,1):(-5,-13),('nar',128,2):(0,-10),('nar',64,1):(-14,-9)}
    ordered=sorted(data.itertuples(),key=lambda r: 0 if (r.method,int(r.g),int(r.m))==('nar',256,3) else 1)
    for row in ordered:
        label=f'({int(row.g)}, {int(row.m)})'
        color=BLUE
        preferred=hints[(row.method,int(row.g),int(row.m))]
        candidates=[preferred]+[(dx,dy) for dy in [8,-9,16,-17,24,-25,32,-33] for dx in [0,12,-12,24,-24,36,-36]]
        if row.method=='nar' and int(row.g)==256 and int(row.m)==3:
            candidates=[(22,1),(22,0),(24,2),(24,-2)]+[offset for offset in candidates if offset[0]>=12 and offset[1]<=0]
        best=None
        for dx,dy in candidates:
            t=ax.annotate(label,(row.effective_bits,row.ppl),xytext=(dx,dy),textcoords='offset points',
                ha='center',va='center',fontsize=6,color=color,annotation_clip=False)
            box=t.get_window_extent(renderer).expanded(1.03,1.08)
            penalty=sum(box.overlaps(b) for b in occupied)*1000+sum(box.overlaps(b) for b in point_boxes)*300
            penalty+=500*sum(path.intersects_bbox(box,filled=False) for path in paths)
            if abs(dx)+abs(dy)>25:
                start=ax.transData.transform((row.effective_bits,row.ppl))
                end=start+np.array([dx,dy])*fig.dpi/72
                leader=MplPath(np.array([start,end]))
                penalty+=1000*sum(leader.intersects_bbox(b,filled=False) for b in occupied)
            penalty+=10000*int(box.x0<allowed.x0 or box.x1>allowed.x1 or box.y0<allowed.y0 or box.y1>allowed.y1)
            penalty+=0.05*(abs(dx)+abs(dy))
            if best is None or penalty<best[0]: best=(penalty,dx,dy,box)
            t.remove()
            if penalty<2: break
        _,dx,dy,box=best
        annotation=ax.annotate(label,(row.effective_bits,row.ppl),xytext=(dx,dy),textcoords='offset points',
            ha='center',va='center',fontsize=6,color=color,annotation_clip=False,
            arrowprops={'arrowstyle':'-','color':color,'lw':.35,'shrinkA':2,'shrinkB':3} if abs(dx)+abs(dy)>25 else None)
        occupied.append(box)
        if annotation.arrow_patch is not None:
            annotation.update_positions(renderer)
            paths.append(annotation.arrow_patch.get_path().transformed(annotation.arrow_patch.get_transform()))

def render_budget(data, outbase):
    configure_style(); fig=plt.figure(figsize=(WIDTH,HEIGHT)); ax=fig.add_axes([.20,.27,.75,.49]);ax.set_facecolor('none')
    pts=data[data.kind.eq('point')]; bf=float(data[data.kind.eq('bf16_reference')].ppl.iloc[0])
    for method,color,lw,marker in [('hadamard',LIGHT,1.2,'o'),('nar',BLUE,1.6,'s')]:
        part=pts[pts.method.eq(method)&pts.m.eq(1)].sort_values('effective_bits')
        ax.plot(part.effective_bits,part.ppl,color=color,lw=lw,marker=marker,ms=3.6,mec=color,mew=.6,zorder=4)
    extra_had=pts[pts.method.eq('hadamard')&pts.m.gt(1)]
    ax.plot(extra_had.effective_bits,extra_had.ppl,ls='none',marker='o',ms=3.8,mfc='white',mec=LIGHT,mew=.9,zorder=5)
    extra=pts[pts.method.eq('nar')&pts.m.gt(1)]
    for row in extra.itertuples():
        origin=pts[pts.method.eq('nar')&pts.g.eq(row.g)&pts.m.eq(1)].iloc[0]
        ax.plot([origin.effective_bits,row.effective_bits],[origin.ppl,row.ppl],color=BLUE,lw=.65,ls=(0,(1,2)),zorder=2)
    ax.plot(extra.effective_bits,extra.ppl,ls='none',marker='^',ms=4,mfc=BLUE,mec=BLUE,mew=.4,zorder=5)
    ax.axhline(bf,color=BLUE,lw=.7,ls=(0,(3,2)))
    ax.annotate('bf16',xy=(1,bf),xycoords=('axes fraction','data'),xytext=(-1,3),textcoords='offset points',ha='right',fontsize=6.5)
    ymin=min(bf,float(pts.ppl.min())); ymax=max(bf,float(pts.ppl.max())); pad=.06*(ymax-ymin)
    ax.set_ylim(ymin-pad,ymax+pad); ax.set_xlim(4.085,4.545)
    ax.set_xticks([4.125,4.1875,4.25,4.375,4.5]); ax.set_xticklabels(['4.125','4.1875','4.25','4.375','4.5'],rotation=35,ha='right')
    ax.set_xlabel('effective bits per value',fontsize=7); ax.set_ylabel('WikiText-2 PPL (64 chunks)',fontsize=7)
    ax.tick_params(labelsize=6); ax.spines[['top','right']].set_visible(False)
    ax.axvline(4.25,color=STEEL,lw=.45,alpha=.65,zorder=0)
    ax.annotate('4.25 b',xy=(4.25,1),xycoords=('data','axes fraction'),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.5,color=STEEL)
    def pick(method,g,m): return pts[pts.method.eq(method)&pts.g.eq(g)&pts.m.eq(m)].iloc[0]
    had128=pick('hadamard',128,1); prism128=pick('nar',128,1); had256=pick('hadamard',256,1)
    gap_x=4.271
    ax.plot([gap_x-.005,gap_x,gap_x,gap_x-.005],[prism128.ppl,prism128.ppl,had128.ppl,had128.ppl],color=STEEL,lw=.6,zorder=3)
    ax.text(gap_x+.013,(prism128.ppl+had128.ppl)/2,'null-space\nterm',fontsize=6.5,color=STEEL,ha='left',va='center',linespacing=1.2)
    dy=.025*(ymax-ymin)
    ax.plot([had256.effective_bits,had128.effective_bits],[had256.ppl+dy,had128.ppl+dy],color=STEEL,lw=.6)
    for row in [had256,had128]: ax.plot([row.effective_bits,row.effective_bits],[row.ppl+dy*.55,row.ppl+dy*1.45],color=STEEL,lw=.6)
    ax.annotate('scale-resolution term',xy=((had256.effective_bits+had128.effective_bits)/2,(had256.ppl+had128.ppl)/2+dy),
        xytext=(.16,.825),textcoords='figure fraction',fontsize=6.5,color=STEEL,ha='left',va='center',
        arrowprops={'arrowstyle':'-','lw':.45,'color':STEEL,'shrinkA':2,'shrinkB':3})
    handles=[Line2D([],[],color=LIGHT,lw=1.2,marker='o',ms=3.5,label='Hadamard'),
        Line2D([],[],color=LIGHT,lw=0,marker='o',mfc='white',mec=LIGHT,ms=3.5,label='Hadamard + extra directions'),
        Line2D([],[],color=BLUE,lw=1.6,marker='s',ms=3.5,label='PrismQuant'),
        Line2D([],[],color=BLUE,lw=.65,ls=':',marker='^',ms=3.5,label='PrismQuant + extra directions')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.98,.985),fontsize=6.5,frameon=False,handlelength=1.6,labelspacing=.6)
    point_labels(fig,ax,pts)
    save_panel(fig,outbase,dpi=300,axes=[ax])
    plt.close(fig)
    return {'size_inches':[WIDTH,HEIGHT],'x_limits':[4.085,4.545],'y_limits':[ymin-pad,ymax+pad],
            'null_space_gap_at_4_25':float(had128.ppl-prism128.ppl),'scale_resolution_gap':float(had256.ppl-had128.ppl),
            'annotation_sources':{'null_space_term':[int(had128.source_csv_line),int(prism128.source_csv_line)],
                                  'scale_resolution_term':[int(had256.source_csv_line),int(had128.source_csv_line)]}}

def render_knobs(data,outbase):
    configure_style(); fig=plt.figure(figsize=(WIDTH,HEIGHT)); ax=fig.add_axes([.22,.27,.54,.49]); cost=ax.twinx()
    ax.axvspan(-.23,.23,facecolor=PALETTE['zero'],zorder=0)
    ax.text(0,1.065,'deployed',ha='center',va='bottom',fontsize=6.5,clip_on=False)
    specs=[('llama32_3b',LIGHT,'Llama-3.2-3B',1.2),('llama31_8b',STEEL,'Llama-3.1-8B',1.2),('qwen3_8b_base',BLUE,'Qwen3-8B-Base',1.6)]
    handles=[]
    for model,color,label,lw in specs:
        part=data[data.kind.eq('recovery')&data.model.eq(model)].copy()
        part['x']=part.k_category.map({k:i for i,k in enumerate(CATEGORIES)}); part=part.sort_values('x')
        line,=ax.plot(part.x,part.recovery,color=color,marker='o',ms=3.5,lw=lw,label=label,zorder=4); handles.append(line)
    for model,label,style,offset in [('llama32_3b','3B','-',6),('llama31_8b','8B',(0,(1.5,1.5)),-6)]:
        part=data[data.kind.eq('kernel_share')&data.model.eq(model)].sort_values('k')
        x=[CATEGORIES.index(str(int(k))) for k in part.k]
        cost.plot(x,part.share_percent,color=RED,lw=.7,ls=style,marker='D',ms=3.5,mfc='white',mec=RED,mew=.8,zorder=3)
        cost.annotate(label,xy=(x[-1],float(part.share_percent.iloc[-1])),xytext=(-12,8) if label=='3B' else (5,-3),textcoords='offset points',fontsize=6,color=RED,va='center')
        base=float(data[data.kind.eq('hadamard_kernel_share')&data.model.eq(model)].share_percent.iloc[0])
        cost.axhline(base,color=RED,lw=.6,ls=(0,(3,2)),alpha=.85)
        cost.annotate(f'Hadamard, {label}',xy=(4.1,base),xytext=(-1,offset),textcoords='offset points',ha='right',va='center',fontsize=6,color=RED,
                      arrowprops={'arrowstyle':'-','color':RED,'lw':.35,'shrinkA':2,'shrinkB':1})
    ax.set_xlim(-.35,4.35); ax.set_ylim(0,1.05); cost.set_ylim(0,10)
    ax.set_xticks(range(5),CATEGORIES); ax.set_yticks([0,.25,.5,.75,1]); ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel('directions retained, k',fontsize=7); ax.set_ylabel('recovered fraction of Hadamard gap',fontsize=7)
    cost.set_yticks([0,2.5,5,7.5,10],['0%','2.5%','5%','7.5%','10%']); cost.tick_params(axis='y',colors=RED,labelsize=6,pad=2)
    cost.set_ylabel('share of decoder-layer time',fontsize=7,color=BLUE,labelpad=4)
    ax.tick_params(labelsize=6); ax.spines['top'].set_visible(False); cost.spines['top'].set_visible(False)
    cost.spines['right'].set_color(RED)
    handles.append(Line2D([],[],color=RED,lw=.7,marker='D',ms=3.5,mfc='white',label='kernel share (right axis)'))
    fig.legend(handles=handles,loc='lower right',bbox_to_anchor=(.985,.005),fontsize=6.5,frameon=False,handlelength=1.6,labelspacing=.6)
    save_panel(fig,outbase,dpi=300,axes=[ax])
    plt.close(fig)
    return {'size_inches':[WIDTH,HEIGHT],'x_categories':CATEGORIES,'recovery_limits':[0,1.05],'kernel_share_percent_limits':[0,10],
            'hadamard_timing_reference':'k=8 row for each model','share_denominator':'decoder_layer_ms + online_transform_ms'}

def compose(here):
    import pymupdf
    import xml.etree.ElementTree as ET
    from audit_panel_alignment import audit_layout_manifest, write_json_report, exit_code
    panels=[]
    for i,stem in enumerate(['fig4a','fig4b']):
        layout=json.loads((here/'qa'/f'{stem}.alignment.json').read_text())['layout']
        box=layout['panels'][0]['bbox_pt']
        panels.append({'id':stem,'bbox_pt':[box[0]+i*WIDTH*72,box[1],box[2]+i*WIDTH*72,box[3]]})
    report=audit_layout_manifest({'schema_version':1,'backend':'composed matplotlib panels',
        'figure':{'width_pt':2*WIDTH*72,'height_pt':HEIGHT*72},'panels':panels,'row_groups':[['fig4a','fig4b']],
        'exemptions':[{'panels':['fig4b'],'checks':['panel-width'],'reason':'Room for the requested second y axis and its labels.'}]})
    write_json_report(report,here/'qa'/'fig4.alignment.json')
    if exit_code(report,strict=True): raise RuntimeError('Figure 4 alignment failed')
    fig=plt.figure(figsize=(WIDTH*2,HEIGHT))
    doc=pymupdf.open(); page=doc.new_page(width=WIDTH*2*72,height=HEIGHT*72)
    ns='http://www.w3.org/2000/svg'; ET.register_namespace('',ns)
    root=ET.Element(f'{{{ns}}}svg',width=f'{WIDTH*2*72}pt',height=f'{HEIGHT*72}pt',viewBox=f'0 0 {WIDTH*2*72} {HEIGHT*72}')
    for i,stem in enumerate(['fig4a','fig4b']):
        ax=fig.add_axes([i*.5,0,.5,1]); ax.imshow(plt.imread(here/f'{stem}.png'));ax.axis('off')
        with pymupdf.open(here/f'{stem}.pdf') as src: page.show_pdf_page(pymupdf.Rect(i*WIDTH*72,0,(i+1)*WIDTH*72,HEIGHT*72),src,0)
        node=ET.parse(here/f'{stem}.svg').getroot(); raw=ET.tostring(node,encoding='unicode')
        ids=[e.attrib['id'] for e in node.iter() if 'id' in e.attrib]
        for key in sorted(ids,key=len,reverse=True):
            raw=raw.replace(f'id="{key}"',f'id="{stem}_{key}"').replace(f'#{key})',f'#{stem}_{key})').replace(f'"#{key}"',f'"#{stem}_{key}"')
        node=ET.fromstring(raw);node.set('x',str(i*WIDTH*72));root.append(node)
    fig.savefig(here/'fig4_preview.png',dpi=300);plt.close(fig);doc.save(here/'fig4.pdf',deflate=True);doc.close()
    ET.ElementTree(root).write(here/'fig4.svg',encoding='unicode',xml_declaration=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.repo.resolve();here=root/'figures';metadata={'panels':{},'backend':'plain matplotlib'}
    for model,stem in [('llama32_3b','fig4a'),('llama31_8b','fig4a_8b')]:
        data,missing=budget_data(root,model);data.to_csv(here/f'{stem}.csv',index=False)
        metadata['panels'][stem]={'model':model,'missing_configurations':missing,**render_budget(data,here/stem)}
    data,missing,protocol=knobs_data(root);data.to_csv(here/'fig4b.csv',index=False)
    metadata['panels']['fig4b']={**render_knobs(data,here/'fig4b'),'missing_k_categories':missing,'protocols':protocol,
        'recovery_formula':'ratio computed from mean PPLs; not the legacy stored recovery column'}
    compose(here);(here/'fig4_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
