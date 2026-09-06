"""Compact, causal Figure 1 layout; semantic icons are vector schematics only."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from figure_style import PALETTE, SEQUENTIAL_CMAP, configure_style, save_panel

INK='#24354B'; RAW='#64748B'
HAD=PALETTE['hadamard']; PRISM=PALETTE['prismquant']
CYAN=PALETTE['hadamard']; ZERO='#73CC80'; GRID=PALETTE['grid']
BURGUNDY='#601D49'; ROSE='#BD5579'; BLUSH='#EA9D9D'
WIDTH,HEIGHT=6.6,4.25
# Dimensions are inches. Dense panels keep all original sample segments.
PLACES={'a':(.00,2.24,1.88,1.53),'b':(2.20,2.85,1.82,1.35),
        'c':(4.40,2.85,2.16,1.35),'d':(4.40,1.49,2.16,1.35),
        'e':(.10,.08,2.04,1.10),'f':(2.28,.08,2.04,1.10),'g':(4.46,.08,2.04,1.10)}
RANGE_MAP=SEQUENTIAL_CMAP
RAW_SURFACE_COLORS=['#FFF5F5',BLUSH,ROSE,BURGUNDY]
FIG1_PALETTE={'raw':BURGUNDY,'hadamard':HAD,'prismquant':PRISM,
    'range_map_shared':True,'zero_point':ZERO,'panel_a':BURGUNDY,
    'panel_e':BURGUNDY,'panel_g':PRISM,'surface_a_colormap':RAW_SURFACE_COLORS}

def style():
    configure_style()
    plt.rcParams.update({'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':RAW,'xtick.color':RAW,'ytick.color':RAW})

def landscape(values, letter, arrays, metadata, here):
    style();_,_,w,h=PLACES[letter];fig=plt.figure(figsize=(w,h))
    ax=fig.add_axes([0,0,1,1],projection='3d')
    ranges=letter in 'cd';zmax=10 if ranges else (40 if letter=='a' else 4)
    x=np.arange(values.shape[1])+(0 if ranges else metadata['channel_windows']['raw'][0])
    tokens=arrays['token_axis']
    assert np.isfinite(values).all() and values.min()>=0 and values.max()<=zmax
    coords=np.empty((values.shape[1],values.shape[0],3),dtype=np.float32)
    coords[:,:,0]=x[:,None];coords[:,:,1]=tokens[None,:];coords[:,:,2]=values.T
    segments=np.stack([coords[:,:-1],coords[:,1:]],axis=2).reshape(-1,2,3)
    cmap=LinearSegmentedColormap.from_list('magnitude',RAW_SURFACE_COLORS) if letter=='a' else RANGE_MAP
    mark=Line3DCollection(segments,cmap=cmap,norm=Normalize(0,zmax),linewidths=.9 if ranges else .7)
    mark.set_array(segments[:,:,2].max(1));mark.set_rasterized(True);ax.add_collection3d(mark)
    ax.set(xlim=(int(x[0]),int(x[-1])),ylim=(int(tokens[0]),int(tokens[-1])),zlim=(0,zmax))
    ax.set_xticks([0,32,63] if ranges else [1500,3000])
    ax.set_yticks([200,600]);ax.set_zticks([0,zmax/2,zmax])
    ax.set_xlabel('');ax.set_ylabel('')
    fig.text(.43,.07,'group' if ranges else 'channel',fontsize=6,ha='center',color=RAW)
    fig.text(.88,.045 if letter=='b' else .10,'token',fontsize=6,ha='center',color=RAW)
    ax.set_zlabel('');ax.tick_params(labelsize=6,pad=-3,length=1.5)
    ax.tick_params(axis='y',pad=0)
    ax.view_init(elev=18 if ranges else 22,azim=-62 if ranges else -60)
    ax.set_box_aspect((2.6,1.2,.85),zoom=.94 if ranges else .88)
    ax.patch.set_alpha(0)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False;axis.pane.set_edgecolor(GRID);axis.pane.set_linewidth(.3)
        axis._axinfo['grid'].update(color='#E8EDF1',linewidth=.3)
        axis.line.set_color(GRID);axis.line.set_linewidth(.4)
    # Projection is invariant across SVG/PDF/PNG DPI changes at fixed geometry.
    # Cache it to avoid repeating a million-segment 3D projection per export.
    project=mark.do_3d_projection;projection_cache={}
    def project_once():
        key=tuple(ax.get_proj().ravel())
        if key not in projection_cache:projection_cache[key]=project()
        return projection_cache[key]
    mark.do_3d_projection=project_once
    save_panel(fig,here/f'fig1{letter}',dpi=300,transparent=True,axes=[ax])
    return {'array':{'a':'raw_magnitude','b':'hadamard_magnitude','c':'hadamard_range','d':'nar_kmax_range'}[letter],
            'shape':list(values.shape),'segments':len(segments),'z_limits':[0,zmax],
            'color_normalization':[0,zmax],'view':[18,-62] if ranges else [22,-60],
            'size_inches':[w,h],'sample_removal':False,'clipping':False}

def trace(values,letter,metadata,here):
    style();_,_,w,h=PLACES[letter];fig=plt.figure(figsize=(w,h))
    ax=fig.add_axes([.16,.25,.81,.62]);color={'e':BURGUNDY,'f':HAD,'g':PRISM}[letter]
    ax.hlines(0,0,127,color=GRID,lw=.5,zorder=0)
    ax.plot(np.arange(128),values,color=color,lw=.8)
    low,high=map(float,[values.min(),values.max()])
    ax.vlines(133,low,high,color=color,lw=.65);ax.hlines([low,high],130.5,135.5,color=color,lw=.65)
    ax.text(138,(low+high)/2,f'{high-low:.3f}',ha='left',va='center',fontsize=6.5,color=color)
    if letter=='g':
        offset=float(np.float16(values.min()))
        ax.hlines(offset,0,127,color=ZERO,lw=.7,ls=(0,(3,2)))
        ax.annotate('zero-point offset',(4,offset),xytext=(0,-5),textcoords='offset points',
                    ha='left',va='top',fontsize=6,color=ZERO)
    ax.set(xlim=(0,165),ylim=metadata['trace_rendering']['y_limits'])
    ax.set_xticks([0,64,127]);ax.set_yticks([-5,0,5]);ax.tick_params(labelsize=6,pad=2)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('channel in group',fontsize=6,labelpad=2)
    if letter=='e':ax.set_ylabel('signed value',fontsize=6,labelpad=2)
    save_panel(fig,here/f'fig1{letter}',dpi=300,transparent=True,axes=[ax])

def arrow(ax,start,end,color=RAW,lw=.8,style='-|>',**kw):
    p=FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=6,linewidth=lw,
                     color=color,shrinkA=0,shrinkB=0,**kw);ax.add_patch(p)

def matrix_icon(ax,x,y):
    pattern=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
    size=.042
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x+j*size,y+i*size),size*.82,size*.82,
                                  facecolor=HAD if pattern[i,j]>0 else PALETTE['zero'],edgecolor='none'))
    ax.text(x+.082,y+.21,'H',color=HAD,fontsize=7,ha='center')

def alignment_icon(ax,x,y):
    # Unscaled conceptual glyph, not an empirical ellipse or energy measurement.
    ax.add_patch(Ellipse((x,y),.35,.12,angle=32,facecolor=PALETTE['zero'],edgecolor=CYAN,lw=.6))
    arrow(ax,(x-.13,y-.08),(x+.15,y+.09),PRISM,lw=.75)
    ax.text(x-.04,y+.16,'v₁ … vₖ',fontsize=6,color=PRISM,ha='center')
    arrow(ax,(x+.25,y),(x+.51,y),PRISM)
    for i in range(4):ax.plot([x+.61+i*.045]*2,[y-.075,y+.075],color=PRISM,lw=.8)
    ax.text(x+.68,y+.16,r'$\mathbf{1}_g$',fontsize=9,color=PRISM,ha='center')

def quantizer_icon(ax,x,y):
    for i,level in enumerate([.07,.12,.095,.145]):
        ax.plot([x+i*.042]*2,[y,y+level],color=ZERO,lw=.8)
    ax.plot([x-.02,x+.15],[y,y],color=ZERO,lw=.6,ls=(0,(2,1)))
    ax.text(x+.18,y-.01,'z',fontsize=6,color=ZERO,va='center')
    ax.text(x-.10,y+.07,'s',fontsize=6,color=ZERO,ha='center',va='center')
    ax.annotate('',(x-.05,y+.14),(x-.05,y),arrowprops={'arrowstyle':'|-|','color':ZERO,'lw':.5})

def compose(here,metadata):
    import pymupdf
    from audit_panel_alignment import audit_layout_manifest,write_json_report,exit_code
    style();fig=plt.figure(figsize=(WIDTH,HEIGHT));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT));ax.axis('off')
    def text(x,y,t,size=7,color=INK,**kw):ax.text(x,y,t,fontsize=size,color=color,**kw)
    # An explicit fork: both branches start at the same raw activation.
    ax.plot([1.88,1.98],[2.98,2.98],color=BURGUNDY,lw=.8)
    ax.plot([1.98,1.98],[2.98,3.53],color=HAD,lw=.8)
    arrow(ax,(1.98,3.53),(2.22,3.53),HAD)
    ax.plot([1.98,1.98],[2.98,2.12],color=PRISM,lw=.8)
    arrow(ax,(1.98,2.12),(2.28,2.12),PRISM)
    matrix_icon(ax,2.015,3.70)
    # Identical quantizer glyphs encode the same operation on both branches.
    for y,color in [(3.53,HAD),(2.12,PRISM)]:
        arrow(ax,(3.94,y),(4.40,y),color)
        quantizer_icon(ax,4.08,y+.10)
    alignment_icon(ax,2.51,2.12)
    arrow(ax,(3.36,2.12),(3.91,2.12),PRISM)
    text(2.28,2.54,'PrismQuant',size=8,color=PRISM,weight='bold')
    text(2.28,1.93,'align dominant eigendirections\nwith group-constant directions',size=6.5,color=PRISM,linespacing=1.25,va='top')
    # Offset absorption changes the affine baseline, not max-minus-min range.
    arrow(ax,(2.40,1.64),(2.40,1.43),ZERO,lw=.65,style='<->')
    ax.plot([2.50,2.72],[1.57,1.57],color=ZERO,lw=.7,ls=(0,(2,1)))
    text(2.81,1.50,'common offset → zero-point',size=6,color=ZERO,va='center')
    text(.12,3.69,'(a)  Raw activations',size=8,color=BURGUNDY,weight='bold')
    text(2.28,4.07,'(b)  Hadamard: spread energy',size=7.5,color=HAD,weight='bold')
    text(4.52,4.07,'(c)  Hadamard: spread',size=8,color=HAD,weight='bold')
    text(4.52,2.71,'(d)  PrismQuant: align',size=8,color=PRISM,weight='bold')
    ax.plot([.12,6.49],[1.22,1.22],color=GRID,lw=.6)
    text(.13,1.31,'One token · measured group traces',size=7,color=INK)
    for letter,title,color in [('e','Raw: concentrated',BURGUNDY),('f','Hadamard: spread',HAD),('g','PrismQuant: shared offset',PRISM)]:
        x,y,w,h=PLACES[letter];text(x+.33,1.10,f'({letter})  {title}',size=7,color=color,weight='bold')
    fig.savefig(here/'qa/fig1_annotations.pdf',transparent=True);fig.savefig(here/'qa/fig1_annotations.svg',transparent=True);plt.close(fig)
    annotation_svg=here/'qa/fig1_annotations.svg'
    annotation_svg.write_text('\n'.join(line.rstrip() for line in annotation_svg.read_text().splitlines())+'\n')
    # Compose vectors at their native size; only scientific 3D marks are raster.
    doc=pymupdf.open();page=doc.new_page(width=WIDTH*72,height=HEIGHT*72)
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    root=ET.Element(f'{{{ns}}}svg',width=f'{WIDTH*72}pt',height=f'{HEIGHT*72}pt',viewBox=f'0 0 {WIDTH*72} {HEIGHT*72}')
    panels=[]
    for letter,(x,y,w,h) in PLACES.items():
        with pymupdf.open(here/f'fig1{letter}.pdf') as src:
            page.show_pdf_page(pymupdf.Rect(x*72,(HEIGHT-y-h)*72,(x+w)*72,(HEIGHT-y)*72),src,0)
        node=ET.parse(here/f'fig1{letter}.svg').getroot();raw=ET.tostring(node,encoding='unicode')
        for key in sorted([e.attrib['id'] for e in node.iter() if 'id' in e.attrib],key=len,reverse=True):
            raw=raw.replace(f'id="{key}"',f'id="{letter}_{key}"').replace(f'#{key})',f'#{letter}_{key})').replace(f'"#{key}"',f'"#{letter}_{key}"')
        node=ET.fromstring(raw);node.set('x',str(x*72));node.set('y',str((HEIGHT-y-h)*72));root.append(node)
        box=json.loads((here/'qa'/f'fig1{letter}.alignment.json').read_text())['layout']['panels'][0]['bbox_pt']
        panels.append({'id':letter,'bbox_pt':[box[0]+x*72,box[1]+y*72,box[2]+x*72,box[3]+y*72]})
    with pymupdf.open(here/'qa/fig1_annotations.pdf') as src:page.show_pdf_page(page.rect,src,0)
    root.append(ET.parse(here/'qa/fig1_annotations.svg').getroot())
    doc.save(here/'fig1.pdf',deflate=True)
    page.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72),alpha=False).save(here/'fig1_preview.png');doc.close()
    ET.ElementTree(root).write(here/'fig1.svg',encoding='unicode',xml_declaration=True)
    svg_path=here/'fig1.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    report=audit_layout_manifest({'schema_version':1,'backend':'native-size matplotlib vector composition',
        'figure':{'width_pt':WIDTH*72,'height_pt':HEIGHT*72},'panels':panels,
        'row_groups':[['e','f','g']],'column_groups':[['c','d']]})
    write_json_report(report,here/'qa/fig1.alignment.json')
    if exit_code(report,strict=True):raise RuntimeError('Figure 1 alignment failed')
    metadata['rendering_contract']['panel_placements_inches']=PLACES

def render_figure(arrays,metadata,here):
    rendered={}
    for letter,key in [('a','raw_magnitude'),('b','hadamard_magnitude'),('c','hadamard_range'),('d','nar_kmax_range')]:
        rendered[letter]=landscape(arrays[key],letter,arrays,metadata,here)
    for letter,key in [('e','trace_raw'),('f','trace_hadamard'),('g','trace_nar_kmax')]:trace(arrays[key],letter,metadata,here)
    metadata['rendering_contract']={'layout':'raw activation forks into Hadamard and PrismQuant; matched c/d outcomes; shared bottom trace strip',
        'size_inches':[WIDTH,HEIGHT],'previous_size_inches':[6.6,7.0],'png_dpi':300,
        'scientific_panels':rendered,'icons':'unscaled vector schematics only; no new experimental measurements',
        'preserved_content':'all original arrays, windows, numeric scales, traces, brackets and affine offset',
        'range_semantics':'Rotations change within-group range. Subtracting an affine offset alone does not change max-minus-min.',
        'palette':FIG1_PALETTE.copy()}
    metadata['trace_rendering'].update({'size_inches':[2.04,1.10],'x_ticks':[0,64,127],
        'y_ticks':[-5,0,5],'all_axes_labeled':False,'shared_y_label':'signed value','y_tick_labels_on_all_panels':True})
    metadata['palette']=metadata['rendering_contract']['palette']
    metadata['source_array_sha256']={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in arrays.items()}
    compose(here,metadata)
