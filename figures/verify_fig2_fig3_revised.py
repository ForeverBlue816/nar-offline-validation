#!/usr/bin/env python3
"""Independent source, observation-count, font, geometry, SVG and raster export checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from verify_deployment_figures import cairo_font

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;QA=HERE/'qa/fig2_fig3'
BASES=[HERE/'fig2_revised',HERE/'fig3_revised',HERE/'appendix/fig3_energy_all_token_context',HERE/'fig2_caption',HERE/'fig3_caption']
NS={'s':'http://www.w3.org/2000/svg'}


def verify(promote=False):
    metadata=json.loads((HERE/'fig2_fig3_source_metadata.json').read_text())
    for path,sha in metadata['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    d2=pd.read_csv(HERE/'fig2_revised_data.csv');d3=pd.read_csv(HERE/'fig3_revised_data.csv');bins=pd.read_csv(HERE/'fig3_moe_binned_summary.csv')
    assert len(d2)==252 and d2.groupby(['panel','method']).size().eq(28).all()
    raw2=pd.read_csv(HERE/'fig2_capture.csv');raw2=raw2[raw2.model.eq('llama32_3b')&raw2.site.eq('down')]
    for row in d2.itertuples():
        original=raw2[raw2.layer.eq(row.layer)&raw2.method.eq(row.method)].iloc[0]
        np.testing.assert_allclose(row.y,original[row.metric],rtol=1e-12,atol=1e-12)
    expected={'a':8066,'b':1024,'c':2912,'d':5342};assert d3.groupby('panel').size().to_dict()==expected
    for layer in [1,13,27]:
        before=pd.read_csv(HERE/'fig3_eigenspace_r256.csv');before=before[before.layer.eq(layer)].sort_values('rank')
        after=d3[d3.panel.eq('b')&d3.layer.eq(layer)].sort_values('x')
        np.testing.assert_array_equal(before['rank'],after.x);np.testing.assert_allclose(before.cumulative_fraction_total_energy,after.y,rtol=1e-12,atol=1e-12)
    assert sorted(d3[d3.panel.eq('b')].layer.unique())==metadata['energy_layers']
    dense=d3[d3.panel.eq('c')];moe=d3[d3.panel.eq('d')]
    xy=dense[['x','y']].to_numpy();co=np.linalg.lstsq(np.column_stack([np.ones(len(xy)),xy[:,0]]),xy[:,1],rcond=None)[0]
    np.testing.assert_allclose(co,[metadata['reference_fit'][k] for k in ['intercept','slope']],rtol=1e-12,atol=1e-12)
    assert not moe.point_id.duplicated().any() and len(bins)==10 and bins.n.sum()==5342 and bins.n.max()-bins.n.min()==1
    for row in bins.itertuples():
        points=moe[moe.f_bin.eq(row.f_bin)]
        assert set(row.point_ids.split(';'))==set(points.point_id)
        np.testing.assert_allclose([row.x_median,row.y_median,row.y_q1,row.y_q3],
            [points.x.median(),points.y.median(),points.y.quantile(.25),points.y.quantile(.75)],rtol=1e-12,atol=1e-12)
    geometry=json.loads((HERE/'fig3_geometry_metadata.json').read_text())
    vectors=np.load(HERE/'fig3_geometry_vectors.npz')['vectors'];np.testing.assert_allclose(vectors.T@vectors,np.eye(2),atol=1e-5)
    assert abs(geometry['arrows']['nar']['in_plane_length']-1)<1e-5 and geometry['arrows']['hadamard']['in_plane_length']<.03
    for name in ['make_fig2_revised.py','make_fig3_revised.py']:
        cmd=[sys.executable,str(HERE/'qa_tools/validate_figure.py'),str(HERE/name),'--font-family','Times New Roman','--include-source',str(HERE/'deployment_figure_style.py'),'--json']
        r=subprocess.run(cmd,capture_output=True,text=True,check=True);(QA/f'{name}.source.json').write_text(r.stdout)
    font=cairo_font();import cairosvg
    report={'status':'PASS','source_commit':metadata['source_commit'],'frozen_files':len(metadata['source_hashes']),
        'figure2_points':252,'figure3_points_by_panel':expected,'all_expert_points_and_bin_membership_verified':True,
        'reference_fit_unchanged':True,'original_three_energy_curves_unchanged':True,'font':font,'assets':[]}
    for base in BASES+[p.with_suffix('') for p in sorted((HERE/'panels/fig2_fig3').glob('*.pdf'))]:
        pdf=pymupdf.open(base.with_suffix('.pdf'));assert len(pdf)==1;page=pdf[0]
        assert not page.get_images(),base
        fonts=page.get_fonts(full=True);assert fonts and all('TimesNewRomanPS-BoldMT' in f[3] for f in fonts)
        assert all(len(pdf.extract_font(f[0])[3]) for f in fonts)
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        minimum=min(s['size'] for s in spans);assert minimum>=7.49,(base,minimum)
        for s in spans:assert page.rect.contains(pymupdf.Rect(s['bbox'])),(base,s)
        collision=json.loads((QA/f'{base.name}.collision.json').read_text())
        assert collision['auditable'] and collision['summary']['fail']==collision['summary']['warn']==0,(base,collision['summary'])
        svg=ET.parse(base.with_suffix('.svg')).getroot();ids=[e.get('id') for e in svg.iter() if e.get('id')]
        assert len(ids)==len(set(ids)) and not svg.findall('.//s:svg',NS) and not svg.findall('.//s:image',NS)
        references=[]
        for element in svg.iter():
            for key,value in element.attrib.items():
                references+=re.findall(r'url\(#([^)]*)\)',value)
                if key.endswith('href') and value.startswith('#'):references.append(value[1:])
        assert not set(references)-set(ids)
        texts=svg.findall('.//s:text',NS);assert texts and all('Times New Roman' in t.attrib.get('style','') for t in texts)
        with Image.open(base.with_suffix('.png')) as im:
            assert min(im.info['dpi'])>599 and abs(im.width-page.rect.width/72*600)<3 and abs(im.height-page.rect.height/72*600)<3
        if base in BASES:
            assert abs(page.rect.width/72-5.5)<.001
            alignment=json.loads((QA/f'{base.name}.alignment.json').read_text());assert alignment['verdict'] in ['PASS','NOT APPLICABLE']
            page.get_pixmap(dpi=150).save(QA/f'{base.name}.pdf-render.png')
            cairosvg.svg2png(url=str(base.with_suffix('.svg')),write_to=str(QA/f'{base.name}.svg-render.png'),output_width=round(page.rect.width/72*150),output_height=round(page.rect.height/72*150))
            Image.open(QA/f'{base.name}.pdf-render.png').convert('L').save(QA/f'{base.name}.grayscale.png')
            if base.name in ['fig2_revised','fig3_revised']:
                # Both independent renderers must retain substantial content in every grid cell.
                pdfim=np.asarray(Image.open(QA/f'{base.name}.pdf-render.png').convert('RGB'))
                svgim=np.asarray(Image.open(QA/f'{base.name}.svg-render.png').convert('RGBA'))
                rgb=svgim[:,:,:3]*(svgim[:,:,3:4]/255)+255*(1-svgim[:,:,3:4]/255)
                rows,cols=(1,3) if base.name=='fig2_revised' else (2,2)
                for row in range(rows):
                    for col in range(cols):
                        def ink(im):
                            y0,y1=[round(v*im.shape[0]/rows) for v in [row,row+1]];x0,x1=[round(v*im.shape[1]/cols) for v in [col,col+1]]
                            return int(np.any(im[y0:y1,x0:x1,:3]<230,axis=2).sum())
                        ratio=ink(rgb)/ink(pdfim);assert .65<ratio<1.35,(base,row,col,ratio)
        if base.name=='fig3_revised':
            groups=[e for e in svg.findall('.//s:g',NS) if (e.get('id') or '').startswith('PathCollection_')]
            counts=[len(g.findall('.//s:use',NS))+len(g.findall('s:path',NS)) for g in groups]
            assert counts==[8064,2520,280,112,1200,4142],counts
            report['svg_scatter_counts']=counts
            all_text=' '.join(t.text or '' for t in texts)
            assert all(tag not in all_text for tag in ['E1c','E7','E20','E26'])
            assert all(label in page.get_text() for label in ['Activations','V-cache','Multi-slot','0.61','0.43'])
        report['assets'].append({'file':str(base.relative_to(ROOT)),'minimum_font_pt':minimum,'vector_only':True,'embedded_Times':True,'editable_SVG':True,'PNG_dpi':600,'collisions':0})
    captions=(HERE/'captions_fig2_fig3.txt').read_text();assert not re.search(r'\bE(?:1c|7|20|26)\b',captions)
    (QA/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in QA.glob('*.json'):path.write_text(path.read_text().replace(str(ROOT)+'/', ''))
    if promote:
        old2=json.loads((HERE/'archive/fig2_fig3_before_oral_revision/fig2_metadata.json').read_text())
        for number,letters in [(2,'abc'),(3,'abcd')]:
            for ext in ['pdf','svg','png']:
                shutil.copy2(HERE/f'fig{number}_revised.{ext}',HERE/f'fig{number}.{ext}')
                for letter in letters:shutil.copy2(HERE/f'panels/fig2_fig3/fig{number}_revised_{letter}.{ext}',HERE/f'fig{number}{letter}.{ext}')
            shutil.copy2(HERE/f'fig{number}_revised.png',HERE/f'fig{number}_preview.png')
            meta=json.loads((HERE/f'fig{number}_revised_metadata.json').read_text());meta['revision']='oral-2026-09-13'
            if number==2:
                meta['duquant_offline_addendum']=old2['duquant_offline_addendum']
                meta['mean_range_reduction_percent']=meta['mean_reduction_percent']['b'];meta['mean_nmse_reduction_percent']=meta['mean_reduction_percent']['c']
            else:meta['geometry']=json.loads((HERE/'archive/fig2_fig3_before_oral_revision/fig3_metadata.json').read_text())['geometry']
            (HERE/f'fig{number}_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f"PASS: {len(report['assets'])} Figure 2/3 vector/600-dpi assets; all observations, font, SVG counts and independent renders verified.")
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--promote',action='store_true');args=ap.parse_args();verify(args.promote)

if __name__=='__main__':main()
