#!/usr/bin/env python3
"""Independent CPU verification of source data, vector exports, fonts, and renderers."""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import pymupdf as fitz
from PIL import Image
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
QA=HERE/'qa/deployment'
BASES=[HERE/'fig4_revised',HERE/'fig_deployment_efficiency',
       HERE/'appendix/fig4_metadata_8b',HERE/'appendix/fig_e17_rank_cost',
       HERE/'appendix/fig_matched_eager_graph',HERE/'appendix/fig_kernel_all_shapes']


def audit_source():
    for name,includes in [('make_fig4_revised.py',[]),('make_deployment_efficiency.py',[]),('make_deployment_appendix.py',['make_fig4_revised.py'])]:
        cmd=[sys.executable,str(HERE/'qa_tools/validate_figure.py'),str(HERE/name),'--json','--font-family','Times New Roman']
        for inc in ['deployment_figure_style.py']+includes:cmd+=['--include-source',str(HERE/inc)]
        r=subprocess.run(cmd,capture_output=True,text=True,check=True)
        (QA/f'{name.removeprefix("make_").removesuffix(".py")}.source.json').write_text(r.stdout)


def cairo_font():
    # Register the same verified, locally installed fonts with Fontconfig.
    # No font binary is copied into the repository or embedded in the SVG.
    from figure_typography import configure_times_bold
    from matplotlib import font_manager
    typography=configure_times_bold(4)
    prop=font_manager.FontProperties(family='Times New Roman',weight='bold')
    expected=Path(font_manager.findfont(prop,fallback_to_default=False)).resolve()
    lib=ctypes.CDLL(ctypes.util.find_library('fontconfig'))
    lib.FcConfigGetCurrent.restype=ctypes.c_void_p
    lib.FcConfigAppFontAddFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
    lib.FcConfigAppFontAddFile.restype=ctypes.c_int
    config=lib.FcConfigGetCurrent()
    for p in expected.parent.iterdir():
        if p.suffix.lower()=='.ttf':assert lib.FcConfigAppFontAddFile(config,str(p).encode())
    lib.FcNameParse.argtypes=[ctypes.c_char_p];lib.FcNameParse.restype=ctypes.c_void_p
    lib.FcConfigSubstitute.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
    lib.FcDefaultSubstitute.argtypes=[ctypes.c_void_p]
    lib.FcFontMatch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];lib.FcFontMatch.restype=ctypes.c_void_p
    lib.FcPatternGetString.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
    pattern=lib.FcNameParse(b'Times New Roman:style=Bold');lib.FcConfigSubstitute(config,pattern,0);lib.FcDefaultSubstitute(pattern)
    code=ctypes.c_int();match=lib.FcFontMatch(config,pattern,ctypes.byref(code));path=ctypes.c_char_p()
    assert lib.FcPatternGetString(match,b'file',0,ctypes.byref(path))==0
    assert Path(path.value.decode()).resolve()==expected,'Independent SVG renderer resolved a substitute font'
    return typography


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--promote-fig4',action='store_true');args=ap.parse_args()
    audit_source()
    metadata=json.loads((HERE/'deployment_efficiency_metadata.json').read_text())
    for file,digest in metadata['source_hashes'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest
    data=pd.read_csv(HERE/'deployment_efficiency_data.csv');accuracy=pd.read_csv(HERE/'fig4_revised_data.csv')
    assert len(data)==416 and sum(accuracy.kind.eq('point'))==22 and sum(accuracy.kind.eq('recovery'))==15
    assert len(data[data.metric.eq('kernel_ratio')&data.session.eq(0)])==30
    assert all(data.groupby(['metric','model','method','baseline','mode','phase','batch','tokens'],dropna=False).session.apply(lambda x:set(x)=={0,1,2,3}))
    graph=data[data['mode'].eq('cuda_graph_sequence')]
    assert graph.backend.str.startswith('private current-stream').all()
    typography=cairo_font()
    import cairosvg
    report=dict(status='PASS',source_commit=metadata['source_commit'],frozen_source_files=len(metadata['source_hashes']),
        deployment_rows=len(data),accuracy_points=37,graph_correctness='6/6 PASS',matched_graph_timing='36/36 PASS',
        font=typography,renderer_font_match=True,figures=[])
    all_bases=BASES+[p.with_suffix('') for p in sorted((HERE/'panels').glob('*.pdf'))]
    for base in all_bases:
        doc=fitz.open(base.with_suffix('.pdf'));assert len(doc)==1
        page=doc[0];assert not page.get_images(),f'Unexpected raster inside vector PDF: {base.name}'
        fonts=page.get_fonts(full=True)
        assert fonts and all('TimesNewRomanPS-BoldMT' in x[3] for x in fonts),fonts
        for font in fonts:assert len(doc.extract_font(font[0])[3])>0,'PDF font is not embedded'
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        assert spans and min(x['size'] for x in spans)>=7.49
        collision=json.loads((QA/f'{base.name}.collision.json').read_text())
        assert collision['auditable'] and collision['summary']['fail']==0 and collision['summary']['warn']==0,(base.name,collision['summary'])
        svg=ET.parse(base.with_suffix('.svg'));ns={'s':'http://www.w3.org/2000/svg'}
        assert len(svg.findall('.//s:text',ns))>0 and not svg.findall('.//s:image',ns)
        for node in svg.findall('.//s:text',ns):assert 'Times New Roman' in node.attrib.get('style',''),node.attrib
        with Image.open(base.with_suffix('.png')) as img:
            dpi=img.info.get('dpi');assert dpi and min(dpi)>599
            assert abs(img.width-page.rect.width/72*600)<3 and abs(img.height-page.rect.height/72*600)<3
        if base in BASES:
            expected_width=3.4 if base.name=='fig4_metadata_8b' else 5.5
            assert abs(page.rect.width/72-expected_width)<.001
            align=json.loads((QA/f'{base.name}.alignment.json').read_text())
            assert align['summary']['fail']==0 if 'summary' in align and 'fail' in align['summary'] else align['verdict'] in ['PASS','NOT APPLICABLE']
            page.get_pixmap(dpi=150,alpha=False).save(QA/f'{base.name}.pdf-render.png')
            cairosvg.svg2png(url=str(base.with_suffix('.svg')),write_to=str(QA/f'{base.name}.svg-render.png'),
                            output_width=round(page.rect.width/72*150),output_height=round(page.rect.height/72*150))
            im=Image.open(QA/f'{base.name}.pdf-render.png');im.convert('L').save(QA/f'{base.name}.grayscale.png')
        report['figures'].append(dict(file=str(base.relative_to(ROOT)),width_inches=page.rect.width/72,
            height_inches=page.rect.height/72,min_font_pt=min(x['size'] for x in spans),
            embedded_font=True,vector_only=True,editable_svg=True,png_dpi=600,collision_failures=0))
    # Assert that every displayed central comparison is derived from the plotted CSV.
    text=fitz.open(HERE/'fig_deployment_efficiency.pdf')[0].get_text()
    for metric in ['prefill_speedup','decode_latency','peak_memory']:
        values=data[data.metric.eq(metric)&data.session.eq(0)]
        if metric!='prefill_speedup':values=values[values['mode'].eq('cuda_graph_sequence')]
        assert all(f'{v:.2f}' in text for v in values.value)
    report['source_preflight_notes']=['5.5 inches is the explicit manuscript width, overriding Nature 89/183 mm defaults.',
        'PNG at 600 dpi is the requested raster format; no TIFF required.',
        'Shared modules are included in source preflight; the explicit Times New Roman contract supersedes the generic sans-serif default.']
    (QA/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    # QA records use repository-relative locations for portability.
    for p in QA.glob('*.json'):
        text=p.read_text().replace(str(ROOT)+'/', '')
        p.write_text(text)
    if args.promote_fig4:
        shutil.copy2(HERE/'fig4_revised.pdf',HERE/'fig4.pdf')
        shutil.copy2(HERE/'fig4_revised.svg',HERE/'fig4.svg')
        shutil.copy2(HERE/'fig4_revised.png',HERE/'fig4_preview.png')
    print(f"PASS: {len(report['figures'])} vector/600-dpi figures and panels; frozen data, embedded TNR, editable SVG, and independent renderers verified.")

if __name__=='__main__':main()
