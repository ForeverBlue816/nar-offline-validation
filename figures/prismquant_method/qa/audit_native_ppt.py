#!/usr/bin/env python3
"""Run the collision engine on native PDF text traces instead of inferred lines.

The generic line parser merges the boxed legend symbol with its description,
and unions stacked W/subscript/superscript text into overlapping paragraphs.
This adapter measures glyph outlines from the embedded TrueType fonts for independent drawing runs. Stroke intersection,
text overlap and page-clipping thresholds are unchanged.
"""
import argparse,importlib.util,sys,json,io,logging
import pymupdf as fitz
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
logging.getLogger("fontTools").setLevel(logging.ERROR)
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--engine',type=Path,default=Path(__file__).with_name('audit_figure_collisions.py'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
spec=importlib.util.spec_from_file_location('collision_engine',args.engine)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
original=m.extract_pdf_geometry

def extract_native(path):
    pages=original(path)
    doc=fitz.open(path)
    for page,rendered in zip(pages,doc):
        fonts={}
        for font in rendered.get_fonts(full=True):
            name,ext,kind,data=doc.extract_font(font[0])
            if ext!='ttf':raise RuntimeError('Only embedded TrueType glyphs are supported.')
            ft=TTFont(io.BytesIO(data));gs=ft.getGlyphSet();order=ft.getGlyphOrder();bounds={}
            for gid,gname in enumerate(order):
                pen=BoundsPen(gs);gs[gname].draw(pen);bounds[gid]=pen.bounds
            fonts[name.split('+')[-1]]=(ft['head'].unitsPerEm,bounds)
        texts=[]
        for trace in rendered.get_texttrace():
            upem,bounds=fonts[trace['font']];scale=trace['size']/upem;boxes=[];letters=[]
            for cp,gid,origin,_ in trace['chars']:
                bound=bounds.get(gid)
                if bound is None:continue
                gx0,gy0,gx1,gy1=bound;ox,oy=origin
                boxes.append((ox+gx0*scale,oy-gy1*scale,ox+gx1*scale,oy-gy0*scale))
                letters.append(chr(cp))
            if boxes:texts.append(m.TextBox(index=len(texts),text=''.join(letters),bbox=m.union_rects(boxes)))
        page.texts=texts
        page.traces=[m.TraceBox(index=t.index,text=t.text,bbox=t.bbox) for t in texts]
    return pages
m.extract_pdf_geometry=extract_native
report=m.audit_pdf(args.pdf)
report['extraction']='Native PDF text traces with actual bounds from embedded TrueType glyph outlines; no paragraph unions or empty em-box space. Original collision thresholds unchanged.'
args.out.write_text(json.dumps(report,indent=2))
print(m.render_text(report))
sys.exit(m.exit_code(report,False))
