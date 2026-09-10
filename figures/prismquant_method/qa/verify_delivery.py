#!/usr/bin/env python3
"""Check native editability, measured panel geometry and actual PDF typography."""
import json,sys,zipfile,hashlib
from pathlib import Path
import pymupdf as fitz
from lxml import etree
ROOT=Path(__file__).resolve().parent.parent;QA=ROOT/'qa'
PDF=ROOT/'PrismQuant_method_refined.pdf';PPT=ROOT/'PrismQuant_method_refined.pptx'
page=fitz.open(PDF)[0];k=page.rect.width/1600
source=json.loads((QA/'layout_source.json').read_text())
expected=source['panels'];drawings=page.get_drawings();measured=[]
labels={'construct':'Construct','represent':'Represent','deploy':'Deploy','attention':'(a)  Attention','ffn':'(b)  FFN'}
traces=page.get_texttrace()
for panel in expected:
 x,y,w,h=panel['rect'];target=fitz.Rect(x*k,y*k,(x+w)*k,(y+h)*k)
 match=min(drawings,key=lambda d:sum(abs(a-b) for a,b in zip(d['rect'],target)))
 error=max(abs(a-b) for a,b in zip(match['rect'],target))
 assert error<.15,(panel['id'],error)
 label=labels[panel['id']]
 options=[t for t in traces if ''.join(chr(c[0]) for c in t['chars'])==label]
 assert options,label
 t=options[0]
 # The layout auditor uses a bottom-left coordinate system.
 r=match['rect'];bbox=[r.x0,page.rect.height-r.y1,r.x1,page.rect.height-r.y0]
 measured.append({'id':panel['id'],'bbox_pt':bbox,'panel_label_anchor_pt':[t['bbox'][0],page.rect.height-t['chars'][0][2][1]]})
manifest={'schema_version':1,'backend':'Python native PowerPoint; geometry measured from actual LibreOffice PDF','figure':{'width_pt':page.rect.width,'height_pt':page.rect.height},'panels':measured,'row_groups':[['construct','represent','deploy'],['attention','ffn']],'column_groups':[],'exemptions':[{'panels':['construct','represent','deploy'],'checks':['panel-width'],'reason':'The user requests a 24% / 26% / 50% card-width ratio.'}]}
(QA/'alignment_measured.json').write_text(json.dumps(manifest,indent=2))
fonts=sorted(set(t['font'] for t in traces));minimum=min(t['size'] for t in traces)
assert minimum>=5,(minimum,fonts)
assert all(s.startswith('TimesNewRoman') for s in fonts),fonts
assert not page.get_images(),'Raster content in vector PDF'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(PPT) as z:
 slide=etree.fromstring(z.read('ppt/slides/slide1.xml'));pres=etree.fromstring(z.read('ppt/presentation.xml'))
 media=[n for n in z.namelist() if n.startswith('ppt/media/')]
 assert not media,media
 pictures=slide.xpath('//p:pic',namespaces=NS);assert not pictures
 text=slide.xpath('//a:t/text()',namespaces=NS)
 text_xml=' '.join(text)
 for bad in ['single-pass','1.8×','~1%','slots filled','\\frac','\\sqrt']:
  assert bad not in text_xml,bad
 declared=slide.xpath('//a:latin/@typeface',namespaces=NS)
 assert set(declared)=={'Times New Roman'},set(declared)
 size=pres.find('p:sldSz',NS);assert int(size.get('cx'))==2*int(size.get('cy'))
 summary={'pptx':{'native_shapes':len(slide.xpath('//p:sp',namespaces=NS)),'native_connectors':len(slide.xpath('//p:cxnSp',namespaces=NS)),'module_groups':len(slide.xpath('//p:grpSp',namespaces=NS)),'editable_text_runs':len(text),'picture_objects':len(pictures),'media_files':len(media),'font_names':sorted(set(declared))},'pdf':{'width_mm':page.rect.width*25.4/72,'height_mm':page.rect.height*25.4/72,'minimum_rendered_font_pt':minimum,'fonts':fonts,'embedded_fonts':page.get_fonts(full=True),'raster_images':len(page.get_images()),'text_characters':len(page.get_text())},'geometry':'Five panel frames measured directly in the final rendered PDF, with native label baselines.','file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [PPT,PDF,ROOT/'PrismQuant_method_refined.png']}}
(QA/'delivery_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
