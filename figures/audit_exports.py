#!/usr/bin/env python3
"""Run source and rendered preflight using the installed figure QA tools."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qa-tools',type=Path,default=Path.home()/'.codex/skills/nature-figure/scripts')
    parser.add_argument('--figure',choices=['all','4'],default='all')
    args=parser.parse_args();tools=args.qa_tools
    root=Path(__file__).resolve().parent; qa=root/'qa';qa.mkdir(exist_ok=True)
    summary={}
    panels=[f'fig1{x}' for x in 'abcdefg']+[f'fig2{x}' for x in 'abc']+[f'fig3{x}' for x in ['a','b','c','c1','c2']]
    targets=panels+['fig1','fig2','fig3']+['fig4a','fig4a_8b','fig4b','fig4']
    if args.figure=='4': targets=['fig4a','fig4a_8b','fig4b','fig4']
    for stem in targets:
        p=root/(stem+'.pdf')
        # The auditor omits an overlay for a clean PDF; remove old diagnostics.
        (qa/f'{stem}.collision.pdf').unlink(missing_ok=True)
        commands={
            'text':[str(tools/'audit_pdf_text.py'),str(p),'--min-pt','5','--json'],
            'collision':[str(tools/'audit_figure_collisions.py'),str(p),'--json-out',str(qa/f'{stem}.collision.json'),'--overlay-pdf',str(qa/f'{stem}.collision.pdf')],
        }
        for kind,cmd in commands.items():
            r=subprocess.run([sys.executable]+cmd,capture_output=True,text=True)
            (qa/f'{stem}.{kind}.txt').write_text(r.stdout+r.stderr)
            summary[f'{stem}.{kind}']={'exit_code':r.returncode}
            if r.returncode: print(stem,kind,r.stdout,file=sys.stderr)
    # Audit the source closure, because style/export/alignment live in the
    # shared module. Do not mistake an imported setting for a missing setting.
    with tempfile.TemporaryDirectory() as temp:
        shared=(root/'figure_style.py').read_text().replace('from __future__ import annotations','')
        sources=['make_fig4.py'] if args.figure=='4' else ['make_fig1.py','make_fig2.py','make_fig3.py','make_fig4.py']
        for name in sources:
            combined=Path(temp)/name
            combined.write_text(shared+'\n'+(root/name).read_text().replace('from __future__ import annotations',''))
            r=subprocess.run([sys.executable,str(tools/'validate_figure.py'),str(combined),'--json'],capture_output=True,text=True)
            raw=json.loads(r.stdout); raw['source']=name+' + figure_style.py'
            (qa/(name+'.source-closure.json')).write_text(json.dumps(raw,indent=2)+'\n')
            resolved=[]
            for finding in raw['findings']:
                if finding['level']=='FAIL':
                    if finding['check_id']=='FONT-FAMILY':
                        resolved.append({'check_id':'FONT-FAMILY','resolution':'User explicitly requests serif. The static checker only accepts sans-serif names; actual PDFs use embedded DejaVu Serif and all glyph sizes pass.'})
                    else:
                        raise RuntimeError(f'Unresolved source failure: {name}: {finding}')
            summary[name+'.source']={'raw_exit_code':r.returncode,'resolutions':resolved,'unresolved_failures':0}
    (qa/('fig4.summary.json' if args.figure=='4' else 'summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    if any(v.get('exit_code',0) for v in summary.values()): raise SystemExit(1)
    print('Rendered preflight: no failures. Inspect collision WARNs and source resolutions in qa/.')
if __name__=='__main__':main()
