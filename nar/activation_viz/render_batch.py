"""Parallel CPU rendering from measured caches, followed by PDF preflight."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path

def audit(root):
    root=Path(root);qa=Path(__file__).parent/'qa';reports=[]
    index = root/'qa/rendered_audit_index.json'
    previous = {r['pdf']: r for r in json.loads(index.read_text())} if index.exists() else {}
    for pdf in sorted((root/'figures').rglob('*.pdf')):
        if '.collision-audit' in pdf.name:continue
        relative = str(pdf.relative_to(root))
        if (root/'detail_render_config.json').exists() and 'detail' not in pdf.parts and relative in previous:
            reports.append(previous[relative]); continue
        text=subprocess.run([sys.executable,str(qa/'pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
        pdf.with_suffix('.font-audit.json').write_text(text.stdout)
        collision=subprocess.run([sys.executable,str(qa/'collisions.py'),str(pdf),'--json-out',str(pdf.with_suffix('.collision-audit.json'))],capture_output=True,text=True)
        reports.append({'pdf':str(pdf.relative_to(root)),'font_exit':text.returncode,'collision_exit':collision.returncode,
                        'font_stderr':text.stderr,'collision_summary':collision.stdout,'collision_stderr':collision.stderr})
    (root/'qa').mkdir(exist_ok=True)
    (root/'qa'/'rendered_audit_index.json').write_text(json.dumps(reports,indent=2)+'\n')
    bad=[r for r in reports if r['font_exit'] or r['collision_exit']]
    print('PDF AUDIT',len(reports),'blocking',len(bad),flush=True)
    if bad:raise RuntimeError('PDF checks require repair; see qa/rendered_audit_index.json')

def run(root,workers=8,parts='all'):
    from .full_batch import run as full_run
    root=Path(root)
    manifest=json.loads((root/'run_manifest.json').read_text())
    source=root if (root/'activations').exists() else Path(manifest['raw_activation_root']).parent
    destination=root.with_name(root.name+'_full') if source==root else root
    if parts=='matrix':
        from .plot import run as plot_run
        plot_run(root,selected='matrices')
    else:full_run(source,destination,workers)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--parts',choices=['all','matrix'],default='all');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:audit(a.root)
    else:run(a.root,a.workers,a.parts)
