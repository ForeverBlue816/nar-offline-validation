"""Parallel CPU rendering from measured caches, followed by PDF preflight."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path

def draw(task):
    root,select,parts=task
    from .plot import run
    run(root,select,parts)
    return select

def audit(root):
    root=Path(root);qa=Path(__file__).parent/'qa';reports=[]
    for pdf in sorted((root/'figures').rglob('*.pdf')):
        if '.collision-audit' in pdf.name:continue
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

def run(root,workers=4,parts='all'):
    selectors=[f'{mode}/{site}/{quantity}/{view}' for mode in ('paired_local','end_to_end')
               for site in ('down_proj','q_proj') for quantity in ('raw','residual') for view in ('overview','detail')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for select in pool.map(draw,[(root,s,parts) for s in selectors]):print('FINISHED',select,flush=True)
    from .plot import style,distributions
    style();distributions(Path(root),Path(root)/'figures')
    from .report import run as report_run
    report_run(root)
    audit(root)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--parts',choices=['all','matrix'],default='all');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:audit(a.root)
    else:run(a.root,a.workers,a.parts)
