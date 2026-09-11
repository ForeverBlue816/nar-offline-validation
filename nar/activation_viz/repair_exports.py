"""Repair individually flagged exports and refresh their PDF audit records."""
import argparse,concurrent.futures,hashlib,json,shutil,subprocess,sys
from pathlib import Path

def draw(task):
    source,destination,mode,site,quantity,name=task
    from .full_plot import matrix,style
    style();method,block=name.removeprefix('panel_').rsplit('_block',1)
    matrix(Path(source),Path(destination)/'figures',mode,site,quantity,[method],[int(block)-1],name=name)
    directory=Path(destination)/'figures'/mode/site/quantity
    for p in (directory/'overview').glob(name+'.*'):
        if p.suffix in ('.png','.pdf','.svg') or p.name.endswith(('.geometry.json','.alignment.json','.scales.json')):
            shutil.copy2(p,directory/'detail'/p.name)
    return mode,site,quantity,name

def run(source,destination):
    destination=Path(destination);index=json.loads((destination/'qa/rendered_audit_index.json').read_text())
    targets=set()
    for r in index:
        if r['font_exit'] or r['collision_exit']:
            parts=Path(r['pdf']).parts;name=Path(parts[-1]).stem
            assert name.startswith('panel_'),r['pdf']
            targets.add((str(source),str(destination),parts[1],parts[2],parts[3],name))
    assert targets,'No failed exports to repair'
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:fixed=list(pool.map(draw,sorted(targets)))
    qa=Path(__file__).parent/'qa';updated=[]
    for r in index:
        p=destination/r['pdf'];parts=Path(r['pdf']).parts
        if (parts[1],parts[2],parts[3],p.stem) not in fixed:continue
        text=subprocess.run([sys.executable,str(qa/'pdf_text.py'),str(p),'--min-pt','5','--json'],capture_output=True,text=True)
        p.with_suffix('.font-audit.json').write_text(text.stdout)
        collision=subprocess.run([sys.executable,str(qa/'collisions.py'),str(p),'--json-out',str(p.with_suffix('.collision-audit.json'))],capture_output=True,text=True)
        r.update(font_exit=text.returncode,collision_exit=collision.returncode,font_stderr=text.stderr,collision_summary=collision.stdout,collision_stderr=collision.stderr);updated.append(r['pdf'])
    (destination/'qa/rendered_audit_index.json').write_text(json.dumps(index,indent=2)+'\n')
    provenance=json.loads((destination/'render_provenance.json').read_text())
    provenance['clipped_label_repair']={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':updated,'command':sys.argv,'plot_source_sha256':hashlib.sha256((Path(__file__).parent/'full_plot.py').read_bytes()).hexdigest(),'change':'center single-panel z tick labels only when measured PDF-canvas bounds would overflow; data and all other plots unchanged'}
    (destination/'render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    failures=[r for r in index if r['font_exit'] or r['collision_exit']]
    print('REPAIRED',len(updated),'PDFs; blocking',len(failures),flush=True)
    assert not failures,failures

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();run(a.source,a.destination)
