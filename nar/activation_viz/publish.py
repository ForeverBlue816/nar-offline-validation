"""Copy the reviewable figure bundle to Git; retain multi-GB signed tensors on disk."""
import argparse,hashlib,json,shutil,os
from pathlib import Path
from PIL import Image

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def run(source,destination):
    source=Path(source);destination=Path(destination)
    report=json.loads((source/'validation_report.json').read_text())
    assert report['required_checks_passed'] and report['full_resolution_metrics']['passed']
    audits=json.loads((source/'qa/rendered_audit_index.json').read_text())
    assert len(audits)==308 and all(r['font_exit']==r['collision_exit']==0 for r in audits)
    provenance=json.loads((source/'render_provenance.json').read_text())
    if provenance.get('pooling') is False:
        assert json.loads((source/'qa/full_resolution_integrity.json').read_text())['passed']
        assert json.loads((source/'qa/exact_ecdf_checks.json').read_text())['passed']
        inventory={r['file']:r['sha256'] for r in json.loads((source/'activation_inventory.json').read_text())}
        geometry=list((source/'figures').rglob('*.geometry.json'))
        assert len(geometry)==304
        for p in geometry:
            for panel in json.loads(p.read_text())['panels']:
                assert panel['data_stride']==[1,1] and not panel['pooling'] and not panel['cropping']
                rows,cols=panel['shape']
                assert rows==2048 and cols in (4096,12288)
                assert panel['vertices_processed']==rows*cols
                assert panel['surface_triangles']==2*(rows-1)*(cols-1)
                assert panel['base_z']==0
                assert panel['source_sha256']==inventory[panel['source']]
        for name in ('display_cache.npz','range_ecdf.npz'):
            (destination/name).unlink(missing_ok=True)
    for ext in ('png','pdf','svg'):
        assert len(list((source/'figures').rglob('*.'+ext)))==308
    records=[];dpi=[];indexed={};dpi_index={}
    files=sorted(source.rglob('*'),key=lambda p:('detail' in p.relative_to(source).parts,str(p)))
    for p in files:
        rel=p.relative_to(source)
        if not p.is_file() or rel.parts[0]=='activations' or p.suffix in ('.pt','.tmp'):continue
        if p.stat().st_size>=100*1024*1024:raise RuntimeError(f'GitHub file size gate: {p}')
        canonical=Path(*('overview' if part=='detail' else part for part in rel.parts))
        if provenance.get('pooling') is False and canonical!=rel and p.suffix in ('.png','.pdf','.svg'):
            original=indexed[str(canonical)];assert original['bytes']==p.stat().st_size
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);target.unlink(missing_ok=True)
            os.link(destination/canonical,target)
            records.append({**original,'file':str(rel)})
            if p.suffix=='.png':dpi.append({**dpi_index[str(canonical)],'file':str(rel)})
            continue
        if p.suffix=='.png':
            with Image.open(p) as im:
                resolution=im.info.get('dpi');assert resolution and all(abs(v-600)<.1 for v in resolution)
                entry={'file':str(rel),'pixels':list(im.size),'dpi':list(resolution)}
                dpi.append(entry);dpi_index[str(rel)]=entry
                im.verify()
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        entry={'file':str(rel),'bytes':p.stat().st_size,'sha256':digest(p)}
        records.append(entry);indexed[str(rel)]=entry
    result={'files':records,'figure_sets':308,'png_exports':dpi,
        'signed_raw_tensors':'retained at raw_activation_root in run_manifest.json; all 448 shard hashes are in activation_inventory.json',
        'overall_numerical_checks_passed':report['passed'],'required_checks_passed':report['required_checks_passed'],
        'publication_bytes':sum(r['bytes'] for r in records)}
    (destination/'publication_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BUNDLE COPIED',len(records),'files',result['publication_bytes'],'bytes')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();run(a.source,a.destination)
