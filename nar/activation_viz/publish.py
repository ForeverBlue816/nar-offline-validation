"""Publish independent local and global assets with measured integrity gates."""
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image
from .detail_plot import digest


def validate_figures(source):
    audits = json.loads((source/'qa/rendered_audit_index.json').read_text())
    pdfs = sorted(p for p in (source/'figures').rglob('*.pdf') if '.collision-audit' not in p.name)
    assert {r['pdf'] for r in audits} == {str(p.relative_to(source)) for p in pdfs}
    assert all(r['font_exit'] == r['collision_exit'] == 0 for r in audits)
    assert json.loads((source/'qa/detail_integrity.json').read_text())['passed']
    inventory = {r['file']: r['sha256'] for r in json.loads((source/'activation_inventory.json').read_text())}
    for mode in ('paired_local', 'end_to_end'):
        for site in ('q_proj', 'down_proj'):
            for quantity in ('raw', 'residual'):
                directory = source/'figures'/mode/site/quantity/'detail'
                for name in ('matrix', 'matrix_linear', 'rotated_only_zoom'):
                    geometry = json.loads((directory/f'{name}.geometry.json').read_text())
                    assert len(geometry['panels']) == (12 if name == 'rotated_only_zoom' else 16)
                assert digest(directory/'matrix.pdf') != digest(directory.parent/'overview/matrix.pdf')
    for p in (source/'figures').rglob('*.geometry.json'):
        if 'detail' not in p.parts:
            continue  # Preserved historical overviews carry their own archived renderer contract.
        for panel in json.loads(p.read_text())['panels']:
            assert panel['shape'] == [128, 512]
            assert panel['tokens'] == [0, 128] and panel['channels'] == [0, 512]
            assert panel['data_stride'] == [1, 1] and not panel['pooling']
            assert panel['surface_rcount'] == 128 and panel['surface_ccount'] == 512
            assert panel['vertices_processed'] == 65536 and panel['surface_cells'] == 64897
            assert not panel['draw_height_columns'] and not panel['draw_sidewalls'] and not panel['zero_padding']
            assert panel['vertical_height_segments'] == panel['sidewall_triangles'] == 0
            assert panel['source_sha256'] == inventory[panel['source']]
    return len(pdfs)


def run(source, destination):
    source, destination = Path(source), Path(destination)
    report = json.loads((source/'validation_report.json').read_text())
    assert report['required_checks_passed'] and report['full_resolution_metrics']['passed']
    count = validate_figures(source)
    records, dpi = [], []
    for p in sorted(source.rglob('*')):
        rel = p.relative_to(source)
        if not p.is_file() or rel.parts[0] == 'activations' or p.suffix in ('.pt', '.tmp') or rel.name == 'publication_manifest.json':
            continue
        if p.stat().st_size >= 100*1024*1024:
            raise RuntimeError(f'GitHub file size gate: {p}')
        if p.suffix == '.png' and rel.parts[0] == 'figures':
            with Image.open(p) as im:
                resolution = im.info.get('dpi')
                assert resolution and all(abs(v-600) < .1 for v in resolution)
                dpi.append({'file': str(rel), 'pixels': list(im.size), 'dpi': list(resolution)})
                im.verify()
        target = destination/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.resolve() != target.resolve():
            # No canonical-path substitution and no overview/detail hardlinks.
            if target.exists(): target.unlink()
            shutil.copy2(p, target)
        records.append({'file': str(rel), 'bytes': p.stat().st_size, 'sha256': digest(p)})
    result = {'files': records, 'figure_sets': count, 'png_exports': dpi,
        'signed_raw_tensors': 'retained at raw_activation_root; all 448 hashes in activation_inventory.json',
        'overall_numerical_checks_passed': report['passed'],
        'required_checks_passed': report['required_checks_passed'],
        'publication_bytes': sum(r['bytes'] for r in records),
        'views': {'detail': 'fixed 128 x 512 native upper surfaces', 'overview': 'independent full-domain context'},
        'revision': 'local-upper-surface-v1'}
    (destination/'publication_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PUBLICATION INVENTORY', len(records), 'files;', count, 'figure sets')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination')
    a = p.parse_args(); run(a.source, a.destination)
