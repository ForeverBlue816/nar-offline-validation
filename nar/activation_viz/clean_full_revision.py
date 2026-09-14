"""Reproduce the six user-selected matrices and their scoped integrity/PDF audits.

Descriptive measured surfaces from one fixed sample; no uncertainty estimates.
See clean_full_revision_contract.md in the published activation run.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pymupdf
from PIL import Image

from . import detail_plot as local
from . import full_plot as full
from .plot import METHODS, LAYERS

TARGETS = ['end_to_end/q_proj/raw', 'end_to_end/down_proj/raw', 'paired_local/q_proj/raw']
VIEWS = ('detail', 'overview')
REVISION = 'clean-footer-full-reference-v1'


def stems():
    return [f'figures/{selection}/{view}/matrix' for selection in TARGETS for view in VIEWS]


def prepare(root):
    qa = root/'qa/clean_full_revision'
    qa.mkdir(parents=True, exist_ok=True)
    baseline = qa/'baseline.json'
    if baseline.exists():
        return
    manifest = json.loads((root/'publication_manifest.json').read_text())
    protected = {r['file']: r['sha256'] for r in manifest['files']
                 if r['file'].startswith('figures/') and
                 not any(r['file'].startswith(stem + '.') for stem in stems())}
    immutable_names = ['run_manifest.json', 'activation_inventory.json', 'inputs.json',
                       'metrics_per_sample.csv', 'metrics_summary.csv', 'validation_report.json',
                       'numerical_edge_checks.json', 'measured_summary.md', 'capture_site_report.md']
    local.write_json(baseline, {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'geometry': {s: json.loads((root/(s+'.geometry.json')).read_text()) for s in stems()},
        'protected_figures': protected,
        'immutable_records': {name: local.digest(root/name) for name in immutable_names},
    })


def render(args):
    source, root, selection, view = args
    source, root = Path(source), Path(root)
    mode, site, quantity = selection.split('/')
    print('START', selection, view, flush=True)
    if view == 'detail':
        local.matrix(source, root, mode, site, quantity)
    else:
        full.matrix(source, root/'figures', mode, site, quantity, METHODS)
        p = root/'figures'/selection/'overview/matrix.geometry.json'
        g = json.loads(p.read_text())
        g.update(revision=REVISION, reference_source_mode='paired_local',
                 reference_precision='unquantized norm-fused FP32',
                 plot_source_sha256=local.digest(full.__file__),
                 rasterizer_source_sha256=local.digest(Path(full.__file__).with_name('full_surface.py')),
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 camera={'projection': 'persp', 'elev': 25, 'azim': -60,
                         'box_aspect': [1.28, 1, .72], 'zoom': .88},
                 color={'cmap': full.CMAP.name, 'norm': 'Normalize', 'vmin': 0},
                 intended_insertion_width_inches=7.2,
                 font_family='DejaVu Sans',
                 explanatory_footer_visible=False)
        local.write_json(p, g)
    print('EXPORTED', selection, view, flush=True)


def audit(root):
    out = root/'qa/clean_full_revision'
    baseline = json.loads((out/'baseline.json').read_text())
    inventory = {r['file']: r for r in json.loads((root/'activation_inventory.json').read_text())}
    index_path = root/'qa/rendered_audit_index.json'
    index = {r['pdf']: r for r in json.loads(index_path.read_text())}
    reviews_path = root/'qa/detail_pdf_render_index.json'
    reviews = {r['pdf']: r for r in json.loads(reviews_path.read_text())}
    qa = Path(__file__).parent/'qa'
    records = []
    for s in stems():
        stem = root/s
        pdf = stem.with_suffix('.pdf')
        relative = str(pdf.relative_to(root))
        new = json.loads(stem.with_suffix('.geometry.json').read_text())
        old = baseline['geometry'][s]
        detail = '/detail/' in s
        assert len(new['panels']) == 16
        assert {(p['method'], p['layer']) for p in new['panels']} == {(m, l) for m in METHODS for l in LAYERS}
        scales = json.loads(stem.with_suffix('.scales.json').read_text())
        assert scales['shared_across_methods'] == METHODS
        for p in new['panels']:
            assert p['source_sha256'] == inventory[p['source']]['sha256']
            assert p['data_stride'] == [1, 1] and not p['pooling']
            assert p['draw_surface'] and not p['draw_height_columns'] and not p['draw_sidewalls']
            assert p['vertical_height_segments'] == p['sidewall_triangles'] == 0
            if p['method'] == 'unrotated':
                assert p['source'].startswith('activations/paired_local/unrotated/')
            expected_shape = [128, 512] if detail else inventory[p['source']]['shape']
            assert p['shape'] == expected_shape
            assert p['vertices_processed'] == expected_shape[0]*expected_shape[1]
            if not detail:
                assert not p['cropping']
                assert p['surface_triangles'] == 2*(expected_shape[0]-1)*(expected_shape[1]-1)
        if detail:
            assert new['scales'] == old['scales']
            assert new['height_axis_style'] == old['height_axis_style']
            assert new['palette_style'] == old['palette_style']
            assert not new['explanatory_footer_visible']
            assert abs(old['figure_inches'][1]-new['figure_inches'][1]-.6) < 1e-12
            for a, b in zip(old['panels'], new['panels']):
                for k in a:
                    if k != 'debug_points':
                        assert a[k] == b[k], (s, k)
                for x, y in zip(a['debug_points'], b['debug_points']):
                    assert all(x[k] == y[k] for k in ('channel', 'token', 'z'))
                    assert all(abs(v-w) < 1e-12 for v, w in zip(x['native_projected_xy'], y['native_projected_xy']))
        else:
            for layer in LAYERS:
                col = [p for p in new['panels'] if p['layer'] == layer]
                expected_limit = 1.03*max(p['maximum'] for p in col) or 1.
                assert all(p['z_limit'] == expected_limit for p in col)
                assert scales['z_limits_by_layer'][str(layer)] == expected_limit
        text = subprocess.run([sys.executable, str(qa/'pdf_text.py'), str(pdf), '--min-pt', '5', '--json'], capture_output=True, text=True)
        stem.with_suffix('.font-audit.json').write_text(text.stdout)
        collision = subprocess.run([sys.executable, str(qa/'collisions.py'), str(pdf), '--json-out', str(stem.with_suffix('.collision-audit.json'))], capture_output=True, text=True)
        index[relative] = {'pdf': relative, 'font_exit': text.returncode, 'collision_exit': collision.returncode,
                           'font_stderr': text.stderr, 'collision_summary': collision.stdout, 'collision_stderr': collision.stderr}
        with pymupdf.open(pdf) as doc:
            page = doc[0]
            content = page.get_text()
            assert 'Unrotated' in content and 'Sample 0' not in content
            assert 'square-root color mapping' not in content and 'Height ticks:' not in content
            assert content.count('0.00') >= 16
            spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0 for line in b['lines'] for s in line['spans']]
            min_pt = min(s['size'] for s in spans)
            paper_pt = min_pt*7.2/(page.rect.width/72)
            assert paper_pt >= (6.99 if detail else 5.0), (relative, paper_pt)
            review = out/('__'.join(pdf.relative_to(root/'figures').with_suffix('').parts)+'.png')
            page.get_pixmap(dpi=100, alpha=False).save(review)
            if detail:
                historical_review = root/'qa/detail_pdf_renders'/review.name
                page.get_pixmap(dpi=120, alpha=False).save(historical_review)
                reviews[relative] = {'pdf': relative, 'pdf_sha256': local.digest(pdf),
                    'review_png': str(historical_review.relative_to(root)), 'render_dpi': 120,
                    'page_points': list(page.rect), 'min_text_pt': min_pt,
                    'text_spans': len(spans), 'minimum_text_pt_at_insertion_width': paper_pt,
                    'intended_insertion_width_inches': 7.2, 'review_png_sha256': local.digest(historical_review)}
        with Image.open(stem.with_suffix('.png')) as im:
            assert all(abs(d-600) < .1 for d in im.info['dpi'])
        records.append({'pdf': relative, 'pdf_sha256': local.digest(pdf),
                        'panels': 16, 'minimum_text_pt_at_183mm': paper_pt,
                        'font_exit': text.returncode, 'collision_exit': collision.returncode,
                        'review_png': str(review.relative_to(root)),
                        'full_grid_vertices': sum(p['vertices_processed'] for p in new['panels'])})
        print('AUDITED', relative, 'font', text.returncode, 'collision', collision.returncode, flush=True)
    for name, expected in {**baseline['protected_figures'], **baseline['immutable_records']}.items():
        assert local.digest(root/name) == expected, name
    local.write_json(index_path, [index[k] for k in sorted(index)])
    local.write_json(reviews_path, [reviews[k] for k in sorted(reviews)])
    result = {'revision': REVISION, 'records': records, 'local_panels_data_unchanged': 48,
              'full_domain_panels': 48, 'other_figure_files_unchanged': len(baseline['protected_figures']),
              'experimental_records_unchanged': True, 'reference_pairing_passed': True,
              'visual_review': 'pending inspection of final PDF renders'}
    local.write_json(out/'audit.json', result)
    assert all(r['font_exit'] == r['collision_exit'] == 0 for r in records), 'Repair PDF layouts before publication'


def run(source, root, audit_only=False, views=VIEWS):
    source, root = Path(source), Path(root)
    prepare(root)
    local.write_json(root/'qa/clean_full_revision/reference_audit.json', local.reference_audit(str(source), str(root)))
    if not audit_only:
        for view in views:
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
                list(pool.map(render, [(str(source), str(root), target, view) for target in TARGETS]))
    audit(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source'); p.add_argument('root')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--view', choices=VIEWS, action='append')
    a = p.parse_args()
    run(a.source, a.root, a.audit_only, a.view or VIEWS)
