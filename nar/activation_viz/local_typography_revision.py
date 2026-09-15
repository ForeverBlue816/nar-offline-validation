"""Restore Viridis, enlarge Times New Roman Bold, and remove the reference rule.

Only three local raw matrices are rendered. Existing measured data, camera,
shared scales and complete meshes remain fixed; uncertainty is not applicable
for these descriptive fixed-sample surfaces.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf

from . import detail_plot as plot


TARGETS = [
    'end_to_end/q_proj/raw',
    'end_to_end/down_proj/raw',
    'paired_local/q_proj/raw',
]


def render(args):
    source, root, selection = args
    plot.run(source, root, selection, parts='primary')


def prepare(root):
    out = root/'qa/local_typography_revision'
    out.mkdir(exist_ok=True)
    baseline = out/'baseline.json'
    if not baseline.exists():
        manifest = json.loads((root/'publication_manifest.json').read_text())
        immutable_names = ['run_manifest.json', 'activation_inventory.json', 'inputs.json',
                           'metrics_per_sample.csv', 'metrics_summary.csv', 'validation_report.json',
                           'numerical_edge_checks.json', 'measured_summary.md', 'capture_site_report.md']
        plot.write_json(baseline, {
            'immutable_records': {name: plot.digest(root/name) for name in immutable_names},
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'geometry': {s: json.loads((root/'figures'/s/'detail/matrix.geometry.json').read_text()) for s in TARGETS},
            'protected_figures': {r['file']: r['sha256'] for r in manifest['files']
                if r['file'].startswith('figures/') and not any(r['file'].startswith(f'figures/{s}/detail/matrix.') for s in TARGETS)},
        })


def audit(root):
    out = root/'qa/local_typography_revision'
    baseline = json.loads((out/'baseline.json').read_text())
    audit_path = root/'qa/rendered_audit_index.json'
    audit_index = {r['pdf']: r for r in json.loads(audit_path.read_text())}
    render_path = root/'qa/detail_pdf_render_index.json'
    render_index = {r['pdf']: r for r in json.loads(render_path.read_text())}
    qa = Path(__file__).parent/'qa'
    records = []
    for selection in TARGETS:
        stem = root/'figures'/selection/'detail/matrix'
        pdf = stem.with_suffix('.pdf')
        relative = str(pdf.relative_to(root))
        old = baseline['geometry'][selection]
        new = json.loads(stem.with_suffix('.geometry.json').read_text())
        assert new['height_axis_style'] == plot.HEIGHT_AXIS_STYLE
        assert new['palette_style'] == plot.PALETTE_STYLE
        assert new['scales']['cmap'] == 'viridis'
        assert {k:v for k,v in old['scales'].items() if k != 'cmap'} == {k:v for k,v in new['scales'].items() if k != 'cmap'}
        assert new['reference_separator_count'] == 0
        assert new['typography_style']['font_family'] == 'Times New Roman'
        assert new['typography_style']['font_weight'] == 'bold'
        assert not new['explanatory_footer_visible']
        assert old['figure_inches'] == new['figure_inches']
        assert len(old['panels']) == len(new['panels']) == 16
        panel_records = []
        for a, b in zip(old['panels'], new['panels']):
            for field in ('source', 'source_sha256', 'source_tensor_sha256', 'local_array_sha256',
                          'shape', 'z_limit', 'norm', 'camera', 'surface_cells', 'surface_rcount',
                          'surface_ccount', 'draw_height_columns', 'draw_sidewalls', 'zero_padding'):
                if field == 'norm':
                    assert b[field]['cmap'] == 'viridis'
                    assert {k:v for k,v in a[field].items() if k != 'cmap'} == {k:v for k,v in b[field].items() if k != 'cmap'}
                else:
                    assert a[field] == b[field], (selection, field)
            for x, y in zip(a['debug_points'], b['debug_points']):
                assert all(x[k] == y[k] for k in ('channel', 'token', 'z'))
                assert all(abs(v-w) < 1e-12 for v, w in zip(x['native_projected_xy'], y['native_projected_xy']))
            panel_records.append({'method': b['method'], 'layer': b['layer'],
                                  'local_array_sha256': b['local_array_sha256'], 'unchanged': True})
        text = subprocess.run([sys.executable, str(qa/'pdf_text.py'), str(pdf), '--min-pt', '5', '--json'], capture_output=True, text=True)
        stem.with_suffix('.font-audit.json').write_text(text.stdout)
        collision = subprocess.run([sys.executable, str(qa/'collisions.py'), str(pdf), '--json-out', str(stem.with_suffix('.collision-audit.json'))], capture_output=True, text=True)
        entry = {'pdf': relative, 'font_exit': text.returncode, 'collision_exit': collision.returncode,
                 'font_stderr': text.stderr, 'collision_summary': collision.stdout, 'collision_stderr': collision.stderr}
        audit_index[relative] = entry
        with pymupdf.open(pdf) as doc:
            page = doc[0]
            spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0 for line in b['lines'] for s in line['spans']]
            paper_font = min(s['size'] for s in spans)*7.2/(page.rect.width/72)
            assert min(s['size'] for s in spans) >= 14
            assert paper_font >= 8.36
            assert all('TimesNewRoman' in s['font'] and 'Bold' in s['font'] and s['flags'] & 16 for s in spans)
            for font in page.get_fonts():
                assert 'TimesNewRoman' in font[3] and 'Bold' in font[3], font
                assert len(doc.extract_font(font[0])[3]) > 0
            # The removed rule ran from x=0.3 to x=68.4 pt in the left label gutter.
            gutter_rules = [item for drawing in page.get_drawings() for item in drawing['items']
                            if item[0] == 'l' and abs(item[1].y-item[2].y) < .05
                            and max(item[1].x,item[2].x) < 70 and abs(item[1].x-item[2].x) > 40]
            assert not gutter_rules, gutter_rules
            content = page.get_text()
            assert 'Sample 0' not in content and 'Height ticks:' not in content
            assert content.count('0.00') >= 16
            review = root/'qa/detail_pdf_renders'/('__'.join(pdf.relative_to(root/'figures').with_suffix('').parts)+'.png')
            page.get_pixmap(dpi=120, alpha=False).save(review)
            render_index[relative] = {'pdf': relative, 'pdf_sha256': plot.digest(pdf),
                'review_png': str(review.relative_to(root)), 'render_dpi': 120,
                'page_points': list(page.rect), 'min_text_pt': min(s['size'] for s in spans),
                'text_spans': len(spans), 'minimum_text_pt_at_insertion_width': paper_font,
                'intended_insertion_width_inches': 7.2, 'review_png_sha256': plot.digest(review)}
        svg = ET.parse(stem.with_suffix('.svg')).getroot()
        texts = svg.findall('.//{http://www.w3.org/2000/svg}text')
        assert texts and all('Times New Roman' in t.get('style', '') and
                             ('700' in t.get('style', '') or 'bold' in t.get('style', '')) for t in texts)
        records.append({'font_family': 'Times New Roman', 'font_weight': 'bold',
            'embedded_font_verified': True, 'reference_separator_absent': True, 'pdf': relative, 'pdf_sha256': plot.digest(pdf),
            'panels': panel_records, 'minimum_text_pt_at_183mm': paper_font,
            'palette_before': old['scales']['cmap'], 'palette_after': new['scales']['cmap'],
            'font_exit': text.returncode, 'collision_exit': collision.returncode})
    for file, expected in baseline['protected_figures'].items():
        assert plot.digest(root/file) == expected, file
    for file, expected in baseline['immutable_records'].items():
        assert plot.digest(root/file) == expected, file
    plot.write_json(audit_path, [audit_index[k] for k in sorted(audit_index)])
    plot.write_json(render_path, [render_index[k] for k in sorted(render_index)])
    result = {'scope': TARGETS, 'records': records, 'panels_unchanged': 48,
              'other_figure_files_unchanged': len(baseline['protected_figures']),
              'experimental_artifacts_unchanged': True,
              'revision': 'priority-times-bold-viridis-v1', 'palette': plot.PALETTE_STYLE,
              'typography': plot.TYPOGRAPHY_STYLE, 'reference_separator_absent': True,
              'visual_review': 'pending inspection of the final PDF renders'}
    plot.write_json(out/'audit.json', result)
    failures = [r for r in records if r['font_exit'] or r['collision_exit']]
    print('LOCAL TYPOGRAPHY AUDIT:', len(records), 'PDFs;', len(failures), 'blocking layouts', flush=True)
    if failures:
        raise RuntimeError('Repair PDF layout before delivery')


def run(source, root, audit_only=False):
    root = Path(root)
    prepare(root)
    if not audit_only:
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(render, [(source, str(root), target) for target in TARGETS]))
    audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('root')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    run(args.source, args.root, args.audit_only)
