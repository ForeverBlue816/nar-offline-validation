"""Rerender and audit only the three user-selected raw activation matrices.

The existing quantitative grid and all source data, camera and scales are fixed.
Height labels, the zero-plane cue and pale floor edges are preserved.
The latest follow-up restores the original blue-orange colormap only. There are
no uncertainty intervals because the surfaces describe one fixed sample.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
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
    out = root/'qa/blue_orange_revision'
    out.mkdir(exist_ok=True)
    baseline = out/'baseline.json'
    if not baseline.exists():
        manifest = json.loads((root/'publication_manifest.json').read_text())
        plot.write_json(baseline, {
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'geometry': {s: json.loads((root/'figures'/s/'detail/matrix.geometry.json').read_text()) for s in TARGETS},
            'protected_figures': {r['file']: r['sha256'] for r in manifest['files']
                if r['file'].startswith('figures/') and not any(r['file'].startswith(f'figures/{s}/detail/matrix.') for s in TARGETS)},
        })


def audit(root):
    out = root/'qa/blue_orange_revision'
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
        assert new['scales']['cmap'] == plot.CMAP.name
        assert {k:v for k,v in old['scales'].items() if k != 'cmap'} == {k:v for k,v in new['scales'].items() if k != 'cmap'}
        assert old['figure_inches'] == new['figure_inches']
        assert len(old['panels']) == len(new['panels']) == 16
        panel_records = []
        for a, b in zip(old['panels'], new['panels']):
            for field in ('source', 'source_sha256', 'source_tensor_sha256', 'local_array_sha256',
                          'shape', 'z_limit', 'norm', 'camera', 'surface_cells', 'surface_rcount',
                          'surface_ccount', 'draw_height_columns', 'draw_sidewalls', 'zero_padding'):
                if field == 'norm':
                    assert b[field]['cmap'] == plot.CMAP.name
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
            assert paper_font >= 6.99
            content = page.get_text()
            assert 'pale floor: z=0' in content
            assert content.count('0.00') >= 16
            review = root/'qa/detail_pdf_renders'/('__'.join(pdf.relative_to(root/'figures').with_suffix('').parts)+'.png')
            page.get_pixmap(dpi=120, alpha=False).save(review)
            render_index[relative] = {'pdf': relative, 'pdf_sha256': plot.digest(pdf),
                'review_png': str(review.relative_to(root)), 'render_dpi': 120,
                'page_points': list(page.rect), 'min_text_pt': min(s['size'] for s in spans),
                'text_spans': len(spans), 'minimum_text_pt_at_insertion_width': paper_font,
                'intended_insertion_width_inches': 7.2, 'review_png_sha256': plot.digest(review)}
        records.append({'pdf': relative, 'pdf_sha256': plot.digest(pdf),
            'panels': panel_records, 'minimum_text_pt_at_183mm': paper_font,
            'palette_before': old['scales']['cmap'], 'palette_after': new['scales']['cmap'],
            'font_exit': text.returncode, 'collision_exit': collision.returncode})
    for file, expected in baseline['protected_figures'].items():
        assert plot.digest(root/file) == expected, file
    from .detail_checks import snapshot
    assert snapshot(root) == json.loads((root/'qa/detail_immutable_before.json').read_text())
    plot.write_json(audit_path, [audit_index[k] for k in sorted(audit_index)])
    plot.write_json(render_path, [render_index[k] for k in sorted(render_index)])
    result = {'scope': TARGETS, 'records': records, 'panels_unchanged': 48,
              'other_figure_files_unchanged': len(baseline['protected_figures']),
              'experimental_artifacts_unchanged': True,
              'palette_only_change': True, 'palette': plot.PALETTE_STYLE,
              'visual_review': 'pending inspection of the final PDF renders'}
    plot.write_json(out/'audit.json', result)
    failures = [r for r in records if r['font_exit'] or r['collision_exit']]
    print('BLUE-ORANGE AUDIT:', len(records), 'PDFs;', len(failures), 'blocking layouts', flush=True)
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
