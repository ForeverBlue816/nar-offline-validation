"""Review artifacts for the current local revision, using final PDF page renders."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pymupdf
from .detail_plot import digest, write_json


def pdf_reviews(root):
    root = Path(root)
    out = root/'qa/detail_pdf_renders'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf in sorted((root/'figures').rglob('*.pdf')):
        if 'detail' not in pdf.parts or '.collision-audit' in pdf.name:
            continue
        rel = pdf.relative_to(root/'figures')
        name = '__'.join(rel.with_suffix('').parts)
        with pymupdf.open(pdf) as doc:
            assert len(doc) == 1
            page = doc[0]
            spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0 for line in b['lines'] for s in line['spans']]
            geometry = json.loads(pdf.with_suffix('.geometry.json').read_text())
            intended = geometry['intended_insertion_width_inches']
            paper_font = min(s['size'] for s in spans) * intended / (page.rect.width/72)
            assert spans and paper_font >= 6.99, (pdf, paper_font)
            pix = page.get_pixmap(dpi=120, alpha=False)
            review = out/(name+'.png')
            pix.save(review)
            record = {'pdf':str(pdf.relative_to(root)), 'pdf_sha256':digest(pdf),
                      'review_png':str(review.relative_to(root)), 'render_dpi':120,
                      'page_points':list(page.rect), 'min_text_pt':min(s['size'] for s in spans),
                      'text_spans':len(spans), 'minimum_text_pt_at_insertion_width':paper_font,
                      'intended_insertion_width_inches':intended, 'review_png_sha256':digest(review)}
            records.append(record)
    write_json(root/'qa/detail_pdf_render_index.json', records)
    print('PDF RE-RENDERS', len(records), flush=True)


def provenance(root, slurm_job_id=None):
    root = Path(root)
    geometries = [json.loads(p.read_text()) for p in (root/'figures').rglob('*.geometry.json') if 'detail' in p.parts]
    commits = sorted({g['git_commit'] for g in geometries})
    source_hashes = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    p = root/'render_provenance.json'
    original = json.loads(p.read_text())
    revision = {
        'name':'local-upper-surface-v1', 'rendering_git_commits':commits,
        'source_hashes':source_hashes, 'scope':'all eight detail combinations; original full-domain overviews and exact ECDFs preserved',
        'supersedes':['identical overview/detail','mandatory height columns','mandatory closed sidewalls'],
        'slurm_job_id':slurm_job_id, 'command':'python -m nar.activation_viz.full_batch $RAW_RUN $RUN --workers 8',
        'text_and_axes':'vector', 'surface_dpi':600,
        'provenance_note':'older fields describe archived full-resolution assets; current detail geometry/config are authoritative'}
    historical = original.get('historical_full_resolution', original)
    write_json(p, {**revision, 'git_commit': commits[0] if len(commits)==1 else commits,
        'surface': 'native Matplotlib 128 x 512 measured upper surface',
        'draw_surface': True, 'draw_height_columns': False, 'draw_sidewalls': False,
        'data_stride': [1,1], 'pooling': False, 'cropping': True,
        'window': {'tokens':[0,128], 'channels':[0,512]},
        'compatibility': 'overview and detail are independent; overview preserves full-domain context',
        'historical_full_resolution': historical})
    config = json.loads((root/'detail_render_config.json').read_text())
    config.update(rendering_git_commits=commits, source_hashes=source_hashes)
    write_json(root/'detail_render_config.json',config)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--slurm-job-id');a=p.parse_args()
    pdf_reviews(a.root);provenance(a.root,a.slurm_job_id)
