"""Render local detail independently of full-domain overview and exact ECDFs."""
import argparse
import concurrent.futures
import shutil
import subprocess
from pathlib import Path


def draw(args):
    source, destination, selector, parts, overview = args
    from .full_plot import run
    if overview:
        run(source, destination, selector, parts=parts, view='overview')
    run(source, destination, selector, parts=parts, view='detail')
    return selector


def run(source, destination, workers=4, parts='all', overview=False):
    from .detail_plot import CONFIG, reference_audit, write_json
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Initialize missing metadata only; never replace an existing experiment report.
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.tex') and not (destination/p.name).exists():
            shutil.copy2(p, destination/p.name)
    config = {**CONFIG, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write_json(destination/'detail_render_config.json', config)
    write_json(destination/'qa/detail_reference_audit.json', reference_audit(str(source), str(destination)))
    selectors = [f'{mode}/{site}/{quantity}' for mode in ('paired_local', 'end_to_end')
                 for site in ('down_proj', 'q_proj') for quantity in ('raw', 'residual')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for selector in pool.map(draw, [(source, destination, s, parts, overview) for s in selectors]):
            print('FINISHED LOCAL DETAIL', selector, flush=True)
    from .render_batch import audit
    audit(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--parts', choices=['all', 'matrix', 'zoom'], default='all')
    p.add_argument('--redraw-overview', action='store_true', help='also rebuild full-domain overviews')
    p.add_argument('--redraw-surfaces-only', action='store_true', help='legacy alias; metrics/ECDFs are always left untouched')
    a = p.parse_args()
    run(a.source, a.destination, a.workers, a.parts, a.redraw_overview)
