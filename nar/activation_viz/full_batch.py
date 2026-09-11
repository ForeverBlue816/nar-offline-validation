"""Batch full-resolution rendering, exact ECDFs and compatibility exports."""
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path


def draw(args):
    source, destination, selector = args
    from .full_plot import run
    run(source, destination, selector)
    # Both historical URLs now contain identical complete data. No cropped
    # or pooled scientific figure remains at the old overview/detail paths.
    directory = Path(destination)/'figures'/selector
    shutil.copytree(directory/'overview', directory/'detail', dirs_exist_ok=True)
    return selector


def run(source, destination, workers=8):
    import matplotlib, numpy, numba
    from matplotlib import font_manager
    from .plot import style
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.tex'):
            if p.name == 'render_provenance.json':
                shutil.copy2(p, destination/'previous_render_provenance.json')
            else: shutil.copy2(p, destination/p.name)
    shutil.copytree(source/'qa', destination/'qa', dirs_exist_ok=True)
    for name in ('full_resolution_contract.md', 'qa/full_surface_checks.json'):
        shutil.copy2(Path('outputs/qwen_activation_viz/qwen3_8b_seed42')/name, destination/name)
    style()
    font = Path(font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False))
    provenance = {'command': sys.argv, 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'matplotlib': matplotlib.__version__,
        'numpy': numpy.__version__, 'numba': numba.__version__, 'workers': workers,
        'source_hashes': {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.rglob('*.py')},
        'font': {'file': font.name, 'sha256': hashlib.sha256(font.read_bytes()).hexdigest()},
        'dpi': 600, 'formats': ['png','pdf','svg'], 'surface': 'all measured vertices and grid cells, depth-tested',
        'data_stride': [1,1], 'pooling': False, 'cropping': False, 'sidewall_base_z': 0,
        'text_and_axes': 'vector and editable', 'compatibility': 'overview and detail URLs both show identical complete data',
        'ecdf': 'exact unique values and multiplicities; no quantile thinning or path simplification'}
    (destination/'render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    selectors = [f'{mode}/{site}/{quantity}' for mode in ('paired_local','end_to_end')
                 for site in ('down_proj','q_proj') for quantity in ('raw','residual')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for selector in pool.map(draw, [(source,destination,s) for s in selectors]):
            print('FINISHED FULL GRID', selector, flush=True)
    from .full_ecdf import run as ecdf_run
    ecdf_run(source, destination)
    from .report import run as report_run
    report_run(destination)
    from .render_batch import audit
    audit(destination)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('destination')
    p.add_argument('--workers',type=int,default=8); a=p.parse_args(); run(a.source,a.destination,a.workers)
