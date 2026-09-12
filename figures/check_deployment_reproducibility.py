#!/usr/bin/env python3
"""Rebuild frozen source data, then require two identical complete render bundles."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    csvs=[HERE/'deployment_efficiency_data.csv',HERE/'fig4_revised_data.csv']
    before={p.name:digest(p) for p in csvs}
    subprocess.run([sys.executable,str(HERE/'build_e28_figure_data.py')],check=True,cwd=ROOT)
    assert before=={p.name:digest(p) for p in csvs},'Data rebuild changed a plotted value or source record'
    def render():
        for name in ['make_fig4_revised.py','make_deployment_efficiency.py','make_deployment_appendix.py']:
            subprocess.run([sys.executable,str(HERE/name),'--reuse-data'],check=True,cwd=ROOT)
        files=[HERE/(base+ext) for base in ['fig4_revised','fig_deployment_efficiency'] for ext in ['.pdf','.svg','.png']]
        files += [p for folder in [HERE/'appendix',HERE/'panels'] for p in folder.iterdir() if p.suffix in ['.pdf','.svg','.png']]
        return {str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}
    first=render();second=render();assert first==second,'Repeated render changed artifact bytes'
    report=dict(status='PASS',data_rebuild_identical=True,full_render_repeats=2,artifact_count=len(first),sha256=second)
    (HERE/'qa/deployment/reproducibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: frozen CSV rebuild and two byte-identical renders of {len(first)} PDF/SVG/PNG artifacts.')

if __name__=='__main__':main()
