#!/usr/bin/env python3
"""Require deterministic frozen-data preparation and two complete figure renders."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    tables=[HERE/name for name in ['fig2_revised_data.csv','fig3_revised_data.csv','fig3_moe_binned_summary.csv','fig3_energy_all_token_context.csv','fig2_fig3_source_metadata.json']]
    old={p.name:digest(p) for p in tables}
    subprocess.run([sys.executable,str(HERE/'build_fig3_moe_summary.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    assert old=={p.name:digest(p) for p in tables},'Frozen preparation changed points, statistics or provenance'
    def render():
        for name in ['make_fig2_revised.py','make_fig3_revised.py','write_fig2_fig3_captions.py']:
            subprocess.run([sys.executable,str(HERE/name)],cwd=ROOT,check=True)
        bases=[HERE/name for name in ['fig2_revised','fig3_revised','fig2_caption','fig3_caption','appendix/fig3_energy_all_token_context']]
        bases += [p.with_suffix('') for p in sorted((HERE/'panels/fig2_fig3').glob('*.pdf'))]
        return {str(p.relative_to(ROOT)):digest(p) for base in bases for ext in ['.pdf','.svg','.png'] for p in [base.with_suffix(ext)]}
    a=render();b=render();assert a==b,'Repeated rendering changed artifact bytes'
    report={'status':'PASS','preparation_byte_identical':True,'full_render_repeats':2,'artifact_count':len(b),'sha256':b}
    (HERE/'qa/fig2_fig3/reproducibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: source rebuild and two byte-identical renders of {len(b)} Figure 2/3 artifacts.')

if __name__=='__main__':main()
