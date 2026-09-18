"""Hash the read-only historical factors and calibration spectra consumed by E33."""
from pathlib import Path
from collections import defaultdict
import hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
from nar.e33_rangelaw import grid


def main():
    paths=defaultdict(set)
    for m in ['llama32_3b','llama31_8b','qwen3_4b_base']:
        if not m.startswith('qwen'):
            paths[a.WORK/'results'/m/'e11_calibration_eigenspace.csv'].add('E20 energy weights and E11 fallback fractions')
        for site,layer,n in a.keys(m):
            for name,g,mm in grid(m):
                if name.startswith('E20'):
                    p=a.WORK/'activations'/m/'e11_calibration/factors/nar_b64_kmax'/f'{site}_layer_{layer:02d}.pt'
                    paths[p].add('E20 recovered eigendirections and permutation energy')
                elif name.startswith('E22'):
                    rank=name.split('_')[1];p=a.WORK/'activations'/m/'e18v2_factors'/f'nar_{rank}'/f'{site}_layer_{layer:02d}.pt';paths[p].add(name)
                else:
                    label=name[4:];folder=a.act.factor_dir(a.WORK,m) if label=='g128_kmax' else a.WORK/'activations'/m/'e11_calibration/factors'/('nar_b'+label[1:])
                    paths[folder/f'{site}_layer_{layer:02d}.pt'].add(name)
                    if m=='llama32_3b' and label=='g128_kmax':
                        src='q_input' if site=='qkv' else 'down_input'
                        paths[a.WORK/'activations'/m/'wide_cal_a/analysis/eigenspaces'/f'{src}_layer_{layer:02d}.pt'].add('frozen E1c eigenvalues/trace for default factor')
    records=[]
    for i,(p,roles) in enumerate(sorted(paths.items())):
        before=p.stat();digest=hashlib.sha256()
        with p.open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Historical asset changed while hashing'
        records.append(dict(file=str(p.relative_to(a.WORK)),bytes=before.st_size,sha256=digest.hexdigest(),roles='; '.join(sorted(roles))))
        if (i+1)%100==0:print('Hashed',i+1,'historical read-only source assets',flush=True)
    a.write(a.REPO/'results/e33_source_assets.csv',records)
    print('Complete:',len(records),'historical source assets',flush=True)
if __name__=='__main__':main()
