#!/usr/bin/env python3
"""Audit Figure 4 against source rows and raw sequence losses, independently."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'

def close(a,b): np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
def source(path,line): return pd.read_csv(ROOT/path).iloc[int(line)-2]
def mean_ppl(samples):
    return samples.groupby('seed').apply(lambda s: np.exp(np.average(s.nll,weights=s.tokens_scored)),include_groups=False).mean()

def main():
    report={'panels':{},'measurements':{}}
    for stem,model in [('fig4a','llama32_3b'),('fig4a_8b','llama31_8b')]:
        table=pd.read_csv(FIG/f'{stem}.csv');raw=pd.read_csv(ROOT/'results'/model/'e20_per_sequence.csv')
        assert len(table)==12
        for row in table.itertuples():
            original=source(row.source_file,row.source_csv_line)
            assert original.row==row.source_row
            close(row.ppl,original.mean_ppl)
            per=raw[raw.row.eq(row.source_row)]
            close(row.ppl,mean_ppl(per))
            assert per.groupby('seed').chunk.nunique().eq(64).all()
            if row.kind=='point':
                close(row.effective_bits,4+16*(row.m+1)/row.g)
                assert per.seed.nunique()==row.seeds==3
                assert json.loads(row.seed_ids)==sorted(per.seed.unique().tolist())
        pts=table[table.kind.eq('point')]
        for g in [64,128,256]:
            a=pts[pts.g.eq(g)&pts.m.eq(1)]
            assert a.loc[a.method.eq('nar'),'ppl'].item()<a.loc[a.method.eq('hadamard'),'ppl'].item()
        equal=pts[np.isclose(pts.effective_bits,4.25)]
        best=equal.loc[equal.ppl.idxmin()]
        assert best.method=='nar' and best.g==128 and best.m==1
        report['panels'][stem]={'verified_source_rows':len(table),'all_three_readings_supported':True}
    table=pd.read_csv(FIG/'fig4b.csv',dtype={'k_category':str})
    assert len(table)==21
    for row in table.itertuples():
        original=source(row.source_file,row.source_csv_line)
        if row.kind=='recovery':
            assert original.method==row.source_row
            qwen=row.model=='qwen3_8b_base'
            column='ppl' if qwen else 'mean_ppl'
            close(row.ppl_k,original[column])
            close(row.ppl_bf16,source(row.bf16_source_file,row.bf16_source_csv_line)[column])
            close(row.ppl_hadamard,source(row.hadamard_source_file,row.hadamard_source_csv_line)[column])
            close(row.recovery,(row.ppl_hadamard-row.ppl_k)/(row.ppl_hadamard-row.ppl_bf16))
            perfile='e18v2_per_sequence.csv' if qwen else ('e11_k64_per_sequence.csv' if row.k_category=='64' else 'e11_per_sequence.csv')
            raw=pd.read_csv(ROOT/'results'/row.model/perfile);per=raw[raw.method.eq(row.source_row)]
            assert len(per)==row.chunks*row.seeds and per.groupby('seed').sequence.nunique().eq(row.chunks).all()
            close(row.ppl_k,mean_ppl(per))
            reported_seeds=json.loads(row.seed_ids)
            assert ([reported_seeds] if isinstance(reported_seeds,int) else reported_seeds)==sorted(per.seed.unique().tolist())
            kmax=json.loads(row.site_kmax);actual=json.loads(row.site_k_actual)
            for site,maximum in kmax.items():
                assert actual[site]==(maximum if row.k_category=='max' else min(int(row.k),maximum))
        else:
            column='nar' if row.kind=='kernel_share' else 'hadamard'
            assert int(original.k)==int(row.k)
            close(row.share_percent,100*original[column+'_share_of_layer'])
            close(row.share_percent,100*row.kernel_ms/(row.decoder_layer_ms+row.kernel_ms))
    for model,part in table[table.kind.eq('recovery')].groupby('model'):
        assert list(part.k_category)==['8','16','32','64','max']
        if model=='qwen3_8b_base': continue
        root=ROOT/'results'/model
        basis=pd.read_csv(root/'e11_k64_basis_audit.csv');factors=pd.read_csv(root/'e11_k64_factor_audit.csv')
        assert basis.orthogonality_error.max()<1e-4 and basis.reconstructed_k32_reflector_error.max()<1e-4
        assert factors.anchor_error.max()<1e-4
        done=json.loads((root/'E11_K64_DONE.json').read_text())
        raw=pd.read_csv(root/'e11_k64_per_sequence.csv')
        assert not raw.duplicated(['seed','sequence']).any()
        assert sorted(raw.seed.unique())==[20260902,20260903,20260904]
        close(done['summary']['mean_ppl'],mean_ppl(raw))
        report['measurements'][model]={'sequences':len(raw),'mean_ppl':mean_ppl(raw),
            'max_basis_reconstruction_error':float(basis.reconstructed_k32_reflector_error.max()),
            'evaluation_token_sha256':done['evaluation_token_sha256']}
    report['panels']['fig4b']={'recovery_points':15,'kernel_points':4,'hadamard_references':2,'missing_k':[]}
    for stem,mask in [('fig4b1',table.kind.eq('recovery')),('fig4b2',~table.kind.eq('recovery'))]:
        split=pd.read_csv(FIG/f'{stem}.csv',dtype={'k_category':str})
        pd.testing.assert_frame_equal(split,table[mask].reset_index(drop=True),check_dtype=False)
        report['panels'][stem]={'verified_source_rows':len(split)}
    layout=json.loads((FIG/'qa/fig4b.alignment.json').read_text())['layout']['panels']
    top,bottom=[p['bbox_pt'] for p in layout]
    close((top[3]-top[1])/(bottom[3]-bottom[1]),62/38)
    close([top[0],top[2]],[bottom[0],bottom[2]])
    for stem in ['fig4a','fig4a_8b','fig4b','fig4b1','fig4b2']:
        height=4.2*({'fig4b1':.62,'fig4b2':.38}.get(stem,1))
        doc=pymupdf.open(FIG/f'{stem}.pdf');page=doc[0]
        np.testing.assert_allclose([page.rect.width,page.rect.height],[2.7*72,height*72],rtol=0,atol=1e-4)
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        assert all(5.99<=s['size']<=7.01 and 'Serif' in s['font'] for s in spans)
        for span in spans: assert page.rect.contains(pymupdf.Rect(span['bbox']))
        with Image.open(FIG/f'{stem}.png') as png: assert png.size==(810,int(height*300))
        svg=(FIG/f'{stem}.svg').read_text();assert '<text' in svg
        report['panels'][stem]['export_dimensions_inches']=[2.7,height]
    (FIG/'qa'/'fig4.integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
