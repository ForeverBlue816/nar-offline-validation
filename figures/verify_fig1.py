#!/usr/bin/env python3
"""Verify frozen Figure 1 evidence and the final vector/raster delivery bundle."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pymupdf

HERE=Path(__file__).resolve().parent

def main():
    arrays=dict(np.load(HERE/'fig1_source_arrays.npz'))
    meta=json.loads((HERE/'fig1_metadata.json').read_text())
    baseline=json.loads((HERE/'qa/fig1.data-baseline.json').read_text())
    for key,value in arrays.items():
        assert hashlib.sha256(value.tobytes()).hexdigest()==baseline['array_sha256'][key],key
    for name,expected in baseline['source_file_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    for key,value in baseline['scientific_metadata'].items():
        assert meta[key]==value,key
    for method,key in [('hadamard','hadamard_range'),('prismquant_kmax','nar_kmax_range')]:
        values=arrays[key];stats=meta['range_statistics'][method]
        np.testing.assert_allclose([stats['mean'],stats['median'],stats['percentile_95'],stats['maximum']],
            [values.mean(dtype=np.float64),np.median(values),np.quantile(values,.95),values.max()],rtol=1e-12)
        assert values.shape==(512,64) and values.min()>=0 and values.max()<=10
    counts=arrays['peak_density_window_counts'];window=meta['peak_density_selection']
    assert int(counts.argmax())==window['start_channel'] and int(counts.max())==window['qualifying_channel_count']
    medians=arrays['all_channel_median_magnitudes'];prefix=np.r_[0,np.cumsum(medians>1)]
    np.testing.assert_array_equal(counts,prefix[2048:]-prefix[:-2048])
    trace_csv=pd.read_csv(HERE/'fig1_ranges.csv')
    for method in ['raw','hadamard','nar_kmax']:
        values=arrays['trace_'+method]
        np.testing.assert_allclose(trace_csv[trace_csv.method.eq(method)].signed_value,values,rtol=1e-12)
        assert float(np.ptp(values))==meta['trace_ranges'][method]
        assert float(np.float16(values.min()))==meta['trace_zero_points'][method]
    row=meta['hero']['token_position']-meta['token_window']['position_start'];group=meta['prismquant_receiving_group']
    for method in ['hadamard','nar_kmax']:
        np.testing.assert_allclose(np.ptp(arrays['trace_'+method]),arrays[method+'_range'][row,group],rtol=1e-6)
    panels=meta['rendering_contract']['scientific_panels']
    for key in ['z_limits','color_normalization','view','size_inches']:
        assert panels['c'][key]==panels['d'][key],key
    for panel in panels.values():
        values=arrays[panel['array']]
        assert panel['segments']==values.shape[1]*(values.shape[0]-1)
    exports={}
    for stem in ['fig1','fig1_caption']+['fig1'+x for x in 'abcdefg']:
        with pymupdf.open(HERE/(stem+'.pdf')) as doc:
            page=doc[0];spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
            assert spans and min(s['size'] for s in spans)>=5.0
            assert all('TimesNewRomanPS-BoldMT' in s['font'] and s['flags'] & 16 for s in spans),(stem,'font substitution or nonbold text')
            for font in page.get_fonts(full=True):
                assert 'TimesNewRomanPS-BoldMT' in font[3] and doc.extract_font(font[0])[3],(stem,'font not embedded')
            if stem=='fig1_caption':
                assert ''.join(''.join(s['text'] for s in spans).split())==''.join((HERE/'fig1_caption.txt').read_text().split())
            assert all(page.rect.contains(pymupdf.Rect(s['bbox'])) for s in spans),(stem,'page clipping')
            assert '<text' in (HERE/(stem+'.svg')).read_text()
            exports[stem]={'size_inches':[page.rect.width/72,page.rect.height/72],
                           'minimum_font_pt':min(s['size'] for s in spans),'vector_text_spans':len(spans),'font':'TimesNewRomanPS-BoldMT','all_text_bold':True,'fonts_embedded':True}
    report={'verdict':'PASS','arrays_unchanged':list(arrays),'scientific_metadata_unchanged':list(baseline['scientific_metadata']),
            'all_samples_preserved':True,'matched_range_panels':True,'exports':exports}
    (HERE/'qa/fig1.integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
