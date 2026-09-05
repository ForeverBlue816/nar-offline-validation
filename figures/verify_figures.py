#!/usr/bin/env python3
"""Independent numerical/export regression checks for the reported figure bugs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pymupdf

HERE=Path(__file__).resolve().parent

def main():
    m=json.loads((HERE/'fig1_metadata.json').read_text())
    arrays=np.load(HERE/'fig1_source_arrays.npz')
    assert m['channel_windows']['raw']==m['channel_windows']['hadamard_and_prismquant']
    assert m["row1_shared_z_limits"] == [0.0, 40.0]
    assert m["row1_z_limits"] == {"raw": [0.0, 40.0], "hadamard": [0.0, 40.0]}
    assert m["row2_shared_z_limits"] == [0.0, 8.92]
    assert m["rendering_contract"]["row1"]["normalization"] == [0.0, 40.0]
    assert m["rendering_contract"]["row2"]["normalization"] == [0.0, 8.92]
    assert m["rendering_contract"]["row2"]["view"] == {"elevation": 18, "azimuth": -62}
    assert m["rendering_contract"]["dense_grid"] == {
        "channel_interval": 250,
        "token_interval": 100,
        "group_interval": 10,
        "row1_z_interval": 5,
        "row2_z_interval": 1,
    }
    forbidden = ("local height scale", "shared height scale", "mean range", "Hadamard", "PrismQuant")
    for letter in "abcd":
        with pymupdf.open(HERE / f"fig1{letter}.pdf") as doc:
            text = doc[0].get_text()
            assert not any(token in text for token in forbidden), (letter, text)
            assert "token" in text
            assert ("channel" if letter in "ab" else "group") in text
        from PIL import Image
        with Image.open(HERE / f"fig1{letter}.png") as image:
            assert image.mode == "RGBA", (letter, image.mode)
            assert image.size == (960, 735), (letter, image.size)
            assert image.getpixel((0, 0))[3] == 0, letter
    for letter, trace_method, statistics_method, key in (
        ("c", "hadamard", "hadamard", "hadamard_range"),
        ("d", "nar_kmax", "prismquant_kmax", "nar_kmax_range"),
    ):
        values = arrays[key]
        assert values.shape == (512, 64)
        assert np.isfinite(values).all() and values.min() >= 0
        assert float(values.max()) <= m["row2_shared_z_limits"][1]
        expected = {
            "median": float(np.median(values)),
            "mean": float(values.mean(dtype=np.float64)),
            "percentile_95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "count": int(values.size),
        }
        actual = m["range_statistics"][statistics_method]
        for statistic, value in expected.items():
            if isinstance(value, float):
                assert abs(actual[statistic] - value) < 1e-12
            else:
                assert actual[statistic] == value
        trace = arrays[f"trace_{trace_method}"]
        row = m["hero"]["token_position"] - m["token_window"]["position_start"]
        cell = values[row, m["prismquant_receiving_group"]]
        np.testing.assert_allclose(np.ptp(trace), cell, rtol=1e-6)
        assert m["trace_zero_points"][trace_method] == float(np.float16(trace.min()))
    f3=json.loads((HERE/'fig3_metadata.json').read_text())
    assert f3['layer']==m['layer'] and f3['token_window']==m['token_window']
    score=arrays['geometry_scores']; covariance=np.cov(score,rowvar=False,ddof=1)
    vals=np.linalg.eigvalsh(covariance)[::-1]
    np.testing.assert_allclose(vals,f3['covariance']['top_two_eigenvalues'],rtol=1e-10)
    np.testing.assert_allclose(2*np.sqrt(vals),f3['ellipse_semiaxes_two_sd'],rtol=1e-10)
    assert f3['cloud_counts']['inside_frame']==[512,512]
    assert f3['ranges']['before_raw']==float(np.ptp(arrays['trace_raw']))
    assert f3['ranges']['after_prismquant']==float(np.ptp(arrays['trace_nar_kmax']))
    assert not any((HERE/f'fig3c.{suffix}').exists() for suffix in ['pdf','png','svg'])
    df=pd.read_csv(HERE/'fig2_capture.csv'); meta2=json.loads((HERE/'fig2_metadata.json').read_text())
    for col,out in [('mean_group_range','mean_range_reduction_percent'),('nmse','mean_nmse_reduction_percent')]:
        part=df[df.model.eq('llama32_3b')&df.site.eq('down')&df.method.isin(['hadamard','nar'])]
        pivot=part.pivot(index='layer',columns='method',values=col)
        assert len(pivot)==28
        np.testing.assert_allclose(100*((pivot.hadamard-pivot.nar)/pivot.hadamard).mean(),meta2[out],rtol=1e-12)
    panels=[f'fig1{x}' for x in 'abcdefg']+[f'fig2{x}' for x in 'abc']+[f'fig3{x}' for x in 'ab']
    sizes={}
    for name in panels+['fig1','fig2','fig3']:
        for suffix in ['pdf','svg']:
            assert (HERE/f'{name}.{suffix}').stat().st_size>1000
        with pymupdf.open(HERE/f'{name}.pdf') as doc:
            assert len(doc)==1 and doc[0].get_text().strip()
            sizes[name]=[doc[0].rect.width/72,doc[0].rect.height/72]
            assert doc[0].get_fonts()
    for name in ['fig1a','fig1b','fig1c','fig1d']:
        np.testing.assert_allclose(sizes[name],[3.2,2.45],atol=1e-6)
    result={'status':'PASS','checks':['bare a–d panels contain axes/ticks/labels only',
        'transparent 300-dpi a–d PNG exports','shared 0–40 row-1 limits and normalization',
        'shared 0–8.92 row-2 limits and normalization','dense grid contract recorded',
        'median/mean/95th-percentile independently recomputed in metadata',
        'full c/d extrema inside common normalization','identical numerical channel windows',
        'trace ranges equal selected landscape cells','zero point equals fp16 minimum',
        'Figure 3 eigenvalues independently recovered from centered score covariance',
        'same-layer/window linkage and measured-range linkage','28 unchanged paired Figure 2 measurements',
        'removed Figure 3c exports','editable text and complete vector export bundle'],
        'source_array_sha256':hashlib.sha256((HERE/'fig1_source_arrays.npz').read_bytes()).hexdigest(),
        'pdf_sizes_inches':sizes}
    (HERE/'qa'/'numerical-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
