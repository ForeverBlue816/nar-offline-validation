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
    assert m["row1_z_limits"] == {"raw": [0.0, 40.0], "hadamard": [0.0, 4.0]}
    assert m["row2_shared_z_limits"] == [0.0, 10.0]
    assert m["rendering_contract"]["row1"]["normalization"] == m["row1_z_limits"]
    assert m["rendering_contract"]["row2"]["normalization"] == [0.0, 10.0]
    assert m["rendering_contract"]["row2"]["view"] == {"elevation": 18, "azimuth": -62}
    assert m["rendering_contract"]["dense_grid"] == {
        "channel_interval": 250,
        "token_interval": 100,
        "group_interval": 10,
        "row1_z_interval": 5,
        "row2_z_interval": 1.25,
    }
    qualifying = arrays["all_channel_median_magnitudes"] > 1.0
    cs = np.r_[0, np.cumsum(qualifying)]
    counts = cs[2048:] - cs[:-2048]
    np.testing.assert_array_equal(counts, arrays["peak_density_window_counts"])
    best = int(counts.argmax())
    assert best == m["peak_density_selection"]["start_channel"] == m["channel_windows"]["raw"][0]
    assert int(counts[best]) == m["peak_density_selection"]["qualifying_channel_count"]
    np.testing.assert_array_equal(np.median(arrays["raw_magnitude"], axis=0), arrays["all_channel_median_magnitudes"][best:best+2048])
    assert arrays["raw_magnitude"].max() <= 40 and arrays["hadamard_magnitude"].max() <= 4
    for letter in "efg":
        from PIL import Image
        with Image.open(HERE / f"fig1{letter}.png") as panel_image:
            assert panel_image.mode == "RGBA" and panel_image.size == (630,525)
            assert panel_image.getpixel((0,0))[3] == 0
        with pymupdf.open(HERE / f"fig1{letter}.pdf") as doc:
            text = doc[0].get_text()
            for label in ("signed value", "channel in group", "32", "64", "96", "127"):
                assert label in text, (letter, label, text)
            assert "5" in text
    assert m["trace_rendering"]["x_ticks"] == [0,32,64,96,127]
    assert m["trace_rendering"]["y_ticks"] == [-5,0,5]
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
    assert f3["layer"] == m["layer"] == 27
    assert f3["restored_from_commit"] == "721f253"
    projections = pd.read_csv(HERE / "fig3_token_projections.csv")
    assert (projections.layer == m["layer"]).all() and projections.bos_excluded.all()
    assert (projections.token_position > 0).all()
    cloud = projections[["projection_v1", "projection_v2"]].to_numpy()
    cloud = (cloud - cloud.mean(0)) / cloud.std(0)
    limits = np.asarray(f3["geometry"]["cloud"]["axis_limits"])
    assert np.all((cloud >= limits[:,0]) & (cloud <= limits[:,1]))
    assert len(cloud) == f3["geometry"]["cloud"]["inside_frame"] == 8064
    vectors = np.load(HERE / "fig3_geometry_vectors.npz")["vectors"]
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), atol=1e-5)
    arrows = f3["geometry"]["arrows"]
    assert abs(arrows["nar"]["in_plane_length"] - 1) < 1e-5
    assert abs(arrows["nar"]["projection_v2"]) < 1e-5
    assert 0 < arrows["hadamard"]["in_plane_length"] < 0.03
    assert f3["range_law_subpanels"]["fig3c1"]["points"] == 2520
    assert f3["range_law_subpanels"]["fig3c2"]["points"] == 392
    with pymupdf.open(HERE / "fig3c1.pdf") as doc:
        text = doc[0].get_text()
        assert "E1c activations" in text and "E7 V cache" not in text
        assert "Pooled R" in text and "0.86" in text
    with pymupdf.open(HERE / "fig3c2.pdf") as doc:
        text = doc[0].get_text()
        assert "E7 V cache" in text and "E20 multi-slot" in text
        assert "E1c activations" not in text and "Pooled R" in text
    law = pd.read_csv(HERE / "fig3_range_law.csv")
    x = law.sqrt_one_minus_f.to_numpy(); y = law.range_ratio_vs_hadamard.to_numpy()
    np.testing.assert_allclose(x, np.sqrt(1 - law.absorbed_energy_fraction.to_numpy()), atol=1e-12)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    r2 = 1 - np.sum((y - design @ [intercept,slope])**2) / np.sum((y-y.mean())**2)
    np.testing.assert_allclose([intercept,slope,r2], [f3["fit_intercept"],f3["fit_slope"],f3["fit_r_squared"]], atol=1e-12)
    assert f"{r2:.2f}" == "0.86" and len(law) == f3["range_law_points"] == 2912
    assert f3["point_counts"] == {str(k): int(v) for k,v in law.source_family.value_counts().items()}
    assert x.min() >= 0 and x.max() <= f3["main_axis_limits"][0][1]
    assert y.min() >= 0 and y.max() <= f3["main_axis_limits"][1][1]
    df=pd.read_csv(HERE/'fig2_capture.csv'); meta2=json.loads((HERE/'fig2_metadata.json').read_text())
    for col,out in [('mean_group_range','mean_range_reduction_percent'),('nmse','mean_nmse_reduction_percent')]:
        part=df[df.model.eq('llama32_3b')&df.site.eq('down')&df.method.isin(['hadamard','nar'])]
        pivot=part.pivot(index='layer',columns='method',values=col)
        assert len(pivot)==28
        np.testing.assert_allclose(100*((pivot.hadamard-pivot.nar)/pivot.hadamard).mean(),meta2[out],rtol=1e-12)
    panels=[f'fig1{x}' for x in 'abcdefg']+[f'fig2{x}' for x in 'abc']+[f'fig3{x}' for x in ['a','b','c','c1','c2']]
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
        'transparent 300-dpi a–g PNG exports','independent raw 0–40 and Hadamard 0–4 height/color scales',
        'shared 0–10 row-2 height/color scale','dense grid contract recorded',
        'median/mean/95th-percentile independently recomputed in metadata',
        'full c/d extrema inside common normalization','identical numerical channel windows',
        'trace ranges equal selected landscape cells','zero point equals fp16 minimum',
        'peak-density selection independently recovered over all candidate windows',
        'standalone trace tick numbers and both axis labels',
        'all 8064 layer-27 non-BOS cloud rows inside the frame',
        'unit PrismQuant direction aligned with frozen v1',
        'all 2912 range-law points included; pooled R-squared independently reproduced','28 unchanged paired Figure 2 measurements',
        'restored Figure 3c exports','editable text and complete vector export bundle'],
        'source_array_sha256':hashlib.sha256((HERE/'fig1_source_arrays.npz').read_bytes()).hexdigest(),
        'pdf_sizes_inches':sizes}
    (HERE/'qa'/'numerical-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
