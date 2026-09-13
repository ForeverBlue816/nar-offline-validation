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
    from verify_fig1 import main as verify_figure1
    verify_figure1()
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
            assert panel_image.mode == "RGBA" and panel_image.size == (612,330)
            assert panel_image.getpixel((0,0))[3] == 0
        with pymupdf.open(HERE / f"fig1{letter}.pdf") as doc:
            text = doc[0].get_text()
            for label in ("channel in group", "64", "127"):
                assert label in text, (letter, label, text)
            assert "5" in text
    assert m["trace_rendering"]["x_ticks"] == [0,64,127]
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
            expected=m['rendering_contract']['scientific_panels'][letter]['size_inches']
            assert image.size == tuple(int(v*300) for v in expected), (letter, image.size)
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
    from verify_fig2_fig3_revised import verify
    verify()
    from verify_fig2_duquant import main as verify_duquant
    verify_duquant()
    print('PASS: Figure 1 numerical checks and active Figure 2/3 source/export checks.')

if __name__=='__main__': main()
