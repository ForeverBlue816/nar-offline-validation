#!/usr/bin/env python3
"""Verify the DuQuant addendum and Figure 2 source-row linkage."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
def main():
    source=ROOT/'results/llama32_3b/e1c_per_layer.csv';raw=pd.read_csv(source)
    add=json.loads((source.parent/'E1C_DONE.json').read_text())['duquant_addendum']
    prefix=source.read_bytes()[:add['original_csv_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==add['original_csv_sha256']
    du=raw[raw.method.eq('duquant')];assert len(du)==56 and not du.duplicated(['site','layer']).any()
    assert du.groupby('site').layer.apply(lambda v:set(v)==set(range(28))).all()
    assert du.evaluation_tokens.eq(8192).all() and du.b.eq(128).all()
    audit=pd.read_csv(source.parent/'e1c_duquant_sanity.csv');assert len(audit)==56
    for row in du.itertuples():
        h=raw[raw.site.eq(row.site)&raw.layer.eq(row.layer)&raw.method.eq('hadamard_full')].iloc[0]
        np.testing.assert_allclose(row.range_reduction_vs_hadamard,(h.mean_group_range-row.mean_group_range)/h.mean_group_range,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(row.nmse_delta_vs_hadamard,row.relative_quantization_error_nmse-h.relative_quantization_error_nmse,rtol=1e-12,atol=1e-12)
        a=audit[audit.site.eq(row.site)&audit.layer.eq(row.layer)].iloc[0]
        np.testing.assert_allclose(row.mean_group_range,a.duquant_range,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose([a.hadamard_replay_range,a.hadamard_replay_nmse],[h.mean_group_range,h.relative_quantization_error_nmse],rtol=2e-5,atol=1e-8)
    assert audit.inside_paired_bracket.all() and add['plotting_gate']=='PASS'
    capture=pd.read_csv(FIG/'fig2_capture.csv')
    for stem,column in [('fig2b','mean_group_range'),('fig2c','nmse')]:
        table=pd.read_csv(FIG/f'{stem}.csv');assert len(table)==84
        for row in table.itertuples():
            frame=pd.read_csv(ROOT/row.source_file);original=frame.iloc[row.source_csv_line-2]
            assert original.method==row.method and original.layer==row.layer and original.site=='down_input'
            np.testing.assert_allclose(getattr(row,column),original[row.source_column],rtol=1e-12,atol=1e-12)
            method={'hadamard_full':'hadamard','duquant':'duquant_style','nar_kmax':'nar'}[row.method]
            plotted=capture[capture.model.eq(row.model)&capture.site.eq(row.site)&capture.layer.eq(row.layer)&capture.method.eq(method)]
            assert len(plotted)==1
            np.testing.assert_allclose(getattr(row,column),plotted.iloc[0][column],rtol=1e-12,atol=1e-12)
        assert '#e63946' in (FIG/f'{stem}.svg').read_text().lower()
    report={'main_figure_status':'PASS','original_e1c_rows_byte_identical':True,'appended_rows':56,'range_bracket_violations':0,
        'plotted_rows_per_metric':84,'duquant_layers_per_panel':28,
        'missing_frozen_e1c_models':json.loads((FIG/'fig2_metadata.json').read_text())['duquant_offline_addendum']['missing_frozen_e1c_models']}
    (FIG/'qa'/'fig2.duquant-integrity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
