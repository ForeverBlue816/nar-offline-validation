"""Write a bounded interpretation directly from full-resolution measured metrics."""
import argparse,csv,json
from pathlib import Path

def run(root):
    root=Path(root);rows=list(csv.DictReader((root/'metrics_summary.csv').open()))
    table={(r['mode'],r['site'],int(r['layer']),r['method']):r for r in rows}
    methods=['hadamard','nar_k8','nar_kmax'];names=['Hadamard','PrismQuant (k=8)','PrismQuant (k=max)']
    text=['# Measured results','',
      'Qwen3-8B-Base; eight fixed WikiText-2 test windows; 2048 tokens/window; seed 42; frozen rotation seed 0. The following results are for paired local down_proj inputs. They use complete tensors and pooled energy ratios.','',
      '| Block | Method | Maximum absolute value | Mean group range | NMSE | Common-component energy (%) |',
      '|---:|---|---:|---:|---:|---:|']
    for l in [0,12,23,35]:
        for m,name in zip(methods,names):
            r=table[('paired_local','down_proj',l,m)]
            text.append(f"| {l+1} | {name} | {float(r['absmax']):.2f} | {float(r['range_mean']):.2f} | {float(r['nmse']):.2e} | {100*float(r['rho']):.2f} |")
    rho_percent=100*float(table[('paired_local','down_proj',35,'nar_kmax')]['rho'])
    text+=['',
      'Across these four preselected blocks, both PrismQuant settings have larger maximum absolute values than Hadamard and smaller mean group ranges and actual activation-QDQ NMSE. This directly shows why raw peak height and quantization quality need to be evaluated separately. Group centering is an analysis decomposition: it leaves the range of every token/group unchanged. It is not a deployed subtraction or the quantizer’s stored offset.','',
      f'The strongest measured change is at Block 36. For k=max, common-component energy accounts for {rho_percent:.2f}% of total activation energy. This is consistent with group alignment of dominant activation directions. It does not establish a downstream perplexity/accuracy improvement or attention-sink causality. The q_proj control and separately captured end-to-end rows are available in the complete CSV tables.','',
      'Required data/energy/complete-projection/capture checks pass. Three supplementary probes exceed their unchanged tolerances: one inverse check and two narrow-output checks on Block 36 frozen factors. Overall all-checks status remains FAIL; the initial failed probes and exact factor normalization deviations are retained in qa/. No checkpoint or factor was renormalized.','',
      '[All metrics](metrics_summary.csv) · [Per-sample variation](metrics_per_sample.csv) · [Validation](validation_report.json)','']
    (root/'measured_summary.md').write_text('\n'.join(text))
    for layer in (0,12,23,35):
        h=table[('paired_local','down_proj',layer,'hadamard')]
        for method in ('nar_k8','nar_kmax'):
            r=table[('paired_local','down_proj',layer,method)]
            assert float(r['absmax'])>float(h['absmax'])
            assert float(r['range_mean'])<float(h['range_mean']) and float(r['nmse'])<float(h['nmse'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
