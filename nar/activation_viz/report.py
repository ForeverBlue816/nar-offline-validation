"""Build captions and an index from measured CSVs; no assumed performance claims."""
import argparse,csv,json
from pathlib import Path
LABELS={'unrotated':'Unrotated','hadamard':'Hadamard','nar_k8':'PrismQuant (k=8)','nar_kmax':'PrismQuant (k=max)'}
def num(x):
    v=float(x)
    return f'{v:.2e}' if v and (abs(v)<.01 or abs(v)>=1000) else f'{v:.2f}'
def run(root):
    root=Path(root);rows=list(csv.DictReader((root/'metrics_summary.csv').open()))
    manifest=json.loads((root/'run_manifest.json').read_text())
    text=['# Qwen3-8B activation diagnostics','',
      'Real Qwen/Qwen3-8B-Base forwards; 8 fixed WikiText-2 test windows of 2048 tokens. Rotation seed 0; sample selection seed 42. Main surfaces show sample 0; statistics use all samples at full resolution.','',
      '## Figures','',
      'Main mechanism figures use paired local inputs. End-to-end figures separately show the three existing E22 W4A4KV4 evaluation rows; no unquantized reference is mislabelled as an end-to-end quantized row.','',
      '| Site | Raw magnitude | Group-centered residual | Range distribution |','|---|---|---|---|']
    for site in ('down_proj','q_proj'):
        text.append(f'| {site} | [Overview](figures/paired_local/{site}/raw/overview/matrix.png) · [Detail](figures/paired_local/{site}/raw/detail/matrix.png) | [Overview](figures/paired_local/{site}/residual/overview/matrix.png) · [Detail](figures/paired_local/{site}/residual/detail/matrix.png) | [ECDF](figures/paired_local/{site}/group_range_ecdf.pdf) |')
    text+=['','PDF and SVG siblings accompany every PNG; individual panels and method rows are in the same directories. Overview colors encode max magnitude within each bin. Zero-based bin IDs map to exact token/channel edges in display_cache.npz. Different block columns may use different z limits; comparisons across methods within a column share z and color limits.','',
      '## Full-resolution paired results','',
      'NMSE is pooled error energy divided by pooled activation energy. rho is pooled signed group-mean energy divided by activation energy. Range means include all non-padding token/group observations. Sample variability is in metrics_summary.csv; these eight windows are descriptive replicates, not independently trained models.','',
      '| Site | Block | Method | NMSE | rho | Mean group range | Token 0 energy fraction |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['mode']=='paired_local':text.append(f"| {r['site']} | {int(r['layer'])+1} | {LABELS[r['method']]} | {num(r['nmse'])} | {num(r['rho'])} | {num(r['range_mean'])} | {num(r['token0_energy_fraction'])} |")
    text+=['','## Interpretation boundaries','',
      'Group centering removes a common component for visualization. It preserves every signed group range (verified numerically) and is not an extra deployed operation. Its mean is not the quantizer offset. Raw peak suppression, residual energy, range, and actual QDQ error answer different questions. A local NMSE change alone does not establish perplexity or downstream accuracy. Large token-0 activations alone do not establish attention-sink causality. Rotated channels represent a new basis.','',
      '## Provenance and reproduction','',
      '[Run manifest](run_manifest.json) · [Capture sites](capture_site_report.md) · [Validation](validation_report.json) · [Predeclared contract](figure_contract.md) · [Captions](captions.tex) · [Per-sample metrics](metrics_per_sample.csv) · [Pooled metrics](metrics_summary.csv).','',
      'Full signed FP32 tensors are retained at raw_activation_root in the run manifest; activation_inventory.json records every shard hash and shared canonical input hash. Binary checkpoints and multi-GB raw shards are not committed to Git. The compact display cache is derived only from sample 0; all statistics use full-resolution shards.','',
      'Style reference: [SpinQuant Appendix C, Figures 8 and 9](https://arxiv.org/pdf/2405.16406). These Qwen3 measurements and their fixed layer/sample choices are independent of the Llama illustrations in that paper.','',
      '```bash',
      'python -m nar.activation_viz.capture --workdir "$NAR_WORKDIR" --output "$RUN"',
      'python -m nar.activation_viz.metrics "$RUN" --device cuda',
      'python -m nar.activation_viz.plot "$RUN"',
      'python -m nar.activation_viz.report "$RUN"','```','']
    (root/'README.md').write_text('\n'.join(text))
    captions=[]
    for mode in ('paired_local','end_to_end'):
        for site,location in [('down_proj',r'\texttt{mlp.down\_proj}, after $\mathrm{SiLU}(\mathrm{gate})\odot\mathrm{up}$'),('q_proj',r'\texttt{self\_attn.q\_proj}, using the deployed global $R_1$')]:
            for quantity in ('raw','residual'):
                desc=(r'$|Y|$, where $Y$ is floating activation after rotation and before activation QDQ' if quantity=='raw' else r'$|E|$, with $E=Y-\operatorname{mean}_{\mathrm{group}}(Y)$ computed from signed post-rotation, pre-QDQ activations')
                rowsdesc=('Rows show Unrotated, Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) applied to identical canonical inputs from a norm-fused FP32 reference.' if mode=='paired_local' else 'Rows show the Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) E22 GPTQ group-128 asymmetric W4A4KV4 evaluation checkpoints. Each includes upstream quantization; these inputs are not paired canonical activations.')
                site_tex=site.replace('_',chr(92)+'_')
                cap=f"\\paragraph{{{mode.replace('_',' ')}; {site_tex}; {quantity}.}} Qwen3-8B-Base activations at {location}. Panels show {desc}. {rowsdesc} Columns are Blocks 1, 13, 24, and 36 (zero-based layers 0, 12, 23, and 35). Surfaces show sample 0 of eight WikiText-2 test windows, each 2048 tokens, selected with seed 42; no calibration uses these windows. Rotation seed is 0. Actual ranks are 8/32 for R1 and 8/96 for R4. Overview uses maximum-absolute pooling into 128 token bins and 256 channel bins with token 0 separate. Detail shows the predetermined first 128 tokens and 512 channels (four groups), without pooling. All peaks are retained. Linear z/color scales are shared across methods within each block/site/quantity/view but may differ across blocks. View: elevation 25 degrees, azimuth -60 degrees. Rotated channel coordinates denote a new basis. Quantizer groups contain 128 contiguous channels; scale and real offset are rounded to FP16 before QDQ. Full-resolution statistics use all eight windows. Group centering is diagnostic only and preserves signed group ranges; it is not the stored offset or a deployed operation. Surface marks are rasterized at 600 dpi with vector text/axes."
                captions.append(cap)
    captions.append(r'\paragraph{Group-range ECDF.} Full-resolution signed group ranges over all eight test windows, displayed at 1001 fixed empirical quantiles including the extrema. The x axis is symmetric-log with a linear region below 0.01; the y axis is cumulative probability. No pooling is applied before calculating the empirical distribution. The four block columns use the same sampling and method definitions as the corresponding surface panels. These are descriptive activation distributions, not confidence intervals or evidence of downstream accuracy.')
    (root/'captions.tex').write_text('\n\n'.join(captions)+'\n')
    print('REPORT COMPLETE')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
