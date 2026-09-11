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
    validation=json.loads((root/'validation_report.json').read_text())
    failures=[r for r in validation['checks'] if not r['passed']]
    text=['# Qwen3-8B activation diagnostics','',
      'Real Qwen/Qwen3-8B-Base forwards; 8 fixed WikiText-2 test windows of 2048 tokens. Rotation seed 0; sample selection seed 42. Main surfaces show sample 0; statistics use all samples at full resolution.','',
      '[Measured findings](measured_summary.md) · [Figure audit](qa/delivery_review.json) · [Publication inventory](publication_manifest.json)','',
      '## Figures','',
      '![Group-centered down-projection activations, rotated methods](figures/paired_local/down_proj/residual/overview/rotated_only_zoom.png)','',
      'Main mechanism figures use paired local inputs. End-to-end figures separately show the three existing E22 W4A4KV4 evaluation rows; no unquantized reference is mislabelled as an end-to-end quantized row.','',
      '| Site | Raw magnitude | Group-centered residual | Range distribution |','|---|---|---|---|']
    for site in ('down_proj','q_proj'):
        text.append(f'| {site} | [Full matrix](figures/paired_local/{site}/raw/overview/matrix.png) · [Unrotated](figures/paired_local/{site}/raw/overview/row_unrotated.png) | [Full matrix](figures/paired_local/{site}/residual/overview/matrix.png) · [Rotated zoom](figures/paired_local/{site}/residual/overview/rotated_only_zoom.png) | [ECDF](figures/paired_local/{site}/group_range_ecdf.pdf) |')
    text+=['','PDF and SVG siblings accompany every PNG; individual panels and method rows are in the same directories. Every surface uses all 2048 tokens and all 12288 down-projection or 4096 query-projection channels. The axes show actual indices. All adjacent grid cells and measured vertices are rasterized without pooling, strides or cropping. Exterior sidewalls close the height surface to z=0 without changing any measured height. The historical overview and detail directories now contain identical full-resolution figures. Different block columns may use different z limits; comparisons across methods within a column share z and color limits.','',
      '## Full-resolution paired results','',
      'NMSE is pooled error energy divided by pooled activation energy. rho is pooled signed group-mean energy divided by activation energy. Range means include all non-padding token/group observations. Sample variability is in metrics_summary.csv; these eight windows are descriptive replicates, not independently trained models.','',
      '| Site | Block | Method | NMSE | rho | Mean group range | Token 0 energy fraction |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if r['mode']=='paired_local':text.append(f"| {r['site']} | {int(r['layer'])+1} | {LABELS[r['method']]} | {num(r['nmse'])} | {num(r['rho'])} | {num(r['range_mean'])} | {num(r['token0_energy_fraction'])} |")
    text+=['','## Numerical validation','',f"Required-check status: {validation.get('required_checks_passed')}. Overall all-checks status: {validation['passed']}.",'','Frozen factors are reused unchanged. Supplementary failures are retained at their original thresholds:','']
    for failure in failures:
        text.append(f"- {failure['check']}, {failure.get('method')}, Block {failure.get('layer',-1)+1}: {num(failure['value'])}; threshold {num(failure['limit'])}.")
    text+=['','The initial eight-output-row projection probes are retained alongside checks of all actual output channels on the same real tokens. This explicitly broadens the tested operator; it does not make the narrow-probe failures pass. See the predeclared contract addendum and archived failed probes. Stored reflectors have small normalization deviations; these diagnostic figures are not a claim of exact finite-precision orthogonality.','']
    text+=['','## Interpretation boundaries','',
      'Group centering removes a common component for visualization. It preserves every signed group range (verified numerically) and is not an extra deployed operation. Its mean is not the quantizer offset. Raw peak suppression, residual energy, range, and actual QDQ error answer different questions. A local NMSE change alone does not establish perplexity or downstream accuracy. Large token-0 activations alone do not establish attention-sink causality. Rotated channels represent a new basis.','',
      '## Provenance and reproduction','',
      '[Run manifest](run_manifest.json) · [Capture sites](capture_site_report.md) · [Validation](validation_report.json) · [Predeclared contract](figure_contract.md) · [Captions](captions.tex) · [Per-sample metrics](metrics_per_sample.csv) · [Pooled metrics](metrics_summary.csv).','',
      'Full signed FP32 tensors are retained at raw_activation_root in the run manifest; activation_inventory.json records every shard hash and shared canonical input hash. Binary checkpoints and multi-GB raw shards are not committed to Git. Figures directly read the complete sample-0 signed shards. The old pooled display cache is no longer used or distributed. Exact ECDF files retain every distinct group-range value and its full multiplicity across all eight samples. See full_resolution_contract.md and per-figure geometry.json for the rendering contract and source vertex counts.','',
      'Style reference: [SpinQuant Appendix C, Figures 8 and 9](https://arxiv.org/pdf/2405.16406). These Qwen3 measurements and their fixed layer/sample choices are independent of the Llama illustrations in that paper.','',
      '```bash',
      'python -m nar.activation_viz.capture --workdir "$NAR_WORKDIR" --output "$RUN"',
      'python -m nar.activation_viz.metrics "$RUN" --device cuda',
      'python -m nar.activation_viz.full_batch "$RAW_RUN" "$RUN" --workers 8',
      'python -m nar.activation_viz.report "$RUN"',
      'python -m nar.activation_viz.summarize "$RUN"',
      'python -m nar.activation_viz.render_batch "$RUN" --audit-only',
      'python -m nar.activation_viz.publish "$RUN" "$PUBLICATION_DIR"','```','']
    manifest['bit_widths']={'linear_weights':4,'activation_inputs':4,'keys':4,'values':4,'exceptions':'embeddings, lm_head, norms and recent KV residual are floating; evaluation containers are FP32'}
    manifest['weight_quantizer']={'implementation':'nar.quarot_gptq.WeightQuantizer','group_size':128,'group_axis':'input-channel groups within each output row','clipping':'per-output-row/per-group MSE search; norm=2.4, grid=100, maxshrink=0.8; detailed checkpoint settings retained','scale_definition':'clipped span / 15, span floor 1e-5; FP32 builder arithmetic','zero_point':'round(-clipped_min/scale), integer-valued zero point in FP32 builder container','saved_form':'only dequantized floating weights, not packed codes or separately serialized scale/zero arrays','nominal_packed_bit_budget':4.15625,'budget_assumption':'4-bit codes + 16-bit scale + 4-bit zero per 128 weights; not actual checkpoint file storage'}
    manifest['module_paths']=[f'model.layers.{layer}.{site}' for layer in manifest['layers_zero_based'] for site in manifest['sites'].values()]
    manifest['mathematical_audit_status']={'all_checks_passed':validation['passed'],'required_checks_passed':validation.get('required_checks_passed'),'failed_supplementary_checks':failures}
    manifest.setdefault('previous_display_settings',manifest.get('display'))
    manifest['display']={'tokens':2048,'down_proj_channels':12288,'q_proj_channels':4096,'sample':0,'data_stride':[1,1],'pooling':False,'cropping':False,'sidewall_base_z':0,'channel_tick_step':2000,'token_ticks':[0,1000,2000],'elev':25,'azim':-60,'overview_and_detail':'identical complete-data compatibility exports','ecdf':'exact values and multiplicities; no quantile thinning'}
    manifest['render_commands']=['python -m nar.activation_viz.full_batch $RAW_RUN $RUN --workers 8','python -m nar.activation_viz.report $RUN']
    (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    (root/'README.md').write_text('\n'.join(text))
    captions=[]
    for mode in ('paired_local','end_to_end'):
        for site,location in [('down_proj',r'\texttt{mlp.down\_proj}, after $\mathrm{SiLU}(\mathrm{gate})\odot\mathrm{up}$'),('q_proj',r'\texttt{self\_attn.q\_proj}, using the deployed global $R_1$')]:
            for quantity in ('raw','residual'):
                desc=(r'$|Y|$, where $Y$ is floating activation after rotation and before activation QDQ' if quantity=='raw' else r'$|E|$, with $E=Y-\operatorname{mean}_{\mathrm{group}}(Y)$ computed from signed post-rotation, pre-QDQ activations')
                rowsdesc=('Rows show Unrotated, Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) applied to identical canonical inputs from a norm-fused FP32 reference.' if mode=='paired_local' else 'Rows show the Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) E22 GPTQ group-128 asymmetric W4A4KV4 evaluation checkpoints. Each includes upstream quantization; these inputs are not paired canonical activations.')
                site_tex=site.replace('_',chr(92)+'_')
                cap=f"\\paragraph{{{mode.replace('_',' ')}; {site_tex}; {quantity}.}} Qwen3-8B-Base activations at {location}. Panels show {desc}. {rowsdesc} Columns are Blocks 1, 13, 24, and 36 (zero-based layers 0, 12, 23, and 35). Surfaces show sample 0 of eight WikiText-2 test windows, each 2048 tokens, selected with seed 42; no calibration uses these windows. Rotation seed is 0. Actual ranks are 8/32 for R1 and 8/96 for R4. All 2048 tokens and all channels enter each surface: 12288 channels at down-projection input and 4096 at query-projection input. There is no pooling, stride, smoothing, or crop. Every measured vertex and both triangles of every adjacent grid cell enter a depth-tested rasterizer. The exterior sides extend to zero as a geometric closure, not additional observations. Real channel indices are ticked every 2000. The historical overview/detail paths now show identical full-resolution figures. Finite output pixels and occlusion limit distinguishable detail without selecting input data. Linear z/color scales are shared across methods within each block/site/quantity/view but may differ across blocks. View: elevation 25 degrees, azimuth -60 degrees. Rotated channel coordinates denote a new basis. Quantizer groups contain 128 contiguous channels; scale and real offset are rounded to FP16 before QDQ. Full-resolution statistics use all eight windows. Group centering is diagnostic only and preserves signed group ranges; it is not the stored offset or a deployed operation. A separate rotated-only zoom uses a shared scale across the three rotated methods and a different scale from the main matrix. Supplementary inverse/narrow-output probes detect small frozen-factor deviations and are reported as failures; no factor is renormalized. Surface marks are rasterized at 600 dpi with vector text/axes."
                captions.append(cap)
    for mode in ('paired_local','end_to_end'):
        for site in ('down_proj','q_proj'):
            site_tex=site.replace('_',chr(92)+'_')
            captions.append(f"\\paragraph{{Group-range ECDF; {mode.replace('_',' ')}; {site_tex}.}} Qwen3-8B-Base, Blocks 1, 13, 24, and 36. Group ranges are computed from signed floating inputs immediately before activation QDQ at the specified projection, using contiguous channel groups of 128. All eight WikiText-2 test windows of 2048 tokens enter the full-resolution distribution; selection seed is 42 and rotation seed is 0. Paired local rows share canonical norm-fused reference inputs, whereas end-to-end rows include upstream E22 W4A4KV4 quantization. The curves display the exact empirical distribution as steps over every distinct observed range and its exact multiplicity, including both extrema; no quantile thinning or path simplification is used. Curve marks are rasterized at 600 dpi, and all distribution values and counts are archived. The nonnegative x axis uses a symmetric-log transform with linear threshold 0.01; the y axis is cumulative probability. Methods share each block's x scale. No activation pooling precedes the distribution calculation. These descriptive distributions do not establish downstream accuracy or exact finite-precision orthogonality; see the retained supplementary numerical failures.")
    (root/'captions.tex').write_text('\n\n'.join(captions)+'\n')
    print('REPORT COMPLETE')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
