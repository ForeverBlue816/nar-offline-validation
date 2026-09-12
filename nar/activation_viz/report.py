"""Describe revised figures without rewriting experimental results or metadata."""
import argparse
import json
from pathlib import Path
from .detail_plot import CONFIG, HEIGHT_AXIS_TARGETS, write_json


def run(root):
    root = Path(root)
    validation = json.loads((root/'validation_report.json').read_text())
    failures = [r for r in validation['checks'] if not r['passed']]
    captions_path = root/'captions.tex'
    previous = captions_path.read_text() if captions_path.exists() else ''
    ecdf_captions = [cap for cap in previous.split('\n\n') if cap.startswith(r'\paragraph{Group-range ECDF;')]
    captions = []
    text = ['# Qwen3-8B activation diagnostics', '',
        'Real Qwen/Qwen3-8B-Base activations, fixed sample 0 from eight WikiText-2 test windows. The current detail revision changes only presentation. Experimental metrics, frozen factors, quantizers and numerical limitations are unchanged.', '',
        '[Revision report](figure_revision_report.md) · [Rendering contract](full_resolution_contract.md) · [Render settings](detail_render_config.json) · [Measured findings](measured_summary.md) · [Publication inventory](publication_manifest.json)', '',
        '## Local surfaces', '',
        '![Local group-centered down-projection activations](figures/paired_local/down_proj/residual/detail/rotated_only_zoom.png)', '',
        '| Forward | Site | Raw detail | Residual detail | Full-domain context |',
        '|---|---|---|---|---|']
    for mode in ('paired_local', 'end_to_end'):
        for site in ('q_proj', 'down_proj'):
            prefix = f'figures/{mode}/{site}'
            text.append(f'| {mode} | {site} | [PDF]({prefix}/raw/detail/matrix.pdf) | [PDF]({prefix}/residual/detail/matrix.pdf) | [Raw overview]({prefix}/raw/overview/matrix.pdf) · [Residual overview]({prefix}/residual/overview/matrix.pdf) |')
            for quantity in ('raw', 'residual'):
                desc = (r'$|Y[0:128,0:512]|$, with signed floating $Y$ immediately before activation QDQ' if quantity == 'raw'
                        else r'$|E[0:128,0:512]|$, where $E=Y-\operatorname{mean}_{g128}(Y)$ is computed per token on complete signed $Y$ before cropping and taking absolute values')
                rows = ('Rows apply Unrotated, Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) to the same canonical input.' if mode == 'paired_local'
                        else 'The top row reuses the matching paired-local Unrotated reference cache, from the BF16 checkpoint loaded and norm-fused in FP32 without rotation or quantization. The lower rows are observed Hadamard, PrismQuant ($k=8$), and PrismQuant ($k=\max$) E22 W4A4KV4 floating QDQ forwards, including upstream quantization; their intermediate activations are not identical canonical inputs.')
                palette_note = ('The main matrix uses the original blue-to-orange colormap; companion exports retain Viridis.' if (mode, site, quantity) in HEIGHT_AXIS_TARGETS else 'Viridis is used throughout.')
                site_tex = site.replace('_', r'\_')
                captions.append(f'\\paragraph{{Local detail; {mode.replace("_", " ")}; {site_tex}; {quantity}.}} '
                    f'Qwen3-8B-Base, Blocks 1, 13, 24, and 36. Surfaces show {desc}. {rows} '
                    'Sample 0, tokens 0--127 and channels 0--511, includes four complete contiguous groups of 128 channels; floor guides mark boundaries at 127.5, 255.5, and 383.5. '
                    'All 65,536 local vertices enter the native 3D surface without pooling, data stride, padding, smoothing, height columns or sidewalls. '
                    f'Linear heights; square-root color mapping. {palette_note} Fixed gamma 0.5 maps colors only; each column shares zero to 1.03 times the maximum across its displayed methods, with a separate colorbar in measured units. '
                    'The linear-color companion keeps the same heights, camera and limits. The rotated-only zoom uses a separately labelled scale shared by its three methods. '
                    'Orthographic view: elevation 28 degrees, azimuth -55 degrees; box aspect 2:1.2:1. '
                    'Unrotated reference means norm-fused FP32, not BF16 inference. Rotated channels denote a different basis. '
                    'Group centering is diagnostic, not QDQ reconstruction error or the quantizer offset. '
                    'The full-domain overview provides global context; this fixed local window need not include global outliers. Full-data metrics and ECDFs still use all eight 2048-token samples. '
                    'Surface marks are rasterized at 600 dpi; text and axes remain vector. Supplementary validation failures remain unchanged.')
                if (mode, site, quantity) in HEIGHT_AXIS_TARGETS:
                    captions[-1] += ' The main matrix labels linear height ticks; the pale floor denotes z=0, and space below a positive surface is unfilled.'
    text += ['',
        'Every detail directory contains `matrix`, `matrix_linear`, `rotated_only_zoom`, method rows and individual panels in PDF, PNG and SVG. Each column has its own measured-unit colorbar. Raw magnitudes may remain similar between methods; no visual separation is imposed.', '',
        'Detail uses exactly tokens [0,128), channels [0,512), sample 0 and four g128 groups. Raw heights are abs(Y). Residual heights are abs(Y − mean_group(Y)), with the signed mean computed on full Y before slicing. Heights are linear; only color uses fixed square-root mapping. All methods use the same window and camera. The separate rotated-only view has a different, explicitly shared scale.', '',
        'The unrotated row is an unquantized norm-fused FP32 reference. In end-to-end matrices it comes from the matching paired_local/unrotated cache. Its input IDs, sample, layer, site and norm-fusion metadata are checked; the three quantized forwards include upstream QDQ and are not claimed to share identical intermediate inputs.', '',
        'Full-domain overviews remain available, unchanged from the preceding publication. Their historical solid geometry is documented in the archived full-resolution contract; it is not used for current local detail. Overview and detail are independent assets. Metrics and exact ECDF values/counts retain all eight full samples. The local window is not an exhaustive model-outlier survey.', '',
        'Four-column matrices use 11.75 pt text at 12.05-inch export width, retaining 7.02 pt text when inserted at 7.2 inches (183 mm). Individual panels use 8.5 pt native text and remain readable at 3.5-inch insertion width. Channel ticks 0, 256 and 511 and token ticks 0, 64 and 127 avoid crowding. All vertices are still drawn. Floor guides and short front-edge ticks mark the three g128 boundaries; per-column colorbars give the shared height/color limits. PDF/SVG axes and text are vector; only surface marks are rasterized.', '',
        'The three priority raw matrices (end_to_end q_proj/down_proj and paired_local q_proj) restore the original blue-to-orange colormap with gamma 0.5, label linear height ticks and identify the pale floor as z=0. These axis annotations clarify real nonzero surface heights; their linear-color controls and zoom companions retain their previous Viridis palette and layout. These retained companions are not matched-colormap controls for the newly restored blue-orange matrices.', '',
        '## Numerical results and limitations', '',
        '[Per-sample metrics](metrics_per_sample.csv) · [Full-data summary](metrics_summary.csv) · [Original measured interpretation](measured_summary.md) · [Validation](validation_report.json)', '',
        f'Required-check status: {validation.get("required_checks_passed")}. Overall all-checks status: {validation["passed"]}. The {len(failures)} supplementary failures below are retained at their original thresholds:', '']
    for failure in failures:
        text.append(f'- {failure["check"]}, {failure.get("method")}, Block {failure.get("layer",-1)+1}: {failure["value"]}; threshold {failure["limit"]}.')
    text += ['', 'Group means are not actual quantizer offsets. Local raw peaks, group-centered residuals, signed group ranges and QDQ errors answer different questions. These figures do not create a new downstream-accuracy conclusion.', '',
        '## Reproduction', '',
        'The raw signed FP32 shards remain at `raw_activation_root` in [run_manifest.json](run_manifest.json). [activation_inventory.json](activation_inventory.json) records all 448 shard hashes. No model rerun is needed.', '',
        '```bash',
        'python -m nar.activation_viz.full_batch "$RAW_RUN" "$RUN" --workers 4',
        'python -m nar.activation_viz.detail_checks "$RAW_RUN" "$RUN"',
        'python -m nar.activation_viz.report "$RUN"',
        'python -m nar.activation_viz.render_batch "$RUN" --audit-only',
        'python -m nar.activation_viz.publish "$RUN" "$PUBLICATION_DIR"',
        '```', '',
        'This is a user-requested visualization revision. Earlier predeclared experiment rules and superseded display records are retained; the new window and color mapping are not presented as the original preregistration.', '']
    (root/'README.md').write_text('\n'.join(text))
    captions_path.write_text('\n\n'.join(captions + ecdf_captions) + '\n')
    manifest = json.loads((root/'run_manifest.json').read_text())
    manifest.setdefault('display_revision_history', []).append(manifest['display']) if manifest.get('display', {}).get('revision') != CONFIG['revision'] else None
    manifest['display'] = {**CONFIG, 'overview': 'previous full-domain exports preserved independently',
                           'ecdf': 'unchanged exact full-data values and multiplicities'}
    manifest['render_commands'] = ['python -m nar.activation_viz.full_batch $RAW_RUN $RUN --workers 4']
    write_json(root/'run_manifest.json', manifest)
    if not (root/'detail_render_config.json').exists():
        write_json(root/'detail_render_config.json', CONFIG)
    print('FIGURE REPORT COMPLETE')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); run(p.parse_args().root)
