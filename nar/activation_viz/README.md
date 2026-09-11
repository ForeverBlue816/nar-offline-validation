# Qwen3 activation figure revision

The current renderer restores sample 0, tokens [0,128), channels [0,512), as
native Matplotlib 3D upper surfaces. All 65,536 vertices are used without
pooling, stride, zero padding, height columns or sidewalls. Four complete g128
groups are located by real integer indices and floor-only boundary guides.
See [the rendering contract](detail_contract.md).

```bash
python -m nar.activation_viz.full_batch "$RAW_RUN" "$RUN" --workers 4
python -m nar.activation_viz.detail_checks "$RAW_RUN" "$RUN"
python -m nar.activation_viz.report "$RUN"
python -m nar.activation_viz.render_batch "$RUN" --audit-only
python -m nar.activation_viz.publish "$RUN" "$PUBLICATION_DIR"
```

Existing full-domain overviews and exact full-data ECDFs are preserved. Detail
is rendered independently and is never copied or hardlinked from overview.
`--redraw-overview` explicitly regenerates global surfaces with the optional
Python/Numba renderer; its surface, height-column and sidewall switches are
separate and the latter two default to false. It is unnecessary for the current
local revision. No capture, calibration, metric or quantizer rerun is performed.

`detail_plot.py` validates the run-wide input IDs and reference semantics, loads
signed FP32 shards, checks their hashes and computes residuals before cropping
and absolute values. Both modes have four main rows. End-to-end references
reuse paired-local unrotated caches and are labelled norm-fused FP32; the other
three rows remain the original floating QDQ forwards. No second rotation is
applied. Each block shares its z limit and Viridis/PowerNorm(gamma=0.5) among
methods. Linear heights are unchanged. Linear-color controls and rotated-only
zooms are separate exports. Raw similarities remain similarities.

Plotting uses the existing Python 3.11/Torch/Matplotlib environment and DejaVu
Sans text: 11.75 pt for four-column matrices, 8.5 pt for individual panels. PDF/SVG text and axes are vector;
surfaces are rasterized at 600 dpi. The 12.05-inch matrix export is intended for 7.2-inch (183 mm) paper
placement, retaining 7.02 pt text; individual panels support 3.5-inch placement. Panel
alignment, rendered PDF glyphs, collisions, exact corner/peak coordinates and
source hashes are audited. Debug arrays may be synthetic only in independent
unit tests, never in scientific exports. Preserve supplementary validation
failures and previous revision records.
