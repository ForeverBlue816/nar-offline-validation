# PrismQuant manuscript figures — revision 6

Scientific plots are generated with Python/matplotlib. User-specified presentation
choices take precedence over journal-style defaults. SVG/PDF axes and text remain
editable; dense scientific marks are embedded rasters.

## Current exports

- Figure 1: fig1a through fig1g, each in SVG/PDF/transparent 300-dpi PNG.
  fig1_preview.png and fig1.pdf/svg compose the seven bare panels.
- Figure 2: fig2a/b/c and assembly; Hadamard, DuQuant, and PrismQuant in all three panels.
- Figure 3: fig3a (cloud), fig3b (energy), and fig3c (two-part range law).
  fig3c1 and fig3c2 are also exported independently in SVG/PDF/PNG.
  fig3_preview.png is the complete 2-by-2 review sheet; fig3c_preview.png is
  the side-by-side range-law comparison. No in-panel titles or footer captions.

## Figure 2 DuQuant addendum

Panels b/c now include the measured DuQuant-style diagnostic in #E63946.
The 3B down-input main figure retains the existing Hadamard/PrismQuant
measurements, capture values, sizes, axes, palette, and shared three-method
legend. `fig2b.csv` and `fig2c.csv` each contain 84 plotted values with exact
physical source-CSV line numbers and source method/site/layer keys.
The displayed percentage reduction still compares PrismQuant with Hadamard.

The 3B addendum applies E16's existing `e11._duquant_blocks` construction to
all 8192 original E1c evaluation rows at both sites and all 28 layers.
Block size is 128; seed is 20260902, with the original site/layer offsets.
The descending-absmax zigzag permutation uses the frozen E11 channel scores,
and the original seeded QR construction builds each block rotation.
No model is loaded or rerun. Computation ran on an allocated CPU node;
parallel exact bf16 row reads reduce network-file latency without changing
which rows are evaluated. Hadamard was replayed on every layer/site to
check agreement with its frozen range and NMSE. `experiment.quant_metrics`
computes the exact same range, global relative error, and paired deltas.

The 56 new `method=duquant` rows are appended to the original
`results/llama32_3b/e1c_per_layer.csv`. Every byte of the existing CSV prefix
is preserved. `E1C_DONE.json` records an independently timestamped
`duquant_addendum`, construction/source hashes, row count, and checks.
`e1c_duquant_sanity.csv` records every paired range comparison, sampled-row
hash, permutation hash, and measured null-space capture. The renderer refuses
to plot if the DuQuant addendum is incomplete or outside the requested
paired Hadamard/PrismQuant bracket (relative numerical tolerance 2e-5).

The requested 8B offline addendum is **not complete**: no frozen 8B
`wide_cal_a` dump, `e1c_per_layer.csv`, or `E1C_DONE.json` exists in the checked
repository or `/projects/nar/nar-validation` asset root. The existing 8B E16
aggregates and E11 factors do not contain the individual activation rows
needed to reconstruct range or NMSE. The missing-input record is in
`fig2_duquant_inputs.json`. No 8B data is fabricated and no model rerun is
substituted for the requested frozen-dump diagnostic. An existing 8B frozen
dump location is needed to complete that part.

```bash
python figures/make_fig2.py
python figures/verify_fig2_duquant.py
python figures/audit_exports.py --figure 2
```

`measure_fig2_duquant.py` performs the offline addendum. It accepts the same
asset and frozen-code roots as the Figure 4 measurement helper, supports CPU
or CUDA, and resumes complete per-layer checkpoints. Original E1c summaries
are retained; the addendum lives in the per-layer CSV and done metadata.

## Figure 4

`fig4a` (3B metadata budget), `fig4b` (recovery and kernel cost), and
`fig4a_8b` (appendix) are separate 2.7-by-4.2-inch SVG/PDF/300-dpi PNG panels.
`fig4_preview.png` and `fig4.pdf/svg` combine the two main panels. The figures
contain no titles, footers, protocol notes, or missing-data notices. The
requested short deployment interpretation appears inside panel b.

The budget panels retain every requested existing E20 configuration, with
mean PPL recomputed from three seeds on 64 chunks and effective bits
`4 + 16*(m+1)/g`. The unaccounted fp32 coefficient controls are excluded.
Each panel's CSV lists the exact summary-file row and physical CSV line for
every point and the bf16 reference. Bracket endpoints are those same plotted
points; their source lines and PPL differences are in `fig4_metadata.json`.

The revised presentation keeps both columns at 2.7 by 4.2 inches. Panel a
spans the two stacked panels at right: b1 shows recovery, b2 shows kernel
cost. Both right-hand axes have the same plot width as a, share their x
limits, and use a 62:38 ratio of plot heights. Their combined top and bottom
edges align with a. The two legends occupy a dedicated bottom row with
matching rectangular frames and 6.5-pt type. Only the three model entries
appear in the right-hand legend; b2 identifies the timing models directly.

The separate bare exports `fig4b1` and `fig4b2` have widths of 2.7 inches
and heights of 2.604 and 1.596 inches (62:38, summing to a's 4.2 inches).
They preserve the marks and font sizes while giving each panel its own
margins. `fig4b` retains the assembled right column with its shared legend.
`fig4b1.csv` and `fig4b2.csv` partition the unchanged `fig4b.csv`, preserving
source rows for all 15 recovery points, four kernel points and two references.

Budget tick labels are horizontal; the 4.1875 tick remains without text.
The scale-resolution bracket spans the Hadamard (256,1)/(128,1) pair with
its label above and no leader. The null-space bracket lies at x=4.25; its
label sits to the right, clear of the (256,3) point label. A small stroke
gap at the intervening measured triangle avoids crossing the marker.
The 3B lower bound stays at 7.60, and the measured bf16 line at 7.61675.
The 8B appendix uses the same presentation.

The green deployed band extends through both right-hand axes. Recovery
labels (0.39, 0.36, 0.59) sit to the left of k=8, staggered for clearance.
The two-line interpretation sits at the upper left of b1. Only b2 carries
x tick labels and the shared x-axis label. Its two solid red diamond lines
connect measured k=8 and k=32 costs, with percentages beside the markers
and model names at the ends. Dashed Hadamard references have right-aligned
labels. All source CSV numbers remain unchanged by this layout revision.

The recovery panel evaluates `(mean PPL_Hadamard - mean PPL_k) /
(mean PPL_Hadamard - PPL_bf16)`, directly from PPLs. It does not reuse legacy
recovery columns with different averaging or rotation-only corrections.
Llama uses the E11 activation-only protocol (64 chunks; seeds 20260902,
20260903, 20260904). Qwen uses the existing E18 v2 activation-only results
(146 chunks; seed 20260902). All five requested categories are measured for
all three models; there are no missing k values and no interpolated points.
The per-site maxima (qkv/down) are 24/64 for 3B, 32/112 for Llama 8B,
and 32/96 for Qwen 8B. At each nominal k, the actual site value is capped
by that site's maximum. The CSV records both values.

The previously absent Llama k=64 rows were measured on September 5, 2026,
using the original E11 code at `f424b82`, original test tokens and rotation
seeds, bf16 weights/KV, and group-128 activation quantization at both sites.
The frozen E11 eigenvectors were recovered by inverting its b64 Householder
factors, checked against the stored b128 k32 reflectors, then used for a
fresh original-protocol 128-sequence permutation calibration. There is no
new eigensolver fit. Raw losses, factor/basis audits, summary PPLs, and device
and token-hash provenance are in `results/<model>/e11_k64_*` and
`E11_K64_DONE.json`. Original E11 tables are preserved.

Kernel costs come from E17 v3's `*_share_of_layer` columns:
`100 * kernel_ms / (decoder_layer_ms + kernel_ms)`. The k=8 timing row
supplies each model's single horizontal Hadamard reference. PrismQuant has
only the measured k=8 and k=32 timing points; no costs are estimated at other
k values. `fig4b.csv` carries separate source files and exact CSV line numbers
for every recovery value, each baseline, and each kernel measurement.

To reproduce the exports and independently verify raw-loss and CSV linkage:

```bash
python figures/make_fig4.py
python figures/verify_fig4.py
python figures/audit_exports.py --figure 4
```

To repeat a missing-k experiment, export the `nar/` directory from commit
`f424b82` into a separate code directory, then run on an allocated GPU:

```bash
python figures/measure_fig4_k64.py --model llama32_3b \
  --repo "$PWD" --assets "$NAR_WORKDIR" --scratch "$FIG4_SCRATCH" \
  --code-root "$FIG4_FROZEN_CODE"
```

Use `llama31_8b` for the other model. Existing completed measurements are
reused. The required frozen factors and model cache reside in the asset root.

## Figure 1

All contiguous, stride-1, 2048-channel windows are ranked by the number of
channels whose median absolute activation over the 512 displayed tokens exceeds
1.0. The best count is 21. Of 183 tied best windows, the earliest starts at 1254;
a and b therefore both show channels 1254–3301. The metadata records the rule,
count, tie handling, and all 6145 candidate-window counts are preserved in the
source NPZ together with all 8192 channel medians. This is a density-selected
illustration, not an average-case estimate.

Panel a uses z/color limits 0–40; the selected window's maximum is 18.125.
Panel b uses its own 0–4 z/color scale. Both use elev=22, azim=-60 and identical
channel/token ticks. Panels c/d use the same 0–10 height/color scale, elev=18,
azim=-62, and 0.9-pt lines. The aspect ratio is 2.6:1.2:0.85. Grids are denser
than the labeled ticks. No values are clipped, subsampled, or smoothed.

Panels a–d contain only axes, tick labels, axis labels, and data. Median, mean,
and 95th-percentile ranges are recorded in fig1_metadata.json. The 32768 range
cells and all three signed traces remain numerically identical to revision 4.
The f/g traces are exact token-416, group-0 cells of c/d; e is raw group 25.

Panels e/f/g have equal 2.1-by-1.75-inch canvases, common y limits, y ticks
−5/0/5, and x ticks 0/32/64/96/127. Every trace has both “signed value” and
“channel in group” labels. Range brackets remain; g retains the actual affine
INT4 offset fp16(min x) as its dashed zero-point line. Panels a–d remain
3.2 by 2.45 inches.

## Figure 3

make_fig3.py was restored from the scatter implementation at commit 721f253,
then updated. Its ellipse replacement is no longer rendered.

Panel a projects 8064 non-BOS, stride-32 tokens from layer 27 onto the same
frozen uncentered second-moment v1/v2 basis used by Figure 1. Each coordinate
is centered and divided by its own sample standard deviation (ddof=0).
The 0.5–99.5 percentile frame is expanded to the full observed extrema and
padded 8%, so every point lies inside it. Metadata records both frames.

The arrows show the unit receiving-group DC direction pulled back through the
full-width Hadamard and frozen PrismQuant transforms, projected onto v1/v2.
Their lengths are 0.0150056 and 0.9999949. Arrow coordinates use unit-direction
cosines overlaid on the standardized token cloud; their lengths are not token
standard deviations. Their common plotting multiplier is 1.0. The sign of a
null direction is arbitrary and is oriented toward positive v1.

Panel b retains the existing rank-256 energy measurements for layers 1/13/27.
The x axis is logarithmic from 1 to 256. Layer 27 uses deep blue; the other two
use frosted blue with solid/dashed lines and direct endpoint labels.

Panel c is split by source population: fig3c1 contains all 2520 E1c activation
points; fig3c2 contains all 280 E7 V-cache and 112 E20 multi-slot points. Both
use identical axes, the dashed identity line, and the same thin pooled OLS fit.
Both explicitly label “Pooled R² = 0.86”: this is not a fit estimated within
either subpanel. The exact pooled fit is y = 0.0598022 + 0.8665148 x,
R² = 0.8613894973, with x = sqrt(1-f). Small upper-left insets enlarge the
0.85–1.0 corner. Every point remains in its main plot; inset filtering only
selects the zoom region. The measured range-law table is unchanged.

Standalone Figure 3 panels are 2.65 by 2.35 inches with 600-dpi PNGs.
Combined review PNGs are 300 dpi. The paired fig3c canvas is 5.3 by 2.35 inches.

## Palette and type

Height map: #F1FAEE → #A8DADC → #457B9D → #1D3557.
Text/PrismQuant: #1D3557; raw/cloud/E7: #457B9D; Hadamard/E20: #A8DADC.
Pane edges: #C9D6DF; grids: #DCE4EA. All labels use upright DejaVu Serif,
6-pt ticks, 7-pt axis labels, and 6.5-pt legends.

## Reproduce

Use the existing environment with numpy, pandas, torch, matplotlib, Pillow,
and PyMuPDF. From the committed derived arrays and tables:

```bash
python figures/make_fig1.py --reuse-data
python figures/make_fig3.py
python figures/verify_figures.py
python figures/audit_exports.py
```

To refresh from the frozen activation artifacts, first run:

```bash
python figures/make_fig1.py --workdir "$NAR_WORKDIR"
python figures/prepare_fig3.py --workdir "$NAR_WORKDIR" --geometry-only
python figures/make_fig3.py
```

The geometry-only route reuses the frozen layer-27 basis and factor. It does
not rerun a model or eigensolver. The old fig3_eigvecs_layer1.npz is retained as
historical provenance and is not used by the current renderer.

Scientific linkage, data-integrity comparisons, and rendered export audits are
in qa/. Figure 2's measurements and files are preserved.
