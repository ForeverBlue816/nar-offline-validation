# PrismQuant manuscript figures — revision 5

Scientific plots are generated with Python/matplotlib. User-specified presentation
choices take precedence over journal-style defaults. SVG/PDF axes and text remain
editable; dense scientific marks are embedded rasters.

## Current exports

- Figure 1: fig1a through fig1g, each in SVG/PDF/transparent 300-dpi PNG.
  fig1_preview.png and fig1.pdf/svg compose the seven bare panels.
- Figure 2: the existing fig2a/b/c and assembly, unchanged.
- Figure 3: fig3a (cloud), fig3b (energy), and fig3c (two-part range law).
  fig3c1 and fig3c2 are also exported independently in SVG/PDF/PNG.
  fig3_preview.png is the complete 2-by-2 review sheet; fig3c_preview.png is
  the side-by-side range-law comparison. No in-panel titles or footer captions.

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
