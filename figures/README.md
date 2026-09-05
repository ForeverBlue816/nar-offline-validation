# PrismQuant manuscript figures — revision 4

All scientific drawing uses plain Python/matplotlib. The method's display name
is PrismQuant; existing CSV/code keys retain `nar` / `nar_kmax`. Outputs include
individual panels, complete PDF/SVG figures, PNG previews, reproducible scripts,
source tables/arrays, metadata, captions, and rendered QA reports.

Current panels: `fig1a`–`fig1g`, `fig2a`–`fig2c`, `fig3a`–`fig3b`.
`fig3a` contains the two adjacent geometric views. Figure 3c has been removed;
its historical range-law observations remain in `fig3_range_law.csv`.

## Fixed palette and typography

| Role | Color |
|---|---|
| PrismQuant; axes and text | `#1D3557` (Deep Space Blue) |
| Hadamard | `#A8DADC` (Frosted Blue) |
| Raw / identity | `#457B9D` (Steel Blue) |
| DuQuant | `#E63946` (Strawberry Red) |
| Light fills / floor | `#F1FAEE` (Honeydew) |
| Pane edges | `#C9D6DF` |
| Pane grid | `#DCE4EA` |

Height map: `#F1FAEE → #A8DADC → #457B9D → #1D3557`.
Every 3D segment is colored by its local maximum; unlike the previous renderer,
a high point does not determine the color of an entire token-long polyline.
Figure 1a/b share the 0–40 height and color scale; c/d share the 0–8.92
height and color scale. Panels a–d are bare 3D exports with axes, ticks, labels,
and surface only. Three back panes, thin borders, and dense light pane grids are
enabled. Row 1 uses elev=22, azim=-60; row 2 uses elev=18, azim=-62. The box
aspect is 2.6:1.2:0.85. All tokens/channels/groups remain represented.

Actual font: DejaVu Serif. Ticks 6 pt, labels/annotations 7 pt, legends 6.5 pt;
upright axis labels. Standalone panels omit LaTeX panel letters and method titles. Figure 1a–d
also omit every scale/statistic annotation so slide captions can be added
externally. Individual 3D panels are 3.2 × 2.45 in and export transparent
SVG/PDF plus 300-dpi transparent PNG; traces remain 1.8 × 1.52 in with their
existing exports. Figure 2 panels are 1.85 × 1.72 in. The complete Figure 1
PDF/SVG and 300-dpi PNG are annotation-free review compositions.

## Figure 1: correctness findings and resolution

**The arrays and labels were not swapped.** The old whole-polyline max coloring
made dense regions and isolated peaks hard to compare. More fundamentally,
a smaller mean range does not imply smaller maxima or a smoother surface.
The new plot preserves the actual distributions. Median, mean, and 95th
percentile are computed from each complete array with float64 accumulation and
stored in `fig1_metadata.json`; they are not printed on panels a–d.

A further real bug was found: the old c/d upper limit was Hadamard's maximum
6.389837, whereas PrismQuant contains a maximum of 8.919331. The revised common
limit covers **both** datasets. No peaks are clipped or removed to make the
method appear flatter. `fig1_metadata.json` records this finding explicitly.

**The two range summaries have different populations.** c/d average all 32,768
cells (512 tokens × 64 groups) from sequence 118, tokens 160–671, layer-27
`down_input`. Their means are 1.764014 / 0.840427, a 52.36% reduction.
f/g are the single token-416, group-0 cells of those exact arrays, with ranges
1.707881 / 0.763428. The script asserts trace/cell equality. The raw trace is
group 25, selected by the frozen leading direction's peak loading. Full E1c
means (1.914824 / 0.885016) are a third, separately labeled population used
for site/layer selection. No means are silently substituted for trace ranges.

Both a/b now use numerical channel indices 2176–4223 with equal width 2048;
b is in the rotated basis. All 8192 channels contribute to the c/d group ranges.
The former trace label also called a group mean a zero point: the new dashed
line uses the actual experimental quantizer offset `fp16(min(values))`.
Group means and quantizer offsets are separately recorded in metadata.

## Figure 2: preserve measurements

All 28 layers and original mean reductions (25.30% range, 40.41% NMSE) are
unchanged. The 6.5-pt boxed legend sits in the entire figure's upper-right
margin, clear of every curve, with labels PrismQuant / Hadamard / DuQuant.
DuQuant remains the requested display name for `duquant_style` source rows;
this is the prior simplified diagnostic, not the official complete algorithm.

## Figure 3: calibrated geometry, with its limits stated

The previous geometry came from layer 1; the new geometry uses the same
layer-27 window as Figure 1. The **centered** sample covariance eigenvalues
are 97.5283125 and 57.3704360; the 2-s.d. semiaxes are calculated from them.
Their modest aspect ratio is a measured fact: the ellipse is not elongated
artificially. Every score is inside the shared frame; percentile-initialized
bounds expand to complete score/ellipse extents, with zero discarded rows.

The 2D rigid rotation illustrates exact alignment with the free (1,1)
direction. It is not claimed to be an exact projection of the full k=max
transform. The mapped centered-PC1 DC cosine and the frozen uncentered-v1 DC
energy are both recorded, so these different directions cannot be confused.
Actual raw and PrismQuant group ranges use their own common 0–12 scale below
the ellipses. Bracket lengths are measured values and are **not** the width of
the 2D covariance ellipse. All numerical/calibration choices are in
`fig3_metadata.json`. Figure 3b preserves the measured uncentered rank-256
energy curves from layers 1/13/27; no 64-slot line is drawn.

## Reproduce and audit

Use the existing Python environment with numpy, pandas, torch, matplotlib,
Pillow, and PyMuPDF. Re-render from the committed derived arrays:

```bash
python figures/make_fig1.py --reuse-data
python figures/make_fig2.py
python figures/make_fig3.py
python figures/verify_figures.py
python figures/audit_exports.py
```

To recompute the Figure 1 arrays from the frozen experiment artifacts, replace
its command with `python figures/make_fig1.py --workdir "$NAR_WORKDIR"`.
No model run or rotation training is needed. `fig1_source_arrays.npz` stores
all plotted landscape/trace arrays and centered covariance scores; the
original large activation dumps remain outside Git. Selection rules are
unchanged and recorded in metadata. These diagnostic figures do not imply
new seed-level confidence intervals.

Render-time alignment reports and final PDF text/collision audit results are
in `qa/`; human panel-by-panel review is summarized in `qa/README.md`.

The legacy layer-1 projection CSV and its geometry metadata are retained solely
as historical source provenance; the current Figure 3 reads the layer-27
arrays and `fig3_metadata.json`.
