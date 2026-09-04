Naming: the method is PrismQuant in the paper. Code, CSV method columns and artifact filenames retain the development name 'nar' (nar_k8 = PrismQuant k=8, nar_kmax = PrismQuant k=max, etc.). The development abbreviation was retired because 'NAR' is the established abbreviation for non-autoregressive in NLP.

# Manuscript figures

Figures 1–3 use plain Python/matplotlib and the single shared style module
`figure_style.py`. Each manuscript panel is exported at its final physical
size as SVG, PDF, and 300-dpi PNG. Panel letters, method titles, and footer
captions are deliberately absent because LaTeX supplies them.

## Fixed paper palette

- PrismQuant: `#004C94`; light PrismQuant accent: `#3C93FA`
- Hadamard: `#73CC80`
- DuQuant: `#0FA69D`
- raw / identity: `#52647A`
- bf16 / reference and neutral text/axes: `#2D3F54`
- sequential 3D map: `#F2F4F6 → #3C93FA → #004C94`
- signed diverging map: `#73CC80 → #F2F4F6 → #004C94`, centered at zero

The requested Times/Liberation Serif fonts are not installed on this machine;
matplotlib resolves the declared serif stack to **DejaVu Serif**. Rendered text
is 7 pt for ticks/legends, 8 pt for axis labels, and 7.5 pt for annotations.
No panel contains an in-file letter or footer.

Previously committed A/B palette comparisons and their audit files are retained because they were explicitly requested not to be deleted. They are archival only; the current manuscript inputs are exactly `fig1a`–`fig1g`, `fig2a`–`fig2c`, and `fig3a`–`fig3c`.

## Figure 1

- Files: `fig1a`–`fig1g` in SVG/PDF/PNG; `fig1_preview.png` is checking-only.
- Data: frozen Llama-3.2-3B E1c `down_input`, layer 27. This case maximizes the
  absolute measured mean-range decrease over both sites: 1.914824 under
  Hadamard versus 0.885016 under PrismQuant k=max (−53.78%).
- Selection: hero sequence 118, token 416, the non-BOS stride-32 row nearest
  the 95th percentile of absolute projection on frozen v1. The plotted token
  window is positions 160–671. Raw channels are 2176–4223 around v1's largest
  loading; transformed channels are 0–2047 around the group receiving v1.
- Quantizer view: panels c/d include all 64 groups and share Hadamard's z limit.
  On the plotted 512-token window the mean range is 1.7640 versus 0.8404
  (−52.36%); the p90 reduction is 49.28%.
- Source tables: `fig1_ranges.csv`, `fig1_landscape_channels.csv`, and
  `fig1_metadata.json`. Large frozen dumps remain outside Git.

## Figure 2

- Files: `fig2a`–`fig2c` in SVG/PDF/PNG; `fig2_preview.png` is checking-only.
- Data: measured per-layer Llama-3.2-3B down-input rows in `fig2_capture.csv`.
- Panel a retains the thin G/d reference without an in-panel label and uses the
  display name DuQuant. Panel b contains no secondary dotted series: the old
  gray line was a percent reduction, not raw range, so it was removed.
- Mean per-layer PrismQuant reductions are 25.30% for range and 40.41% for NMSE.

## Figure 3

- Files: `fig3a`–`fig3c` in SVG/PDF/PNG; `fig3_preview.png` is checking-only.
- Panel b omits the former 64-slot line. Panel c keeps all 2,912 measured rows:
  2,520 E1c activation points, 280 E7 V-cache points, and 112 E20 multi-slot
  points. The pooled fit is `range_ratio = 0.0598 + 0.8665 sqrt(1-f)`, R²=0.861.
- The upper-right inset is included because 365 observations lie above 0.90 on
  both axes. Source files are `fig3_token_projections.csv`,
  `fig3_eigenspace_r256.csv`, `fig3_range_law.csv`, and metadata JSONs.

## Reproduce

```bash
python figures/make_fig1.py --workdir /projects/nar/nar-validation
python figures/make_fig2.py
python figures/make_fig3.py
```

Before commit, all 13 standalone PNGs were inspected at their native 300-dpi
pixel dimensions. The 3D panels use 3.2-in canvases, full-frame Axes3D,
`box_aspect=(2.4, 1.3, 0.9)`, view `(24°, −58°)`, no pane fill, and one 0.7-pt
polyline per channel/group. Figure 2 and 3 panels are 1.85 in wide; Figure 1
trace panels are 1.8 in wide.
