# Manuscript figure revisions

## 2026-09-13: Figure 2 original colors and lighter strokes

- Restored the original deep-blue PrismQuant, pale-cyan Hadamard and pale-pink DuQuant colors, including matching legend handles.
- Reduced main/comparison/reference widths from 2.3/1.85/1.5 pt to 1.5/1.05/0.85 pt, and softened marker edges and axes. Kept the current font sizes, layout, exact data and statistics.
- Refreshed main/standalone PDF, editable SVG, 600-dpi PNG, manuscript aliases, palette-sensitive checks and export QA. Figures 3, 4 and deployment retain their existing rendered appearance.


## 2026-09-13: unified Figures 2/3 and restored decoder-layer cost

- Restored Figure 4c from all six original decoder-layer cost/reference records, using restrained blue/teal colors. Kept the wider k=8 category highlight; Figure 4 is now 5.5 × 5.55 inches with a left-spanning accuracy panel. The E28 Graph-overhead experiment remains archived and is not conflated with this independently benchmarked cost statistic.
- Restyled Figure 2 at 5.5 × 2.55 inches with 2.3-pt main curves, 1.85-pt comparators and ≥7.5-pt labels, preserving every one of 252 values and both mean reductions.
- Rebuilt Figure 3 at 5.5 × 6.05 inches as a four-panel evidence grid. Kept every original geometry, rank-256 energy and reference-law observation. Added four existing depth-spanning rank-64 spectra with their different BOS protocol explicitly visible and a same-protocol seven-layer appendix.
- Merged activation, V-cache and multi-slot law observations; replaced internal experiment identifiers with scientific names. Kept all 5,342 experts in the main MoE panel, adding equal-count f-bin median/IQR summaries. Kept original reference coefficients; displayed full-row and all-expert R² together. Verified 263 high-f experts, 0.19 restricted-range R², 0.95 median y/x and 3.82× cold/hot residual spread.
- Corrected the holdout wording after inspecting the original collector: MoE observations are excluded from the dense reference fit, while retained expert rows come from expert calibration capture. No independent calibration/test split or causal cold-expert noise claim is made.
- Added traceable point/bin tables, source hashes, scientific captions and LaTeX snippets. Archived old figure files/scripts; active aliases and independent panel exports now point to the revised figures.
- Updated the legacy DuQuant gate to check complete frozen tables and all 56 numerical brackets directly because the retained DONE metadata lacks its former addendum field; measurements are unchanged.
- Both verification suites enforce embedded Times New Roman Bold, actual ≥7.5-pt text, editable vector SVG/PDF, 600-dpi PNG, full point counts, no clipping, zero collision findings and strict 1.5-pt rendered alignment. The deployment bundle has 21 assets / 63 exports; the Figure 2/3 bundle has 12 assets / 36 exports, each rendered twice identically.

## 2026-09-13: earlier visual clarity revision (superseded layout)

- Figure 4 now has only a and b, side by side. The former c and the preceding
  assembly are archived in `archive/fig4_before_two_panel_refinement/`.
  All measurement CSVs remain byte-for-byte unchanged.
- Enlarged b to the full plot height and widened the k=8 highlight from
  0.34 to 0.95 categorical axis units; it stops halfway before rank 16.
- Replaced opaque B1/B16 prefill labels with `batch 1` / `batch 16` and an
  explicit sequence-count axis description.
- Tightened deployment d to a labeled linear 0.25–1.05 range. All points,
  session extrema and the neutral reference 1.00 remain visible. Thicker
  segments and larger markers accompany ratio and percentage-reduction labels
  calculated from unrounded measurements.
- Updated captions, metadata, independent panels, compatibility copies and QA.
  Current bundle: 20 figures/panels, 60 PDF/SVG/PNG artifacts. The two original
  main-figure point tables and the full 30-comparison appendix remain intact.

## 2026-09-12: original deployment revision

Sources frozen at 6747960b96b0b03e626e98eaeeccf0d1abc34e79 and
results/e28_v2/20260912_a40_v2_full_int4. No experiments or kernels changed.

- Preserved all 22 E20 budget observations, two bf16 references, all 15 E11/E18
  rank observations, and all six original E17 cost/reference rows. The original
  CSVs and component exports are unchanged; the old assembly is archived.
- Figure 4 retains its left-spanning budget panel and two stacked right panels.
  Recovery is now percent, rank remains categorical, and c uses a separate x
  axis with private Graph overhead. The green k=8 band is confined to b.
  The horizontal bracket explains scale resolution; the vertical bracket's
  group-constant/null-space interpretation is in the caption. Every point stays
  at its original coordinate; 4.125 is displayed as 4.13 only in the tick label.
- Replaced the main E17 proxy-cost view with 1.96% / 2.35% measured additional
  Graph decode latency. Retained the full E17 view in the appendix with its
  original RTX PRO 6000 hardware and denominator definition.
- Added an E28 implementation strip showing X to both kernels, offline factors,
  FP16 B output, shared symmetric token quantizer, and CUTLASS INT4 GEMM.
- Added four efficiency panels and all 30 kernel ablations in the appendix.
  Preserved 3B decode being slower than FP16 and TC-B losing at T=1. Prefill
  uses the original eager cohort; decode/memory use matching private Graph
  records. No ratios cross backends, cohorts, modes, or commits.
- All labels use verified Times New Roman Bold. Main width is 5.5 inches;
  labels are 7.5–10.5 pt. Strokes, markers, Hadamard edges, and separation are
  strengthened without changing data. The decode y-axis extends to 40 ms to
  accommodate two-decimal labels and the 8B bracket at final font size.
- Added raw-value/provenance CSVs, source-derived captions, a LaTeX snippet,
  plotting dependencies, source/geometry/font audits, independent PDF/SVG
  renders, grayscale checks, and exact render reproducibility checks.
- Exported pure-vector PDFs, SVGs with editable text, and 600-dpi PNGs for
  both main figures, four appendices, and 15 standalone panels/strip.
  Compatibility fig4.pdf/svg/preview files are promoted only after QA.
- Kept the neutral deployment filename because the complete manuscript's
  numbering is unavailable. No Figure 5 number is hard-coded.

The plotting scripts never launch training, calibration, GPU benchmarks, or
model loading. The original generic QA utility was narrowly extended to
inspect shared modules and honor an explicit manuscript font family; geometry
thresholds remain unchanged. Generic Nature TIFF/89–183 mm warnings are
resolved by the user's requested PNG/5.5-inch contract and actual export checks.
