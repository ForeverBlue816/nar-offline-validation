# Figure 4 and deployment-efficiency revisions

## 2026-09-13: visual clarity revision

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
