# Figure contracts — revision 6

User-requested bare panels; upright serif, 6-pt ticks, 7-pt axis labels,
6.5-pt legends. Editable SVG/PDF text. No in-panel titles or footer captions.

1. Figure 1: density-selected activation illustration. Rank every contiguous
2048-wide window by count(median |x| > 1), earliest start breaks ties. Both a/b
use the same window and camera, with independent z/color scales 0–40 and 0–4.
Panels c/d share 0–10 limits and normalization, camera 18/−62, and 0.9-pt lines.
No sample removal or clipping. Report range statistics in metadata only.
The e/f/g signed traces each carry both axes, ticks and labels; equal sizes,
common y limits, measured brackets, actual zero-point line in g. Transparent
300-dpi PNGs accompany all seven PDF/SVG panels.

2. Figure 2: preserve all 28 paired Hadamard/PrismQuant layer measurements and
existing mean reductions. Add the E16 DuQuant-style rotation on frozen E1c
rows, seed 20260902, group 128. All measured layer/site ranges must pass the
paired bracket gate before plotting. Append original per-layer CSV rows
without modifying its existing byte prefix. Three series in b/c share the
existing palette and legend; each point has a source-CSV line. No invented
uncertainty intervals or new data exclusions. The 8B addendum requires its
currently missing frozen E1c dump; do not substitute a model rerun.

3. Figure 3: restore the historical scatter implementation from 721f253.
Panel a uses non-BOS tokens and frozen v1/v2 from the Figure-1 layer. Standardize
token projections; retain all observations within percentile-initialized,
full-extrema-expanded bounds. Overlay measured unit-direction projections,
recording their distinct coordinate meaning. Panel b retains all rank-256
spectra, log-x, direct labels, and no slot threshold line. Panel c comprises
two standalone source-population views, E1c versus E7/E20, with shared axes,
identity lines, and explicitly pooled OLS fit. Every scientific point remains
in the main plots. Insets only enlarge the declared 0.85–1.0 corner.

QA independently recomputes selection/statistics/fit, confirms full cloud and
range-law bounds, checks PDF glyph sizes and overlaps, and reviews final renders.

4. Figure 4: distinguish metadata scale resolution from aligned null-space
allocation, and show measured activation recovery alongside measured online
cost. Panel a uses all requested accounted E20 rows and measured bf16;
a second version uses 8B. Panel b uses activation-only E11/E18 v2 PPL means,
including direct supplementary E11 k64 measurements, and E17 v3 timing shares.
All five categorical recovery positions contain actual measurements. No
interpolation, smoothing, invented uncertainty, or substituted protocols.
Independent source-row and per-sequence-loss audits accompany each CSV.
The metadata-budget panel spans two stacked shared-x panels: b1 measures
recovery, b2 measures kernel cost, with a 62:38 plot-height ratio. All three
plot areas have the same width. The right stack shares its outer top and
bottom edges with a. Both framed legends remain below, aligned side by side;
the right-hand legend lists only the recovery models. The green deployed
band extends through b1 and b2. No x tick labels appear on b1. The two-line
interpretation is in b1's upper left; direct timing labels identify b2's
solid PrismQuant lines and dashed Hadamard references. The 4.1875 budget
tick is unlabelled, and the null-space label clears the (256,3) label.
Standalone b1/b2 canvases are 2.7 inches wide, with heights 2.604/1.596
inches, summing to a's height. The assembled right column remains 2.7 by
4.2 inches including the shared legend. Every plotted number is preserved
in the per-panel source CSVs; no new measurements or exclusions are needed.
