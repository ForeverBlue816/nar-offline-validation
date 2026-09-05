# Figure contracts — revision 5

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

2. Figure 2: preserve all 28 paired layer measurements and existing mean
reductions. No invented uncertainty intervals or new data exclusions.

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
