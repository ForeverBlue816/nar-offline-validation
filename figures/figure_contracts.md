# Figure contracts — revision 4

Python/matplotlib; serif; ticks 6 pt, axes and annotations 7 pt, legend 6.5 pt.
Editable SVG/PDF and exact physical sizes. Figure 1a–d use transparent
SVG/PDF and 300-dpi transparent PNG; their preview is annotation-free.

1. Figure 1 (image plate + quantitative traces): rotation suppresses raw
outliers; PrismQuant further reduces group range. a/b compare magnitudes in
identical numerical channel windows and one shared 0–40 height/color scale;
c/d compare every paired token/group range on one shared 0–8.92 scale; e/g/f
describe a single token. Panels a–d contain only axes, ticks, axis labels, and
the surface. Row 2 uses elev=18, azim=-62 and 0.9-pt surface lines. Range
median/mean/95th percentile belong in metadata only. The landscape statistics
must come from the drawn arrays; f/g must equal the exact selected c/d cells.
All samples are retained, including isolated PrismQuant peaks.

2. Figure 2 (quantitative grid): layerwise null-space capture accompanies lower
range and activation NMSE. Preserve all 28 paired layers and the existing
estimators. Descriptive diagnostic rows, no invented uncertainty intervals.
Shared boxed figure legend occupies a dedicated top strip clear of data.

3. Figure 3 (schematic-led composite): explain alignment with a free direction
and show how much energy a limited number of directions captures. a contains
two equal-scale covariance ellipses using the SAME layer/window as Figure 1;
b preserves the three measured rank-256 energy curves. Remove c from the
current manuscript exports but retain its source measurements in Git.
The 2D ellipse is a calibrated rigid-rotation illustration, not an exact
projection of the 8192D transform. Centered covariance and uncentered second
moment are distinct. Measured 128-channel ranges have their own linear scale;
never equate those to the 2D ellipse width. All such distinctions go in metadata
and the caption. Axis bounds include full transformed score extents and the
complete 2-s.d. ellipses, using percentile bounds only as an initial estimate.

QA: measured comparable plot rectangles within 1.5 pt; source, PDF glyph-size
and collision audits; inspect each rendered panel and complete composition.
