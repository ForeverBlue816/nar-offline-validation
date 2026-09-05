# Final rendered QA — revision 5

Independent scientific/export verification: PASS (numerical-verification.json).
The saved range and trace arrays, rank-256 energy table, and range-law table
remain identical to commit 91b2ffe (final-data-integrity.json).

All 18 audited PDFs have embedded fonts, editable text, and no rendered text
below 6 pt. The final audit reports zero clipping, text-text, or text-stroke
FAILs. Figure 1a/b/e/f/g and Figure 3a/b/c1/c2 have zero WARNs. Figure 2 remains
unchanged. Render-time alignment checks pass; the two range-law subpanels
have identical comparable plot rectangles. The energy plot reserves room
for direct labels and is not treated as geometrically comparable to the cloud.

## Resolved visual-review findings

The unmodified collision checker retains the following ambiguous warnings:

- Figure 1c/d: three fill-edge WARNs each for group, group tick 40, and token
  tick 400. Transparent raster/pane bounds intersect the text rectangles;
  the actual visible surfaces and grid lines do not obscure the labels.
- Combined Figure 1: 12 fill/image-edge WARNs involving transparent surface
  bounds and channel/group/token labels. These are composed bounds rather
  than actual opaque marks at the text locations.
- Combined Figure 3c: three image-edge WARNs; combined Figure 3: six. These
  concern labels next to the transparent point-cloud and inset raster
  rectangles. All original Figure 3 standalone panels have zero WARNs.

The actual composed PDFs were rasterized and visually inspected after the
last render. All flagged labels are clear. Raw REVIEW REQUIRED verdicts are
retained, with this visual resolution; no tolerance was relaxed and no data
were masked to suppress the audit.

## Scientific checks

- Recomputed all 6145 candidate peak-density counts from the stored full-width
  channel medians; start 1254, count 21, earliest of 183 ties.
- Confirmed the selected raw/Hadamard values fit the independent 0–40/0–4
  axes, and every c/d value fits the shared 0–10 axis.
- Recomputed median, mean, and 95th-percentile ranges from complete arrays.
- Confirmed exact trace/cell equality, fp16(min) zero-point, complete ticks,
  both axis labels, equal trace sizes, and transparent 300-dpi Figure 1 PNGs.
- Confirmed all 8064 layer-27 projection rows exclude BOS, use an orthonormal
  frozen basis, and lie inside the recorded expanded frame.
- Confirmed PrismQuant's unit direction is aligned with v1 to numerical
  tolerance; the measured Hadamard in-plane length is 0.0150056.
- Recomputed x=sqrt(1-f), the pooled fit, R²=0.8613894973, and all 2912 source
  counts. Source populations are disjoint between c1 and c2; both label the
  fit as pooled, and both main plots include their complete observations.

## Style overrides and reproduction

The user explicitly chose upright serif, 6/7/6.5-pt typography and the supplied
palette. Generic Nature font/width/TIFF defaults are not the design authority.
The source checker itself is unchanged; its serif FONT-FAMILY finding is
resolved in summary.json. Other source diagnostics are advisory for the
requested export formats and dimensions.

Run python figures/verify_figures.py and python figures/audit_exports.py after
regeneration. The latter refreshes source-closure, PDF text, collision, and
composition reports. See figures/README.md for the frozen-data commands.
