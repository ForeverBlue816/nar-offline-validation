# Final rendered QA — revision 3

Numerical regression checks: PASS (`numerical-verification.json`).
All 15 PDFs (12 standalone panels + 3 full figures) have editable text, embedded
fonts, and a rendered minimum size of 6 pt. No PDF has a clipping, text–text,
or text–stroke FAIL in the final audit. Individual panels and Figures 2/3
also have zero WARNs. All complete-figure comparable-panel alignment gates
pass at 1.5 pt with no exemptions. Figure 3's energy axis is deliberately
outside the ellipse/range comparison groups: its units and aspect ratio differ.

Figure 1's composed PDF has 16 ambiguous fill/image-edge WARNs. Each was
reviewed against the actual rendered PDF, not merely its separate PNG preview:

- 11 fill-edge WARNs concern a/b channel labels, c/d group labels, all three
  range numbers, and the four token-600 tick labels. Source-page background
  and pane bounding rectangles in the composed PDF create these candidate
  overlaps; the actual text is clear and does not cross a visible edge.
- 5 image-edge WARNs concern a/b channel labels, d's group label, d's group-40
  tick and d's token-400 tick. The rectangular bounds of the rasterized 3D
  marks include transparent regions. These text items occupy clear space
  outside the visible data marks. The original standalone PDFs have 0 WARNs.

The raw composed-PDF verdict remains REVIEW REQUIRED; the 16 items are
resolved by the documented visual inspection above. No global tolerance was
relaxed, no WARN was recast as an automatic PASS, and no data was masked.

## Panel-by-panel review

All panels use the requested method mapping and 6/7/6.5-pt typography.
No stochastic uncertainty interval is inferred for these fixed diagnostic
measurements. Data exclusions are the inherited, disclosed layer/window
selection and BOS rule; no extra rows are removed for appearance.

| Panel | Evidence role / statistic | Unit and spread | Visual and rendered result |
|---|---|---|---|
| 1a | Raw activation magnitude | 512 tokens × 2048 channels; every value | Full box/grid, local scale, identical a/b channel interval; pass |
| 1b | Hadamard magnitude | Same token/channel counts; rotated coordinates | Independent stated scale; no label crop; pass |
| 1c | Hadamard quantizer range | 32768 token/group cells; mean 1.764 | Complete shared height range; array-derived annotation; pass |
| 1d | PrismQuant quantizer range | Same cells; mean 0.840 | Lower overall ranges AND larger isolated extrema retained; pass |
| 1e | Raw signed trace | One token × 128 channels; min/max bracket | Actual range, shared signed-value scale; pass |
| 1f | Hadamard signed trace | One selected cell of 1c | Exact trace/cell agreement; bracket avoids zero baseline; pass |
| 1g | PrismQuant signed trace | One selected cell of 1d | Actual fp16(min) zero point, not mean; pass |
| 2a | Null-space energy fraction | All 28 layer rows for each of 3 methods | Shared figure legend separated from curves; pass |
| 2b | Mean group range | 28 paired layer rows | Original mean reduction 25.30%; pass |
| 2c | Activation NMSE | 28 paired layer rows | Original mean reduction 40.41%; pass |
| 3a | Calibrated 2D alignment illustration | Covariance ellipse, 512 scores, 2-s.d. semiaxes; not a CI | Both complete ellipses and all score extents inside common axes; measured ranges separately scaled; pass |
| 3b | Spectral energy capacity | 256 measured ranks in each of 3 layers | All curves, distinguishable line styles, no 64-slot marker; pass |

## Source preflight interpretation

The unmodified source checker is designed for Nature-style defaults and
recognizes only sans-serif font names. This user explicitly requires serif.
Its FONT-FAMILY finding is therefore a documented user-instruction override;
we verified the actual embedded DejaVu Serif font. The checker was not edited.

Source-closure reports inspect each plot script together with its imported
shared style/export helper, avoiding false reports of missing SVG/PDF export,
editable-text configuration, or alignment integration. No other source FAIL
remains. `summary.json` records the raw checker return codes and resolution.
Other source warnings are reviewed as follows: TIFF is not requested (PNG at
600 dpi and vector PDF/SVG are supplied); the complete Figure 1 PNG is a
300-dpi preview; requested standalone widths differ from generic Nature
89/183-mm defaults; and a deterministic random-sign seed does not imply
multiple-seed aggregates needing error bars. Full raw data are preserved in
the committed derived arrays, with an independent covariance/range check.

## Re-run

`python figures/audit_exports.py` runs installed source/PDF auditors and writes
fresh reports. Supply `--qa-tools` if their installation is elsewhere.
`python figures/verify_figures.py` independently checks scientific linkage and
export completeness. Regeneration already runs the bundled alignment gate.
