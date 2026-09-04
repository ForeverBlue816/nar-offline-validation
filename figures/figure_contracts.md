# Figure contracts

All paper panels use the fixed PrismQuant palette in `figure_style.py`, plain
Python/matplotlib, editable SVG/PDF and 300-dpi PNG. Panels are exported at
their final physical width with no in-file panel letter, title, or footer;
LaTeX supplies subcaptions and panel letters.

## Figure 1 — outliers, quantizer range, and the null-space mechanism

- Core conclusion: ordinary rotation flattens raw activation outliers, while
  PrismQuant specifically lowers the per-group range that sets the asymmetric
  INT4 step by moving the leading persistent direction toward a groupwise
  zero-point-removable plateau.
- Results-level question: does PrismQuant improve what the quantizer actually
  sees, rather than merely making a full-tensor activation landscape look flat?
- Archetype: image plate plus paired quantitative traces.
- Evidence chain: a–b establish raw-versus-Hadamard outlier flattening; c–d
  compare the exact paired token-by-group range seen by the quantizer; e–g show
  raw concentration, signed Hadamard mixing, and the PrismQuant plateau for one
  deterministic non-BOS token.
- Source and selection: frozen Llama-3.2-3B E1c rows. The site/layer maximizes
  the absolute measured mean-range decrease of `nar_kmax` versus
  `hadamard_full` over both activation sites. The hero is the stride-32 non-BOS
  row nearest the 95th percentile of absolute projection on frozen v1, inside
  the requested top decile. The trace group receives v1 under the frozen
  PrismQuant transform.
- Integrity: a and b use independent z limits; c and d share c's limit. The
  transformed landscapes use a matched 2,048-channel window and c/d use all 64
  groups across the same 512 tokens. No clipping or data adjustment is allowed.

## Figure 2 — per-layer causal chain

- Core conclusion: layer-wise null-space energy placement is accompanied by
  lower group range and lower dynamic INT4 activation NMSE.
- Results-level question: does the proposed mechanism track both the
  intermediate range and final quantization error across layers?
- Archetype: quantitative grid exported as three independent 1.85-in panels.
- Evidence chain: a measures null-space energy against Hadamard and DuQuant;
  b measures paired range; c measures paired activation NMSE.
- Integrity: all points remain measured paired rows in `fig2_capture.csv`; no
  unexplained secondary series, interpolation, or model/site mixing.

## Figure 3 — geometry, capacity, and predictive law

- Core conclusion: PrismQuant aligns a leading activation direction with a
  quantizer null-space direction, and residual energy predicts measured range
  across the existing diagnostic families.
- Results-level question: is the mechanism visible geometrically and does its
  square-root energy law hold beyond one activation site?
- Archetype: three complementary quantitative panels, each 1.85 in wide.
- Evidence chain: a shows input-space geometry; b shows the retained
  eigenspectrum; c tests the pooled law with source-family-specific points and
  an identity reference.
- Integrity: panel c retains every measured E1c, E7 and E20 point; dense marks
  use alpha and rasterization rather than exclusion.
