# Figure contracts

All final paper labels use **PrismQuant**. Internal `nar` method values and artifact names remain unchanged for reproducibility. Main-text figures are 5.5 in wide and use the shared serif typography and palette audit defined in `palette.py`.

## Figure 1 — persistent outliers become removable common mode

- Core conclusion: PrismQuant converts token-persistent activation outliers into groupwise common-mode plateaus, so the asymmetric quantizer's zero point removes most—but not all—of their range.
- Results-level question: What does zero-point-aware alignment do to a persistent down-projection outlier before and after group-mean removal?
- Figure archetype: Quantitative image plate + paired trace controls.
- Target/output: Manuscript figure, Python/matplotlib only, 5.5 in wide; editable SVG/PDF plus 300 dpi PNG and plotted-data CSV.
- Panel map: a–d are equal-size signed all-channel 3D landscapes (raw, random-sign block-H128, PrismQuant k=max, and PrismQuant after group-zero-point removal); e–h are paired 128-channel traces for one deterministic non-BOS token.
- Evidence hierarchy: a–d establish the token-persistent wall, distinguish random mixing from alignment, and expose the quantizer null-space operation. e–h report exact signed values, group means and ranges, retaining the non-zero PrismQuant residual.
- Statistics/source data: Frozen Llama-3.2-3B E1c down-input activations, layer 1, 1,024 stride-32 token rows after excluding sequence position 0, all 8,192 channels. The strongest persistent channel maximizes median absolute activation over these non-BOS rows; the displayed token maximizes absolute activation on that channel. No uncertainty interval is applicable to this representative paired view.
- Integrity notes: All z values remain signed. Panel a uses its own symmetric z limit; b–d share b's symmetric z limit. Color normalization is separately symmetric about zero for every panel. No global color clipping, simulated data, or convenience downsampling of channels is allowed; dense 3D line art may be rasterized inside the vector container.
- Reviewer risk: A BOS spike, global clipping, or a trivially zero residual could manufacture the mechanism. BOS is excluded explicitly, the transform/group/token selection is deterministic and exported, and the residual panel reports the real non-zero range.

## Figure 2 — layer-wise causal chain from null-space filling to quantization error

- Core conclusion: Across layers, greater placement of activation energy in the quantizer null space lowers dynamic INT4 group range and thereby lowers activation NMSE.
- Results-level question: Does the proposed null-space mechanism track the measured range and quantization-error reductions layer by layer?
- Figure archetype: Three-panel quantitative causal chain.
- Target/output: Manuscript figure, Python/matplotlib only, 5.5 in wide; editable SVG/PDF plus 300 dpi PNG and plotted-data CSV. Llama-3.2-3B is the main figure; Llama-3.1-8B versions follow in the appendix after the audited gaps are measured.
- Panel map:
  - a: Fraction of total activation energy in the quantizer null space for Hadamard, DuQuant-style and PrismQuant k=max, with the random-rotation reference G/d.
  - b: Measured mean group range for Hadamard and PrismQuant, with the paired per-layer reduction and its across-layer mean.
  - c: Measured asymmetric INT4 activation NMSE for Hadamard and PrismQuant, with the paired per-layer reduction and its across-layer mean.
- Evidence hierarchy: The a→b→c layer-wise chain is the hero evidence; no panel alone establishes the mechanism. The second activation site and 8B outputs are replication evidence. DuQuant-style in a is the control for generic outlier rotation.
- Statistics/source data: Every point is computed from real paired activation rows under fixed transforms and group size 128. Frozen E1c rows and results are reused for 3B range/NMSE. Measured whole-activation f is added for both models. The 8B E1c diagnostic is run once with the released 3B settings (stride-32 positions, seed 20260902); no tuning, interpolation, cross-model copying, or synthetic completion is allowed.
- Integrity notes: Panel a uses f = ||P_DC R X||² / ||X||² over evaluated activation rows, not top-direction capture s_i. Percent reductions in b/c are paired within layer. The source CSV records model, site, layer, method, row count, group count and metric definitions.
- Reviewer risk: Replacing f with top-eigenvector alignment, mixing models/sites, or inferring missing 8B points from 3B would overstate the mechanism. The plotting script asserts full layer coverage and finite measured values; missing statistics block rendering until the real pipeline completes.

## Multi-panel audit

| Figure/panel | Scientific question | Evidence role | Decisive comparison | Unique inference gain | Destination |
|---|---|---|---|---|---|
| 1a–d | What does the transform do across tokens and all channels? | Representative mechanism | raw vs H128 vs PrismQuant vs PrismQuant−mean | Establishes persistence and the common-mode wall | Main |
| 1e–h | Is the apparent plateau quantitatively removable but non-trivial? | Paired control | exact group means and ranges | Prevents a color-scale-only or zero-residual story | Main |
| 2a | Is energy actually placed in the quantizer null space? | Mechanistic measurement | Hadamard/DuQuant-style/PrismQuant vs G/d | Measures the proposed cause | Main |
| 2b | Does null-space filling reduce the quantity that sets the step size? | Intermediate validation | Hadamard vs PrismQuant range | Links mechanism to quantizer range | Main |
| 2c | Does lower range reduce quantization error? | Endpoint validation | Hadamard vs PrismQuant NMSE | Completes the causal chain | Main |
| 2 appendix | Does the chain repeat across site/model? | Replication | q/k/v and 8B | Bounds generality without mixing models | Appendix |

## Figure 3 — geometry and predictive law of null-space alignment

- Core conclusion: PrismQuant makes the leading second-moment direction an available groupwise null-space direction; the number of such slots limits the absorbed spectrum, and the resulting range follows the pre-registered square-root energy law across activation and V-cache diagnostics.
- Results-level question: Is the mechanism visible in input-space geometry, capacity-limited in the eigenspectrum, and quantitatively predictive across the existing paired experiments?
- Figure archetype: Three-panel mechanism-to-law quantitative composite.
- Target/output: Main-text Llama-3.2-3B down-input figure at 5.5 in, Python/matplotlib only; editable SVG/PDF, 300 dpi PNG, plotted-data CSVs and reconstruction metadata.
- Panel map:
  - a: Non-BOS calibration tokens projected onto recomputed `(v1, v2)`, with `R^T e_null` for the matched Hadamard and PrismQuant group. Arrow length is the in-plane component; metadata records its cosine with `v1`.
  - b: Recomputed rank-256 cumulative second-moment energy for down-input layers 1, 13 and 27, with the 64-slot group-128 capacity marked explicitly.
  - c: Measured range ratio versus `sqrt(1-f)` pooled from 3B E1c activations, E7 V cache and E20 multi-slot experiments. Every y value is the mean group range divided by the paired Hadamard range at the same layer/site/group size.
- Evidence hierarchy: Panel a is the geometric mechanism, panel b is the finite-capacity qualification, and panel c is the pooled quantitative check. Layer 1 is visually primary; additional layers and experiment families are supporting evidence.
- Statistics/source data: Panel-a/b eigenvectors are recomputed from the frozen E1c dump after excluding sequence position 0, with seed 20260902, oversampling 16, one power iteration and three covariance passes. Panel c consumes only existing labeled result rows; no values are interpolated, extrapolated or copied across models/sites/layers.
- Reviewer risk: Recovering a stale factor would silently reintroduce BOS and a different sample. The preparation script records the exclusion rule, seed, row count and Ritz residuals and saves the actual eigenvectors used by the figure. The pooled range-law table keeps source artifact and configuration provenance per point.
