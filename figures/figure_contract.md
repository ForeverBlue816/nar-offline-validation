# Figure contract: design choices and deployment efficiency

Frozen source: commit 6747960b96b0b03e626e98eaeeccf0d1abc34e79;
run results/e28_v2/20260912_a40_v2_full_int4. Main was checked before work.
Backend: Python/matplotlib; actual 5.5-inch final width, Times New Roman Bold.

Figure 4 asks how activation metadata and rank affect PPL. The 2026-09-13
user revision uses two side-by-side quantitative panels: all 11 budget points
plus BF16 (3B main, 8B appendix), and all 15 rank points. A wider green band
marks the k=8 category only. The former Graph overhead panel is archived;
its complete source values remain in the CSV and the deployment evidence.
Accuracy and runtime remain distinct protocols, not a joint Pareto frontier.

The efficiency figure asks whether the compact transform preserves practical
integer-pipeline efficiency. The small implementation strip defines the actual
FP16-output R4 interface; a compares eager input throughput, b bounds the decode
benefit by model, c gives independent allocated-memory evidence, d tests dispatch
and bulk R4 implementation choices. No unverified Figure 5 assignment is made:
the complete manuscript's figure numbering is unavailable in this repository.

Center: official collector median over 150 samples for throughput/latency;
ratios of pooled medians for corresponding method comparisons. Session ratios
pair the same session/workload/backend. Kernel comparison centers are medians
of three session wall-time ratios. Memory center follows the collector maximum
of independent-session inference peaks. All three session values are retained;
range bars are min–max, not confidence intervals. No significance testing.

Risks checked: original/private backend confusion; different initialization
cohorts; FP16 versus BF16; graph allocated versus reserved or file bytes; E17
asymmetric versus E28 per-token symmetric quantizer; full-path versus B-only
ratios; nonmonotone recovery; all point and uncertainty extents retained.

Exports: true vector PDF with embedded font, editable SVG text, PNG at 600 dpi,
independent panels and appendix, exact source CSVs, metadata, captions, LaTeX,
source and rendered QA. Final rendered axes alignment tolerance is 1.5 pt.

Visual refinement: deployment prefill uses explicit batch-size wording. The
kernel dot plot uses a labeled linear 0.25–1.05 axis containing every session
and the reference 1.00, thicker comparisons, and derived percentage reductions.
No point is shifted, omitted, transformed nonlinearly, or truncated.


## September 13 — Figures 2/3 typography and evidence structure; Figure 4 cost restoration

Python/matplotlib is the saved backend. The scientific question is whether quantizer-aware null-space energy placement reduces activation range/error, generalizes across layers and measurement families, and is consistent with an independent model's expert observations. Figure 2 is the paired layer-level quantitative grid, retaining all 252 plotted values and both original mean percentage reductions. Figure 3 is a 2×2 evidence grid: geometry → depth coverage → pooled reference law → cross-model experts. Figure 4 restores the original independently benchmarked decoder-layer transform share after accuracy and rank design.

All main figures are 5.5 inches wide, with true embedded Times New Roman Bold, 7.5 pt minimum glyph size, 8.5 pt axis labels, 10.5 pt panel letters, blue/teal primary encoding, 2.1–2.3 pt main curves, editable vector PDF/SVG and 600-dpi PNG. Every export must pass 1.5-pt rendered alignment, clipping, collisions and actual PDF font audits. No experiments, GPU runs, model calibration or new fitted line is authorized or needed.

Reference-law coefficients remain exactly those frozen at commit 519ddad: intercept 0.05980223276470587, slope 0.8665147718600141, pooled R² 0.8613894973078933. All 2912 reference and 5342 expert points remain. Internal experiment IDs are provenance only. The expert panel adds equal-count f-decile medians and Q1–Q3 intervals, never an MoE fitted line. Its within-MoE R² diagnostics describe association, not predictive R² of the transferred dense reference. Full-row means the original 256-row collection cap, not a selected correlation threshold; cold means the original <2048 routed-token shrinkage threshold.

Energy coverage retains every original BOS-excluded rank-256 point for layers 1/13/27 and adds existing all-token rank-64 curves for layers 5/9/18/22. The prespecified layers [1,5,9,13,18,22,27] cover depth evenly. Solid versus dashed curves and a visible panel note distinguish the protocols; added curves stop at rank 64. Different BOS and spectral-rank protocols are never silently spliced. Existing all-token layer-1 energy is BOS-dominated and is not substituted for the original BOS-excluded curve. The full same-protocol seven-layer rank-64 view is exported as a clearly identified appendix. Geometry remains the original 8064 non-BOS projections. The original expert collector keeps rows from its own covariance capture; experts are held out of the dense reference fit, but their rows are not claimed independent of expert calibration. Cold-expert association does not establish a causal noise explanation.

Figure 4c uses all four original rank-cost observations and both Hadamard references; its ordinate is 100 t_transform/(t_layer+t_transform) from RTX PRO 6000 Blackwell Server Edition, T=2048. It is not A40 whole-model Graph overhead. The green k=8 highlight in 4b remains a category band, not uncertainty.


## Figure 2 palette and stroke refinement

User-directed styling revision: restore original PrismQuant deep blue #1D3557, Hadamard pale cyan #A8DADC and DuQuant pale pink #F5CBCB. Reduce curve widths to 1.5 pt / 1.05 pt and the original deep-blue dashed reference to 0.85 pt, with lighter marker boundaries and axes. Preserve the three-panel evidence structure, all 252 observations, existing axis limits, layout and ≥7.5-pt typography. This supersedes the earlier stronger-stroke/warm-accent preference for Figure 2 only.
