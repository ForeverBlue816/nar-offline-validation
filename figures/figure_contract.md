# Figure contract: design choices and deployment efficiency

Frozen source: commit 6747960b96b0b03e626e98eaeeccf0d1abc34e79;
run results/e28_v2/20260912_a40_v2_full_int4. Main was checked before work.
Backend: Python/matplotlib; actual 5.5-inch final width, Times New Roman Bold.

Figure 4 asks how activation metadata and rank affect PPL, and what k=8 adds
to measured deployment latency. An asymmetric quantitative layout preserves
all 11 budget points plus BF16 (3B main, 8B appendix), all 15 rank points, and
uses two same-backend Graph overhead estimates. The original E17 cost evidence
is retained in the appendix. Accuracy and runtime are different protocols and
are not a joint checkpoint Pareto frontier.

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
