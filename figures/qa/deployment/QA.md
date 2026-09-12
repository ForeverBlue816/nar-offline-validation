# Final figure QA

The frozen raw samples reproduce the collector's centers and paired ratios.
All 416 deployment rows and all original accuracy/cost rows are retained.
Independent original Figure 4 verification also reproduced source PPLs from
raw per-sequence losses, checked seed IDs, and checked per-site rank caps.

Every final PDF was checked for text collisions and clipping, embedded Times
New Roman Bold, a 7.5 pt minimum text size, and absence of raster images. Every
SVG retains text nodes and has no embedded images. Every final PNG is 600 dpi.
Shared rendered axes and panel-letter anchors pass the 1.5 pt alignment gate.
The source data and uncertainty ranges fit inside all plot limits. PDF and SVG
were rendered independently, using the same verified font through Fontconfig,
and each panel was visually inspected along with its complete assembly.

| Panel | Role / summary | Spread and unit | Visual review |
|---|---|---|---|
| Fig.4 a | E20 3B metadata/PPL; 11 points + bf16 | Retained mean PPL; no invented interval | All labels readable; exact x preserved; two brackets explained in caption |
| Fig.4 b | 15 categorical-rank recovery values | Recovery from recorded mean PPLs; heterogeneous original seed protocols documented | Real nonmonotone paths retained; narrow k=8 band stops within b; model markers and legend explicit |
| Fig.4 c | 3B/8B additional Graph latency | Ratio of pooled medians; three paired-session ratios and min–max | Both models/percentages visible; axis starts at zero and includes every session |
| Efficiency strip | Actual R4 interface | Diagram only; widths do not encode time | X branch, FP16 output, offline factors, shared token quantizer, and CUTLASS all visible |
| Efficiency a | Four prefill workloads, both integer methods | Ratio of pooled throughput medians; session-ratio min–max | Equal-width bars and FP16=1 line; all eight labels; input throughput explicitly named |
| Efficiency b | Six matched Graph latency values | Pooled medians and session-median min–max | Zero origin, complete FP16 bars, truthful 3B slowdown and 8B 1.22× gain |
| Efficiency c | Six matched Graph memory values | Maximum inference allocated peak; session-peak min–max | Decimal GB, both savings, similar integer-pipeline resource levels; not weight size |
| Efficiency d | Dispatch and complete-R4 B comparisons | Median of three paired-session wall ratios; min–max | Own reference=1 explicit, four values, two baseline groups, no B-only/full-model claim |
| Appendix metadata 8B | 11 E20 points + bf16 | Original summary values | All labels and repeated x coordinates retained |
| Appendix E17 cost | Four k=8/32 points and two references | Original local bench aggregates; no invented intervals | Original RTX PRO 6000 protocol and proxy denominator in caption |
| Appendix matched modes a/b | Six latencies per model | Same-private-backend medians; min–max | Both modes, all three methods, distinct mode markers; no original eager denominator |
| Appendix all kernels a–e | All 30 comparisons | Median of three paired-session ratios; min–max | All three shapes, two models, and five families; T=1 TC-B losses included |

The automated collision reports have zero FAIL and zero WARN findings.
The main figures are 5.5 inches wide (Figure 4 height 4.65; deployment 5.35).
The 8B standalone appendix is 3.4 inches wide, with unchanged physical text size.
Standalone panels are bare assembly assets and use the corresponding main
figure legend. Grayscale previews preserve markers, ordering, strokes and
readable baseline edges; color is not the only model/mode encoding.

Source-preflight warnings reviewed: TIFF is a generic Nature export default,
while this task specifies 600-dpi PNG; 139.7 mm is the explicit 5.5-inch width,
not a Nature 89/183 mm column. Rendered checks enforce the actual requested
contract. The shared-module and explicit-font options make the source audit
inspect the real export code without substituting a sans-serif font.

See verification.json for all 21 PDF/SVG/PNG asset checks, reproducibility.json
for the frozen-CSV rebuild and two byte-identical runs of all 63 figure assets,
and the individual alignment, extent, source, text, and collision reports.

Protocol limits are documented in captions and metadata: random performance
weights, distinct accuracy protocols, non-native per-token symmetric E28
quantization/KV interfaces, one preallocated page2176 during Graph timing,
independent page64 correctness, and retained large-accumulator stress failures.
There are no significance stars, hidden points, enlarged error bars, smoothing,
profiler-time substitutions, or full-checkpoint Pareto claims.
