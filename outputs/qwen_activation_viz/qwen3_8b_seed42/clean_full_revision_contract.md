# Clean local matrices and four-row full-domain matrices

User-requested presentation revision, 2026-09-14. Scope: raw matrix for
end_to_end/q_proj, end_to_end/down_proj and paired_local/q_proj, in detail and overview.

Claim: show measured activation magnitude across methods and blocks without
imposing an advantage. The 4 x 4 diagnostic grid compares the unrotated reference,
Hadamard, PrismQuant k=8 and k=max at Blocks 1, 13, 24 and 36.

Backend: existing Python / Matplotlib workflow, native local surfaces and the
existing Numba full-grid upper-surface renderer. No AI-generated scientific data.
Detail retains all 128 x 512 measured vertices, camera, blue-orange PowerNorm(0.5),
linear heights and shared per-column limits. Remove the three explanatory footer
lines and 0.60 inches of corresponding page space, preserving physical panel size,
axis labels, colorbars, height ticks and the pale z=0 floor.

Overview covers all 2,048 tokens and all 4,096 q_proj or 12,288 down_proj channels.
Each of the four rows uses the same full-domain limit in a given column. All
vertices and adjacent cells are processed, without pooling, stride or cropping;
only measured upper surfaces are drawn, without artificial columns or sidewalls.
Retain the established full-view camera and blue-orange linear color mapping.
The reference is the existing unquantized norm-fused FP32 paired_local cache.
For end_to_end the rotated rows include upstream QDQ, so identical intermediate
inputs across all four rows are not claimed.

Export: PDF/SVG with vector axes and editable text, surfaces at 600 dpi, PNG at
600 dpi. Double-column insertion width 7.2 inches. Minimum rendered text at that
width: detail about 7.02 pt, overview at least 5 pt. Inspect all 96 panels and
assembled pages; enforce final panel alignment and audit PDF glyphs/collisions.
These are descriptive surfaces for fixed sample 0 (WikiText-2 sample index 52),
not replicate aggregates: uncertainty intervals and statistical tests do not apply.
No recapture, activation edits, rotation changes, metric updates or new accuracy
claims. Existing numerical limitations and all unrelated figure sets are preserved.
