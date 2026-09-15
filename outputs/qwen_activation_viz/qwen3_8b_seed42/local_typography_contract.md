# Local matrix typography follow-up, 2026-09-15

Scope: only raw/detail/matrix for end_to_end/q_proj, end_to_end/down_proj and
paired_local/q_proj. Keep all measured data, four method rows, four block columns,
local window, physical dimensions, camera, color normalization, height ticks,
zero plane and color scales. Core claim: compare the recorded activation
magnitudes without manufacturing a method advantage. Quantitative 4 x 4 grid;
each panel is a method/block diagnostic for fixed sample 0, not a replicate
aggregate, so uncertainty intervals do not apply.

Restore the original Viridis palette, matching the user's blue-green reference.
Keep PowerNorm(gamma=0.5), linear heights and all measured shared limits.
Only color values change; no surface heights are normalized or changed.

Use the existing Python/Matplotlib renderer. Per user request use actual Times
New Roman Bold, increasing body/tick/row text from 11.75 to 14 pt, block titles
to 15 pt and figure title to 16 pt. At 7.2-inch insertion width, body text is
8.37 pt (12.05-inch export width). Remove the figure-level separator below the
Unrotated reference. Keep the explanatory footer absent. If any enlarged text
collides, repair spacing without altering measured surfaces or shared scales.

Export synchronized PDF/SVG/PNG with editable vector text/axes and 600-dpi
surface marks. Verify actual embedded Times New Roman Bold, increased text size,
no reference separator, every source/geometry/scale unchanged, final 1.5-pt panel
alignment and PDF text/collision audits. Inspect all 48 panels and three pages.
Other figures, including all full-domain exports, and numerical records remain
unchanged. Historical revision audits retain their original scope.

Colorbar tick-label padding is 4 pt to clear the enlarged bold numbers; other
plot and text positions remain fixed. Initial spacing failures are archived
under qa/local_typography_revision/initial_layout.
