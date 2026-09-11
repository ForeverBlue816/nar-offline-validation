# Local surface revision contract

This user-directed revision supersedes the earlier identical overview/detail,
height-column and closed-sidewall presentation rules. The earlier records are
historical; this is not a retroactive preregistration.

Question: what local activation structure is present before activation QDQ,
and what remains after signed per-token g128 centering? No method is required
to look better. Quantitative grid: columns hold Blocks 1, 13, 24, 36; rows hold
Unrotated reference, Hadamard, PrismQuant (k=8), PrismQuant (k=max). Paired
local rows share canonical inputs. End-to-end uses the matching paired-local
unrotated norm-fused FP32 reference above three observed quantized forwards.

Python/Matplotlib native 3D upper surfaces use sample 0, tokens [0,128), channels
[0,512), all 65,536 vertices and all 64,897 adjacent cells. Heights are linear,
unchanged absolute raw values or absolute residuals computed from the complete
signed tensor before cropping. No pooling, data stride, zero padding, smoothing,
columns or walls. Four complete groups are located by floor-only boundaries
at 127.5, 255.5 and 383.5.

The common orthographic camera is elevation 28, azimuth -55, aspect (2,1.2,1).
Viridis and PowerNorm(gamma=0.5) map colors only; each column shares 1.03 times
the maximum of its displayed methods. All-zero columns explicitly use a
display-only upper bound of 1. The linear-color control uses identical heights,
camera and limits. Rotated-only zoom explicitly uses its own shared limits.

PDF/SVG preserve vector text/axes and 600-dpi rasterized surfaces. Four-column matrices use 11.75 pt native text at 12.05 inches and are intended
for 7.2-inch (183 mm) paper placement, retaining 7.02 pt text. Single-block
panels use 8.5 pt native text and support 3.5-inch placement. Three token ticks
(0, 64, 127) avoid crowding; all 128 token coordinates remain unchanged. Every export has geometry, hashes, alignment and PDF audits.
Debug exports mark exact corners and real maxima; PDF pages are rasterized
again for visual inspection. Uncertainty is not drawn because each surface is
one fixed observed sample, not an estimate across independent models.

Overview remains the full-domain context; metrics and exact ECDFs retain all
eight samples. The local window need not contain global outliers. Raw tensors,
frozen factors, quantizers, measured statistics and supplementary failures are
immutable. Integrity snapshots and the publication manifest record this.


---

## Archived preceding full-resolution revision (superseded for detail)

The record below describes the previous publication. Its identical-view, zero-column and wall rules do not govern current detail outputs. Existing overview images are preserved as global context.

# Full-resolution rendering revision

User-directed revision: preserve the complete 2048-token sequence and every
channel (12288 at down_proj; 4096 at q_proj). Do not use the previous 128 x 256
maximum-pooling cache, channel/token strides, smoothing, or a cropped detail
window. The fixed sample remains sample 0 for every method; all eight samples
continue to determine reported metrics.

Plot the actual absolute post-rotation/pre-QDQ values, or the absolute signed
group-centered residual, as a height surface. Close its four exterior boundaries
to z=0, so the display reads as height rising from a base. These geometric
sidewalls are not additional activation observations. Never shift, clip, or
zero measured surface heights. Use shared linear z/color limits across methods
within each layer/site/quantity. Unrotated reference rows remain labelled.

Render every grid vertex and both triangles of every adjacent grid cell through
a depth buffer. A streaming Python/Numba rasterizer avoids allocating tens of
millions of Matplotlib polygon objects. Its camera comes directly from the
Matplotlib axes. Record the number of vertices and triangles, raw shard hash,
array shape, source maximum, and zero base for every panel. Depth-tested vertex
coverage retains even subpixel projected peaks. Pixel resolution and physical
occlusion still limit what can be distinguished on a printed page; neither is
input-data subsampling. SVG/PDF keep axes/text editable and embed the surface
raster at 600 dpi.

Tick labels are real channel/token indices. Channel ticks advance by 2000;
token ticks are 0, 1000, 2000. Exact array domains remain 0..12287/4095 and
0..2047. Typography is Times New Roman. Multicolumn exports are 12 inches wide;
8.5-point tick labels remain at least 5.1 points if reduced to 7.2 inches.

The archived capture and all numerical limitations remain unchanged, including
three failed supplementary frozen-factor checks. This revision changes the
presentation and does not support a new scientific-performance claim.

QA: verify projection, depth ordering, wall closure and complete vertex/cell
counts on small deterministic geometry tests; inspect the complete measured
Block 36 panel before batch rendering; audit panel alignment, PDF text and
collisions; inspect all revised matrix panels. No synthetic test surface may
appear among the scientific figures.

Final height-column rule: every measured vertex also contributes a depth-tested vertical segment from z=0 to its unchanged value. This connects even subpixel-width peaks continuously to the base. The segment represents geometric height, not additional activation observations. The complete grid-cell mesh and exterior closure remain present. Geometry reports record one vertical segment per source element.
